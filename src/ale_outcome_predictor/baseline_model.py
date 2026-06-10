"""Baseline ALE-outcome model: gradient-boosted tree + cluster-aware splits  (milestone 15.4).

Consumes the canonical feature table from ``feature_engineer.build_feature_matrix``
(one row per ``(gene, selection_condition)``, columns = ``feature_engineer.FEATURE_COLUMNS``)
and trains a gradient-boosted-tree classifier that scores, for a given chassis +
selection condition, how likely each gene is to acquire a causal mutation. The
output is a ranked gene list with (optionally calibrated) probabilities — the
deliverable the research note's acceptance test grades.

Why cluster-aware splits
------------------------
A naive random train/test split leaks signal whenever near-identical rows sit on
both sides of the split — exactly the AUC-inflation trap documented for the AMP
work (the ``bio-ml-eval-protocols`` line). Here the leakage surfaces two ways:

* **Homology leakage** — orthologous genes (same COG / eggNOG cluster) carry
  almost the same GSM-flux features, so a model can "memorise" a gene family.
  ``cluster_aware_split_indices`` keeps a whole cluster on one side of the split.
* **Chassis leakage** — the headline acceptance test is *held-out chassis* recall
  (train on chassis != X, predict on X). ``held_out_group_splits`` is the
  leave-one-group-out generator for that, keyed on organism / base strain.

Both are framed as *grouped* splits so the evaluation never reports a number a
random split would have inflated.

What lives here
---------------
* **Pure-Python core** (no third-party deps — import-light + unit-testable):
  ``cluster_aware_split_indices``, ``held_out_group_splits``, ``year_split_indices``,
  ``top_k_recall`` / ``top_k_recall_by_group``, ``uniform_essential_baseline_scores``,
  ``brier_score``, ``reliability_bins``, ``extract_year``.
* **Heavy pass** (sklearn / lightgbm imported *inside* the functions):
  ``build_design_matrix``, ``train_model``, ``predict_scores``,
  ``predict_ranked_genes``, ``evaluate_held_out_chassis``,
  ``cross_validate_cluster_aware``, ``evaluate_calibration``.

Design notes
------------
* **Dependency-light import.** ``numpy`` / ``pandas`` / ``sklearn`` / ``lightgbm``
  are imported only inside the functions that need them (and under
  ``TYPE_CHECKING`` for annotations), so importing this module during the CI
  smoke test never pulls in a model stack. The split / metric core runs on the
  standard library alone.
* **Estimator is swappable.** ``_build_estimator`` prefers LightGBM (the
  ``models`` extra) and falls back to sklearn's ``HistGradientBoostingClassifier``
  — both are gradient-boosted trees that ingest NaN natively, so the NA-safe
  feature table needs no imputation.
* **Held-out chassis label-leak guard.** Recommendations for an in-house
  prospective chassis must use an anonymised feature view until that chassis's
  own paper publishes (research_notes/15 BLOCKERS). This module never embeds a
  chassis genotype; it only consumes the generic feature columns.

Citations (from ``research_notes/15_ale_outcome_predictor.md`` LINKS)
--------------------------------------------------------------------
* Anand et al. 2021, bioRxiv 2021.07.19.452699 — aggregated-ALE workflow; the
  held-out structure this evaluator targets extends theirs with GSM features.
* Zampieri et al. 2019, PLOS Comput Biol 10.1371/journal.pcbi.1007084 — the
  GSM-feature + ML lane.
* Phaneuf et al. 2019, NAR 47(D1):D1164 — ALEdb 1.0, the training corpus.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type hints; keeps runtime import dep-light
    import numpy as np
    import pandas as pd

from .feature_engineer import FEATURE_COLUMNS, STRESS_CLASSES

__all__ = [
    "JOIN_KEYS",
    "LABEL_COLUMN",
    "CATEGORICAL_INPUTS",
    "MODEL_INPUT_COLUMNS",
    "ModelConfig",
    "ModelBundle",
    "cluster_aware_split_indices",
    "held_out_group_splits",
    "year_split_indices",
    "extract_year",
    "top_k_recall",
    "top_k_recall_by_group",
    "uniform_essential_baseline_scores",
    "brier_score",
    "reliability_bins",
    "build_design_matrix",
    "train_model",
    "predict_scores",
    "predict_ranked_genes",
    "evaluate_held_out_chassis",
    "cross_validate_cluster_aware",
    "evaluate_calibration",
]

# --------------------------------------------------------------------------- #
# Schema contract (mirrors feature_engineer.FEATURE_COLUMNS)
# --------------------------------------------------------------------------- #

# Columns that identify a row but are NOT model inputs.
JOIN_KEYS: tuple[str, ...] = ("gene", "selection_condition", "experiment_id")
# Supervised target.
LABEL_COLUMN: str = "mutated"
# Categorical inputs that get one-hot expanded (stable order from STRESS_CLASSES).
CATEGORICAL_INPUTS: tuple[str, ...] = ("stress_class",)
# Everything in FEATURE_COLUMNS that is neither a join key, the label, nor a
# categorical — i.e. the numeric / boolean predictors fed to the tree as-is.
MODEL_INPUT_COLUMNS: tuple[str, ...] = tuple(
    col
    for col in FEATURE_COLUMNS
    if col not in JOIN_KEYS and col != LABEL_COLUMN and col not in CATEGORICAL_INPUTS
)


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters + evaluation knobs for the baseline model.

    Defaults are deliberately small/conservative starting points; the feature
    ablation notebook (research_notes/15 DELIVERABLE) is where these get tuned.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31          # LightGBM; mapped to max_leaf_nodes for sklearn
    max_depth: int = -1           # -1 = unbounded (LightGBM convention)
    min_child_samples: int = 20
    random_state: int = 0
    n_jobs: int = 1
    k_values: tuple[int, ...] = (10, 25)   # top-k recall reporting (acceptance test)
    calibrate: bool = False                # wrap CalibratedClassifierCV after fit
    calibration_method: str = "isotonic"   # "isotonic" | "sigmoid"
    year_cutoff: int = 2020                 # pre-cutoff train / cutoff+ test (calibration)
    use_lightgbm: bool | None = None        # None = auto (LightGBM if importable)


@dataclass
class ModelBundle:
    """A fitted estimator plus everything needed to score a new feature frame."""

    estimator: object                       # fitted sklearn/lightgbm classifier
    design_columns: list[str]               # exact column order the estimator saw
    config: ModelConfig
    n_train: int = 0
    backend: str = "unknown"                # "lightgbm" | "sklearn"
    calibrated: bool = False
    notes: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Splitters (pure Python — grouped so nothing leaks across the split)
# --------------------------------------------------------------------------- #


def cluster_aware_split_indices(
    groups: "list[object]",
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """K grouped folds where a whole group is never split across train/test.

    ``groups[i]`` is the cluster id of row ``i`` (a homology cluster — COG /
    eggNOG — or the gene id itself when no clustering is available). Groups are
    assigned whole to the currently-smallest fold (greedy, size-balanced) after a
    seeded shuffle for tie variety, so the folds are roughly row-balanced and
    deterministic given the same inputs. Returns ``[(train_idx, test_idx), ...]``.

    This is the homology-leakage guard: it is sklearn ``GroupKFold`` semantics,
    re-implemented dependency-free so it is unit-testable in the CI smoke test.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    n = len(groups)
    if n == 0:
        return []

    group_rows: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        group_rows.setdefault(g, []).append(i)

    unique = list(group_rows)
    random.Random(seed).shuffle(unique)
    # Largest groups first so the size-balancing has room to even out.
    unique.sort(key=lambda g: len(group_rows[g]), reverse=True)

    effective = min(n_splits, len(unique))
    fold_sizes = [0] * effective
    fold_rows: list[list[int]] = [[] for _ in range(effective)]
    for g in unique:
        target = min(range(effective), key=lambda f: fold_sizes[f])
        fold_rows[target].extend(group_rows[g])
        fold_sizes[target] += len(group_rows[g])

    all_idx = range(n)
    splits: list[tuple[list[int], list[int]]] = []
    for f in range(effective):
        test = sorted(fold_rows[f])
        test_set = set(test)
        train = [i for i in all_idx if i not in test_set]
        splits.append((train, test))
    return splits


def held_out_group_splits(
    groups: "list[object]",
) -> list[tuple[object, list[int], list[int]]]:
    """Leave-one-group-out generator, keyed on group value.

    For the held-out-chassis acceptance test: pass per-row chassis (organism /
    base strain) and get ``[(held_out_value, train_idx, test_idx), ...]`` with one
    entry per distinct chassis. Groups are emitted in sorted order for
    determinism.
    """
    by_group: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    out: list[tuple[object, list[int], list[int]]] = []
    for g in sorted(by_group, key=lambda x: str(x)):
        test = by_group[g]
        test_set = set(test)
        train = [i for i in range(len(groups)) if i not in test_set]
        out.append((g, train, test))
    return out


def year_split_indices(
    years: "list[int | None]",
    cutoff: int = 2020,
) -> tuple[list[int], list[int]]:
    """Temporal split: train on publications before ``cutoff``, test on ``cutoff``+.

    Rows whose year is ``None`` go to neither side (they cannot be placed on the
    timeline). Backs the calibration year-split in the acceptance test (train
    pre-2020, test 2020-2025).
    """
    train = [i for i, y in enumerate(years) if y is not None and y < cutoff]
    test = [i for i, y in enumerate(years) if y is not None and y >= cutoff]
    return train, test


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_year(text: str | None) -> int | None:
    """Pull a 4-digit publication year (1900-2099) out of a citation / DOI string.

    Heuristic only — returns the first plausible year found, else ``None``. The
    canonical mutation table's ``publication`` field is free text, so this is a
    best-effort extractor feeding ``year_split_indices``.

    TODO: once the real ALEdb export's publication field format is
    known, replace with a structured year column rather than regex-scraping.
    """
    if not text:
        return None
    m = _YEAR_RE.search(text)
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- #
# Metrics (pure Python)
# --------------------------------------------------------------------------- #


def top_k_recall(
    y_true: "list[int]",
    scores: "list[float]",
    k: int,
) -> float:
    """Fraction of the positives that land in the top-``k`` highest-scored rows.

    Ties are broken by original row order (stable). Returns ``nan`` when there are
    no positives (recall undefined). This is the per-condition metric the
    acceptance test reports at k = 10 and k = 25.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    n_pos = sum(1 for v in y_true if v)
    if n_pos == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    hits = sum(1 for i in order[:k] if y_true[i])
    return hits / n_pos


def top_k_recall_by_group(
    groups: "list[object]",
    y_true: "list[int]",
    scores: "list[float]",
    k: int,
) -> float:
    """Macro-average ``top_k_recall`` across groups (e.g. per selection condition).

    Groups with no positives are skipped (their recall is undefined). Returns
    ``nan`` if no group has a positive.
    """
    buckets: dict[object, list[int]] = {}
    for i, g in enumerate(groups):
        buckets.setdefault(g, []).append(i)

    per_group: list[float] = []
    for idx in buckets.values():
        gt = [y_true[i] for i in idx]
        sc = [scores[i] for i in idx]
        r = top_k_recall(gt, sc, k)
        if r == r:  # skip nan (no positives in this group)
            per_group.append(r)
    if not per_group:
        return float("nan")
    return sum(per_group) / len(per_group)


def uniform_essential_baseline_scores(
    is_essential: "list[bool | None]",
    seed: int = 0,
) -> list[float]:
    """Acceptance-test baseline: rank essential genes uniformly above the rest.

    Essential genes get a random score in ``[0.5, 1.0)``, every other gene a random
    score in ``[0.0, 0.5)`` — so any essential gene outranks any non-essential one,
    but the order *within* the essential set is uniform (no GSM-flux information).
    The model must beat this by >= 2x top-10 recall on >= 3 chassis to pass.

    ``None`` essentiality (solver failure / gene absent) is treated as
    non-essential.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for ess in is_essential:
        if ess:
            out.append(0.5 + 0.5 * rng.random())
        else:
            out.append(0.5 * rng.random())
    return out


def brier_score(y_true: "list[int]", prob: "list[float]") -> float:
    """Mean squared error between predicted probabilities and outcomes (lower=better)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    return sum((p - y) ** 2 for y, p in zip(y_true, prob)) / n


def reliability_bins(
    y_true: "list[int]",
    prob: "list[float]",
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Reliability-diagram data: per equal-width probability bin, mean predicted vs
    observed frequency and the bin count.

    Bins span ``[0, 1]``; a probability of exactly 1.0 lands in the last bin.
    Empty bins are omitted. Feeds the calibration reliability plot in the
    acceptance test.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = [i / n_bins for i in range(n_bins + 1)]
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            members = [i for i, p in enumerate(prob) if lo <= p <= hi]
        else:
            members = [i for i, p in enumerate(prob) if lo <= p < hi]
        if not members:
            continue
        count = len(members)
        mean_pred = sum(prob[i] for i in members) / count
        frac_pos = sum(y_true[i] for i in members) / count
        rows.append(
            {
                "bin_lower": lo,
                "bin_upper": hi,
                "mean_predicted": mean_pred,
                "observed_frequency": frac_pos,
                "count": float(count),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Design matrix + estimator (heavy: pandas / numpy / sklearn / lightgbm inside)
# --------------------------------------------------------------------------- #


def build_design_matrix(features: "pd.DataFrame") -> "pd.DataFrame":
    """Turn the canonical feature table into an all-numeric design matrix.

    * Drops the join keys and the label.
    * One-hot expands ``stress_class`` over the full ``STRESS_CLASSES`` vocabulary
      (stable column order, so train and inference matrices always align — an
      unseen/absent class becomes an all-zero block, never a new column).
    * Casts booleans to ``0.0/1.0`` and leaves NaN in place (the tree backends
      ingest NaN natively, so no imputation is needed).

    Returns a float DataFrame whose columns are the model's design columns.
    """
    import numpy as np
    import pandas as pd

    cols: dict[str, object] = {}
    for col in MODEL_INPUT_COLUMNS:
        series = features[col] if col in features.columns else pd.Series(pd.NA, index=features.index)
        cols[col] = pd.to_numeric(series, errors="coerce").astype("float64")

    # One-hot the categorical(s) against the fixed vocabulary.
    for cat in CATEGORICAL_INPUTS:
        raw = (
            features[cat].astype("string")
            if cat in features.columns
            else pd.Series(pd.NA, index=features.index, dtype="string")
        )
        if cat == "stress_class":
            vocabulary: tuple[str, ...] = STRESS_CLASSES
        else:  # pragma: no cover - only stress_class today
            vocabulary = tuple(sorted(v for v in raw.dropna().unique()))
        for value in vocabulary:
            cols[f"{cat}={value}"] = (raw == value).astype("float64")

    design = pd.DataFrame(cols, index=features.index)
    # Deterministic column order: numeric inputs first, then one-hot blocks.
    return design.reindex(columns=list(cols.keys())).astype(np.float64)


def _build_estimator(config: ModelConfig) -> tuple[object, str]:
    """Construct an unfitted gradient-boosted-tree classifier + its backend name.

    Prefers LightGBM (the ``models`` extra); falls back to sklearn's
    ``HistGradientBoostingClassifier`` — both handle NaN inputs natively.
    """
    want_lgbm = config.use_lightgbm if config.use_lightgbm is not None else True
    if want_lgbm:
        try:
            from lightgbm import LGBMClassifier

            est = LGBMClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                max_depth=config.max_depth,
                min_child_samples=config.min_child_samples,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=-1,
            )
            return est, "lightgbm"
        except ImportError:
            if config.use_lightgbm:  # explicitly requested but unavailable
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    est = HistGradientBoostingClassifier(
        max_iter=config.n_estimators,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.num_leaves,
        max_depth=None if config.max_depth in (-1, 0) else config.max_depth,
        min_samples_leaf=config.min_child_samples,
        random_state=config.random_state,
    )
    return est, "sklearn"


def _labels(features: "pd.DataFrame") -> tuple["np.ndarray", "np.ndarray"]:
    """Return (mask of non-NA labels, integer label vector over that mask)."""
    import numpy as np

    raw = features[LABEL_COLUMN]
    mask = raw.notna().to_numpy()
    y = raw[mask].astype("boolean").astype(int).to_numpy(dtype=np.int64)
    return mask, y


def train_model(features: "pd.DataFrame", config: ModelConfig | None = None) -> ModelBundle:
    """Fit the baseline gradient-boosted tree on a labelled feature table.

    Rows with a missing ``mutated`` label are dropped. If ``config.calibrate`` is
    set, the fitted estimator is wrapped in a ``CalibratedClassifierCV``.

    TODO: ``CalibratedClassifierCV``'s internal CV is *not*
    group-aware, so calibrating here can reintroduce the homology leakage the
    cluster-aware splits remove. Prefer calibrating on a held-out chassis fold
    (see ``evaluate_calibration``) over this convenience flag for the paper.
    """
    cfg = config or ModelConfig()
    X = build_design_matrix(features)
    mask, y = _labels(features)
    X_fit = X.iloc[mask]

    estimator, backend = _build_estimator(cfg)
    calibrated = False
    if cfg.calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        estimator = CalibratedClassifierCV(estimator, method=cfg.calibration_method)
        calibrated = True
    estimator.fit(X_fit, y)

    return ModelBundle(
        estimator=estimator,
        design_columns=list(X.columns),
        config=cfg,
        n_train=int(mask.sum()),
        backend=backend,
        calibrated=calibrated,
        notes={"n_positive": int(y.sum()), "n_features": X.shape[1]},
    )


def predict_scores(bundle: ModelBundle, features: "pd.DataFrame") -> "np.ndarray":
    """P(mutated = 1) for every row of ``features``, aligned to the bundle's columns."""
    import numpy as np

    X = build_design_matrix(features).reindex(columns=bundle.design_columns, fill_value=0.0)
    proba = bundle.estimator.predict_proba(X)
    classes = list(getattr(bundle.estimator, "classes_", [0, 1]))
    pos_col = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return np.asarray(proba[:, pos_col], dtype=float)


def predict_ranked_genes(bundle: ModelBundle, features: "pd.DataFrame") -> "pd.DataFrame":
    """Ranked gene list with probabilities — the deliverable output.

    Returns a frame ``[gene, selection_condition, probability]`` sorted by
    descending probability. Pass the feature rows for the chassis + condition you
    want a recommendation for.
    """
    import pandas as pd

    scores = predict_scores(bundle, features)
    out = pd.DataFrame(
        {
            "gene": features.get("gene", pd.Series(index=features.index, dtype="string")),
            "selection_condition": features.get(
                "selection_condition", pd.Series(index=features.index, dtype="string")
            ),
            "probability": scores,
        }
    )
    return out.sort_values("probability", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Evaluation harnesses (acceptance test)
# --------------------------------------------------------------------------- #


def evaluate_held_out_chassis(
    features: "pd.DataFrame",
    chassis: "list[object]",
    config: ModelConfig | None = None,
) -> "pd.DataFrame":
    """Leave-one-chassis-out top-k recall vs the uniform-essential baseline.

    For each distinct chassis: train on every other chassis, predict on the held-
    out one, and report macro top-k recall per ``selection_condition`` (model and
    baseline) plus their ratio — the headline acceptance-test number (target:
    model >= 2x baseline top-10 recall on >= 3 chassis).

    ``chassis`` is the per-row chassis label (organism / base strain), length =
    ``len(features)``; map it from the canonical mutation table's ``organism`` /
    ``base_strain`` via ``experiment_id`` before calling.
    """
    import pandas as pd

    cfg = config or ModelConfig()
    if len(chassis) != len(features):
        raise ValueError("chassis labels must align row-for-row with features")

    conditions = features["selection_condition"].astype("string").tolist()
    rows: list[dict[str, object]] = []
    for held, train_idx, test_idx in held_out_group_splits(list(chassis)):
        train_df = features.iloc[train_idx]
        test_df = features.iloc[test_idx]
        if test_df.empty or train_df.empty:
            continue

        bundle = train_model(train_df, cfg)
        model_scores = predict_scores(bundle, test_df).tolist()

        ess = [
            None if pd.isna(v) else bool(v)
            for v in test_df.get("is_essential", pd.Series(index=test_df.index)).tolist()
        ]
        base_scores = uniform_essential_baseline_scores(ess, seed=cfg.random_state)

        y_test = [0 if pd.isna(v) else int(bool(v)) for v in test_df[LABEL_COLUMN].tolist()]
        cond_test = [conditions[i] for i in test_idx]

        row: dict[str, object] = {"held_out_chassis": held, "n_test": len(test_idx)}
        for k in cfg.k_values:
            m = top_k_recall_by_group(cond_test, y_test, model_scores, k)
            b = top_k_recall_by_group(cond_test, y_test, base_scores, k)
            row[f"model_recall@{k}"] = m
            row[f"baseline_recall@{k}"] = b
            row[f"ratio@{k}"] = (m / b) if (b and b == b and b > 0) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cross_validate_cluster_aware(
    features: "pd.DataFrame",
    clusters: "list[object]",
    config: ModelConfig | None = None,
    n_splits: int = 5,
) -> "pd.DataFrame":
    """Cluster-aware K-fold top-k recall (homology-leakage-free CV).

    ``clusters`` is the per-row homology cluster id (COG / eggNOG; fall back to the
    gene id). Each fold trains on the other clusters and reports macro top-k recall
    per ``selection_condition`` on the held-out cluster fold.
    """
    import pandas as pd

    cfg = config or ModelConfig()
    if len(clusters) != len(features):
        raise ValueError("cluster labels must align row-for-row with features")

    conditions = features["selection_condition"].astype("string").tolist()
    splits = cluster_aware_split_indices(list(clusters), n_splits=n_splits, seed=cfg.random_state)

    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = features.iloc[train_idx]
        test_df = features.iloc[test_idx]
        if test_df.empty or train_df.empty:
            continue
        bundle = train_model(train_df, cfg)
        scores = predict_scores(bundle, test_df).tolist()
        y_test = [0 if pd.isna(v) else int(bool(v)) for v in test_df[LABEL_COLUMN].tolist()]
        cond_test = [conditions[i] for i in test_idx]
        row: dict[str, object] = {"fold": fold, "n_test": len(test_idx)}
        for k in cfg.k_values:
            row[f"model_recall@{k}"] = top_k_recall_by_group(cond_test, y_test, scores, k)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_calibration(
    features: "pd.DataFrame",
    years: "list[int | None]",
    config: ModelConfig | None = None,
    n_bins: int = 10,
) -> dict[str, object]:
    """Year-split calibration: Brier score + reliability bins on the recent split.

    Trains on rows with ``year < config.year_cutoff`` and evaluates on
    ``year >= config.year_cutoff`` (acceptance test: train pre-2020, test
    2020-2025). ``years`` is the per-row publication year (see ``extract_year``).
    Returns ``{"brier", "reliability", "n_train", "n_test"}``.
    """
    import pandas as pd

    cfg = config or ModelConfig()
    if len(years) != len(features):
        raise ValueError("years must align row-for-row with features")

    train_idx, test_idx = year_split_indices(list(years), cutoff=cfg.year_cutoff)
    train_df = features.iloc[train_idx]
    test_df = features.iloc[test_idx]
    if train_df.empty or test_df.empty:
        return {
            "brier": float("nan"),
            "reliability": [],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "note": "empty train or test side at this cutoff",
        }

    bundle = train_model(train_df, cfg)
    prob = predict_scores(bundle, test_df).tolist()
    y_test = [0 if pd.isna(v) else int(bool(v)) for v in test_df[LABEL_COLUMN].tolist()]
    return {
        "brier": brier_score(y_test, prob),
        "reliability": reliability_bins(y_test, prob, n_bins=n_bins),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
