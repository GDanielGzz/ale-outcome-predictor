"""End-to-end pipeline: corpus -> GSM-flux features -> trained model -> evaluation.

This is the driver the scaffold was missing — it wires ``data_loader`` /
``feature_engineer`` / ``media`` / ``baseline_model`` into a single reproducible
run and is what turns "nothing here trains yet" into a trained, evaluated model.

Stages
------
1. ``load_gsm``        fetch/cache the chassis GSM (iJO1366) from the cobrapy repo.
2. ``load_corpus``     read the committed, cited convergence corpus into the
                       canonical mutation schema (``data_loader.CANONICAL_COLUMNS``).
                       Swap to ``data_loader.load_aledb()`` once a real ALEdb pull
                       lands in ``data/`` — same schema, nothing else changes.
3. ``build_features``  per (gene, condition) flux + structural features, with each
                       condition scored **under its own growth medium** (``media``).
4. ``attach_labels``   join the corpus positives onto the feature matrix.
5. ``train``           fit the gradient-boosted-tree baseline.
6. ``evaluate``        leave-one-condition-out + cluster-aware CV top-k recall vs
                       the uniform-essential baseline, plus a GSM-coverage stat.

Run it: ``python -m ale_outcome_predictor.pipeline`` (see ``scripts/run_experiment.py``
for the figure-producing wrapper).

Honesty note
------------
The committed corpus is **small** (a handful of literature-anchored convergent
metabolic genes). Numbers from it are a *proof-of-concept that the pipeline runs
end-to-end and that the flux features carry condition-specific signal* — not a
powered benchmark. The headline limitation (regulatory hotspots are invisible to
a flux model) is reported as ``gsm_coverage`` rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from cobra import Model

    from .baseline_model import ModelBundle

from .data_loader import CANONICAL_COLUMNS

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_CSV = REPO_ROOT / "corpus" / "curated_ale_corpus.csv"
GSM_CACHE = REPO_ROOT / "data" / "iJO1366.json"      # gitignored; re-fetchable
FEATURES_CACHE = REPO_ROOT / "features" / "feature_matrix.parquet"  # gitignored
DEFAULT_GSM = "iJO1366"


@dataclass
class PipelineResult:
    """Everything a run produces, so callers (CLI / notebook) can render it."""

    features: "pd.DataFrame"
    bundle: "ModelBundle"
    loco: "pd.DataFrame"               # leave-one-condition-out recall table
    cluster_cv: "pd.DataFrame"          # cluster-aware CV recall table
    importances: "pd.DataFrame"         # gain/permutation feature importances
    gsm_coverage: dict[str, object]     # how many corpus targets the GSM can see
    meta: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage 1 — GSM
# --------------------------------------------------------------------------- #


def load_gsm(name: str = DEFAULT_GSM, cache: Path | None = None) -> "Model":
    """Load the chassis GSM, caching a JSON copy under ``data/`` after first fetch.

    Tries the local cache first (offline-friendly), then ``cobra.io.load_model``
    (downloads from the cobrapy model repository). iJO1366 is the default chassis:
    a real, genome-scale *E. coli* K-12 model whose gene ids are b-numbers — the
    join key for the corpus.
    """
    from cobra.io import load_json_model, load_model, save_json_model

    cache = cache or GSM_CACHE
    if cache.exists():
        return load_json_model(str(cache))

    model = load_model(name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_json_model(model, str(cache))
    return model


# --------------------------------------------------------------------------- #
# Stage 2 — corpus
# --------------------------------------------------------------------------- #


def load_corpus(path: Path | None = None) -> "pd.DataFrame":
    """Read the curated convergence corpus into the canonical mutation schema.

    The CSV already uses canonical-ish names; this coerces it to exactly
    ``data_loader.CANONICAL_COLUMNS`` (adding any missing columns as NA) so the
    downstream join is identical to a real ``load_aledb()`` table. The corpus
    ``gene`` column holds **b-numbers** (model gene ids) so it joins straight onto
    the feature matrix; the readable name rides along in ``df.attrs['gene_name']``.
    """
    import pandas as pd

    path = path or CORPUS_CSV
    raw = pd.read_csv(path)

    # Keep a name<->bnumber map for reporting before we coerce to canonical.
    name_map = dict(zip(raw.get("gene", []), raw.get("gene_name", [])))

    df = raw.copy()
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(CANONICAL_COLUMNS)]
    for col, dtype in CANONICAL_COLUMNS.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    df.attrs["gene_name"] = name_map
    df.attrs["source"] = str(path)
    return df


# --------------------------------------------------------------------------- #
# Stage 3 — features  (heavy: cobrapy FVA + single-gene-deletion)
# --------------------------------------------------------------------------- #


def build_features(
    model: "Model",
    corpus: "pd.DataFrame",
    config: "object | None" = None,
    cache: Path | None = None,
    use_cache: bool = True,
) -> "pd.DataFrame":
    """Build the (gene x condition) feature matrix with per-condition media.

    One row per (model gene, distinct selection_condition in the corpus). Flux
    features are computed **under each condition's growth medium** via ``media``;
    structural features are shared across conditions.

    Optimisation: conditions that resolve to the *same* growth medium (e.g. the
    37 °C and 42 °C glucose conditions) share one FVA/deletion pass — the heavy
    cobrapy work is keyed on the medium, not the condition, then the cheap
    condition columns (temperature / stress / minimal-media) are stamped per
    condition. Caches the assembled matrix to parquet so it runs once.
    """
    import pandas as pd

    from .feature_engineer import (
        FEATURE_COLUMNS,
        FeatureConfig,
        attach_labels,
        build_feature_matrix,
        parse_selection_condition,
    )
    from .media import resolve_exchange_bounds, resolve_environment

    cache = cache or FEATURES_CACHE
    if use_cache and cache.exists():
        feats = pd.read_parquet(cache)
    else:
        cfg = config or FeatureConfig()
        conditions = sorted(corpus["selection_condition"].dropna().astype(str).unique())
        gene_ids = [g.id for g in model.genes]

        # Group conditions by their resolved medium signature (hashable).
        def medium_key(text: str) -> tuple:
            return tuple(sorted(resolve_exchange_bounds(text, model).items()))

        by_medium: dict[tuple, list[str]] = {}
        for cond in conditions:
            by_medium.setdefault(medium_key(cond), []).append(cond)

        blocks: list[pd.DataFrame] = []
        for _key, conds in by_medium.items():
            rep = conds[0]
            # One heavy pass under this medium, using the representative condition.
            rep_block = build_feature_matrix(
                model,
                conditions=[rep],
                genes=gene_ids,
                config=cfg,
                env_resolver=lambda t: resolve_environment(t, model),
            )
            # Reuse flux+structural columns for every condition on this medium;
            # restamp only the condition-specific columns.
            flux_struct_cols = [
                c for c in rep_block.columns
                if c not in ("selection_condition", "experiment_id", "temperature_c",
                             "stress_class", "is_minimal_media", "mutated")
            ]
            for cond in conds:
                env = parse_selection_condition(cond)
                blk = rep_block[flux_struct_cols].copy()
                blk["selection_condition"] = cond
                blk["experiment_id"] = cond
                blk["temperature_c"] = env.temperature_c if env.temperature_c is not None else pd.NA
                blk["stress_class"] = env.stress_class
                blk["is_minimal_media"] = env.is_minimal_media
                blk["mutated"] = pd.NA
                blocks.append(blk)

        feats = pd.concat(blocks, ignore_index=True)
        # Quantise sub-1e-4 solver noise: a couple of FVA reactions have
        # alternate optima that differ ~1e-7 run-to-run (degenerate LPs), and the
        # small-corpus model amplifies that into unstable ranks. Rounding the
        # float features to 4 dp makes the build bit-reproducible.
        for _c, _d in FEATURE_COLUMNS.items():
            if _d == "Float64" and _c in feats.columns:
                feats[_c] = pd.to_numeric(feats[_c], errors="coerce").round(4)
        cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(cache)

    feats = attach_labels(feats, corpus, key="selection_condition")
    return feats


# --------------------------------------------------------------------------- #
# Stage 4/5 — train
# --------------------------------------------------------------------------- #


def train(features: "pd.DataFrame", config: "object | None" = None) -> "ModelBundle":
    """Fit the gradient-boosted-tree baseline on the labelled feature matrix."""
    from .baseline_model import ModelConfig, train_model

    return train_model(features, config or ModelConfig())


# --------------------------------------------------------------------------- #
# Stage 6 — evaluate
# --------------------------------------------------------------------------- #


def gsm_coverage(model: "Model", corpus_csv: Path | None = None) -> dict[str, object]:
    """How many corpus convergence targets the GSM can even represent.

    Reads the *raw* corpus (with gene_name) and checks each positive's b-number
    against the model. The headline limitation number: a flux model is blind to
    every regulatory/non-metabolic hotspot, so this bounds the achievable ceiling.
    """
    import pandas as pd

    raw = pd.read_csv(corpus_csv or CORPUS_CSV)
    in_model = [str(b) in model.genes for b in raw["gene"]]
    covered = int(sum(in_model))
    total = int(len(raw))
    return {
        "targets_total": total,
        "targets_in_gsm": covered,
        "coverage_fraction": round(covered / total, 3) if total else float("nan"),
        "note": (
            "Regulatory/non-metabolic convergent ALE hotspots (rpoB, rpoC, rho, "
            "hns, ...) are out of any metabolic model by construction; this corpus "
            "is pre-filtered to in-GSM genes, so coverage here is 1.0 by design and "
            "the real ceiling is set by how many literature targets were metabolic "
            "in the first place (see corpus/CURATION.md 'GSM-coverage gap')."
        ),
    }


def _row_chassis(features: "pd.DataFrame", corpus: "pd.DataFrame") -> list[object]:
    """Per-row chassis label (organism/base_strain), mapped via selection_condition."""
    cond2chassis = (
        corpus.dropna(subset=["selection_condition"])
        .assign(_c=lambda d: d["organism"].astype(str) + " " + d["base_strain"].astype(str))
        .groupby("selection_condition")["_c"]
        .first()
        .to_dict()
    )
    return [cond2chassis.get(str(c), "unknown") for c in features["selection_condition"]]


def evaluate(
    features: "pd.DataFrame",
    corpus: "pd.DataFrame",
    config: "object | None" = None,
) -> dict[str, object]:
    """Leave-one-condition-out + cluster-aware CV recall, vs uniform-essential base.

    * **Leave-one-condition-out (LOCO):** train on the other conditions, rank genes
      for the held-out condition, report top-k recall of its convergent genes vs
      the baseline. The honest transfer test for a small, condition-diverse corpus.
    * **Cluster-aware CV:** pooled across conditions, whole gene held out per fold
      (homology-leakage-free), measures whether flux features rank ALE-prone genes
      above the essential-gene baseline at all.
    """
    import pandas as pd

    from .baseline_model import (
        ModelConfig,
        evaluate_held_out_chassis,
        cross_validate_cluster_aware,
    )

    cfg = config or ModelConfig()

    # LOCO: treat each selection_condition as its own held-out "group". Reuse the
    # held-out-chassis harness by passing selection_condition as the group label.
    loco = evaluate_held_out_chassis(
        features,
        chassis=features["selection_condition"].astype(str).tolist(),
        config=cfg,
    ).rename(columns={"held_out_chassis": "held_out_condition"})

    # Cluster-aware CV: cluster = gene id (homology clustering would group orthologs;
    # gene id is the dependency-free fallback the scaffold documents).
    clusters = features["gene"].astype(str).tolist()
    cl = cross_validate_cluster_aware(features, clusters=clusters, config=cfg, n_splits=5)

    return {"loco": loco, "cluster_cv": cl}


def feature_importances(bundle: "ModelBundle") -> "pd.DataFrame":
    """Gain (LightGBM) or permutation-free fallback importances as a tidy frame."""
    import numpy as np
    import pandas as pd

    est = bundle.estimator
    cols = bundle.design_columns
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame({"feature": cols, "importance": [float("nan")] * len(cols)})
    return (
        pd.DataFrame({"feature": cols, "importance": np.asarray(imp, dtype=float)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )



# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(
    gsm: str = DEFAULT_GSM,
    config: "object | None" = None,
    use_cache: bool = True,
) -> PipelineResult:
    """Run every stage and return a ``PipelineResult``."""
    from .baseline_model import ModelConfig
    from .feature_engineer import FeatureConfig

    model = load_gsm(gsm)
    corpus = load_corpus()
    features = build_features(model, corpus, config=FeatureConfig(), use_cache=use_cache)
    bundle = train(features, config or ModelConfig())
    ev = evaluate(features, corpus, config or ModelConfig())
    importances = feature_importances(bundle)
    coverage = gsm_coverage(model)

    n_pos = int(features["mutated"].fillna(False).astype("boolean").sum())
    meta = {
        "gsm": gsm,
        "n_genes": len(model.genes),
        "n_conditions": int(features["selection_condition"].nunique()),
        "n_rows": int(len(features)),
        "n_positive": n_pos,
        "backend": bundle.backend,
        "n_train": bundle.n_train,
    }
    return PipelineResult(
        features=features,
        bundle=bundle,
        loco=ev["loco"],
        cluster_cv=ev["cluster_cv"],
        importances=importances,
        gsm_coverage=coverage,
        meta=meta,
    )


def _summary_dict(res: PipelineResult) -> dict[str, object]:
    """JSON-serialisable summary of a run for the CLI / metrics artifact."""
    return {
        "meta": res.meta,
        "gsm_coverage": res.gsm_coverage,
        "leave_one_condition_out": json.loads(res.loco.to_json(orient="records")),
        "cluster_aware_cv": json.loads(res.cluster_cv.to_json(orient="records")),
        "top_features": json.loads(res.importances.head(12).to_json(orient="records")),
    }


def main() -> None:
    """CLI entry point: run the pipeline and print a JSON summary."""
    res = run()
    print(json.dumps(_summary_dict(res), indent=2, default=str))


if __name__ == "__main__":
    main()
