"""GSM-flux feature engineering: per gene, per selection condition  (milestone 15.3).

Turns a chassis genome-scale metabolic model (GSM) + a set of ALE selection
conditions into the project's **canonical feature table** — one row per
``(gene, selection_condition)`` with flux-derived predictors the baseline model
(15.4) trains on. The supervised label (did this gene acquire a causal mutation
under this selection?) is joined from the canonical mutation table produced by
``data_loader.normalize_mutations`` / ``load_aledb``.

What lives here
---------------
* ``parse_selection_condition``  free-text ALEdb condition -> ``SelectionEnvironment``.
* ``apply_environment``          push a ``SelectionEnvironment`` onto a cobra Model
  (media exchange bounds; temperature is carried as a feature, not simulated).
* ``gpr_complexity``             structural features from a gene-reaction-rule string.
* ``gene_structural_features``   per-gene GPR / reaction-degree features (no solve).
* ``gene_flux_features``         per-gene FVA span, flux-at-optimum, essentiality
  (cobrapy solves; the same GSM flux features used in the upstream chassis work).
* ``flux_coupling_features``     directional flux-coupling degree (expensive; proxy
  here, real FCA is a TODO).
* ``build_feature_matrix``       the pipeline: genes x conditions -> feature frame.
* ``attach_labels``              join the mutation table -> boolean ``mutated`` label.

Design notes
------------
* **Dependency-light import.** ``cobra``/``pandas``/``numpy`` are imported *inside*
  the functions that need them (and under ``TYPE_CHECKING`` for annotations), so
  importing this module during the CI smoke test never requires a solver stack.
  The pure-Python helpers (``parse_selection_condition``, ``gpr_complexity``,
  ``gene_structural_features``) run with no third-party deps at all.
* **Feature schema is explicit and stable.** ``FEATURE_COLUMNS`` is the contract
  the baseline model relies on, mirroring ``data_loader.CANONICAL_COLUMNS``.
* **FBA has no temperature.** ALE stressors like "42C" cannot be simulated by an
  unmodified FBA model. Temperature + a coarse stressor class are carried as
  *condition* features so the model can still condition on them; modelling the
  thermal effect on enzyme capacity is left to a v2 (FoldX/AF features, see the
  design note's "Optional v2").

Citations (from ``research_notes/15_ale_outcome_predictor.md`` LINKS)
--------------------------------------------------------------------
* Anand et al. 2021, bioRxiv 2021.07.19.452699 — aggregated-ALE workflow; the
  hitchhiker filter feeding the label set lives in ``data_loader``.
* Zampieri et al. 2019, PLOS Comput Biol 10.1371/journal.pcbi.1007084 — the
  GSM-feature + ML lane this feature set sits in.
* Phaneuf et al. 2019, NAR 47(D1):D1164 — ALEdb 1.0, the training corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type hints; keeps runtime import dep-light
    import pandas as pd
    from cobra import Model

__all__ = [
    "SelectionEnvironment",
    "FeatureConfig",
    "FEATURE_COLUMNS",
    "STRESS_CLASSES",
    "parse_selection_condition",
    "apply_environment",
    "gpr_complexity",
    "gene_structural_features",
    "gene_flux_features",
    "flux_coupling_features",
    "build_feature_matrix",
    "attach_labels",
    "validate_features",
]

# --------------------------------------------------------------------------- #
# Canonical feature schema
# --------------------------------------------------------------------------- #
# One row per (gene, selection_condition). pandas dtype *strings* only, same
# idiom as data_loader.CANONICAL_COLUMNS, so importing this module needs no
# pandas. Nullable Int64/Float64/boolean keep solver-failures / missing GPRs
# NA-safe rather than collapsing to NaN-float or False.
FEATURE_COLUMNS: dict[str, str] = {
    # --- join keys ---------------------------------------------------------
    "gene": "string",
    "selection_condition": "string",
    "experiment_id": "string",        # carried through for label joining
    # --- condition (environment) features ----------------------------------
    "temperature_c": "Float64",
    "stress_class": "string",         # see STRESS_CLASSES; "none" if unstressed
    "is_minimal_media": "boolean",
    # --- structural (GPR / topology) features — no solve required -----------
    "n_reactions": "Int64",           # reactions this gene participates in
    "gpr_n_isozyme_groups": "Int64",  # max OR-arity across the gene's reactions
    "gpr_max_complex_size": "Int64",  # max AND-arity across the gene's reactions
    "is_isozyme": "boolean",          # gene appears in an OR group anywhere
    "is_in_complex": "boolean",       # gene appears in an AND group anywhere
    "subsystem_count": "Int64",       # distinct subsystems the gene spans
    # --- flux features — cobrapy solves at the condition's medium -----------
    "is_essential": "boolean",        # KO growth / WT growth < essential fraction
    "ko_growth_ratio": "Float64",     # single-gene-deletion growth / WT growth
    "fva_max_abs_flux": "Float64",    # max |flux| over the gene's reactions (FVA)
    "fva_max_span": "Float64",        # max (ub - lb) over the gene's reactions (FVA)
    "carries_flux_at_optimum": "boolean",  # any reaction with |v| > tol in pFBA
    "n_blocked_reactions": "Int64",   # gene's reactions blocked under this medium
    # --- coupling (expensive; proxy until real FCA) ------------------------
    "max_flux_coupling_degree": "Int64",
    # --- supervised label (filled by attach_labels) ------------------------
    "mutated": "boolean",
}

# Coarse stressor buckets. ALEdb free-text is mapped onto these so the model
# conditions on a small categorical instead of raw strings.
# TODO: confirm/extend against the real ALEdb condition vocabulary.
STRESS_CLASSES: tuple[str, ...] = (
    "none",
    "temperature",     # e.g. "42C", "heat", "thermal"
    "solvent",         # e.g. "n-butanol", "isobutanol", "ethanol"
    "osmotic",         # e.g. "NaCl", "salt", "sorbitol"
    "ph",              # e.g. "acid", "low pH", "alkaline"
    "oxidative",       # e.g. "H2O2", "paraquat"
    "carbon_limited",  # e.g. "glycerol-min", "minimal", "C-limited"
    "antibiotic",      # e.g. "ampicillin", "tetracycline"
    "aromatic",        # aromatic-acid / phenolic stress: benzoate, ferulic, p-coumaric
    "other",           # matched a stress keyword but not a known bucket
)

# keyword -> stress_class, scanned over the lower-cased condition string.
_STRESS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("butanol", "solvent"),
    ("ethanol", "solvent"),
    ("isobutanol", "solvent"),
    ("solvent", "solvent"),
    ("nacl", "osmotic"),
    ("salt", "osmotic"),
    ("sorbitol", "osmotic"),
    ("osmot", "osmotic"),
    ("benzoate", "aromatic"),
    ("benzoic", "aromatic"),
    ("ferulic", "aromatic"),
    ("coumaric", "aromatic"),
    ("phenol", "aromatic"),
    ("aromatic", "aromatic"),
    ("acid", "ph"),
    ("alkaline", "ph"),
    ("h2o2", "oxidative"),
    ("peroxide", "oxidative"),
    ("paraquat", "oxidative"),
    ("oxidat", "oxidative"),
    ("ampicillin", "antibiotic"),
    ("tetracycline", "antibiotic"),
    ("kanamycin", "antibiotic"),
    ("antibiotic", "antibiotic"),
    ("glycerol", "carbon_limited"),
    ("minimal", "carbon_limited"),
    ("c-limit", "carbon_limited"),
    ("carbon", "carbon_limited"),
)

# Tokens that signal a minimal / defined medium in the free-text condition.
_MINIMAL_MEDIA_HINTS: tuple[str, ...] = ("minimal", "m9", "defined", "-min")


# --------------------------------------------------------------------------- #
# Selection environment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SelectionEnvironment:
    """A parsed ALE selection condition.

    ``raw`` keeps the original free text so nothing is lost; the structured
    fields are best-effort heuristic extractions (see ``parse_selection_condition``).
    """

    raw: str
    temperature_c: float | None = None
    stress_class: str = "none"
    is_minimal_media: bool = False
    exchange_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureConfig:
    """Thresholds / knobs for the flux feature pass.

    Defaults are conservative starting points; the ablation notebook
    (research_notes/15 DELIVERABLE) is where these get tuned.
    """

    essential_growth_fraction: float = 0.01   # KO/WT below this -> "essential"
    flux_tolerance: float = 1e-6              # |v| above this -> "carries flux"
    fva_fraction_of_optimum: float = 0.9      # FVA loopless-window optimum fraction
    compute_flux_coupling: bool = False       # off by default — expensive (see note)
    processes: int = 1                        # cobrapy parallelism for FVA/deletion


_TEMPERATURE_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:°|deg|degrees?\s*)?c\b")


def parse_selection_condition(text: str | None) -> SelectionEnvironment:
    """Heuristically parse an ALEdb free-text condition into a ``SelectionEnvironment``.

    Extracts (a) a temperature in Celsius if one is spelled out (``"42C"``,
    ``"42 °C"``, ``"37 degrees C"``), (b) a coarse ``stress_class`` from
    ``_STRESS_KEYWORDS`` (first match wins), and (c) a minimal-media flag.

    This is intentionally a heuristic, not a parser of a controlled vocabulary —
    ALEdb's condition strings are unconstrained free text.

    TODO: replace with a controlled-vocabulary lookup once the
    real export's distinct condition strings are enumerated; carbon source and
    explicit exchange-bound overrides (``exchange_bounds``) are left empty here
    because they need the model's exchange-reaction IDs to be meaningful.
    """
    if not text:
        return SelectionEnvironment(raw="")
    lower = text.lower()

    temperature_c: float | None = None
    m = _TEMPERATURE_RE.search(lower)
    if m:
        value = float(m.group(1))
        # Plausibility guard: ALE temperatures live in a biological band; ignore
        # spurious matches (year-like or concentration-like numbers).
        if 4.0 <= value <= 70.0:
            temperature_c = value

    stress_class = "none"
    for keyword, bucket in _STRESS_KEYWORDS:
        if keyword in lower:
            stress_class = bucket
            break
    # A temperature well off the 37 °C reference reads as thermal stress even if
    # no explicit stress keyword was present.
    if stress_class == "none" and temperature_c is not None and abs(temperature_c - 37.0) >= 3.0:
        stress_class = "temperature"

    is_minimal_media = any(hint in lower for hint in _MINIMAL_MEDIA_HINTS)

    return SelectionEnvironment(
        raw=text,
        temperature_c=temperature_c,
        stress_class=stress_class,
        is_minimal_media=is_minimal_media,
    )


def apply_environment(model: "Model", env: SelectionEnvironment) -> None:
    """Apply a ``SelectionEnvironment``'s exchange bounds onto ``model`` in place.

    Only the explicit ``exchange_bounds`` overrides are pushed — temperature and
    stress class are *features*, not simulable constraints on an unmodified FBA
    model (see module "Design notes"). Call inside a ``with model:`` block so the
    bounds revert after the condition's features are computed.

    TODO: wire a media -> exchange-reaction mapping (e.g. M9 +
    a carbon source -> open the matching ``EX_*`` reactions, close the rest) once
    the chassis model's exchange IDs are known. Until then this is a no-op when
    ``exchange_bounds`` is empty.
    """
    for reaction_id, (lower, upper) in env.exchange_bounds.items():
        if reaction_id in model.reactions:
            rxn = model.reactions.get_by_id(reaction_id)
            rxn.bounds = (lower, upper)


# --------------------------------------------------------------------------- #
# Structural features (pure Python — no solver, no third-party deps)
# --------------------------------------------------------------------------- #

_GENE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")
_BOOL_TOKENS = {"and", "or", "(", ")"}


def gpr_complexity(rule: str | None) -> dict[str, int | bool]:
    """Structural features of one gene-reaction-rule (GPR) string.

    Returns ``n_genes`` (distinct gene tokens), ``n_isozyme_groups`` (top-level
    OR-arity), ``max_complex_size`` (largest AND group = subunits in a complex),
    ``is_isozyme`` (any OR present), ``is_in_complex`` (any AND present).

    Parsing is structural, not a full boolean-expression evaluator: it splits on
    top-level ``or`` to count isozyme groups and counts ``and`` members within
    the largest group. Matches cobrapy GPR syntax (``"b0001 and b0002 or b0003"``,
    parenthesised groups). Empty / missing rules return all-zero.
    """
    if not rule or not rule.strip():
        return {
            "n_genes": 0,
            "n_isozyme_groups": 0,
            "max_complex_size": 0,
            "is_isozyme": False,
            "is_in_complex": False,
        }

    # Tokenise into genes vs boolean operators / parens.
    raw_tokens = re.findall(r"\(|\)|\b(?:and|or)\b|[A-Za-z0-9_.\-]+", rule.lower())
    genes = {t for t in raw_tokens if t not in _BOOL_TOKENS}

    # Split on top-level OR (parenthesis depth 0) to count isozyme groups.
    groups: list[list[str]] = [[]]
    depth = 0
    for tok in raw_tokens:
        if tok == "(":
            depth += 1
            groups[-1].append(tok)
        elif tok == ")":
            depth -= 1
            groups[-1].append(tok)
        elif tok == "or" and depth == 0:
            groups.append([])
        else:
            groups[-1].append(tok)

    n_isozyme_groups = len([g for g in groups if any(t not in _BOOL_TOKENS for t in g)])
    # Largest AND group = max distinct genes inside any single isozyme group.
    max_complex_size = 0
    for g in groups:
        members = {t for t in g if t not in _BOOL_TOKENS}
        max_complex_size = max(max_complex_size, len(members))

    has_or = "or" in raw_tokens
    has_and = "and" in raw_tokens
    return {
        "n_genes": len(genes),
        "n_isozyme_groups": n_isozyme_groups,
        "max_complex_size": max_complex_size,
        "is_isozyme": has_or,
        "is_in_complex": has_and,
    }


def gene_structural_features(model: "Model", gene_id: str) -> dict[str, object]:
    """Per-gene GPR / topology features that need no solve.

    Aggregates ``gpr_complexity`` across every reaction the gene participates in
    (max OR-arity, max AND-arity, any-isozyme, any-complex) plus the reaction and
    subsystem counts. Returns zeros for a gene absent from the model.
    """
    if gene_id not in model.genes:
        return {
            "n_reactions": 0,
            "gpr_n_isozyme_groups": 0,
            "gpr_max_complex_size": 0,
            "is_isozyme": False,
            "is_in_complex": False,
            "subsystem_count": 0,
        }
    gene = model.genes.get_by_id(gene_id)
    reactions = list(gene.reactions)

    n_isozyme_groups = 0
    max_complex_size = 0
    is_isozyme = False
    is_in_complex = False
    subsystems: set[str] = set()
    for rxn in reactions:
        c = gpr_complexity(getattr(rxn, "gene_reaction_rule", "") or "")
        n_isozyme_groups = max(n_isozyme_groups, int(c["n_isozyme_groups"]))
        max_complex_size = max(max_complex_size, int(c["max_complex_size"]))
        is_isozyme = is_isozyme or bool(c["is_isozyme"])
        is_in_complex = is_in_complex or bool(c["is_in_complex"])
        if getattr(rxn, "subsystem", None):
            subsystems.add(rxn.subsystem)

    return {
        "n_reactions": len(reactions),
        "gpr_n_isozyme_groups": n_isozyme_groups,
        "gpr_max_complex_size": max_complex_size,
        "is_isozyme": is_isozyme,
        "is_in_complex": is_in_complex,
        "subsystem_count": len(subsystems),
    }


# --------------------------------------------------------------------------- #
# Flux features (cobrapy solves)
# --------------------------------------------------------------------------- #


def gene_flux_features(
    model: "Model",
    genes: list[str] | None = None,
    config: FeatureConfig | None = None,
) -> "pd.DataFrame":
    """Per-gene flux features at the model's *current* medium.

    Computes, for every gene (or the supplied subset): FVA span/abs-flux over the
    gene's reactions, whether any reaction carries flux in a parsimonious-FBA
    optimum, the blocked-reaction count, and single-gene-deletion essentiality
    (KO growth / WT growth). Caller is responsible for having applied the
    condition's medium first (see ``apply_environment``).

    Returns a frame indexed by ``gene`` with one column per flux feature.
    """
    import pandas as pd
    from cobra.flux_analysis import (
        flux_variability_analysis,
        pfba,
        single_gene_deletion,
    )

    cfg = config or FeatureConfig()
    gene_ids = genes if genes is not None else [g.id for g in model.genes]

    # WT reference growth + parsimonious flux distribution (one solve each).
    wt_solution = model.optimize()
    wt_growth = float(wt_solution.objective_value or 0.0)
    try:
        pfba_fluxes = pfba(model).fluxes
    except Exception:  # solver/feasibility hiccup — fall back to the FBA optimum
        pfba_fluxes = wt_solution.fluxes

    # FVA across all reactions once; slice per gene below.
    fva = flux_variability_analysis(
        model,
        fraction_of_optimum=cfg.fva_fraction_of_optimum,
        processes=cfg.processes,
    )

    # Single-gene-deletion essentiality in one batched call.
    deletion = single_gene_deletion(model, gene_list=gene_ids, processes=cfg.processes)
    ko_growth = _deletion_growth_map(deletion)

    rows: list[dict[str, object]] = []
    for gid in gene_ids:
        gene = model.genes.get_by_id(gid) if gid in model.genes else None
        reaction_ids = [r.id for r in gene.reactions] if gene is not None else []

        max_abs_flux = 0.0
        max_span = 0.0
        carries_flux = False
        n_blocked = 0
        for rid in reaction_ids:
            lb = float(fva.at[rid, "minimum"]) if rid in fva.index else 0.0
            ub = float(fva.at[rid, "maximum"]) if rid in fva.index else 0.0
            max_abs_flux = max(max_abs_flux, abs(lb), abs(ub))
            max_span = max(max_span, ub - lb)
            if abs(ub) <= cfg.flux_tolerance and abs(lb) <= cfg.flux_tolerance:
                n_blocked += 1
            if rid in pfba_fluxes.index and abs(float(pfba_fluxes[rid])) > cfg.flux_tolerance:
                carries_flux = True

        ko = ko_growth.get(gid)
        ratio = (ko / wt_growth) if (ko is not None and wt_growth > 0) else None
        is_essential = (ratio is not None and ratio < cfg.essential_growth_fraction)

        rows.append(
            {
                "gene": gid,
                "is_essential": is_essential if ratio is not None else pd.NA,
                "ko_growth_ratio": ratio if ratio is not None else pd.NA,
                "fva_max_abs_flux": max_abs_flux,
                "fva_max_span": max_span,
                "carries_flux_at_optimum": carries_flux,
                "n_blocked_reactions": n_blocked,
            }
        )
    return pd.DataFrame(rows).set_index("gene")


def _deletion_growth_map(deletion: "pd.DataFrame") -> dict[str, float]:
    """Flatten cobrapy's single_gene_deletion frame to ``{gene_id: growth}``.

    cobrapy returns a frame whose ``ids`` column holds a frozenset per row; for
    single deletions that set has one member. ``growth`` holds the KO objective.
    """
    out: dict[str, float] = {}
    if "ids" not in deletion.columns or "growth" not in deletion.columns:
        return out
    for ids, growth in zip(deletion["ids"], deletion["growth"]):
        for gid in ids:  # single-member frozenset for single-gene deletions
            out[str(gid)] = float(growth)
    return out


def flux_coupling_features(
    model: "Model",
    genes: list[str] | None = None,
    config: FeatureConfig | None = None,
) -> "pd.DataFrame":
    """Per-gene flux-coupling degree (number of fully-coupled reaction partners).

    Real flux-coupling analysis (FCA, Burgard 2004) is not in cobrapy core and is
    expensive on genome-scale models; this is a placeholder that returns a zero
    degree column with the right shape so ``build_feature_matrix`` stays whole
    when coupling is switched off.

    TODO (15.x): port a proper FCA — e.g. via ``cobamp`` / ``fastFVA`` flux-ratio
    bounds — and cache per (model, medium); the design note flags flux-coupling
    on iML1515 as expensive-but-tractable-with-caching.
    """
    import pandas as pd

    gene_ids = genes if genes is not None else [g.id for g in model.genes]
    return pd.DataFrame(
        {"max_flux_coupling_degree": [0] * len(gene_ids)},
        index=pd.Index(gene_ids, name="gene"),
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build_feature_matrix(
    model: "Model",
    conditions: "list[str] | dict[str, str]",
    genes: list[str] | None = None,
    config: FeatureConfig | None = None,
    env_resolver: "object | None" = None,
) -> "pd.DataFrame":
    """Build the genes x conditions feature matrix for one chassis model.

    Parameters
    ----------
    model:
        The chassis GSM (cobrapy ``Model``). Read from BiGG per the design note.
    conditions:
        Either a list of free-text ``selection_condition`` strings, or a mapping
        ``{experiment_id: selection_condition}`` so the label join can key on the
        experiment. A bare list uses the condition string itself as the key.
    genes:
        Restrict to these gene IDs (default: all model genes).
    config:
        ``FeatureConfig`` knobs; flux coupling is computed only if
        ``config.compute_flux_coupling`` is True.
    env_resolver:
        Optional ``callable(condition_text) -> SelectionEnvironment``. Defaults to
        ``parse_selection_condition`` (heuristic temp/stress, *no* medium swap).
        Pass ``media.make_env_resolver(model)`` to make the flux features genuinely
        condition-specific by applying each condition's growth medium — without
        this the per-condition flux features are all computed at the model's
        default medium and only the condition (temperature / stress) columns vary.

    Returns
    -------
    A frame with the columns of ``FEATURE_COLUMNS`` except ``mutated`` (filled by
    ``attach_labels``). Structural features are computed once and reused across
    conditions; flux features are recomputed per condition under that medium.
    """
    import pandas as pd

    cfg = config or FeatureConfig()
    gene_ids = genes if genes is not None else [g.id for g in model.genes]
    resolve = env_resolver or parse_selection_condition

    if isinstance(conditions, dict):
        condition_items = list(conditions.items())            # (experiment_id, text)
    else:
        condition_items = [(c, c) for c in conditions]        # text doubles as key

    # Structural features are condition-independent — compute once.
    structural = {gid: gene_structural_features(model, gid) for gid in gene_ids}

    frames: list[pd.DataFrame] = []
    for experiment_id, condition_text in condition_items:
        env = resolve(condition_text)
        with model:  # bounds revert on exit
            apply_environment(model, env)
            flux = gene_flux_features(model, genes=gene_ids, config=cfg)
            coupling = (
                flux_coupling_features(model, genes=gene_ids, config=cfg)
                if cfg.compute_flux_coupling
                else None
            )

        block: list[dict[str, object]] = []
        for gid in gene_ids:
            row: dict[str, object] = {
                "gene": gid,
                "selection_condition": condition_text,
                "experiment_id": experiment_id,
                "temperature_c": env.temperature_c if env.temperature_c is not None else pd.NA,
                "stress_class": env.stress_class,
                "is_minimal_media": env.is_minimal_media,
            }
            row.update(structural[gid])
            if gid in flux.index:
                row.update(flux.loc[gid].to_dict())
            row["max_flux_coupling_degree"] = (
                int(coupling.loc[gid, "max_flux_coupling_degree"])
                if (coupling is not None and gid in coupling.index)
                else 0
            )
            block.append(row)
        frames.append(pd.DataFrame(block))

    features = pd.concat(frames, ignore_index=True)
    features["mutated"] = pd.NA  # label slot; filled by attach_labels
    return _coerce_feature_dtypes(features)


def attach_labels(
    features: "pd.DataFrame",
    mutations: "pd.DataFrame",
    *,
    key: str = "selection_condition",
) -> "pd.DataFrame":
    """Set the boolean ``mutated`` label on ``features`` from the mutation table.

    A ``(key, gene)`` pair present in the (already hitchhiker-filtered) mutation
    table is a positive; every other ``(key, gene)`` row is a negative. ``key``
    is ``"selection_condition"`` by default, or ``"experiment_id"`` if the
    conditions were supplied as an ``{experiment_id: text}`` mapping.

    ``mutations`` is expected to be a canonical table from ``data_loader`` (see
    ``data_loader.CANONICAL_COLUMNS``); only its ``gene`` + ``key`` columns are
    used here.

    TODO: the negative set currently = "every gene-condition not
    observed mutated", which conflates "truly not selected" with "not yet seen in
    ALEdb". The design note's held-out-chassis split is what keeps this honest;
    consider a presence-only / PU-learning framing in 15.4 instead of hard zeros.
    """
    import pandas as pd

    out = features.copy()
    if key not in mutations.columns or "gene" not in mutations.columns:
        out["mutated"] = pd.NA
        return _coerce_feature_dtypes(out)

    positives = set(
        zip(
            mutations[key].astype("string"),
            mutations["gene"].astype("string"),
        )
    )
    out["mutated"] = [
        (str(k), str(g)) in positives
        for k, g in zip(out[key].astype("string"), out["gene"].astype("string"))
    ]
    return _coerce_feature_dtypes(out)


def _coerce_feature_dtypes(df: "pd.DataFrame") -> "pd.DataFrame":
    """Reorder to ``FEATURE_COLUMNS`` and coerce dtypes (NA-safe nullable types)."""
    import pandas as pd

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(FEATURE_COLUMNS)]
    for col, dtype in FEATURE_COLUMNS.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            df.attrs.setdefault("coerce_failed", []).append(col)
    return df


def validate_features(df: "pd.DataFrame") -> None:
    """Assert the frame carries exactly the canonical feature columns."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"Non-canonical feature table. missing={missing} extra={extra}. "
            "Build it with build_feature_matrix()."
        )
