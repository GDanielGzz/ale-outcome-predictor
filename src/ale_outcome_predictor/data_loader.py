"""ALEdb 1.0 import + mutation-table normalisation  (milestone 15.2).

Turns a *local* ALEdb export into the project's **canonical mutation table** — a
tidy, one-row-per-mutation frame with stable column names and dtypes that the
feature engineer (15.3) and baseline model (15.4) can rely on without caring how
ALEdb happens to spell its columns.

What lives here
---------------
* ``load_raw_export``   read a local ALEdb dump (CSV / TSV / SQLite) from ``data/``.
* ``normalize_mutations``  rename + coerce the raw dump to the canonical schema.
* ``filter_hitchhikers``  drop passenger / clonal-interference mutations, keeping
  the causal candidates (reuse the Anand-2021 logic rather than re-inventing it).
* ``load_aledb``       the convenience pipeline wiring the three together.

Design notes
------------
* **Dependency-light import.** pandas is only touched *inside* the functions and
  under ``TYPE_CHECKING`` for annotations, so importing this module (e.g. during
  the CI smoke test) never requires pandas/cobra to be installed.
* **No corpus is shipped.** ALEdb bulk-export licensing is unverified (see the
  design note's BLOCKERS), so the loader reads whatever the user pulled locally
  via ``data/HOWTO_PULL.md`` and never bundles the data itself. ``data/`` is
  gitignored.
* **Column aliases are unverified guesses.** ``ALEDB_COLUMN_ALIASES`` lists the
  *candidate* raw names per canonical field. They are marked TODO until checked
  against a real ALEdb 1.0 export — the exact header spelling is set there, not
  here.

ALEdb 2.0 watch  (Idea Bank, re-audited Run 35 · 2026-05-29)
------------------------------------------------------------
ALEdb 1.0 (Phaneuf et al., *NAR* 2019, 47(D1):D1164) remains the latest release;
no v2 announcement in PubMed / NAR / aledb.org as of 2026-05-29. If a 2026/2027
v2 lands with normalised cross-organism entries, this loader is the swap point:
add a ``ALEDB_VERSION == "2.0"`` branch to ``normalize_mutations`` and re-point
``ALEDB_COLUMN_ALIASES``. Re-audit on the next mod-2 prune.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type hints; keeps runtime import dep-light
    import pandas as pd

__all__ = [
    "ALEDB_HOMEPAGE",
    "ALEDB_VERSION",
    "ALEDB_CITATION",
    "ALEDB_2_0_WATCH",
    "CANONICAL_COLUMNS",
    "ALEDB_COLUMN_ALIASES",
    "default_export_path",
    "load_raw_export",
    "normalize_mutations",
    "filter_hitchhikers",
    "load_aledb",
    "validate_canonical",
]

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

ALEDB_HOMEPAGE = "http://aledb.org"
ALEDB_VERSION = "1.0"
ALEDB_CITATION = (
    "Phaneuf, P.V. et al. (2019) ALEdb 1.0: a database of mutations from "
    "adaptive laboratory evolution experimentation. Nucleic Acids Research "
    "47(D1):D1164-D1171. doi:10.1093/nar/gky983"
)
ALEDB_2_0_WATCH = (
    "ALEdb 1.0 (2019) is the latest release as of 2026-05-29; no v2 found in "
    "PubMed / NAR / aledb.org. If v2 ships, branch normalize_mutations() on "
    "ALEDB_VERSION and re-point ALEDB_COLUMN_ALIASES. Re-audit next mod-2 prune."
)

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
# One row per (sample, mutation). pandas dtype strings only — no pandas import
# needed to read this. "string" / "boolean" / nullable "Int64"/"Float64" are
# chosen so ALEdb's frequent missing fields stay NA-safe rather than coercing to
# NaN-floats or empty strings.
CANONICAL_COLUMNS: dict[str, str] = {
    # --- experiment / selection context -----------------------------------
    "experiment_id": "string",       # ALEdb experiment identifier
    "publication": "string",         # source citation or DOI
    "organism": "string",            # e.g. "Escherichia coli"
    "base_strain": "string",         # e.g. "K-12 MG1655"
    "selection_condition": "string",  # free-text media+stress; parsed in 15.3
    "media": "string",
    "stressor": "string",            # e.g. "42C", "n-butanol", "glycerol-min"
    "temperature_c": "Float64",
    # --- sample (flask / isolate) ------------------------------------------
    "lineage": "string",             # flask / lineage id within the experiment
    "generation": "Int64",           # cumulative generations (or flask index)
    "sample_type": "string",         # "population" | "clone"
    # --- mutation ----------------------------------------------------------
    "gene": "string",
    "locus_tag": "string",
    "genomic_position": "Int64",
    "ref_allele": "string",
    "alt_allele": "string",
    "mutation_type": "string",       # breseq class: SNP/INS/DEL/MOB/AMP/SUB/CON
    "coding_effect": "string",       # regulatory|synonymous|missense|nonsense|frameshift|...
    "frequency": "Float64",          # 0..1 allele frequency in the sample
    "is_key_mutation": "boolean",    # ALEdb causal-candidate flag (vs hitchhiker)
}

# Canonical field -> candidate raw column names in an ALEdb 1.0 export.
# TODO: verify each tuple against a real export header; ALEdb's
# CSV vs SQLite-dump column names differ and are not documented field-for-field.
ALEDB_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "experiment_id": ("experiment", "experiment_id", "ale_experiment"),
    "publication": ("publication", "reference", "doi", "source"),
    "organism": ("organism", "species", "strain_species"),
    "base_strain": ("strain", "base_strain", "host_strain"),
    "selection_condition": ("condition", "environment", "selection"),
    "media": ("media", "medium"),
    "stressor": ("stress", "stressor", "perturbation"),
    "temperature_c": ("temperature", "temp_c", "temperature_c"),
    "lineage": ("flask", "lineage", "population_id"),
    "generation": ("generation", "cumulative_generations", "flask_number"),
    "sample_type": ("sample_type", "isolate_type", "clonality"),
    "gene": ("gene", "gene_name"),
    "locus_tag": ("locus_tag", "b_number", "bnum"),
    "genomic_position": ("position", "genome_position", "coord"),
    "ref_allele": ("ref", "reference_base", "ref_seq"),
    "alt_allele": ("alt", "new_base", "mutation_seq"),
    "mutation_type": ("mutation_type", "type", "variant_type"),
    "coding_effect": ("snp_type", "annotation", "effect", "consequence"),
    "frequency": ("frequency", "freq", "allele_frequency"),
    "is_key_mutation": ("key_mutation", "is_key", "causal"),
}

# breseq / ALEdb annotation strings -> canonical coding_effect buckets.
# TODO: extend once the real export's annotation vocabulary is confirmed.
_CODING_EFFECT_MAP: dict[str, str] = {
    "synonymous": "synonymous",
    "nonsynonymous": "missense",
    "missense": "missense",
    "nonsense": "nonsense",
    "stop": "nonsense",
    "intergenic": "regulatory",
    "noncoding": "regulatory",
    "pseudogene": "regulatory",
    "frameshift": "frameshift",
    "indel": "frameshift",
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def default_export_path() -> Path:
    """Return the conventional local ALEdb export path: ``<repo>/data/aledb_export``.

    No file is guaranteed to exist there — ``data/`` is gitignored and populated
    by the user per ``data/HOWTO_PULL.md``.
    """
    # repo_root = .../ale-outcome-predictor ; this file is src/ale_outcome_predictor/
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "aledb_export"


def load_raw_export(path: str | Path | None = None) -> "pd.DataFrame":
    """Load a local ALEdb dump (CSV / TSV / SQLite) into a raw DataFrame.

    Parameters
    ----------
    path:
        File to read. Defaults to ``default_export_path()`` with a ``.csv``
        suffix probe. ``.csv``/``.tsv`` are read as delimited text; ``.sqlite``/
        ``.db`` read the ``mutations`` table (TODO: confirm table name).

    Raises
    ------
    FileNotFoundError
        If no local export is present — directs the user to ``data/HOWTO_PULL.md``.
    """
    import pandas as pd

    candidate = Path(path) if path is not None else default_export_path()

    # If a bare stem was given, probe the supported suffixes in priority order.
    if candidate.suffix == "":
        for suffix in (".csv", ".tsv", ".sqlite", ".db"):
            if candidate.with_suffix(suffix).exists():
                candidate = candidate.with_suffix(suffix)
                break

    if not candidate.exists():
        raise FileNotFoundError(
            f"No local ALEdb export at {candidate!s}. ALEdb is not redistributed "
            f"with this repo (licensing unverified). Pull it per data/HOWTO_PULL.md "
            f"({ALEDB_HOMEPAGE}), then re-run."
        )

    suffix = candidate.suffix.lower()
    if suffix in (".csv",):
        return pd.read_csv(candidate)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(candidate, sep="\t")
    if suffix in (".sqlite", ".db"):
        import sqlite3

        # TODO: confirm the mutations table name in the ALEdb
        # SQLite dump; "mutations" is a placeholder.
        with sqlite3.connect(candidate) as conn:
            return pd.read_sql_query("SELECT * FROM mutations", conn)

    raise ValueError(f"Unsupported export format: {candidate.suffix!r}")


def _resolve_aliases(raw_columns: list[str]) -> dict[str, str]:
    """Map raw -> canonical column names using ``ALEDB_COLUMN_ALIASES``.

    Case-insensitive. Returns only the columns that matched; unmatched canonical
    fields are added (empty, NA) downstream by ``normalize_mutations``.
    """
    lower = {c.lower(): c for c in raw_columns}
    rename: dict[str, str] = {}
    for canonical, aliases in ALEDB_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                rename[lower[alias.lower()]] = canonical
                break
    return rename


def normalize_mutations(raw: "pd.DataFrame") -> "pd.DataFrame":
    """Coerce a raw ALEdb dump to ``CANONICAL_COLUMNS`` (names, order, dtypes).

    Steps: alias-rename known columns -> add any missing canonical columns as NA
    -> map the annotation vocabulary into ``coding_effect`` buckets -> coerce
    dtypes. Unknown raw columns are dropped (kept on ``df.attrs['dropped']`` for
    debugging).
    """
    import pandas as pd

    rename = _resolve_aliases(list(raw.columns))
    df = raw.rename(columns=rename)

    dropped = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    df = df.drop(columns=dropped)

    # Add missing canonical columns as all-NA.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Map free-text annotations into canonical coding_effect buckets.
    if "coding_effect" in df.columns:
        df["coding_effect"] = df["coding_effect"].map(_map_coding_effect)

    # Reorder + coerce dtypes.
    df = df[list(CANONICAL_COLUMNS)]
    for col, dtype in CANONICAL_COLUMNS.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Leave un-coercible columns as-is; surface in attrs for inspection.
            df.attrs.setdefault("coerce_failed", []).append(col)

    df.attrs["dropped"] = dropped
    df.attrs["aledb_version"] = ALEDB_VERSION
    return df


def _map_coding_effect(value: object) -> object:
    """Map one raw annotation string to a canonical coding_effect bucket.

    Matches on whole **word tokens**, not substrings, so "nonsynonymous" can
    never be swallowed by the "synonymous" it literally contains (the classic
    trap here). Tries the full normalised string first, then each alphabetic
    word left-to-right — so "nonsynonymous (S450L)" and "stop_gained" both
    resolve correctly. Result is independent of dict iteration order.
    """
    if not isinstance(value, str):
        return value
    token = value.strip().lower()
    if token in _CODING_EFFECT_MAP:
        return _CODING_EFFECT_MAP[token]
    for word in re.findall(r"[a-z]+", token):
        if word in _CODING_EFFECT_MAP:
            return _CODING_EFFECT_MAP[word]
    return token  # unknown vocabulary passes through for later inspection


def filter_hitchhikers(
    df: "pd.DataFrame",
    *,
    keep_key_only: bool = True,
    min_frequency: float = 0.0,
) -> "pd.DataFrame":
    """Drop passenger / clonal-interference mutations, keep causal candidates.

    First pass implemented here: keep rows flagged ``is_key_mutation`` (when the
    flag is present) and above ``min_frequency``. The richer hitchhiker /
    clonal-interference model is intentionally **not** re-invented — reuse the
    Anand et al. 2021 annotation pipeline (bioRxiv 2021.07.19.452699) for the
    co-occurrence / lineage-sharing logic.

    TODO (15.x): port the Anand-2021 hitchhiker filter (shared-lineage
    co-occurrence + convergence test) rather than the frequency proxy below.
    """
    import pandas as pd

    out = df
    if keep_key_only and "is_key_mutation" in out.columns:
        flagged = out["is_key_mutation"]
        # Only filter where the flag is actually populated; if the whole column
        # is NA the export lacks key-mutation calls, so don't drop everything.
        if flagged.notna().any():
            out = out[flagged.fillna(False).astype("boolean")]
    if min_frequency > 0.0 and "frequency" in out.columns:
        out = out[out["frequency"].fillna(0.0) >= min_frequency]
    return out.reset_index(drop=True) if isinstance(out, pd.DataFrame) else out


def load_aledb(
    path: str | Path | None = None,
    *,
    drop_hitchhikers: bool = True,
    min_frequency: float = 0.0,
) -> "pd.DataFrame":
    """End-to-end: read local export -> normalise -> (optionally) de-hitchhike.

    The single entry point the feature engineer (15.3) should import:

    >>> from ale_outcome_predictor.data_loader import load_aledb
    >>> muts = load_aledb()                      # doctest: +SKIP
    >>> muts.columns.tolist() == list(CANONICAL_COLUMNS)   # doctest: +SKIP
    True
    """
    raw = load_raw_export(path)
    norm = normalize_mutations(raw)
    if drop_hitchhikers:
        norm = filter_hitchhikers(norm, min_frequency=min_frequency)
    validate_canonical(norm)
    return norm


def validate_canonical(df: "pd.DataFrame") -> None:
    """Assert the frame carries exactly the canonical columns. Raises ValueError."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"Non-canonical mutation table. missing={missing} extra={extra}. "
            "Run normalize_mutations() first."
        )


# --------------------------------------------------------------------------- #
# Real ALEdb export loader (wide mutation matrix + Resequencing-Runs metadata)
# --------------------------------------------------------------------------- #
# Verified against real ALEdb wide-matrix exports (the public experiments below;
# ALE). The mutations CSV is a WIDE matrix — annotation columns then one column
# per sample labelled "A<ale> F<flask> I<isolate> R<techrep>", with per-caller
# frequency strings like "1.00/1.00" (breseq / GATK). The "Resequencing Runs"
# CSV is the metadata that decodes each sample label into strain / media / temp.

ALEDB_MATRIX_ANNOTATION: tuple[str, ...] = (
    "Reference Seq", "Position", "Mutation Type", "Sequence Change",
    "Gene (Scrollable)", "Gene", "Details",
)
# Minimal NCBI taxid -> organism map (extend as needed).
_TAXID_ORGANISM: dict[str, str] = {
    "511145": "Escherichia coli",
    "83333": "Escherichia coli",
}
_AA_MISSENSE = re.compile(r"^[A-Z]\d+[A-Z]\b")
_AA_NONSENSE = re.compile(r"^[A-Z]\d+\*")


def _mut_present(value: object) -> bool:
    """A sample 'carries' a mutation if any caller frequency parses > 0."""
    if value is None:
        return False
    s = str(value).strip()
    if not s or s.lower() in ("nan", "na", ""):
        return False
    nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", s)]
    return bool(nums) and max(nums) > 0.0


def _mut_frequency(value: object) -> float | None:
    nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", str(value))]
    return max(nums) if nums else None


def _details_to_effect(details: object) -> str | None:
    """Map an ALEdb 'Details' string to a canonical coding_effect bucket."""
    if not isinstance(details, str) or not details.strip():
        return None
    d = details.strip()
    if _AA_NONSENSE.match(d):
        return "nonsense"
    if _AA_MISSENSE.match(d):
        # synonymous if the aa letters are equal (e.g. "L21L") — rare; treat as missense otherwise
        m = re.match(r"^([A-Z])\d+([A-Z])", d)
        if m and m.group(1) == m.group(2):
            return "synonymous"
        return "missense"
    low = d.lower()
    if "intergenic" in low or "noncoding" in low or "pseudogene" in low:
        return "regulatory"
    if "coding" in low:
        return "coding"
    return _map_coding_effect(low)


def _split_sequence_change(value: object) -> tuple[object, object]:
    """Parse 'G→C' / 'G->C' into (ref, alt) for SNPs; else (NA, raw)."""
    import pandas as pd
    s = str(value)
    parts = re.split(r"→|->", s)
    if len(parts) == 2 and len(parts[0].strip()) == 1 and len(parts[1].strip()) == 1:
        return parts[0].strip(), parts[1].strip()
    return pd.NA, s


def load_resequencing_metadata(path: str | Path) -> "pd.DataFrame":
    """Load an ALEdb 'Resequencing Runs' metadata CSV, keyed on the sample label.

    Returns a frame indexed by the sample label (matching the mutation-matrix
    sample columns) with strain / media / temperature / organism columns.
    """
    import pandas as pd

    md = pd.read_csv(path)
    md.columns = [c.replace("\n", " ").strip() for c in md.columns]
    key = next((c for c in md.columns if "Isolate" in c or c.lower().startswith("ale")), md.columns[0])
    md = md.rename(columns={key: "lineage"})
    md["lineage"] = md["lineage"].astype(str).str.strip()

    # Robust to column mislabelling (seen in real exports where the "Strain"
    # column holds the taxid): identify organism by scanning every cell for a
    # known NCBI taxid, and base_strain by a strain-name-like token.
    def _row_organism(row) -> str:
        for v in row:
            t = str(v).strip()
            if t in _TAXID_ORGANISM:
                return _TAXID_ORGANISM[t]
        for v in row:
            t = str(v).strip()
            if t.isdigit() and 1000 <= int(t) <= 9999999:
                return f"taxid:{t}"
        return "unknown"

    def _row_strain(row) -> object:
        # prefer a non-numeric token from Strain / Additional Strain Details
        for col in ("Strain", "Additional Strain Details"):
            if col in md.columns:
                t = str(row.get(col, "")).strip()
                if t and t.lower() != "nan" and not t.isdigit():
                    return t
        return pd.NA

    md["organism"] = md.apply(_row_organism, axis=1)
    md["base_strain"] = md.apply(_row_strain, axis=1)
    return md.set_index("lineage")


def load_aledb_matrix(
    mutations_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    control_lineage: str | None = None,
    keep: str = "denovo",
    map_ecoli_bnumbers: bool = True,
) -> "pd.DataFrame":
    """Load a real ALEdb wide mutation matrix into the canonical long schema.

    Parameters
    ----------
    mutations_path : the wide ALEdb mutations CSV (annotation cols + sample cols).
    metadata_path  : optional 'Resequencing Runs' CSV to decode sample labels.
    control_lineage: the un-evolved reference sample (e.g. "A0 F0 I1 R1"). Mutations
        present in it are background/strain-construction and flagged
        ``is_key_mutation = False``; mutations absent in it but present in an
        evolved lineage are de-novo (``is_key_mutation = True``). Auto-detected as
        the "...I1..." sample if not given.
    keep : "denovo" (only de-novo rows), "all" (every observed sample×mutation),
        or "observed" (alias of all).

    Returns a frame with exactly ``CANONICAL_COLUMNS``.
    """
    import pandas as pd

    mut = pd.read_csv(mutations_path)
    ann = [c for c in mut.columns if c in ALEDB_MATRIX_ANNOTATION]
    samples = [c for c in mut.columns if c not in ann]
    if control_lineage is None:
        control_lineage = next((s for s in samples if re.search(r"\bI1\b", s)), samples[0])

    long = mut.melt(id_vars=ann, value_vars=samples, var_name="lineage", value_name="_freq")
    long = long[long["_freq"].apply(_mut_present)].copy()

    gene_col = "Gene (Scrollable)" if "Gene (Scrollable)" in mut.columns else "Gene"
    long["gene"] = long[gene_col].astype("string")
    long["genomic_position"] = pd.to_numeric(
        long["Position"].astype(str).str.replace(",", "", regex=False), errors="coerce").astype("Int64")
    long["mutation_type"] = long["Mutation Type"].astype("string")
    long["coding_effect"] = long["Details"].apply(_details_to_effect).astype("string")
    long["frequency"] = long["_freq"].apply(_mut_frequency).astype("Float64")
    refalt = long["Sequence Change"].apply(_split_sequence_change)
    long["ref_allele"] = [r for r, _ in refalt]
    long["alt_allele"] = [a for _, a in refalt]
    long["lineage"] = long["lineage"].astype(str).str.strip()

    # de-novo flag: present in this lineage, absent in the control lineage
    control_genes = set(
        zip(mut.loc[mut[control_lineage].apply(_mut_present), gene_col].astype(str),
            mut.loc[mut[control_lineage].apply(_mut_present), "Position"].astype(str)))
    long["is_key_mutation"] = [
        (str(g), str(p)) not in control_genes
        for g, p in zip(long[gene_col].astype(str), long["Position"].astype(str))
    ]
    long = long[long["lineage"] != control_lineage]   # the control itself isn't a sample of interest

    # metadata join
    if metadata_path is not None:
        md = load_resequencing_metadata(metadata_path)
        for canon, raw in [("base_strain", "base_strain"), ("media", "Base Media"),
                           ("organism", "organism")]:
            if raw in md.columns:
                long[canon] = long["lineage"].map(md[raw])
        if "Temperature (Celsius)" in md.columns:
            long["temperature_c"] = pd.to_numeric(
                long["lineage"].map(md["Temperature (Celsius)"]), errors="coerce").astype("Float64")
        long["selection_condition"] = (
            long.get("media", pd.Series("", index=long.index)).astype(str) + " "
            + long.get("temperature_c", pd.Series("", index=long.index)).astype(str) + "C"
        ).str.strip()

    long["experiment_id"] = str(Path(mutations_path).stem)
    long["sample_type"] = "clone"

    # For E. coli rows, map the ALEdb gene *name* to the iJO1366 *b-number*
    # (the model gene id / join key) so the table drops straight onto the GSM.
    if map_ecoli_bnumbers:
        from .gene_map import load_gene_aliases, map_name
        al = load_gene_aliases()
        is_ecoli = long["organism"].astype(str).str.contains("Escherichia", case=False, na=False) \
            if "organism" in long.columns else pd.Series(False, index=long.index)
        long["locus_tag"] = [
            map_name(g, al) if e else pd.NA
            for g, e in zip(long[gene_col].astype(str), is_ecoli)
        ]

    if keep == "denovo":
        long = long[long["is_key_mutation"]]

    # coerce to canonical
    for col in CANONICAL_COLUMNS:
        if col not in long.columns:
            long[col] = pd.NA
    out = long[list(CANONICAL_COLUMNS)].reset_index(drop=True)
    for col, dtype in CANONICAL_COLUMNS.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            out.attrs.setdefault("coerce_failed", []).append(col)
    out.attrs["control_lineage"] = control_lineage
    out.attrs["n_samples"] = len(samples)
    return out
