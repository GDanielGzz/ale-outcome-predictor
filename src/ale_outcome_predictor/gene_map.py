"""Map ALEdb gene *names* to iJO1366 *b-numbers* (the model's gene ids).

ALEdb reports gene symbols (`pyrE`, `fabA`) and intergenic loci (`pyrE/rph`),
while iJO1366 keys genes on b-numbers (`b3642`). This module bridges them so a
loaded ALEdb table joins straight onto the model.

Source of the alias table (`corpus/ecoli_gene_aliases.csv`): the UniProt
`organism_id:83333 reviewed` dump (`gene_primary` / `gene_oln` / `gene_synonym`),
where `gene_oln` is the b-number. Seeded with the convergent genes of the target
ALEdb conditions (glucose-growth, 42 °C thermal, glycerol); extend the CSV from
the same UniProt query to cover more (see `data/HOWTO_PULL.md`). Primary symbols
win over synonyms on conflict (e.g. `fabG` is b1093, not accC's synonym).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["DEFAULT_ALIAS_CSV", "load_gene_aliases", "map_name", "map_names_to_bnumbers"]

DEFAULT_ALIAS_CSV = Path(__file__).resolve().parents[2] / "corpus" / "ecoli_gene_aliases.csv"

# split an intergenic / multi-gene locus string into candidate gene tokens
_SPLIT = re.compile(r"[\s/,;]+|←|→|<-|->|\.\.\.|–|—")


def load_gene_aliases(csv: str | Path | None = None) -> dict[str, str]:
    """Return a lower-cased ``{name_or_synonym: b_number}`` map.

    Primary names are inserted last so they overwrite any synonym collision.
    """
    import pandas as pd

    df = pd.read_csv(csv or DEFAULT_ALIAS_CSV).fillna("")
    syn: dict[str, str] = {}
    prim: dict[str, str] = {}
    for _, r in df.iterrows():
        b = str(r["b_number"]).strip()
        if not b:
            continue
        for s in str(r["synonyms"]).split("|"):
            s = s.strip().lower()
            if s:
                syn[s] = b
        prim[str(r["gene_name"]).strip().lower()] = b
    return {**syn, **prim}     # primary wins


def map_name(name: object, aliases: dict[str, str]) -> str | None:
    """Map one ALEdb gene/locus string to a b-number, or ``None``.

    Handles exact symbol, synonym, and intergenic/multi-gene strings (returns the
    first flanking gene that maps — intergenic mutations are regulatory and sit
    between two genes; the first mappable flank is the conventional anchor).
    """
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip().lower()
    if key in aliases:
        return aliases[key]
    for tok in _SPLIT.split(key):
        tok = tok.strip()
        if tok and tok in aliases:
            return aliases[tok]
    return None


def map_names_to_bnumbers(
    table: "pd.DataFrame",
    aliases: dict[str, str] | None = None,
    *,
    name_col: str = "gene",
    out_col: str = "locus_tag",
) -> "pd.DataFrame":
    """Fill ``out_col`` with b-numbers mapped from ``name_col``.

    Unmappable names are left NA and collected on ``df.attrs['unmapped_genes']``
    so the alias CSV can be extended in one place.
    """
    import pandas as pd

    al = aliases if aliases is not None else load_gene_aliases()
    out = table.copy()
    mapped = [map_name(n, al) for n in out[name_col]]
    out[out_col] = pd.Series(mapped, index=out.index, dtype="string")
    unmapped = sorted({
        str(n) for n, m in zip(out[name_col], mapped) if m is None and isinstance(n, str) and n.strip()
    })
    out.attrs["unmapped_genes"] = unmapped
    out.attrs["n_mapped"] = int(sum(m is not None for m in mapped))
    return out
