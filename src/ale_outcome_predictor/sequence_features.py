"""Sequence-derived features to complement GSM-flux features  (milestone 16).

The flux features are blind to ~60-80 % of real convergent ALE targets (regulatory,
transport, structural, prophage genes — see README "GSM-coverage gap"). Sequence
features are *gene-identity-rich and flux-independent*, so they can reach those genes.

Two paths, same output schema (one row per gene, numeric columns the GBT ingests):

* **biophysical** (`mode="biophysical"`) — torch-free, runs anywhere via Biopython
  ProtParam: length, MW, pI, GRAVY hydropathy, aromaticity, instability index,
  secondary-structure fractions, and charged/hydrophobic AA fractions.
* **esm** (`mode="esm"`) — mean-pooled ESM2 embeddings via HuggingFace
  ``transformers`` (480-dim for esm2_t12_35M). Needs torch + model download, so it is
  run **locally / on GPU** (see ``scripts/extract_esm_features.py``) and the cached
  ``.npz`` is dropped in — the build sandbox cannot reach pytorch.org / huggingface.co.

Both merge onto the flux feature matrix on the model gene id (b-number); sequences
come from a proteome FASTA keyed by gene name (UniProt ``GN=``) or locus tag, resolved
to b-numbers via ``gene_map``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BIOPHYS_COLUMNS",
    "load_proteome_fasta",
    "biophysical_features",
    "build_biophysical_table",
    "model_uniprot_map",
    "esm_embed",
    "merge_sequence_features",
]

# Charged / hydrophobic residue groups for composition features.
_CHARGED = set("DEKR")
_HYDROPHOBIC = set("AILMFWVC")
_AA = "ACDEFGHIKLMNPQRSTVWY"

BIOPHYS_COLUMNS: tuple[str, ...] = (
    "seq_length", "seq_mw", "seq_pi", "seq_gravy", "seq_aromaticity",
    "seq_instability", "seq_helix_frac", "seq_turn_frac", "seq_sheet_frac",
    "seq_charged_frac", "seq_hydrophobic_frac",
)

_GN_RE = re.compile(r"\bGN=([^\s]+)")
_LOCUS_RE = re.compile(r"(?:locus_tag=|\[locus_tag=)?\b(b\d{4})\b")
_ACC_RE = re.compile(r"^(?:sp|tr)\|([A-Z0-9]+)\|")


def load_proteome_fasta(path: str | Path) -> dict[str, str]:
    """Parse a proteome FASTA → ``{gene_key: sequence}``.

    Keys are, in priority order, a b-number locus tag found in the header, else the
    UniProt ``GN=`` gene name (lower-cased). Handles UniProt and NCBI/RefSeq headers.
    Resolution of gene-name keys → b-numbers happens in ``build_biophysical_table``
    via ``gene_map``.
    """
    out: dict[str, str] = {}
    header, seq = None, []

    def _flush(h, s):
        if not h or not s:
            return
        sequence = "".join(s).replace("*", "").strip().upper()
        if not sequence:
            return
        # index the sequence under every identifier we can find in the header:
        # UniProt accession (sp|ACC|), a b-number locus tag, and the GN= gene name.
        keys = []
        acc = _ACC_RE.search(h)
        if acc:
            keys.append(acc.group(1).lower())
        b = _LOCUS_RE.search(h)
        if b:
            keys.append(b.group(1))
        gn = _GN_RE.search(h)
        if gn:
            keys.append(gn.group(1).lower())
        if not keys:
            keys = [h.split()[0]]
        for k in keys:
            out.setdefault(k, sequence)

    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                _flush(header, seq)
                header, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
        _flush(header, seq)
    return out


def biophysical_features(seq: str) -> dict[str, float]:
    """Per-protein biophysical features from an amino-acid sequence (Biopython)."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    clean = "".join(c for c in seq.upper() if c in _AA)
    if len(clean) < 5:
        return {c: float("nan") for c in BIOPHYS_COLUMNS}
    pa = ProteinAnalysis(clean)
    helix, turn, sheet = pa.secondary_structure_fraction()
    n = len(clean)
    return {
        "seq_length": float(n),
        "seq_mw": float(pa.molecular_weight()),
        "seq_pi": float(pa.isoelectric_point()),
        "seq_gravy": float(pa.gravy()),
        "seq_aromaticity": float(pa.aromaticity()),
        "seq_instability": float(pa.instability_index()),
        "seq_helix_frac": float(helix),
        "seq_turn_frac": float(turn),
        "seq_sheet_frac": float(sheet),
        "seq_charged_frac": sum(clean.count(a) for a in _CHARGED) / n,
        "seq_hydrophobic_frac": sum(clean.count(a) for a in _HYDROPHOBIC) / n,
    }


def model_uniprot_map(model) -> dict[str, str]:
    """{gene_id (b-number): UniProt accession} from a cobrapy model's gene annotations."""
    out = {}
    for g in model.genes:
        acc = (g.annotation or {}).get("uniprot")
        if isinstance(acc, list):
            acc = acc[0] if acc else None
        if acc:
            out[g.id] = acc
    return out


def build_biophysical_table(
    gene_ids: list[str], proteome: dict[str, str], aliases: dict[str, str] | None = None,
    bnum_to_uniprot: dict[str, str] | None = None,
) -> "pd.DataFrame":
    """Biophysical feature table indexed by model gene id (b-number).

    For each ``gene_id`` (b-number), find its sequence in ``proteome`` — directly by
    b-number key, else by any proteome gene-name key whose ``gene_map`` b-number matches.
    Genes with no sequence get all-NA rows (the GBT ingests NaN natively).
    """
    import pandas as pd

    from .gene_map import load_gene_aliases, map_name

    al = aliases if aliases is not None else load_gene_aliases()
    bnum_to_uniprot = bnum_to_uniprot or {}
    # gene-name keys -> b-number (via gene_map), so name-keyed entries resolve too
    name_resolved: dict[str, str] = {}
    for key, seq in proteome.items():
        if not re.fullmatch(r"b\d{4}", key) and not re.fullmatch(r"[a-z0-9]{6,10}", key):
            b = map_name(key, al)
            if b:
                name_resolved.setdefault(b, seq)

    rows = []
    for gid in gene_ids:
        gid = str(gid)
        acc = str(bnum_to_uniprot.get(gid, "")).lower()
        seq = (proteome.get(gid)                       # b-number header
               or (proteome.get(acc) if acc else None)  # UniProt accession (model map)
               or name_resolved.get(gid))               # gene-name fallback
        feats = biophysical_features(seq) if seq else {c: pd.NA for c in BIOPHYS_COLUMNS}
        feats["gene"] = gid
        rows.append(feats)
    return pd.DataFrame(rows).set_index("gene")


def esm_embed(seqs: list[str], model_name: str = "facebook/esm2_t12_35M_UR50D",
              batch_size: int = 16) -> "object":
    """Mean-pooled ESM2 embeddings via HuggingFace transformers (local/GPU only).

    Mirrors the AMP-classifier extractor. Imports torch/transformers lazily; the build
    sandbox cannot reach the weights, so this is invoked by
    ``scripts/extract_esm_features.py`` on a machine with the deps + network.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [s[:1022] for s in seqs[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            hidden = model(**enc).last_hidden_state            # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


def merge_sequence_features(features: "pd.DataFrame", seq_table: "pd.DataFrame") -> "pd.DataFrame":
    """Left-join a per-gene sequence-feature table onto the (gene x condition) matrix."""
    import pandas as pd

    return features.merge(seq_table, how="left", left_on="gene", right_index=True)
