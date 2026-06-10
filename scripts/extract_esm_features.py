"""Extract ESM2 embeddings for the model's genes — RUN LOCALLY (needs torch + HF).

No cobra / solver required: the model gene list and the gene->UniProt-accession map
are read from the committed ``data/model_gene_uniprot.csv`` (exported once from the GSM).
So a plain ``pip install torch transformers biopython pandas numpy`` env is enough.

    pip install torch transformers biopython pandas numpy
    python scripts/extract_esm_features.py --proteome data/proteome_ecoli.fasta --out data/esm_features.npz
    # optional: --model facebook/esm2_t33_650M_UR50D   (strong; GPU recommended)
    # sanity-check sequence resolution without downloading weights:
    python scripts/extract_esm_features.py --proteome data/proteome_ecoli.fasta --dry-run

Then back in the build env:
    python scripts/run_fusion_experiment.py --seq esm:data/esm_features.npz

Output npz: arrays ``genes`` (b-numbers) and ``embeddings`` (N x H mean-pooled ESM2).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GENE_MAP_CSV = REPO / "data" / "model_gene_uniprot.csv"


def _load_gene_uniprot(path: Path) -> dict:
    """{gene_id (b-number): UniProt accession} from the exported CSV.

    Falls back to loading the GSM (needs cobra) only if the CSV is missing, so the
    common case stays solver-free.
    """
    if path.exists():
        out = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["gene"]] = (row.get("uniprot") or "").strip()
        return out
    from ale_outcome_predictor import pipeline as P
    from ale_outcome_predictor.sequence_features import model_uniprot_map
    model = P.load_gsm()
    umap = model_uniprot_map(model)
    return {g.id: umap.get(g.id, "") for g in model.genes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteome", required=True, help="proteome FASTA (UniProt/NCBI)")
    ap.add_argument("--out", default="data/esm_features.npz")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve sequences and report coverage; skip the (heavy) embedding")
    args = ap.parse_args()

    from ale_outcome_predictor.gene_map import load_gene_aliases, map_name
    from ale_outcome_predictor.sequence_features import load_proteome_fasta

    gene_uniprot = _load_gene_uniprot(GENE_MAP_CSV)
    gene_ids = list(gene_uniprot)
    prot = load_proteome_fasta(args.proteome)

    al = load_gene_aliases()
    name_seq = {}
    for k, v in prot.items():
        if not re.fullmatch(r"b\d{4}", k) and not re.fullmatch(r"[a-z0-9]{6,10}", k):
            b = map_name(k, al)
            if b:
                name_seq.setdefault(b, v)

    have = []
    for g in gene_ids:
        acc = str(gene_uniprot.get(g, "")).lower()
        seq = prot.get(g) or (prot.get(acc) if acc else None) or name_seq.get(g)
        if seq:
            have.append((g, seq))

    print(f"[esm] {len(have)}/{len(gene_ids)} model genes resolved to a sequence "
          f"({len(prot)} proteome entries)")
    if args.dry_run:
        resolved = {h for h, _ in have}
        missing = [g for g in gene_ids if g not in resolved]
        if missing:
            print(f"[esm] unresolved ({len(missing)}): {', '.join(missing[:10])}"
                  + (" ..." if len(missing) > 10 else ""))
        print("[esm] dry run — no embedding performed.")
        return

    from ale_outcome_predictor.sequence_features import esm_embed
    print(f"[esm] embedding with {args.model} ...")
    genes = [g for g, _ in have]
    emb = esm_embed([s for _, s in have], model_name=args.model)
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, genes=np.array(genes), embeddings=emb)
    print(f"[esm] wrote {args.out}: {emb.shape[0]} genes x {emb.shape[1]} dims")


if __name__ == "__main__":
    main()
