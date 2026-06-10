"""Fuse sequence features with GSM-flux features and compare (milestone 16).

Tests the central hypothesis: do *sequence* features (which are flux-independent and
gene-identity-rich) help predict the convergent genes the flux model misses?

Compares three feature sets under the same cluster-aware CV (gene held out, per
condition, recall@10 vs the uniform-essential baseline):
  - flux   : the GSM-flux + structural features (current model)
  - seq    : sequence features only (biophysical, or ESM)
  - fused  : flux + seq

Usage:
  # biophysical (runs in-sandbox; needs a proteome FASTA)
  python scripts/run_fusion_experiment.py --seq biophysical:data/proteome_ecoli.fasta
  # ESM (drop in the .npz produced locally by scripts/extract_esm_features.py)
  python scripts/run_fusion_experiment.py --seq esm:data/esm_features.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ale_outcome_predictor import pipeline as P
from ale_outcome_predictor.baseline_model import (
    ModelConfig, build_design_matrix, cluster_aware_split_indices,
    top_k_recall, uniform_essential_baseline_scores,
)

CFG = ModelConfig(n_estimators=150, num_leaves=7, min_child_samples=1, learning_rate=0.05)


def _seq_table(spec: str, gene_ids: list[str], model) -> tuple[pd.DataFrame, str]:
    """Return (per-gene seq-feature table indexed by b-number, label)."""
    mode, _, src = spec.partition(":")
    if mode == "biophysical":
        from ale_outcome_predictor.sequence_features import (
            build_biophysical_table, load_proteome_fasta, model_uniprot_map)
        prot = load_proteome_fasta(src)
        tbl = build_biophysical_table(gene_ids, prot, bnum_to_uniprot=model_uniprot_map(model))
        cov = int(tbl.notna().any(axis=1).sum())
        return tbl, f"biophysical ({cov}/{len(gene_ids)} genes matched, {len(prot)} seqs)"
    if mode == "esm":
        npz = np.load(src, allow_pickle=True)
        genes = [str(g) for g in npz["genes"]]
        emb = npz["embeddings"]
        cols = [f"esm_{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb, index=genes, columns=cols), f"ESM ({emb.shape[1]}-dim)"
    raise SystemExit(f"--seq must be biophysical:<fasta> or esm:<npz>, got {spec!r}")


def _fit_predict(Xtr, ytr, Xte):
    from lightgbm import LGBMClassifier
    est = LGBMClassifier(n_estimators=CFG.n_estimators, num_leaves=CFG.num_leaves,
                         min_child_samples=CFG.min_child_samples, learning_rate=CFG.learning_rate,
                         random_state=0, n_jobs=1, verbose=-1)
    est.fit(Xtr, ytr)
    cls = list(est.classes_); pos = cls.index(1) if 1 in cls else len(cls) - 1
    return est.predict_proba(Xte)[:, pos]


def _cv_recall(design: pd.DataFrame, feats: pd.DataFrame, k: int = 10):
    """Per-condition macro recall@k under cluster-aware (gene held-out) CV."""
    clusters = feats["gene"].astype(str).tolist()
    conds = feats["selection_condition"].astype(str).tolist()
    y = feats["mutated"].fillna(False).astype("boolean").astype(int).to_numpy()
    splits = cluster_aware_split_indices(clusters, n_splits=5, seed=0)
    per = defaultdict(list)
    X = design.to_numpy(dtype=float)
    for tr, te in splits:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        sc = _fit_predict(X[tr], y[tr], X[te])
        tc = [conds[i] for i in te]
        for cond in set(tc):
            idx = [j for j, c in enumerate(tc) if c == cond]
            yy = [int(y[te[j]]) for j in idx]
            if sum(yy) == 0:
                continue
            per[cond].append(top_k_recall(yy, [sc[j] for j in idx], k))
    return {c: float(np.mean(v)) for c, v in per.items()}




def _fig(summary, path, seq_label="sequence"):
    short = {"glucose M9 minimal 37C": "glucose", "glucose M9 minimal 42C thermal": "thermal",
             "glucose M9 benzoate stress 37C": "benzoate", "glycerol M9 minimal 37C": "glycerol",
             "acetate defined medium 37C": "acetate"}
    conds = sorted(summary["flux"]["per_condition"], key=lambda c: short.get(c, c))
    labels = [short.get(c, c) for c in conds]
    x = np.arange(len(conds)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for off, fs, col, lab in [(-w, "flux", "#2f6f6f", "flux (GSM)"),
                              (0.0, "seq", "#c98b3a", seq_label),
                              (w, "fused", "#7a4fa3", "fused")]:
        vals = [summary[fs]["per_condition"].get(c, 0.0) for c in conds]
        ax.bar(x + off, vals, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("cluster-aware CV recall@10")
    ax.set_title("%s vs GSM-flux features, and their fusion\n(macro recall@10: flux %.2f | seq %.2f | fused %.2f)" %
                 (seq_label, summary["flux"]["macro"], summary["seq"]["macro"], summary["fused"]["macro"]))
    peak = max([v for f in summary for v in summary[f]["per_condition"].values()] + [0.25])
    ax.legend(frameon=False, fontsize=8.5); ax.set_ylim(0, peak + 0.04)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="biophysical:<fasta> | esm:<npz>")
    args = ap.parse_args()

    # Only the biophysical path needs the GSM (for model_uniprot_map); the ESM path
    # reads a pre-built .npz and uses the committed flux-feature cache, so it runs
    # with no cobra / solver installed. Load the model lazily to keep that true.
    need_model = args.seq.startswith("biophysical")
    model = P.load_gsm() if need_model else None
    feats = P.build_features(model, P.load_corpus(), use_cache=True)
    if feats is None or len(feats) == 0:
        raise SystemExit("no cached features at features/feature_matrix.parquet — "
                         "run the flux pipeline once (needs cobra) to build the cache.")
    gene_ids = sorted(feats["gene"].astype(str).unique())
    seq_tbl, seq_label = _seq_table(args.seq, gene_ids, model)

    seq_cols = list(seq_tbl.columns)
    flux_design = build_design_matrix(feats).reset_index(drop=True)
    # align seq features to feats row order by gene (guarantees row-for-row match)
    seq_design = (seq_tbl.reindex(feats["gene"].astype(str).tolist())
                  .reset_index(drop=True).apply(pd.to_numeric, errors="coerce").astype(float))
    assert len(seq_design) == len(flux_design) == len(feats)
    fused_design = pd.concat([flux_design, seq_design], axis=1)

    cov = float(seq_tbl.notna().any(axis=1).mean())
    print(f"[fusion] seq source: {seq_label} | genes with sequence: {cov:.0%}")
    results = {"flux": _cv_recall(flux_design, feats),
               "seq": _cv_recall(seq_design, feats),
               "fused": _cv_recall(fused_design, feats)}
    summary = {fs: {"per_condition": r, "macro": float(np.mean(list(r.values()))) if r else 0.0}
               for fs, r in results.items()}
    out = {"seq_source": seq_label, "sequence_coverage": cov, "recall_at_10": summary}
    (REPO / "report" / "fusion_metrics.json").write_text(json.dumps(out, indent=2))
    _fig(summary, REPO / "figures" / "fig8_sequence_fusion.png",
         seq_label="sequence (ESM2)" if seq_label.startswith("ESM") else "sequence (biophysical)")
    print(json.dumps({fs: round(summary[fs]["macro"], 3) for fs in summary}, indent=2))
    print("wrote report/fusion_metrics.json")


if __name__ == "__main__":
    main()
