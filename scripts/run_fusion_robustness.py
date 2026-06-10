"""Multi-seed robustness for the flux/seq/fused comparison (milestone 16c).

The fusion result (flux+ESM2 macro recall@10 = 0.088 vs flux 0.037) comes from a single
CV seed on an 86-positive corpus, so it needs error bars. This re-runs the *identical*
cluster-aware (gene held-out) CV across many seeds — varying both the fold-split seed and
the LightGBM seed — and reports mean / std / 95% interval per feature set plus paired
differences (fused-flux, seq-flux, fused-seq) and the fraction of seeds where fusion wins.

Resumable: per-seed macros are cached to ``report/_robustness_raw.json``; each run extends
the cache and refreshes the aggregate (``report/fusion_robustness.json``) + ``figures/fig9``.

    python scripts/run_fusion_robustness.py --seq esm:data/esm_features.npz --seeds 30
    # add more later (cobra-free ESM path; uses the committed flux cache):
    python scripts/run_fusion_robustness.py --seq esm:data/esm_features.npz --seed-start 30 --seeds 20
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
    build_design_matrix, cluster_aware_split_indices, top_k_recall)

RAW = REPO / "report" / "_robustness_raw.json"
OUT = REPO / "report" / "fusion_robustness.json"
FIG = REPO / "figures" / "fig9_fusion_robustness.png"


def _seq_table(spec, gene_ids, model):
    mode, _, src = spec.partition(":")
    if mode == "biophysical":
        from ale_outcome_predictor.sequence_features import (
            build_biophysical_table, load_proteome_fasta, model_uniprot_map)
        prot = load_proteome_fasta(src)
        tbl = build_biophysical_table(gene_ids, prot, bnum_to_uniprot=model_uniprot_map(model))
        return tbl, f"biophysical ({len(prot)} seqs)"
    if mode == "esm":
        npz = np.load(src, allow_pickle=True)
        genes = [str(g) for g in npz["genes"]]; emb = npz["embeddings"]
        cols = [f"esm_{i}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb, index=genes, columns=cols), f"ESM ({emb.shape[1]}-dim)"
    raise SystemExit(f"--seq must be biophysical:<fasta> or esm:<npz>, got {spec!r}")


def _macro_at_seed(design, feats, seed, k=10):
    from lightgbm import LGBMClassifier
    clusters = feats["gene"].astype(str).tolist()
    conds = feats["selection_condition"].astype(str).tolist()
    y = feats["mutated"].fillna(False).astype("boolean").astype(int).to_numpy()
    X = design.to_numpy(dtype=float)
    splits = cluster_aware_split_indices(clusters, n_splits=5, seed=seed)
    per = defaultdict(list)
    for tr, te in splits:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        est = LGBMClassifier(n_estimators=150, num_leaves=7, min_child_samples=1,
                             learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1)
        est.fit(X[tr], y[tr])
        cl = list(est.classes_); pos = cl.index(1) if 1 in cl else len(cl) - 1
        sc = est.predict_proba(X[te])[:, pos]
        tc = [conds[i] for i in te]
        for cond in set(tc):
            idx = [j for j, c in enumerate(tc) if c == cond]
            yy = [int(y[te[j]]) for j in idx]
            if sum(yy):
                per[cond].append(top_k_recall(yy, [sc[j] for j in idx], k))
    pc = {c: float(np.mean(v)) for c, v in per.items()}
    return float(np.mean(list(pc.values()))) if pc else 0.0


def _ci(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "p2.5": float(np.percentile(a, 2.5)), "p50": float(np.percentile(a, 50)),
            "p97.5": float(np.percentile(a, 97.5)), "n": int(len(a))}


def _aggregate(raw, seq_label):
    seeds = sorted(raw, key=int)
    F = np.array([raw[s]["flux"] for s in seeds]); S = np.array([raw[s]["seq"] for s in seeds])
    Fu = np.array([raw[s]["fused"] for s in seeds])
    def paired(d):
        d = np.asarray(d, float)
        return {"mean": float(d.mean()), "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "frac_positive": float((d > 0).mean()), "n": int(len(d))}
    agg = {"seq_source": seq_label, "n_seeds": len(seeds),
           "macro": {"flux": _ci(F), "seq": _ci(S), "fused": _ci(Fu)},
           "paired": {"fused_minus_flux": paired(Fu - F), "seq_minus_flux": paired(S - F),
                      "fused_minus_seq": paired(Fu - S)}}
    OUT.write_text(json.dumps(agg, indent=2))
    _fig(F, S, Fu, agg, seq_label)
    return agg


def _fig(F, S, Fu, agg, seq_label):
    short = "sequence (ESM2)" if seq_label.startswith("ESM") else "sequence (biophysical)"
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    names = ["flux (GSM)", short, "fused"]; cols = ["#2f6f6f", "#c98b3a", "#7a4fa3"]
    means = [agg["macro"][k]["mean"] for k in ("flux", "seq", "fused")]
    stds = [agg["macro"][k]["std"] for k in ("flux", "seq", "fused")]
    a0.bar(range(3), means, yerr=stds, capsize=6, color=cols)
    for i, (m, s) in enumerate(zip(means, stds)):
        a0.text(i, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=9)
    a0.set_xticks(range(3)); a0.set_xticklabels(names, fontsize=9)
    a0.set_ylabel("macro recall@10  (mean ± SD)")
    a0.set_title(f"Feature-set comparison across {agg['n_seeds']} CV seeds")
    a0.spines[["top", "right"]].set_visible(False); a0.grid(axis="y", alpha=0.25)
    d = Fu - F
    a1.hist(d, bins=15, color="#7a4fa3", alpha=0.85)
    a1.axvline(0, color="k", lw=1)
    a1.axvline(d.mean(), color="#c0392b", lw=2, ls="--", label=f"mean {d.mean():+.3f}")
    win = agg["paired"]["fused_minus_flux"]["frac_positive"]
    a1.set_title(f"Paired (fused − flux): wins {win:.0%} of seeds")
    a1.set_xlabel("Δ macro recall@10 per seed"); a1.set_ylabel("seeds")
    a1.legend(frameon=False, fontsize=8.5)
    a1.spines[["top", "right"]].set_visible(False); a1.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix outputs (e.g. 650m) to keep runs separate")
    ap.add_argument("--pca", type=int, default=0, help="reduce seq embeddings to N PCA components (unsupervised)")
    args = ap.parse_args()

    global RAW, OUT, FIG
    if args.tag:
        RAW = REPO / "report" / f"_robustness_raw_{args.tag}.json"
        OUT = REPO / "report" / f"fusion_robustness_{args.tag}.json"
        FIG = REPO / "figures" / f"fig9_fusion_robustness_{args.tag}.png"
    need_model = args.seq.startswith("biophysical")
    model = P.load_gsm() if need_model else None
    feats = P.build_features(model, P.load_corpus(), use_cache=True)
    gene_ids = sorted(feats["gene"].astype(str).unique())
    seq_tbl, seq_label = _seq_table(args.seq, gene_ids, model)
    if args.pca:
        from sklearn.decomposition import PCA
        k = min(args.pca, seq_tbl.shape[1], seq_tbl.shape[0])
        filled = seq_tbl.apply(pd.to_numeric, errors="coerce")
        filled = filled.fillna(filled.mean())
        Z = PCA(n_components=k, random_state=0).fit_transform(filled.values)
        seq_tbl = pd.DataFrame(Z, index=seq_tbl.index, columns=[f"pc_{i}" for i in range(k)])
        seq_label += f" PCA{k}"

    flux_design = build_design_matrix(feats).reset_index(drop=True)
    seq_design = (seq_tbl.reindex(feats["gene"].astype(str).tolist())
                  .reset_index(drop=True).apply(pd.to_numeric, errors="coerce").astype(float))
    fused_design = pd.concat([flux_design, seq_design], axis=1)

    raw = json.loads(RAW.read_text()) if RAW.exists() else {}
    todo = [s for s in range(args.seed_start, args.seed_start + args.seeds) if str(s) not in raw]
    print(f"[robust] seq={seq_label} | cached {len(raw)} | computing {len(todo)} seeds")
    for s in todo:
        raw[str(s)] = {"flux": _macro_at_seed(flux_design, feats, s),
                       "seq": _macro_at_seed(seq_design, feats, s),
                       "fused": _macro_at_seed(fused_design, feats, s)}
        RAW.write_text(json.dumps(raw))
        print(f"  seed {s}: flux {raw[str(s)]['flux']:.3f}  seq {raw[str(s)]['seq']:.3f}  fused {raw[str(s)]['fused']:.3f}", flush=True)
    agg = _aggregate(raw, seq_label)
    m = agg["macro"]; p = agg["paired"]["fused_minus_flux"]
    print(json.dumps({"n_seeds": agg["n_seeds"],
                      "flux": round(m["flux"]["mean"], 3), "seq": round(m["seq"]["mean"], 3),
                      "fused": round(m["fused"]["mean"], 3),
                      "fused_minus_flux_mean": round(p["mean"], 3),
                      "fused_wins_frac": round(p["frac_positive"], 2)}, indent=2))


if __name__ == "__main__":
    main()
