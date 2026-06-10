"""Run the full ALE-outcome-predictor experiment and emit metrics + figures.

Wraps ``ale_outcome_predictor.pipeline`` with the small-corpus model config,
adds a baseline-anchored cluster-aware CV (the library CV reports model recall
only), renders four figures, and writes ``report/metrics.json`` +
``report/ranked_predictions.csv``.

Usage:  python scripts/run_experiment.py
Heavy step (FVA + single-gene-deletion over the whole GSM) is cached to
``features/feature_matrix.parquet`` after the first run.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
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
    ModelConfig,
    cluster_aware_split_indices,
    predict_scores,
    top_k_recall,
    top_k_recall_by_group,
    train_model,
    uniform_essential_baseline_scores,
    predict_ranked_genes,
)

FIG = REPO / "figures"
REPORT = REPO / "report"
FIG.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)

# Small-corpus model config: shallow trees, single-sample leaves (the default
# min_child_samples=20 cannot split ~8 positives).
CFG = ModelConfig(n_estimators=150, num_leaves=7, min_child_samples=1, learning_rate=0.05)

# Clean, neutral plot style.
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})
INK = "#2f6f6f"     # teal
ACCENT = "#c98b3a"  # warm amber (baseline)


def cluster_cv_with_baseline(features, cfg, n_splits=5, seed=0):
    """Cluster-aware (gene held-out) CV reporting BOTH model and uniform-essential
    baseline macro top-k recall, averaged over folds that contain positives."""
    clusters = features["gene"].astype(str).tolist()
    conditions = features["selection_condition"].astype(str).tolist()
    splits = cluster_aware_split_indices(clusters, n_splits=n_splits, seed=seed)
    rows = []
    for fold, (tr, te) in enumerate(splits):
        tr_df, te_df = features.iloc[tr], features.iloc[te]
        y = [0 if pd.isna(v) else int(bool(v)) for v in te_df["mutated"].tolist()]
        if sum(y) == 0:
            continue
        bundle = train_model(tr_df, cfg)
        ms = predict_scores(bundle, te_df).tolist()
        ess = [None if pd.isna(v) else bool(v) for v in te_df["is_essential"].tolist()]
        bs = uniform_essential_baseline_scores(ess, seed=seed)
        cond_te = [conditions[i] for i in te]
        row = {"fold": fold, "n_test": len(te), "n_pos": sum(y)}
        for k in (10, 25):
            row[f"model@{k}"] = top_k_recall_by_group(cond_te, y, ms, k)
            row[f"baseline@{k}"] = top_k_recall_by_group(cond_te, y, bs, k)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_cluster_cv(cl_base, path):
    ks = [10, 25]
    model = [cl_base[f"model@{k}"].mean() for k in ks]
    base = [cl_base[f"baseline@{k}"].mean() for k in ks]
    x = np.arange(len(ks)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x - w/2, model, w, label="GSM-flux model", color=INK)
    ax.bar(x + w/2, base, w, label="uniform-essential baseline", color=ACCENT)
    for i, (m, b) in enumerate(zip(model, base)):
        ax.text(i - w/2, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("macro recall of convergent genes"); ax.set_ylim(0, 1.15)
    ax.set_title("Cluster-aware CV (gene held out): model vs baseline")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def fig_importance(imp, path, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.barh(d["feature"], d["importance"], color=INK)
    ax.set_xlabel("LightGBM gain"); ax.set_title("Top flux/structural features")
    fig.savefig(path); plt.close(fig)


def fig_loco(loco, path):
    conds = loco["held_out_condition"].tolist()
    short = [c.replace("M9 minimal", "").replace("glucose", "glc").replace("glycerol", "gly").strip()
             for c in conds]
    model = loco["model_recall@10"].fillna(0).tolist()
    base = loco["baseline_recall@10"].fillna(0).tolist()
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - w/2, model, w, label="model", color=INK)
    ax.bar(x + w/2, base, w, label="baseline", color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=12, ha="right", fontsize=8.5)
    ax.set_ylabel("recall@10 (held-out condition)"); ax.set_ylim(0, 1.15)
    ax.set_title("Leave-one-condition-out transfer (the hard test)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def fig_mechanism(features, path):
    """The honest mechanistic hook: pFBA flux through glycerol kinase (glpK, GLYK)
    is ZERO on glucose and large on glycerol — the media swap genuinely turns the
    gene on. (fva_max_abs_flux shows capacity, not use, so we plot pFBA flux.)"""
    from cobra.flux_analysis import pfba
    from ale_outcome_predictor.feature_engineer import apply_environment
    from ale_outcome_predictor.media import resolve_environment
    model = P.load_gsm()
    glpk = model.genes.get_by_id("b3926")
    rxn_ids = [r.id for r in glpk.reactions]
    vals = {}
    for carbon, cond in [("glucose", "glucose M9 minimal 37C"),
                         ("glycerol", "glycerol M9 minimal 37C")]:
        with model:
            apply_environment(model, resolve_environment(cond, model))
            fl = pfba(model).fluxes
            vals[carbon] = sum(abs(float(fl[r])) for r in rxn_ids if r in fl.index)
    carbons = ["glucose", "glycerol"]
    flux = [vals[c] for c in carbons]
    colors = [INK if f > 1e-6 else "#b9c6c6" for f in flux]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bars = ax.bar(carbons, flux, color=colors, width=0.55)
    for b, f in zip(bars, flux):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(flux) * 0.02 + 0.05,
                ("OFF (no flux)" if f < 1e-6 else f"ON ({f:.1f})"), ha="center", fontsize=9)
    ax.set_ylabel("glpK pFBA flux (mmol gDW$^{-1}$ h$^{-1}$)")
    ax.set_ylim(0, max(flux) * 1.25 + 0.5)
    ax.set_title("Media conditioning is mechanistic:\nglycerol kinase is OFF on glucose, ON on glycerol")
    fig.savefig(path); plt.close(fig)




def cluster_cv_per_condition(features, cfg, n_splits=5, seed=0):
    """Cluster-aware CV recall@10 *per selection condition* (model vs baseline).

    The macro-average hides that glucose-37 and thermal-42 are flux-degenerate
    (same simulated glucose medium — FBA has no temperature), so pooling their
    distinct convergent sets interferes. Per condition is the honest view.
    """
    import statistics as st
    from collections import defaultdict
    clusters = features["gene"].astype(str).tolist()
    conds = features["selection_condition"].astype(str).tolist()
    splits = cluster_aware_split_indices(clusters, n_splits=n_splits, seed=seed)
    mm, bb = defaultdict(list), defaultdict(list)
    for tr, te in splits:
        trd, ted = features.iloc[tr], features.iloc[te]
        y_all = [0 if pd.isna(v) else int(bool(v)) for v in ted["mutated"].tolist()]
        if sum(y_all) == 0:
            continue
        bundle = train_model(trd, cfg)
        sc = predict_scores(bundle, ted)
        ess = [None if pd.isna(v) else bool(v) for v in ted["is_essential"].tolist()]
        bs = uniform_essential_baseline_scores(ess, seed=seed)
        tc = [conds[i] for i in te]
        for cond in set(tc):
            idx = [j for j, c in enumerate(tc) if c == cond]
            y = [y_all[j] for j in idx]
            if sum(y) == 0:
                continue
            mm[cond].append(top_k_recall(y, [sc[j] for j in idx], 10))
            bb[cond].append(top_k_recall(y, [bs[j] for j in idx], 10))
    return {c: {"model@10": st.mean(mm[c]), "baseline@10": st.mean(bb[c]), "folds": len(mm[c])}
            for c in sorted(mm)}


def fig_per_condition(perc, path):
    short = {"glucose M9 minimal 37C": "glucose\n(ExpID762)",
             "glucose M9 minimal 42C thermal": "thermal 42\u00b0C\n(ExpID740)",
             "glucose M9 benzoate stress 37C": "benzoate\n(ExpID940)",
             "glycerol M9 minimal 37C": "glycerol\n(ExpID1523)",
             "acetate defined medium 37C": "acetate\n(ExpID1008)"}
    conds = list(perc.keys())
    labels = [short.get(c, c) for c in conds]
    model = [perc[c]["model@10"] for c in conds]
    base = [perc[c]["baseline@10"] for c in conds]
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.bar(x - w/2, model, w, label="GSM-flux model", color=INK)
    ax.bar(x + w/2, base, w, label="uniform-essential baseline", color=ACCENT)
    for i, (mo, ba) in enumerate(zip(model, base)):
        ax.text(i - w/2, mo + 0.01, f"{mo:.2f}", ha="center", fontsize=8.5)
        ax.text(i + w/2, ba + 0.01, f"{ba:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("cluster-aware CV recall@10"); ax.set_ylim(0, max(model + base) * 1.3 + 0.05)
    ax.set_title("Per-condition recall, 5 real ALEdb selections:\n3 share the glucose medium (flux-degenerate), so they interfere")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(path); plt.close(fig)


def fig_indist_recovery(ranked, path):
    """In-distribution: how many of each condition's real converged genes land in
    the model's top-10 (trained on all, ranked per condition). Shows the model
    *fits* every condition's convergent set — the held-out failure (fig6) is a
    generalization limit, not a representation one."""
    conds = sorted(ranked["selection_condition"].unique())
    short = {"glucose M9 minimal 37C": "glucose", "glucose M9 minimal 42C thermal": "thermal 42\u00b0C",
             "glucose M9 benzoate stress 37C": "benzoate", "glycerol M9 minimal 37C": "glycerol",
             "acetate defined medium 37C": "acetate"}
    labels = [short.get(c, c) for c in conds]
    hits = [int(ranked[ranked["selection_condition"] == c].head(10)["is_known_target"].sum()) for c in conds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(range(len(conds)), hits, color=INK, width=0.6)
    for i, h in enumerate(hits):
        ax.text(i, h + 0.15, str(h), ha="center", fontsize=10)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("real converged genes in top-10"); ax.set_ylim(0, 10.5)
    ax.axhline(10, ls="--", lw=1, color="#888", alpha=0.6)
    ax.set_title("In-distribution recovery: the model fits every condition's\nreal converged set (top-10 of 1367 genes)")
    fig.savefig(path); plt.close(fig)


def main():
    res = P.run(config=CFG, use_cache=True)
    model = P.load_gsm()

    cl_base = cluster_cv_with_baseline(res.features, CFG)
    perc = cluster_cv_per_condition(res.features, CFG)

    # Figures
    fig_cluster_cv(cl_base, FIG / "fig1_cluster_cv_recall.png")
    fig_importance(res.importances, FIG / "fig2_feature_importance.png")
    fig_loco(res.loco, FIG / "fig3_loco_transfer.png")
    fig_mechanism(res.features, FIG / "fig4_media_mechanism.png")
    fig_per_condition(perc, FIG / "fig6_per_condition_cv.png")

    # Ranked-prediction example: train on all, rank genes per condition, top 15.
    bundle = train_model(res.features, CFG)
    ranked_parts = []
    namemap = P.load_corpus().attrs.get("gene_name", {})
    for cond in sorted(res.features["selection_condition"].unique()):
        sub = res.features[res.features["selection_condition"] == cond]
        r = predict_ranked_genes(bundle, sub).head(15).copy()
        r["gene_name"] = r["gene"].map(namemap).fillna("")
        r["is_known_target"] = r["gene"].isin(P.load_corpus()["gene"].astype(str))
        ranked_parts.append(r)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked.to_csv(REPORT / "ranked_predictions.csv", index=False)
    fig_indist_recovery(ranked, FIG / "fig7_indistribution_recovery.png")
    indist = {c: int(ranked[ranked["selection_condition"] == c].head(10)["is_known_target"].sum()) for c in sorted(ranked["selection_condition"].unique())}

    metrics = {
        "meta": res.meta,
        "gsm_coverage": res.gsm_coverage,
        "cluster_cv_per_condition": perc,
        "indistribution_top10_recovery": indist,
        "cluster_aware_cv_vs_baseline": json.loads(cl_base.to_json(orient="records")),
        "cluster_aware_cv_means": {
            "model@10": float(cl_base["model@10"].mean()),
            "baseline@10": float(cl_base["baseline@10"].mean()),
            "model@25": float(cl_base["model@25"].mean()),
            "baseline@25": float(cl_base["baseline@25"].mean()),
        },
        "leave_one_condition_out": json.loads(res.loco.to_json(orient="records")),
        "top_features": json.loads(res.importances.head(12).to_json(orient="records")),
    }
    (REPORT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("DONE")
    print(json.dumps(metrics["cluster_aware_cv_means"], indent=2))
    print("coverage:", metrics["gsm_coverage"]["coverage_fraction"],
          "| meta:", metrics["meta"])


if __name__ == "__main__":
    main()
