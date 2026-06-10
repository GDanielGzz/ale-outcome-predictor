# ALE outcome predictor

**Predict which genes a genome is most likely to mutate under a defined adaptive-laboratory-evolution (ALE) selection — conditioned on genome-scale metabolic-model (GSM) flux features.**

> Status: **working end-to-end** on **five real ALEdb *E. coli* selections** — glucose,
> 42 °C thermal, glycerol, benzoate, acetate (86 convergent positives, 85 real ALEdb) — on a
> committed, literature-cited convergence corpus. `python -m ale_outcome_predictor.pipeline`
> trains a gradient-boosted-tree model and runs the held-out evaluation. The corpus is
> deliberately small (a proof-of-concept; see [Results](#results) and the honesty notes) and
> the pipeline drops straight onto the full ALEdb the moment it is pulled locally.

---

## Overview

Adaptive laboratory evolution selects a population in a defined environment and lets mutations accumulate. Aggregated ALE corpora (notably **ALEdb 1.0**, 11,000+ mutations across 11 publications) show that the genes which acquire causal mutations are far from random — they cluster on metabolic bottlenecks created by the selection. Prior meta-analyses stop at *statistical* and *structural* enrichment of those mutations.

This project asks a sharper, prospective question: **given a chassis GSM and a selection environment, can we rank the genes by their probability of acquiring a causal mutation?**

The model takes three inputs:

1. the chassis **genome-scale metabolic model** (*E. coli* `iJO1366`, read directly by cobrapy);
2. the **selection environment**, parsed into a growth medium + a temperature/stress vector;
3. **flux-derived features** computed on that GSM *under the selection medium* — FVA ranges, single-gene-deletion essentiality, parsimonious-FBA flux usage, blocked-reaction counts, and GPR/topology structure.

It outputs a **ranked gene list with probabilities** — the genes most likely to be the next ALE target under that selection.

## Results

**Five real ALEdb E. coli selections** train the model — glucose (ExpID762, LaCroix),
42 °C thermal (ExpID740, Tenaillon's 113 lineages), glycerol (ExpID1523), benzoate
aromatic-acid stress (ExpID940), acetate (ExpID1008) — **86 convergent positives, 85 of
them real ALEdb converged-mutation calls**. `python scripts/run_experiment.py`; figures in `figures/`, numbers in
`report/metrics.json`. The headline is an honest, two-part characterisation of *where
GSM-flux ALE-prediction works and where it breaks*.

**1. In-distribution, the model fits every condition's real converged set (figures/fig7).**
Trained on all 86 positives and ranked per condition, the real ALEdb converged genes land
in the top-10 of all 1,367 genes: glucose **9/10**, thermal **9/10**, benzoate **8/10**,
glycerol **7/10**, acetate **4/4**. The flux + condition features *represent* real
convergence across five selections — top glucose hits are oppA/cueO/fadE/pykF, thermal the
rbs/glm/glg/murA/cls set, glycerol glpK.

**2. Held-out generalisation is bounded by flux-degeneracy — demonstrated decisively
(figures/fig6).** Held-out-*gene* cluster-aware CV tells the real limitation. Three of the
five conditions (glucose, thermal, benzoate) resolve to the **same simulated glucose
medium** — FBA has no temperature or stress, so their flux features are identical and their
distinct convergent sets become mutually unpredictable. Adding benzoate as a *third*
glucose-medium condition **collapsed glucose held-out recall from 0.18 to 0.04** — a clean
empirical proof that flux features cannot resolve *which* selection drives convergence when
media are shared. Signal survives only where one condition's positives dominate the shared
medium (thermal, 46 genes: **0.15 vs 0.06** baseline) or where the medium is genuinely
distinct (glycerol/acetate — too few positives to measure).


**3. The GSM-coverage gap, quantified on real convergence.** glucose **19/51 = 37 %** in
iJO1366; thermal **46/266 = 17 %**. The majority of convergent ALE targets are
non-metabolic — RNA polymerase (rpoB/C/D), rho/nusA, the cell-shape system (mreB/C/D,
mrdA/B), transport, prophage loci, y-genes — invisible to any flux model.

**4. Mechanistic sanity (figures/fig4).** glpK carries 0 pFBA flux on glucose, 5.4 on
glycerol — the media map genuinely turns genes on; without it every condition would share
flux features by construction.

**5. Sequence features reach the flux-blind genes — and ESM fusion finally beats flux
(figures/fig8).** A sequence-feature track tests the coverage gap directly, in two flavours
over the UniProt K-12 proteome (1,366/1,367 genes matched): **biophysical** (Biopython
ProtParam, torch-free) and **ESM2** embeddings (480-dim, `esm2_t12_35M`). Scored under the *same* cluster-aware CV as flux (single-seed snapshot — see the
30-seed error bars below):

| feature set | macro recall@10 | where it wins |
|---|---|---|
| flux (GSM) | 0.037 | thermal (0.15) |
| biophysical seq-only | **0.12** | acetate/benzoate/glucose (flux-blind) |
| ESM2 seq-only | 0.05 | benzoate/glucose |
| flux + biophysical | 0.01 | — (naive concat collapses) |
| **flux + ESM2** | **0.088** | **acetate (0.33)** |

Two findings. (a) *Sequence alone reaches the genes flux can't see*: crude biophysical
features already score macro **0.12 vs flux 0.037**, winning every condition where flux is
~0. (b) *With a rich enough representation, fusion works*: flux + ESM2 reaches **0.088 — 2.4×
flux** and above ESM-seq alone, whereas naive flux+biophysical concatenation collapsed to
0.01. The ESM gain is concentrated in **acetate (fused 0.33)**, a flux-blind condition where
flux *and* ESM-seq each score 0 alone — the two families recover convergent genes *jointly*
that neither finds alone, a genuine interaction rather than addition. (ESM-seq's lower
*solo* macro than biophysical is the tiny-corpus signature: 480 dense dims need a partner —
or more than 86 positives — to pay off; `esm2_t12_35M` is the small model, 650M is the next
lever.)

**Robustness across 30 CV seeds (figures/fig9).** The corpus is small, so the single-seed
numbers above are noisy; the comparison was repeated over **30 independent CV seeds**
(re-shuffling the gene-held-out folds and the model seed). The rank order is stable in the
mean — **flux 0.056 ± 0.037  <  ESM2-seq 0.078 ± 0.048  <  fused 0.095 ± 0.056** (mean ± SD)
— and **fusion beats flux in 73 % of seeds** (paired mean +0.039). The direction is therefore
robust, not a single-seed artifact (the seed-0 fused 0.088 sat right at the multi-seed mean).
But the error bars overlap and the paired SD (0.068) exceeds the mean gain: with 86 positives
the corpus pins the *direction* of the fusion advantage, not its *magnitude*. Adding flux *on
top of* sequence is the weakest link (fused > seq in only 57 % of seeds) — most of the lift is
sequence reaching the flux-blind genes; flux adds a smaller, noisier increment. Dimensionality
reduction (PCA) tightened it more than a bigger model did — see the size comparison below.

**Model size & dimensionality — bigger did not help fusion (figures/fig10).** Re-embedding
with the 30× larger **ESM2-650M** (1280-dim) tested whether a stronger representation widens
the fusion gap. It does not. Compared first at matched PCA-100 the 650M fused *worse* (0.086
vs the 35M's 0.133) — but that comparison is confounded, because 100 components retain only
87 % of the 1280-dim model's variance vs 92 % of the 480-dim one. So a **variance-matched**
check was added (650M at PCA-200 = 94 % variance): the 650M still does not fuse better
(**fused 0.083**), and paired on shared seeds the 35M-PCA100 fuses higher in **73 %** of them
(+0.050 mean). The 650M's apparent sequence-only edge (0.089 at PCA-100) is fragile — it
collapses to 0.039 at PCA-200 (over-fitting more dims on 86 positives), so it is an
operating-point artifact, not a durable advantage. The standout configuration remains
**35M at PCA-100: fused 0.133 ± 0.055, +0.077 over flux, fusion winning 87 % of seeds**, with
flux genuinely additive on top of sequence. *(Audited: gene order is identical across the two
embedding files, no NaNs, and the flux scores are byte-identical between runs — the harness is
deterministic, so the differences are driven only by the embedding.)* The lesson on this small
corpus: **feature complementarity and dimensionality reduction beat raw model scale**; the one
result robust across every configuration is simply that fusion beats flux (67–87 % of seeds).

The headline holds and sharpens: **sequence/identity signal reaches the regulatory/
transport/structural convergent genes the flux model is structurally blind to, and fused with
ESM embeddings it beats flux alone in ~3 of every 4 CV seeds.**

**Conclusion.** GSM-flux features are **necessary but not sufficient** for ALE-target
prediction: they represent real convergence in-distribution and flag ALE-prone metabolic
genes, but cannot resolve selection when the simulated media match (FBA's medium/temperature/
stress-blindness), and they are blind to the regulatory/structural majority of real targets.
The path forward is **condition-aware features** (temperature-dependent enzyme capacity,
stress-response flux) and **sequence/structural features** (ESM embeddings, ΔΔG). The ESM-fusion result
above (#5) is the first quantitative win on this front — flux + ESM2 (0.088) beats flux
alone (0.037) by recovering flux-blind targets neither feature set finds in isolation.

*Reproducibility:* a couple of FVA reactions have alternate optima differing ~1e-7
run-to-run; `build_features` quantises float features to 4 dp, so every number above
reproduces bit-for-bit from a cold rebuild. Top features (figures/fig2): reaction
connectivity, single-gene-deletion growth ratio, isozyme redundancy, FVA span, temperature.

## Why this, why now

- **The corpus and the meta-analysis pipeline already exist.** Anand et al. (2021, 2023) proved that aggregated ALE mutations carry transferable, design-relevant signal — but their recommendations are frequency/structure statistics, not a GSM-feature-conditioned, gene-level probabilistic predictor with a held-out test.
- **GSM + ML is an established venue lane.** Zampieri et al. (2019, *PLOS Comput. Biol.*) framed the integration; the venue keeps publishing GSM-feature ML. No dedicated ALE-outcome predictor occupies that niche yet.
- **It is a complete, legible artifact.** One repository exercises cobrapy FVA/FBA/deletion, condition-specific feature engineering, a calibrated-ready ML model with leakage-free evaluation, biological validation, and a strain-design output — end to end.

## Repository structure

```
ale-outcome-predictor/
├── README.md
├── corpus/
│   ├── curated_ale_corpus.csv     # committed, literature-cited convergence corpus
│   └── CURATION.md                # provenance + scope + honesty ledger
├── src/ale_outcome_predictor/
│   ├── data_loader.py             # ALEdb 1.0 import + canonical mutation schema
│   ├── feature_engineer.py        # GSM-flux + GPR features / gene / condition
│   ├── media.py                   # selection-condition -> growth-medium mapping
│   ├── baseline_model.py          # gradient-boosted tree + leakage-free splits/metrics
│   └── pipeline.py                # end-to-end driver (corpus -> features -> model -> eval)
├── scripts/run_experiment.py      # full run: metrics.json + figures + ranked predictions
├── scripts/extract_esm_features.py # ESM2 embeddings (local/GPU) → data/esm_features.npz
├── scripts/run_fusion_experiment.py# flux vs sequence vs fused (single seed)
├── scripts/run_fusion_robustness.py# multi-seed error bars + PCA / model-size sweep
├── tests/                         # pure-Python core + fast end-to-end (e_coli_core) tests
├── figures/                       # rendered result figures
├── report/                        # metrics.json, ranked_predictions.csv, methods note
├── data/                          # GSM/ALEdb pulls — gitignored, re-fetchable
│   └── HOWTO_PULL.md               # ALEdb account export -> drop-in (4 steps)
└── features/                      # cached feature matrix — gitignored
```

## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,models]"          # cobra, scikit-learn, lightgbm, pandas
pip install pyarrow matplotlib          # feature cache + figures
```

## Quickstart

```bash
# Full experiment: builds features (cached), trains, evaluates, writes figures.
python scripts/run_experiment.py

# Or programmatically:
python -c "from ale_outcome_predictor import pipeline; r = pipeline.run(); print(r.meta)"
```

```python
from ale_outcome_predictor import pipeline
res = pipeline.run()                     # PipelineResult
res.loco                                 # leave-one-condition-out recall table
res.cluster_cv                           # cluster-aware CV recall table
res.importances                          # feature importances
```

Run the tests:

```bash
PYTHONPATH=src pytest          # core unit tests + a fast e_coli_core end-to-end pass
```

## Data & licensing

- The committed `corpus/curated_ale_corpus.csv` is transcribed from **published** ALE studies (LaCroix 2015, Herring 2006, Cheng/Conrad 2014, Tenaillon 2012, Sandberg 2014), each row cited — so it is redistributable. See `corpus/CURATION.md`.
- **ALEdb** (http://aledb.org) is the full training corpus; bulk-export licensing is unverified, so it is **never committed**. `data_loader.load_aledb()` reads a local pull (`data/HOWTO_PULL.md`); the canonical schema is identical to the corpus, so switching sources is a one-line change.
- **GSMs** come from the cobrapy model repository and are re-fetchable; `data/` is gitignored.

## Acceptance test

1. **Held-out evaluation** — cluster-aware CV (homology-leakage-free) and leave-one-condition-out top-k recall vs a uniform-over-essential-genes baseline. ✅ implemented & reported.
2. **Mechanistic validity** — condition-specific media produce condition-specific flux features. ✅ (`glpK` glucose-vs-glycerol).
3. **Reproducible env + CI test** — `pytest` runs a fast synthetic / e_coli_core end-to-end pass offline. ✅
4. **Scale path** — full ALEdb pull swaps in via `data_loader.load_aledb()` with no downstream change. ✅ wired.

## Prior art

- Anand et al., 2021 — *Data-Driven Strain Design Using Aggregated ALE Mutational Data*, bioRxiv 2021.07.19.452699 (closest prior art).
- Phaneuf et al., 2019 — *ALEdb 1.0*, *Nucleic Acids Research* 47(D1):D1164.
- Zampieri et al., 2019 — *Machine and deep learning meet genome-scale metabolic modeling*, *PLOS Comput. Biol.* 10.1371/journal.pcbi.1007084.
- LaCroix et al., 2015 — glucose-M9 growth ALE, *AEM* 81:17 (corpus source).
- Herring et al., 2006 — glycerol ALE / glpK, *Nat. Genet.* 38:1406 (corpus source).
- Tenaillon et al., 2012 — 42 °C thermal ALE, *Science* 335:457 (corpus source).

## Roadmap

| Milestone | Deliverable | State |
|---|---|---|
| **15.1** | Repository scaffold | ✅ |
| **15.2** | `data_loader.py` — ALEdb import + canonical schema | ✅ |
| **15.3** | `feature_engineer.py` — GSM-flux features per gene per condition | ✅ |
| **15.3b** | `media.py` — condition→medium mapping (real per-condition flux) | ✅ new |
| **15.4** | `baseline_model.py` — GBT + cluster-aware / held-out splits | ✅ |
| **15.5** | `pipeline.py` + `run_experiment.py` — end-to-end train + evaluate + figures | ✅ new |
| **15.6** | Expand corpus (more carbon sources / stressors) or pull full ALEdb | next |
| **15.7** | Sequence features (ESM2) + flux/sequence fusion with multi-seed error bars | ✅ |
| **15.8** | Expand the convergence corpus (more ALEdb selections) to tighten the error bars | next |

## License

MIT © 2026 Daniel González Lozano. See [LICENSE](LICENSE).
