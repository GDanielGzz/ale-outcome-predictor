# Ranking adaptive-laboratory-evolution targets from genome-scale metabolic-model flux features

*Methods note — proof-of-concept. Daniel González Lozano, 2026.*

## Abstract

Adaptive laboratory evolution (ALE) selects microbial populations in a defined
environment; the genes that acquire causal mutations are not random but cluster
on the metabolic bottlenecks the selection creates. We ask whether a model can
rank genes by ALE-mutation propensity *prospectively*, from genome-scale
metabolic-model (GSM) flux features computed under the selection medium. We wire
a complete pipeline — cobrapy FVA / FBA / single-gene-deletion feature
engineering, a gradient-boosted-tree classifier, and leakage-free held-out
evaluation — over the *E. coli* iJO1366 model and a small, literature-cited
convergence corpus. The condition-conditioned flux features are mechanistically
correct (glycerol kinase is flux-bearing only on glycerol), the model recovers
known convergent metabolic targets in-distribution, and a homology-leakage-free
cross-validation beats a uniform-over-essential-genes baseline ~2× at top-10.
Cross-selection transfer (leave-one-condition-out) fails at this corpus size — an
honest bound that, together with the observation that the dominant convergent
ALE targets are regulatory genes absent from any metabolic model, defines the
path to a powered model: the full ALEdb corpus plus sequence/structural features.

## 1. Motivation

Aggregated-ALE meta-analyses (Anand et al., 2021; Phaneuf et al., 2019) establish
that convergent ALE mutations carry transferable, design-relevant signal, but
report frequency/structure statistics rather than a gene-level, GSM-feature-
conditioned predictor with a held-out test. GSM + ML is an established lane
(Zampieri et al., 2019), yet no dedicated ALE-outcome predictor occupies it. The
prospective question — *given a chassis GSM and a selection, rank the genes by
mutation probability* — is the natural follow-on to constraint-based strain
design, where flux analysis already nominates engineering targets.

## 2. Methods

**Chassis model.** *E. coli* K-12 iJO1366 (1,367 genes, 2,583 reactions), loaded
via cobrapy from the cobrapy model repository and cached locally. Gene identifiers
are b-numbers, used as the join key to the corpus.

**Selection → medium mapping (`media.py`).** Each free-text selection condition is
parsed for (a) a carbon source, mapped to its `EX_<met>_e` exchange reaction and
opened at −10 mmol gDW⁻¹ h⁻¹ while the other carbons are closed; (b) a temperature
and a coarse stress class, carried as *features* (an unmodified FBA model has no
temperature). This makes the flux features genuinely condition-specific; without
it every condition collapses to the model's default medium.

**Features (`feature_engineer.py`).** Per (gene, condition): single-gene-deletion
growth ratio and essentiality; FVA max |flux| and span over the gene's reactions
at 90 % of optimum; parsimonious-FBA flux usage; blocked-reaction count; and
solver-free GPR/topology structure (reaction count, isozyme-group and complex
arity, subsystem span). Flux features are computed once per *distinct medium* and
reused across conditions that share it.

**Labels.** A `(condition, gene)` pair is positive if a peer-reviewed ALE study
reports that gene as a convergent causal target under that selection; every other
model gene under that condition is a negative. The committed corpus has eight
in-GSM positives across three conditions (glucose-M9 37 °C, glycerol-M9 37 °C,
glucose-M9 42 °C), each row cited (`corpus/CURATION.md`).

**Model + evaluation (`baseline_model.py`).** A LightGBM gradient-boosted tree
(small-corpus config: 150 trees, 7 leaves, single-sample leaves), NaN-native so
the nullable feature table needs no imputation. Evaluation is grouped to avoid
leakage: (i) cluster-aware K-fold holding whole genes out (homology-leakage-free),
(ii) leave-one-condition-out for cross-selection transfer. Both report macro
top-k recall per condition against a uniform-over-essential-genes baseline.

## 3. Results

| Test | Metric | Model | Baseline |
|---|---|---|---|
| Cluster-aware CV (pooled) | macro recall@10 | **0.15** | 0.03 |
| Cluster-aware CV — glucose / thermal | recall@10 per cond | **0.18 / 0.17** | 0.00 / 0.06 |
| Leave-one-condition-out | recall@10 | 0.00 | 0.0–0.2 |

- **Mechanistic validity.** glpK parsimonious-FBA flux: 0 (glucose) → 5.4
  (glycerol) mmol gDW⁻¹ h⁻¹ (Fig. 4).
- **In-distribution recovery.** Ranking all 1,367 genes per condition, the known
  targets top the list — `pyrE` (glucose); `plsC/plsB/fabZ/fabA/cls` (thermal);
  `glpD/glpK` (glycerol), with `glpK` top-ranked only under glycerol
  (`report/ranked_predictions.csv`).
- **Generalisation.** Cluster-aware CV beats baseline ~2× at top-10 (Fig. 1), with
  high variance across 3 informative folds (8 positives total).
- **Transfer.** Leave-one-condition-out recall is 0 (Fig. 3): per-selection
  metabolic targets are idiosyncratic and do not transfer at this corpus size.
- **Feature drivers** (Fig. 2): reaction connectivity, single-gene-deletion growth
  ratio, isozyme redundancy, FVA span, and the temperature feature.

## 4. Limitations (stated, not hidden)

1. **Corpus size.** Eight in-GSM positives is a proof-of-concept, not a powered
   benchmark; the CV numbers carry wide intervals. The pipeline reads the full
   ALEdb (`data_loader.load_aledb()`) unchanged once pulled locally.
2. **GSM-coverage ceiling.** The most frequent convergent ALE targets (`rpoB`,
   `rpoC`, `rho`, `hns`) are regulatory and outside any metabolic model. A
   flux-only predictor is structurally blind to them; the metrics report this gap.
3. **FBA has no temperature.** Thermal selection is encoded only as a feature, so
   the 37 °C and 42 °C glucose conditions share flux features and differ only in
   the condition columns — the model cannot mechanistically "see" thermal stress.
4. **Curation, not extraction.** The corpus is hand-transcribed from headline
   convergence results; a programmatic ALEdb pull would remove curator bias.

## 5. Next steps

- Swap the corpus for a local ALEdb export (already wired) → many conditions,
  hundreds of positives, real held-out-chassis recall.
- Add sequence/structural features (ESM embeddings, ΔΔG) to reach the regulatory
  hotspots GSMs miss — the natural fusion with the portfolio's protein-stability work.
- Flux-coupling features (real FCA) and a presence-only / PU-learning label model
  instead of hard negatives.

## References

LaCroix et al. 2015, *AEM* 81:17 · Herring et al. 2006, *Nat. Genet.* 38:1406 ·
Cheng/Conrad et al. 2014, *Nat. Commun.* 5:3233 · Tenaillon et al. 2012,
*Science* 335:457 · Sandberg et al. 2014, *MBE* 31:2647 · Anand et al. 2021,
bioRxiv 2021.07.19.452699 · Phaneuf et al. 2019, *NAR* 47(D1):D1164 · Zampieri et
al. 2019, *PLOS Comput. Biol.* 10.1371/journal.pcbi.1007084 · Orth et al. 2011
(iJO1366), *Mol. Syst. Biol.* 7:535.

## Update — real ALEdb glucose data (ExpID762, 2026-06)

The glucose condition's hand-curated positive (pyrE) was replaced by the **real
ALEdb converged-mutation set** from experiment ExpID762 (LaCroix 2015 glucose-M9
ALE, 8 independent lineages A3–A10, 48 samples, 732 mutations / 104 converged).
Converged genes are those mutated convergently across independent ALE lineages —
ALEdb's own positive calls. Loaded with `data_loader.load_aledb_matrix`, gene
names mapped to b-numbers via `gene_map` (UniProt-seeded).

**GSM coverage on real convergence data:** of **51 distinct convergent genes**,
**19 (37 %) are in iJO1366** — pykF, glgC, gcd, fadE, putA, xylB, argI, corA,
oppA, cueO, dsbG, fldA, gltP, paaJ, sstT, tdk, tsx, wecA, xanQ. The other 63 % are
regulatory (rpoB/C, hns, evgA), transport (corA*, tsx*, mdtF, sstT*), a chaperone
(dnaK), and uncharacterised y-genes — invisible to a flux model. This is the
GSM-coverage ceiling, now measured on real convergence rather than literature.

**Effect on the model.** Positives 10 → 28. Cluster-aware CV jumps to **model@10 =
0.35 vs 0.03 baseline (~10×)** — the gap widens precisely *because* the real
convergent genes are mostly non-essential, so the uniform-essential baseline barely
flags them while the GSM-flux features rank them highly (top glucose predictions:
oppA, cueO, fadE, tsx, gltP, putA, corA). All numbers reproduce bit-for-bit from a cold rebuild.


## Update — real ALEdb thermal data (ExpID740, Tenaillon 2012)

The hand-curated thermal genes (fab*/cls/pls*) were replaced by the **real ALEdb
converged set** from ExpID740 — Tenaillon's 42 °C ALE, **113 lineages**, E. coli B
REL606 in DM25, 699 converged mutations / 266 distinct genes. Of those, **46 (17 %)
are in iJO1366** (rbs operon, glm/glg, murA, mrdA, cls, met*, asp*, glp*, …). The
real gene-level convergence is dominated by **non-metabolic** targets: RNA polymerase
(rpoB/C/D), termination (rho, nusA), the cell-shape/peptidoglycan system (mreB/C/D,
mrdA/B), transport, E. coli B prophage `ECB_` loci, and y-genes — far more than the
fatty-acid genes literature reviews emphasise. Positives 28 → 67.

**Per-condition result and the key limitation.** On the real sets the model beats the
essential-gene baseline in each condition (glucose 0.18 vs 0.00; thermal 0.17 vs 0.06).
But glucose-37 °C and thermal-42 °C resolve to the **same simulated glucose medium**
(FBA has no temperature), so they are **flux-degenerate**: the model can flag ALE-prone
metabolic genes but not which selection drives them when the media match, and pooling
their distinct convergent sets lowers macro recall (0.35 glucose-only → 0.15 pooled).
This precisely locates where GSM-flux prediction works (within an informative medium) and
fails (across selections that share a medium) — the same temperature-blindness across selections that share a medium. All numbers reproduce from a cold rebuild.

## Update — benzoate + acetate (ExpID940 / ExpID1008): the decisive flux-degeneracy demo

Two more real ALEdb selections: **benzoate** aromatic-acid stress (ExpID940, W3110 in LB,
17 samples, 15 in-GSM converged — new `aromatic` stress class) and **acetate** (ExpID1008,
defined-medium acetate carbon, 2 samples, 4 in-GSM converged — a genuinely flux-*distinct*
medium). Positives 67 → 86; five E. coli conditions, all real ALEdb.

**Two-part finding.** (i) *In-distribution* the model fits every condition's converged set
— top-10 recovery glucose 9, thermal 9, benzoate 8, glycerol 7, acetate 4/4. (ii) *Held-out
generalisation* is bounded by flux-degeneracy: three conditions share the glucose medium,
and adding benzoate as a third collapsed glucose held-out recall **0.18 → 0.04** — the
cleanest empirical proof that GSM-flux cannot resolve selection when media match. The model
keeps signal only where one condition dominates the shared medium (thermal 0.15 vs 0.06) or
the medium is distinct. Conclusion: **GSM-flux is necessary but not sufficient** — path
forward is condition-aware + sequence/structural features. All numbers reproduce from a cold
rebuild.

## Sequence-feature fusion (milestone 16)

A sequence track complements the GSM-flux features to test the coverage gap directly.
Two representations over the UniProt K-12 proteome (UP000000625, 1,366/1,367 genes resolved
by accession via the model's gene annotations): biophysical (Biopython ProtParam) and
mean-pooled ESM2 embeddings (`esm2_t12_35M`, 480-dim). All three feature sets — flux, seq,
flux+seq — are scored under the identical cluster-aware (gene held-out) CV, per condition,
recall@10. Result: flux 0.037, biophysical-seq 0.12, ESM2-seq 0.05, flux+biophysical 0.01,
**flux+ESM2 0.088**. ESM fusion is the first feature set to exceed flux alone, with the gain
concentrated in the flux-blind acetate condition (fused 0.33; flux and ESM-seq each 0 solo) —
a feature-family interaction. ESM extraction (`scripts/extract_esm_features.py`) and the
fusion comparison (`scripts/run_fusion_experiment.py`) are cobra-free (gene→UniProt map
exported to `data/model_gene_uniprot.csv`); flux features come from the committed cache.

## Robustness (milestone 16c)

The fusion comparison was repeated over 30 independent CV seeds (re-shuffling the
gene-held-out folds and the LightGBM seed; `scripts/run_fusion_robustness.py`, resumable).
Mean ± SD macro recall@10: flux 0.056 ± 0.037, ESM2-seq 0.078 ± 0.048, fused 0.095 ± 0.056.
Paired: fusion exceeds flux in 73 % of seeds (mean +0.039), seq exceeds flux in 63 %, fused
exceeds seq in 57 %. The rank order is stable in the mean and the single-seed snapshot (0.088)
matches the multi-seed mean, but the error bars overlap and the paired SD (0.068) exceeds the
mean gain — on 86 positives the result establishes the direction of the fusion advantage, not
its magnitude. See figures/fig9_fusion_robustness.png, report/fusion_robustness.json.

## ESM model-size comparison (milestone 16d)

ESM2-650M (1280-dim) was compared to ESM2-35M (480-dim) at matched PCA-100 (unsupervised
reduction to 100 components; `run_fusion_robustness.py --pca 100`, 30 CV seeds each). The
650M yields the best sequence-only macro recall@10 (0.089 vs 0.075) but the worst fused
(0.086 vs the 35M-PCA100's 0.133), and flux is non-additive on top of the 650M (fused−seq
−0.003). The best configuration is 35M at PCA-100: fused 0.133 ± 0.055, +0.077 over flux,
fusion winning 87 % of seeds and flux additive on top of sequence in 90 %. On this 86-positive
corpus, feature complementarity and dimensionality reduction outweigh model scale. See
figures/fig10_esm_size_comparison.png and report/fusion_robustness_{35m,650m}_pca100.json.

## Audit of the model-size comparison (milestone 16e)

The "650M fuses worse" finding was checked for artifacts. Gene order is identical across the
two embedding files, there are no NaN/inf values, and flux scores are byte-identical between
the 35M and 650M robustness runs (shared seeds → shared folds), confirming a deterministic
harness in which only the embedding differs. Because PCA-100 retains unequal variance (92.3 %
of the 480-dim 35M vs 86.9 % of the 1280-dim 650M), a variance-matched run was added (650M at
PCA-200, 93.5 % variance); the 650M still does not fuse better (0.083 vs the 35M-PCA100's
0.133), and the 35M fuses higher in 73 % of 30 paired seeds. The 650M's PCA-100 sequence-only
value (0.089) is operating-point-fragile, collapsing to 0.039 at PCA-200. The conclusion is
therefore robust: a larger model did not improve fusion on this 86-positive corpus.
