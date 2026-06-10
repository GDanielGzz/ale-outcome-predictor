# Curated ALE convergence corpus — provenance & scope

`curated_ale_corpus.csv` is a **small, literature-curated** set of convergent
metabolic-gene targets from landmark *E. coli* adaptive-laboratory-evolution
(ALE) studies. It exists so the pipeline can train and be evaluated end-to-end
**without** the full ALEdb bulk export, which is licensing-unverified and not
reachable from the build sandbox (see `data/HOWTO_PULL.md`).

Unlike an ALEdb dump, every row here is transcribed from a **published,
citable** result, so this file *is* redistributable and is committed to the repo
(whereas `data/` stays gitignored for any real ALEdb pull).

## What counts as a positive

A row is a `(selection_condition, gene)` pair that a peer-reviewed ALE study
reports as a **convergent / recurrent causal target** under that selection. Only
genes that are **present in the iJO1366 genome-scale model** are included —
because a flux model can only score genes it represents. This is a deliberate,
disclosed scope limit, not an oversight (see "The GSM-coverage gap" below).

| selection_condition | gene (b#) | study | evidence |
|---|---|---|---|
| glucose M9, 37 °C | pyrE (b3642) | LaCroix 2015, *AEM* | pyrE–rph 82-bp intergenic deletion in **every** sequenced clone |
| glycerol M9, 37 °C | glpK (b3926) | Herring 2006, *Nat Genet* | glpK mutations fixed in **47/50** glycerol lineages |
| glycerol M9, 37 °C | glpD (b3426) | Cheng/Conrad 2014, *Nat Commun* | glp-regulon reorganization on glycerol |
| glucose M9, 42 °C | fabA (b0954) | Tenaillon 2012, *Science* | fatty-acid / membrane functional-group convergence |
| glucose M9, 42 °C | fabZ (b0180) | Tenaillon 2012, *Science* | "" |
| glucose M9, 42 °C | cls (b1249) | Tenaillon 2012, *Science* | cardiolipin synthase, membrane-lipid convergence |
| glucose M9, 42 °C | plsB (b4041) | Tenaillon 2012, *Science* | G3P acyltransferase, membrane-lipid convergence |
| glucose M9, 42 °C | plsC (b3018) | Sandberg 2014, *MBE* | acyl-G3P acyltransferase, 42 °C membrane adaptation |

## The GSM-coverage gap (a headline finding, not a bug)

The **most frequent** convergent ALE targets across these same studies are
*regulatory / non-metabolic*: `rpoB`, `rpoC`, `rpoD` (RNA polymerase), `rho`
(transcription termination), `hns`, `iclR`, `mreB`. None of these exist in a
metabolic model, so a flux-feature predictor is **structurally blind** to them.
Quantifying that blind spot is part of the result the pipeline reports
(`gsm_coverage` in the metrics): it bounds the ceiling on what *any*
GSM-flux-only model can achieve and motivates fusing sequence/structural
features (the design note's "Optional v2").

## Deliberately excluded (honesty ledger)

Candidates left **out** because a clean, specific citation to a convergent call
was not confirmed during curation — rather than guess a supplementary-table
gene list from memory:

- `pgi`, `gnd`, `gltA`, `icd` for glucose growth-rate ALE (plausible central-carbon
  targets, but not pinned to a specific convergence table here).
- `fabB`, `fabF`, `fabR` for thermal (same functional group as the included fab
  genes; included only the ones with the firmest attribution).
- Carbon sources (lactate, xylose, acetate) whose textbook gating gene I could
  not attach to a specific ALE convergence report.

These are the obvious **growth path** for the corpus: each added with its own
citation, ideally superseded wholesale by a local ALEdb pull through
`data_loader.load_aledb()` (already wired).

## How to scale past this file

`pipeline.load_corpus()` reads this CSV. The moment a real ALEdb export lands in
`data/` (per `data/HOWTO_PULL.md`), switch the pipeline's source to
`data_loader.load_aledb()` — the canonical schema is identical, so nothing
downstream changes.

## Update — glucose condition is now real ALEdb data (ExpID762, 2026-06-09)

The hand-curated glucose positive (`pyrE`) was **replaced by the real ALEdb
converged-mutation set** from experiment **ExpID762** (the published LaCroix 2015
glucose-M9 ALE; reference BOP27 = K-12 MG1655 / NC_000913.3; 8 lineages A3–A10,
48 samples). The converged set is ALEdb's cross-lineage convergence call — the
authentic positive labels. Files in `data/aledb/GLU_ExpID762/` (gitignored);
loaded via `data_loader.load_aledb_matrix`, names → b-numbers via `gene_map`.

**Real-data GSM coverage:** of **51 distinct convergent genes, 19 (37 %) are in
iJO1366** (pykF, glgC, gcd, fadE, putA, xylB, argI, corA, oppA, cueO, dsbG, fldA,
gltP, paaJ, sstT, tdk, tsx, wecA, xanQ). The rest are regulatory / transport /
chaperone / y-genes — the GSM blind spot, now quantified on real convergence.
Note the famous pyrE–rph operon appears here as out-of-model `rph`, so the earlier
hand-curated `pyrE` was a model-gene proxy for it; the 19 real converged genes
replace that proxy. Positives: 10 → 28. `corpus/ecoli_gene_aliases.csv` extended
+36 entries (UniProt-verified) to map the real gene names.

## Update — thermal condition is now real ALEdb data (ExpID740 / Tenaillon, 2026-06-09)

The hand-curated thermal genes (fabA/fabZ/cls/plsB/plsC/fabB/fabF) were **replaced by
the real ALEdb converged set** from **ExpID740** — Tenaillon 2012's 42 °C ALE, **113
lineages**, E. coli B REL606 in DM25 (699 converged mutations, 266 distinct genes).
Files in `data/aledb/42C_Tenaillon_ExpID740/`; names → b-numbers via `gene_map`
(+72 UniProt-verified aliases added across GLU + Tenaillon).

**Real-data GSM coverage:** **46/266 = 17 %** in iJO1366 (rbs operon, glm/glg, murA,
mrdA, cls, met*, asp*, glp*, …). The convergence is dominated by non-metabolic targets —
RNAP (rpoB/C/D), rho/nusA, the cell-shape system (mreB/C/D, mrdA/B), transport, E. coli B
`ECB_` prophage loci, y-genes. The literature's fatty-acid emphasis is *not* the gene-level
convergence here (cls is in; fab genes are not in the converged calls). Positives 28 → 67.

**Key limitation surfaced:** glucose-37 °C and thermal-42 °C share the same simulated
glucose medium (FBA has no temperature) → flux-degenerate. The model beats baseline per
condition (glucose 0.18, thermal 0.17) but pooling lowers macro recall — flux features
identify ALE-prone genes but not which selection drives them when media match.

## Update — glycerol confirmed in real ALEdb data (ExpID1523, 2026-06-09)

The third E. coli condition is now grounded in real ALEdb too. Public experiment
**ExpID1523** (Ecoli_sexualRecomb_glycerol, *asexual* control strain; M9 glycerol
5 % v/v, 37 °C; Nat Commun 2017, 10.1038/s41467-017-02323-4) is small (6 samples)
but its converged set cleanly recovers the canonical target: **glpK** (I238T +
R189S — glycerol kinase, the textbook glycerol-ALE gene). The regulatory co-target
**rpoB** (H526Y + duplication) is convergent but out-of-model. The other "converged"
calls — `araD/araC` intergenic SNPs in the **arabinose promoter** — are artifacts of
the strain's ara-inducible recombination cassette, **not** glycerol adaptation, and
are excluded. `glpD` is retained as literature support (Cheng 2014) since this small
recombination-focused experiment is underpowered for glycerol convergence.

**Status: all three E. coli conditions now use real ALEdb convergence** — glucose
(ExpID762), thermal (ExpID740), glycerol (ExpID1523, glpK confirmed). Picking the
*public* asexual experiment over the private EEP-Glycerol project (ETC-knockout
backgrounds) keeps the corpus clean and redistributable.

## Update — benzoate + acetate selections (ExpID940 / ExpID1008, 2026-06-09)

Two more real ALEdb selections, 67 → 86 positives (85 real ALEdb). **Benzoate** (ExpID940,
W3110 benzoate tolerization in LB, 10.1128/AEM.02736-16): a new *aromatic-acid stress axis*
(relevant to plant-hydrolysate phenolics), 15 in-GSM converged genes (aceA, add, apt, asnS, fabB,
fdnI, folD, fucA, gatA, icd, ltaE, narH, ompF, tdcD, dcuC). Modelled on the glucose medium
(benzoate toxicity isn't FBA-simulable; real medium was LB), `stress_class=aromatic`.
**Acetate** (ExpID1008, defined-medium acetate, 10.1007/s00253-016-7724-0): a flux-*distinct*
carbon condition (its own acetate flux pass), small (2 samples), 4 in-GSM converged (dacD,
gsiB, oppA, xylB). Many converged genes are strain-specific locus tags (Y75_RS / ECOLC_RS)
that don't map to K-12 b-numbers; `gene_map` aliases extended +18 (UniProt-verified).

**Result:** adding benzoate as a *third* glucose-medium condition collapsed the held-out
glucose recall (0.18 → 0.04) — the decisive demonstration of the flux-degeneracy limitation
(see README "Results" / methods note). In-distribution the model still fits all five
conditions' converged sets (top-10 recovery 7–9/10, acetate 4/4).

## Update — sequence-feature fusion (milestone 16, 2026-06-09)

Added a sequence-feature track to test the GSM-coverage gap directly. Proteome:
UniProt reference **UP000000625** (E. coli K-12, 4,403 seqs); keyed to iJO1366 by
UniProt accession (`model_uniprot_map`) → **1,366/1,367 genes matched (100%)**.
Two paths in `sequence_features.py`: **biophysical** (Biopython ProtParam, torch-free,
runs in-sandbox) and **ESM2** (`transformers`, run on GPU via
`scripts/extract_esm_features.py`, drop in the `.npz`). `scripts/run_fusion_experiment.py`
compares flux vs seq vs fused under the cluster-aware CV.

**Result (biophysical):** sequence-only macro recall@10 = **0.12** vs flux **0.04** — and
sequence *wins in every flux-blind condition* (acetate 0.17, benzoate 0.20, glucose 0.10;
flux ~0 there). Driven by a spread of biophysical features, not a single trivial one.
**Naive flux+seq fusion underperforms both (0.01)** — with 86 positives the GBT can't
combine the feature families stably; regularised/stacked fusion + ESM-strength embeddings
+ more data is the next step. Confirms the core hypothesis: sequence signal reaches the
regulatory/transport/structural convergent genes the flux model misses. Proteome FASTA in
`data/` (gitignored). See `figures/fig8_sequence_fusion.png`, `report/fusion_metrics.json`.

## Update — ESM2 fusion clears flux alone (milestone 16b, 2026-06-09)

Ran the strong sequence track: **ESM2 embeddings** (`facebook/esm2_t12_35M_UR50D`, 480-dim,
mean-pooled) over the same UP000000625 proteome, 1,366/1,367 genes (100 %). Generated locally
(`scripts/extract_esm_features.py`, now cobra-free — reads `data/model_gene_uniprot.csv`),
compared via `scripts/run_fusion_experiment.py --seq esm:...`. Cross-platform reproducible
(Windows run == Linux re-run, identical macro to 4 dp).

**Result (macro recall@10):** flux 0.037 · biophysical-seq 0.12 · ESM2-seq 0.05 ·
flux+biophysical 0.01 · **flux+ESM2 0.088**. ESM fusion is the first feature set to clear
flux alone (2.4×), and beats ESM-seq alone. The gain is concentrated in **acetate
(fused 0.33)** — a flux-blind condition where flux *and* ESM-seq each score 0 solo, so the
two families recover convergent genes jointly that neither finds alone (interaction, not
addition). ESM-seq's lower *solo* macro than biophysical is the tiny-corpus signature
(480 dense dims need a partner or >86 positives). Small model used; 650M is the next lever.
See `figures/fig8_sequence_fusion.png`, `report/fusion_metrics.json`.

## Update — multi-seed robustness on the fusion result (milestone 16c, 2026-06-10)

The flux+ESM2 fusion win was a single CV seed, so it was repeated over **30 independent
seeds** (`scripts/run_fusion_robustness.py`, re-shuffling the gene-held-out folds + model
seed; resumable raw cache in `report/_robustness_raw.json`). Result (mean ± SD macro
recall@10): **flux 0.056 ± 0.037 · ESM2-seq 0.078 ± 0.048 · fused 0.095 ± 0.056**. Paired:
**fused beats flux in 73 % of seeds** (mean +0.039), seq beats flux in 63 %, fused beats seq
in only 57 %. The rank order (fused > seq > flux) holds in the mean and the seed-0 fused
(0.088) sat at the multi-seed mean — so the direction is real, not a cherry-pick. But the
error bars overlap and the paired SD (0.068) exceeds the mean gain: with 86 positives the
corpus fixes the *direction* of the advantage, not its *magnitude*. Most of the lift is
sequence reaching flux-blind genes; flux adds a smaller, noisier increment on top.
See `figures/fig9_fusion_robustness.png`, `report/fusion_robustness.json`.

## Update — ESM size comparison: 650M vs 35M at matched PCA-100 (milestone 16d, 2026-06-10)

Tested whether the 30× larger **ESM2-650M** (1280-dim) widens the fusion gap. Embedded
locally (`scripts/extract_esm_features.py --model facebook/esm2_t33_650M_UR50D`,
1366/1367 genes, 1280-dim); raw 1280-dim is too slow for the GBT sweep, so both models were
compared at **matched PCA-100** (`run_fusion_robustness.py --pca 100`, unsupervised, 30 seeds).

Result (mean ± SD macro recall@10, fused-minus-flux, win-rate):
- 35M raw-480:  flux 0.056 · seq 0.078 · **fused 0.095** · +0.039 · 73 %
- 35M PCA-100:  flux 0.056 · seq 0.075 · **fused 0.133** · +0.077 · 87 %
- 650M PCA-100: flux 0.056 · seq **0.089** · fused 0.086 · +0.030 · 67 %

**Bigger isn't better here.** The 650M has the best *sequence-only* signal (0.089) but the
worst *fusion* (0.086), and flux adds nothing on top of it (fused−seq −0.003, coin-flip) —
its embeddings are rich enough to be redundant with flux. The 35M's embeddings are
*complementary* to flux, and **PCA-100 on the 35M is the best configuration** (fused 0.133,
+0.077 over flux, 87 % win, flux additive on top of seq in 90 % of seeds). Dimensionality
reduction — not model size — unlocked fusion on the 86-positive corpus. See
`figures/fig10_esm_size_comparison.png`, `report/fusion_robustness_{35m,650m}_pca100.json`.

## Audit — is "bigger isn't better" real? (milestone 16e, 2026-06-10)

The 650M-fuses-worse result was audited three ways. (1) **Not a bug:** gene order is identical
across the 35M and 650M npz, no NaN/inf, and the flux scores are byte-identical between the two
robustness runs (same seeds → same folds), so only the embedding differs. (2) **Not a PCA-
compression artifact:** PCA-100 retains 92.3 % of the 35M's variance but only 86.9 % of the
650M's, so a **variance-matched** run was added — 650M at PCA-200 (93.5 % variance). The 650M
still does not fuse better (fused 0.083 vs 35M-PCA100 0.133); paired on 30 shared seeds the
35M-PCA100 fuses higher in **73 %** of them (+0.050 mean). (3) The 650M's seq-only "edge"
(0.089 at PCA-100) is **operating-point-fragile** — it collapses to 0.039 at PCA-200, i.e.
over-fitting more dims on 86 positives, not a durable advantage. Conclusion stands: a bigger
model did not improve fusion here; 35M-PCA100 is the best config; the only configuration-robust
result is fusion > flux (67–87 % of seeds). fig10 now includes the variance-matched bar.
