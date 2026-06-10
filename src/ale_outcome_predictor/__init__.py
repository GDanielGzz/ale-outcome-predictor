"""ALE outcome predictor.

A supervised model that predicts the gene(s) most likely to acquire causal
mutations under a defined adaptive-laboratory-evolution (ALE) selection, given
a chassis genome-scale metabolic model (GSM), the selection environment, and
flux-derived features (FVA / flux-coupling / essentiality) computed on that GSM.

Package layout (modules land in later scaffold sub-tasks):
    data_loader.py       ALEdb 1.0 import + mutation-table normalisation   (15.2)
    feature_engineer.py  GSM-flux features per gene per condition          (15.3)
    baseline_model.py    gradient-boosted-tree baseline, cluster-aware     (15.4)
                         splits + calibration

This file deliberately exposes only metadata; importing the package must stay
dependency-light so the CI smoke test can run without cobra/lightgbm installed.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
