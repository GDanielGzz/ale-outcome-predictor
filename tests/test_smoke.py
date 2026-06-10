"""Smoke test — the minimal CI gate for the scaffold.

Acceptance test, repo gate (research_notes/15, §"Acceptance test" item 4):
"CI runs a small synthetic-data smoke test." This file is the placeholder for
that gate: it asserts the package imports and exposes a version. The real
synthetic-data FVA-pipeline integration test arrives once feature_engineer.py
(15.3) and baseline_model.py (15.4) exist.
"""

import ale_outcome_predictor as aop


def test_package_imports() -> None:
    """The package imports without pulling heavy optional deps (cobra/lightgbm)."""
    assert aop is not None


def test_version_is_exposed() -> None:
    """__version__ is a non-empty, dotted version string."""
    assert isinstance(aop.__version__, str)
    assert aop.__version__.count(".") >= 2


# TODO (15.3 / 15.4): replace this with a synthetic-GSM smoke test once the
# feature pipeline exists — load a tiny toy cobra Model, run a one-reaction FVA
# feature, assert the engineered feature frame has the expected shape. Keep it
# offline + seconds-fast so CI stays green without an ALEdb pull.
