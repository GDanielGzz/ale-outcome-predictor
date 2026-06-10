"""Selection-condition -> growth-medium mapping  (milestone 15.3b).

The feature engineer (``feature_engineer.apply_environment``) can push exchange-
reaction bound overrides onto a model, but on its own it has **no map** from a
free-text ALE condition ("glycerol M9 minimal 37C") to the exchange reactions
that encode that medium — so without this module every condition is scored at
the model's *default* medium and the flux features never actually change between
conditions. That defeats the point: the whole premise is that the flux signature
of a gene *under the selection medium* carries the signal.

This module closes that gap for BiGG-style *E. coli* models (iJO1366 / iML1515),
whose exchange reactions follow the ``EX_<met>_e`` convention. It detects the
carbon source named in a condition string, opens that uptake, closes the other
common carbons, and (optionally) sets aerobic/anaerobic O2. Temperature and
abstract stressors are **not** simulable on an unmodified FBA model and stay
*features*, not constraints (see ``feature_engineer`` "FBA has no temperature").

Public API
----------
* ``CARBON_EXCHANGES``      carbon-source keyword -> BiGG exchange id.
* ``resolve_exchange_bounds(text, model)``  condition text -> ``{EX_id: (lb, ub)}``.
* ``resolve_environment(text, model)``      a fully-populated ``SelectionEnvironment``
  (heuristic temp/stress from ``feature_engineer`` + medium exchange bounds here).
* ``make_env_resolver(model)``  a one-arg resolver to hand to
  ``feature_engineer.build_feature_matrix(..., env_resolver=...)``.

Design notes
------------
* **Dependency-light import.** ``cobra`` is only referenced under
  ``TYPE_CHECKING`` and inside functions, mirroring the rest of the package, so
  importing this module in the CI smoke test pulls no solver stack.
* **Bounds are uptake-side only.** We set the *lower* bound (uptake is negative
  flux in cobrapy) and leave secretion (upper bound) at +1000. Closing a carbon
  means lower bound 0, not blocking secretion.
* **Unknown carbon -> default medium.** If no known carbon keyword is found the
  resolver returns no overrides, so the model keeps its shipped medium (glucose
  aerobic for iJO1366) — a safe, documented fallback rather than a guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-hint only; no runtime cobra dependency for import
    from cobra import Model

    from .feature_engineer import SelectionEnvironment

__all__ = [
    "CARBON_EXCHANGES",
    "DEFAULT_CARBON_UPTAKE",
    "AEROBIC_O2_UPTAKE",
    "resolve_exchange_bounds",
    "resolve_environment",
    "make_env_resolver",
]

# Carbon-source keyword (scanned over the lower-cased condition string) -> the
# BiGG exchange reaction that supplies it. Order matters only for reporting; the
# scan tries every key and the matched ones are opened.
CARBON_EXCHANGES: dict[str, str] = {
    "glucose": "EX_glc__D_e",
    "glycerol": "EX_glyc_e",
    "acetate": "EX_ac_e",
    "lactate": "EX_lac__D_e",
    "succinate": "EX_succ_e",
    "xylose": "EX_xyl__D_e",
    "galactose": "EX_gal_e",
    "fructose": "EX_fru_e",
    "gluconate": "EX_glcn_e",
    "pyruvate": "EX_pyr_e",
}

# Standard ALE uptake bound (mmol gDW^-1 h^-1). 10 is the iJO1366 default for
# glucose; we reuse it for every swapped carbon so growth rates stay comparable.
DEFAULT_CARBON_UPTAKE: float = 10.0
# Aerobic O2 uptake bound; -1000 (effectively unconstrained) matches iJO1366.
AEROBIC_O2_UPTAKE: float = 1000.0

# Tokens that imply anaerobic selection (rare in this corpus, supported anyway).
_ANAEROBIC_HINTS: tuple[str, ...] = ("anaerob", "anoxic", "fermentat")


def _present_carbon_exchanges(model: "Model") -> dict[str, str]:
    """Subset of ``CARBON_EXCHANGES`` whose exchange id exists in ``model``."""
    return {kw: ex for kw, ex in CARBON_EXCHANGES.items() if ex in model.reactions}


def resolve_exchange_bounds(
    text: str | None,
    model: "Model",
    *,
    uptake: float = DEFAULT_CARBON_UPTAKE,
) -> dict[str, tuple[float, float]]:
    """Map a condition string to ``{exchange_id: (lower, upper)}`` overrides.

    Opens every carbon named in ``text`` (lower bound ``-uptake``) and closes the
    other known carbons (lower bound 0) so the medium is unambiguous. If ``text``
    names no known carbon, returns ``{}`` (keep the model's default medium). O2 is
    set to aerobic unless an anaerobic hint is present.

    Only exchange ids actually present in ``model`` are emitted, so the result is
    safe to pass straight to ``feature_engineer.apply_environment``.
    """
    present = _present_carbon_exchanges(model)
    if not present:
        return {}

    lower = (text or "").lower()
    named = [ex for kw, ex in present.items() if kw in lower]

    bounds: dict[str, tuple[float, float]] = {}
    if named:
        named_set = set(named)
        for ex in present.values():
            if ex in named_set:
                bounds[ex] = (-abs(uptake), 1000.0)   # open this carbon's uptake
            else:
                bounds[ex] = (0.0, 1000.0)             # close the others' uptake

    # Oxygen regime (only if the model has an O2 exchange).
    if "EX_o2_e" in model.reactions:
        anaerobic = any(h in lower for h in _ANAEROBIC_HINTS)
        bounds["EX_o2_e"] = (0.0, 1000.0) if anaerobic else (-abs(AEROBIC_O2_UPTAKE), 1000.0)

    return bounds


def resolve_environment(text: str | None, model: "Model") -> "SelectionEnvironment":
    """A ``SelectionEnvironment`` with heuristic temp/stress **and** medium bounds.

    Wraps ``feature_engineer.parse_selection_condition`` (temperature, stress
    class, minimal-media flag) and attaches the carbon/O2 exchange bounds resolved
    here. This is the object ``apply_environment`` consumes to make the flux
    features genuinely condition-specific.
    """
    from dataclasses import replace

    from .feature_engineer import parse_selection_condition

    base = parse_selection_condition(text)
    bounds = resolve_exchange_bounds(text, model)
    return replace(base, exchange_bounds=bounds)


def make_env_resolver(model: "Model"):
    """Return a one-arg ``resolver(text) -> SelectionEnvironment`` bound to ``model``.

    Convenience for ``build_feature_matrix(..., env_resolver=make_env_resolver(model))``
    so the matrix builder can resolve media without knowing about this module.
    """

    def _resolver(text: str | None) -> "SelectionEnvironment":
        return resolve_environment(text, model)

    return _resolver
