"""Typed random-AST null candidates and reading baselines (vehicle calibration).

Calibration generates deterministic, catalog-legal random factor candidates from
the same DSL, field set and window choices as the research search, so that any
evaluation batch can be compared with a random-factor vehicle baseline instead
of assuming zero.

Scope discipline:
  - readings are descriptive; calibration never grants acceptance, economic
    rewards or JQ authority, and never modifies thresholds, windows or the
    catalog;
  - the generator is additive: it does not change any existing search arm
    (frozen batch code identities must stay reproducible);
  - determinism: same seed and parameters reproduce the same population.
"""

from copy import deepcopy
import random

from .contracts import require
from .dsl import CATALOG, compile_ast
from .search import factor_spec

DEFAULT_WINDOWS = (2, 3, 5, 7, 10, 15, 20)
_MECHANISM = "typed_random_null"
_ACTION = "calibration_random_null"


def _leaf(name):
    return {"field": name}


def _time_op(op, name, window):
    return {"op": op, "args": [{"field": name}], "window": window}


def _sample_atom(rng, fields, windows):
    """Raw field leaf or a time operator on a field leaf (both always legal)."""
    names = tuple(sorted(fields))
    name = rng.choice(names)
    if rng.random() < 0.40:
        return _leaf(name)
    op = rng.choice(("lag", "return", "mean", "std", "min", "max", "delta"))
    return _time_op(op, name, rng.choice(windows))


def _sample_expr(rng, fields, windows, depth):
    """Grammar-biased expression sampler; whole-tree compile_ast validates."""
    if depth <= 0 or rng.random() < 0.42:
        return _sample_atom(rng, fields, windows)
    roll = rng.random()
    if roll < 0.22:
        return {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]}
    if roll < 0.30:
        return {"op": "zscore", "args": [_sample_expr(rng, fields, windows, depth - 1)]}
    if roll < 0.42:
        return {"op": rng.choice(("neg", "abs")),
                "args": [_sample_expr(rng, fields, windows, depth - 1)]}
    if roll < 0.55:
        return {"op": "sub", "args": [
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]},
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]}]}
    if roll < 0.65:
        return {"op": "add", "args": [
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]},
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]}]}
    if roll < 0.75:
        # Same-field ratio: unit-consistent by construction.
        name = rng.choice(tuple(sorted(fields)))
        return {"op": "div", "args": [_leaf(name), _time_op("mean", name, rng.choice(windows))]}
    if roll < 0.85:
        op = rng.choice(("mul", "div"))
        return {"op": op, "args": [
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]},
            {"op": "rank", "args": [_sample_expr(rng, fields, windows, depth - 1)]}]}
    # Price/volume level relative to its own time reference (same units).
    name = rng.choice(tuple(sorted(fields)))
    op = rng.choice(("sub", "div"))
    return {"op": op, "args": [_leaf(name), _time_op("mean", name, rng.choice(windows))]}


def random_ast(fields, seed, *, max_depth=4, windows=DEFAULT_WINDOWS):
    """One catalog-legal random AST (or None if the bounded retries fail)."""
    rng = random.Random(seed)
    for _ in range(64):
        ast = _sample_expr(rng, fields, tuple(windows), max_depth)
        try:
            compile_ast(ast, fields)
        except Exception:  # ContractError from the catalog contract
            continue
        return ast
    return None


def fields_used(ast, found=None):
    """Set of field names referenced by an AST (used by audits and tests)."""
    found = set() if found is None else found
    if "field" in ast:
        found.add(ast["field"])
    for arg in ast.get("args", ()):
        fields_used(arg, found)
    return found


def random_population(fields, count, seed, *, max_depth=4, windows=DEFAULT_WINDOWS,
                      node_budget=32, max_attempts_per_candidate=64):
    """Deterministic null population of `count` distinct catalog-legal candidates.

    Returns (candidates, report). Candidates are full factor specs from
    search.factor_spec with action 'calibration_random_null'; the report carries
    the seed manifest and generation statistics for the frozen batch record.
    """
    require(type(count) is int and count >= 1, "positive candidate count required")
    require(type(max_depth) is int and 1 <= max_depth <= CATALOG["max_depth"], "depth within catalog")
    require(type(node_budget) is int and 1 <= node_budget <= CATALOG["max_nodes"], "node budget within catalog")
    rng = random.Random(seed)
    candidates, seen = [], set()
    report = {"seed": seed, "count": count, "attempts": 0, "compiler_rejected": 0,
              "budget_rejected": 0, "duplicate_rejected": 0,
              "max_depth": max_depth, "node_budget": node_budget,
              "window_choices": list(windows)}
    budget = max_attempts_per_candidate * count
    while len(candidates) < count and report["attempts"] < budget:
        report["attempts"] += 1
        ast = _sample_expr(rng, fields, tuple(windows), max_depth)
        try:
            compiled = compile_ast(ast, fields)
        except Exception:  # ContractError: catalog-legal rejections retried
            report["compiler_rejected"] += 1
            continue
        if compiled["node_count"] > node_budget:
            report["budget_rejected"] += 1
            continue
        spec = factor_spec(deepcopy(ast), fields, _MECHANISM, action=_ACTION)
        if spec["formula_hash"] in seen:
            report["duplicate_rejected"] += 1
            continue
        seen.add(spec["formula_hash"])
        candidates.append(spec)
    require(len(candidates) == count, "random population generation budget exhausted")
    report["generated"] = len(candidates)
    report["node_counts"] = [compile_ast(c["canonical_ast"], fields)["node_count"]
                             for c in candidates]
    return candidates, report


def baseline_summary(values):
    """Quantile summary of readings; None and non-finite excluded (missing stays missing)."""
    xs = [float(v) for v in values
          if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
          and v not in (float("inf"), float("-inf"))]
    n = len(xs)
    if n == 0:
        return {"n": 0}
    xs.sort()

    def q(p):
        if n == 1:
            return xs[0]
        pos = (n - 1) * p
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    center = sum(xs) / n
    sd = (sum((x - center) ** 2 for x in xs) / (n - 1)) ** 0.5 if n >= 2 else None
    return {"n": n, "min": xs[0], "p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
            "p75": q(0.75), "p90": q(0.90), "max": xs[-1], "mean": center, "sd": sd}


def percentile_of(reading, values):
    """Fraction of baseline values <= reading (descriptive position of a reading)."""
    if not isinstance(reading, (int, float)) or isinstance(reading, bool) or reading != reading:
        return None
    xs = [float(v) for v in values
          if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]
    if not xs:
        return None
    return sum(1 for v in xs if v <= reading) / len(xs)


# --- economic distance: combination-layer breakeven positioning (M4) ---
# Frozen breakeven IC anchors (equal-weight weekly rebalance, 15 bps roundtrip cost), fitted in
# reports/fuxi_ic_ceiling_20260924.md: top-tail excess = -0.00843 + 0.0915 * IC (equal-weight arm,
# level t=-6.35; -0.00720 / t=-5.78 after trimming 4 overlapping sections). Solving for zero net
# excess at 15 bps yields +0.108 (equal weight) and +0.083 (R4 reference arm).
# Descriptive positioning only: never an acceptance, threshold, reward or eligibility input.
BREAKEVEN_IC = {"equal_weight_weekly_15bps": 0.108, "r4_reference_weekly_15bps": 0.083}
BREAKEVEN_SOURCE = {
    "report": "reports/fuxi_ic_ceiling_20260924.md",
    "fit": "top_tail_excess = -0.00843 + 0.0915 * IC (equal-weight; trimmed -0.00720 / t=-5.78)",
    "cost_bps_roundtrip": 15, "rebalance": "weekly",
    "usage": "descriptive_triage_positioning_not_reward"}
DEFAULT_BREAKEVEN_KEY = "equal_weight_weekly_15bps"


def breakeven_for(key=DEFAULT_BREAKEVEN_KEY):
    """Frozen breakeven IC anchor by key (audit trail in BREAKEVEN_SOURCE)."""
    require(key in BREAKEVEN_IC, "unknown breakeven key")
    return BREAKEVEN_IC[key]


def economic_distance(mean_rank_ic, breakeven_ic):
    """Descriptive ratio d = IC / breakeven; missing IC stays missing, anchor must be positive."""
    require(isinstance(breakeven_ic, (int, float)) and not isinstance(breakeven_ic, bool)
            and breakeven_ic == breakeven_ic and 0 < breakeven_ic < float("inf"),
            "positive finite breakeven IC required")
    if not isinstance(mean_rank_ic, (int, float)) or isinstance(mean_rank_ic, bool) or mean_rank_ic != mean_rank_ic:
        return None
    return float(mean_rank_ic) / float(breakeven_ic)
