"""Small typed AST interpreter. Time operators intentionally accept field leaves only."""

import hashlib
import math
import os
from pathlib import Path
from .contracts import digest, keys, require, finite

CATALOG = {
    "version": "fuxi.dsl.2", "max_nodes": 64, "max_depth": 12, "max_window": 252,
    "operators": {"neg": 1, "add": 2, "sub": 2, "mul": 2, "div": 2,
                  "rank": 1, "lag": 1, "return": 1, "mean": 1,
                  "std": 1, "min": 1, "max": 1, "delta": 1,
                  "abs": 1, "square": 1, "log": 1, "zscore": 1},
    "rank": "eligible current cross-section; average ties; (rank-0.5)/n; n>=2",
    "time": "explicit session calendar; field leaves only; available_at<=decision_time",
    "missing": "propagate; div zero=>missing; complete mean window required",
}
UNITS = {"price": (1, 0), "shares": (0, 1), "ratio": (0, 0), "currency": (1, 1)}


# ---------------------------------------------------------------- ScoreRef cache key (P5)
# P5.1: the per-snapshot score cache key upgrades from a bare digest(ast) to the
# nine-component content-addressed key (score_ref.score_key) behind
# FUXI_V1_SCORE_REF_KEY. Within one snapshot instance every component except the
# AST is constant, so on/off produce identical values and identical hit rates;
# the tightened key earns safe cross-instance reuse and computable evidence
# counters. Default on (v0.7.186, four-gate verified: getter / A-B counterfactual /
# default-identity invariant / end-to-end reconciliation); explicit off restores the
# historical bare-digest key.

def score_ref_mode():
    mode = os.environ.get("FUXI_V1_SCORE_REF_KEY", "on")
    require(mode in {"off", "on"}, "unknown score ref key mode")
    return mode


def _eval_impl_ref():
    """Implementation identity: source bytes of the two evaluators."""
    parts = []
    for name in ("dsl.py", "columnar.py"):
        try:
            parts.append(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest())
        except FileNotFoundError:
            continue
    return digest({"evaluators": parts})


def score_ref_context(snapshot, universe=None):
    """Lazily build + cache the nine-component key context on the snapshot.

    ``universe`` defaults to ``snapshot.universe`` (dict form); the columnar
    snapshot passes ``snapshot.instruments`` instead. Built once per snapshot
    instance (all components except the AST are instance-constant).
    """
    ctx = getattr(snapshot, "_score_ref_ctx", None)
    if ctx is not None:
        return ctx
    payload = snapshot.payload
    if universe is None:
        universe = snapshot.universe
    ctx = {"operator_catalog_hash": digest({"catalog_version": CATALOG["version"],
                                            "fields": payload["fields"]}),
           "snapshot_ref": digest(payload),
           "universe_ref": digest(universe),
           "eval_window": [snapshot.sessions[snapshot.start], snapshot.sessions[-1]],
           "point_in_time_rule": "snapshot_decision_clocks",
           "adjustment_rule": "declared_adjustment_raw_or_not_applicable",
           "missing_rule": "null_propagates_div_zero_missing",
           "eval_impl_ref": _eval_impl_ref()}
    snapshot._score_ref_ctx = ctx
    return ctx


def _score_cache_key(ast, snapshot, universe=None):
    if score_ref_mode() == "on":
        from .score_ref import score_key
        _, score_id = score_key(canonical_ast=ast,
                                **score_ref_context(snapshot, universe))
        return score_id
    return digest(ast)


def _score_stats(snapshot, cache):
    stats = getattr(snapshot, "_score_ref_stats", None)
    if stats is None:
        stats = {"mode": score_ref_mode(), "computes": 0, "hits": 0,
                 "cache_present": cache is not None}
        snapshot._score_ref_stats = stats
    return stats


def compile_ast(ast, fields):
    count = 0
    required_fields = set()
    def walk(node, depth):
        nonlocal count
        count += 1
        require(count <= CATALOG["max_nodes"] and depth <= CATALOG["max_depth"], "AST budget exceeded")
        if isinstance(node, dict) and "field" in node:
            keys(node, ("field",))
            require(node["field"] in fields, "unknown field")
            required_fields.add(node["field"])
            return UNITS[fields[node["field"]]["unit"]], 0
        keys(node, ("op", "args"), ("window",))
        op = node["op"]
        require(op in CATALOG["operators"], "unsupported operator")
        require(isinstance(node["args"], list) and len(node["args"]) == CATALOG["operators"][op],
                "operator arity mismatch")
        temporal = op in {"lag", "return", "mean", "std", "min", "max", "delta"}
        require(("window" in node) == temporal, "window only/always required for time operators")
        units, lookbacks = zip(*(walk(arg, depth + 1) for arg in node["args"]))
        if temporal:
            w = node["window"]
            require(type(w) is int and 1 <= w <= CATALOG["max_window"], "invalid window")
            require(set(node["args"][0]) == {"field"}, "time operators currently require field leaves")
            require(op != "std" or w >= 2, "std requires at least two sessions")
            return ((0, 0) if op == "return" else units[0]), w if op in {"lag", "return", "delta"} else w - 1
        if op in {"add", "sub"}:
            require(units[0] == units[1], "incompatible addition/subtraction units")
        unit = units[0]
        if op in {"rank", "zscore"}:
            unit = (0, 0)
        elif op == "log":
            require(unit == (0, 0), "log requires dimensionless input")
        elif op == "square":
            unit = tuple(2*x for x in unit)
        elif op in {"mul", "div"}:
            sign = 1 if op == "mul" else -1
            unit = tuple(a + sign * b for a, b in zip(units[0], units[1]))
        return unit, max(lookbacks)
    unit, lookback = walk(ast, 1)
    return dict(formula_hash=digest(ast), catalog_hash=digest(CATALOG),
                required_fields=sorted(required_fields), output_unit=list(unit),
                lookback=lookback, node_count=count)


def ranks(values):
    """Average ties; no arbitrary order among equal scores."""
    valid = sorted((value, key) for key, value in values.items() if value is not None)
    result = {key: None for key in values}
    n = len(valid)
    if n < 2:
        return result
    index = 0
    while index < n:
        end = index + 1
        while end < n and valid[end][0] == valid[index][0]:
            end += 1
        rank = ((index + 1 + end) / 2 - 0.5) / n
        for _, key in valid[index:end]:
            result[key] = rank
        index = end
    return result


def evaluate_ast(ast, snapshot):
    if snapshot.payload.get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import evaluate_columnar_ast
        return evaluate_columnar_ast(ast, snapshot)
    compile_ast(ast, snapshot.payload["fields"])
    cache = getattr(snapshot, '_score_cache', None)
    stats = _score_stats(snapshot, cache)
    cache_key = _score_cache_key(ast, snapshot) if cache is not None else None
    if cache is not None and cache_key in cache:
        stats["hits"] += 1
        return cache[cache_key]
    output = {}
    for index in range(snapshot.start, len(snapshot.sessions)):
        session = snapshot.sessions[index]
        members, asof = snapshot.universe[session], snapshot.clocks[session]
        def visit(node):
            if "field" in node:
                return {s: snapshot.value(index, s, node["field"], asof) for s in members}
            op, args = node["op"], node["args"]
            if op in {"lag", "return", "mean", "std", "min", "max", "delta"}:
                field, w = args[0]["field"], node["window"]
                result = {}
                for s in members:
                    current = snapshot.value(index, s, field, asof)
                    previous = snapshot.value(index - w, s, field, asof)
                    if op == "lag":
                        value = previous
                    elif op == "return":
                        value = None if current is None or previous in (None, 0) else current / previous - 1
                    elif op == "delta":
                        value = None if current is None or previous is None else current-previous
                    else:
                        window = [snapshot.value(i, s, field, asof) for i in range(index-w+1, index+1)]
                        value = None
                        if all(v is not None for v in window):
                            if op == "mean":
                                value = math.fsum(v/w for v in window)
                            elif op == "min":
                                value = min(window)
                            elif op == "max":
                                value = max(window)
                            else:
                                from statistics import stdev
                                value = stdev(window)
                    result[s] = value if finite(value) else None
                return result
            left = visit(args[0])
            if op == "rank":
                return ranks(left)
            if op == "zscore":
                from statistics import mean, stdev
                valid = [v for v in left.values() if v is not None]
                center = mean(valid) if valid else 0
                scale = stdev(valid) if len(valid) >= 2 else 0
                return {s: (v-center)/scale if v is not None and scale > 0 else None for s, v in left.items()}
            if op in {"abs", "square", "log"}:
                result = {}
                for s, v in left.items():
                    value = None if v is None or op == "log" and v <= 0 else (
                        abs(v) if op == "abs" else v*v if op == "square" else math.log(v))
                    result[s] = value if finite(value) else None
                return result
            if op == "neg":
                return {s: -v if v is not None else None for s, v in left.items()}
            right, result = visit(args[1]), {}
            for s in members:
                a, b = left[s], right[s]
                if a is None or b is None or (op == "div" and b == 0):
                    result[s] = None
                    continue
                value = {"add": lambda: a+b, "sub": lambda: a-b,
                         "mul": lambda: a*b, "div": lambda: a/b}[op]()
                result[s] = value if finite(value) else None
            return result
        output[session] = visit(ast)
    stats["computes"] += 1
    if cache is not None:
        cache[cache_key] = output
    return output


def generate_candidates(fields, windows=(2, 5)):
    """Deterministic bounded enumeration. Both signs are independent research trials."""
    candidates = []
    for window in windows:
        for polarity in (1, -1):
            ast = {"op": "rank", "args": [{"op": "return", "window": window,
                                           "args": [{"field": "close"}]}]}
            if polarity == -1:
                ast = {"op": "neg", "args": [ast]}
            compiled = compile_ast(ast, fields)
            spec = dict(schema_version="fuxi.factor.1", canonical_ast=ast,
                        polarity=polarity, hypothesis_id="price_persistence_vs_reversal",
                        frequency="daily", missing_policy="propagate", parent_ids=[],
                        role="research_feature", **compiled)
            spec["candidate_id"] = "factor_" + digest(spec)
            candidates.append(spec)
    require(len({x["candidate_id"] for x in candidates}) == len(candidates), "duplicate generation")
    return candidates
