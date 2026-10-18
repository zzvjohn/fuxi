"""Compile a read-only JQ diagnostic. No trading API, no promotion or provider parity claim."""

import ast as python_ast
import hashlib
from pathlib import Path
from .contracts import canonical, digest, require
from .dsl import compile_ast


CALLBACK = '''
def initialize(context):
    log.info("FUXI V1 DIAGNOSTIC ONLY: no orders, no strategy-performance eligibility")

def local_replay():
    require(digest(FROZEN) == FROZEN_SHA256, "frozen diagnostic input mismatch")
    snapshot = Snapshot(FROZEN["snapshot"])
    require(snapshot.hash == FROZEN["snapshot_hash"], "frozen snapshot mismatch")
    return evaluate_ast(FROZEN["ast"], snapshot)

def provider_diagnostic(get_price_fn, session):
    """Re-fetch raw closes for frozen cells; paused/missing bars become null.

    Local declared availability is retained ONLY for a reproducibility diagnostic.
    Provider historical revision/PIT correctness is not certified by this adapter.
    """
    from copy import deepcopy
    require(digest(FROZEN) == FROZEN_SHA256, "frozen diagnostic input mismatch")
    require(session in FROZEN["expected_scores"], "session outside frozen evaluation scope")
    original = FROZEN["snapshot"]
    observed = deepcopy(original)
    stocks = sorted({row["instrument"] for row in original["observations"]})
    fetched = {}
    for stock in stocks:
        require(stock.endswith((".XSHE", ".XSHG")), "explicit JQ instrument mapping required")
        frame = get_price_fn(stock, start_date=original["sessions"][0], end_date=session,
                             frequency="daily", fields=["close", "paused"], skip_paused=False, fq=None)
        require(frame is not None and len(frame) > 0, "empty provider response")
        require({"close", "paused"} <= set(frame.columns), "provider capability mismatch")
        for index, row in frame.iterrows():
            day = str(index)[:10]
            require(day <= session, "provider returned future bar")
            key = (day, stock)
            require(key not in fetched, "duplicate provider bar")
            value = float(row["close"])
            paused = float(row["paused"])
            fetched[key] = value if paused == 0 and math.isfinite(value) else None
    price_pairs = [(row["value"], fetched.get((row["session"], row["instrument"])))
                   for row in original["observations"] if row["session"] <= session]
    paired_prices = [(a, b) for a, b in price_pairs if a is not None and b is not None]
    for row in observed["observations"]:
        row["value"] = fetched.get((row["session"], row["instrument"]))
    actual = evaluate_ast(FROZEN["ast"], Snapshot(observed))[session]
    expected = FROZEN["expected_scores"][session]
    pairs = [(expected[s], actual[s]) for s in expected if expected[s] is not None and actual[s] is not None]
    require(len(pairs) > 0, "diagnostic comparison has zero matched pairs")
    return {"session": session, "n_pairs": len(pairs), "universe_count": len(expected),
            "missing_pattern_equal": {s for s in expected if expected[s] is None} ==
                                     {s for s in actual if actual[s] is None},
            "max_abs_score_difference": max(abs(a-b) for a, b in pairs),
            "raw_price_pair_count": len(paired_prices),
            "raw_price_missing_pattern_equal": all((a is None) == (b is None) for a, b in price_pairs),
            "max_abs_raw_price_difference": max((abs(a-b) for a, b in paired_prices), default=None),
            "actual_scores": actual, "provider_parity_accepted": False,
            "historical_pit_certified": False, "formal_eligibility": False}

def before_trading_start(context):
    session = str(context.previous_date)[:10]
    if session in FROZEN["expected_scores"]:
        result = provider_diagnostic(get_price, session)
        log.info(json.dumps(result, ensure_ascii=False, sort_keys=True))
'''


def compile_diagnostic(candidate, snapshot, scores):
    compiled = compile_ast(candidate["canonical_ast"], snapshot.payload["fields"])
    require(compiled["formula_hash"] == candidate["formula_hash"], "candidate formula mismatch")
    require(compiled["required_fields"] == ["close"], "JQ adapter currently supports close only")
    spec = snapshot.payload["fields"]["close"]
    require(spec["unit"] == "price" and spec["adjustment"] == "raw", "unsupported JQ price semantics")
    modules = []
    # Embed precisely the same pure kernel code; no hand-maintained second evaluator.
    for name in ("contracts.py", "data.py", "dsl.py"):
        source = (Path(__file__).parent / name).read_text(encoding="utf-8")
        modules.append("\n".join(line for line in source.splitlines() if not line.startswith("from .")))
    frozen = dict(snapshot=snapshot.payload, snapshot_hash=snapshot.hash,
                  candidate_id=candidate["candidate_id"], ast=candidate["canonical_ast"],
                  expected_scores=scores, exposure=snapshot.payload["exposure"],
                  purpose="jq_diagnostic", formal_eligibility=False)
    text = "# Generated Fuxi v1 read-only diagnostic; never calls order APIs.\n"
    text += "\n\n".join(modules) + "\nFROZEN = json.loads(" + repr(canonical(frozen).decode("utf-8")) + ")\n"
    text += "FROZEN_SHA256 = " + repr(digest(frozen)) + "\n"
    text += CALLBACK
    tree = python_ast.parse(text)
    forbidden = {"order", "order_target", "order_value", "order_target_value", "order_target_percent"}
    require(not any(isinstance(n, python_ast.Call) and isinstance(n.func, python_ast.Name)
                    and n.func.id in forbidden for n in python_ast.walk(tree)), "order call in diagnostic")
    return text.encode("utf-8"), dict(schema_version="fuxi.jq_capability.1",
                                     purpose="jq_diagnostic", supported_fields=["close"],
                                     compiled=True, observed_on_jq=False, parity_accepted=False,
                                     script_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                     frozen_input_hash=digest(frozen))
