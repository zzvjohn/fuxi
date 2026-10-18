"""Causal offline mechanism replay; incomplete counterfactuals fail closed."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from mechanism_search import load_catalog, load_evidence, select_plan  # noqa: E402

CATALOG = ROOT / "research/factor_alchemy/mechanism_catalog_20260925.json"
EVIDENCE = ROOT / "reports/fuxi_mechanism_quantum_20260925/economic_evidence.jsonl"
OUT = ROOT / "reports/fuxi_a4_20260926"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iso(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("replay timestamps require a timezone")
    return result


def replay_complete_matrix(plans, rounds, *, seed):
    """Compare policies only on complete, time-stamped per-round action matrices.

    Each row has decision_at and outcomes for every plan, each outcome with a
    mature_at strictly after that decision. Previous selected rewards become
    visible to each policy only once mature_at <= its next decision time.
    The caller must verify result provenance before supplying a real matrix.
    """
    plan_ids = {p["plan_id"] for p in plans}
    if not plan_ids or len(plan_ids) != len(plans) or not rounds:
        raise ValueError("nonempty unique plans and rounds required")
    last = None
    selected = {"selector": [], "uniform": []}
    result = []
    for index, row in enumerate(rounds):
        at = _iso(row["decision_at"])
        if last is not None and at <= last:
            raise ValueError("decision timestamps must increase")
        last = at
        outcomes = row.get("outcomes", {})
        if set(outcomes) != plan_ids:
            raise ValueError("complete outcome matrix required")
        for pid, outcome in outcomes.items():
            if (outcome.get("status") != "valid" or
                    outcome.get("verified_trading_label") is not True or
                    isinstance(outcome.get("economic_lcb_bps"), bool) or
                    not isinstance(outcome.get("economic_lcb_bps"), (int, float))):
                raise ValueError("only verified finite economic outcomes support policy comparison")
            score = float(outcome["economic_lcb_bps"])
            if not math.isfinite(score) or _iso(outcome["mature_at"]) <= at:
                raise ValueError("invalid or prematurely mature economic outcome")
        visible = [{"plan_id": event["plan_id"], "status": "valid",
                    "verified_trading_label": True,
                    "economic_lcb_bps": event["economic_lcb_bps"]}
                   for event in selected["selector"] if _iso(event["mature_at"]) <= at]
        choice, audit = select_plan(plans, visible, seed=seed, round_index=index)
        baseline = random.Random(seed + 1_000_003 * index).choice(sorted(plan_ids))
        for label, pid in (("selector", choice["plan_id"]), ("uniform", baseline)):
            event = {"plan_id": pid, **outcomes[pid]}
            selected[label].append(event)
        result.append({"round": index, "decision_at": row["decision_at"],
                       "visible_valid_before_decision": len(visible),
                       "selector_plan": choice["plan_id"],
                       "selector_mode": audit["mode"], "uniform_plan": baseline,
                       "selector_lcb_bps": float(outcomes[choice["plan_id"]]["economic_lcb_bps"]),
                       "uniform_lcb_bps": float(outcomes[baseline]["economic_lcb_bps"])})
    return {"status": "paired_complete_matrix_diagnostic", "rounds": result,
            "paired_rounds": len(result), "promotes_plan": False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "mechanism_replay.json"
    if output.exists():
        raise RuntimeError("immutable replay result exists")
    plans = load_catalog(CATALOG, "daily")
    observed = load_evidence(EVIDENCE, ROOT, plans)
    valid = [row for row in observed if row["status"] == "valid"]
    result = {
        "schema": "fuxi_a4_mechanism_replay/v1",
        "status": "not_identifiable_without_complete_matured_matrix",
        "catalog_sha256": sha(CATALOG), "evidence_sha256": sha(EVIDENCE),
        "source_result_sha256": {row["observation_id"]: row["source_sha256"] for row in observed},
        "plans": [p["plan_id"] for p in plans],
        "observations": len(observed), "valid_economic_observations": len(valid),
        "timed_matured_observations": sum(bool(row.get("mature_at")) for row in observed),
        "complete_counterfactual_rounds": 0,
        "selector_vs_uniform_paired_rounds": 0,
        "comparison_count_assertion": "no comparison attempted; 0 is not an advantage estimate",
        "reason": "two data-insufficient observations have no verified economic reward, maturity timestamp, or all-action outcomes",
        "economic_verdict": "not_evaluated", "formal_eligibility": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("status", "observations",
        "valid_economic_observations", "selector_vs_uniform_paired_rounds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
