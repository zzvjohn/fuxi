"""Trace possible held-event demand after A1's first strict stop.

This is a diagnostic sensitivity run. It deliberately uses the isolated
ledger's fractional floor, single-security exclusion, missing-limit defer,
and full-halt vacuity options. None of its NAVs are economic labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio_execution_ledger import (CorporateAction, LotRule,
                                        assess_economic_book, run_ledger)
from fuxi_a1_three_arm_strict_probe_20260926 import (
    ADJUDICATION, END, OUT, START, load, load_bars, sha, targets_for,
)
from run_contract import FuxiWriterLock

DELIST_TRACE_FACTS = {
    "300208.SZ": {
        "date": "2025-07-21",
        "evidence": "https://static.cninfo.com.cn/finalpage/2025-07-21/1224215557.PDF",
        "event": "delisted; prior session was last trade",
    },
    "300391.SZ": {
        "date": "2026-04-13",
        "evidence": "https://disc.static.szse.cn/disc/disk03/finalpage/2026-04-13/58b8875f-ce5e-40ee-a271-6ef6effedf67.PDF",
        "event": "delisted; held position's actual later value remains unknown",
    },
    "600190.SH": {
        "date": "2025-07-25",
        "evidence": "https://www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250718_10785880.shtml",
        "event": "delisted; held position's actual later value remains unknown",
    },
    "601989.SH": {
        "date": "2025-09-05",
        "evidence": "https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250829_10790128.shtml",
        "event": "merger share exchange; old shares convert into 600150, not a cash zero",
    },
}


def trace(arm: str) -> dict:
    target_path, targets = targets_for(arm)
    codes = {c for weights in targets.values() if weights for c in weights}
    days, bars, bar_coverage = load_bars(codes)
    evidence = load(ADJUDICATION)
    if evidence["counts"] != {"approved_for_gross_cash_stress_only": 229}:
        raise RuntimeError("unexpected frozen A1 adjudication")
    actions = [CorporateAction(**item["action"]) for item in evidence["events"]
               if item["action"] is not None and item["code"] in codes
               and item["action"]["record_date"] in days and item["ex_date"] in days]
    extension_path = OUT / "actual_holdings_extension_v1.json"
    if arm == "amount_top80":
        extension = load(extension_path)
        if extension["status"] != "approved_for_gross_cash_stress_only":
            raise RuntimeError("unapproved extension")
        actions.append(CorporateAction(**extension["action"]))
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    delist_trace = {code: fact for code, fact in DELIST_TRACE_FACTS.items()
                    if code in codes}
    ledger = run_ledger(
        days, bars, targets, initial_cash=10_000_000, cost_bps=15,
        rules=rules, actions=actions, fractional_share_policy="floor",
        unresolved_action_policy="exclude_held_security_day",
        missing_limit_policy="defer", halted_session_check_policy="vacuous",
        delist_evidence=delist_trace,
    )
    book_assessment = assess_economic_book(ledger, days, 10_000_000)
    row_by_day = {row["date"]: row for row in ledger.days}
    held_exclusions = []
    for entry in ledger.excluded:
        row = row_by_day[entry["date"]]
        held_exclusions.append({"date": entry["date"], "code": entry["code"],
                                "shares_under_sensitivity": row["positions"].get(
                                    entry["code"], {}).get("shares", 0),
                                "action_check": bars.get((entry["date"], entry["code"]), {})
                                .get("action_required")})
    fractional = [{k: entry.get(k) for k in (
        "date", "code", "event_id", "theoretical_extra", "granted_extra",
        "retired_fraction")}
                  for entry in ledger.journal if entry.get("kind") == "fractional_retired"]
    blocking_positions = []
    if ledger.errors and len(ledger.days) >= 2:
        prior = ledger.days[-2]
        for code in sorted({error["code"] for error in ledger.errors
                            if error["date"] == ledger.days[-1]["date"]}):
            position = prior["positions"].get(code, {})
            blocking_positions.append({
                "code": code, "last_finite_day": prior["date"],
                "shares_under_sensitivity": position.get("shares", 0),
                "last_mark": position.get("mark"),
                "last_marked_value": (position.get("shares", 0) * position.get("mark", 0)
                                      if position.get("mark") is not None else None),
                "last_portfolio_nav": prior["nav"],
            })
    supplied = sorted({(a.code, a.ex_date) for a in actions})
    missing_keys = sorted({(r["code"], r["date"]) for r in held_exclusions
                           if r["shares_under_sensitivity"] > 0 and r["action_check"]})
    return {
        "arm": arm, "status": "diagnostic_sensitivity_only",
        "formal_economic_label": False,
        "window": [START, END], "expected_days": len(days),
        "processed_days": len(ledger.days), "ledger_status": ledger.status,
        "economic_status": "data_insufficient",
        "book_assessment": book_assessment,
        "error_counts": dict(Counter(e["reason"] for e in ledger.errors)),
        "first_errors": ledger.errors[:20],
        "blocking_positions": blocking_positions,
        "supplied_action_keys": len(supplied),
        "held_missing_action_keys": [{"code": c, "ex_date": d} for c, d in missing_keys],
        "held_exclusions": held_exclusions,
        "fractional_retirements": fractional,
        "documented_lifecycle_facts_used": delist_trace,
        "bar_coverage": bar_coverage,
        "policies": {
            "fractional_share_policy": "floor",
            "unresolved_action_policy": "exclude_held_security_day",
            "missing_limit_policy": "defer",
            "halted_session_check_policy": "vacuous",
            "cash": "gross dividend; investor tax unmodeled",
        },
        "limitations": [
            "The floor rule is unverified for each account and can change later holdings.",
            "Excluded positions carry stale marks and lack entitlements; NAV is invalid.",
            "The resulting event demand is a sensitivity trace, not the strict realized book.",
            "Delisted/merged positions are stale-frozen solely to continue demand tracing.",
        ],
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in (
                ADJUDICATION, target_path, ROOT / "data/raw/daily_prices.csv",
                ROOT / "data/raw/adj_factor_daily.csv",
                ROOT / "data/raw/stk_limit.csv", ROOT / "data/raw/suspend_d.csv",
                ROOT / "research/factor_alchemy/portfolio_execution_ledger.py",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("v2", "v3", "amount_top80"))
    args = parser.parse_args()
    output = OUT / ("held_event_trace_" + args.arm + "_v5.json")
    with FuxiWriterLock(run_id="a1_held_event_trace_" + args.arm):
        if output.exists():
            raise RuntimeError("trace already sealed: " + str(output))
        payload = trace(args.arm)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    print(json.dumps({k: payload[k] for k in (
        "arm", "processed_days", "ledger_status", "error_counts")},
        ensure_ascii=False), flush=True)
    print("held_missing_action_keys=", len(payload["held_missing_action_keys"]),
          "fractional_retirements=", len(payload["fractional_retirements"]))


if __name__ == "__main__":
    main()
