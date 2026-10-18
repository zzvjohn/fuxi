"""Read-only-inference sensitivity trace for missing rights in matched control.

The traced NAV is deliberately discarded. Excluded holdings and carried marks
cannot become economic labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger  # noqa: E402
from fuxi_a1_three_arm_strict_probe_20260926 import (  # noqa: E402
    ADJUDICATION, OUT as A1_OUT, load, load_bars,
)
from fuxi_a1_actual_holdings_trace_20260926 import DELIST_TRACE_FACTS  # noqa: E402
from fuxi_a1_model_account_replay_20260926 import EXCHANGE  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402

OUT = ROOT / "reports/fuxi_v3_matched_book_20260926/event_demand.json"
MATCH = ROOT / "reports/fuxi_v3_matched_control_20260926_v3/matched_targets.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    with FuxiWriterLock(run_id="v3_matched_event_demand"):
        if OUT.exists():
            raise RuntimeError("event demand already frozen")
        target = load(MATCH)["control_targets"]
        codes = {c for basket in target.values() for c in basket}
        if "601989.SH" in codes:
            codes.add("600150.SH")
        days, bars, coverage = load_bars(codes)
        planned = load(ADJUDICATION)
        held = load(A1_OUT / "held_extension_audit_v5.json")
        by_key = {}
        for item in planned["events"] + held["events"]:
            if item.get("status", "approved_for_gross_cash_stress_only") != "approved_for_gross_cash_stress_only":
                continue
            action = item.get("action")
            if action and action["code"] in codes and action["record_date"] in days and action["ex_date"] in days:
                by_key[(action["code"], action["ex_date"])] = CorporateAction(**action)
        extra = load(A1_OUT / "actual_holdings_extension_v1.json")["action"]
        if extra["code"] in codes:
            by_key[(extra["code"], extra["ex_date"])] = CorporateAction(**extra)
        rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
                 LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
        ledger = run_ledger(
            days, bars, target, initial_cash=10_000_000, cost_bps=15,
            rules=rules, actions=list(by_key.values()), fractional_share_policy="floor",
            gross_cash_tax_rate=0.2, delist_mark_fraction=0,
            reference_only_events={("002822.SZ", "2025-12-30")},
            share_exchanges=[EXCHANGE] if "601989.SH" in codes else [],
            unresolved_action_policy="exclude_held_security_day",
            missing_limit_policy="defer", halted_session_check_policy="vacuous",
            delist_evidence={c: f for c, f in DELIST_TRACE_FACTS.items() if c in codes},
        )
        by_day = {r["date"]: r for r in ledger.days}
        held_missing = sorted({(item["code"], item["date"])
                               for item in ledger.excluded
                               if by_day[item["date"]]["positions"].get(item["code"], {}).get("shares", 0) > 0
                               and bars.get((item["date"], item["code"]), {}).get("action_required")})
        result = {
            "schema": "fuxi_v3_matched_event_demand/v1",
            "status": "diagnostic_sensitivity_only", "economic_status": "data_insufficient",
            "formal_eligibility": False,
            "processed_days": len(ledger.days), "expected_days": len(days),
            "error_counts": dict(Counter(e["reason"] for e in ledger.errors)),
            "first_errors": ledger.errors[:30],
            "held_missing_action_keys": [{"code": c, "ex_date": d} for c, d in held_missing],
            "held_exclusions": [{"date": e["date"], "code": e["code"],
                                 "action_required": bars.get((e["date"], e["code"]), {}).get("action_required")}
                                for e in ledger.excluded],
            "fractional_events": [{k: e.get(k) for k in
                                   ("date", "code", "event_id", "theoretical_extra", "granted_extra")}
                                  for e in ledger.journal if e.get("kind") in
                                  ("fractional_retired", "fractional_rounded_up")],
            "delisted_freezes": [{"date": e["date"], "code": e["code"]}
                                 for e in ledger.journal if e.get("kind") == "delisted_freeze"],
            "quote_coverage": coverage,
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                MATCH, ADJUDICATION, A1_OUT / "held_extension_audit_v5.json",
                ROOT / "data/raw/daily_prices.csv", ROOT / "data/raw/adj_factor_daily.csv",
                ROOT / "data/raw/stk_limit.csv", ROOT / "data/raw/suspend_d.csv",
                ROOT / "research/factor_alchemy/portfolio_execution_ledger.py",
                Path(__file__).resolve())},
            "not_reported": "NAV/returns are invalid under excluded held-security days and are intentionally omitted",
        }
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("processed_days", "expected_days",
                                                "error_counts", "held_missing_action_keys")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
