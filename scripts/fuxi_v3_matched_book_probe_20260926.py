"""Strict account eligibility probe for the frozen v3 matched subset and control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio_execution_ledger import (  # noqa: E402
    CorporateAction, LotRule, assess_economic_book, run_ledger,
)
from fuxi_a1_three_arm_strict_probe_20260926 import (  # noqa: E402
    ADJUDICATION, OUT as A1_OUT, load, load_bars,
)
from fuxi_a1_actual_holdings_trace_20260926 import DELIST_TRACE_FACTS  # noqa: E402
from fuxi_a1_model_account_replay_20260926 import EXCHANGE, independent_calculation  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402

OUT = ROOT / "reports/fuxi_v3_matched_book_20260926"
MATCH = ROOT / "reports/fuxi_v3_matched_control_20260926_v3/matched_targets.json"
SOURCE = (
    MATCH, ADJUDICATION,
    A1_OUT / "held_extension_audit_v5.json",
    A1_OUT / "actual_holdings_extension_v1.json",
    ROOT / "data/raw/daily_prices.csv",
    ROOT / "data/raw/stk_limit.csv",
    ROOT / "data/raw/adj_factor_daily.csv",
    ROOT / "data/raw/suspend_d.csv",
    ROOT / "research/factor_alchemy/portfolio_execution_ledger.py",
    Path(__file__).resolve(),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    with FuxiWriterLock(run_id="v3_matched_book_prepare"):
        OUT.mkdir(exist_ok=True)
        protocol_path = OUT / "protocol.json"
        if protocol_path.exists():
            raise RuntimeError("matched book protocol already frozen")
        match = load(MATCH)
        if match["status"] != "matched_complete" or match["matched_dates"] != 69:
            raise RuntimeError("matched set incomplete")
        protocol = {
            "schema": "fuxi_v3_matched_book_protocol/v1",
            "arms": ["trimmed_v3", "matched_control"],
            "window": ["2025-01-02", "2026-04-30"], "expected_days": 320,
            "initial_cash": 10_000_000, "cost_bps_each_side": 15,
            "signal_cutoff": "post_close", "earliest_fill": "next_session_open_proxy",
            "reference_scenario": "floor_tax20_delist0",
            "same_execution_model": "frozen A1 ledger; missing limit defer, halted check vacuous, issuer actions only from sealed adjudication/held extension; no hypothetical action fill",
            "economic_success": "both arms complete 320-day book, zero errors/exclusions, independent same-input arithmetic replay, paired count>0",
            "if_failure": "return data_insufficient and blockers; no NAV difference may be called cost-after advantage",
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCE},
        }
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("protocol_sha256", sha(protocol_path))


def construct(arm, targets):
    codes = {code for basket in targets.values() if basket for code in basket}
    if "601989.SH" in codes:
        codes.add("600150.SH")
    days, bars, coverage = load_bars(codes)
    if len(days) != 320:
        raise RuntimeError("frozen calendar changed")
    planned = load(ADJUDICATION)
    held = load(A1_OUT / "held_extension_audit_v5.json")
    if planned["counts"] != {"approved_for_gross_cash_stress_only": 229} or held["union_keys"] != 24:
        raise RuntimeError("action seal changed")
    by_key = {}
    for item in planned["events"] + held["events"]:
        if item.get("status", "approved_for_gross_cash_stress_only") != "approved_for_gross_cash_stress_only":
            continue
        action = item.get("action")
        if not action or action["code"] not in codes:
            continue
        if action["record_date"] not in days or action["ex_date"] not in days:
            continue
        key = (action["code"], action["ex_date"])
        old = by_key.get(key)
        if old and (old.cash_per_share_net != action["cash_per_share_net"]
                    or old.share_multiplier != action["share_multiplier"]):
            raise RuntimeError("conflicting action key: " + str(key))
        by_key[key] = CorporateAction(**action)
    extra = load(A1_OUT / "actual_holdings_extension_v1.json")
    if extra["action"]["code"] in codes:
        action = CorporateAction(**extra["action"])
        by_key[(action.code, action.ex_date)] = action
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    delisted = {c: fact for c, fact in DELIST_TRACE_FACTS.items() if c in codes}
    actions = list(by_key.values())
    exchanges = [EXCHANGE] if "601989.SH" in codes else []
    ledger = run_ledger(
        days, bars, targets, initial_cash=10_000_000, cost_bps=15,
        rules=rules, actions=actions, fractional_share_policy="floor",
        gross_cash_tax_rate=0.2, delist_mark_fraction=0,
        reference_only_events={("002822.SZ", "2025-12-30")},
        share_exchanges=exchanges, missing_limit_policy="defer",
        halted_session_check_policy="vacuous", delist_evidence=delisted,
    )
    assessment = assess_economic_book(ledger, days, 10_000_000)
    complete = (assessment["status"] == "book_complete_under_declared_assumptions"
                and len(ledger.days) == 320 and not ledger.errors and not ledger.excluded
                and all(math.isfinite(row["nav"]) for row in ledger.days))
    independent = None
    if complete:
        independent = independent_calculation(
            ledger, bars, actions, delisted,
            fractional_policy="floor", tax_rate=0.2, delist_mark_fraction=0,
            share_exchanges=exchanges)
    return {
        "arm": arm, "status": "model_complete" if complete else "data_insufficient",
        "economic_status": "model_complete_under_declared_assumptions" if complete else "data_insufficient",
        "formal_eligibility": False,
        "assessment": assessment, "expected_days": 320,
        "processed_days": len(ledger.days), "finite_nav_days": sum(math.isfinite(r["nav"]) for r in ledger.days),
        "first_errors": ledger.errors[:40],
        "error_counts": dict(Counter(e["reason"] for e in ledger.errors)),
        "excluded_count": len(ledger.excluded),
        "action_keys_supplied": len(by_key), "quote_coverage": coverage,
        "filled_orders": sum(o.get("filled", 0) > 0 for o in ledger.orders),
        "end_nav": ledger.days[-1]["nav"] if complete else None,
        "daily_nav": [{"date": r["date"], "nav": r["nav"]} for r in ledger.days] if complete else [],
        "independent_calculation": {k: v for k, v in independent.items() if k != "daily_parity"}
        if independent else None,
    }


def run():
    with FuxiWriterLock(run_id="v3_matched_book_run"):
        protocol_path = OUT / "protocol.json"
        output_path = OUT / "probe.json"
        if output_path.exists():
            raise RuntimeError("matched book probe already frozen")
        protocol = load(protocol_path)
        if protocol["schema"] != "fuxi_v3_matched_book_protocol/v1":
            raise RuntimeError("unknown protocol")
        for p in SOURCE:
            if sha(p) != protocol["input_sha256"][str(p.relative_to(ROOT))]:
                raise RuntimeError("frozen source changed: " + str(p))
        match = load(MATCH)
        results = {}
        for arm, key in (("trimmed_v3", "trimmed_v3_targets"),
                         ("matched_control", "control_targets")):
            results[arm] = construct(arm, match[key])
            print(json.dumps({k: results[arm][k] for k in (
                "arm", "status", "processed_days", "error_counts")}, ensure_ascii=False), flush=True)
        valid_pairs = (sum(a["date"] == b["date"] for a, b in zip(
            results["trimmed_v3"]["daily_nav"], results["matched_control"]["daily_nav"]))
                       if all(x["status"] == "model_complete" for x in results.values()) else 0)
        result = {"schema": "fuxi_v3_matched_book_probe/v1",
                  "protocol_sha256": sha(protocol_path), "arms": results,
                  "paired_days": valid_pairs,
                  "economic_status": "conditional_exploratory_only" if valid_pairs == 320
                  else "data_insufficient",
                  "matched_return_gap": (
                      (results["trimmed_v3"]["end_nav"] - results["matched_control"]["end_nav"])
                      / 10_000_000 if valid_pairs == 320 else None),
                  "formal_eligibility": False}
        for p in SOURCE:
            if sha(p) != protocol["input_sha256"][str(p.relative_to(ROOT))]:
                raise RuntimeError("source changed during probe: " + str(p))
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("paired_days", "economic_status",
                                                "matched_return_gap")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else run()
