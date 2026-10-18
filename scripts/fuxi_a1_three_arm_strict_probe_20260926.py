"""Re-run frozen A1 targets with adjudicated events through the strict ledger.

The source actions use a declared gross-cash stress convention because the
investor account tax treatment is not frozen. Every output remains non-formal.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from portfolio_execution_ledger import (CorporateAction, LotRule,
                                        assess_economic_book, run_ledger)

OUT = ROOT / "reports/fuxi_a1_source_audit_20260926"
ADJUDICATION = OUT / "adjudication_v2.json"
START, END = "2025-01-01", "2026-04-30"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quoted_session_state(key, full_halts, partial_halts):
    if key in full_halts:
        return "unknown"
    if key in partial_halts:
        return "partial_halt"
    return "trading"


def targets_for(arm):
    path = (ROOT / "reports/fuxi_a1_20260926/baseline_targets.json"
            if arm == "amount_top80" else
            ROOT / ("reports/fuxi_quality_gate_20260925/targets_" + arm + ".json"))
    payload = load(path)
    return path, payload["targets"]


def load_bars(codes):
    import numpy as np
    import pandas as pd

    def small_panel(name, fields):
        parts = []
        for piece in pd.read_csv(ROOT / "data/raw" / name,
                                 usecols=["ts_code", "trade_date", *fields],
                                 dtype={"trade_date": str}, chunksize=500_000):
            piece = piece[piece.ts_code.isin(codes)]
            if not piece.empty:
                piece = piece.copy()
                piece["date"] = pd.to_datetime(piece.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
                parts.append(piece[piece.date.between("2024-12-01", END)])
        frame = pd.concat(parts, ignore_index=True)
        if frame.duplicated(["date", "ts_code"]).any():
            raise RuntimeError("duplicate source keys: " + name)
        return frame

    prices = small_panel("daily_prices.csv", ["open", "close", "pre_close"])
    quote = {(r.date, r.ts_code): {"open": r.open, "close": r.close,
                                   "pre_close": r.pre_close}
             for r in prices.itertuples() if START <= r.date <= END}
    days = sorted(prices[prices.date.between(START, END)].date.unique().tolist())
    if len(days) != 320 or days[-1] != END:
        raise RuntimeError("price calendar not the frozen 320 sessions")
    del prices
    limits = small_panel("stk_limit.csv", ["up_limit", "down_limit"])
    boundaries = {(r.date, r.ts_code): {"up_limit": r.up_limit,
                                        "down_limit": r.down_limit}
                  for r in limits.itertuples() if START <= r.date <= END}
    del limits
    factors = small_panel("adj_factor_daily.csv", ["adj_factor"])
    factors = factors.sort_values(["ts_code", "date"])
    factors["previous"] = factors.groupby("ts_code").adj_factor.shift()
    checks = {(r.date, r.ts_code):
              (bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0),
               bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0
                    and not np.isclose(r.previous, r.adj_factor, rtol=1e-7, atol=0)))
              for r in factors.itertuples() if START <= r.date <= END}
    del factors
    susp = pd.read_csv(ROOT / "data/raw/suspend_d.csv", dtype=str).fillna("")
    susp = susp[susp.ts_code.isin(codes)].copy()
    susp["date"] = pd.to_datetime(susp.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    full = {(r.date, r.ts_code) for r in susp.itertuples()
            if r.suspend_type == "S" and not r.suspend_timing}
    partial = {(r.date, r.ts_code) for r in susp.itertuples()
               if r.suspend_type == "S" and r.suspend_timing}

    class Bars:
        def get(self, key, default=None):
            available, required = checks.get(key, (False, False))
            if key in quote:
                return {**quote[key], **boundaries.get(key, {}),
                        "state": quoted_session_state(key, full, partial),
                        "action_check_available": available, "action_required": required}
            if key in full:
                return {"state": "halted", "evidence": "suspend_d.csv/full-day-S",
                        "action_check_available": available, "action_required": required}
            return default if default is not None else {"state": "unknown"}

    return days, Bars(), {"quote_keys": len(quote), "limit_keys": len(boundaries),
                          "adjustment_checks": len(checks), "full_halt_keys": len(full)}


def replay(arm, *, extension=None):
    import math

    output = OUT / ("strict_probe_" + arm +
                    ("_extension1" if extension else "") + ".json")
    if output.exists():
        raise RuntimeError("sealed probe exists: " + str(output))
    evidence = load(ADJUDICATION)
    if evidence["counts"] != {"approved_for_gross_cash_stress_only": 229}:
        raise RuntimeError("frozen action adjudication is incomplete")
    target_path, targets = targets_for(arm)
    codes = {c for weights in targets.values() if weights for c in weights}
    days, bars, coverage = load_bars(codes)
    actions = [CorporateAction(**r["action"]) for r in evidence["events"]
               if r["action"] is not None and r["code"] in codes
               and r["action"]["record_date"] in days and r["ex_date"] in days]
    if extension:
        if arm != "amount_top80":
            raise RuntimeError("actual-holdings extension is baseline-only")
        item = load(extension)
        if item["status"] != "approved_for_gross_cash_stress_only":
            raise RuntimeError("extension is not adjudicated")
        actions.append(CorporateAction(**item["action"]))
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    ledger = run_ledger(days, bars, targets, initial_cash=10_000_000,
                        cost_bps=15, rules=rules, actions=actions)
    assessment = assess_economic_book(ledger, days, 10_000_000)
    finite = [row for row in ledger.days if math.isfinite(row["nav"])]
    result = {
        "schema": "fuxi_a1_strict_three_arm_probe/v1", "arm": arm,
        "window": [START, END], "signal_dates": len(targets),
        "selected_unique_codes": len(codes), "adjudicated_action_keys_supplied": len(actions),
        "source_coverage": coverage, "expected_days": len(days),
        "processed_days": len(ledger.days), "finite_nav_days": len(finite),
        "finite_end": finite[-1]["date"] if finite else None,
        "status": assessment["status"], "assessment": assessment,
        "ledger_status": ledger.status, "errors": ledger.errors[:30],
        "excluded": len(ledger.excluded),
        "filled_orders": sum(o.get("filled", 0) > 0 for o in ledger.orders),
        "order_reasons": dict(Counter(o.get("reason") for o in ledger.orders if o.get("reason"))),
        "daily_nav": [{"date": r["date"], "nav": r["nav"]} for r in finite],
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            ADJUDICATION, target_path, ROOT / "data/raw/daily_prices.csv",
            ROOT / "data/raw/stk_limit.csv", ROOT / "data/raw/adj_factor_daily.csv",
            ROOT / "data/raw/suspend_d.csv",
            ROOT / "research/factor_alchemy/portfolio_execution_ledger.py")},
        "cash_assumption": "gross dividend paid; deferred account tax unmodeled",
        "actual_holdings_extension": str(extension.relative_to(ROOT)) if extension else None,
        "economic_status": "data_insufficient", "formal_eligibility": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("arm", "processed_days",
         "finite_nav_days", "finite_end", "status", "errors")}, ensure_ascii=False), flush=True)
    del ledger, bars
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("v2", "v3", "amount_top80"))
    replay(parser.parse_args().arm)
