"""Run only the frozen simple baseline through the strict isolated cash ledger."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from portfolio_execution_ledger import (  # noqa: E402
    CorporateAction, LotRule, assess_economic_book, run_ledger,
)

OUT = ROOT / "reports/fuxi_a1_20260926"
START, END = "2025-01-01", "2026-04-30"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    import pandas as pd

    output = OUT / "baseline_probe.json"
    if output.exists():
        raise RuntimeError("baseline probe already frozen")
    target_file = OUT / "baseline_targets.json"
    target_payload = json.loads(target_file.read_text(encoding="utf-8"))
    if sha(ROOT / "data/raw/daily_prices.csv") != target_payload["source_sha256"]:
        raise RuntimeError("baseline source changed")
    targets = target_payload["targets"]
    codes = {code for weights in targets.values() for code in weights}
    panels = {}
    for name, fields in (
        ("daily_prices.csv", ["open", "close", "pre_close"]),
        ("stk_limit.csv", ["up_limit", "down_limit"]),
        ("adj_factor_daily.csv", ["adj_factor"]),
    ):
        chunks = []
        for part in pd.read_csv(ROOT / "data/raw" / name,
                                usecols=["ts_code", "trade_date", *fields],
                                dtype={"trade_date": str}, chunksize=500_000):
            part = part[part.ts_code.isin(codes)]
            if not part.empty:
                chunks.append(part)
        frame = pd.concat(chunks, ignore_index=True)
        frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
        if frame.duplicated(["date", "ts_code"]).any():
            raise RuntimeError("duplicate source keys: " + name)
        if name == "adj_factor_daily.csv":
            frame = frame.sort_values(["ts_code", "date"])
            frame["previous"] = frame.groupby("ts_code").adj_factor.shift()
        panels[name] = frame[frame.date.between(START, END)]
    price = panels["daily_prices.csv"]
    days = sorted(price.date.unique().tolist())
    if not days or days[-1] != END:
        raise RuntimeError("price window not complete to requested end")
    quote = {(r.date, r.ts_code): {"open": r.open, "close": r.close,
                                   "pre_close": r.pre_close}
             for r in price.itertuples()}
    limits = {(r.date, r.ts_code): {"up_limit": r.up_limit,
                                    "down_limit": r.down_limit}
              for r in panels["stk_limit.csv"].itertuples()}
    adjust = {(r.date, r.ts_code): (
        bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0),
        bool(pd.notna(r.previous) and r.previous > 0 and r.adj_factor > 0
             and not np.isclose(r.previous, r.adj_factor, rtol=1e-7, atol=0)))
              for r in panels["adj_factor_daily.csv"].itertuples()}
    suspension = pd.read_csv(ROOT / "data/raw/suspend_d.csv", dtype=str).fillna("")
    suspension["date"] = pd.to_datetime(suspension.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    full_halt = {(r.date, r.ts_code) for r in suspension.itertuples()
                 if r.ts_code in codes and r.suspend_type == "S" and not r.suspend_timing}
    partial_halt = {(r.date, r.ts_code) for r in suspension.itertuples()
                    if r.ts_code in codes and r.suspend_type == "S" and r.suspend_timing}

    class Bars:
        def get(self, key, default=None):
            available, required = adjust.get(key, (False, False))
            if key in quote:
                return {**quote[key], **limits.get(key, {}),
                        "state": "unknown" if key in full_halt or key in partial_halt else "trading",
                        "action_check_available": available, "action_required": required}
            if key in full_halt:
                return {"state": "halted", "evidence": "suspend_d.csv/full-day-S",
                        "action_check_available": available, "action_required": required}
            return default if default is not None else {"state": "unknown"}

    events = []
    for path, key in (
        (ROOT / "reports/execution_events_20260923/approved_actions.json", "actions"),
        (ROOT / "reports/execution_followup_20260923/validation.json", "new_actions"),
        (ROOT / "reports/fuxi_quality_gate_20260925/event_920718/validation.json", "action"),
        (ROOT / "reports/fuxi_quality_gate_20260925/event_920405/validation.json", "action"),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        events.extend([data[key]] if key == "action" else data[key])
    actions = [CorporateAction(**event) for event in events
               if event["code"] in codes and event["record_date"] in days
               and event["ex_date"] in days]
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    ledger = run_ledger(days, Bars(), targets, initial_cash=10_000_000,
                        cost_bps=15, rules=rules, actions=actions)
    assessment = assess_economic_book(ledger, days, 10_000_000)
    finite = [r for r in ledger.days if np.isfinite(r["nav"])]
    result = {
        "schema": "fuxi_a1_baseline_probe/v1", "arm": "amount_top80",
        "window": [START, END], "book_kind": "fixed_shares_cash",
        "baseline_targets_sha256": sha(target_file),
        "ledger_code_sha256": sha(ROOT / "research/factor_alchemy/portfolio_execution_ledger.py"),
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": {name: sha(ROOT / "data/raw" / name) for name in
                          ("daily_prices.csv", "stk_limit.csv", "adj_factor_daily.csv", "suspend_d.csv")},
        "cost_bps": 15, "expected_days": len(days), "processed_days": len(ledger.days),
        "finite_nav_days": len(finite), "finite_end": finite[-1]["date"] if finite else None,
        "status": assessment["status"], "economic_verdict": "data_insufficient"
            if assessment["status"] == "data_insufficient" else "not_evaluated",
        "assessment": assessment, "errors": ledger.errors[:30],
        "filled_orders": sum(o.get("filled", 0) > 0 for o in ledger.orders),
        "formal_eligibility": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in (
        "status", "expected_days", "processed_days", "finite_nav_days", "finite_end", "errors")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
