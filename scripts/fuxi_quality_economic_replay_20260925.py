"""Frozen, isolated next-open cash-ledger probe for one LLM factor.

No factor registration, production S5 adjudication, or data collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from formula_quality_precheck import assess_candidate
from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal

OUT = ROOT / "reports/fuxi_quality_gate_20260925"
SOURCE = ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget/result.json"
INPUTS = ["data/raw/daily_prices.csv", "data/raw/stk_limit.csv",
          "data/raw/adj_factor_daily.csv", "data/raw/suspend_d.csv",
          "data/raw/index_000985_daily.csv", "data/raw/trade_calendar.csv"]
START, END = "2025-01-01", "2026-04-30"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    if path.exists():
        raise RuntimeError("immutable evidence exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def prepare(revision=""):
    suffix = "_" + revision if revision else ""
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for arm, group in source["arms"].items():
        for index, candidate in enumerate(group["candidates"], 1):
            quality = assess_candidate(candidate)
            rows.append({"arm": arm, "index": index, "name": candidate["factor_name"],
                         "direction": candidate["direction"], **quality})
    accepted = [r for r in rows if r["status"] == "pass"]
    assert [(r["arm"], r["index"]) for r in accepted] == [("semantic_plan", 3)]
    chosen = accepted[0]
    protocol = {
        "schema": "fuxi_quality_economic_probe/v1",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "source_result_sha256": sha(SOURCE),
        "quality_code_sha256": sha(ROOT / "research/factor_alchemy/formula_quality_precheck.py"),
        "s5_code_sha256": sha(ROOT / "research/factor_alchemy/s5_joint_filter.py"),
        "ledger_code_sha256": sha(ROOT / "research/factor_alchemy/portfolio_execution_ledger.py"),
        "script_sha256": sha(Path(__file__)),
        "inputs": {name: sha(ROOT / name) for name in INPUTS},
        "window": {"start": START, "end": END, "research_exposed": True},
        "selection": "all static pass candidates; currently one semantic-plan candidate and zero baseline candidates",
        "chosen": chosen,
        "signal_cutoff": "post_close", "earliest_fill": "next_session_open_proxy",
        "portfolio": "Top80 long only, deterministic factor/amount/code tie order",
        "capital": 10_000_000, "cost_bps": 15,
        "corporate_events": "only prior separately evidenced events; unknown adjustment blocks",
        "promotion_eligible": False,
    }
    if revision:
        protocol["supersedes"] = "protocol.json"
        protocol["correction"] = ("Full-day halt evidence must be carried into the ledger; "
                                  "finite NAV requires isfinite, not merely non-None.")
    write_once(OUT / ("protocol" + suffix + ".json"), protocol)
    write_once(OUT / ("precheck" + suffix + ".json"), {"source_result_sha256": sha(SOURCE),
                                        "rows": rows, "accepted_n": len(accepted)})
    print(json.dumps({"accepted": [(r["arm"], r["index"]) for r in accepted],
                      "rejected": sum(r["status"] == "reject" for r in rows),
                      "review": sum(r["status"] == "review" for r in rows)}, ensure_ascii=False))


def evaluate(revision=""):
    import numpy as np
    import pandas as pd
    from s5_joint_filter import S5JointFilter

    suffix = "_" + revision if revision else ""
    protocol_file = OUT / ("protocol" + suffix + ".json")
    result_file = OUT / ("economic_probe" + suffix + ".json")
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if result_file.exists():
        raise RuntimeError("economic result already exists")
    assert sha(SOURCE) == protocol["source_result_sha256"]
    for path, key in (("research/factor_alchemy/formula_quality_precheck.py", "quality_code_sha256"),
                      ("research/factor_alchemy/s5_joint_filter.py", "s5_code_sha256"),
                      ("research/factor_alchemy/portfolio_execution_ledger.py", "ledger_code_sha256"),
                      ("scripts/fuxi_quality_economic_replay_20260925.py", "script_sha256")):
        assert sha(ROOT / path) == protocol[key], path
    for name, expected in protocol["inputs"].items():
        assert sha(ROOT / name) == expected, name
    choice = protocol["chosen"]
    assert choice["status"] == "pass" and choice["direction"] == "long"

    obj = S5JointFilter(cache_dir=OUT / "cache")
    assert obj.top_n == 80 and obj.sample_stocks is None
    obj._load_data()
    raw_price = obj._price_df
    signal = obj.evaluate_factor_formula(choice["canonical_formula"], "quality_probe_semantic_3")
    if signal is None or signal.empty:
        write_once(result_file, {"status": "evaluation_error",
                                                "reason": "empty_formula_signal"})
        return
    dates = pd.DatetimeIndex(raw_price.trade_date.drop_duplicates()).sort_values()
    dates = dates[(dates >= START) & (dates <= END)]
    days = [d.strftime("%Y-%m-%d") for d in dates]
    week_last = pd.Series(dates, index=dates.to_period("W-SUN")).groupby(level=0).max()
    signal_dates = {d.strftime("%Y-%m-%d") for d in week_last}
    selected_signal = signal[signal.trade_date.dt.strftime("%Y-%m-%d").isin(signal_dates)].copy()
    selected_signal = selected_signal.merge(
        raw_price[["trade_date", "ts_code", "close", "amount"]],
        on=["trade_date", "ts_code"], how="inner", validate="one_to_one")
    selected_signal = selected_signal[np.isfinite(selected_signal.factor_value)
                                      & (selected_signal.close > 0)]
    targets, coverage = {}, []
    grouped = {d.strftime("%Y-%m-%d"): g for d, g in selected_signal.groupby("trade_date")}
    for day in sorted(signal_dates):
        group = grouped.get(day)
        if group is None or len(group) < 80:
            targets[day] = None
            coverage.append({"date": day, "available": 0 if group is None else len(group)})
            continue
        ranked = group.sort_values(["factor_value", "amount", "ts_code"],
                                   ascending=[False, False, True], kind="mergesort")
        targets[day] = {code: 1 / 80 for code in ranked.head(80).ts_code}
    write_once(OUT / ("targets" + suffix + ".json"), {"targets": targets, "insufficient": coverage})
    selected_codes = {c for weights in targets.values() if weights for c in weights}
    del signal, selected_signal

    price = raw_price[raw_price.ts_code.isin(selected_codes)
                      & raw_price.trade_date.between(START, END)].copy()
    price["date"] = price.trade_date.dt.strftime("%Y-%m-%d")
    if price.duplicated(["date", "ts_code"]).any():
        raise RuntimeError("duplicate raw price keys")
    quote = {(r.date, r.ts_code): {"open": r.open, "close": r.close,
                                  "pre_close": r.pre_close}
             for r in price.itertuples()}

    def small_panel(name, columns):
        chunks = []
        for part in pd.read_csv(ROOT / name, usecols=["trade_date", "ts_code"] + columns,
                                dtype={"trade_date": str}, chunksize=300_000):
            part = part[part.ts_code.isin(selected_codes)].copy()
            part["date"] = pd.to_datetime(part.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
            chunks.append(part[part.date.between(START, END)])
        frame = pd.concat(chunks, ignore_index=True)
        if frame.duplicated(["date", "ts_code"]).any():
            raise RuntimeError("duplicate market keys: " + name)
        return frame

    limits = small_panel("data/raw/stk_limit.csv", ["up_limit", "down_limit"])
    boundaries = {(r.date, r.ts_code): {"up_limit": r.up_limit, "down_limit": r.down_limit}
                  for r in limits.itertuples()}
    adjust = small_panel("data/raw/adj_factor_daily.csv", ["adj_factor"])
    adjust = adjust.sort_values(["ts_code", "date"])
    adjust["previous"] = adjust.groupby("ts_code").adj_factor.shift()
    adjustments = {(r.date, r.ts_code): (bool(pd.notna(r.previous) and r.adj_factor > 0
                                             and r.previous > 0),
                                         bool(pd.notna(r.previous) and r.adj_factor > 0
                                              and r.previous > 0 and not np.isclose(
                                                  r.adj_factor, r.previous, rtol=1e-7, atol=0)))
                   for r in adjust.itertuples()}
    susp = pd.read_csv(ROOT / "data/raw/suspend_d.csv", dtype=str).fillna("")
    susp["date"] = pd.to_datetime(susp.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    full_halt = {(r.date, r.ts_code) for r in susp.itertuples()
                 if r.suspend_type == "S" and not r.suspend_timing}
    partial_halt = {(r.date, r.ts_code) for r in susp.itertuples()
                    if r.suspend_type == "S" and r.suspend_timing}

    class Bars:
        def get(self, key, default=None):
            if key in quote:
                check, required = adjustments.get(key, (False, False))
                return {**quote[key], **boundaries.get(key, {}),
                        "state": "unknown" if key in full_halt or key in partial_halt else "trading",
                        "action_check_available": check, "action_required": required}
            if key in full_halt:
                check, required = adjustments.get(key, (False, False))
                return {"state": "halted", "action_check_available": check,
                        "action_required": required,
                        "evidence": "suspend_d.csv/full-day-S"}
            return default if default is not None else {"state": "unknown"}

    known = []
    for path, key in ((ROOT / "reports/execution_events_20260923/approved_actions.json", "actions"),
                      (ROOT / "reports/execution_followup_20260923/validation.json", "new_actions")):
        known.extend(json.loads(path.read_text(encoding="utf-8"))[key])
    actions = [CorporateAction(**r) for r in known
               if r["record_date"] in days and r["ex_date"] in days]
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule()
             for c in selected_codes}
    ledger = run_ledger(days, Bars(), targets, initial_cash=protocol["capital"],
                        cost_bps=protocol["cost_bps"], rules=rules, actions=actions)
    parity = reconcile_journal(ledger, protocol["capital"])
    finite = [r for r in ledger.days if np.isfinite(r.get("nav", np.nan))]
    summary = {"status": ledger.status, "formal_eligibility": False,
               "source_protocol_sha256": sha(protocol_file),
               "candidate": {k: choice[k] for k in ("arm", "index", "name", "direction")},
               "window": protocol["window"], "signal_dates": len(targets),
               "insufficient_signal_dates": len(coverage),
               "selected_unique_codes": len(selected_codes),
               "known_actions_supplied": len(actions),
               "processed_days": len(ledger.days), "finite_nav_days": len(finite),
               "finite_end": finite[-1]["date"] if finite else None,
               "errors": ledger.errors[:20], "parity": parity,
               "filled_orders": sum(o.get("filled", 0) > 0 for o in ledger.orders),
               "return_status": "data_insufficient" if ledger.status != "complete_under_declared_assumptions"
                                else "isolated_execution_diagnostic_only",
               "economic_verdict": "not_evaluated" if len(finite) < 30 else "diagnostic_only",
               "cost_bps": protocol["cost_bps"],
               "limits": "next-open proxy; no auction/liquidity/tax model; missing corporate events fail closed"}
    write_once(result_file, summary)
    print(json.dumps({k: summary[k] for k in ("status", "processed_days", "finite_nav_days",
                                           "finite_end", "filled_orders", "return_status")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--revision", default="", choices=("", "v2"))
    args = parser.parse_args()
    prepare(args.revision) if args.action == "prepare" else evaluate(args.revision)
