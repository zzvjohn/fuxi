"""Read-only v3 source and point-in-time audit of sealed research books."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger  # noqa: E402
from fuxi_a1_three_arm_strict_probe_20260926 import (  # noqa: E402
    ADJUDICATION, OUT, load, load_bars, targets_for,
)
from fuxi_a1_actual_holdings_trace_20260926 import DELIST_TRACE_FACTS  # noqa: E402
from fuxi_a1_model_account_replay_20260926 import EXCHANGE  # noqa: E402

DEST = ROOT / "reports/fuxi_v3_source_audit_20260926"
SCENARIO = "floor_tax20_delist0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sealed_daily():
    path = ROOT / "reports/fuxi_a1_model_account_20260926_v4"
    out = {}
    for arm in ("v3", "amount_top80"):
        source = path / f"{arm}.json"
        obj = load(source)
        row = next(s for s in obj["scenarios"] if s["scenario"] == SCENARIO)
        assert row["complete_model_book"] and len(row["daily_balances"]) == 320
        out[arm] = (source, row)
    return out


def prefix_replay():
    frozen = load(ROOT / "reports/fuxi_quality_gate_20260925/targets_v3.json")["targets"]
    signal_dates = sorted(frozen)
    raw = ROOT / "data/raw/daily_prices.csv"
    # Only the prior global session and signal sessions can affect this formula.
    calendar = sorted({str(x)[:10] for x in pd.read_csv(
        ROOT / "data/raw/trade_calendar.csv", usecols=["cal_date", "is_open"],
        dtype=str).query("is_open == '1'").cal_date})
    def norm(s):
        return pd.to_datetime(s, format="mixed").strftime("%Y-%m-%d")
    calendar = sorted({norm(x) for x in calendar})
    preceding = {d: calendar[calendar.index(d) - 1] for d in signal_dates}
    needed = set(signal_dates) | set(preceding.values())
    frames = []
    for part in pd.read_csv(raw, usecols=["ts_code", "trade_date", "open", "close", "amount"],
                            dtype={"ts_code": str, "trade_date": str}, chunksize=500_000):
        dates = pd.to_datetime(part.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
        take = dates.isin(needed)
        if take.any():
            x = part.loc[take].copy()
            x["date"] = dates[take].to_numpy()
            frames.append(x)
    price = pd.concat(frames, ignore_index=True)
    assert not price.duplicated(["date", "ts_code"]).any()
    for field in ("open", "close", "amount"):
        price[field] = pd.to_numeric(price[field], errors="coerce").astype("float32")
    groups = {d: g.set_index("ts_code") for d, g in price.groupby("date")}
    checks = []
    for day in signal_dates:
        now = groups[day]
        prior = groups[preceding[day]][["close"]].rename(columns={"close": "prior_close"})
        panel = now.join(prior, how="left")
        gap = panel.open / panel.prior_close - 1
        intraday = panel.close / panel.open - 1
        panel["factor_value"] = gap.rank(pct=True) - intraday.rank(pct=True)
        panel = panel[np.isfinite(panel.factor_value) & (panel.close > 0)]
        ranked = panel.reset_index().sort_values(
            ["factor_value", "amount", "ts_code"],
            ascending=[False, False, True], kind="mergesort")
        expected = list(frozen[day]); observed = ranked.head(80).ts_code.tolist()
        checks.append({"date": day, "eligible": len(panel),
                       "exact_order": observed == expected,
                       "intersection": len(set(observed) & set(expected)),
                       "first_expected": expected[:3], "first_observed": observed[:3]})
    return {"signal_dates": len(signal_dates),
            "exact_order_dates": sum(x["exact_order"] for x in checks),
            "min_top80_intersection": min(x["intersection"] for x in checks),
            "checks": checks,
            "interpretation": "independent price-only as-of-date reconstruction; no post-signal rows"}


def replay(arm, sealed):
    _, targets = targets_for(arm)
    codes = {c for weights in targets.values() if weights for c in weights}
    if arm == "amount_top80":
        codes.add("600150.SH")
    days, bars, _ = load_bars(codes)
    planned = load(ADJUDICATION)
    extension = load(OUT / "held_extension_audit_v5.json")
    by_key = {}
    for item in planned["events"] + extension["events"]:
        if item.get("status", "approved_for_gross_cash_stress_only") != "approved_for_gross_cash_stress_only":
            continue
        a = item.get("action")
        if a and a["code"] in codes and a["record_date"] in days and a["ex_date"] in days:
            by_key[(a["code"], a["ex_date"])] = CorporateAction(**a)
    if arm == "amount_top80":
        a = CorporateAction(**load(OUT / "actual_holdings_extension_v1.json")["action"])
        by_key[(a.code, a.ex_date)] = a
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    ledger = run_ledger(
        days, bars, targets, initial_cash=10_000_000, cost_bps=15,
        rules=rules, actions=list(by_key.values()), fractional_share_policy="floor",
        gross_cash_tax_rate=0.2, delist_mark_fraction=0,
        reference_only_events={("002822.SZ", "2025-12-30")},
        share_exchanges=[EXCHANGE] if arm == "amount_top80" else [],
        missing_limit_policy="defer", halted_session_check_policy="vacuous",
        delist_evidence={c: f for c, f in DELIST_TRACE_FACTS.items() if c in codes},
    )
    frozen = sealed[arm][1]["daily_balances"]
    assert len(ledger.days) == len(frozen) == 320
    assert all(a["date"] == b["date"] and math.isclose(a["nav"], b["nav"], abs_tol=1e-6)
               for a, b in zip(ledger.days, frozen))
    return ledger


def contributions(ledger):
    previous = {}
    previous_nav = 10_000_000.0
    journal = defaultdict(list)
    for item in ledger.journal:
        journal[item["date"]].append(item)
    daily = []
    by_code = defaultdict(float)
    cost_total = 0.0
    for row in ledger.days:
        current = {code: p["shares"] * p["mark"] for code, p in row["positions"].items()}
        pnl = defaultdict(float)
        for code in set(previous) | set(current):
            pnl[code] += current.get(code, 0) - previous.get(code, 0)
        for event in journal[row["date"]]:
            pnl[event["code"]] += event["cash_delta"] + event["receivable_delta"]
            if event["kind"] == "fill":
                cost_total += event["fee"]
        total = sum(pnl.values())
        nav_delta = row["nav"] - previous_nav
        if not math.isclose(total, nav_delta, abs_tol=1e-5):
            raise AssertionError((row["date"], total, nav_delta))
        for code, value in pnl.items():
            by_code[code] += value
        daily.append({"date": row["date"], "pnl": nav_delta,
                      "by_code": {k: v for k, v in pnl.items() if abs(v) > 1e-8},
                      "nav": row["nav"]})
        previous, previous_nav = current, row["nav"]
    return daily, by_code, cost_total


def attribution():
    sealed = sealed_daily()
    ledgers = {arm: replay(arm, sealed) for arm in sealed}
    parts = {arm: contributions(ledger) for arm, ledger in ledgers.items()}
    daily = []
    active_code = defaultdict(float)
    for a, b in zip(parts["v3"][0], parts["amount_top80"][0]):
        assert a["date"] == b["date"]
        for code in set(a["by_code"]) | set(b["by_code"]):
            active_code[code] += a["by_code"].get(code, 0) - b["by_code"].get(code, 0)
        daily.append({"date": a["date"], "v3_pnl": a["pnl"], "baseline_pnl": b["pnl"],
                      "active_pnl": a["pnl"] - b["pnl"]})
    active_total = sum(x["active_pnl"] for x in daily)
    assert math.isclose(active_total, sum(active_code.values()), abs_tol=1e-5)
    months = Counter()
    for item in daily:
        months[item["date"][:7]] += item["active_pnl"]
    ordered = sorted(active_code.items(), key=lambda x: x[1], reverse=True)
    positive_days = sorted(daily, key=lambda x: x["active_pnl"], reverse=True)
    negative_days = sorted(daily, key=lambda x: x["active_pnl"])
    return {"scenario": SCENARIO, "full_active_pnl_yuan": active_total,
            "full_active_return_points": active_total / 10_000_000,
            "year_active_pnl_yuan": {year: sum(x["active_pnl"] for x in daily if x["date"].startswith(year))
                                     for year in ("2025", "2026")},
            "monthly_active_pnl_yuan": dict(sorted(months.items())),
            "top_positive_days": positive_days[:10], "top_negative_days": negative_days[:10],
            "top_positive_codes": ordered[:20], "top_negative_codes": ordered[-20:][::-1],
            "top10_positive_codes_sum": sum(v for _, v in ordered[:10]),
            "top10_positive_days_sum": sum(x["active_pnl"] for x in positive_days[:10]),
            "v3_cost_yuan": parts["v3"][2], "baseline_cost_yuan": parts["amount_top80"][2],
            "positive_active_days": sum(x["active_pnl"] > 0 for x in daily),
            "negative_active_days": sum(x["active_pnl"] < 0 for x in daily),
            "paired_days": len(daily),
            "accounting_identity": "per-code delta marked value plus journal cash/receivable; sum equals daily NAV change"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pit", "attribution"))
    action = parser.parse_args().action
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("Fuxi writer lock present")
    DEST.mkdir(exist_ok=True)
    target = DEST / f"{action}.json"
    if target.exists():
        raise RuntimeError("audit output already frozen")
    result = prefix_replay() if action == "pit" else attribution()
    result["input_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "reports/fuxi_quality_gate_20260925/targets_v3.json",
        ROOT / "data/raw/daily_prices.csv",
        ROOT / "reports/fuxi_a1_model_account_20260926_v4/v3.json",
        ROOT / "reports/fuxi_a1_model_account_20260926_v4/amount_top80.json",
    )}
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("writer appeared during audit")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "input_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
