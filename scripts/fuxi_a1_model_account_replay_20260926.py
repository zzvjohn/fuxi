"""Frozen A1 reference-account scenarios; never a broker or JQ account label."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from portfolio_execution_ledger import CorporateAction, LotRule, assess_economic_book, run_ledger
from fuxi_a1_three_arm_strict_probe_20260926 import (
    ADJUDICATION, OUT, load, load_bars, targets_for,
)
from fuxi_a1_actual_holdings_trace_20260926 import DELIST_TRACE_FACTS
from run_contract import FuxiWriterLock

ARMS = ("v2", "v3", "amount_top80")
EXCHANGE_EVIDENCE = (
    "https://dataclouds.cninfo.com.cn/shgonggao/2025/2025-09-12/"
    "d59cd0378efa11f0ad48fa163e957f7a.pdf"
)
EXCHANGE = dict(old_code="601989.SH", new_code="600150.SH",
                date="2025-09-16", ratio=0.1339, evidence=EXCHANGE_EVIDENCE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dec(value) -> Decimal:
    return Decimal(str(value))


def independent_calculation(ledger, bars, actions, delisted, *,
                            fractional_policy, tax_rate, delist_mark_fraction,
                            share_exchanges=(), initial_cash=10_000_000,
                            cost_bps=15):
    """Rebuild rights/balances in Decimal and verify raw-source fill/mark prices."""
    cash, receivable = dec(initial_cash), Decimal(0)
    holdings: dict[str, int] = {}
    source_marks: dict[str, float] = {}
    actions_by_id = {a.event_id: a for a in actions}
    recorded: dict[str, int] = {}
    payments: dict[str, Decimal] = {}
    exchanges = {x["old_code"]: x for x in share_exchanges}
    exchange_out: dict[str, int] = {}
    cursor = 0
    worst_nav_gap = Decimal(0)
    checked_fills = 0
    checked_marks = 0
    checked_rights = 0
    checked_model_marks = 0
    daily_parity = []
    for row in ledger.days:
        day = row["date"]
        while cursor < len(ledger.journal) and ledger.journal[cursor]["date"] <= day:
            entry = ledger.journal[cursor]
            code = entry["code"]
            quantity = entry["quantity"]
            kind = entry["kind"]
            if kind == "entitlement":
                action = actions_by_id[entry["event_id"]]
                count = recorded[action.event_id]
                theoretical = dec(count) * (dec(action.share_multiplier) - 1)
                if theoretical == theoretical.to_integral_value():
                    expected_extra = int(theoretical)
                else:
                    expected_extra = (math.floor if fractional_policy == "floor"
                                      else math.ceil)(theoretical)
                due = dec(count) * dec(action.cash_per_share_net) * (1 - dec(tax_rate))
                if quantity != expected_extra or abs(dec(entry["receivable_delta"]) - due) > Decimal("0.000001"):
                    raise AssertionError(f"entitlement mismatch: {day} {code}")
                payments[action.event_id] = due
                checked_rights += 1
            elif kind == "payment":
                if abs(dec(entry["cash_delta"]) - payments[entry["event_id"]]) > Decimal("0.000001"):
                    raise AssertionError(f"payment mismatch: {day} {code}")
            elif kind == "share_exchange_out":
                exchange = exchanges[code]
                if day != exchange["date"] or quantity != -holdings.get(code, 0):
                    raise AssertionError(f"exchange old shares mismatch: {day} {code}")
                exchange_out[code] = -quantity
            elif kind == "share_exchange_in":
                exchange = next(x for x in share_exchanges if x["new_code"] == code and x["date"] == day)
                theoretical = dec(exchange_out[exchange["old_code"]]) * dec(exchange["ratio"])
                expected_new = (math.floor if fractional_policy == "floor" else math.ceil)(theoretical)
                if quantity != expected_new:
                    raise AssertionError(f"exchange new shares mismatch: {day} {code}")
            holdings[code] = holdings.get(code, 0) + quantity
            cash += dec(entry["cash_delta"])
            receivable += dec(entry["receivable_delta"])
            if kind == "fill":
                raw_open = bars.get((day, code), {}).get("open")
                if raw_open is None or not math.isclose(entry["price"], raw_open, abs_tol=1e-9):
                    raise AssertionError(f"fill not matched to source open: {day} {code}")
                expected_fee = dec(abs(quantity)) * dec(raw_open) * dec(cost_bps) / 10000
                if abs(dec(entry["fee"]) - expected_fee) > Decimal("0.000001"):
                    raise AssertionError(f"fee mismatch: {day} {code}")
                checked_fills += 1
            if holdings[code] == 0:
                source_marks.pop(code, None)
            cursor += 1
        expected = {code: n for code, n in holdings.items() if n}
        observed = {code: p["shares"] for code, p in row["positions"].items()}
        if expected != observed:
            raise AssertionError(f"position mismatch: {day}")
        cash_gap = abs(cash - dec(row["cash"]))
        receivable_gap = abs(receivable - dec(row["receivable"]))
        if cash_gap > Decimal("0.00001"):
            raise AssertionError(f"cash mismatch: {day}")
        if receivable_gap > Decimal("0.00001"):
            raise AssertionError(f"receivable mismatch: {day}")
        marked = Decimal(0)
        for code, position in row["positions"].items():
            mark = position["mark"]
            if not math.isfinite(mark):
                raise AssertionError(f"nonfinite mark: {day} {code}")
            source = bars.get((day, code), {})
            delist = delisted.get(code)
            if delist and day >= delist["date"]:
                exchange = exchanges.get(code)
                if exchange and day < exchange["date"]:
                    new_close = bars.get((day, exchange["new_code"]), {}).get("close")
                    expected_mark = exchange["ratio"] * new_close if new_close else math.nan
                else:
                    expected_mark = source_marks[code] * delist_mark_fraction
                checked_model_marks += 1
            elif source.get("state") in ("trading", "partial_halt"):
                expected_mark = source.get("close")
                source_marks[code] = expected_mark
                checked_marks += 1
            else:
                expected_mark = source_marks.get(code)
                checked_model_marks += 1
            if expected_mark is None or not math.isclose(mark, expected_mark, abs_tol=1e-8):
                raise AssertionError(f"mark not matched to model/source: {day} {code}")
            marked += dec(position["shares"]) * dec(mark)
        nav = cash + receivable + marked
        gap = abs(nav - dec(row["nav"]))
        worst_nav_gap = max(worst_nav_gap, gap)
        if gap > Decimal("0.0001"):
            raise AssertionError(f"NAV mismatch: {day} gap={gap}")
        daily_parity.append({"date": day, "cash_gap": float(cash_gap),
                             "receivable_gap": float(receivable_gap),
                             "nav_gap": float(gap), "positions_equal": True})
        for action in actions:
            if action.record_date == day:
                recorded[action.event_id] = holdings.get(action.code, 0)
    if cursor != len(ledger.journal) or len(ledger.days) != 320:
        raise AssertionError("incomplete independent replay")
    return dict(compared_days=len(ledger.days), checked_fills=checked_fills,
                checked_rights=checked_rights, checked_live_marks=checked_marks,
                checked_model_marks=checked_model_marks,
                max_nav_abs_gap=float(worst_nav_gap),
                daily_parity=daily_parity,
                independent_data_source=False, broker_or_jq_parity=False)


def run_arm(arm: str) -> dict:
    target_path, targets = targets_for(arm)
    codes = {c for weights in targets.values() if weights for c in weights}
    if arm == "amount_top80":
        codes.add("600150.SH")
    days, bars, coverage = load_bars(codes)
    if len(days) != 320:
        raise AssertionError("frozen calendar changed")
    planned = load(ADJUDICATION)
    extension = load(OUT / "held_extension_audit_v5.json")
    if planned["counts"] != {"approved_for_gross_cash_stress_only": 229}:
        raise AssertionError("planned action seal changed")
    if extension["union_keys"] != 24:
        raise AssertionError("held-event seal changed")
    actions_by_key = {}
    for item in planned["events"] + extension["events"]:
        if item.get("status", "approved_for_gross_cash_stress_only") != "approved_for_gross_cash_stress_only":
            continue
        action = item.get("action")
        if action is None or action["code"] not in codes:
            continue
        if action["record_date"] not in days or action["ex_date"] not in days:
            continue
        key = (action["code"], action["ex_date"])
        previous = actions_by_key.get(key)
        if previous and (previous.cash_per_share_net != action["cash_per_share_net"]
                         or previous.share_multiplier != action["share_multiplier"]):
            raise AssertionError(f"conflicting event rights: {key}")
        actions_by_key[key] = CorporateAction(**action)
    if arm == "amount_top80":
        item = load(OUT / "actual_holdings_extension_v1.json")
        action = CorporateAction(**item["action"])
        actions_by_key[(action.code, action.ex_date)] = action
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    delisted = {code: fact for code, fact in DELIST_TRACE_FACTS.items() if code in codes}
    variants = []
    for fraction in ("floor", "ceil"):
        for tax in (0.0, 0.2):
            for delist_mark in (0.0, 1.0):
                scenario = f"{fraction}_tax{int(tax*100)}_delist{int(delist_mark*100)}"
                selected_actions = list(actions_by_key.values())
                model_exchanges = [EXCHANGE] if arm == "amount_top80" else []
                ledger = run_ledger(
                    days, bars, targets, initial_cash=10_000_000, cost_bps=15,
                    rules=rules, actions=selected_actions,
                    fractional_share_policy=fraction,
                    gross_cash_tax_rate=tax,
                    delist_mark_fraction=delist_mark,
                    reference_only_events={("002822.SZ", "2025-12-30")},
                    share_exchanges=model_exchanges,
                    missing_limit_policy="defer",
                    halted_session_check_policy="vacuous",
                    delist_evidence=delisted,
                )
                assessment = assess_economic_book(ledger, days, 10_000_000)
                complete = (len(ledger.days) == len(days) and not ledger.errors
                            and not ledger.excluded and
                            all(math.isfinite(row["nav"]) for row in ledger.days))
                parity = (independent_calculation(
                    ledger, bars, selected_actions, delisted,
                    fractional_policy=fraction, tax_rate=tax,
                    delist_mark_fraction=delist_mark,
                    share_exchanges=model_exchanges) if complete else None)
                variants.append(dict(
                    scenario=scenario, complete_model_book=complete,
                    status=("model_complete" if complete else "data_insufficient"),
                    processed_days=len(ledger.days),
                    first_errors=ledger.errors[:5],
                    exclusion_count=len(ledger.excluded),
                    assessment=assessment,
                    independent_calculation=parity,
                    end_nav=ledger.days[-1]["nav"] if complete else None,
                    net_return=(ledger.days[-1]["nav"] / 10_000_000 - 1 if complete else None),
                    filled_orders=sum(o.get("filled", 0) > 0 for o in ledger.orders),
                    event_counts=dict(Counter(e.get("kind") for e in ledger.journal)),
                    daily_balances=[{
                        **{k: row[k] for k in ("date", "cash", "receivable", "nav")},
                        "positions": {code: [p["shares"], p["mark"]]
                                      for code, p in row["positions"].items()},
                    } for row in ledger.days] if complete else [],
                ))
    return dict(arm=arm, window=[days[0], days[-1]], expected_days=len(days),
                signal_dates=len(targets), action_keys=len(actions_by_key),
                quote_coverage=coverage, scenarios=variants,
                input_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
                    target_path, ADJUDICATION, OUT / "held_extension_audit_v5.json",
                    ROOT / "data/raw/daily_prices.csv",
                    ROOT / "data/raw/stk_limit.csv",
                    ROOT / "data/raw/adj_factor_daily.csv",
                    ROOT / "data/raw/suspend_d.csv",
                    ROOT / "research/factor_alchemy/portfolio_execution_ledger.py",
                    ROOT / "scripts/fuxi_a1_model_account_replay_20260926.py",
                    OUT / "issuer_exceptions.json",
                    OUT / "actual_holdings_extension_v1.json",
                )})


def main():
    outdir = ROOT / "reports/fuxi_a1_model_account_20260926_v4"
    if outdir.exists():
        raise RuntimeError("sealed model-account directory already exists")
    with FuxiWriterLock(run_id="a1_model_account_three_arm"):
        payload = [run_arm(arm) for arm in ARMS]
        outdir.mkdir(parents=True, exist_ok=False)
        for arm in payload:
            path = outdir / f"{arm['arm']}.json"
            path.write_text(json.dumps(arm, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        brief = [{"arm": x["arm"], "complete": sum(s["complete_model_book"] for s in x["scenarios"]),
                  "errors": [s["first_errors"] for s in x["scenarios"] if s["first_errors"]][:2]}
                 for x in payload]
        print(json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
