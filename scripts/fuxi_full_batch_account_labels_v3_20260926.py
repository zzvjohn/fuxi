"""Resumable cost-after model books for every finite-proxy full-batch formula."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

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

OUT = ROOT / "reports/fuxi_full_batch_account_labels_20260926_v3"
MATCH = ROOT / "reports/fuxi_full_batch_account_targets_20260926_v2/targets.json"
CATALOG = ROOT / "reports/fuxi_full_batch_event_catalog_20260926/event_catalog.json"
SOURCE = (
    MATCH, CATALOG, ROOT / "reports/fuxi_full_batch_event_catalog_20260926/event_catalog_protocol.json",
    ADJUDICATION,
    A1_OUT / "held_extension_audit_v5.json",
    A1_OUT / "actual_holdings_extension_v1.json",
    ROOT / "data/raw/daily_prices.csv",
    ROOT / "data/raw/stk_limit.csv",
    ROOT / "data/raw/adj_factor_daily.csv",
    ROOT / "data/raw/suspend_d.csv",
    ROOT / "data/raw/delist_dates.csv",
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
    with FuxiWriterLock(run_id="full_batch_account_labels_prepare"):
        OUT.mkdir(exist_ok=True)
        protocol_path = OUT / "protocol.json"
        if protocol_path.exists():
            raise RuntimeError("candidate account protocol already frozen")
        match = load(MATCH)
        if len(match["candidate_rows"]) != 252 or len(match["targets_by_expression_sha256"]) != 98:
            raise RuntimeError("candidate target set incomplete")
        protocol = {
            "schema": "fuxi_full_batch_account_label_protocol/v3",
            "supersedes": "v1 lacked provider delist facts; v2 proved later quote gaps remain; v3 shares one immutable in-memory all-candidate quote panel to complete the same declared model efficiently; prior labels remain sealed",
            "scope": "all 98 finite-proxy distinct formulas and all 252 row identities; invalid sentinels retain explicit no-label status",
            "window": ["2025-01-02", "2026-04-30"], "expected_days": 320,
            "initial_cash": 10_000_000, "cost_bps_each_side": 15,
            "signal_cutoff": "post_close", "earliest_fill": "next_session_open_proxy",
            "reference_scenario": "floor_tax20_delist0",
            "same_execution_model": "frozen A1 ledger; missing limit defer, halted check vacuous, issuer actions from sealed A1 plus separately frozen all-candidate-union adjudication catalog; no hypothetical action fill",
            "delist_source": "existing data/raw/delist_dates.csv D rows supply delist effective date for zero-value model mark; issuer-confirmed special cases retained; actual recovery remains unknown",
            "numeric_noise_exception": "every candidate: no implementation key in complete sealed catalog, action_required solely by adj_factor change, relative factor change <=0.00025, and same-day pre_close equals prior session close within 0.011; set action_required=false for those audited keys, never fabricate rights",
            "issuer_reference_only": "all catalog_v2 not_holder_distribution keys with dated issuer exception are price-reference events only; old shareholders receive no new shares",
            "economic_success": "each candidate complete 320-day book, zero errors/exclusions, independent same-input arithmetic replay",
            "if_failure": "candidate label remains data_insufficient with blocker; no candidate or arm receives a fabricated reward",
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCE},
        }
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("protocol_sha256", sha(protocol_path))


def numeric_noise_keys(codes, days, bars, catalog):
    known = {(e["code"], e["ex_date"]) for e in catalog["events"]}
    pieces = []
    for chunk in pd.read_csv(ROOT / "data/raw/adj_factor_daily.csv",
                             usecols=["ts_code", "trade_date", "adj_factor"],
                             dtype={"ts_code": str, "trade_date": str}, chunksize=500_000):
        chunk = chunk[chunk.ts_code.isin(codes)]
        if not chunk.empty:
            chunk = chunk.copy()
            chunk["date"] = pd.to_datetime(chunk.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
            pieces.append(chunk[chunk.date.between("2024-12-01", days[-1])])
    factors = pd.concat(pieces, ignore_index=True).sort_values(["ts_code", "date"])
    if factors.duplicated(["date", "ts_code"]).any():
        raise RuntimeError("duplicate factor key in numeric-noise audit")
    factors["previous"] = factors.groupby("ts_code").adj_factor.shift()
    prior_day = {date: days[i-1] for i, date in enumerate(days) if i}
    exceptions = []
    for row in factors.itertuples():
        if row.date not in prior_day or (row.ts_code, row.date) in known:
            continue
        if not (math.isfinite(row.adj_factor) and math.isfinite(row.previous)
                and row.adj_factor > 0 and row.previous > 0):
            continue
        change = abs(row.adj_factor / row.previous - 1)
        if not 0 < change <= 0.00025:
            continue
        current = bars.get((row.date, row.ts_code), {})
        prior = bars.get((prior_day[row.date], row.ts_code), {})
        if (current.get("action_required") is True
                and current.get("action_check_available") is True
                and current.get("pre_close") is not None
                and prior.get("close") is not None
                and math.isclose(current["pre_close"], prior["close"], abs_tol=0.011, rel_tol=0)):
            exceptions.append({"code": row.ts_code, "date": row.date,
                               "adj_prev": float(row.previous), "adj_today": float(row.adj_factor),
                               "relative_change": float(change),
                               "pre_close": float(current["pre_close"]),
                               "prior_close": float(prior["close"])})
    return exceptions


def construct(arm, targets, shared_bars=None, shared_noise=None):
    codes = {code for basket in targets.values() if basket for code in basket}
    if "601989.SH" in codes:
        codes.add("600150.SH")
    days, bars, coverage = load_bars(codes) if shared_bars is None else shared_bars
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
    catalog = load(CATALOG)
    if catalog["schema"] != "fuxi_full_batch_event_catalog/v1":
        raise RuntimeError("unknown event catalog")
    reference_only = {(e["code"], e["ex_date"]) for e in catalog["events"]
                      if e["status"] == "not_holder_distribution"}
    reference_only.add(("002822.SZ", "2025-12-30"))
    new_actions = 0
    for item in catalog["events"]:
        if item["status"] != "approved_for_gross_cash_stress_only" or not item.get("action"):
            continue
        action = item["action"]
        if action["code"] not in codes or action["record_date"] not in days or action["ex_date"] not in days:
            continue
        key = (action["code"], action["ex_date"])
        if key in by_key:
            prior = by_key[key]
            if (not math.isclose(prior.cash_per_share_net, action["cash_per_share_net"], abs_tol=1e-8)
                    or not math.isclose(prior.share_multiplier, action["share_multiplier"], abs_tol=1e-8)):
                raise RuntimeError("sealed action conflicts with broad catalog: " + str(key))
            continue
        by_key[key] = CorporateAction(**action)
        new_actions += 1
    exceptions = (numeric_noise_keys(codes, days, bars, catalog) if shared_noise is None
                  else [e for e in shared_noise if e["code"] in codes]) if arm in (
                      "matched_control", "candidate") else []
    noise_set = {(e["date"], e["code"]) for e in exceptions}
    if noise_set:
        original_bars = bars
        class CorrectedBars:
            def get(self, key, default=None):
                value = original_bars.get(key, default)
                if key in noise_set and isinstance(value, dict):
                    return {**value, "action_required": False}
                return value
        bars = CorrectedBars()
    rules = {c: LotRule(200, 1) if c.startswith(("688", "689")) else
             LotRule(100, 1) if c.endswith(".BJ") else LotRule() for c in codes}
    delisted = {}
    registry = pd.read_csv(ROOT / "data/raw/delist_dates.csv", dtype=str)
    registry_hash = sha(ROOT / "data/raw/delist_dates.csv")
    for row in registry.itertuples():
        if row.ts_code in codes and row.list_status == "D" and isinstance(row.delist_date, str):
            day = pd.to_datetime(row.delist_date, format="mixed").strftime("%Y-%m-%d")
            if days[0] <= day <= days[-1]:
                delisted[row.ts_code] = {"date": day,
                    "evidence": "data/raw/delist_dates.csv sha256=" + registry_hash,
                    "event": "provider-delisted; model zero recovery, actual residual unknown"}
    delisted.update({c: fact for c, fact in DELIST_TRACE_FACTS.items() if c in codes})
    actions = list(by_key.values())
    exchanges = [EXCHANGE] if "601989.SH" in codes else []
    ledger = run_ledger(
        days, bars, targets, initial_cash=10_000_000, cost_bps=15,
        rules=rules, actions=actions, fractional_share_policy="floor",
        gross_cash_tax_rate=0.2, delist_mark_fraction=0,
        reference_only_events=reference_only,
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
        "action_keys_supplied": len(by_key), "new_catalog_action_keys": new_actions,
        "numeric_noise_exceptions": exceptions,
        "reference_only_event_keys": sorted([list(k) for k in reference_only]),
        "quote_coverage": coverage,
        "filled_orders": sum(o.get("filled", 0) > 0 for o in ledger.orders),
        "end_nav": ledger.days[-1]["nav"] if complete else None,
        "daily_nav": [{"date": r["date"], "nav": r["nav"]} for r in ledger.days] if complete else [],
        "independent_calculation": {k: v for k, v in independent.items() if k != "daily_parity"}
        if independent else None,
    }


def checked_protocol():
    path = OUT / "protocol.json"
    protocol = load(path)
    if protocol["schema"] != "fuxi_full_batch_account_label_protocol/v3":
        raise RuntimeError("unknown candidate account protocol")
    for source in SOURCE:
        if sha(source) != protocol["input_sha256"][str(source.relative_to(ROOT))]:
            raise RuntimeError("frozen source changed: " + str(source))
    return protocol, sha(path)


def run_one(expression_hash):
    with FuxiWriterLock(run_id="full_batch_account_label_" + expression_hash[:10]):
        protocol, protocol_hash = checked_protocol()
        match = load(MATCH)
        targets = match["targets_by_expression_sha256"].get(expression_hash)
        if targets is None:
            raise RuntimeError("unknown or ineligible expression hash")
        folder = OUT / "labels"
        folder.mkdir(exist_ok=True)
        path = folder / (expression_hash + ".json")
        if path.exists():
            raise RuntimeError("candidate label already frozen")
        result = construct("candidate", targets)
        result["expression_sha256"] = expression_hash
        result["protocol_sha256"] = protocol_hash
        result["economic_label_status"] = (
            "exploratory_model_complete" if result["status"] == "model_complete"
            else "data_insufficient")
        result["formal_eligibility"] = False
        for source in SOURCE:
            if sha(source) != protocol["input_sha256"][str(source.relative_to(ROOT))]:
                raise RuntimeError("source changed during candidate book: " + str(source))
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "expression_sha256", "status", "processed_days", "end_nav", "error_counts")},
        ensure_ascii=False), flush=True)


def run_batch():
    with FuxiWriterLock(run_id="full_batch_account_labels_v3_batch"):
        protocol, protocol_hash = checked_protocol()
        match = load(MATCH)
        schedules = match["targets_by_expression_sha256"]
        all_codes = {code for targets in schedules.values() for basket in targets.values()
                     for code in basket}
        if "601989.SH" in all_codes:
            all_codes.add("600150.SH")
        if len(schedules) != 98:
            raise RuntimeError("candidate schedule cardinality changed")
        source_stat = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in SOURCE}
        days, bars, coverage = load_bars(all_codes)
        catalog = load(CATALOG)
        shared_noise = numeric_noise_keys(all_codes, days, bars, catalog)
        checked_protocol()  # detects changes during the expensive preload
        print(json.dumps({"preloaded_codes": len(all_codes), "days": len(days),
                          "quote_keys": coverage["quote_keys"],
                          "numeric_noise_keys": len(shared_noise)}), flush=True)
        folder = OUT / "labels"
        folder.mkdir(exist_ok=True)
        total = len(schedules)
        for index, expression_hash in enumerate(sorted(schedules), 1):
            path = folder / (expression_hash + ".json")
            if path.exists():
                old = load(path)
                if old["protocol_sha256"] != protocol_hash:
                    raise RuntimeError("existing label has different protocol")
                print(json.dumps({"index": index, "total": total,
                                  "expression_sha256": expression_hash,
                                  "status": "already_sealed"}), flush=True)
                continue
            if any((p.stat().st_size, p.stat().st_mtime_ns) != source_stat[str(p)]
                   for p in SOURCE):
                raise RuntimeError("source changed during batch before candidate " + expression_hash)
            result = construct("candidate", schedules[expression_hash],
                               shared_bars=(days, bars, coverage), shared_noise=shared_noise)
            result["expression_sha256"] = expression_hash
            result["protocol_sha256"] = protocol_hash
            result["shared_union_coverage"] = True
            result["economic_label_status"] = (
                "exploratory_model_complete" if result["status"] == "model_complete"
                else "data_insufficient")
            result["formal_eligibility"] = False
            if any((p.stat().st_size, p.stat().st_mtime_ns) != source_stat[str(p)]
                   for p in SOURCE):
                raise RuntimeError("source changed during candidate " + expression_hash)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            print(json.dumps({"index": index, "total": total,
                              "expression_sha256": expression_hash,
                              "status": result["status"],
                              "processed_days": result["processed_days"],
                              "end_nav": result["end_nav"],
                              "error_counts": result["error_counts"]}, ensure_ascii=False), flush=True)
        checked_protocol()


def summarize():
    with FuxiWriterLock(run_id="full_batch_account_labels_summary"):
        _, protocol_hash = checked_protocol()
        match = load(MATCH)
        output = OUT / "summary.json"
        if output.exists():
            raise RuntimeError("candidate summary already frozen")
        rows = []
        for candidate in match["candidate_rows"]:
            item = {k: candidate[k] for k in (
                "candidate_id", "arm", "seed", "ordinal", "expression_sha256",
                "legacy_fitness", "proxy_status")}
            path = OUT / "labels" / (candidate["expression_sha256"] + ".json")
            if candidate["proxy_status"] == "invalid_sentinel":
                item.update(economic_label_status="proxy_invalid", end_nav=None)
            elif not path.exists():
                item.update(economic_label_status="not_evaluated", end_nav=None)
            else:
                book = load(path)
                if book["protocol_sha256"] != protocol_hash:
                    raise RuntimeError("candidate book protocol mismatch")
                item.update(economic_label_status=book["economic_label_status"],
                            end_nav=book["end_nav"], processed_days=book["processed_days"],
                            error_counts=book["error_counts"])
            rows.append(item)
        result = {"schema": "fuxi_full_batch_account_label_summary/v3",
                  "protocol_sha256": protocol_hash, "candidate_rows": rows,
                  "row_count": len(rows), "distinct_expressions": len({r["expression_sha256"] for r in rows}),
                  "status_counts": dict(Counter(r["economic_label_status"] for r in rows)),
                  "formal_eligibility": False,
                  "scope": "same-source model arithmetic and research-exposed historical window; not JQ or broker parity"}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["status_counts"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run-one", "run-batch", "summarize"))
    parser.add_argument("expression_hash", nargs="?")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run-one":
        if not args.expression_hash or len(args.expression_hash) != 64:
            raise ValueError("run-one needs a full expression SHA256")
        run_one(args.expression_hash)
    elif args.action == "run-batch":
        run_batch()
    else:
        summarize()
