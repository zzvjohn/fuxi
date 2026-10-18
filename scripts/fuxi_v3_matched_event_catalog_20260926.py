"""Adjudicate matched-control ex-date events from already sealed daily responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from a1_corporate_action_adapter import adjudicate_bundle  # noqa: E402
from fuxi_a1_adjudicate_20260926 import measured_pairs  # noqa: E402
from fuxi_a1_held_extension_audit_20260926 import SPECIAL  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402

OUT = ROOT / "reports/fuxi_v3_matched_book_20260926"
MATCH = ROOT / "reports/fuxi_v3_matched_control_20260926_v3/matched_targets.json"
RAW = ROOT / "reports/window_evidence_20260923/raw"
EXCEPTIONS = ROOT / "reports/fuxi_a1_source_audit_20260926/issuer_exceptions.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_paths():
    calendar = pd.read_csv(ROOT / "data/raw/trade_calendar.csv",
                           usecols=["cal_date", "is_open"], dtype=str)
    dates = pd.to_datetime(calendar.loc[calendar.is_open == "1", "cal_date"],
                           format="mixed").dt.strftime("%Y-%m-%d")
    days = sorted(set(d for d in dates if "2025-01-02" <= d <= "2026-04-30"))
    if len(days) != 320:
        raise RuntimeError("frozen 320-session calendar changed")
    return [RAW / ("dividend_" + day.replace("-", "") + ".json") for day in days]


def inputs():
    return [MATCH, EXCEPTIONS, ROOT / "data/raw/trade_calendar.csv",
            ROOT / "data/raw/daily_prices.csv", ROOT / "data/raw/adj_factor_daily.csv",
            ROOT / "research/factor_alchemy/a1_corporate_action_adapter.py",
            ROOT / "research/factor_alchemy/corporate_action_inventory.py",
            Path(__file__).resolve(), *source_paths()]


def prepare():
    with FuxiWriterLock(run_id="v3_matched_event_catalog_prepare"):
        path = OUT / "event_catalog_protocol.json"
        if path.exists():
            raise RuntimeError("event catalog protocol already sealed")
        paths = inputs()
        if any(not p.is_file() for p in paths):
            raise RuntimeError("required sealed response missing")
        protocol = {
            "schema": "fuxi_v3_matched_event_catalog_protocol/v1",
            "scope": "already-sealed daily ex-date responses for matched-control selected-union codes; no network collection",
            "selection_rule": "all implementation rows for selected-union codes and ex_dates within frozen 69-signal calendar; adjudicate all before seeing later account returns",
            "account_rule": "approved gross-cash stress actions only; unresolved keys not filled or treated as zero",
            "source_paths_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        }
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("protocol_sha256", sha(path))


def run():
    with FuxiWriterLock(run_id="v3_matched_event_catalog_run"):
        protocol_path = OUT / "event_catalog_protocol.json"
        output = OUT / "event_catalog.json"
        if output.exists():
            raise RuntimeError("event catalog already frozen")
        protocol = load(protocol_path)
        if protocol["schema"] != "fuxi_v3_matched_event_catalog_protocol/v1":
            raise RuntimeError("unknown protocol")
        for relative, expected in protocol["source_paths_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("frozen input changed: " + relative)
        match = load(MATCH)
        codes = {c for basket in match["control_targets"].values() for c in basket}
        exceptions = {(e["code"], e["ex_date"]): e for e in load(EXCEPTIONS)["events"]}
        exceptions.update(SPECIAL)
        rows_by_key = defaultdict(list)
        source_by_key = {}
        incomplete_response_dates = []
        for path in source_paths():
            payload = load(path)
            day = path.stem.split("_")[-1]
            if payload.get("status") != "response_received" or payload.get("possible_truncation"):
                incomplete_response_dates.append(day)
                continue
            for row in payload.get("records", []):
                if (row.get("ts_code") in codes and row.get("div_proc") == "实施"
                        and str(row.get("ex_date")) == day):
                    ex_date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
                    key = (row["ts_code"], ex_date)
                    rows_by_key[key].append(row)
                    source_by_key[key] = path
        if incomplete_response_dates:
            raise RuntimeError("daily response coverage changed: " + str(incomplete_response_dates[:5]))
        pairs = measured_pairs(set(rows_by_key))
        events = []
        for (code, ex_date), rows in sorted(rows_by_key.items()):
            path = source_by_key[(code, ex_date)]
            p = pairs["daily_prices.csv"].get((code, ex_date))
            a = pairs["adj_factor_daily.csv"].get((code, ex_date))
            if p is None or a is None or any(x != x for x in (*p, *a)):
                events.append({"code": code, "ex_date": ex_date, "status": "unresolved",
                               "reason": "market_reference_unavailable", "action": None,
                               "source_rows": len(rows), "source_sha256": sha(path)})
                continue
            verdict = adjudicate_bundle(
                rows, code=code, ex_date=ex_date,
                pre_close=float(p[0]), adj_prev=float(a[0]), adj_ex=float(a[1]),
                issuer_exception=exceptions.get((code, ex_date)),
                source_evidence=str(path.relative_to(ROOT)) + " sha256=" + sha(path))
            events.append({"code": code, "ex_date": ex_date,
                           "status": verdict.status, "reason": verdict.reason,
                           "action": verdict.action, "detail": verdict.detail,
                           "source_rows": len(rows), "source_sha256": sha(path)})
        result = {
            "schema": "fuxi_v3_matched_event_catalog/v1",
            "protocol_sha256": sha(protocol_path),
            "selected_union_codes": len(codes),
            "event_keys": len(rows_by_key), "status_counts": dict(Counter(e["status"] for e in events)),
            "reason_counts": dict(Counter(e["reason"] for e in events)),
            "events": events,
            "economic_status": "not_evaluated", "formal_eligibility": False,
            "source_limit": "same-provider sealed ex-date responses and issuer exceptions; not actual account tax/fill evidence",
        }
        for relative, expected in protocol["source_paths_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("input changed during catalog: " + relative)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("selected_union_codes", "event_keys",
                                                "status_counts", "reason_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else run()
