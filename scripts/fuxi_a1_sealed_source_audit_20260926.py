"""Audit bounded A1 corporate-action evidence without collecting or promoting it.

The sealed daily Tushare responses can provide within-provider negative coverage
only if every exchange session has a non-truncated response. A reconstructed
price identity is a separate, weaker accounting check, not issuer verification.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMAND = ROOT / "reports/fuxi_a1_20260926/demand.json"
PROTOCOL = ROOT / "reports/window_evidence_20260923/protocol.json"
RAW = ROOT / "reports/window_evidence_20260923/raw"
INVENTORY = ROOT / "reports/event_inventory_20260923/inventory.json"
CALENDAR = ROOT / "data/raw/trade_calendar.csv"
OUT = ROOT / "reports/fuxi_a1_source_audit_20260926"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict:
    demand, protocol, inventory = load(DEMAND), load(PROTOCOL), load(INVENTORY)
    start, end = demand["window"]
    requested = {
        req["params"]["ex_date"]: req for req in protocol["requests"]
        if req["api"] == "dividend" and req["id"].startswith("dividend_")
    }
    calendar = pd.read_csv(CALENDAR, dtype=str)
    sessions = set(pd.to_datetime(calendar.loc[
        calendar.exchange.eq("SSE") & calendar.is_open.eq("1"), "cal_date"
    ], format="mixed").dt.strftime("%Y%m%d"))
    sessions = sorted(day for day in sessions if start.replace("-", "") <= day <= end.replace("-", ""))

    response_checks = []
    source_rows = {}
    for day in sessions:
        req = requested.get(day)
        item = {"date": day, "request_present": req is not None,
                "response_received": False, "not_truncated": False,
                "identity_consistent": False, "raw_sha256": None}
        if req is not None:
            path = RAW / (req["id"] + ".json")
            if path.exists():
                response = load(path)
                rows = response.get("records") or []
                item.update(response_received=response.get("status") == "response_received",
                            not_truncated=(response.get("possible_truncation") is False
                                           and len(rows) < req["cap"]),
                            identity_consistent=response.get("request") == req
                            and all(str(row.get("ex_date") or "") == day for row in rows),
                            raw_sha256=sha256(path), rows=len(rows))
                if all(item[k] for k in ("response_received", "not_truncated", "identity_consistent")):
                    for row in rows:
                        source_rows.setdefault((row.get("ts_code"), day), []).append(row)
        response_checks.append(item)

    complete_days = {r["date"] for r in response_checks if all(r[k] for k in (
        "request_present", "response_received", "not_truncated", "identity_consistent"))}
    indexed = {(row["code"], row["ex_date"].replace("-", "")): row
               for bucket in ("approved", "rejected", "unresolved")
               for row in inventory[bucket]}
    union = demand["union_request"]
    events = []
    for need in union:
        code, day = need["code"], need["ex_date"].replace("-", "")
        raw_rows = source_rows.get((code, day), [])
        implemented = [row for row in raw_rows if row.get("div_proc") == "实施"]
        verdict = indexed.get((code, day))
        events.append({
            "code": code, "ex_date": need["ex_date"], "arms": need["arms"],
            "source_day_complete": day in complete_days,
            "source_rows": len(raw_rows), "implemented_rows": len(implemented),
            "row_identity_matches": all(row.get("ts_code") == code and row.get("ex_date") == day
                                        for row in raw_rows),
            "inventory_status": verdict["status"] if verdict else "absent",
            "inventory_reason": verdict["reason"] if verdict else None,
            "source_file": "dividend_" + day + ".json",
            "raw_sha256": next((r["raw_sha256"] for r in response_checks if r["date"] == day), None),
            "source_verdict": "provider_response_present" if day in complete_days and implemented
                              else "unresolved",
            "issuer_verified": False, "trading_economic_valid": False,
        })
    statuses = Counter(row["inventory_status"] for row in events)
    anomalies = [row for row in events if row["inventory_status"] != "approved"]
    missing_days = [r["date"] for r in response_checks if r["date"] not in complete_days]
    return {
        "schema": "fuxi_a1_sealed_source_audit/v1",
        "scope": "frozen three arms; sealed Tushare dividend ex-date responses only",
        "window": [start, end], "calendar_sessions": len(sessions),
        "daily_responses_complete": len(complete_days), "daily_response_gaps": missing_days,
        "provider_negative_coverage": len(missing_days) == 0,
        "provider_negative_coverage_limit": "same-provider ex-date query coverage; not issuer-level completeness or executed-holding proof",
        "union_demand_events": len(union), "event_statuses": dict(statuses),
        "demand_source_rows_found": sum(row["implemented_rows"] > 0 for row in events),
        "anomalies": anomalies, "events": events,
        "daily_response_checks": response_checks,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p)
                         for p in (DEMAND, PROTOCOL, INVENTORY, CALENDAR)},
        "economic_status": "data_insufficient", "formal_eligibility": False,
    }


def main():
    result = audit()
    assert result["union_demand_events"] == 229
    assert result["daily_responses_complete"] == result["calendar_sessions"]
    assert result["demand_source_rows_found"] == 229
    assert sum(result["event_statuses"].values()) == 229
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "audit.json"
    if path.exists():
        raise RuntimeError("A1 source audit is already frozen")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in (
        "calendar_sessions", "daily_responses_complete", "union_demand_events",
        "event_statuses", "demand_source_rows_found", "economic_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
