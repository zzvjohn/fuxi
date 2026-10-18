"""Audit the held-event demand found by the non-economic sensitivity trace."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from a1_corporate_action_adapter import adjudicate_bundle
from fuxi_a1_adjudicate_20260926 import measured_pairs
from run_contract import FuxiWriterLock

OUTDIR = ROOT / "reports/fuxi_a1_source_audit_20260926"
ARMS = ("v2", "v3", "amount_top80")
SPECIAL = {
    ("300963.SZ", "2025-06-09"): {
        "code": "300963.SZ", "ex_date": "2025-06-09",
        "notice_date": "2025-05-30",
        "notice_url": "https://static.cninfo.com.cn/finalpage/2025-05-30/1223733028.PDF",
        "reason": "issuer_explicit_no_cash_pure_transfer",
        "holder_cash_pre_tax": 0.0, "holder_share_rate": 0.4,
        "reference_cash": 0.0, "reference_share_rate": 0.4,
    },
    ("002822.SZ", "2025-12-30"): {
        "code": "002822.SZ", "ex_date": "2025-12-30",
        "notice_date": "2025-12-23",
        "notice_url": "https://disc.static.szse.cn/disc/disk03/finalpage/2025-12-30/66a4c578-0e57-4445-8d44-9f85d24d6f6e.PDF",
        "reason": "restructuring_shares_not_distributed_to_existing_holders",
        "holder_cash_pre_tax": 0.0, "holder_share_rate": 0.0,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict:
    traces = {arm: OUTDIR / f"held_event_trace_{arm}_v5.json" for arm in ARMS}
    payloads = {arm: load(path) for arm, path in traces.items()}
    if any(p["expected_days"] != 320 or p["processed_days"] != 320 or
           p["economic_status"] != "data_insufficient" or
           p["book_assessment"]["status"] != "data_insufficient"
           for p in payloads.values()):
        raise RuntimeError("sensitivity trace incomplete or misclassified")
    by_key = {}
    for arm, payload in payloads.items():
        for item in payload["held_missing_action_keys"]:
            key = (item["code"], item["ex_date"])
            by_key.setdefault(key, []).append(arm)
    if not by_key:
        raise RuntimeError("nonempty held-event comparison required")
    panels = measured_pairs(set(by_key))
    events = []
    for (code, ex_date), arms in sorted(by_key.items(), key=lambda x: (x[0][1], x[0][0])):
        special = SPECIAL.get((code, ex_date))
        path = ROOT / "reports/window_evidence_20260923/raw" / (
            "dividend_" + ex_date.replace("-", "") + ".json")
        if not path.is_file():
            events.append({"code": code, "ex_date": ex_date, "arms": arms,
                           "status": "unresolved", "reason": "no_sealed_daily_response"})
            continue
        source = load(path)
        if source.get("possible_truncation") or source.get("status") != "response_received":
            events.append({"code": code, "ex_date": ex_date, "arms": arms,
                           "status": "unresolved", "reason": "daily_response_incomplete",
                           "source_sha256": digest(path)})
            continue
        rows = [row for row in source.get("records", [])
                if row.get("ts_code") == code and row.get("div_proc") == "实施"
                and row.get("ex_date") == ex_date.replace("-", "")]
        p = panels["daily_prices.csv"].get((code, ex_date))
        a = panels["adj_factor_daily.csv"].get((code, ex_date))
        if not rows or p is None or a is None or any(x != x for x in (*p, *a)):
            events.append({"code": code, "ex_date": ex_date, "arms": arms,
                           "status": "unresolved",
                           "reason": ("no_implemented_source_row" if not rows else
                                      "market_reference_unavailable"),
                           "source_sha256": digest(path), "source_rows": len(rows)})
            continue
        verdict = adjudicate_bundle(
            rows, code=code, ex_date=ex_date, pre_close=float(p[0]),
            adj_prev=float(a[0]), adj_ex=float(a[1]),
            issuer_exception=special,
            source_evidence=str(path.relative_to(ROOT)) + " sha256=" + digest(path),
        )
        events.append({"code": code, "ex_date": ex_date, "arms": arms,
                       "status": verdict.status, "reason": verdict.reason,
                       "source_rows": len(rows), "source_sha256": digest(path),
                       "issuer": special, "detail": verdict.detail,
                       "action": verdict.action})
    return {
        "schema": "fuxi_a1_held_extension_audit/v5",
        "scope": "24 union held-event keys from invalid sensitivity path; no production eligibility",
        "trace_input_sha256": {str(path.relative_to(ROOT)): digest(path)
                               for path in traces.values()},
        "implementation_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in (
                ROOT / "research/factor_alchemy/a1_corporate_action_adapter.py",
                ROOT / "research/factor_alchemy/corporate_action_inventory.py",
                ROOT / "data/raw/daily_prices.csv",
                ROOT / "data/raw/adj_factor_daily.csv",
            )
        },
        "union_keys": len(by_key), "events": events,
        "status_counts": dict(Counter(item["status"] for item in events)),
        "reason_counts": dict(Counter(item["reason"] for item in events)),
        "economic_status": "data_insufficient", "formal_eligibility": False,
        "limitation": "Floor, stale-frozen lifecycle positions and excluded actions make this a demand sensitivity only.",
    }


def main() -> None:
    output = OUTDIR / "held_extension_audit_v5.json"
    with FuxiWriterLock(run_id="a1_held_extension_audit"):
        if output.exists():
            raise RuntimeError("audit already sealed: " + str(output))
        result = audit()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("union_keys", "status_counts",
          "reason_counts", "economic_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
