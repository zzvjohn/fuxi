"""Adjudicate the baseline's first actual-held action beyond planned demand."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from a1_corporate_action_adapter import adjudicate_bundle
from fuxi_a1_adjudicate_20260926 import measured_pairs
from fuxi_a1_three_arm_strict_probe_20260926 import replay

CODE, EX = "000034.SZ", "2025-05-12"
RAW = ROOT / "reports/window_evidence_20260923/raw/dividend_20250512.json"
OUT = ROOT / "reports/fuxi_a1_source_audit_20260926/actual_holdings_extension_v1.json"
NOTICE = "https://paper.cnstock.com/html/2025-04/30/content_2062841.htm"


def main():
    if OUT.exists():
        raise RuntimeError("extension already sealed")
    response = json.loads(RAW.read_text(encoding="utf-8"))
    if response.get("possible_truncation") or response.get("status") != "response_received":
        raise RuntimeError("sealed response incomplete")
    rows = [r for r in response["records"] if r.get("ts_code") == CODE
            and r.get("ex_date") == "20250512" and r.get("div_proc") == "实施"]
    if len(rows) != 1:
        raise RuntimeError("extension implementation count changed")
    panels = measured_pairs({(CODE, EX)})
    prices = panels["daily_prices.csv"][(CODE, EX)]
    factors = panels["adj_factor_daily.csv"][(CODE, EX)]
    issuer = {"code": CODE, "ex_date": EX, "notice_date": "2025-04-30",
              "notice_url": NOTICE, "reason": "repurchased_shares_excluded",
              "holder_cash_pre_tax": .267, "holder_share_rate": 0.0,
              "reference_cash": .2646736, "reference_share_rate": 0.0}
    verdict = adjudicate_bundle(
        rows, code=CODE, ex_date=EX, pre_close=float(prices[0]),
        adj_prev=float(factors[0]), adj_ex=float(factors[1]),
        issuer_exception=issuer,
        source_evidence=str(RAW.relative_to(ROOT)) + " sha256=" + hashlib.sha256(RAW.read_bytes()).hexdigest())
    payload = {"schema": "fuxi_a1_actual_holding_extension/v1",
               "scope": "one actual-held baseline key beyond 229 planned actions",
               "code": CODE, "ex_date": EX,
               "status": verdict.status, "reason": verdict.reason,
               "detail": verdict.detail, "action": verdict.action,
               "issuer": issuer,
               "raw_response_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
               "economic_status": "data_insufficient", "formal_eligibility": False}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps({"code": CODE, "status": verdict.status,
                      "reason": verdict.reason}, ensure_ascii=False), flush=True)
    if verdict.status == "approved_for_gross_cash_stress_only":
        replay("amount_top80", extension=OUT)


if __name__ == "__main__":
    main()
