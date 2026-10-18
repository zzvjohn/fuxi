"""Adjudicate only the frozen A1 demand against sealed rows and issuer exceptions.

This produces a gross-cash stress input, not an investor-tax or formal economic
label. Existing sealed A1 evidence is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from a1_corporate_action_adapter import adjudicate_bundle

DEMAND = ROOT / "reports/fuxi_a1_20260926/demand.json"
SOURCE_AUDIT = ROOT / "reports/fuxi_a1_source_audit_20260926/audit.json"
ISSUER = ROOT / "reports/fuxi_a1_source_audit_20260926/issuer_exceptions.json"
OUT = ROOT / "reports/fuxi_a1_source_audit_20260926/adjudication_v2.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measured_pairs(keys):
    import pandas as pd

    codes = {c for c, _d in keys}
    panels = {}
    for filename, value in (("daily_prices.csv", "close"),
                            ("adj_factor_daily.csv", "adj_factor")):
        rows = []
        for part in pd.read_csv(ROOT / "data/raw" / filename,
                                usecols=["ts_code", "trade_date", value],
                                dtype={"trade_date": str}, chunksize=500_000):
            part = part[part.ts_code.isin(codes)]
            if not part.empty:
                rows.append(part)
        frame = pd.concat(rows, ignore_index=True)
        frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
        if frame.duplicated(["ts_code", "date"]).any():
            raise RuntimeError("duplicate market key: " + filename)
        frame = frame.sort_values(["ts_code", "date"])
        frame["previous"] = frame.groupby("ts_code")[value].shift()
        panels[filename] = {(r.ts_code, r.date): (r.previous, getattr(r, value))
                            for r in frame.itertuples() if (r.ts_code, r.date) in keys}
    return panels


def adjudicate():
    demand, source, issuer = load(DEMAND), load(SOURCE_AUDIT), load(ISSUER)
    for name, expected in source["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("sealed A1 source changed: " + name)
    if not source["provider_negative_coverage"] or source["daily_responses_complete"] != 320:
        raise RuntimeError("supplier daily coverage is not complete")
    exceptions = {(x["code"], x["ex_date"]): x for x in issuer["events"]}
    if len(exceptions) != 10:
        raise RuntimeError("issuer exception key count changed")
    keys = {(r["code"], r["ex_date"]) for r in demand["union_request"]}
    if len(keys) != 229 or not set(exceptions).issubset(keys):
        raise RuntimeError("frozen A1 demand changed")
    panels = measured_pairs(keys)
    evidence = { (r["code"], r["ex_date"]): r for r in source["events"] }
    out = []
    for code, day in sorted(keys, key=lambda x: (x[1], x[0])):
        check = evidence[(code, day)]
        if not check["source_day_complete"]:
            raise RuntimeError("source day no longer complete")
        path = ROOT / "reports/window_evidence_20260923/raw" / check["source_file"]
        if digest(path) != check["raw_sha256"]:
            raise RuntimeError("sealed source response changed: " + str(path))
        raw = load(path)
        rows = [r for r in raw["records"] if r.get("ts_code") == code
                and r.get("div_proc") == "实施" and r.get("ex_date") == day.replace("-", "")]
        p, a = panels["daily_prices.csv"].get((code, day)), panels["adj_factor_daily.csv"].get((code, day))
        if p is None or a is None:
            out.append({"code": code, "ex_date": day, "status": "unresolved",
                        "reason": "missing_market_reference_pair"})
            continue
        import pandas as pd
        if any(not pd.notna(x) for x in (p[0], a[0], a[1])):
            out.append({"code": code, "ex_date": day, "status": "unresolved",
                        "reason": "nonfinite_market_reference_pair"})
            continue
        result = adjudicate_bundle(
            rows, code=code, ex_date=day, pre_close=float(p[0]),
            adj_prev=float(a[0]), adj_ex=float(a[1]),
            issuer_exception=exceptions.get((code, day)),
            source_evidence=check["source_file"] + " sha256=" + check["raw_sha256"],
        )
        out.append({"code": code, "ex_date": day, "status": result.status,
                    "reason": result.reason, "detail": result.detail,
                    "action": result.action, "source_sha256": check["raw_sha256"]})
    counts = dict(Counter(r["status"] for r in out))
    return {
        "schema": "fuxi_a1_frozen_adjudication/v2",
        "scope": "229 frozen planned events only; gross-cash engineering stress",
        "window": demand["window"], "counts": counts,
        "issuer_exceptions_used": len(exceptions), "events": out,
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p)
                          for p in (DEMAND, SOURCE_AUDIT, ISSUER,
                                    ROOT / "data/raw/daily_prices.csv",
                                    ROOT / "data/raw/adj_factor_daily.csv",
                                    ROOT / "research/factor_alchemy/corporate_action_inventory.py",
                                    ROOT / "research/factor_alchemy/a1_corporate_action_adapter.py")},
        "account_tax_model": "unspecified; gross-cash stress is not a net investor return",
        "economic_status": "data_insufficient", "formal_eligibility": False,
    }


def main():
    if OUT.exists():
        raise RuntimeError("A1 v2 adjudication is already sealed")
    result = adjudicate()
    if sum(result["counts"].values()) != 229:
        raise RuntimeError("not every frozen event was adjudicated")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
