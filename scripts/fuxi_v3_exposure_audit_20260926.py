"""Descriptive signal-date style/industry comparison for frozen v3 and Top80."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fuxi_v3_source_audit_20260926/exposure_v2.json"
V3 = ROOT / "reports/fuxi_quality_gate_20260925/targets_v3.json"
BASE = ROOT / "reports/fuxi_a1_20260926/baseline_targets.json"
DAILY = ROOT / "data/raw/daily_basic.csv"
INDUSTRY = ROOT / "data/raw/sw_index_member_all.csv"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def board(code):
    if code.endswith(".BJ"):
        return "Beijing"
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301")):
        return "ChiNext"
    return "Other_A"


def main():
    if (ROOT / "data/fuxi_runtime.lock").exists() or OUT.exists():
        raise RuntimeError("writer present or exposure output already sealed")
    arms = {"v3": json.loads(V3.read_text(encoding="utf-8"))["targets"],
            "amount_top80": json.loads(BASE.read_text(encoding="utf-8"))["targets"]}
    dates = sorted(arms["v3"])
    assert dates == sorted(arms["amount_top80"]) and len(dates) == 69
    pairs = {(d, c) for targets in arms.values() for d, weights in targets.items() for c in weights}
    frames = []
    for part in pd.read_csv(DAILY,
                            usecols=["ts_code", "trade_date", "total_mv", "circ_mv", "turnover_rate"],
                            dtype={"trade_date": str, "ts_code": str}, chunksize=500_000):
        part["date"] = pd.to_datetime(part.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
        part = part[part.date.isin(dates)]
        part = part[["ts_code", "date", "total_mv", "circ_mv", "turnover_rate"]]
        if len(part):
            frames.append(part)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily[daily.apply(lambda r: (r.date, r.ts_code) in pairs, axis=1)]
    duplicate_keys = int(daily.duplicated(["date", "ts_code"]).sum())
    conflicting_keys = 0
    if duplicate_keys:
        conflicting_keys = int((daily.groupby(["date", "ts_code"])[
            ["total_mv", "circ_mv", "turnover_rate"]].nunique(dropna=False) > 1).any(axis=1).sum())
        daily = daily.sort_values(["date", "ts_code"]).drop_duplicates(
            ["date", "ts_code"], keep="last")
    basic = {(r.date, r.ts_code): r for r in daily.itertuples()}
    membership = pd.read_csv(INDUSTRY, dtype=str).fillna("")
    membership = membership[membership.ts_code.isin({c for _, c in pairs})]
    members = defaultdict(list)
    for r in membership.itertuples():
        members[r.ts_code].append((r.in_date, r.out_date, r.l1_name))
    output = {}
    for arm, targets in arms.items():
        boards = Counter()
        industries = Counter()
        mv, circ, turnover = [], [], []
        miss_industry = miss_basic = 0
        for date, weights in targets.items():
            day = date.replace("-", "")
            assert weights and len(weights) == 80
            for code in weights:
                boards[board(code)] += 1
                matches = {name for start, end, name in members[code]
                           if start <= day and (not end or day < end)}
                if len(matches) == 1:
                    industries[next(iter(matches))] += 1
                else:
                    miss_industry += 1
                row = basic.get((date, code))
                if row is None:
                    miss_basic += 1
                else:
                    mv.append(float(row.total_mv))
                    circ.append(float(row.circ_mv))
                    turnover.append(float(row.turnover_rate))
        output[arm] = {
            "signal_slots": 69 * 80, "board_slots": dict(boards),
            "industry_slots_top10": industries.most_common(10),
            "industry_unresolved_slots": miss_industry,
            "daily_basic_missing_slots": miss_basic,
            "median_total_mv_10k_yuan": float(np.nanmedian(mv)),
            "median_circ_mv_10k_yuan": float(np.nanmedian(circ)),
            "median_turnover_rate_pct": float(np.nanmedian(turnover)),
        }
    overlap = [len(set(arms["v3"][d]) & set(arms["amount_top80"][d])) for d in dates]
    result = {"schema": "fuxi_v3_exposure_audit/v1",
              "sample": "69 already-exposed weekly post-close signals; 80 equal weights each",
              "arms": output, "mean_same_date_overlap": float(np.mean(overlap)),
              "min_same_date_overlap": min(overlap), "max_same_date_overlap": max(overlap),
              "selected_daily_basic_duplicate_rows_resolved_last": duplicate_keys,
              "selected_daily_basic_conflicting_keys": conflicting_keys,
              "risk": "descriptive source diagnosis, not PIT-neutralized alpha or causal attribution; historical membership effective dates may be vendor-restated",
              "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V3, BASE, DAILY, INDUSTRY)}}
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("writer appeared during exposure audit")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
