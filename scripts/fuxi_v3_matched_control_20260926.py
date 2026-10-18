"""Freeze and construct a board/size matched control from v3's as-of priced pool.

Selection never reads post-signal returns. Economic interpretation requires a
separate complete fixed-share cash book for newly selected controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fuxi_v3_matched_control_20260926"
SOURCE = {
    "targets": ROOT / "reports/fuxi_quality_gate_20260925/targets_v3.json",
    "daily_prices": ROOT / "data/raw/daily_prices.csv",
    "daily_basic": ROOT / "data/raw/daily_basic.csv",
    "trade_calendar": ROOT / "data/raw/trade_calendar.csv",
    "matching_code": Path(__file__).resolve(),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lock_check():
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("Fuxi writer lock present")


def board(code):
    if code.endswith(".BJ"):
        return "Beijing"
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301")):
        return "ChiNext"
    return "Other_A"


def selected_rows(path, needed, fields):
    parts = []
    for chunk in pd.read_csv(path, usecols=["ts_code", "trade_date", *fields],
                             dtype={"ts_code": str, "trade_date": str}, chunksize=500_000):
        dates = pd.to_datetime(chunk.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
        mask = dates.isin(needed)
        if mask.any():
            item = chunk.loc[mask, ["ts_code", *fields]].copy()
            item["date"] = dates[mask].to_numpy()
            parts.append(item)
    return pd.concat(parts, ignore_index=True)


def prepare():
    lock_check()
    OUT.mkdir(exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        raise RuntimeError("matched control protocol already frozen")
    targets = json.loads(SOURCE["targets"].read_text(encoding="utf-8"))["targets"]
    if len(targets) != 69 or any(not v or len(v) != 80 for v in targets.values()):
        raise RuntimeError("v3 target shape changed")
    protocol = {
        "schema": "fuxi_v3_matched_control_protocol/v1",
        "sample": "research-exposed 69 weekly signals, 2025-01-03 to 2026-04-30",
        "selection_cutoff": "signal_day_post_close",
        "eligible_pool": "all same-day priced names with positive open/close, previous global-session close and positive same-day total_mv; no post-signal rows or survivorship list",
        "v3_arm": "frozen 80 equal-weight targets",
        "control_arm": "80 equal-weight non-v3 names, same signal-day board, globally minimum absolute log(total_mv) assignment within each board",
        "board_rule": "Beijing / STAR(688,689) / ChiNext(300,301) / Other_A",
        "size_caliper_max_ratio": 1.5,
        "tie_break": "lexical ts_code sorted before Hungarian assignment; no future-return optimization",
        "window": ["2025-01-02", "2026-04-30"],
        "first_stage_success": "69 complete matched dates, exact board count each date, all cap ratios <=1.5, no selected v3 name in control",
        "economic_gate": "same next-session open fixed-share 15bps cash book with complete rights/fill evidence; otherwise data_insufficient",
        "input_sha256": {name: sha(path) for name, path in SOURCE.items()},
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("protocol_sha256", sha(protocol_path))


def run():
    lock_check()
    protocol_path = OUT / "protocol.json"
    output_path = OUT / "matched_targets.json"
    if output_path.exists():
        raise RuntimeError("matched targets already sealed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "fuxi_v3_matched_control_protocol/v1":
        raise RuntimeError("unknown matched protocol")
    for name, path in SOURCE.items():
        if sha(path) != protocol["input_sha256"][name]:
            raise RuntimeError("frozen source changed: " + name)
    frozen = json.loads(SOURCE["targets"].read_text(encoding="utf-8"))["targets"]
    dates = sorted(frozen)
    calendar = pd.read_csv(SOURCE["trade_calendar"],
                           usecols=["cal_date", "is_open"], dtype=str)
    calendar = sorted(set(pd.to_datetime(calendar.loc[calendar.is_open == "1", "cal_date"],
                                               format="mixed").dt.strftime("%Y-%m-%d")))
    prev = {date: calendar[calendar.index(date) - 1] for date in dates}
    price = selected_rows(SOURCE["daily_prices"], set(dates) | set(prev.values()),
                          ["open", "close", "amount"])
    if price.duplicated(["date", "ts_code"]).any():
        raise RuntimeError("duplicate daily price key")
    for field in ("open", "close", "amount"):
        price[field] = pd.to_numeric(price[field], errors="coerce").astype("float32")
    basic = selected_rows(SOURCE["daily_basic"], set(dates), ["total_mv"])
    basic["total_mv"] = pd.to_numeric(basic.total_mv, errors="coerce")
    grouped = basic.groupby(["date", "ts_code"], sort=False).total_mv
    conflicted = grouped.nunique(dropna=False)
    conflicts = set(conflicted[conflicted > 1].index)
    basic = basic.drop_duplicates(["date", "ts_code"], keep="last")
    if conflicts:
        idx = pd.MultiIndex.from_frame(basic[["date", "ts_code"]])
        basic = basic[~idx.isin(conflicts)]
    caps = {(r.date, r.ts_code): r.total_mv for r in basic.itertuples()}
    priced = {d: frame.set_index("ts_code") for d, frame in price.groupby("date")}
    matched = {}
    diagnostics = []
    max_ratio = protocol["size_caliper_max_ratio"]
    for date in dates:
        today = priced[date]
        prior = priced[prev[date]][["close"]].rename(columns={"close": "previous_close"})
        pool = today.join(prior, how="left")
        pool["total_mv"] = [caps.get((date, code), np.nan) for code in pool.index]
        pool = pool[np.isfinite(pool.open) & (pool.open > 0)
                    & np.isfinite(pool.close) & (pool.close > 0)
                    & np.isfinite(pool.previous_close) & (pool.previous_close > 0)
                    & np.isfinite(pool.total_mv) & (pool.total_mv > 0)].copy()
        v3_codes = set(frozen[date])
        if not v3_codes <= set(pool.index):
            raise RuntimeError("frozen v3 target outside as-of eligible pool: " + date)
        pool["board"] = [board(code) for code in pool.index]
        chosen = []
        ratios = []
        per_board = {}
        for group in ("Beijing", "STAR", "ChiNext", "Other_A"):
            treated = sorted(c for c in v3_codes if board(c) == group)
            controls = sorted(c for c in pool.index if c not in v3_codes and board(c) == group)
            if len(controls) < len(treated):
                per_board[group] = {"treated": len(treated), "available_controls": len(controls),
                                    "matched": 0, "reason": "insufficient_same_board_pool"}
                continue
            if not treated:
                per_board[group] = {"treated": 0, "available_controls": len(controls), "matched": 0}
                continue
            treated_log = np.log(pool.loc[treated, "total_mv"].to_numpy(dtype=float))
            control_log = np.log(pool.loc[controls, "total_mv"].to_numpy(dtype=float))
            distance = abs(treated_log[:, None] - control_log[None, :])
            i, j = linear_sum_assignment(distance)
            picked = [controls[k] for k in j]
            distances = distance[i, j]
            chosen.extend(picked)
            ratios.extend(np.exp(distances).tolist())
            per_board[group] = {"treated": len(treated), "available_controls": len(controls),
                                "matched": len(picked), "max_cap_ratio": float(np.exp(distances.max())),
                                "median_cap_ratio": float(np.exp(np.median(distances)))}
        complete = len(chosen) == 80 and len(set(chosen)) == 80 and not (set(chosen) & v3_codes)
        within = complete and max(ratios) <= max_ratio
        diagnostics.append({"date": date, "eligible_pool": len(pool),
                            "matched": len(chosen), "complete": complete,
                            "within_caliper": within,
                            "max_cap_ratio": max(ratios) if ratios else None,
                            "median_cap_ratio": float(np.median(ratios)) if ratios else None,
                            "boards": per_board})
        matched[date] = {code: 1 / 80 for code in chosen} if within else None
    result = {
        "schema": "fuxi_v3_matched_targets/v1", "protocol_sha256": sha(protocol_path),
        "status": "matched_complete" if all(x["within_caliper"] for x in diagnostics)
                  else "data_insufficient",
        "matched_dates": sum(x["within_caliper"] for x in diagnostics),
        "expected_dates": len(dates), "daily_basic_conflicting_keys_excluded": len(conflicts),
        "max_cap_ratio": max(x["max_cap_ratio"] for x in diagnostics if x["max_cap_ratio"]),
        "median_date_cap_ratio": float(np.median([x["median_cap_ratio"] for x in diagnostics
                                                  if x["median_cap_ratio"] is not None])),
        "targets": matched, "diagnostics": diagnostics,
        "economic_status": "not_evaluated", "formal_eligibility": False,
    }
    lock_check()
    for name, path in SOURCE.items():
        if sha(path) != protocol["input_sha256"][name]:
            raise RuntimeError("source changed during match: " + name)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "matched_dates", "expected_dates",
                                             "daily_basic_conflicting_keys_excluded",
                                             "max_cap_ratio", "median_date_cap_ratio")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else run()
