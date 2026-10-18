"""Read-only replay of A3 forecast-revision density after bounded PIT backfill.

Reproducing the archived pre-backfill counts is a required guard; no economic
returns are read. The optional output is a new audit artifact, never a source edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "reports" / "fuxi_a3_feasibility_20260926.json"
BACKFILL = ROOT / "data" / "raw" / "forecast_pit_backfill_20260926.csv"
MANIFEST = BACKFILL.with_suffix(".manifest.json")
ORIGINAL = ROOT / "data" / "raw" / "forecast.csv"
BLOCK = ROOT / "data" / "raw" / "block_trade_daily.csv"
PRICES = ROOT / "data" / "raw" / "daily_prices.csv"
CALENDAR = ROOT / "data" / "raw" / "trade_calendar.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision_events(frame, dates, calendar):
    df = frame.sort_values(["ts_code", "end_date", "ann_date"]).copy()
    low = pd.to_numeric(df.p_change_min, errors="coerce")
    high = pd.to_numeric(df.p_change_max, errors="coerce")
    df["mid"] = (low + high) / 2
    df["prev"] = df.groupby(["ts_code", "end_date"], dropna=False).mid.shift(1)
    df["prev_ann"] = df.groupby(["ts_code", "end_date"], dropna=False).ann_date.shift(1)
    changed = df.prev_ann.notna() & np.isfinite(df.mid) & np.isfinite(df.prev) & \
        (low <= high) & (df.mid != df.prev)
    revision = df[changed]
    event = {}
    active = set()
    for day in dates:
        position = calendar.index(day)
        cutoff = calendar[position - 1]
        oldest = calendar[max(0, position - 60)]
        names = set(revision[(revision.ann_date <= cutoff) &
                             (revision.ann_date >= oldest)].ts_code)
        event[day] = names
        active.update((day, code) for code in names)
    return event, {
        "rows": len(df),
        "code_periods": int(df.groupby(["ts_code", "end_date"], dropna=False).ngroups),
        "transitions": int(df.prev_ann.notna().sum()),
        "finite_changed_transitions": int(changed.sum()),
        "active_revision_name_dates": len(active),
    }


def selected_top(prior, dates):
    pieces = []
    for chunk in pd.read_csv(PRICES, usecols=["trade_date", "ts_code", "close", "amount"],
                             dtype={"trade_date": str, "ts_code": str}, chunksize=300000):
        chunk.trade_date = chunk.trade_date.str.replace("-", "", regex=False)
        pieces.append(chunk[chunk.trade_date.isin(dates)])
    prices = pd.concat(pieces, ignore_index=True)
    prices = prices[(prices.close > 0) & np.isfinite(prices.close)]
    top = []
    datetimes = [pd.Timestamp(day) for day in dates]
    for name, expected_hash in prior["protocol_signals"].items():
        path = ROOT / "reports" / "drought_20260923" / "signals" / (name + ".parquet")
        assert sha(path) == expected_hash, f"signal bytes changed: {name}"
        factor = pd.read_parquet(path, filters=[("trade_date", "in", datetimes)])
        factor.trade_date = factor.trade_date.dt.strftime("%Y%m%d")
        factor = factor[np.isfinite(factor.factor_value)]
        joined = factor.merge(prices, on=["trade_date", "ts_code"], how="inner")
        for day, cross in joined.groupby("trade_date"):
            picks = cross.sort_values(["factor_value", "amount", "ts_code"],
                                      ascending=[False, False, True]).head(80)
            top.append((name, day, set(picks.ts_code)))
    return top


def top_density(top, events):
    hits = [(arm, day, code) for arm, day, names in top
            for code in (names & events.get(day, set()))]
    unique = {(day, code) for _, day, code in hits}
    a1 = {(day, code) for day, code in unique if "20250101" <= day <= "20260430"}
    return {
        "slots_across_four_arms": len(hits),
        "sections_with_event": len({(arm, day) for arm, day, _ in hits}),
        "unique_name_dates": len(unique),
        "unique_codes": len({code for _, code in unique}),
        "unique_weeks": len({day for day, _ in unique}),
        "a1_calendar_overlap_name_dates": len(a1),
        "a1_calendar_overlap_unique_codes": len({code for _, code in a1}),
        "a1_calendar_overlap_unique_weeks": len({day for day, _ in a1}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert sha(BACKFILL) == manifest["sha256"]
    assert sha(ORIGINAL) == manifest["original_forecast_sha256_after"]
    assert len(manifest["per_day"]) == 249
    assert all(item["status"] == "answered" and item["rows"] < 3500
               for item in manifest["per_day"].values())
    assert sha(PRICES) == prior["source_hashes"]["data/raw/daily_prices.csv"]
    assert sha(CALENDAR) == prior["source_hashes"]["data/raw/trade_calendar.csv"]
    before = pd.read_csv(ORIGINAL, dtype=str)
    stage = pd.read_csv(BACKFILL, dtype=str)
    combined = pd.concat([stage, before], ignore_index=True)
    keys = ["ts_code", "end_date", "ann_date"]
    assert not combined.duplicated(keys).any(), "ambiguous PIT version key"
    dates = sorted({item["date"].replace("-", "") for item in prior["per_section"]})
    calendar_frame = pd.read_csv(CALENDAR, encoding="utf-8-sig", dtype=str,
                                 on_bad_lines="skip")
    calendar = sorted(set(calendar_frame[(calendar_frame.exchange == "SSE") &
                                         (calendar_frame.is_open == "1")].cal_date.str
                          .replace("-", "", regex=False)))
    old_events, old = revision_events(before, dates, calendar)
    new_events, new = revision_events(combined, dates, calendar)
    top = selected_top(prior, dates)
    old["top"] = top_density(top, old_events)
    new["top"] = top_density(top, new_events)
    archived = prior["density"]["forecast_revision"]
    assert old["active_revision_name_dates"] == prior["active_revision_name_dates"]
    assert old["top"]["slots_across_four_arms"] == archived["top_event_name_sections"]
    assert old["top"]["sections_with_event"] == archived["sections_with_top_event"]
    first = combined.groupby(["ts_code", "end_date"], dropna=False).ann_date.transform("min")
    first_meta = combined.first_ann_date.fillna("")
    block = pd.read_csv(BLOCK, dtype=str)
    buyer_contains = block.buyer.fillna("").str.contains("机构专用", regex=False)
    buyer_exact = block.buyer.fillna("").eq("机构专用")
    on_signals = block.trade_date.isin(dates)
    output = {
        "schema": "fuxi_a3_backfill_replay/v1",
        "status": "source_day_coverage_complete_economic_not_evaluated",
        "hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                   for path in (BACKFILL, MANIFEST, ORIGINAL, BLOCK, PRICES, CALENDAR,
                                Path(__file__), PRIOR)},
        "source_response": {
            "requested_days": len(manifest["per_day"]),
            "answered_days": sum(v["status"] == "answered"
                                 for v in manifest["per_day"].values()),
            "empty_days": sum(v["rows"] == 0 for v in manifest["per_day"].values()),
            "max_rows_per_day": max(v["rows"] for v in manifest["per_day"].values()),
            "max_rows_day": max(manifest["per_day"],
                                key=lambda day: manifest["per_day"][day]["rows"]),
            "source_row_cap": 3500,
            "original_bytes_unchanged": manifest["original_forecast_sha256_before"] ==
                                        manifest["original_forecast_sha256_after"],
            "overlap_spotcheck_20250807_api_local_keys": "3/3",
            "full_original_window_requery": False,
        },
        "versions": {
            "stage_rows": len(stage), "original_rows": len(before),
            "combined_rows": len(combined),
            "min_ann_date": combined.ann_date.min(),
            "max_ann_date": combined.ann_date.max(),
            "duplicate_code_period_ann_date": int(combined.duplicated(keys).sum()),
            "same_code_ann_date_extra_rows": int(combined.duplicated(
                ["ts_code", "ann_date"]).sum()),
            "first_ann_before_local_first_rows": int(((first_meta != "") &
                                                      (first_meta < first)).sum()),
            "first_ann_after_local_first_rows": int(((first_meta != "") &
                                                     (first_meta > first)).sum()),
        },
        "frozen_replay": {"signal_dates": len(dates), "sections": len(top),
                          "old": old, "combined": new},
        "institution_buyer_label": {
            "contains_rows": int(buyer_contains.sum()),
            "exact_rows": int(buyer_exact.sum()),
            "contains_but_not_exact_rows": int((buyer_contains & ~buyer_exact).sum()),
            "on_signal_dates_contains_name_days": len(block[on_signals & buyer_contains]
                                                       [["ts_code", "trade_date"]]
                                                       .drop_duplicates()),
            "on_signal_dates_exact_name_days": len(block[on_signals & buyer_exact]
                                                    [["ts_code", "trade_date"]]
                                                    .drop_duplicates()),
        },
        "economic_backtests": 0,
        "qualification": {"s5_new": 0, "jq_new": 0, "r5_unknown": 0},
    }
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
