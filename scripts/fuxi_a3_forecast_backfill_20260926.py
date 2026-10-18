"""Bounded, immutable PIT forecast backfill. Does not touch forecast.csv.

The only permitted range in this batch is the missing interval observed in A3.
Every calendar day, including holidays, receives an explicit source response.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))
from credentials import get_tushare_api  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402

START = date(2024, 12, 1)
END = date(2025, 8, 6)
FIELDS = ("ts_code", "ann_date", "end_date", "type", "p_change_min", "p_change_max",
          "net_profit_min", "net_profit_max", "last_parent_net", "first_ann_date",
          "summary", "change_reason", "update_flag")
OUT = ROOT / "data" / "raw" / "forecast_pit_backfill_20260926.csv"
MANIFEST = ROOT / "data" / "raw" / "forecast_pit_backfill_20260926.manifest.json"
SOURCE = ROOT / "data" / "raw" / "forecast.csv"
ROW_CAP = 3500


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def days():
    day = START
    while day <= END:
        yield day.strftime("%Y%m%d")
        day += timedelta(days=1)


def fetch(pro, day):
    last = None
    for trial in range(3):
        try:
            return pro.forecast(ann_date=day)
        except Exception as exc:
            last = exc
            time.sleep(1.0 + trial)
    raise RuntimeError(f"forecast({day}) failed after 3 tries: {type(last).__name__}: {last}")


def serialise(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main():
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError("immutable staging output already exists")
    dates = list(days())
    pro = get_tushare_api()
    rows, per_day, duplicates = [], {}, {}
    source_before = digest(SOURCE.read_bytes())
    with FuxiWriterLock(run_id="a3_forecast_backfill_20260926"):
        for index, day in enumerate(dates, 1):
            frame = fetch(pro, day)
            if frame is None:
                raise RuntimeError(f"source returned None, not certified empty: {day}")
            if set(frame.columns) != set(FIELDS):
                raise RuntimeError(f"source schema changed on {day}: {list(frame.columns)}")
            if len(frame) >= ROW_CAP:
                raise RuntimeError(f"response at API row cap on {day}: {len(frame)}")
            records = []
            for record in frame.to_dict("records"):
                row = {field: "" if record[field] is None or str(record[field]) == "nan"
                       else str(record[field]) for field in FIELDS}
                if row["ann_date"] != day or not row["ts_code"] or not row["end_date"]:
                    raise RuntimeError(f"bad source key on {day}: {row['ts_code']}/{row['ann_date']}/{row['end_date']}")
                records.append(row)
            day_raw = json.dumps(records, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            per_day[day] = {"status": "answered", "rows": len(records),
                            "response_sha256": digest(day_raw)}
            rows.extend(records)
            if index % 25 == 0 or index == len(dates):
                print(f"{index}/{len(dates)} dates; {len(rows)} rows", flush=True)
            time.sleep(0.35)
        by_key = {}
        for row in rows:
            key = (row["ts_code"], row["end_date"], row["ann_date"])
            value = tuple(row[field] for field in FIELDS)
            if key in by_key and by_key[key] != value:
                raise RuntimeError(f"conflicting version key: {key}")
            if key in by_key:
                duplicates[key] = duplicates.get(key, 1) + 1
            by_key[key] = value
        payload = serialise(rows)
        manifest = {
            "schema": "fuxi_forecast_pit_backfill/v1",
            "status": "complete_daily_source_response_not_exhaustive_exchange_archive",
            "api": "Tushare forecast(ann_date=YYYYMMDD)",
            "range": [dates[0], dates[-1]],
            "days_requested": len(dates), "days_answered": len(per_day),
            "empty_response_days": sum(x["rows"] == 0 for x in per_day.values()),
            "source_row_cap": ROW_CAP, "rows": len(rows),
            "same_version_key_exact_duplicate_groups": len(duplicates),
            "fields": FIELDS, "per_day": per_day,
            "file": str(OUT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(payload),
            "original_forecast_sha256_before": source_before,
            "original_forecast_sha256_after": digest(SOURCE.read_bytes()),
            "code_sha256": digest(Path(__file__).read_bytes()),
        }
        if manifest["original_forecast_sha256_after"] != source_before:
            raise RuntimeError("original forecast changed during collection")
        if len(per_day) != len(dates):
            raise RuntimeError("missing day response")
        OUT.write_bytes(payload)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        if digest(OUT.read_bytes()) != manifest["sha256"]:
            raise RuntimeError("staging readback mismatch")
    print(json.dumps({k: manifest[k] for k in ("status", "days_answered", "rows",
                    "empty_response_days", "same_version_key_exact_duplicate_groups", "sha256")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
