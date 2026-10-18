"""Read-only audit of future-label eligibility in the frozen model comparison.

This reproduces the old equal-rank arm on its 47 trained sections, then selects
the same score before asking whether entry/exit adjusted opens exist. No new
portfolio return is inferred when a selected stock lacks a future label.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))

import fuxi_model_compare_20260924 as old  # noqa: E402
from fuxi_execution_lab_20260923 import digest, write  # noqa: E402
from run_contract import RunContract  # noqa: E402

OUT = ROOT / "reports/fuxi_drought_execution_20260925"
OLD = ROOT / "reports/model_compare_20260924"


def audit():
    protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))
    # Raw daily panels and the calendar have since been updated. The frozen
    # snapshot/price caches supply signal-time features; exact old label-pool
    # counts and both returns must be reproduced for every section below.
    changed_raw_inputs = []
    for name, expected in protocol["hashes"].items():
        if digest(ROOT / name) == expected:
            continue
        if name in {"data/raw/trade_calendar.csv", "data/raw/daily_basic.csv",
                    "data/raw/daily_prices.csv"}:
            changed_raw_inputs.append(name)
        else:
            raise ValueError("frozen input changed: " + name)
    original = json.loads((OLD / "sections.json").read_text(encoding="utf-8"))
    rows = sorted((r for r in original if r["arm"] == "equal_rank" and r["complete"]),
                  key=lambda r: r["stamp"])
    if len(rows) != 47:
        raise ValueError("frozen equal-rank denominator changed")
    features = tuple((f["name"], f["kind"], f["direction"], f["source"])
                     for f in protocol["features"])
    names = [f[0] for f in features]
    dates, stamps, index, lag, close, amount, snapshot = old.load_cache()
    wanted = set()
    for row in rows:
        if old.label_dates(dates, row["stamp"]) != (row["entry"], row["exit"]):
            raise ValueError("calendar changed a frozen entry/exit window")
        wanted.update((row["entry"], row["exit"]))

    opens = old._chunked("data/raw/daily_prices.csv",
                         ["ts_code", "trade_date", "open"], wanted)
    adj = old._chunked("data/raw/adj_factor_daily.csv",
                       ["ts_code", "trade_date", "adj_factor"], wanted)
    for table, label in ((opens, "open"), (adj, "adjustment")):
        if table.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("duplicate " + label + " keys")
    open_wide = opens.pivot(index="trade_date", columns="ts_code", values="open")
    adj_wide = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor")

    readings = []
    for row in rows:
        stamp = row["stamp"]
        frame = old.features_at(dates, index, stamp, close, amount, snapshot,
                                lag.get(stamp), names)
        if frame is None:
            raise ValueError("nonempty frozen section became empty: " + stamp)
        label = (open_wide.loc[row["exit"]] * adj_wide.loc[row["exit"]] /
                 (open_wide.loc[row["entry"]] * adj_wide.loc[row["entry"]]) - 1)
        frame = frame.assign(label=frame.index.map(label))
        labelled = frame[np.isfinite(pd.to_numeric(frame.label, errors="coerce"))]
        if len(labelled) != row["pool"]:
            raise ValueError("frozen labelled pool mismatch on " + stamp)
        old_score = old.normalise(labelled, features)[names].mean(axis=1)
        old_top = old.top_selection(labelled, old_score)
        if not math.isclose(float(old_top.label.mean()), row["gross_return"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("frozen top return failed parity on " + stamp)
        if not math.isclose(float(labelled.label.mean()), row["pool_return"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("frozen pool return failed parity on " + stamp)

        pre_score = old.normalise(frame, features)[names].mean(axis=1)
        pre_top = old.top_selection(frame, pre_score)
        missing_top = pre_top[~np.isfinite(pd.to_numeric(pre_top.label, errors="coerce"))]
        old_set = set(old_top.ts_code)
        pre_set = set(pre_top.ts_code)
        readings.append({
            "stamp": stamp,
            "ex_ante_pool": int(len(frame)),
            "future_labelled_pool": int(len(labelled)),
            "excluded_by_future_label": int(len(frame) - len(labelled)),
            "ex_ante_top_label_missing": int(len(missing_top)),
            "ex_ante_top_missing_codes": sorted(missing_top.ts_code.tolist()),
            "old_top_vs_ex_ante_top_replaced": int(len(old_set - pre_set)),
            "old_top_vs_ex_ante_top_overlap": int(len(old_set & pre_set)),
            "old_gross_return_parity": True,
            "old_pool_return_parity": True,
        })

    return {
        "status": "historical_diagnostic_only",
        "raw_inputs_changed_since_frozen_run": changed_raw_inputs,
        "signal_features_from_frozen_cache": True,
        "window": [rows[0]["stamp"], rows[-1]["stamp"]],
        "label": protocol["label"],
        "cost_bps_primary": protocol["primary_cost_bps"],
        "sections": len(readings),
        "future_label_excluded_total": sum(r["excluded_by_future_label"] for r in readings),
        "sections_with_future_exclusions": sum(r["excluded_by_future_label"] > 0 for r in readings),
        "sections_with_missing_ex_ante_top": sum(r["ex_ante_top_label_missing"] > 0 for r in readings),
        "sections_with_changed_top": sum(r["old_top_vs_ex_ante_top_replaced"] > 0 for r in readings),
        "changed_top_names_total": sum(r["old_top_vs_ex_ante_top_replaced"] for r in readings),
        "old_return_parity_sections": len(readings),
        "readings": readings,
        "claims_not_made": [
            "The audit does not impute unavailable future prices or calculate an alternative return.",
            "A changed Top80 is evidence of selection dependence, not its performance direction.",
            "Ridge and tree require separately refitting their training sets; this audits equal rank only.",
            "All sections were already exposed to research and cannot serve as R5 unknown samples.",
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "label_pool_audit.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite existing label-pool audit")
    with RunContract(
        parameters={"frozen_protocol_sha256": digest(OLD / "protocol.json"),
                    "window": ["2025-08-08", "2026-08-28"],
                    "caliber": "signal-time equal rank versus future-label-screened equal rank"},
        run_id="label_pool_" + datetime.now().strftime("%H%M%S_%f"),
        runs_dir=OUT / "runs", lock_path=OUT / "owner.lock",
        extra_code=[__file__, ROOT / "scripts/fuxi_model_compare_20260924.py"],
        extra_data=[OLD / "protocol.json", OLD / "sections.json",
                    OLD / "cache/wide_close.parquet", OLD / "cache/wide_amount.parquet",
                    OLD / "cache/folded_snapshot.parquet",
                    ROOT / "data/raw/daily_prices.csv",
                    ROOT / "data/raw/adj_factor_daily.csv"],
    ) as contract:
        result = audit()
        write(target, result)
        contract.record_evaluation(result, [])
    print(json.dumps({k: v for k, v in result.items() if k != "readings"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
