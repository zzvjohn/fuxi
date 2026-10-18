"""Read-only inventory of adjustment changes in frozen candidate target sets.

Target membership is a planned weekly exposure proxy, not actual ledger
holdings. This inventory sizes the event-data problem; it cannot settle PnL.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fuxi_quality_gate_20260925/event_inventory.json"
START, END = "2025-01-01", "2026-04-30"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import pandas as pd

    if OUT.exists():
        raise RuntimeError("immutable inventory already exists")
    arms = {}
    inputs = {}
    for revision in ("v2", "v3"):
        path = ROOT / f"reports/fuxi_quality_gate_20260925/targets_{revision}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        targets = {day: set(weights or {}) for day, weights in row["targets"].items()}
        arms[revision] = (sorted(targets), targets)
        inputs[str(path.relative_to(ROOT))] = sha(path)
    codes = set().union(*(set().union(*targets.values()) for _, targets in arms.values()))
    source = ROOT / "data/raw/adj_factor_daily.csv"
    inputs[str(source.relative_to(ROOT))] = sha(source)
    chunks = []
    for part in pd.read_csv(source, usecols=["ts_code", "trade_date", "adj_factor"],
                            dtype={"trade_date": str}, chunksize=500_000):
        chunks.append(part[part.ts_code.isin(codes)])
    frame = pd.concat(chunks, ignore_index=True)
    frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    if frame.duplicated(["ts_code", "date"]).any():
        raise RuntimeError("duplicate adjustment keys")
    frame = frame.sort_values(["ts_code", "date"])
    frame["prior"] = frame.groupby("ts_code").adj_factor.shift()
    frame = frame[frame.date.between(START, END)
                  & frame.prior.notna() & frame.adj_factor.notna()
                  & ~np.isclose(frame.adj_factor, frame.prior, rtol=1e-7, atol=0)]

    known = set()
    for path, key in ((ROOT / "reports/execution_events_20260923/approved_actions.json", "actions"),
                      (ROOT / "reports/execution_followup_20260923/validation.json", "new_actions"),
                      (ROOT / "reports/fuxi_quality_gate_20260925/event_920718/validation.json", "action"),
                      (ROOT / "reports/fuxi_quality_gate_20260925/event_920405/validation.json", "action")):
        data = json.loads(path.read_text(encoding="utf-8"))
        actions = [data[key]] if key == "action" else data[key]
        known.update((a["code"], a["ex_date"]) for a in actions)
        inputs[str(path.relative_to(ROOT))] = sha(path)

    results = {}
    for revision, (days, targets) in arms.items():
        planned = []
        for row in frame.itertuples():
            pos = bisect.bisect_left(days, row.date) - 1
            if pos >= 0 and row.ts_code in targets[days[pos]]:
                planned.append((row.ts_code, row.date))
        known_planned = [key for key in planned if key in known]
        results[revision] = {
            "target_union_codes": len(set().union(*targets.values())),
            "adjustment_changes_in_target_union": int(frame.ts_code.isin(
                set().union(*targets.values())).sum()),
            "planned_weekly_exposure_change_events": len(planned),
            "planned_weekly_exposure_known_cash_events": len(known_planned),
            "planned_weekly_exposure_unresolved": len(planned) - len(known_planned),
            "planned_weekly_exposure_unresolved_examples": [
                {"code": code, "date": day} for code, day in planned if (code, day) not in known][:15],
        }
    result = {"schema": "fuxi_quality_event_inventory/v1",
              "window": {"start": START, "end": END},
              "interpretation": "planned weekly target exposure proxy, not executed holdings or formal coverage",
              "inputs": inputs, "results": results}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
