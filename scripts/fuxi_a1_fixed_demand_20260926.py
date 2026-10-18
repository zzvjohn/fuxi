"""Freeze A1's three-arm local event demand; never infer missing events as absent.

The demand uses intended weekly weights, not actual filled holdings. Its event
rows are a bounded acquisition plan; they cannot certify a cash-book return.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/fuxi_a1_20260926"
FROZEN = ROOT / "reports/fuxi_quality_gate_20260925"
START, END = "2025-01-01", "2026-04-30"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    if path.exists():
        raise RuntimeError("frozen A1 output exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def build_baseline(signal_days):
    """Simple post-close top-amount Top80, next-open fill, no future selection."""
    import numpy as np
    import pandas as pd

    rows = []
    for part in pd.read_csv(ROOT / "data/raw/daily_prices.csv",
                            usecols=["ts_code", "trade_date", "close", "amount"],
                            dtype={"trade_date": str}, chunksize=500_000):
        part = part[part.trade_date.isin({day.replace("-", "") for day in signal_days})]
        if not part.empty:
            rows.append(part)
    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    if frame.duplicated(["date", "ts_code"]).any():
        raise RuntimeError("duplicate baseline price keys")
    frame = frame[np.isfinite(frame.amount) & (frame.amount > 0)
                  & np.isfinite(frame.close) & (frame.close > 0)]
    targets = {}
    for day in signal_days:
        group = frame[frame.date == day].sort_values(
            ["amount", "ts_code"], ascending=[False, True], kind="mergesort")
        if len(group) < 80:
            raise RuntimeError("baseline fewer than 80 priced names on " + day)
        targets[day] = {code: 1 / 80 for code in group.head(80).ts_code}
    return targets


def main():
    import numpy as np
    import pandas as pd

    if (OUT / "demand.json").exists() or (OUT / "baseline_targets.json").exists():
        raise RuntimeError("A1 demand was already frozen")
    inputs = {}
    arms = {}
    for name in ("v2", "v3"):
        path = FROZEN / ("targets_" + name + ".json")
        row = json.loads(path.read_text(encoding="utf-8"))
        arms[name] = row["targets"]
        inputs[str(path.relative_to(ROOT))] = digest(path)
    days = sorted(arms["v2"])
    if days != sorted(arms["v3"]) or len(days) != 69:
        raise RuntimeError("frozen candidate signal calendars differ")
    arms["amount_top80"] = build_baseline(days)
    price_path = ROOT / "data/raw/daily_prices.csv"
    adjust_path = ROOT / "data/raw/adj_factor_daily.csv"
    inputs[str(price_path.relative_to(ROOT))] = digest(price_path)
    inputs[str(adjust_path.relative_to(ROOT))] = digest(adjust_path)
    baseline = {
        "schema": "fuxi_a1_baseline_targets/v1", "window": [START, END],
        "signal_cutoff": "post_close", "earliest_fill": "next_session_open_proxy",
        "selection": "all locally priced names, descending same-day amount, ts_code tie",
        "weight": "equal Top80", "signal_dates": len(days),
        "target_union_codes": len({c for weights in arms["amount_top80"].values() for c in weights}),
        "targets": arms["amount_top80"], "source_sha256": inputs[str(price_path.relative_to(ROOT))],
        "formal_eligibility": False,
    }
    codes = {c for arm in arms.values() for weights in arm.values() if weights for c in weights}
    chunks = []
    for part in pd.read_csv(adjust_path,
                            usecols=["ts_code", "trade_date", "adj_factor"],
                            dtype={"trade_date": str}, chunksize=500_000):
        subset = part[part.ts_code.isin(codes)]
        if not subset.empty:
            chunks.append(subset)
    frame = pd.concat(chunks, ignore_index=True)
    frame["date"] = pd.to_datetime(frame.trade_date, format="mixed").dt.strftime("%Y-%m-%d")
    if frame.duplicated(["ts_code", "date"]).any():
        raise RuntimeError("duplicate adjustment keys")
    frame = frame.sort_values(["ts_code", "date"])
    frame["prior"] = frame.groupby("ts_code").adj_factor.shift()
    frame = frame[frame.date.between(START, END) & frame.prior.notna()
                  & frame.adj_factor.notna()
                  & ~np.isclose(frame.adj_factor, frame.prior, rtol=1e-7, atol=0)]

    known = set()
    for rel, key in (
        ("reports/execution_events_20260923/approved_actions.json", "actions"),
        ("reports/execution_followup_20260923/validation.json", "new_actions"),
        ("reports/fuxi_quality_gate_20260925/event_920718/validation.json", "action"),
        ("reports/fuxi_quality_gate_20260925/event_920405/validation.json", "action"),
    ):
        path = ROOT / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = [payload[key]] if key == "action" else payload[key]
        known.update((a["code"], a["ex_date"]) for a in actions)
        inputs[rel] = digest(path)
    per_arm = {}
    union = {}
    for arm, targets in arms.items():
        rows = []
        for row in frame.itertuples():
            prior_index = bisect.bisect_left(days, row.date) - 1
            if prior_index < 0 or row.ts_code not in targets[days[prior_index]]:
                continue
            key = (row.ts_code, row.date)
            entry = {"code": key[0], "ex_date": key[1],
                     "prior_signal_date": days[prior_index],
                     "local_evidence": "known_action" if key in known else "unresolved_adjustment_change",
                     "source": "adj_factor_daily.csv change; actual ownership unverified"}
            rows.append(entry)
            union.setdefault(key, set()).add(arm)
        per_arm[arm] = {
            "signal_dates": len(days),
            "target_union_codes": len({c for weights in targets.values() if weights for c in weights}),
            "planned_exposure_adjustment_changes": len(rows),
            "known_actions": sum(r["local_evidence"] == "known_action" for r in rows),
            "unresolved": sum(r["local_evidence"] != "known_action" for r in rows),
            "demand_rows": sorted(rows, key=lambda r: (r["ex_date"], r["code"])),
        }
    demand = {
        "schema": "fuxi_a1_fixed_demand/v1", "window": [START, END],
        "arms": per_arm,
        "union_request": [{"code": code, "ex_date": day,
                           "arms": sorted(arm_names),
                           "known_action": (code, day) in known}
                          for (code, day), arm_names in sorted(union.items(), key=lambda x: (x[0][1], x[0][0]))],
        "required_evidence_classes": [
            "implemented cash and share actions with record/ex/pay/listing dates and net entitlement",
            "supplier-complete positive and negative event coverage over executed holdings",
            "effective-date trading status, precise price limits and raw marks",
            "PIT listing, board, ST and delisting/cash-recovery lifecycle",
            "same-window index benchmark and independent execution/data reconciliation",
        ],
        "interpretation": "planned targets only; fills, actual holdings and formal economic labels remain unknown",
        "status": "data_insufficient", "formal_eligibility": False,
        "input_sha256": inputs,
    }
    write_once(OUT / "baseline_targets.json", baseline)
    write_once(OUT / "demand.json", demand)
    print(json.dumps({arm: {k: row[k] for k in (
        "target_union_codes", "planned_exposure_adjustment_changes", "unresolved")}
        for arm, row in per_arm.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
