"""Freeze all full-batch candidate identities and historical account target schedules.

This is an exposed-sample signal construction step, not an economic label.
No future return enters selection; 2021-2024 weekly IC fixes sign beforehand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from forge.expression import ExpressionEvaluator, parse_expression  # noqa: E402
from fuxi_quantum_forge_challenge_20260925 import real_panel  # noqa: E402
from fuxi_v3_matched_control_v3_20260926 import selected_rows  # noqa: E402
from run_contract import FuxiWriterLock  # noqa: E402

OUT = ROOT / "reports/fuxi_full_batch_account_targets_20260926_v2"
INPUTS = {
    "batch": ROOT / "reports/fuxi_generator_full_batch_20260926/full_batch.json",
    "matched": ROOT / "reports/fuxi_v3_matched_control_20260926_v3/matched_targets.json",
    "weekly_prices": ROOT / "output/ap_batch/cache/weekly_prices.parquet",
    "weekly_ohlc": ROOT / "output/ap_batch/cache/weekly_ohlc.parquet",
    "daily_prices": ROOT / "data/raw/daily_prices.csv",
    "daily_basic": ROOT / "data/raw/daily_basic.csv",
    "calendar": ROOT / "data/raw/trade_calendar.csv",
    "forge": ROOT / "research/factor_alchemy/forge/expression.py",
    "loader": ROOT / "research/factor_alchemy/ralph_loop.py",
    "code": Path(__file__).resolve(),
}
ARMS = ("forge", "quantum", "same_grammar_random", "same_decoder_eda")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    with FuxiWriterLock(run_id="full_batch_account_targets_prepare"):
        OUT.mkdir(exist_ok=True)
        path = OUT / "protocol.json"
        if path.exists():
            raise RuntimeError("candidate target protocol already frozen")
        batch = json.loads(INPUTS["batch"].read_text(encoding="utf-8"))
        matched = json.loads(INPUTS["matched"].read_text(encoding="utf-8"))
        assert batch["schema"] == "fuxi_generator_full_batch/v1"
        assert matched["status"] == "matched_complete" and matched["matched_dates"] == 69
        dump(path, {
            "schema": "fuxi_full_batch_account_target_protocol/v2",
            "supersedes": "v1 target run discarded because ralph_loop.py changed during computation; no target output was sealed",
            "purpose": "all 252 archived candidate rows; 144 distinct formulas; exposed historical signal schedules only",
            "sample": "2025-01-02 to 2026-04-30, same 69 signal dates and same-day PIT priced pool as matched v3",
            "weekly_clock": "Sunday cache label maps to the immediately preceding signal-day week; OHLC and close may be used only after that week's last session close",
            "direction": "mean weekly cross-sectional Spearman IC sign using only weekly sections <=2024-12-22; zero/insufficient IC means no target",
            "selection": "signed factor value descending, lexical code tie break; top 80 in same signal-day eligible pool; each weight 1/80",
            "sentinel": "fitness<=-999 retained in roster as proxy_invalid; never silently dropped or credited with economic label",
            "account_model": "subsequent separate frozen 15bps each-side, next-session-open, fixed-share cash book with rights and independent arithmetic replay",
            "economic_status": "not_evaluated",
            "formal_eligibility": False,
            "input_sha256": {k: sha(v) for k, v in INPUTS.items()},
        })
    print("protocol_sha256", sha(path))


def eligible_pools(signal_dates):
    cal = pd.read_csv(INPUTS["calendar"], usecols=["cal_date", "is_open"], dtype=str)
    days = sorted(set(pd.to_datetime(cal.loc[cal.is_open == "1", "cal_date"],
                                          format="mixed").dt.strftime("%Y-%m-%d")))
    previous = {d: days[days.index(d) - 1] for d in signal_dates}
    price = selected_rows(INPUTS["daily_prices"], set(signal_dates) | set(previous.values()),
                          ["open", "close", "amount"])
    if price.duplicated(["date", "ts_code"]).any():
        raise RuntimeError("duplicate daily price key")
    basic = selected_rows(INPUTS["daily_basic"], set(signal_dates), ["total_mv"])
    basic["total_mv"] = pd.to_numeric(basic.total_mv, errors="coerce")
    grouped = basic.groupby(["date", "ts_code"], sort=False).total_mv
    conflicts = grouped.nunique(dropna=False)
    bad = set(conflicts[conflicts > 1].index)
    basic = basic.drop_duplicates(["date", "ts_code"], keep="last")
    if bad:
        idx = pd.MultiIndex.from_frame(basic[["date", "ts_code"]])
        basic = basic[~idx.isin(bad)]
    basic = basic.set_index(["date", "ts_code"]).total_mv
    priced = {d: f.set_index("ts_code") for d, f in price.groupby("date")}
    result = {}
    for day in signal_dates:
        today = priced[day].join(priced[previous[day]][["close"]].rename(
            columns={"close": "previous_close"}), how="left")
        today["total_mv"] = [basic.get((day, c), np.nan) for c in today.index]
        for col in ("open", "close", "previous_close", "total_mv"):
            today[col] = pd.to_numeric(today[col], errors="coerce")
        good = np.logical_and.reduce([np.isfinite(today[c]) & (today[c] > 0)
                                      for c in ("open", "close", "previous_close", "total_mv")])
        result[day] = set(today.index[good])
    return result, len(bad)


def weekly_ic_direction(values, fwd, train_rows):
    ics = []
    for i in train_rows:
        a, b = values[i], fwd[i]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 30:
            continue
        ar, br = rankdata(a[ok]), rankdata(b[ok])
        if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
            continue
        ic = float(np.corrcoef(ar, br)[0, 1])
        if math.isfinite(ic):
            ics.append(ic)
    mean_ic = float(np.mean(ics)) if ics else None
    direction = 1 if mean_ic is not None and mean_ic > 0 else (
        -1 if mean_ic is not None and mean_ic < 0 else None)
    return direction, mean_ic, len(ics)


def run():
    with FuxiWriterLock(run_id="full_batch_account_targets_run"):
        protocol_path = OUT / "protocol.json"
        output_path = OUT / "targets.json"
        if output_path.exists():
            raise RuntimeError("candidate target output already frozen")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert protocol["schema"] == "fuxi_full_batch_account_target_protocol/v2"
        for name, path in INPUTS.items():
            if sha(path) != protocol["input_sha256"][name]:
                raise RuntimeError("frozen input changed: " + name)
        batch = json.loads(INPUTS["batch"].read_text(encoding="utf-8"))
        matched = json.loads(INPUTS["matched"].read_text(encoding="utf-8"))
        signals = sorted(matched["control_targets"])
        pools, conflicts = eligible_pools(signals)
        weekly = pd.read_parquet(INPUTS["weekly_prices"], columns=["trade_date", "stock_code", "close"])
        pivot = weekly.pivot_table(index="trade_date", columns="stock_code", values="close", aggfunc="last")
        index = [pd.Timestamp(x).date().isoformat() for x in pivot.index]
        codes = list(pivot.columns)
        week_rows = {}
        for signal in signals:
            d = date.fromisoformat(signal)
            sunday = (d + timedelta(days=(6 - d.weekday()) % 7)).isoformat()
            if sunday not in index:
                raise RuntimeError("weekly panel missing mapped signal: " + signal)
            week_rows[signal] = index.index(sunday)
        data = real_panel()
        if data["close"].shape != pivot.shape or not np.allclose(
                data["close"], pivot.to_numpy(), equal_nan=True):
            raise RuntimeError("weekly row/column mapping differs from Forge loader")
        train_rows = [i for i, d in enumerate(index) if d <= "2024-12-22"]
        if len(train_rows) < 100:
            raise RuntimeError("insufficient pre-account-window training sections")
        roster = []
        distinct = {}
        for row in batch["rows"]:
            for arm in ARMS:
                for ordinal, candidate in enumerate(row[arm]["candidates"]):
                    expr = candidate["expression"]
                    cid = f"{arm}:s{row['seed']}:e{ordinal:02d}"
                    roster.append({"candidate_id": cid, "arm": arm, "seed": row["seed"],
                                   "ordinal": ordinal, "expression": expr,
                                   "expression_sha256": hashlib.sha256(expr.encode()).hexdigest(),
                                   "legacy_fitness": candidate["fitness"],
                                   "proxy_status": "invalid_sentinel" if candidate["fitness"] <= -999
                                   else "finite_proxy"})
                    distinct.setdefault(expr, candidate["fitness"])
        if len(roster) != 252 or len(distinct) != 144:
            raise RuntimeError("frozen full batch cardinality changed")
        evaluator = ExpressionEvaluator(data)
        targets = {}
        statuses = {}
        for count, (expr, fitness) in enumerate(distinct.items(), 1):
            expr_hash = hashlib.sha256(expr.encode()).hexdigest()
            if fitness <= -999:
                statuses[expr_hash] = {"status": "proxy_invalid", "targets": 0}
                continue
            try:
                values = evaluator.evaluate(parse_expression(expr))
                direction, train_ic, sections = weekly_ic_direction(
                    values, data["fwd_return"], train_rows)
                schedule = {}
                if direction is not None:
                    for signal in signals:
                        i = week_rows[signal]
                        eligible = pools[signal]
                        choices = [(direction * float(v), code) for code, v in zip(codes, values[i])
                                   if code in eligible and math.isfinite(float(v))]
                        choices.sort(key=lambda x: (-x[0], x[1]))
                        if len(choices) < 80:
                            raise ValueError(f"coverage_below_80:{signal}:{len(choices)}")
                        schedule[signal] = {c: 1 / 80 for _, c in choices[:80]}
                if len(schedule) != 69:
                    raise ValueError("direction_or_schedule_missing")
                targets[expr_hash] = schedule
                statuses[expr_hash] = {"status": "targets_frozen", "direction": direction,
                                       "train_mean_ic": train_ic, "train_ic_sections": sections,
                                       "targets": len(schedule)}
            except Exception as exc:
                statuses[expr_hash] = {"status": "target_ineligible", "reason": str(exc)[:250],
                                       "targets": 0}
            if count % 20 == 0:
                print(json.dumps({"expressions_processed": count, "total": len(distinct)}), flush=True)
        result = {"schema": "fuxi_full_batch_account_targets/v2",
                  "protocol_sha256": sha(protocol_path), "candidate_rows": roster,
                  "expression_status": statuses, "targets_by_expression_sha256": targets,
                  "signal_dates": signals, "training_weekly_sections": len(train_rows),
                  "eligible_pool_counts": {d: len(x) for d, x in pools.items()},
                  "daily_basic_conflicting_keys_excluded": conflicts,
                  "economic_status": "not_evaluated", "formal_eligibility": False}
        for name, path in INPUTS.items():
            if sha(path) != protocol["input_sha256"][name]:
                raise RuntimeError("input changed during target construction: " + name)
        dump(output_path, result)
    print(json.dumps({"rows": len(roster), "distinct": len(distinct),
                      "targets_frozen": len(targets), "statuses": {
                          status: sum(x["status"] == status for x in statuses.values())
                          for status in set(x["status"] for x in statuses.values())}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    prepare() if parser.parse_args().action == "prepare" else run()
