"""Summarize sealed three-arm reference accounts without promoting economic labels."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from run_contract import FuxiWriterLock

SOURCE = ROOT / "reports/fuxi_a1_model_account_20260926_v4"
OUTPUT = SOURCE / "comparison.json"
ARMS = ("v3", "amount_top80", "v2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(values):
    return {"min": min(values), "max": max(values)}


def year_return(scenario, year):
    rows = scenario["daily_balances"]
    end = [row["nav"] for row in rows if row["date"].startswith(str(year))][-1]
    start = (10_000_000 if year == 2025 else
             [row["nav"] for row in rows if row["date"].startswith(str(year - 1))][-1])
    return end / start - 1


def main():
    with FuxiWriterLock(run_id="a1_model_account_summary"):
        if OUTPUT.exists():
            raise RuntimeError("comparison already sealed")
        paths = {arm: SOURCE / f"{arm}.json" for arm in ARMS}
        books = {arm: json.loads(path.read_text(encoding="utf-8"))
                 for arm, path in paths.items()}
        scenarios = {arm: {s["scenario"]: s for s in books[arm]["scenarios"]}
                     for arm in ARMS}
        names = sorted(scenarios[ARMS[0]])
        if len(names) != 8 or any(sorted(scenarios[arm]) != names for arm in ARMS):
            raise AssertionError("scenario identity mismatch")
        for arm in ARMS:
            book = books[arm]
            if book["expected_days"] != 320 or book["window"] != ["2025-01-02", "2026-04-30"]:
                raise AssertionError("frozen window changed")
            for name in names:
                item = scenarios[arm][name]
                calc = item["independent_calculation"]
                if (not item["complete_model_book"] or item["status"] != "model_complete"
                        or item["assessment"]["status"] != "book_complete_under_declared_assumptions"
                        or item["assessment"]["formal_eligibility"]
                        or calc["compared_days"] != 320
                        or len(calc["daily_parity"]) != 320
                        or not all(r["positions_equal"] and r["nav_gap"] <= 0.0001
                                   for r in calc["daily_parity"])
                        or calc["max_nav_abs_gap"] > 0.0001
                        or calc["independent_data_source"] or calc["broker_or_jq_parity"]):
                    raise AssertionError("incomplete or misclassified model scenario")
                dates = [r["date"] for r in item["daily_balances"]]
                if len(dates) != 320 or dates != sorted(set(dates)):
                    raise AssertionError("daily balance calendar mismatch")
        arm_bands = {
            arm: {
                "full_return": band([scenarios[arm][n]["net_return"] for n in names]),
                "year_2025": band([year_return(scenarios[arm][n], 2025) for n in names]),
                "year_2026_ytd": band([year_return(scenarios[arm][n], 2026) for n in names]),
                "max_daily_nav_replay_gap": max(scenarios[arm][n]["independent_calculation"]
                                                ["max_nav_abs_gap"] for n in names),
            } for arm in ARMS
        }
        pairs = {}
        for lhs, rhs in (("v3", "amount_top80"), ("amount_top80", "v2"), ("v3", "v2")):
            gaps = [scenarios[lhs][n]["net_return"] - scenarios[rhs][n]["net_return"]
                    for n in names]
            pairs[f"{lhs}_vs_{rhs}"] = {
                "matched_scenario_wins": sum(g > 0 for g in gaps),
                "paired_scenarios": len(gaps), "return_gap": band(gaps),
                "cross_scenario_conservative_gap": (
                    arm_bands[lhs]["full_return"]["min"] -
                    arm_bands[rhs]["full_return"]["max"]),
            }
        result = {
            "schema": "fuxi_a1_model_account_comparison/v1",
            "status": "conditional_exploratory_only",
            "window": ["2025-01-02", "2026-04-30"],
            "signal_window_declared": ["2025-01-01", "2026-04-30"],
            "days": 320, "initial_cash": 10_000_000,
            "cost_bps_each_side": 15,
            "scenario_grid": names,
            "arm_bands": arm_bands, "paired_comparisons": pairs,
            "full_window_rank_stable_in_declared_grid": all(
                scenarios["v3"][n]["net_return"] > scenarios["amount_top80"][n]["net_return"]
                > scenarios["v2"][n]["net_return"] for n in names),
            "formal_s5_or_jq_eligibility": False,
            "generator_method_winner": None,
            "external_account_or_jq_parity": False,
            "research_exposed_window": True,
            "limitations": [
                "same-source independent arithmetic replay, not broker/JQ parity",
                "flat immediate 0/20% dividend haircut is not holding-period tax settlement",
                "delist zero/last-mark are declared stress endpoints, not proven recovery bounds",
                "fractional floor/ceil endpoints do not enumerate every joint account allocation",
                "next-open fills and 15bps aggregate costs are execution assumptions",
                "three fixed portfolios do not identify superiority of a generator family",
            ],
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        }
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
        print(json.dumps({"status": result["status"],
                          "rank_stable": result["full_window_rank_stable_in_declared_grid"],
                          "pairs": result["paired_comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
