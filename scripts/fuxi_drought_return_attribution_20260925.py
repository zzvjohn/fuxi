"""Arithmetic attribution of the frozen 47-section model comparison.

This reads historical result rows only. It does not re-run a strategy, assert
causality, or treat correlated weekly sections as independent tests.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fuxi_execution_lab_20260923 import digest, write  # noqa: E402
from run_contract import RunContract  # noqa: E402

SOURCE = ROOT / "reports/model_compare_20260924/sections.json"
OUT = ROOT / "reports/fuxi_drought_execution_20260925"
ARMS = ("equal_rank", "ridge", "tree", "r4_equal_rank")


def mean(items):
    if not items:
        raise ValueError("empty comparison")
    return sum(items) / len(items)


def components(rows):
    selection = [r["gross_excess_vs_pool"] for r in rows]
    pool = [r["pool_return"] - r["benchmark"] for r in rows]
    cost = [-r["cost_15"] for r in rows]
    net = [r["net_excess_15"] for r in rows]
    for i, row in enumerate(rows):
        if not math.isclose(selection[i] + pool[i] + cost[i], net[i],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("return identity failed: " + row["arm"] + " " + row["stamp"])
    return {"n": len(rows), "mean_selection_vs_pool": mean(selection),
            "mean_pool_vs_benchmark": mean(pool), "mean_cost_15": mean(cost),
            "mean_net_vs_benchmark_15": mean(net),
            "net_positive_sections": sum(x > 0 for x in net),
            "mean_turnover_one_way": mean([r["one_way_turnover"] for r in rows]),
            "identity_max_abs_residual": max(abs(selection[i] + pool[i] + cost[i] - net[i])
                                             for i in range(len(rows)))}


def audit():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in data:
        if row["arm"] in ARMS and row["complete"]:
            grouped[row["arm"]].append(row)
    if set(grouped) != set(ARMS):
        raise ValueError("frozen arms changed")
    stamps = None
    detail = {}
    for arm in ARMS:
        rows = sorted(grouped[arm], key=lambda r: r["stamp"])
        if len(rows) != 47 or len({r["stamp"] for r in rows}) != 47:
            raise ValueError("frozen 47-section denominator changed: " + arm)
        current = [(r["stamp"], r["entry"], r["exit"]) for r in rows]
        if stamps is None:
            stamps = current
        elif current != stamps:
            raise ValueError("arms have different windows")
        detail[arm] = {"all": components(rows),
                       "by_year": {y: components([r for r in rows if r["stamp"].startswith(y)])
                                   for y in ("2025", "2026")}}
    return {
        "status": "historical_descriptive_only",
        "source_sha256": digest(SOURCE),
        "window": [stamps[0][0], stamps[-1][0]],
        "n_per_arm": 47,
        "cost_bps": 15,
        "identity": "net_excess_15 = gross_excess_vs_pool + (pool_return - benchmark) - cost_15",
        "units": "simple fractional return per section; multiply by 10000 for basis points",
        "arms": detail,
        "limits": [
            "Pool definitions differ by arm; compare components within an arm only.",
            "Weekly return components are arithmetic means, not compounded total returns.",
            "The historical label screens with future entry/exit availability; this is no proof of executable performance.",
            "All sections are research-exposed, not R5 unknown samples.",
            "This identity attributes arithmetic terms, not causal risk or sector exposures.",
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "return_attribution.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite return attribution")
    with RunContract(
        parameters={"frozen_sections_sha256": digest(SOURCE), "cost_bps": 15,
                    "caliber": "same-window return identity only"},
        run_id="attribution_" + datetime.now().strftime("%H%M%S_%f"),
        runs_dir=OUT / "runs", lock_path=OUT / "owner.lock",
        extra_code=[__file__], extra_data=[SOURCE],
    ) as contract:
        result = audit()
        write(target, result)
        contract.record_evaluation(result, [])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
