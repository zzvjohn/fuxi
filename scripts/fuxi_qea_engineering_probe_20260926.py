"""Synthetic, read-only QEA decoder and budget diagnostic.

Fitness is a fixed hash of the formula, deliberately unrelated to returns.
This probes search mechanics only and cannot support an alpha claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from forge import FactorForge  # noqa: E402
from forge.primitives import NON_INPUT_PRIMITIVES  # noqa: E402
from forge.quantum_challenger import QuantumForgeChallenger  # noqa: E402


def run() -> dict:
    panel = {name: np.ones((24, 64)) for name in
             ("open", "high", "low", "close", "volume", "returns")}
    seeds = (17, 41, 73, 101, 137)
    rows = []
    all_ops = {primitive.name for primitive in NON_INPUT_PRIMITIVES}
    for seed in seeds:
        for label, rotation in (("qea", 0.04), ("same_decoder_random", 0.0)):
            forge = FactorForge(data=panel, max_depth=3, max_complexity=12,
                                seed=seed, dim_prune="off", fitness_mode="legacy")
            forge._fitness = lambda tree: int.from_bytes(
                hashlib.sha256(tree.to_string().encode("utf-8")).digest()[:4], "big"
            ) / (2 ** 32)
            result = QuantumForgeChallenger(
                forge, seed=seed, unique_budget=40, population_size=8,
                islands=4, rotation=rotation, max_draw_multiplier=30,
            ).run()
            roots = set(result["root_ops_covered"])
            rows.append({"seed": seed, "arm": label,
                         "budget_met": result["budget_met"],
                         "unique_evaluated": result["unique_evaluated"],
                         "draws": result["draws"],
                         "duplicates": result["duplicates"],
                         "pre_eval_rejected": result["pre_eval_rejected"],
                         "rotation_batches": result["rotation_batches"],
                         "rotation_updates": result["rotation_updates"],
                         "mean_bit_entropy": result["mean_bit_entropy"],
                         "root_ops_covered_n": len(roots),
                         "root_ops_covered": sorted(roots),
                         "valid_unique_per_draw": result["unique_evaluated"] / result["draws"]})
    return {"schema": "fuxi_qea_engineering_probe/v1",
            "purpose": "mechanical reachability, actual unique-evaluation budget and diversity",
            "fitness": "deterministic_sha256_formula_proxy_no_market_data",
            "source": "synthetic_ones_24x64_six_verified_terminals",
            "seeds": list(seeds), "unique_budget_per_arm_seed": 40,
            "all_root_ops_n": len(all_ops), "all_root_ops": sorted(all_ops),
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run(), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        destination = args.output.resolve()
        if ROOT.resolve() not in destination.parents or destination.exists():
            raise ValueError("output must be a new file inside the workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload.encode("utf-8"))
        print(destination)
