"""Read-only, paired Forge versus classical quantum-inspired search pilot.

The shared fitness is an ICIR proxy.  These results are search diagnostics,
never S5, cash-book economics, JQ eligibility, or unknown-sample evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

from forge import FactorForge  # noqa: E402
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES  # noqa: E402
from forge.quantum_challenger import QuantumForgeChallenger  # noqa: E402
from ralph_loop import RalphLoop  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while part := handle.read(4 * 1024 * 1024):
            digest.update(part)
    return digest.hexdigest()


def real_panel() -> dict:
    loader = object.__new__(RalphLoop)
    loader.data_dir = ROOT / "data"
    panel = loader._load_forge_data()
    if not panel or "fwd_return" not in panel:
        raise RuntimeError("verified Forge weekly panel/forward label unavailable")
    return panel


def smoke_panel() -> dict:
    rng = np.random.RandomState(2909)
    returns = rng.normal(0, 0.025, (42, 90))
    close = 20 * np.cumprod(1 + returns, axis=0)
    open_ = close * (1 + rng.normal(0, 0.008, close.shape))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.015, close.shape))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.015, close.shape))
    volume = rng.lognormal(10, 0.8, close.shape)
    fwd = np.roll(returns, -1, axis=0)
    fwd[-1] = np.nan
    return {"open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
            "returns": returns, "fwd_return": fwd}


def run_pair(data: dict, *, seed: int, paradigm: str, pop_size: int,
             generations: int) -> dict:
    common = dict(data=data, max_depth=6, max_complexity=12.0,
                  icir_threshold=0.1, forward_returns=data["fwd_return"],
                  seed=seed, paradigm_profiles=MATURE_PARADIGM_PROFILES,
                  fitness_mode="legacy", dim_prune="enforce")
    forge = FactorForge(**common)
    started = time.perf_counter()
    best = forge.evolve(n_generations=generations, pop_size=pop_size,
                        tournament_size=3, elite_count=2, mutation_rate=0.3,
                        crossover_rate=0.5, workers=1, verbose=False,
                        top_k=1, icir_top_n=0, target_paradigm=paradigm)
    forge_seconds = time.perf_counter() - started
    actual_unique = len(forge.fitness_cache)
    if actual_unique <= 0 or not best:
        raise AssertionError("Forge arm has no actual unique evaluated candidates")
    quantum_forge = FactorForge(**common)
    started = time.perf_counter()
    quantum = QuantumForgeChallenger(
        quantum_forge, seed=seed, unique_budget=actual_unique,
        population_size=pop_size, target_paradigm=paradigm,
        max_draw_multiplier=80,
    ).run()
    quantum_seconds = time.perf_counter() - started
    random_forge = FactorForge(**common)
    started = time.perf_counter()
    random_control = QuantumForgeChallenger(
        random_forge, seed=seed, unique_budget=actual_unique,
        population_size=pop_size, target_paradigm=paradigm,
        rotation=0.0, max_draw_multiplier=80,
    ).run()
    random_seconds = time.perf_counter() - started
    forge_best = max(best, key=lambda row: row["fitness"])
    qbest = quantum["candidates"][0] if quantum["candidates"] else None
    rbest = random_control["candidates"][0] if random_control["candidates"] else None
    return {
        "seed": seed,
        "comparison_nonempty": (actual_unique > 0 and quantum["unique_evaluated"] > 0
                                and random_control["unique_evaluated"] > 0),
        "same_unique_evaluation_budget": (
            quantum["unique_evaluated"] == actual_unique ==
            random_control["unique_evaluated"]),
        "forge": {"unique_evaluated": actual_unique,
                  "valid_fitness_count": sum(v > -999 for v in forge.fitness_cache.values()),
                  "seconds": forge_seconds,
                  "best_expression": forge_best["expression"],
                  "best_fitness": float(forge_best["fitness"])},
        "quantum": {"unique_evaluated": quantum["unique_evaluated"],
                    "draws": quantum["draws"],
                    "duplicates": quantum["duplicates"],
                    "pre_eval_rejected": quantum["pre_eval_rejected"],
                    "valid_fitness_count": sum(
                        row["fitness"] > -999 for row in quantum["candidates"]),
                    "seconds": quantum_seconds,
                    "best_expression": qbest["expression"] if qbest else None,
                    "best_fitness": qbest["fitness"] if qbest else None},
        "same_grammar_random": {
            "unique_evaluated": random_control["unique_evaluated"],
            "draws": random_control["draws"],
            "duplicates": random_control["duplicates"],
            "pre_eval_rejected": random_control["pre_eval_rejected"],
            "valid_fitness_count": sum(
                row["fitness"] > -999 for row in random_control["candidates"]),
            "seconds": random_seconds,
            "best_expression": rbest["expression"] if rbest else None,
            "best_fitness": rbest["fitness"] if rbest else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("smoke", "real"), default="smoke")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--paradigm", default="动量反转")
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=2)
    args = parser.parse_args()
    if args.paradigm not in MATURE_PARADIGM_PROFILES:
        raise ValueError("unknown Forge paradigm")
    if args.pop_size < 4 or args.generations < 1:
        raise ValueError("pilot needs pop_size >= 4 and generations >= 1")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique")
    lock = ROOT / "data" / "fuxi_runtime.lock"
    if lock.exists():
        raise RuntimeError("Fuxi writer lock exists; paired readout refused")
    source_files = []
    if args.panel == "real":
        source_files = [ROOT / "output/ap_batch/cache/weekly_prices.parquet",
                        ROOT / "output/ap_batch/cache/weekly_ohlc.parquet"]
        if any(not source.exists() for source in source_files):
            raise RuntimeError("frozen real-panel source is missing")
    before = {source.relative_to(ROOT).as_posix(): sha256_file(source)
              for source in source_files}
    data = real_panel() if args.panel == "real" else smoke_panel()
    rows = [run_pair(data, seed=seed, paradigm=args.paradigm,
                     pop_size=args.pop_size, generations=args.generations)
            for seed in seeds]
    after = {source.relative_to(ROOT).as_posix(): sha256_file(source)
             for source in source_files}
    if lock.exists() or before != after:
        raise RuntimeError("writer/data state changed during paired readout")
    if not all(row["comparison_nonempty"] for row in rows):
        raise AssertionError("paired comparison count must be positive")
    result = {
        "schema": "fuxi_quantum_forge_challenge/v1",
        "panel": args.panel,
        "panel_shape": list(data["close"].shape),
        "source_sha256": before,
        "code_sha256": {
            name: sha256_file(ROOT / name) for name in (
                "scripts/fuxi_quantum_forge_challenge_20260925.py",
                "research/factor_alchemy/forge/__init__.py",
                "research/factor_alchemy/forge/quantum_challenger.py",
                "research/factor_alchemy/forge/primitives.py",
            )
        },
        "paradigm": args.paradigm, "fitness_mode": "legacy",
        "pop_size": args.pop_size, "generations": args.generations,
        "seeds": seeds, "rows": rows,
        "all_unique_budgets_matched": all(
            row["same_unique_evaluation_budget"] for row in rows),
        "economic_status": "not_evaluated",
        "s5_new": 0, "jq_new": 0,
        "claim": "ICIR search pilot only; no economic or R5 promotion inference",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    os.environ["FORGE_WORKERS"] = "1"
    main()
