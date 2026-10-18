"""Frozen four-arm, equal unique-fitness-budget search diagnostic.

Fitness is Forge's legacy ICIR proxy on research-exposed weekly data. This
cannot establish executable alpha, S5 qualification, or QEA superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from forge import FactorForge  # noqa: E402
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES  # noqa: E402
from forge.quantum_challenger import QuantumForgeChallenger  # noqa: E402
from fuxi_quantum_forge_challenge_20260925 import real_panel, run_pair  # noqa: E402

OUT = ROOT / "reports/fuxi_a4_20260926"
SOURCE = ("output/ap_batch/cache/weekly_prices.parquet",
          "output/ap_batch/cache/weekly_ohlc.parquet")
CODE = ("scripts/fuxi_a4_generator_challenge_20260926.py",
        "scripts/fuxi_quantum_forge_challenge_20260925.py",
        "research/factor_alchemy/forge/__init__.py",
        "research/factor_alchemy/forge/quantum_challenger.py",
        "research/factor_alchemy/forge/primitives.py")
SEEDS = (17, 41, 73, 101, 137)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class SameDecoderEDA(QuantumForgeChallenger):
    """Simple independent Bernoulli probability update toward island elite."""

    learning_rate = 0.15

    def _rotate_toward(self, island, elite):
        current = np.sin(self.angles[island]) ** 2
        updated = np.clip((1 - self.learning_rate) * current
                          + self.learning_rate * elite, 0.10, 0.90)
        self.angles[island] = np.arcsin(np.sqrt(updated))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol_v2.json"
    if path.exists():
        raise RuntimeError("A4 protocol already frozen")
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("Fuxi writer lock present")
    protocol = {
        "schema": "fuxi_a4_generator_protocol/v2",
        "supersedes": "protocol.json; v1 preflight failed on arm-key mismatch before any result was written",
        "purpose": "exploratory search-engineering and legacy proxy comparison only",
        "panel": "research-exposed weekly R3 panel; no economic validation",
        "window": ["2021-01-10", "2026-09-13"],
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE},
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "arms": ["forge", "quantum", "same_grammar_random", "same_decoder_eda"],
        "seeds": list(SEEDS), "paradigm": "动量反转",
        "fitness_mode": "legacy", "max_depth": 6, "max_complexity": 12,
        "pop_size": 8, "generations": 2, "workers": 1,
        "budget_unit": "actual unique fitness evaluations; Forge realized count sets paired budget per seed",
        "eda_learning_rate": SameDecoderEDA.learning_rate,
        "primary_engineering_metric": "valid_fitness_count / actual_unique_evaluations",
        "secondary_proxy_metric": "best legacy fitness; nonconfirmatory",
        "selection_gate": "economic comparison requires complete PIT fixed-share 15bps cash books, same-window paired baseline, independent execution parity, future time split and multiplicity plan",
        "stop_rule": "five fixed seeds, no adaptive extension after reading results",
        "s5_or_jq_eligibility": False,
    }
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print("A4 four-arm protocol frozen:", sha(path))


def run():
    protocol_path = OUT / "protocol_v2.json"
    output = OUT / "generator_challenge_v2.json"
    if output.exists():
        raise RuntimeError("A4 challenge result already exists")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "fuxi_a4_generator_protocol/v2":
        raise RuntimeError("unknown A4 protocol")
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("Fuxi writer lock present")
    for name, expected in {**protocol["source_sha256"], **protocol["code_sha256"]}.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("frozen input changed: " + name)
    data = real_panel()
    if list(data["close"].shape) != [291, 5774]:
        raise RuntimeError("unexpected weekly panel shape")
    rows = []
    for seed in protocol["seeds"]:
        trio = run_pair(data, seed=seed, paradigm=protocol["paradigm"],
                        pop_size=protocol["pop_size"],
                        generations=protocol["generations"])
        budget = trio["forge"]["unique_evaluated"]
        if budget <= 0 or not trio["same_unique_evaluation_budget"]:
            raise AssertionError("three-arm realized fitness budgets differ")
        forge = FactorForge(
            data=data, max_depth=protocol["max_depth"],
            max_complexity=protocol["max_complexity"], icir_threshold=0.1,
            forward_returns=data["fwd_return"], seed=seed,
            paradigm_profiles=MATURE_PARADIGM_PROFILES,
            fitness_mode=protocol["fitness_mode"], dim_prune="enforce")
        start = time.perf_counter()
        eda = SameDecoderEDA(
            forge, seed=seed, unique_budget=budget,
            population_size=protocol["pop_size"],
            target_paradigm=protocol["paradigm"],
            max_draw_multiplier=80, rotation=0.04).run()
        seconds = time.perf_counter() - start
        if eda["unique_evaluated"] != budget or not eda["budget_met"]:
            raise AssertionError("EDA did not meet paired unique budget")
        best = eda["candidates"][0]
        trio["same_decoder_eda"] = {
            "unique_evaluated": eda["unique_evaluated"],
            "draws": eda["draws"], "duplicates": eda["duplicates"],
            "pre_eval_rejected": eda["pre_eval_rejected"],
            "valid_fitness_count": sum(item["fitness"] > -999 for item in eda["candidates"]),
            "seconds": seconds, "best_expression": best["expression"],
            "best_fitness": best["fitness"],
            "update_batches": eda["rotation_batches"],
            "updates": eda["rotation_updates"],
        }
        trio["all_four_actual_unique_matched"] = all(
            trio[arm]["unique_evaluated"] == budget for arm in protocol["arms"])
        rows.append(trio)
    for name, expected in {**protocol["source_sha256"], **protocol["code_sha256"]}.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("input changed during four-arm challenge: " + name)
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("writer appeared during four-arm challenge")
    if not rows or not all(row["all_four_actual_unique_matched"] for row in rows):
        raise AssertionError("paired comparison count must be positive")
    metrics = {arm: {
        "valid_fitness_total": sum(row[arm]["valid_fitness_count"] for row in rows),
        "unique_total": sum(row[arm]["unique_evaluated"] for row in rows),
        "best_proxy_wins_over_forge": sum(
            row[arm]["best_fitness"] > row["forge"]["best_fitness"] for row in rows),
    } for arm in protocol["arms"]}
    result = {
        "schema": "fuxi_a4_generator_challenge/v2",
        "source_protocol_sha256": sha(protocol_path),
        "panel_shape": list(data["close"].shape),
        "seeds": list(SEEDS), "rows": rows, "metrics": metrics,
        "matched_seed_count": len(rows),
        "economic_status": "not_evaluated", "formal_eligibility": False,
        "claim": "four-arm legacy fitness engineering diagnostic; no economic winner",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"matched_seed_count": len(rows), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else run()
