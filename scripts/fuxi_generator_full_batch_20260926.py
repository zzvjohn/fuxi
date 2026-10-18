"""Frozen full evaluated-candidate archive for four generator methods.

The shared weekly legacy fitness is an exposed research proxy, never an
executable account label or evidence of S5/JQ qualification.
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
from forge.expression import parse_expression  # noqa: E402
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES  # noqa: E402
from forge.quantum_challenger import QuantumForgeChallenger  # noqa: E402
from fuxi_a4_generator_challenge_20260926 import SameDecoderEDA  # noqa: E402
from fuxi_quantum_forge_challenge_20260925 import real_panel  # noqa: E402

OUT = ROOT / "reports/fuxi_generator_full_batch_20260926"
SEEDS = (19, 43, 79, 103, 139)
ARMS = ("forge", "quantum", "same_grammar_random", "same_decoder_eda")
INPUTS = (
    "output/ap_batch/cache/weekly_prices.parquet",
    "output/ap_batch/cache/weekly_ohlc.parquet",
    "research/factor_alchemy/forge/__init__.py",
    "research/factor_alchemy/forge/expression.py",
    "research/factor_alchemy/forge/primitives.py",
    "research/factor_alchemy/forge/quantum_challenger.py",
    "scripts/fuxi_a4_generator_challenge_20260926.py",
    "scripts/fuxi_quantum_forge_challenge_20260925.py",
    "scripts/fuxi_generator_full_batch_20260926.py",
    "reports/fuxi_v3_source_audit_20260926/pit.json",
    "reports/fuxi_v3_source_audit_20260926/attribution.json",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lock_check():
    if (ROOT / "data/fuxi_runtime.lock").exists():
        raise RuntimeError("Fuxi writer lock present")


def forge_for(data, seed):
    return FactorForge(
        data=data, max_depth=6, max_complexity=12.0, icir_threshold=0.1,
        forward_returns=data["fwd_return"], seed=seed,
        paradigm_profiles=MATURE_PARADIGM_PROFILES,
        fitness_mode="legacy", dim_prune="enforce")


def prepare():
    lock_check()
    pit = json.loads((ROOT / INPUTS[-2]).read_text(encoding="utf-8"))
    attribution = json.loads((ROOT / INPUTS[-1]).read_text(encoding="utf-8"))
    if pit["signal_dates"] != 69 or pit["exact_order_dates"] != 69:
        raise RuntimeError("v3 point-in-time gate failed")
    if attribution["paired_days"] != 320:
        raise RuntimeError("v3 accounting attribution gate failed")
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        raise RuntimeError("protocol already frozen")
    protocol = {
        "schema": "fuxi_generator_full_batch_protocol/v1",
        "purpose": "full evaluated-candidate engineering comparison after v3 PIT and account source audit",
        "arms": ARMS, "seeds": SEEDS,
        "panel_window": ["2021-01-10", "2026-09-13"],
        "panel_research_exposed": True, "panel_shape": [291, 5774],
        "paradigm": "动量反转", "fitness_mode": "legacy",
        "max_depth": 6, "max_complexity": 12.0,
        "pop_size": 8, "generations": 2, "workers": 1,
        "budget_rule": "Forge actual unique fitness evaluations set the paired unique budget per seed",
        "stop_rule": "all five fixed seeds; no adaptive extension or cherry-picking",
        "archive_rule": "all unique evaluated expressions and fitness per arm; all QEA/random/EDA draws with duplicate/pre-evaluation rejection statuses; Forge rejection counters only",
        "primary_metric": "valid fitness share at equal unique evaluation budget",
        "secondary_metrics": ["positive-fitness share", "per-seed best legacy fitness", "inter-arm overlap"],
        "economic_status": "not_evaluated",
        "economic_winner_eligible": False,
        "reason": "weekly legacy ICIR is not a PIT fixed-share cash-book trade label; exposed sample and incomplete independent platform parity",
        "inputs_sha256": {name: sha(ROOT / name) for name in INPUTS},
    }
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("protocol_sha256", sha(path))


def captured_challenger(cls, forge, *, seed, budget, rotation=None):
    kwargs = dict(seed=seed, unique_budget=budget, population_size=8,
                  target_paradigm="动量反转", max_draw_multiplier=80)
    if rotation is not None:
        kwargs["rotation"] = rotation
    engine = cls(forge, **kwargs)
    draws = []
    original = engine._decode
    def captured(tape):
        tree = original(tape)
        draws.append(tree.to_string())
        return tree
    engine._decode = captured
    started = time.perf_counter()
    result = engine.run()
    seconds = time.perf_counter() - started
    evaluated = {row["expression"] for row in result["candidates"]}
    seen = set()
    trace = []
    for expression in draws:
        status = "duplicate" if expression in seen else (
            "evaluated" if expression in evaluated else "pre_eval_rejected")
        trace.append({"expression": expression, "status": status})
        seen.add(expression)
    assert len(trace) == result["draws"]
    assert sum(x["status"] == "duplicate" for x in trace) == result["duplicates"]
    assert sum(x["status"] == "pre_eval_rejected" for x in trace) == result["pre_eval_rejected"]
    assert len(result["candidates"]) == budget == sum(x["status"] == "evaluated" for x in trace)
    return {"seconds": seconds, "draw_trace": trace, **result}


def run():
    lock_check()
    protocol_path = OUT / "protocol.json"
    output_path = OUT / "full_batch.json"
    if output_path.exists():
        raise RuntimeError("full batch already frozen")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["schema"] == "fuxi_generator_full_batch_protocol/v1"
    assert tuple(protocol["seeds"]) == SEEDS and tuple(protocol["arms"]) == ARMS
    for name, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("frozen input changed: " + name)
    data = real_panel()
    if list(data["close"].shape) != protocol["panel_shape"]:
        raise RuntimeError("panel shape changed")
    rows = []
    for seed in SEEDS:
        forge = forge_for(data, seed)
        started = time.perf_counter()
        best = forge.evolve(n_generations=2, pop_size=8, tournament_size=3,
                            elite_count=2, mutation_rate=0.3, crossover_rate=0.5,
                            workers=1, verbose=False, top_k=1, icir_top_n=0,
                            target_paradigm="动量反转")
        seconds = time.perf_counter() - started
        budget = len(forge.fitness_cache)
        if budget <= 0 or not best:
            raise AssertionError("Forge had zero evaluations")
        candidates = []
        for expression, fitness in forge.fitness_cache.items():
            tree = parse_expression(expression)
            candidates.append({
                "expression": expression,
                "pandas_expression": tree.to_pandas_string() if tree else None,
                "fitness": float(fitness) if math.isfinite(fitness) else -999.0,
                "complexity": float(tree.complexity) if tree else None,
                "root_op": tree.primitive.name if tree else None,
            })
        candidates.sort(key=lambda x: (-x["fitness"], x["expression"]))
        row = {"seed": seed, "unique_budget": budget,
               "forge": {"unique_evaluated": budget, "seconds": seconds,
                         "candidates": candidates,
                         "rejection_counters": {"fsa_blocks": forge.fsa_blocks,
                                                "dim_hard": forge.dim_hard,
                                                "sub_blocks": forge.sub_blocks}}}
        for arm, cls, rotation in (
            ("quantum", QuantumForgeChallenger, None),
            ("same_grammar_random", QuantumForgeChallenger, 0.0),
            ("same_decoder_eda", SameDecoderEDA, None),
        ):
            row[arm] = captured_challenger(cls, forge_for(data, seed),
                                           seed=seed, budget=budget, rotation=rotation)
        assert all(row[arm]["unique_evaluated"] == budget for arm in ARMS)
        rows.append(row)
        print(json.dumps({"seed": seed, "budget": budget,
                          "best": {a: row[a]["candidates"][0]["fitness"] for a in ARMS}},
                         ensure_ascii=False), flush=True)
    metrics = {}
    for arm in ARMS:
        all_rows = [c for row in rows for c in row[arm]["candidates"]]
        metrics[arm] = {
            "unique_evaluations": len(all_rows),
            "valid_fitness": sum(c["fitness"] > -999 for c in all_rows),
            "positive_fitness": sum(c["fitness"] > 0 for c in all_rows),
            "best_per_seed": [row[arm]["candidates"][0]["fitness"] for row in rows],
            "distinct_expressions": len({c["expression"] for c in all_rows}),
            "draws": sum(row[arm].get("draws", row[arm]["unique_evaluated"]) for row in rows),
            "pre_eval_rejected": sum(row[arm].get("pre_eval_rejected", 0) for row in rows),
        }
    result = {
        "schema": "fuxi_generator_full_batch/v1",
        "protocol_sha256": sha(protocol_path),
        "matched_seeds": len(rows), "rows": rows, "metrics": metrics,
        "economic_status": "not_evaluated", "formal_eligibility": False,
        "conclusion_scope": "exposed weekly legacy-fitness engineering comparison only",
    }
    for name, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("input changed during batch: " + name)
    lock_check()
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else run()
