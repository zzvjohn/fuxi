"""Read-only evidence probes for the 2026-09-26 Fuxi iteration roadmap.

All deliberately malformed/negative inputs are synthetic and in memory.
Production save methods and evidence file reads are replaced locally. No
market simulation, network call, candidate registration or state write runs.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "research/factor_alchemy"
sys.path.insert(0, str(FA))
from mechanism_search import EVIDENCE_SCHEMA, load_catalog, load_evidence, select_plan
from multiple_testing_guard import MultipleTestingGuard
from forge.expression import ExprNode
from forge.primitives import PRIMITIVE_BY_NAME
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES
from forge.quantum_challenger import QuantumForgeChallenger

SOURCES = [
    "research/factor_alchemy/mechanism_search.py",
    "research/factor_alchemy/mechanism_catalog_20260925.json",
    "research/factor_alchemy/multiple_testing_guard.py",
    "research/factor_alchemy/multi_stage_validator.py",
    "research/factor_alchemy/ralph_loop.py",
    "research/factor_alchemy/factor_model_cooptim.py",
    "research/factor_alchemy/forge/quantum_challenger.py",
    "research/factor_alchemy/s5_joint_filter.py",
    "research/factor_alchemy/portfolio_execution_ledger.py",
    "data/mab_scheduler_state.json", "data/experience_memory.json",
    "data/stage2_s5_pending.json", "data/stage2_s5_verdicts.json",
    "data/ralph_loop_result_20260925_222930.json",
    "reports/fuxi_quality_gate_20260925/economic_probe_v2.json",
    "reports/fuxi_quality_gate_20260925/economic_probe_v3.json",
    "reports/fuxi_drought_execution_20260925/label_pool_audit.json",
    "reports/fuxi_mechanism_quantum_20260925/pilot_readout.json",
    "data/version_registry.json",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES}


def mechanism_probe():
    plans = load_catalog(FA / "mechanism_catalog_20260925.json", "daily")
    pid = "pv_gap_rejection_v1"
    source = {
        "status": "valid", "return_status": "valid", "economic_verdict": "valid",
        "book_kind": "fixed_shares_cash", "full_window_covered": True,
        "cost_bps": 15, "source_protocol_sha256": "",
        "candidate": {"semantic_plan_id": pid}, "metric_kind": "net_excess_bps",
        "paired_net_bps": [8.0] * 30,
        "parity": {"compared_days": 30, "independent_execution_or_data_validation": True},
    }
    raw = json.dumps(source).encode()
    row = {"schema": EVIDENCE_SCHEMA, "observation_id": "synthetic_no_file",
           "plan_id": pid, "status": "valid",
           "source_file": "reports/synthetic_source_not_written.json",
           "source_sha256": hashlib.sha256(raw).hexdigest()}
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "read_text", return_value=json.dumps(row) + "\n"), \
         patch.object(Path, "read_bytes", return_value=raw):
        observations = load_evidence(ROOT / "reports/synthetic_index_not_written.jsonl", ROOT, plans)
    _, audit = select_plan(plans, observations, seed=17, round_index=2)
    return {"synthetic": True, "empty_protocol_digest_accepted": True,
            "missing_dates_and_baseline_accepted": True, "self_asserted_parity_accepted": True,
            "mode": audit["mode"], "lcb_bps": observations[0]["economic_lcb_bps"],
            "grants_promotion": audit["grants_promotion"],
            "catalog_plan_count": len(plans),
            "catalog_uses_only_ohlcv": all(set(p["fields"]) <= {"open", "high", "low", "close", "volume"} for p in plans)}


def s6_probe():
    cases = {"absent": None, "three_returns": [0.01, -0.01, 0.02],
             "all_nan": [float("nan")] * 20,
             "three_finite_and_twenty_nan": [0.01, -0.01, 0.02] + [float("nan")] * 20}
    guard = MultipleTestingGuard(trials=5, gate_enabled=True)
    return {name: guard.evaluate_single(name, values).to_dict() for name, values in cases.items()}


def mab_probe():
    source = (FA / "ralph_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "hasattr(self, 'mab') and self.mab"
             and "local_reward" in ast.get_source_segment(source, n)]
    if len(nodes) != 1:
        raise RuntimeError("local MAB reward block identity changed")
    arm = SimpleNamespace(status="active", pulls=1, expected_reward=.8,
                          reward_source="jq", n_rewards=10)
    scheduler = SimpleNamespace(resolve_arms=lambda _: [arm], save_state=lambda: None)
    result = SimpleNamespace(s1_icir=-1., s1_eval_status="error", eligible_for_jq=False)
    context = {"self": SimpleNamespace(mab=scheduler), "candidates": [{"paradigm": "synthetic"}],
               "results": [result], "np": SimpleNamespace(tanh=math.tanh)}
    with patch.dict(os.environ, {"FUXI_MAB_PARADIGM_BROADCAST": "on"}):
        exec(compile(ast.Module(body=[nodes[0]], type_ignores=[]), "<actual_local_mab_block>", "exec"), context)
    return {"synthetic": True, "source_line": nodes[0].lineno,
            "input_status": "error", "input_icir": -1., "before_reward": .8,
            "after": vars(arm), "production_save_called": False}


def quantum_probe():
    def one(sequence):
        forge = SimpleNamespace(max_depth=3, max_complexity=12, fitness_mode="legacy",
                                dim_prune="off", _fitness=lambda _: 1.0,
                                data={name: np.ones((2, 2)) for name in
                                      ("open", "high", "low", "close", "volume", "returns")})
        q = QuantumForgeChallenger(forge, seed=17, unique_budget=5, population_size=4, islands=1)
        seq = iter(sequence)
        q._decode = lambda _: ExprNode(PRIMITIVE_BY_NAME[next(seq)])
        calls = []
        original = q._rotate_toward
        def rotate(island, elite):
            calls.append(island)
            original(island, elite)
        q._rotate_toward = rotate
        result = q.run()
        return {key: result[key] for key in ("draws", "unique_evaluated", "duplicates")} | {"rotation_calls": len(calls)}
    forge = SimpleNamespace(max_depth=6, max_complexity=12, fitness_mode="legacy", dim_prune="off",
                            paradigm_profiles=MATURE_PARADIGM_PROFILES,
                            data={name: np.ones((2, 2)) for name in
                                  ("open", "high", "low", "close", "volume", "returns")})
    q = QuantumForgeChallenger(forge, seed=17, unique_budget=5, target_paradigm="动量反转")
    reachable = {q._weighted_choice(q.ops, q.op_weights, code >> 1).name for code in range(64, 256)}
    return {"synthetic": True, "all_accepted": one(["close", "open", "high", "low", "volume"]),
            "duplicate_at_generation_boundary": one(["close", "open", "high", "close", "low", "volume"]),
            "decoder_depth1_missing_ops": sorted({p.name for p in q.ops} - reachable)}


def state_snapshot():
    def read(name):
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    mab = read("data/mab_scheduler_state.json")
    arms = mab["directions"]
    arms = list(arms.values()) if isinstance(arms, dict) else arms
    memory = read("data/experience_memory.json")
    attempts = memory["attempts"]
    pending = read("data/stage2_s5_pending.json")
    verdicts = read("data/stage2_s5_verdicts.json")
    return {
        "mab": {"arms": len(arms), "total_pulls": mab["total_pulls"],
                "sum_arm_pulls": sum(d.get("pulls", 0) for d in arms),
                "reward_sources": dict(Counter(d.get("reward_source", "") for d in arms)),
                "nonzero_reward_with_zero_n_rewards": sum(abs(d.get("expected_reward", 0)) > 0 and d.get("n_rewards", 0) == 0 for d in arms)},
        "memory": {"attempts": len(attempts), "outcomes": dict(Counter(x.get("outcome") for x in attempts)),
                   "with_s5_eval_status": sum("s5_eval_status" in x for x in attempts),
                   "with_evaluation_status": sum("evaluation_status" in x for x in attempts),
                   "attempt_count_is_independent_trials": False},
        "stage2_queue": {key: pending.get(key) for key in
                         ("updated_at", "mode", "n_pass_total", "n_already_verified", "n_pending")},
        "stage2_verdict_count": verdicts.get("n_verdicts"),
    }


def main():
    before = hashes()
    result = {"schema": "fuxi_iteration_review/v1", "as_of": "2026-09-26", "version": "v0.7.78",
              "production_mutations": False, "network_calls": 0, "economic_backtests": 0,
              "mechanism_probe": mechanism_probe(), "s6_probe": s6_probe(),
              "mab_probe": mab_probe(), "quantum_probe": quantum_probe(), "state": state_snapshot()}
    after = hashes()
    if before != after:
        raise RuntimeError("source/state changed while reading; snapshot invalid")
    result["sources_unchanged_during_review"] = True
    result["source_sha256"] = before
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
