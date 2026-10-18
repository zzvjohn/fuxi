import hashlib
import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))

from forge import FactorForge  # noqa: E402
from forge.expression import ExprNode  # noqa: E402
from forge.primitives import PRIMITIVE_BY_NAME  # noqa: E402
from forge.quantum_challenger import QuantumForgeChallenger  # noqa: E402
from mechanism_search import (  # noqa: E402
    EVIDENCE_SCHEMA, load_catalog, load_evidence, select_plan,
)


class MechanismEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plans = load_catalog(BASE / "mechanism_catalog_20260925.json", "daily")

    def test_current_books_do_not_become_zero_reward(self):
        path = ROOT / "reports/fuxi_mechanism_quantum_20260925/economic_evidence.jsonl"
        rows = load_evidence(path, ROOT, self.plans)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["status"] for r in rows}, {"data_insufficient"})
        self.assertTrue(all("economic_lcb_bps" not in r for r in rows))
        plan, audit = select_plan(self.plans, rows, seed=17, round_index=2)
        self.assertEqual(audit["mode"], "explore")
        self.assertEqual(audit["valid_economic_observations"], 0)
        self.assertNotEqual(plan["plan_id"], "pv_gap_rejection_v1")
        self.assertFalse(audit["grants_promotion"])

    def test_verified_complete_book_can_drive_exploitation(self):
        pid = "pv_gap_rejection_v1"
        dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(30)]
        parity_evidence = {
            "schema": "fuxi_independent_book_parity/v1",
            "source_protocol_sha256": "a" * 64,
            "candidate_id": "cand_test",
            "compared_dates": dates,
            "complete_event_coverage": True,
            "independent_replay": True,
            "max_abs_nav_error": 0.0,
        }
        parity_raw = json.dumps(parity_evidence).encode("utf-8")
        result = {
            "status": "valid", "return_status": "valid",
            "economic_verdict": "valid", "book_kind": "fixed_shares_cash",
            "full_window_covered": True, "cost_bps": 15,
            "source_protocol_sha256": "a" * 64,
            "candidate": {"semantic_plan_id": pid, "candidate_id": "cand_test"},
            "window": {"start": dates[0], "end": dates[-1]},
            "paired_dates": dates,
            "metric_kind": "net_excess_bps",
            "paired_net_bps": [8.0 + (i % 3) for i in range(30)],
            "parity": {"compared_days": 30,
                       "independent_execution_or_data_validation": True,
                       "independent_artifact_file": "reports/fuxi_mechanism_quantum_20260925/test_parity.json",
                       "independent_artifact_sha256": hashlib.sha256(parity_raw).hexdigest()},
        }
        row = {"schema": EVIDENCE_SCHEMA, "observation_id": "v1",
               "plan_id": pid, "status": "valid",
               "source_file": "reports/fuxi_mechanism_quantum_20260925/test_result.json"}

        def load():
            raw = json.dumps(result).encode("utf-8")
            row["source_sha256"] = hashlib.sha256(raw).hexdigest()
            def read_bytes(path):
                return parity_raw if path.name == "test_parity.json" else raw
            with patch.object(Path, "exists", return_value=True), \
                 patch.object(Path, "read_text", return_value=json.dumps(row) + "\n"), \
                 patch.object(Path, "read_bytes", read_bytes):
                return load_evidence(ROOT / "reports/fuxi_mechanism_quantum_20260925/test_evidence.jsonl",
                                     ROOT, self.plans)

        rows = load()
        self.assertGreater(rows[0]["economic_lcb_bps"], 0)
        plan, audit = select_plan(self.plans, rows, seed=17, round_index=2)
        self.assertEqual(plan["plan_id"], pid)
        self.assertEqual(audit["mode"], "exploit")
        result["parity"]["independent_execution_or_data_validation"] = False
        with self.assertRaisesRegex(ValueError, "complete verified"):
            load()

    def test_source_mutation_and_traversal_fail_closed(self):
        path = ROOT / "reports/fuxi_mechanism_quantum_20260925/economic_evidence.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        target = ROOT / "reports/fuxi_mechanism_quantum_20260925/test_evidence.jsonl"
        row = json.loads(lines[0])
        row["source_sha256"] = "0" * 64
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(row) + "\n"), \
             patch.object(Path, "read_bytes", return_value=b"{}"): 
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_evidence(target, ROOT, self.plans)
        row["source_file"] = "../outside.json"
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(row) + "\n"):
            with self.assertRaisesRegex(ValueError, "leaves"):
                load_evidence(target, ROOT, self.plans)

    def test_self_attested_positive_result_cannot_drive_selection(self):
        pid = "pv_gap_rejection_v1"
        result = {
            "status": "valid", "return_status": "valid",
            "economic_verdict": "valid", "book_kind": "fixed_shares_cash",
            "full_window_covered": True, "cost_bps": 15,
            "source_protocol_sha256": "a" * 64,
            "candidate": {"semantic_plan_id": pid, "candidate_id": "cand_fake"},
            "window": {"start": "2025-01-01", "end": "2025-01-30"},
            "paired_dates": [(date(2025, 1, 1) + timedelta(days=i)).isoformat()
                             for i in range(30)],
            "metric_kind": "net_excess_bps", "paired_net_bps": [8.0] * 30,
            "parity": {"compared_days": 30,
                       "independent_execution_or_data_validation": True},
        }
        raw = json.dumps(result).encode("utf-8")
        row = {"schema": EVIDENCE_SCHEMA, "observation_id": "fake",
               "plan_id": pid, "status": "valid",
               "source_file": "reports/fuxi_mechanism_quantum_20260925/fake.json",
               "source_sha256": hashlib.sha256(raw).hexdigest()}
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(row) + "\n"), \
             patch.object(Path, "read_bytes", return_value=raw):
            with self.assertRaisesRegex(ValueError, "independent parity artifact"):
                load_evidence(ROOT / "reports/fuxi_mechanism_quantum_20260925/fake_evidence.jsonl",
                              ROOT, self.plans)


class QuantumChallengerTests(unittest.TestCase):
    @staticmethod
    def _forge(seed):
        data = {name: np.ones((4, 12)) for name in
                ("open", "high", "low", "close", "volume", "returns")}
        forge = FactorForge(data=data, max_depth=3, max_complexity=12,
                            seed=seed, dim_prune="off", fitness_mode="legacy")
        forge._fitness = lambda tree: float(len(tree.to_string()) % 17)
        return forge

    def test_fixed_budget_reproducible_and_same_field_contract(self):
        a = QuantumForgeChallenger(self._forge(13), seed=13,
                                   unique_budget=8, population_size=4).run()
        b = QuantumForgeChallenger(self._forge(13), seed=13,
                                   unique_budget=8, population_size=4).run()
        self.assertEqual(a, b)
        self.assertTrue(a["budget_met"])
        self.assertEqual(a["unique_evaluated"], 8)
        self.assertEqual(len({r["expression"] for r in a["candidates"]}), 8)
        self.assertTrue(all("vwap" not in r["terminals"] for r in a["candidates"]))
        self.assertTrue(all(r["complexity"] <= 12 for r in a["candidates"]))

    def test_final_population_pairs_scored_trees(self):
        forge = self._forge(7)
        initial = [ExprNode(PRIMITIVE_BY_NAME[name]) for name in
                   ("close", "volume", "high")]
        scores = {"close": 100.0, "volume": 50.0, "high": 10.0}
        forge._gen_tree_fsa_guarded = lambda _kind, _tag: initial.pop(0)
        forge._fitness = lambda tree: scores.get(tree.to_string(), -1.0)
        forge._evaluate_icir = lambda _tree: 0.5
        forge.evolve(n_generations=1, pop_size=3, tournament_size=2,
                     elite_count=1, workers=1, verbose=False, icir_top_n=0)
        self.assertEqual(
            {row["expression"]: row["fitness"] for row in forge.final_population},
            scores,
        )


if __name__ == "__main__":
    unittest.main()
