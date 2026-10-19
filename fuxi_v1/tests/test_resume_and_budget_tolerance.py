"""Regression tests for the 2026-10-19 resume + budget-tolerance fixes.

1. ``normalize_request`` must return a JSON-roundtrippable object so that the
   in-memory ``experiment`` equals ``json.loads(runs.manifest)`` on resume.
   Historically ``model_algorithms`` was a tuple, the manifest a list, so every
   resume raised ``run identity collision``.
2. ``EvidenceStore.finish`` must not poison a campaign for sub-percent
   measurement jitter (300.3s against a 300s reservation), while a genuine
   overrun still halts it.
"""
import json
import tempfile
import unittest
from pathlib import Path

from fuxi_v1.budgets import budget_units, is_overrun, worker_timeout_seconds
from fuxi_v1.contracts import ContractError, canonical
from fuxi_v1.evidence import EvidenceStore


class BudgetToleranceTests(unittest.TestCase):
    def test_default_worker_window_is_420(self):
        self.assertEqual(worker_timeout_seconds({}), 420)
        self.assertEqual(worker_timeout_seconds({"FUXI_V1_WORKER_TIMEOUT": "480"}), 480)
        self.assertEqual(worker_timeout_seconds({"FUXI_V1_WORKER_TIMEOUT": "5"}), 120)
        self.assertEqual(budget_units(20, 4), 20 * 4 * 420 + 420)

    def test_jitter_is_not_an_overrun_but_a_real_breach_is(self):
        self.assertFalse(is_overrun(300.3, 300))
        self.assertFalse(is_overrun(300.0, 300))
        self.assertFalse(is_overrun(305.9, 300))
        self.assertTrue(is_overrun(306.1, 300))
        self.assertTrue(is_overrun(420.0, 300))
        self.assertFalse(is_overrun(None, 300))

    def test_campaign_survives_jitter_and_still_halts_on_breach(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "evidence.sqlite")
            try:
                store.campaign("jitter", {"budget_units": 5000})
                first = store.reserve("jitter", "e1", 300)
                store.finish(first, "completed", 300.3, {"ok": 1})
                self.assertNotEqual(
                    store.db.execute("SELECT status FROM campaigns WHERE id='jitter'").fetchone()[0],
                    "budget_overrun")
                second = store.reserve("jitter", "e2", 300)  # must not raise
                store.finish(second, "completed", 310.0, {"ok": 2})
                self.assertEqual(
                    store.db.execute("SELECT status FROM campaigns WHERE id='jitter'").fetchone()[0],
                    "budget_overrun")
                with self.assertRaisesRegex(ContractError, "budget overrun"):
                    store.reserve("jitter", "e3", 300)
            finally:
                store.close()


class ResumeNormalisationTests(unittest.TestCase):
    def test_normalized_request_roundtrips_through_json(self):
        from fuxi_v1.contracts import load_json
        from fuxi_v1.unified_runner import normalize_request
        root = Path(__file__).resolve().parents[2]
        request = {
            "schema_version": "fuxi.unified_run.1",
            "input": "data/fuxi_unified/snapshots/native64_20210104_20251231_ledger_v1/campaign_input.json",
            "rounds": 1, "candidates_per_round": 1, "seed": 1,
            "mode": "adaptive", "scheduler_policy": "legacy", "include_catalog": False,
            "llm_enabled": False, "model_algorithms": [], "external_hypotheses": [],
        }
        frozen = json.loads(canonical(request).decode("utf-8"))
        frozen["input_sha256"] = __import__("fuxi_v1.unified_runner", fromlist=["file_sha"]).file_sha(
            root / request["input"])
        normalized = normalize_request(root, frozen)
        self.assertIsInstance(normalized["model_algorithms"], list)
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)
        # the manifest is written as canonical(experiment); json.loads must equal it
        experiment = {"request": normalized, "code": {}, "limits": {}, "effective_config": {}}
        self.assertEqual(json.loads(canonical(experiment).decode("utf-8")), experiment)


if __name__ == "__main__":
    unittest.main()
