import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "stage2_s5_chain_preflight_test", ROOT / "scripts" / "stage2_s5_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


class Stage2PreflightContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"factor_name": "bad", "formula": "close - close", "paradigm": "p", "icir": 0.4},
            {"factor_name": "good", "formula": "close.rank(pct=True)", "paradigm": "p", "icir": 0.4},
        ]

    def test_shadow_preserves_candidates_and_reports_quality(self):
        with patch.dict(os.environ, {"FUXI_STAGE2_PREFLIGHT": "shadow"}):
            readings, engine = chain.run_s5(self.rows, 2, "dry", dry_run=True)
        self.assertEqual(readings, [])
        self.assertEqual(engine["generator_preflight"]["input"], 2)
        self.assertEqual(engine["generator_preflight"]["kept"], 2)
        self.assertEqual(self.rows[1]["formula"], "close.rank(pct=True)")

    def test_enforce_rejection_remains_in_five_state_denominator(self):
        seen = []
        class FakeResult:
            error = ""
            reason = "ok"
            elapsed_sec = 0.1
            passed = False
            def to_dict(self):
                return {"passed": False, "elapsed_sec": self.elapsed_sec}
        class FakeS5:
            is_ready = True
            def __init__(self, *, top_n):
                self.top_n = top_n
            def validate_batch(self, rows, *, verbose):
                seen.extend(rows)
                return [FakeResult() for _ in rows]
        fake_module = types.ModuleType("s5_joint_filter")
        fake_module.S5JointFilter = FakeS5
        with patch.dict(os.environ, {"FUXI_STAGE2_PREFLIGHT": "enforce"}), \
             patch.dict(sys.modules, {"s5_joint_filter": fake_module}):
            readings, engine = chain.run_s5(self.rows, 2, "unit")
        self.assertEqual(engine["generator_preflight"]["rejected"], 1)
        self.assertEqual([r["s5_eval_status"] for r in readings],
                         ["precheck_rejected", "valid"])
        self.assertEqual([r["factor_name"] for r in seen], ["good"])
        self.assertIn("axis=1", seen[0]["formula"])
        self.assertEqual(self.rows[1]["formula"], "close.rank(pct=True)")

    def test_nonempty_request_cannot_silently_return_empty_result(self):
        class EmptyS5:
            is_ready = True
            def __init__(self, *, top_n):
                pass
            def validate_batch(self, rows, *, verbose):
                return []
        fake_module = types.ModuleType("s5_joint_filter")
        fake_module.S5JointFilter = EmptyS5
        with patch.dict(os.environ, {"FUXI_STAGE2_PREFLIGHT": "off"}), \
             patch.dict(sys.modules, {"s5_joint_filter": fake_module}):
            readings, engine = chain.run_s5(self.rows, 2, "unit")
        self.assertIsNone(readings)
        self.assertEqual(engine["s5_mode"], "result_count_mismatch")

    def test_cli_refuses_when_fuxi_writer_is_owned_elsewhere(self):
        import run_contract
        class BusyLock:
            def __init__(self, **_kwargs):
                pass
            def __enter__(self):
                raise run_contract.ActiveFuxiRunError("live owner")
            def __exit__(self, *_args):
                return False
        with patch.object(run_contract, "FuxiWriterLock", BusyLock):
            self.assertEqual(chain.main(), 3)


if __name__ == "__main__":
    unittest.main()
