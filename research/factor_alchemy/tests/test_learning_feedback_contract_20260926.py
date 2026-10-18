import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_feedback import candidate_feedback  # noqa: E402
from multiple_testing_guard import MultipleTestingGuard  # noqa: E402
from multi_stage_validator import MultiStageValidator, ValidationResult  # noqa: E402
from ralph_loop import RalphLoop  # noqa: E402


class FeedbackContractTests(unittest.TestCase):
    def test_error_is_not_economic_failure_or_proxy_reward(self):
        result = SimpleNamespace(s1_eval_status="error", s1_icir=-1,
                                 s4_eval_status="not_evaluated",
                                 s5_eval_status="data_insufficient",
                                 s6_eval_status="not_evaluated")
        feedback = candidate_feedback(
            {"factor_name": "x", "_source": "weekly_native_template",
             "_selected_generator": "llm"}, result)
        self.assertEqual(feedback["actual_generator"], "weekly_native_template")
        self.assertEqual(feedback["selected_generator"], "llm")
        self.assertIsNone(feedback["proxy_icir"])
        self.assertIsNone(feedback["economic_reward"])
        self.assertEqual(feedback["economic_reward_status"], "not_evaluated")

    def test_valid_s1_is_proxy_only(self):
        result = SimpleNamespace(s1_eval_status="valid", s1_icir=0.4,
                                 s4_eval_status="valid", s5_eval_status="valid",
                                 s6_eval_status="valid")
        feedback = candidate_feedback({"_source": "factorforge"}, result)
        self.assertEqual(feedback["actual_generator"], "forge")
        self.assertEqual(feedback["proxy_icir"], 0.4)
        self.assertIsNone(feedback["economic_reward"])

    def test_unverified_candidate_does_not_form_weak_memory(self):
        class FailOnWrite:
            def form(self, **_kwargs):
                raise AssertionError("legacy Memory formation must be skipped")
            def evolve(self, **_kwargs):
                raise AssertionError("legacy Memory evolution must be skipped")
        class EmptyTrajectory:
            def analyze_failures(self, **_kwargs):
                return {"total_analyzed": 0}
            def get_distillation_hints(self):
                return {}
        loop = RalphLoop.__new__(RalphLoop)
        loop.memory = FailOnWrite()
        loop.traj_logger = EmptyTrajectory()
        with patch("library_orthogonality.LibraryOrthogonalityManager") as manager:
            manager.return_value._corr_matrix = None
            result = loop._phase_distill([{"factor_name": "unmeasured"}], {}, "p")
        self.assertEqual(result["formation"]["status"],
                         "awaiting_verified_economic_label")


class MultipleTestingStatusTests(unittest.TestCase):
    def setUp(self):
        self.series = np.array([0.01, -0.005, 0.015, 0.008, -0.004,
                                0.007, 0.004, -0.003, 0.014, -0.006,
                                0.012, 0.009, -0.002, 0.006], dtype=float)

    def test_absent_short_and_nonfinite_are_not_passes(self):
        for values, status in ((None, "not_evaluated"),
                               (self.series[:3], "data_insufficient"),
                               (np.r_[self.series[:3], np.full(20, np.nan)],
                                "data_insufficient")):
            for gate in (False, True):
                result = MultipleTestingGuard(trials=5, gate_enabled=gate).evaluate_single("x", values)
                self.assertEqual(result.evaluation_status, status)
                self.assertFalse(result.dsr_passed)
                self.assertFalse(result.s6_passed)
                self.assertEqual(result.policy_allows, not gate)

    def test_valid_dsr_has_separate_shadow_policy(self):
        result = MultipleTestingGuard(trials=5, gate_enabled=False,
                                      dsr_min_probability=0).evaluate_single("x", self.series)
        self.assertEqual(result.evaluation_status, "valid")
        self.assertTrue(result.s6_passed)
        self.assertTrue(result.policy_allows)

    def test_validator_keeps_shadow_eligibility_but_reports_missing(self):
        validator = MultiStageValidator()
        result = ValidationResult(factor_name="x")
        validator.stage6_multiple_testing(
            [{"factor_name": "x"}], [result],
            guard=MultipleTestingGuard(trials=5, gate_enabled=False),
        )
        self.assertFalse(result.s6_computed)
        self.assertEqual(result.s6_eval_status, "not_evaluated")
        self.assertFalse(result.s6_passed)
        self.assertTrue(result.s6_policy_allows)


if __name__ == "__main__":
    unittest.main()
