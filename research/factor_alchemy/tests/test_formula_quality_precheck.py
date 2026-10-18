"""Regression tests for real LLM-output defects, not economic performance."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from formula_quality_precheck import assess_candidate
from llm_generator import LLMGenerator


class FormulaQualityPrecheckTests(unittest.TestCase):
    def test_epsilon_cancellation_and_real_frozen_pair(self):
        frozen = json.loads((ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget/result.json")
                            .read_text(encoding="utf-8"))
        observed = {arm: [assess_candidate(c)["status"] for c in row["candidates"]]
                    for arm, row in frozen["arms"].items()}
        self.assertEqual(observed["baseline"], ["reject", "review", "review", "review"])
        self.assertEqual(observed["semantic_plan"], ["reject", "pass", "pass", "review"])

    def test_positive_price_denominators_are_not_hard_rejected(self):
        valid = assess_candidate({"formula": "open / close.shift(1) - close / open"})
        self.assertEqual(valid["status"], "pass")
        uncertain = assess_candidate({"formula": "volume / volume.rolling(5).mean()"})
        self.assertEqual(uncertain["status"], "review")
        self.assertIn("denominator_domain_unverified", uncertain["review_codes"])

    def test_rank_repair_matches_existing_s5_normalization(self):
        from s5_joint_filter import S5JointFilter
        expression = "(close / (open + 1e-8)).rank(pct=True)"
        result = assess_candidate({"formula": expression})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["repairs"], ["rank_axis_explicit_cross_section"])
        self.assertEqual(
            S5JointFilter._normalize_rank_axis(result["canonical_formula"]),
            result["canonical_formula"])
        self.assertEqual(
            S5JointFilter._normalize_rank_axis(expression).replace(" ", ""),
            result["canonical_formula"].replace(" ", ""))

    def test_time_axis_rejected_and_rolling_rank_left_alone(self):
        wrong = assess_candidate({"formula": "close.rank(pct=True, axis=0)"})
        self.assertIn("rank_time_axis", wrong["hard_codes"])
        rolling = assess_candidate({"formula": "close.rolling(5).rank(pct=True)"})
        self.assertEqual(rolling["repairs"], [])
        self.assertEqual(rolling["status"], "pass")
        different = assess_candidate({"formula": "close.rank(pct=True, axis=1) - close"})
        self.assertNotIn("algebraic_zero", different["hard_codes"])

    def test_enforce_rejects_only_proven_defects_and_preserves_raw(self):
        candidates = [{"factor_name": "bad", "formula": "close / (open + 1e-8) - close / (open + 1e-8)"},
                      {"factor_name": "good", "formula": "close.rank(pct=True)"}]
        gen = LLMGenerator(output_dir=ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget")
        with patch.dict(os.environ, {"FUXI_LLM_QUALITY_GATE": "enforce"}):
            kept = gen.filter_quality_candidates(candidates)
        self.assertEqual([c["factor_name"] for c in kept], ["good"])
        self.assertEqual(kept[0]["raw_formula"], "close.rank(pct=True)")
        self.assertIn("axis=1", kept[0]["formula"])
        self.assertEqual(candidates[0]["formula_quality"]["status"], "reject")

    def test_shadow_does_not_change_candidates(self):
        candidate = {"formula": "close.rank(pct=True)", "factor_name": "shadow"}
        gen = LLMGenerator(output_dir=ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget")
        with patch.dict(os.environ, {"FUXI_LLM_QUALITY_GATE": "shadow"}):
            kept = gen.filter_quality_candidates([candidate])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["formula"], "close.rank(pct=True)")
        self.assertEqual(kept[0]["formula_quality"]["repairs"],
                         ["rank_axis_explicit_cross_section"])

    def test_second_gate_keeps_raw_formula_fingerprint(self):
        formula = "(close / open).rank(pct=True)"
        candidate = {"factor_name": "twice", "formula": formula}
        gen = LLMGenerator(output_dir=ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget")
        with patch.dict(os.environ, {"FUXI_LLM_QUALITY_GATE": "enforce"}):
            first = gen.filter_quality_candidates([candidate])[0]
            original_quality = dict(first["formula_quality"])
            second = gen.filter_quality_candidates([first])[0]
        self.assertEqual(second["raw_formula"], formula)
        self.assertEqual(second["formula_quality"], original_quality)

    def test_response_generation_path_applies_gate(self):
        gen = LLMGenerator(output_dir=ROOT / "reports/fuxi_generator_rebuild_20260925/equal_budget")
        response = json.dumps([
            {"name": "zero", "expression": "close - close", "hypothesis": "none"},
            {"name": "valid", "expression": "(close - open) / (open + 1e-8)",
             "hypothesis": "intraday return"}], ensure_ascii=False)
        with patch.dict(os.environ, {"FUXI_LLM_QUALITY_GATE": "enforce"}):
            _, candidates = gen.generate(response_text=response)
        self.assertEqual([c["factor_name"] for c in candidates], ["valid"])


if __name__ == "__main__":
    unittest.main()
