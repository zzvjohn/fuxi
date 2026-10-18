import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
TEST_ROOT = BASE.parents[1] / "reports" / "fuxi_generator_rebuild_20260925" / "test_tmp"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
from semantic_plan import load_plan, prompt_block, tag_candidates, validate_plan  # noqa: E402
from llm_generator import LLMGenerator  # noqa: E402
from multi_stage_validator import MultiStageValidator  # noqa: E402


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.file = BASE / "semantic_plans_20260925.json"
        self.plan = load_plan(self.file, "pv_gap_rejection_v1", "daily")

    def test_validated_plan_renders_and_tags(self):
        self.assertIn("next_session", prompt_block(self.plan))
        tagged = tag_candidates([{"formula": "close.pct_change(5)"}], self.plan)
        self.assertEqual(tagged[0]["semantic_plan_id"], self.plan["plan_id"])
        self.assertEqual(tagged[0]["semantic_target_use"], "promotion_candidate")
        self.assertEqual(tagged[0]["semantic_plan_violations"], [])
        invalid = tag_candidates([{"formula": "north_money / close"}], self.plan)
        self.assertIn("north_money", invalid[0]["semantic_plan_violations"])

    def test_unverified_field_and_same_day_trade_refused(self):
        for change in ({"fields": ["north_money"]}, {"earliest_trade": "same_session"}):
            with self.assertRaises(ValueError):
                validate_plan({**self.plan, **change})

    def test_wrong_lane_and_duplicate_identity_refused(self):
        with self.assertRaises(ValueError):
            load_plan(self.file, self.plan["plan_id"], "weekly")
        data = json.loads(self.file.read_text(encoding="utf-8"))
        data["plans"].append(data["plans"][1])
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            with self.assertRaises(ValueError):
                load_plan(self.file, self.plan["plan_id"], "daily")

    def test_real_prompt_wiring_and_rollback(self):
        gen = LLMGenerator(output_dir=TEST_ROOT)
        legacy = gen.build_prompt("gap", n_factors=2)
        self.assertNotIn("Forge DSL 语法(如", legacy)
        self.assertIn("无 amount/无基本面", legacy)
        gen.set_semantic_plan(self.plan)
        planned = gen.build_prompt("gap", n_factors=2)
        self.assertIn(self.plan["plan_id"], planned)
        self.assertEqual(
            gen.tag_semantic_candidates([{"formula": "close.pct_change(5)"}])[0]
            ["semantic_target_use"], "promotion_candidate")
        gen.set_semantic_plan(None)
        self.assertEqual(gen.build_prompt("gap", n_factors=2), legacy)

    def test_weekly_prompt_has_single_consistent_field_contract(self):
        gen = LLMGenerator(output_dir=TEST_ROOT)
        prompt = gen.build_prompt(n_factors=1, weekly_mode=True)
        self.assertIn("周频 close open high low volume amount", prompt)
        self.assertNotIn("无 amount/无基本面", prompt)

    def test_research_feature_cannot_enter_jq_even_when_stages_pass(self):
        validator = MultiStageValidator()
        def pass_stages(stage):
            def run(_candidates, results, *_args):
                for result in results:
                    setattr(result, f"s{stage}_passed", True)
                    if stage == 5:
                        result.s5_eval_status = "valid"
                return results
            return run
        for stage, name in enumerate(("stage1_fast_ic", "stage2_correlation",
                                      "stage3_batch_dedup", "stage4_oos_replacement",
                                      "stage5_joint_filter", "stage6_multiple_testing"), 1):
            setattr(validator, name, pass_stages(stage))
        rows = [{"factor_name": "research", "semantic_target_use": "research_feature"},
                {"factor_name": "promotion", "semantic_target_use": "promotion_candidate"},
                {"factor_name": "invalid", "semantic_target_use": "promotion_candidate",
                 "semantic_plan_violations": ["north_money"]},
                {"factor_name": "legacy"},
                {"factor_name": "quality_review", "formula_quality": {"review_codes": ["mechanism_mismatch"]}},
                {"factor_name": "quality_shadow_hard", "formula_quality": {"hard_codes": ["algebraic_zero"]}}]
        results, summary = validator.validate(rows, {}, {}, s5_filter=None)
        self.assertEqual([r.eligible_for_jq for r in results],
                         [False, True, False, True, False, False])
        self.assertEqual(summary["eligible_for_jq"], 2)


if __name__ == "__main__":
    unittest.main()
