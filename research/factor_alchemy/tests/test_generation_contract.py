import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from generation_contract import gp_template_dict, resolve_gp_parent_pool, s5_streak_event  # noqa: E402


class ParentTests(unittest.TestCase):
    def test_empty_head_resolves_sample_and_preserves_quality(self):
        head = SimpleNamespace(pattern_id="p", formula="", sample_factor_ids=["f"],
                               verification_level="jq_single", success_rate=0.7,
                               occurrence_count=12)
        row = gp_template_dict(head, {"f": "close.pct_change(5)"})
        self.assertEqual(row["formula"], "close.pct_change(5)")
        self.assertEqual(row["verification_level"], "jq_single")
        self.assertEqual(row["occurrence_count"], 12)

    def test_pool_fills_past_empty_retrieval_heads(self):
        memory = {"attempts": [], "success_templates": [
            {"pattern_id": "a", "formula": "close.pct_change(5)"},
            {"pattern_id": "b", "formula": "volume.pct_change(5)"}]}
        pool = resolve_gp_parent_pool([{"pattern_id": "top", "formula": ""}], memory)
        self.assertEqual(len(pool), 2)
        self.assertTrue(all(p["formula"] for p in pool))

    def test_live_memory_has_two_resolvable_parents(self):
        path = ROOT / "data/experience_memory.json"
        if not path.exists():
            self.skipTest("live memory absent")
        memory = json.loads(path.read_text(encoding="utf-8"))
        pool = resolve_gp_parent_pool(memory["success_templates"][:5], memory)
        self.assertGreaterEqual(len(pool), 2)
        from factor_expression_tree import GPBreeder
        breeder = GPBreeder(seed=17)
        with patch.object(GPBreeder, "_get_next_breed_id", return_value=900000), \
             patch.object(GPBreeder, "_save_breed_counter"):
            children = breeder.breed_from_templates(
                pool, n_children=4, fsa=None, output_format="pandas")
        self.assertEqual(len(children), 4)
        self.assertTrue(all(c.get("formula") for c in children))

    def test_streak_never_calls_missing_economic_data_a_failure(self):
        missing = SimpleNamespace(s4_passed=True, s5_eval_status="data_insufficient",
                                  s5_passed=False)
        fail = SimpleNamespace(s4_passed=True, s5_eval_status="valid", s5_passed=False)
        success = SimpleNamespace(s4_passed=True, s5_eval_status="valid", s5_passed=True)
        self.assertEqual(s5_streak_event([missing]), "unassessed")
        self.assertEqual(s5_streak_event([missing, fail]), "unassessed")
        self.assertEqual(s5_streak_event([fail]), "economic_failure")
        self.assertEqual(s5_streak_event([fail, success]), "pass")


if __name__ == "__main__":
    unittest.main()
