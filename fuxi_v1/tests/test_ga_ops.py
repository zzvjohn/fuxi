"""Bounded GA operators: legality, boundedness, dedup, and the A/B toggle."""
import os
import random
import unittest

from fuxi_v1.dsl import compile_ast
from fuxi_v1.ga_ops import combine_ga, ga_ops_enabled, refine_ga
from fuxi_v1.search import factor_spec, generate, templates_expanded

FIELDS = {"close": {"unit": "price"}, "open": {"unit": "price"},
          "high": {"unit": "price"}, "low": {"unit": "price"},
          "volume": {"unit": "shares"}, "amount": {"unit": "currency"},
          "turnover_rate": {"unit": "ratio"}, "float_shares": {"unit": "shares"}}


def _parents(n=6):
    specs = [factor_spec(ast, FIELDS, m) for m, ast in templates_expanded(FIELDS)]
    return specs[:n]


class RefineGaTests(unittest.TestCase):
    def test_refine_is_legal_bounded_novel_and_deduped(self):
        parents = _parents()
        proposals, receipt = refine_ga(parents, FIELDS, 8, random.Random(7), seen=set())
        self.assertEqual(len(proposals), 8)
        self.assertGreaterEqual(receipt["attempts"], 8)
        hashes = [p["formula_hash"] for p in proposals]
        self.assertEqual(len(set(hashes)), 8)
        parent_hashes = {p["formula_hash"] for p in parents}
        for p in proposals:
            compiled = compile_ast(p["canonical_ast"], FIELDS)
            self.assertLessEqual(compiled["node_count"], 64)
            self.assertNotIn(p["formula_hash"], parent_hashes)

    def test_refine_inherits_parent_mechanism(self):
        parents = _parents(3)
        proposals, _ = refine_ga(parents, FIELDS, 6, random.Random(3), seen=set())
        parent_mech = {p["candidate_id"]: p["hypothesis"]["mechanism"] for p in parents}
        for p in proposals:
            self.assertIn(p["hypothesis"]["mechanism"], parent_mech.values())
            self.assertEqual(p["action"], "refine_factor")


class CombineGaTests(unittest.TestCase):
    def test_combine_has_combination_and_crossover_axes(self):
        parents = _parents(6)
        proposals, receipt = combine_ga(parents, FIELDS, 12, random.Random(11), seen=set())
        self.assertEqual(len(proposals), 12)
        axes = {(p.get("mutation_patch") or {}).get("axis") for p in proposals}
        self.assertIn("combination", axes)
        self.assertIn("subtree_crossover", axes)
        for p in proposals:
            self.assertEqual(p["hypothesis"]["mechanism"], "combined_independent_hypotheses")
            compiled = compile_ast(p["canonical_ast"], FIELDS)
            self.assertLessEqual(compiled["node_count"], 64)

    def test_combine_is_novel_and_deduped(self):
        parents = _parents(4)
        proposals, _ = combine_ga(parents, FIELDS, 10, random.Random(9), seen=set())
        hashes = [p["formula_hash"] for p in proposals]
        self.assertEqual(len(set(hashes)), 10)
        parent_hashes = {p["formula_hash"] for p in parents}
        self.assertTrue(all(h not in parent_hashes for h in hashes))


class ToggleTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("FUXI_V1_GA_OPS", None)

    def _axes(self, arm, on):
        os.environ["FUXI_V1_GA_OPS"] = "on" if on else "off"
        out = generate(FIELDS, arm=arm, count=6, seed=5, parents=_parents(4), seen=set())
        return sorted({(p.get("mutation_patch") or {}).get("axis", "") for p in out})

    def test_refine_toggle_switches_path(self):
        off = self._axes("refine", False)
        on = self._axes("refine", True)
        self.assertIn("one_window", off)
        self.assertIn("one_node_edit", on)

    def test_combine_toggle_switches_path(self):
        off = self._axes("combine", False)
        on = self._axes("combine", True)
        self.assertNotIn("subtree_crossover", off)
        self.assertIn("subtree_crossover", on)

    def test_ga_ops_defaults_on(self):
        self.assertTrue(ga_ops_enabled())


if __name__ == "__main__":
    unittest.main()
