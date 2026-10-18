"""Calibration null-population contracts: determinism, legality, budget, statistics."""
import unittest

from fuxi_v1.calibration import (baseline_summary, fields_used, percentile_of,
                                 random_ast, random_population)
from fuxi_v1.dsl import CATALOG, compile_ast

FIELDS = {"close": {"unit": "price"}, "open": {"unit": "price"},
          "high": {"unit": "price"}, "low": {"unit": "price"},
          "volume": {"unit": "shares"}, "amount": {"unit": "currency"}}


class CalibrationTests(unittest.TestCase):
    def test_population_is_deterministic_and_seed_sensitive(self):
        a, ra = random_population(FIELDS, 12, 20261005)
        b, rb = random_population(FIELDS, 12, 20261005)
        c, _ = random_population(FIELDS, 12, 20261006)
        self.assertEqual([x["formula_hash"] for x in a], [x["formula_hash"] for x in b])
        self.assertEqual(ra["node_counts"], rb["node_counts"])
        self.assertNotEqual([x["formula_hash"] for x in a], [x["formula_hash"] for x in c])

    def test_population_is_legal_distinct_and_bounded(self):
        pop, report = random_population(FIELDS, 24, 7)
        hashes = [c["formula_hash"] for c in pop]
        self.assertEqual(len(set(hashes)), 24)
        self.assertEqual(report["generated"], 24)
        for c in pop:
            compiled = compile_ast(c["canonical_ast"], FIELDS)
            self.assertLessEqual(compiled["node_count"], report["node_budget"])
            self.assertTrue(set(compiled["required_fields"]) <= set(FIELDS))
        self.assertEqual(len(report["node_counts"]), 24)

    def test_population_uses_only_given_fields(self):
        pop, _ = random_population(FIELDS, 16, 11)
        for c in pop:
            self.assertTrue(fields_used(c["canonical_ast"]) <= set(FIELDS))

    def test_time_operators_only_on_field_leaves(self):
        pop, _ = random_population(FIELDS, 16, 13)
        temporal = {"lag", "return", "mean", "std", "min", "max", "delta"}

        def walk(node):
            if "field" in node:
                return
            if node["op"] in temporal:
                self.assertEqual(set(node["args"][0]), {"field"})
                self.assertIn(node["window"], (2, 3, 5, 7, 10, 15, 20))
            for arg in node["args"]:
                walk(arg)

        for c in pop:
            walk(c["canonical_ast"])

    def test_action_and_scope_metadata(self):
        pop, _ = random_population(FIELDS, 4, 3)
        for c in pop:
            self.assertEqual(c["action"], "calibration_random_null")
            self.assertEqual(c["hypothesis"]["claim_type"], "correlational_hypothesis")

    def test_random_ast_rejects_unknown_field_space(self):
        ast = random_ast({"close": {"unit": "price"}}, 5)
        self.assertIsNotNone(ast)
        self.assertTrue(fields_used(ast) <= {"close"})

    def test_baseline_summary_quantiles_and_missing_filter(self):
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, float("nan")]
        s = baseline_summary(values)
        self.assertEqual(s["n"], 11)
        self.assertEqual(s["min"], 0.0)
        self.assertEqual(s["max"], 10.0)
        self.assertAlmostEqual(s["p50"], 5.0)
        self.assertAlmostEqual(s["mean"], 5.0)
        self.assertIsNotNone(s["sd"])
        self.assertEqual(baseline_summary([None, float("nan")]), {"n": 0})

    def test_percentile_of_positions_reading_in_baseline(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile_of(2.5, xs), 0.5)
        self.assertAlmostEqual(percentile_of(0.0, xs), 0.0)
        self.assertAlmostEqual(percentile_of(9.0, xs), 1.0)
        self.assertIsNone(percentile_of(None, xs))
        self.assertIsNone(percentile_of(1.0, []))

    def test_smoke_many_seeds_stay_legal(self):
        for seed in range(20):
            pop, _ = random_population(FIELDS, 6, seed, max_depth=4)
            for c in pop:
                compile_ast(c["canonical_ast"], FIELDS)


if __name__ == "__main__":
    unittest.main()
