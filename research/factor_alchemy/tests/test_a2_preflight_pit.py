"""A2 contracts: all generators, causality and a prospective selection pool."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from candidate_preflight import precheck_candidates
from formula_quality_precheck import assess_candidate
from point_in_time_probe import probe_future_row_invariance


class A2PreflightTests(unittest.TestCase):
    def test_generator_sources_share_same_hard_defect_contract(self):
        rows = [{"factor_name": "bad", "formula": "close - close"},
                {"factor_name": "good", "formula": "close.rank(pct=True)"}]
        for source in ("llm", "gp_breed", "forge", "gp_breed_fallback", "gp_evolve"):
            with self.subTest(source=source):
                kept, audit = precheck_candidates(rows, source=source, mode="enforce")
                self.assertEqual([c["factor_name"] for c in kept], ["good"])
                self.assertEqual(audit["rejected"], 1)
                self.assertEqual(kept[0]["raw_formula"], "close.rank(pct=True)")
                self.assertIn("axis=1", kept[0]["formula"])
        self.assertNotIn("formula_quality", rows[0])

    def test_off_and_shadow_preserve_formula_and_economic_eligibility(self):
        row = {"formula": "close.rank(pct=True)", "semantic_target_use": "research_feature"}
        off, _ = precheck_candidates([row], source="forge", mode="off")
        shadow, _ = precheck_candidates([row], source="forge", mode="shadow")
        self.assertEqual(off, [row])
        self.assertEqual(shadow[0]["formula"], row["formula"])
        self.assertEqual(shadow[0]["semantic_target_use"], "research_feature")
        self.assertEqual(shadow[0]["formula_quality"]["status"], "pass")
        self.assertNotIn("eligible_for_jq", shadow[0])

    def test_literal_future_access_is_hard_and_dynamic_lag_only_review(self):
        for expression in ("close.shift(periods=-1)", "close.diff(periods=-2)",
                           "close.pct_change(periods=-1)", "close.fillna(method='bfill')",
                           "close.backfill()", "close.rolling(window=5, center=True).mean()"):
            with self.subTest(expression=expression):
                self.assertEqual(assess_candidate({"formula": expression})["status"], "reject")
        self.assertEqual(assess_candidate({"formula": "close.shift(periods=1)"})["status"], "pass")
        dynamic = assess_candidate({"formula": "close.shift(periods=k)"})
        self.assertEqual(dynamic["status"], "review")
        self.assertIn("lag_direction_unverified", dynamic["review_codes"])

    def test_future_rows_cannot_change_history(self):
        panel = pd.DataFrame({"A": [1., 2., 3., 4., 5.],
                              "B": [3., 2., 4., 5., 6.]},
                             index=pd.date_range("2025-01-01", periods=5))
        cutoff = panel.index[2]
        causal = probe_future_row_invariance(
            panel, cutoff, lambda x: x.pct_change(fill_method=None).rank(axis=1))
        self.assertEqual(causal["status"], "probe_pass")
        leaking = probe_future_row_invariance(panel, cutoff,
                                               lambda x: x.rank(axis=0))
        self.assertEqual(leaking["status"], "reject")
        self.assertGreater(leaking["changed_historical_cells"], 0)

    def test_hindsight_label_changes_top_but_not_prospective_top(self):
        spec = importlib.util.spec_from_file_location(
            "fuxi_model_compare_pool_shadow_a2", ROOT / "scripts/fuxi_model_compare_pool_shadow_20260926.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        codes = [f"c{i:03d}" for i in range(81)]
        frame = pd.DataFrame({"amount": np.ones(81)}, index=pd.Index(codes, name="ts_code"))
        score = pd.Series(np.arange(81, 0, -1), index=frame.index)
        label = pd.Series(np.zeros(81), index=frame.index)
        label.iloc[0] = np.nan
        result = module.compare_prospective_vs_hindsight_pool(frame, score, label)
        self.assertEqual(result["prospective_top_codes"][0], "c000")
        self.assertEqual(result["hindsight_top_codes"][0], "c001")
        self.assertEqual(result["prospective_unlabelled_top_codes"], ["c000"])
        self.assertEqual(result["top_overlap_n"], 79)


if __name__ == "__main__":
    unittest.main()
