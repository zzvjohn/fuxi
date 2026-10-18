import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from weekly_panel import weekly_ic

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('drought_campaign', ROOT / 'scripts' / 'fuxi_drought_campaign_20260923.py')
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class ExperimentControls(unittest.TestCase):
    def test_synthetic_positive_and_shuffled_negative_control(self):
        # Deliberately label-derived signal, only to verify the measurement chain.
        rng = np.random.default_rng(20260923)
        labels = pd.DataFrame(rng.normal(size=(80, 100)))
        panel = {'fwd_ret': labels}
        positive = weekly_ic(panel, labels)
        negative = weekly_ic(panel, pd.DataFrame(rng.permuted(labels.to_numpy(), axis=1)))
        self.assertEqual(positive['weekly_n'], 80)
        self.assertAlmostEqual(positive['weekly_ic'], 1.)
        self.assertEqual(negative['weekly_n'], 80)
        self.assertLess(abs(negative['weekly_ic']), .04)

    def test_llm_expression_guard(self):
        check = CAMPAIGN.experiment_formula_issues
        self.assertEqual(check('close.pct_change(fill_method=None).rank(axis=1,pct=True)'), [])
        self.assertEqual(check('close.sub(low).div(high.sub(low).replace(0,np.nan))'), [])
        for expression in ['open("secret")', '__import__("os").system("cmd")',
                           'close.shift(-1)', 'close.pct_change(1)',
                           'close.fillna(0)', 'close.rolling(4,center=True).mean()',
                           'close.rank(pct=True)', 'close.to_csv("x")']:
            self.assertTrue(check(expression), expression)

    def test_broadcast_repair_preserves_non_series_math(self):
        self.assertIsNone(CAMPAIGN.repair_cs_broadcast('close_p - low_p'))
        fixed = CAMPAIGN.repair_cs_broadcast('close_p - close_p.mean(axis=1)')
        self.assertIn('.sub(', fixed)
        self.assertIn('axis=0', fixed)

    def test_version_render_keeps_non_bump_research_notes(self):
        spec = importlib.util.spec_from_file_location('version_bump', ROOT / 'scripts' / 'version_bump.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rendered = module.render_changelog({'versions': [{
            'version': 'v0.7.52', 'date': '2026-09-22',
            'non_bump': ['research-note-sentinel'], 'proposals': []}]}, [])
        self.assertIn('research-note-sentinel', rendered)

    def test_ridge_future_features_and_labels_cannot_change_frozen_weights(self):
        rng = np.random.default_rng(23)
        weeks = pd.date_range('2023-01-01', periods=200, freq='W-SUN')
        availability = pd.Series(weeks, index=weeks).shift(-1)
        values = rng.normal(size=(200, 20, 3))
        target = pd.DataFrame(rng.normal(size=(200, 20)), index=weeks)
        first = CAMPAIGN.fit_frozen_ridge(values, target, availability, '2024-12-31', 100.)
        future = (availability > '2024-12-31').to_numpy()
        changed = values.copy()
        changed[future] = 10000.
        target.loc[future] = -10000.
        second = CAMPAIGN.fit_frozen_ridge(changed, target, availability, '2024-12-31', 100.)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertGreater(first[1], 1000)


if __name__ == '__main__':
    unittest.main()
