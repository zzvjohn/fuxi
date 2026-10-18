import os
import sys
import importlib.util
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from s5_joint_filter import S5JointFilter
from weekly_panel import align_weekly_to_daily, build_weekly_panel


class DroughtS5ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parents[3] / 'reports' / 'drought_20260923' / 'test_tmp' / uuid.uuid4().hex
        self.tmp.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_epsilon_is_not_a_price_level_proxy(self):
        obj = S5JointFilter(cache_dir=self.tmp)
        formula = '(volume_p.rolling(3).mean() / (volume_p + 1e-6)).rank(pct=True, axis=1)'
        with patch.dict(os.environ, {'FUXI_S5_SANITY_AST': 'legacy'}):
            self.assertIn('价格代理', obj._check_formula_sanity(formula, 'test'))
        with patch.dict(os.environ, {'FUXI_S5_SANITY_AST': 'enforce'}):
            self.assertIsNone(obj._check_formula_sanity(formula, 'test'))
            self.assertIn('价格代理', obj._check_formula_sanity('close_p - 2.0', 'test'))
            self.assertIn('量纲混合', obj._check_formula_sanity('close_p + close_p.pct_change(3)', 'test'))

    def test_weekly_enforce_never_falls_back_to_daily(self):
        with patch.dict(os.environ, {'FUXI_S5_WEEKLY_CHANNEL': 'enforce'}):
            obj = S5JointFilter(cache_dir=self.tmp)
        with patch.object(obj, '_load_data'), patch.object(obj, '_check_formula_sanity', return_value=None), \
                patch.object(obj, 'evaluate_factor_formula_weekly', return_value=(None, {}, 'missing')), \
                patch.object(obj, 'evaluate_factor_formula') as daily:
            result = obj.validate_factor(factor_name='test', formula='close_p.rolling(3).mean().rank(axis=1)', natural_freq='weekly')
        self.assertFalse(result.passed)
        self.assertEqual(result.error, 'weekly_data_unavailable')
        daily.assert_not_called()

    def test_weekly_loader_failure_cannot_change_later_candidate_frequency(self):
        with patch.dict(os.environ, {'FUXI_S5_WEEKLY_CHANNEL': 'enforce'}):
            obj = S5JointFilter(cache_dir=self.tmp)
        obj._price_df = pd.DataFrame(columns=['ts_code', 'trade_date', 'close'])
        with patch.object(obj, '_load_data'), patch.object(obj, '_check_formula_sanity', return_value=None), \
                patch('weekly_panel.build_weekly_panel', side_effect=RuntimeError('fixture failure')), \
                patch.object(obj, 'evaluate_factor_formula') as daily:
            for name in ('first', 'second'):
                result = obj.validate_factor(factor_name=name, formula='close.rolling(3).mean()', natural_freq='weekly')
                self.assertEqual(result.error, 'weekly_data_unavailable')
                self.assertEqual(obj.weekly_channel, 'enforce')
        daily.assert_not_called()

    def test_missing_week_expires_previous_signal(self):
        weeks = pd.date_range('2025-01-05', periods=3, freq='W-SUN')
        fridays = weeks - pd.Timedelta(days=2)
        panel = {'week_last_day': pd.Series(fridays, index=weeks)}
        signals = pd.DataFrame({'stock_code': ['a', 'a'], 'trade_date': [weeks[0], weeks[2]], 'factor_value': [2., 3.]})
        keys = pd.DataFrame({'ts_code': 'a', 'trade_date': pd.date_range(fridays[0], fridays[-1], freq='B')})
        with patch.dict(os.environ, {'FUXI_WEEKLY_MISSING_CONTRACT': 'enforce'}):
            result = align_weekly_to_daily(signals, panel, keys)
        self.assertFalse(((result.trade_date >= fridays[1]) & (result.trade_date < fridays[2])).any())
        with patch.dict(os.environ, {'FUXI_WEEKLY_MISSING_CONTRACT': 'legacy'}):
            legacy = align_weekly_to_daily(signals, panel, keys)
        self.assertGreater(len(legacy), len(result))

    def test_all_missing_volume_does_not_become_zero(self):
        daily = pd.DataFrame({'ts_code': ['a'] * 3, 'trade_date': pd.date_range('2025-01-06', periods=3),
                              'close': [1., 2., 3.], 'volume': [np.nan] * 3})
        with patch.dict(os.environ, {'FUXI_WEEKLY_MISSING_CONTRACT': 'enforce'}):
            panel = build_weekly_panel(daily)
        self.assertTrue(panel['env']['volume'].isna().all().all())

    def test_metrics_use_actual_portfolio_dates(self):
        obj = S5JointFilter(cache_dir=self.tmp)
        dates = pd.date_range('2025-01-01', periods=80, freq='B')
        obj._index_df = pd.DataFrame({'trade_date': dates, 'idx_return': 0.001})
        returns = pd.Series(0.002, index=dates[10:70])
        with patch.dict(os.environ, {'FUXI_S5_RETURN_ALIGN': 'on'}):
            matched = obj.compute_metrics(returns, 2025)
        with patch.dict(os.environ, {'FUXI_S5_RETURN_ALIGN': 'off'}):
            old = obj.compute_metrics(returns, 2025)
        self.assertAlmostEqual(matched['benchmark_return'], 1.001 ** 60 - 1)
        self.assertAlmostEqual(old['benchmark_return'], 1.001 ** 80 - 1)
        obj._index_df = obj._index_df.iloc[15:]
        with patch.dict(os.environ, {'FUXI_S5_RETURN_ALIGN': 'on'}):
            missing = obj.compute_metrics(returns, 2025)
        self.assertEqual(missing['error'], 'benchmark_dates_missing')
        obj._index_df = pd.DataFrame({'trade_date': dates - pd.DateOffset(years=1), 'idx_return': .001})
        with patch.dict(os.environ, {'FUXI_S5_RETURN_ALIGN': 'on'}):
            self.assertEqual(obj.compute_metrics(returns, 2025)['error'], 'benchmark_dates_missing')

    def test_alignment_excludes_unobserved_portfolio_dates_and_reports_actual_end(self):
        obj = S5JointFilter(cache_dir=self.tmp)
        dates = pd.date_range('2025-01-01', periods=80, freq='B')
        obj._index_df = pd.DataFrame({'trade_date': dates, 'idx_return': .001})
        returns = pd.Series(.002, index=dates)
        returns.iloc[-1] = np.nan
        with patch.dict(os.environ, {'FUXI_S5_RETURN_ALIGN': 'on'}):
            result = obj.compute_metrics(returns, 2025)
        self.assertEqual(result['n_periods'], 79)
        self.assertEqual(result['window_end'], str(dates[-2].date()))
        self.assertAlmostEqual(result['benchmark_return'], 1.001 ** 79 - 1)

    def test_invalid_weekly_mode_is_not_a_silent_frequency_change(self):
        with patch.dict(os.environ, {'FUXI_S5_WEEKLY_CHANNEL': 'enfroce'}):
            with self.assertRaises(ValueError):
                S5JointFilter(cache_dir=self.tmp)

    def test_default_and_p0_caliber_match_and_ab_overrides_survive(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('fullpool_rerun', root / 'scripts' / 's5_fullpool_rerun.py')
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(root / 'scripts'), *sys.path]):
            spec.loader.exec_module(module)
        keys = {'FUXI_S5_WEEKLY_CHANNEL': 'enforce', 'FUXI_S5_RETURN_ALIGN': 'on',
                'FUXI_S5_SANITY_AST': 'enforce', 'FUXI_WEEKLY_MISSING_CONTRACT': 'enforce',
                'FUXI_S5_RANK_AXIS': 'enforce'}
        clean = {k: v for k, v in os.environ.items() if not k.startswith('FUXI_')}
        for caliber in ('default', 'p0'):
            with patch.dict(os.environ, clean, clear=True):
                module.apply_caliber(caliber)
                obj = S5JointFilter(cache_dir=self.tmp)
                self.assertEqual(obj.weekly_channel, 'enforce')
                for key, default in keys.items():
                    self.assertEqual(os.environ.get(key, default), default)
                self.assertEqual(obj._effective_calmar_threshold(), .7)
        overrides = dict(zip(keys, ['off', 'off', 'legacy', 'legacy', 'legacy']))
        for caliber in ('legacy', 'legacy_e2e', 'p0'):
            with patch.dict(os.environ, {**clean, **overrides}, clear=True):
                module.apply_caliber(caliber)
                self.assertEqual({k: os.environ[k] for k in overrides}, overrides)
        with patch.dict(os.environ, clean, clear=True):
            module.apply_caliber('legacy')
            self.assertEqual(os.environ['FUXI_S5_WEEKLY_CHANNEL'], 'shadow')
            self.assertEqual(os.environ['FUXI_S5_RETURN_ALIGN'], 'off')
        fp = {'caliber': 'default', 'date_ceil': 'auto', 'idx_align': 'off', 'semantic_contracts': dict(keys)}
        code = {'fixture.py': 'hash'}
        previous = {**fp, 'code_fingerprint': code, 'results': [{}]}
        self.assertEqual(module._resume_decision(previous, fp, code)[0], 1)
        changed = {**fp, 'semantic_contracts': {**keys, 'FUXI_S5_RETURN_ALIGN': 'off'}}
        self.assertIsNone(module._resume_decision(previous, changed, code)[0])
        self.assertIsNone(module._resume_decision({**fp, 'results': [{}]}, fp, code)[0])
        with patch.dict(os.environ, clean, clear=True):
            normal_path = module.caliber_out_path('default')
            os.environ['FUXI_S5_RANK_AXIS'] = 'legacy'
            self.assertNotEqual(module.caliber_out_path('default'), normal_path)

    def test_llm_schema_distinguishes_pandas_methods_from_bare_dsl(self):
        from llm_generator import validate_llm_schema
        valid = 'close.sub(low).div(high.sub(low).add(1e-8)).mul(-1).rank(axis=1,pct=True)'
        self.assertFalse(any('DSL' in x for x in validate_llm_schema(valid, weekly_mode=True)))
        self.assertTrue(any('DSL' in x for x in validate_llm_schema('sub(close, low)', weekly_mode=True)))

    def test_daily_rank_axis_is_causal_and_rolling_rank_is_preserved(self):
        obj = S5JointFilter(cache_dir=self.tmp)
        obj._loaded = True
        rng = np.random.default_rng(72)
        dates = pd.date_range('2025-01-01', periods=100)
        prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(scale=.02, size=(100, 5)), axis=0)),
                              index=dates, columns=list('abcde'))
        prices.index.name = 'trade_date'
        prices.columns.name = 'ts_code'
        formula = 'close_p.pct_change(3, fill_method=None).rank(pct=True)'

        def evaluate(frame, mode):
            obj._price_df = frame.stack().rename('close_p').reset_index()
            with patch.dict(os.environ, {'FUXI_S5_RANK_AXIS': mode}), \
                    patch.object(obj, '_load_fundamental_pivots', return_value={}):
                result = obj.evaluate_factor_formula(formula, 'causal-rank')
            return result.pivot(index='trade_date', columns='ts_code', values='factor_value').reindex_like(prices)

        changed = prices.copy()
        changed.iloc[60:] = 10000.
        a, b = evaluate(prices, 'enforce'), evaluate(changed, 'enforce')
        np.testing.assert_allclose(a.iloc[:60], b.iloc[:60], equal_nan=True)
        old_a, old_b = evaluate(prices, 'legacy'), evaluate(changed, 'legacy')
        self.assertFalse(np.allclose(old_a.iloc[:60], old_b.iloc[:60], equal_nan=True))
        rolling = 'close_p.rolling(3).rank(pct=True, method="average")'
        self.assertEqual(obj._normalize_rank_axis(rolling), rolling)
        with self.assertRaises(ValueError):
            obj._normalize_rank_axis('close_p.rank(axis=0)')


if __name__ == '__main__':
    unittest.main()
