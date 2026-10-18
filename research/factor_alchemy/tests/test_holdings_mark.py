import importlib.util
import os
import sys
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from s5_joint_filter import (S5JointFilter, S5Result, _s5_holdings_mark,
                             _shadow_carry_forward_returns)
from multi_stage_validator import MultiStageValidator, ValidationResult


class HoldingsMarkTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / 'reports/holdings_20260923/test_tmp'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = root / uuid.uuid4().hex
        self.tmp.mkdir()
        self.assertTrue(self.tmp.resolve().is_relative_to(root.resolve()))
        self.addCleanup(self.tmp.rmdir)
        self.env = patch.dict(os.environ, {'FUXI_S5_COST_BPS': '0', 'FUXI_S5_ADJ_PRICE': 'on',
                                           'FUXI_S5_TRADABLE_GATE': 'on', 'FUXI_S5_SPARSE_DIAG': 'off',
                                           'FUXI_S5_HOLDINGS_MARK': 'enforce'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dates = pd.bdate_range('2026-01-05', '2026-01-23')
        self.codes = ['000001.SZ', '000002.SZ', '000003.SZ']
        self.price = pd.DataFrame([{'trade_date': day, 'ts_code': code, 'close': 100., 'close_p': 100.,
                                   'pre_close': 100., 'pre_close_p': 100., 'amount': 100000.}
                                  for day in self.dates for code in self.codes])
        self.signal = self.price[['trade_date', 'ts_code']].copy()
        self.signal['factor_value'] = self.signal.ts_code.map(dict(zip(self.codes, [3., 2., 1.])))
        self.obj = S5JointFilter(top_n=2, cache_dir=self.tmp)
        self.obj._price_df = self.price
        self.obj._universe = self.codes
        self.obj._get_limit_up_wide = lambda: None
        self.obj._index_df = pd.DataFrame({'trade_date': self.dates, 'idx_return': 0.})

    def run_signal(self, signal, mode='enforce'):
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': mode}):
            returns, _ = self.obj.backtest_factor(signal)
        return returns

    def test_future_signal_deletion_cannot_remove_positive_or_negative_held_return(self):
        day = pd.Timestamp('2026-01-16')
        for change in (.01, -.01):
            self.price.loc[(self.price.trade_date == day) & (self.price.ts_code == self.codes[0]), 'close_p'] = 100 * (1 + change)
            complete = self.run_signal(self.signal)
            missing = self.signal[~((self.signal.trade_date == day) & (self.signal.ts_code == self.codes[0]))]
            fixed = self.run_signal(missing)
            pd.testing.assert_series_equal(complete.loc[:day], fixed.loc[:day])
            self.assertAlmostEqual(fixed.loc[day], change / 2)
            self.assertAlmostEqual(self.run_signal(missing, 'legacy').loc[day], 0.)

    def test_calendar_does_not_move_when_entire_friday_signal_disappears(self):
        missing = self.signal[self.signal.trade_date != pd.Timestamp('2026-01-16')]
        full = self.run_signal(self.signal)
        fixed = self.run_signal(missing)
        pd.testing.assert_series_equal(full, fixed)
        self.assertEqual(self.obj._last_backtest_diag['carried_periods'], 1)

    def test_entire_missing_signal_week_carries_existing_holdings(self):
        missing = self.signal[~self.signal.trade_date.between('2026-01-12', '2026-01-16')]
        fixed = self.run_signal(missing)
        self.assertEqual(len(fixed), 10)
        self.assertEqual(self.obj._last_backtest_diag['n_periods_skipped'], 1)
        self.assertFalse(self.obj._last_backtest_diag['coverage_ok'])

    def test_real_missing_price_is_unknown_not_zero_or_reweighted(self):
        day = pd.Timestamp('2026-01-14')
        self.obj._price_df = self.price[~((self.price.trade_date == day) & (self.price.ts_code == self.codes[0]))]
        fixed = self.run_signal(self.signal)
        self.assertTrue(np.isnan(fixed.loc[day]))
        self.assertFalse(self.obj._last_backtest_diag['accounting_ok'])
        self.assertEqual(self.obj._last_backtest_diag['missing_held_price_cells'], 1)
        _, _, m26 = self.obj._backtest_and_metrics(self.signal)
        self.assertEqual(m26['error'], 'held_price_observations_missing')
        fake = {'excess_return': 1., 'calmar': 2.}
        self.assertFalse(self.obj._verdict(fake, {**fake, **{'error': m26['error']}})[0])

    def test_all_held_quotes_missing_keeps_the_calendar_day(self):
        day = pd.Timestamp('2026-01-14')
        self.obj._price_df = self.price[~((self.price.trade_date == day) & self.price.ts_code.isin(self.codes[:2]))]
        fixed = self.run_signal(self.signal)
        self.assertIn(day, fixed.index)
        self.assertTrue(np.isnan(fixed.loc[day]))
        self.assertEqual(self.obj._last_backtest_diag['missing_held_price_cells'], 2)

    def test_complete_prices_equal_independent_daily_equal_weight_reference(self):
        day = pd.Timestamp('2026-01-14')
        self.price.loc[self.price.trade_date == day, 'close_p'] = [102., 99., 103.]
        with patch.dict(os.environ, {'FUXI_S5_COST_BPS': '15'}):
            fixed = self.run_signal(self.signal)
            legacy = self.run_signal(self.signal, 'legacy')
        expected = pd.Series(0., index=self.dates[self.dates > '2026-01-09'], name='daily_ret')
        expected.loc['2026-01-12'] = -.0015
        expected.loc[day] = (.02 - .01) / 2
        np.testing.assert_allclose(fixed.values, expected.values, atol=1e-12)
        pd.testing.assert_series_equal(fixed, legacy, check_freq=False)
        with patch.dict(os.environ, {'FUXI_S5_ADJ_PRICE': 'off', 'FUXI_S5_COST_BPS': '0'}):
            raw = self.run_signal(self.signal)
            self.assertIn(pd.Timestamp('2026-01-12'), raw.index)
            self.assertTrue(np.isfinite(raw).all())
            legacy = self.run_signal(self.signal, 'legacy')
            self.assertTrue(np.isnan(legacy.loc['2026-01-12']))

    def test_invalid_mode_and_duplicate_keys_fail_closed(self):
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': 'enfroce'}):
            self.assertRaises(ValueError, _s5_holdings_mark)
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': 'shadow_carry_forward'}):
            self.assertEqual(_s5_holdings_mark(), 'shadow_carry_forward')
        duplicate = pd.concat([self.signal, self.signal.iloc[:1]])
        self.assertRaises(ValueError, self.run_signal, duplicate)

    def test_new_contract_is_in_default_profiles_override_and_resume_identity(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('holdings_fullpool', root / 'scripts/s5_fullpool_rerun.py')
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(root / 'scripts'), *sys.path]):
            spec.loader.exec_module(module)
        key = 'FUXI_S5_HOLDINGS_MARK'
        clean = {k: v for k, v in os.environ.items() if not k.startswith('FUXI_')}
        for caliber, expected in [('default', 'enforce'), ('p0', 'enforce'), ('legacy', 'legacy'), ('legacy_e2e', 'legacy')]:
            with patch.dict(os.environ, clean, clear=True):
                module.apply_caliber(caliber)
                self.assertEqual(_s5_holdings_mark(), expected)
                self.assertEqual(module.semantic_contracts()[key], expected)
        with patch.dict(os.environ, {**clean, key: 'legacy'}, clear=True):
            module.apply_caliber('p0')
            self.assertEqual(_s5_holdings_mark(), 'legacy')
            path = module.caliber_out_path('p0')
        with patch.dict(os.environ, clean, clear=True):
            self.assertNotEqual(path, module.caliber_out_path('p0'))

    def test_partial_s5_measurement_with_missing_prices_is_not_valid_evidence(self):
        self._assert_consumer_status('held_price_observations_missing', 'data_insufficient', False)
        self._assert_consumer_status('benchmark_dates_missing', 'data_insufficient', False)

    def test_partial_s5_measurement_with_runtime_error_is_not_valid_evidence(self):
        self._assert_consumer_status('unexpected exception', 'error', False)

    def test_successful_measured_s5_keeps_valid_status(self):
        self._assert_consumer_status('', 'valid', True)

    def test_missing_mark_diagnostics_reach_serialized_consumer_without_promotion(self):
        mark = {'missing_held_price_cells': 2,
                'missing_held_price_cells_by_year': {'2026': 2},
                'incomplete_marking_days': 1,
                'marking_samples': [{'date': '2026-01-14', 'missing': 2}],
                'weighting_contract': 'daily_equal_weight_membership_weekly; not a fixed-share NAV ledger'}
        measured = S5Result(factor_name='candidate', passed=False,
                            reason='held prices unavailable',
                            error='held_price_observations_missing',
                            marking_diagnostics=mark,
                            economic_readings_status='diagnostic_only_data_insufficient',
                            excess_2026=-0.2)
        self.assertEqual(measured.to_dict()['marking_diagnostics'], mark)
        row = ValidationResult('candidate')
        MultiStageValidator()._apply_stage5_from_s5result(measured, [row], 0)
        self.assertEqual(row.s5_eval_status, 'data_insufficient')
        self.assertFalse(row.s5_passed)
        self.assertEqual(row.to_dict()['s5']['marking_diagnostics'], mark)
        self.assertEqual(row.to_dict()['s5']['economic_readings_status'],
                         'diagnostic_only_data_insufficient')

    def _assert_consumer_status(self, error, status, passed):
        measured = SimpleNamespace(elapsed_sec=1., window_days_2025=243, window_days_2026=167,
                                   reason='measurement', error=error, passed=True,
                                   excess_2025=.1, excess_2026=.1, calmar_2025=1., calmar_2026=1.)
        result = ValidationResult('candidate')
        MultiStageValidator()._apply_stage5_from_s5result(measured, [result], 0)
        self.assertEqual(result.s5_eval_status, status)
        self.assertEqual(result.s5_passed, passed)


class _ValidateStub(S5JointFilter):
    """M6b: validate_factor 全链路桩 — 只替换数据加载/求值/守卫, 其余真实代码。"""

    @property
    def is_ready(self):
        return True

    def _load_data(self):
        return None

    def _check_formula_sanity(self, formula, factor_name, skip_simplify=False):
        return ""

    def evaluate_factor_formula(self, formula, factor_name="unknown"):
        return self._stub_signal

    def _value_guards(self, factor_df):
        return ""

    def _daily_ic_series(self, factor_df, *a, **k):
        return pd.Series(dtype=float)

    def _verdict(self, m25, m26, label=""):
        return True, "synthetic pass"


class ShadowCarryForwardTests(unittest.TestCase):
    """M6b: FUXI_S5_HOLDINGS_MARK=shadow_carry_forward 契约测试。

    · 只填「行情缺失格」(面板无行) 且限段内前 20 个交易日; 超限保持缺失;
    · 缺格计数保留披露 (不得因影子档放宽);
    · 影子读数 passed 恒 False + reason 标注 + economic_readings_status
      == "measured_shadow"; enforce 路径逐位不变 (既有 12 测试即回归)。
    """

    def setUp(self):
        root = Path(__file__).resolve().parents[3] / 'reports/holdings_20260923/test_tmp'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = root / uuid.uuid4().hex
        self.tmp.mkdir()
        self.addCleanup(self.tmp.rmdir)
        self.env = patch.dict(os.environ, {'FUXI_S5_COST_BPS': '0', 'FUXI_S5_ADJ_PRICE': 'on',
                                           'FUXI_S5_TRADABLE_GATE': 'on', 'FUXI_S5_SPARSE_DIAG': 'off',
                                           'FUXI_S5_HOLDINGS_MARK': 'enforce'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dates = pd.bdate_range('2026-01-05', '2026-01-23')
        self.codes = ['000001.SZ', '000002.SZ', '000003.SZ']
        self.price = pd.DataFrame([{'trade_date': day, 'ts_code': code, 'close': 100., 'close_p': 100.,
                                   'pre_close': 100., 'pre_close_p': 100., 'amount': 100000.}
                                  for day in self.dates for code in self.codes])
        self.signal = self.price[['trade_date', 'ts_code']].copy()
        self.signal['factor_value'] = self.signal.ts_code.map(dict(zip(self.codes, [3., 2., 1.])))
        self.obj = S5JointFilter(top_n=2, cache_dir=self.tmp)
        self.obj._price_df = self.price
        self.obj._universe = self.codes
        self.obj._get_limit_up_wide = lambda: None
        self.obj._index_df = pd.DataFrame({'trade_date': self.dates, 'idx_return': 0.})

    def run_signal(self, signal, mode='enforce'):
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': mode}):
            returns, _ = self.obj.backtest_factor(signal)
        return returns

    def test_shadow_fills_missing_quote_cells_and_keeps_disclosure(self):
        day = pd.Timestamp('2026-01-14')
        self.obj._price_df = self.price[~((self.price.trade_date == day)
                                          & (self.price.ts_code == self.codes[0]))]
        enforce = self.run_signal(self.signal, 'enforce')
        self.assertTrue(np.isnan(enforce.loc[day]))          # enforce: 缺格 ⇒ 当日不可计
        shadow = self.run_signal(self.signal, 'shadow_carry_forward')
        self.assertEqual(shadow.loc[day], 0.)                # shadow: carry ⇒ 该格记 0
        d = self.obj._last_backtest_diag
        self.assertEqual(d['missing_held_price_cells'], 1)   # 披露保留
        self.assertEqual(d['carried_forward_cells'], 1)
        self.assertEqual(d['carry_forward_exceeded_cells'], 0)
        self.assertFalse(d['accounting_ok'])
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': 'shadow_carry_forward'}):
            _, _m25, m26 = self.obj._backtest_and_metrics(self.signal)
        self.assertNotIn('error', m26)                       # 不再阻断
        self.assertEqual(m26.get('missing_held_price_cells'), 1)

    def test_shadow_helper_k_cap_and_clipped_boundary(self):
        held = pd.DataFrame(np.nan, index=range(6), columns=['a', 'b'])
        held.loc[0, 'a'] = .01
        held.loc[2, 'b'] = .03
        held.loc[5, 'a'] = .02
        absent = pd.DataFrame(False, index=range(6), columns=['a', 'b'])
        absent.loc[1:2, 'a'] = True
        absent.loc[4, 'a'] = True
        absent.loc[[0, 1, 3, 4, 5], 'b'] = True
        out, stat = _shadow_carry_forward_returns(held, absent, k=1)
        # a: 段 {1,2} 与 {4} 各自只补第 1 格 ⇒ a1 补 0, a2 超限留 NaN, a3 守卫格不动
        self.assertEqual(out.loc[1, 'a'], 0.)
        self.assertTrue(np.isnan(out.loc[2, 'a']))
        self.assertTrue(np.isnan(out.loc[3, 'a']))
        self.assertEqual(out.loc[4, 'a'], 0.)
        self.assertEqual(out.loc[5, 'a'], .02)
        # b: 段 {0,1} 补 1 格; 段 {3,4,5} 补 1 格
        self.assertEqual(out.loc[0, 'b'], 0.)
        self.assertTrue(np.isnan(out.loc[1, 'b']))
        self.assertEqual(out.loc[3, 'b'], 0.)
        self.assertTrue(np.isnan(out.loc[4, 'b']))
        self.assertTrue(np.isnan(out.loc[5, 'b']))
        self.assertEqual(stat, {'carried': 4, 'exceeded': 4})

    def test_shadow_pins_passed_false_and_status_measured_shadow(self):
        stub = _ValidateStub(top_n=2, cache_dir=self.tmp)
        stub._price_df = self.price
        stub._universe = self.codes
        stub._get_limit_up_wide = lambda: None
        stub._index_df = pd.DataFrame({'trade_date': self.dates, 'idx_return': 0.})
        stub._stub_signal = self.signal
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': 'enforce'}):
            control = stub.validate_factor('t_enforce', 'close')
        self.assertTrue(control.passed)                      # 桩裁决 True ⇒ 对照组可达
        self.assertEqual(control.economic_readings_status, 'measured')
        with patch.dict(os.environ, {'FUXI_S5_HOLDINGS_MARK': 'shadow_carry_forward'}):
            shadow = stub.validate_factor('t_shadow', 'close')
        self.assertFalse(shadow.passed)                      # guard: 永不授予
        self.assertEqual(shadow.error, '')
        self.assertEqual(shadow.economic_readings_status, 'measured_shadow')
        self.assertTrue(shadow.reason.startswith('🧪 影子计价读数'))
        self.assertIn('synthetic pass', shadow.reason)


if __name__ == '__main__':
    unittest.main()
