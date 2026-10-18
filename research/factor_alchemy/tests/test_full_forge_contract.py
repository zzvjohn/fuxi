import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forge.expression import ExpressionEvaluator, grow_random_tree, parse_expression
from forge.primitives import ALL_PRIMITIVES, available_input_primitives
from weekly_panel import eval_weekly_formula


class FullForgeContractTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            'FUXI_FORGE_NUMERIC_CONTRACT': 'enforce',
            'FUXI_FORGE_FIELD_CONTRACT': 'enforce',
            'FUXI_FORGE_CS_COMPILE': 'enforce',
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        rng = np.random.RandomState(20260923)
        values = rng.normal(size=(90, 5))
        values[3, :] = 1.0
        values[8:12, 0] = np.nan
        values[4, 1] = 0
        values[12, 2] = -1e-12
        values[17, 4] = np.inf
        self.frame = pd.DataFrame(values, index=pd.date_range('2020-01-01', periods=90),
                                  columns=list('abcde'))
        self.data = {p.name: values.copy() for p in available_input_primitives()}
        self.data['returns'] = (self.frame - self.frame.shift(1)).div(self.frame.shift(1)).to_numpy()
        self.context = {'np': np, 'pd': pd}
        self.context.update({name + '_p': pd.DataFrame(v, index=self.frame.index,
                                                     columns=self.frame.columns)
                             for name, v in self.data.items()})

    def compare(self, expression):
        node = parse_expression(expression)
        self.assertIsNotNone(node, expression)
        with np.errstate(all='ignore'):
            expected = ExpressionEvaluator(self.data).evaluate(node)
            actual = eval(node.to_pandas_string(), self.context)
        self.assertIsInstance(actual, pd.DataFrame, expression)
        self.assertTrue(actual.index.equals(self.frame.index), expression)
        self.assertTrue(actual.columns.equals(self.frame.columns), expression)
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-10,
                                   atol=1e-10, equal_nan=True, err_msg=expression)

    def test_every_registered_operator_and_constant_input(self):
        count = 0
        for category, primitives in ALL_PRIMITIVES.items():
            if category == 'input':
                continue
            for primitive in primitives:
                for operand in ('close', '0.0', '-0.5'):
                    if category == 'timeseries':
                        expr = f'{primitive.name}({operand}, 3)'
                    elif primitive.arity == 2:
                        expr = f'{primitive.name}({operand}, volume)'
                    else:
                        expr = f'{primitive.name}({operand})'
                    with self.subTest(expr=expr):
                        self.compare(expr)
                    count += 1
        self.assertEqual(count, 72)

    def test_nested_signed_missing_and_constant_expressions(self):
        expressions = [
            'ts_pct(ts_delta(close, 3), 5)', 'rank(ts_mean(close, 3))',
            'ts_rank(zscore(close), 5)', 'div(ts_std(close, 3), ts_mean(close, 3))',
            'ts_ema_decay(ts_mean(close, 3), 3)', 'ts_sum(2.0, 3)',
            'inv(sub(close, close))', 'scale(rank(-0.5))',
            'ts_pct(returns, 3)', 'ts_std(close, 120)',
        ]
        for expr in expressions:
            with self.subTest(expr=expr):
                self.compare(expr)

    def test_random_trees(self):
        rng = np.random.RandomState(47)
        for _ in range(60):
            tree = grow_random_tree(rng, max_depth=3, dim_prune='off')
            self.compare(tree.to_string())

    def test_canonical_truth_values_and_legacy_counterfactual(self):
        from forge import primitives as p
        x = np.array([[-2.], [-3.], [-1.], [-2.]])
        self.assertAlmostEqual(p._ts_pct(x, 2)[-1, 0], -1 / 3)
        self.assertAlmostEqual(p._ts_rank(np.ones((4, 1)), 3)[-1, 0], 2 / 3)
        self.assertTrue(np.isnan(p._ts_mean(np.array([[1.], [np.nan], [3.]]), 3)[-1, 0]))
        with patch.dict(os.environ, {'FUXI_FORGE_NUMERIC_CONTRACT': 'legacy'}):
            self.assertAlmostEqual(p._ts_pct(x, 2)[-1, 0], 1 / 3)

    def test_future_perturbation_cannot_change_past(self):
        node = parse_expression('ts_ema_decay(ts_mean(close, 3), 5)')
        original = ExpressionEvaluator(self.data).evaluate(node)
        changed = {k: v.copy() for k, v in self.data.items()}
        changed['close'][50:] = 10000.0
        actual = ExpressionEvaluator(changed).evaluate(node)
        np.testing.assert_allclose(original[:50], actual[:50], equal_nan=True)

    def test_data_dependent_windows_are_rejected_by_both_engines(self):
        for expression in ['ts_mean(close,volume)', 'ts_mean(close,ts_mean(volume,3))']:
            node = parse_expression(expression)
            with self.assertRaises(ValueError):
                node.to_pandas_string()
            with self.assertRaises(ValueError):
                ExpressionEvaluator(self.data).evaluate(node)

    def test_vwap_is_disabled_and_legacy_mapping_is_explicit(self):
        self.assertNotIn('vwap', [p.name for p in available_input_primitives()])
        node = parse_expression('rank(vwap)')
        with self.assertRaises(ValueError):
            node.to_pandas_string()
        with patch.dict(os.environ, {'FUXI_FORGE_FIELD_CONTRACT': 'legacy'}):
            self.assertIn('volume_p', node.to_pandas_string())

    def test_actual_weekly_formula_consumer_preserves_rolling_rank(self):
        node = parse_expression('ts_rank(close, 3)')
        panel = {'env': {k: v for k, v in self.context.items() if isinstance(v, pd.DataFrame)},
                 'universe': list(self.frame.columns)}
        ok, wide, _, reason = eval_weekly_formula(node.to_pandas_string(), panel)
        self.assertTrue(ok, reason)
        np.testing.assert_allclose(wide, ExpressionEvaluator(self.data).evaluate(node), equal_nan=True)

    def test_actual_daily_series_consumer_accepts_constant_nodes(self):
        from s5_joint_filter import S5JointFilter
        root = Path(__file__).resolve().parents[3]
        obj = S5JointFilter(cache_dir=root / 'reports' / 'drought_20260923' / 'test_tmp')
        frame = self.frame.copy()
        frame.index.name = 'trade_date'
        frame.columns.name = 'ts_code'
        obj._price_df = frame.stack().rename('close_p').reset_index()
        obj._loaded = True
        for expression in ['ts_sum(add(close,0.5),3)', 'div(ts_std(close,3),2)', 'ts_rank(close,3)']:
            node = parse_expression(expression)
            actual = obj.evaluate_factor_formula(node.to_pandas_string(), 'daily-contract')
            self.assertIsNotNone(actual, expression)
            wide = actual.pivot(index='trade_date', columns='ts_code', values='factor_value').reindex_like(frame)
            np.testing.assert_allclose(wide, ExpressionEvaluator(self.data).evaluate(node),
                                       equal_nan=True, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
