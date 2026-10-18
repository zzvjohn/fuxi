"""Independent statistical contracts; no real-market research or qualification."""
from copy import deepcopy
from datetime import date, timedelta
import math
import unittest

import numpy as np

from fuxi_v1.contracts import ContractError
from fuxi_v1.full_a_statistics import (analyze_daily_rows, bootstrap_budget,
    bootstrap_diagnostic, calendar_values, calibration_planning, freeze_method,
    paired_account_diagnostic, summarize_support, validate_method)


def days(n):
    return [(date(2024, 1, 1)+timedelta(days=i)).isoformat() for i in range(n)]


class FullAStatisticsTests(unittest.TestCase):
    def test_resolution_and_precision_are_distinct_for_40_trials(self):
        old = bootstrap_budget(family_size=40, replicates=199)
        self.assertEqual(old['minimum_nonzero_p'], .005)
        self.assertAlmostEqual(old['expected_draws_per_corrected_ci_tail'], .125)
        self.assertFalse(old['one_sided_resolution_adequate'])
        self.assertEqual(old['one_sided_resolution_minimum_replicates'], 799)
        self.assertEqual(old['two_sided_resolution_minimum_replicates'], 1599)
        self.assertEqual(old['twenty_draw_tail_minimum_replicates'], 31999)
        self.assertEqual(old['hundred_draw_tail_minimum_replicates'], 159999)
        planned = bootstrap_budget()
        self.assertEqual(planned['expected_draws_per_corrected_ci_tail'], 100.)
        self.assertLess(planned['relative_tail_mc_se_approx'], .1)
        self.assertFalse(planned['adaptive_family_error_control_validated'])

    def test_method_is_bound_and_cannot_advertise_admission(self):
        spec = freeze_method(frozen_at='2026-10-09T00:00:00+00:00')
        self.assertTrue(spec['registered'])
        self.assertFalse(spec['formal_qualification_allowed'])
        self.assertEqual(validate_method(spec), spec['method_hash'])
        changed = deepcopy(spec); changed['block_lengths'] = [6]
        with self.assertRaises(ContractError): validate_method(changed)
        self.assertFalse(freeze_method()['registered'])

    def test_missing_dates_preserved_and_duplicate_dates_refused(self):
        calendar = days(4)
        dates, values = calendar_values([{'session': calendar[0], 'rank_ic': .1},
            {'session': calendar[3], 'rank_ic': -.1}], sessions=calendar)
        self.assertEqual(dates, calendar)
        self.assertEqual(values, [.1, None, None, -.1])
        with self.assertRaises(ContractError):
            calendar_values([{'session': calendar[0], 'rank_ic': .1}]*2)
        with self.assertRaises(ContractError):
            calendar_values([{'session': calendar[0], 'rank_ic': float('nan')}])

    def test_fast_bootstrap_matches_independent_explicit_circular_draws(self):
        values = [.1, .2, None, -.3, .4, .2, .1, .05, .02, .03, .04, .08, .09]
        actual = bootstrap_diagnostic(values, block_length=3, replicates=199, seed=9)
        rng = np.random.default_rng(9); expected = []
        for size in [128, 71]:
            for starts in rng.integers(0, len(values), size=(size, 5)):
                sampled = [values[(int(s)+j) % len(values)] for s in starts for j in range(3)][:len(values)]
                observed = [v for v in sampled if v is not None]
                expected.append(sum(observed)/len(observed))
        self.assertAlmostEqual(actual['nominal_interval']['lower'], float(np.quantile(expected, .025)), 12)
        self.assertAlmostEqual(actual['bonferroni_sensitivity_interval']['lower'],
                               float(np.quantile(expected, .05/40/2)), 12)
        self.assertEqual(actual['calendar_slots'], 13)
        self.assertEqual(actual['n_valid_days'], 12)
        self.assertFalse(actual['formal_qualification_allowed'])

    def test_unknown_support_and_nonfinite_series_are_not_economic_rejections(self):
        insufficient = bootstrap_diagnostic([None]*30, block_length=6, replicates=99)
        self.assertEqual(insufficient['eval_status'], 'data_insufficient')
        self.assertIsNone(insufficient['mean'])
        with self.assertRaises(ContractError):
            bootstrap_diagnostic([float('inf')]*30, block_length=6, replicates=99)

    def test_positive_ar_dependence_increases_planned_calendar_need(self):
        rng = np.random.default_rng(7)
        shocks = rng.normal(size=6000)
        dependent = []; x = 0.
        for shock in shocks:
            x = .8*x+float(shock); dependent.append(x)
        iid = calibration_planning(shocks.tolist())
        ar = calibration_planning(dependent)
        self.assertGreater(ar['estimates'][-1]['long_run_variance'],
                           iid['estimates'][-1]['long_run_variance']*8)
        self.assertGreater(ar['estimates'][-1]['required_calendar_sections_approx'],
                           iid['estimates'][-1]['required_calendar_sections_approx']*8)
        self.assertFalse(ar['is_admission_gate'])

    def test_55_day_calibration_discloses_short_window_and_real_missingness(self):
        values = [math.sin(i/3)*.1 for i in range(55)] + [None]*6
        plan = calibration_planning(values)
        self.assertEqual(plan['calibration_observations'], 55)
        self.assertEqual(plan['calendar_slots'], 61)
        self.assertEqual(plan['unknown_days'], 6)
        self.assertTrue(plan['short_calibration_window'])
        self.assertEqual(plan['planning_status'], 'short_window_uncertain')
        self.assertFalse(plan['formal_qualification_allowed'])
        empty = calibration_planning([None]*55)
        self.assertEqual(empty['planning_status'], 'data_insufficient')

    def test_constant_calibration_does_not_promise_two_day_confirmation(self):
        plan = calibration_planning([.1]*55)
        for estimate in plan['estimates']:
            self.assertEqual(estimate['status'], 'degenerate_variance')
            self.assertIsNone(estimate['required_calendar_sections_approx'])

    def test_all_frozen_blocks_remain_visible_without_qualification(self):
        calendar = days(60)
        rows = [{'session': d, 'rank_ic': .1 if i % 7 else None} for i, d in enumerate(calendar)]
        method = freeze_method(replicates=199)
        output = analyze_daily_rows(rows, method=method, sessions=calendar)
        self.assertEqual([b['block_length'] for b in output['blocks']], [6, 12, 24])
        self.assertEqual(output['valid_days'], 51)
        self.assertTrue(output['nominal_positive_all_fixed_blocks'])
        self.assertFalse(output['formal_qualification_allowed'])
        self.assertFalse(output['adaptive_family_error_control_validated'])

    def test_paired_costs_calendar_and_unknowns_are_checked(self):
        calendar = days(60)
        a = [{'session': d, 'net_return': .002 if i != 4 else None} for i, d in enumerate(calendar)]
        b = [{'session': d, 'net_return': .001} for d in calendar]
        output = paired_account_diagnostic(a, b, method=freeze_method(replicates=199),
            candidate_cost_profile_hash='a'*64, baseline_cost_profile_hash='a'*64)
        self.assertAlmostEqual(output['mean'], .001)
        self.assertEqual(output['unknown_days'], 1)
        self.assertFalse(output['account_paths_identified'])
        self.assertFalse(output['financial_increment_claim_validated'])
        with self.assertRaises(ContractError):
            paired_account_diagnostic(a, b, method=freeze_method(replicates=199),
                candidate_cost_profile_hash='a'*64, baseline_cost_profile_hash='b'*64)
        with self.assertRaises(ContractError):
            paired_account_diagnostic(a[:-1], b, method=freeze_method(replicates=199),
                candidate_cost_profile_hash='a'*64, baseline_cost_profile_hash='a'*64)

    def test_five_board_and_three_exchange_missing_groups_remain_explicit(self):
        rows = [{'session': '2024-12-31', 'rank_ic': .1, 'pair_count': 80, 'universe_count': 100},
                {'session': '2025-01-02', 'rank_ic': -.1, 'pair_count': 10, 'universe_count': 20}]
        north = [{'session': '2024-12-31', 'rank_ic': -.2, 'pair_count': 10, 'universe_count': 20}]
        output = summarize_support(rows, exchange_sections={'XBSE': north}, board_sections={'bj': north})
        self.assertEqual(output['overall']['security_label_pairs'], 90)
        self.assertAlmostEqual(output['overall']['pair_coverage'], .75)
        self.assertEqual(output['overall']['mean_rank_ic'], 0.)
        self.assertEqual(output['exchanges']['XBSE']['mean_rank_ic'], -.2)
        self.assertEqual(output['exchanges']['XBSE']['unreported_calendar_days'], 1)
        self.assertEqual(output['exchanges']['XBSE']['status'], 'incomplete_calendar_support')
        self.assertEqual(output['exchanges']['XSHG']['status'], 'not_provided')
        self.assertEqual(set(output['boards']), {'main_sh', 'main_sz', 'chinext', 'star', 'bj'})
        self.assertEqual(set(output['years']), {'2024', '2025'})
        with self.assertRaises(ContractError):
            summarize_support([dict(rows[0], pair_count=101)])


if __name__ == '__main__':
    unittest.main()
