import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corporate_action_inventory import (ADJ_ROUNDING_TOLERANCE, assess_row, identity_tolerance,
                                        merge_same_implementation, reference_identity,
                                        share_components)
from portfolio_execution_ledger import CorporateAction, run_ledger, reconcile_journal


class InventoryTests(unittest.TestCase):
    """Every positive case is a real event already blocked in the ledger."""

    def setUp(self):
        # Vendor rows exactly as returned by the dividend endpoint, and the
        # measured pre_close / adj_factor pair for the ex session.
        self.cases = [
            (dict(code='601800.SH', end_date='20251231', ex_date='20260109',
                  record_date='20260108', pay_date='20260109', cash_div_tax=0.1178,
                  stk_div=0.0, imp_ann_date='20251230'),
             dict(pre_close=8.42, adj_prev=1.4639, adj_ex=1.4851)),
            (dict(code='688697.SH', end_date='20241231', ex_date='20250618',
                  record_date='20250617', pay_date='20250618', cash_div_tax=0.60,
                  stk_div=0.4, stk_co_rate=0.4, imp_ann_date='20250612'),
             dict(pre_close=18.91, adj_prev=1.07, adj_ex=1.5469)),
            (dict(code='920871.BJ', end_date='20241231', ex_date='20250523',
                  record_date='20250522', pay_date='20250523', cash_div_tax=0.20,
                  stk_div=0.1, stk_co_rate=0.1, imp_ann_date='20250514'),
             dict(pre_close=17.18, adj_prev=1.3759, adj_ex=1.5310)),
        ]

    def test_real_events_satisfy_the_reference_identity(self):
        for row, measured in self.cases:
            verdict = assess_row(row, identity_proven=True, **measured)
            self.assertEqual(verdict.status, 'approved', verdict.reason)
            self.assertLess(verdict.detail['relative_error'], 0.001)
            self.assertGreater(verdict.detail['relative_error'], 0.0)

    def test_lone_component_is_only_a_fallback_when_total_is_absent(self):
        row, measured = self.cases[1]
        # A declared total is what the ledger consumes; the split is irrelevant.
        self.assertEqual(assess_row(row, identity_proven=False, **measured).status, 'approved')
        no_total = {k: v for k, v in row.items() if k != 'stk_div'}
        self.assertEqual(assess_row(no_total, identity_proven=False, **measured).reason,
                         'lone_component_without_identity_proof')
        self.assertEqual(assess_row(no_total, identity_proven=True, **measured).status, 'approved')
        self.assertEqual(share_components(row, True), (0.0, 0.4))

    def test_component_arithmetic_is_enforced(self):
        row, _ = self.cases[1]
        with self.assertRaises(ValueError):
            share_components({**row, 'stk_co_rate': 0.5}, True)

    def test_absent_total_and_components_is_unresolved_not_zero(self):
        row, measured = self.cases[1]
        blank = {k: v for k, v in row.items() if k not in ('stk_div', 'stk_co_rate')}
        self.assertEqual(assess_row(blank, identity_proven=True, **measured).reason,
                         'positive_total_without_components')

    def test_identity_violation_is_rejected_not_approved(self):
        row, measured = self.cases[1]
        bad = {**measured, 'adj_ex': 1.20}
        verdict = assess_row(row, identity_proven=True, **bad)
        self.assertEqual(verdict.status, 'rejected')
        self.assertEqual(verdict.reason, 'reference_identity_failed')

    def test_report_period_duplicates_merge_but_only_on_binding_fields(self):
        base = self.cases[0][0]
        rows = [{**base, 'end_date': p} for p in ('20251231', '20250630', '20250630')]
        merged, _ = merge_same_implementation(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['merged_report_periods'], ['20250630', '20251231'])
        conflicting = rows + [{**base, 'cash_div_tax': 0.9}]
        self.assertEqual(len(merge_same_implementation(conflicting)[0]), 2)

    def test_identical_rows_fold_into_one_event_identity(self):
        # Some issuers are repeated verbatim by the vendor, report period and
        # announcement date included. Leaving those split created two event ids
        # for one entitlement, which surfaced downstream as a same-day collision
        # instead of one recorded event.
        base = self.cases[0][0]
        merged, _ = merge_same_implementation([dict(base), dict(base), dict(base)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['merged_rows'], 3)
        self.assertEqual(merged[0]['merged_report_periods'], ['20251231'])
        self.assertEqual(merged[0]['merged_announcement_dates'], ['20251230'])

    def test_derived_tolerance_adds_half_a_tick_to_the_rounding_bound(self):
        # 601868.SH 2025-01-20: reference 2.2475 published on a 0.01 tick. The
        # half-tick term is what makes a 0.11% error benign rather than a failure.
        cheap = identity_tolerance(2.26, 0.0125)
        rich = identity_tolerance(66.3, 0.6)
        self.assertAlmostEqual(cheap, ADJ_ROUNDING_TOLERANCE + 0.5 * 0.01 / 2.2475)
        self.assertAlmostEqual(rich, ADJ_ROUNDING_TOLERANCE + 0.5 * 0.01 / 65.7)
        self.assertGreater(cheap, rich)
        self.assertEqual(identity_tolerance(0.0, 0.0), ADJ_ROUNDING_TOLERANCE)

    def test_half_tick_event_passes_while_a_real_cash_conflict_fails(self):
        # Same measured error, different price granularity: 601868 is explained
        # by the exchange tick, 600256 is not.
        cheap = {**self.cases[0][0], 'code': '601868.SH', 'cash_div': 0.0125,
                 'cash_div_tax': 0.0125, 'stk_co_rate': None, 'stk_div': 0.0}
        cheap_pair = dict(pre_close=2.26, adj_prev=1.0359, adj_ex=1.0405)
        self.assertAlmostEqual(reference_identity(**cheap_pair, cash_per_share=0.0125,
                                                  share_rate=0.0)[2], 0.001116, places=6)
        self.assertEqual(assess_row(cheap, identity_proven=False, **cheap_pair).status, 'approved')
        rich = {**self.cases[0][0], 'code': '600256.SH', 'cash_div': 0.622,
                'cash_div_tax': 0.622, 'stk_co_rate': None, 'stk_div': 0.0}
        rich_pair = dict(pre_close=6.08, adj_prev=54.7793, adj_ex=60.8881)
        verdict = assess_row(rich, identity_proven=False, **rich_pair)
        self.assertEqual(verdict.status, 'rejected')
        self.assertLess(verdict.detail['tolerance'], verdict.detail['relative_error'])

    def test_explicit_tolerance_still_overrides_the_derived_bound(self):
        row = {**self.cases[0][0], 'code': '601868.SH', 'cash_div': 0.0125,
               'cash_div_tax': 0.0125, 'stk_div': 0.0, 'stk_co_rate': None}
        verdict = assess_row(row, pre_close=2.26, adj_prev=1.0359, adj_ex=1.0405,
                             identity_proven=False, tolerance=0.0005)
        self.assertEqual(verdict.status, 'rejected')

    def test_identity_ignores_missing_or_degenerate_quotes(self):
        self.assertIsNone(reference_identity(0.0, 0.1, 0.0, 1.0, 1.0))
        self.assertIsNone(reference_identity(None, 0.1, 0.0, 1.0, 1.0))
        self.assertIsNone(reference_identity(10.0, 0.1, 0.0, None, 1.0))
        self.assertIsNone(reference_identity(0.05, 0.1, 0.0, 1.0, 1.0))
        # The pre-tax declared cash is what the reference price removes.
        observed, predicted, _ = reference_identity(10.0, 1.0, 0.5, 1.0, 1.1)
        self.assertAlmostEqual(predicted, 10.0 / ((10.0 - 1.0) / 1.5))
        self.assertAlmostEqual(observed, 1.1)

    def test_pre_tax_reference_and_reported_after_tax_are_not_swapped(self):
        row = dict(code='A', ex_date='2025-01-09', record_date='2025-01-08',
                   pay_date='2025-01-09', stk_div=0.0,
                   cash_div=0.08, cash_div_tax=0.10)
        verdict = assess_row(row, pre_close=10.0, adj_prev=1.0,
                             adj_ex=10 / 9.9, identity_proven=False)
        self.assertEqual(verdict.status, 'approved')
        self.assertEqual(verdict.detail['reference_cash'], 0.10)
        self.assertEqual(verdict.detail['entitlement_cash_pre_tax'], 0.10)
        self.assertEqual(verdict.detail['reported_cash_after_tax'], 0.08)
        self.assertIsNone(verdict.detail['account_cash_receipt'])
        self.assertEqual(assess_row({**row, 'cash_div_tax': None},
                                    pre_close=10.0, adj_prev=1.0,
                                    adj_ex=10 / 9.9, identity_proven=False).reason,
                         'missing_pre_tax_cash_component')

    def test_issuer_reference_parameters_do_not_change_holder_entitlement(self):
        row = dict(code='300100.SZ', ex_date='2025-05-15',
                   record_date='2025-05-14', pay_date='2025-05-15',
                   stk_div=0.4, cash_div=0.2, cash_div_tax=0.2)
        ordinary = assess_row(row, pre_close=76.92, adj_prev=1.0,
                              adj_ex=76.92 / 54.94, identity_proven=False)
        self.assertEqual(ordinary.reason, 'reference_identity_failed')
        issuer = assess_row(row, pre_close=76.92, adj_prev=1.0,
                            adj_ex=76.92 / 54.94, identity_proven=False,
                            reference_cash_override=0.1982171,
                            reference_share_override=0.3964343,
                            reference_evidence='issuer implementation announcement')
        self.assertEqual(issuer.status, 'approved')
        self.assertEqual(issuer.detail['share_rate'], 0.4)
        self.assertEqual(issuer.detail['entitlement_cash_pre_tax'], 0.2)
        self.assertEqual(issuer.detail['reference_share_rate'], 0.3964343)
        self.assertEqual(assess_row(row, pre_close=76.92, adj_prev=1.0,
                                    adj_ex=76.92 / 54.94, identity_proven=False,
                                    reference_cash_override=0.1982171).reason,
                         'unproven_reference_override')

    def test_pure_share_requires_explicit_cash_absence_evidence(self):
        row = dict(code='300803.SZ', ex_date='2025-03-17',
                   record_date='2025-03-14', pay_date=None,
                   stk_div=0.45, cash_div=None, cash_div_tax=None)
        pair = dict(pre_close=92.86, adj_prev=1.0, adj_ex=92.86 / 64.04)
        self.assertEqual(assess_row(row, identity_proven=False, **pair).reason,
                         'missing_pre_tax_cash_component')
        verdict = assess_row(row, identity_proven=False, cash_absence_proven=True,
                             reference_evidence='issuer says no cash', **pair)
        self.assertEqual(verdict.status, 'approved')
        self.assertEqual(verdict.detail['entitlement_cash_pre_tax'], 0.0)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.dates = ["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06"]
        self.bars = {(d, c): dict(state="trading", open=10., close=10., pre_close=10.,
                                  up_limit=11., down_limit=9., action_check_available=True)
                     for d in self.dates for c in ["A", "B"]}

    def action(self, multiplier, ex_date="2025-01-06", record_date="2025-01-03"):
        return CorporateAction('e1', 'A', record_date, ex_date, ex_date,
                               cash_per_share_net=0.0, share_multiplier=multiplier,
                               new_shares_date=ex_date, evidence="unit fixture")

    def test_fractional_entitlement_blocks_by_default(self):
        out = run_ledger(self.dates, self.bars, {"2024-12-31": {"A": 0.1}},
                         initial_cash=10000., cost_bps=0., actions=[self.action(1.001)])
        self.assertEqual(out.status, "data_insufficient")
        self.assertIn("fractional_action_requires_cash_in_lieu",
                      [e["reason"] for e in out.errors])

    def test_floor_policy_retires_the_fraction_and_records_it(self):
        out = run_ledger(self.dates, self.bars, {"2024-12-31": {"A": 0.1}},
                         initial_cash=10000., cost_bps=0., actions=[self.action(1.001)],
                         fractional_share_policy="floor")
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertNotIn("fractional_action_requires_cash_in_lieu",
                         [e["reason"] for e in out.errors])
        retired = [e for e in out.journal if e["kind"] == "fractional_retired"]
        self.assertEqual(len(retired), 1)
        self.assertAlmostEqual(retired[0]["retired_fraction"], 0.1)
        self.assertEqual(retired[0]["granted_extra"], 0)
        self.assertEqual(out.days[-1]["positions"]["A"]["shares"], 100)

    def test_floor_policy_never_creates_cash_in_lieu(self):
        args = (self.dates, self.bars, {"2024-12-31": {"A": 0.1}})
        floored = run_ledger(*args, initial_cash=10000., cost_bps=0.,
                             actions=[self.action(1.001)],
                             fractional_share_policy="floor")
        plain = run_ledger(*args, initial_cash=10000., cost_bps=0.)
        self.assertEqual(floored.days[-1]["receivable"], 0.0)
        self.assertEqual(floored.days[-1]["cash"], plain.days[-1]["cash"])
        self.assertEqual(floored.days[-1]["nav"], plain.days[-1]["nav"])

    def test_whole_share_entitlement_is_unaffected_by_policy(self):
        for policy in ("block", "floor"):
            out = run_ledger(self.dates, self.bars, {"2024-12-31": {"A": 0.1}},
                             initial_cash=10000., cost_bps=0.,
                             actions=[self.action(1.1)],
                             fractional_share_policy=policy)
            self.assertEqual(out.status, "complete_under_declared_assumptions")
            self.assertEqual(out.days[-1]["positions"]["A"]["shares"], 110)

    def test_unproven_delisting_fact_is_refused(self):
        for bad in ({'A': {'date': '2025-01-06'}},
                    {'A': {'date': '2025-01-06', 'evidence': ''}},
                    {'A': {'date': '2025-02-01', 'evidence': 'x'}}):
            with self.assertRaises(ValueError):
                run_ledger(self.dates, self.bars, {"2024-12-31": {"A": 1.}},
                           initial_cash=10000., cost_bps=0., delist_evidence=bad)

    def test_delisted_position_freezes_and_refuses_to_trade(self):
        bars = {k: v for k, v in self.bars.items() if k[0] != "2025-01-06"}
        evidence = {"A": {"date": "2025-01-06", "evidence": "exchange notice"}}
        out = run_ledger(self.dates, bars,
                         {"2024-12-31": {"A": 1.}, "2025-01-03": {}},
                         initial_cash=10000., cost_bps=0., delist_evidence=evidence)
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual([e for e in out.journal if e["kind"] == "delisted_freeze"][0]["date"],
                         "2025-01-06")
        self.assertEqual(out.days[-1]["positions"]["A"]["shares"], 1000)
        self.assertIn("unlisted_frozen", [o["reason"] for o in out.orders])
        self.assertAlmostEqual(out.days[-1]["nav"], out.days[1]["nav"])
        reconcile_journal(out, 10000.)

    def test_frozen_position_return_is_not_synthesised(self):
        bars = {k: v for k, v in self.bars.items() if k[0] != "2025-01-06"}
        frozen = run_ledger(self.dates, bars, {"2024-12-31": {"A": 1.}},
                            initial_cash=10000., cost_bps=0.,
                            delist_evidence={"A": {"date": "2025-01-06",
                                                   "evidence": "notice"}})
        self.assertEqual(frozen.days[-1]["return_"], 0.0)

    def test_invalid_policy_name_is_refused(self):
        with self.assertRaises(ValueError):
            run_ledger(self.dates, self.bars, {"2024-12-31": {"A": 1.}},
                       initial_cash=10000., fractional_share_policy="round")


if __name__ == "__main__":
    unittest.main()
