# -*- coding: utf-8 -*-
"""P-20261015-009: vendor action-term resolution rules + exemption registry."""
import json
import tempfile
import unittest
from pathlib import Path

from fuxi_v1.historical_adapter import dividend_terms, apply_action_term_exemptions
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.certification import certify_account_events
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import ContractError


DAYS = ['2024-01-01', '2026-12-31']


class VendorRowResolutionTests(unittest.TestCase):
    def test_rule_a_imp_ann_publication_variants_collapse_to_latest(self):
        base = {'ts_code': '920001.BJ', 'div_proc': '实施', 'ex_date': '20240613',
                'record_date': '20240612', 'pay_date': '20240613', 'imp_ann_date': '20240605',
                'div_listdate': None, 'stk_div': '0', 'cash_div_tax': '0.3', 'end_date': '20231231'}
        term = dividend_terms([base, dict(base, imp_ann_date='20240606')], DAYS)[0]
        self.assertEqual(term['status'], 'cash_terms_supported')
        self.assertEqual(term['ann_date'], '2024-06-06')
        self.assertEqual(term['vendor_row_resolution_audit']['imp_ann_date_variants'],
                         ['2024-06-05', '2024-06-06'])

    def test_rule_b_share_rows_cash_none_equals_zero_and_resolver_upgrades(self):
        a = {'ts_code': '920002.BJ', 'div_proc': '实施', 'ex_date': '20240614',
             'record_date': '20240613', 'imp_ann_date': '20240606', 'div_listdate': '20240614',
             'stk_div': '0.4', 'cash_div_tax': None, 'pay_date': None}
        b = dict(a, cash_div_tax='0.0')
        base = dividend_terms([a, b], DAYS)
        self.assertEqual(len(base), 1)
        self.assertEqual(base[0]['status'], 'share_distribution_requires_settlement_adapter')
        resolved = resolve_action_terms(base, [a, b], [])[0]
        self.assertEqual(resolved['status'], 'ordinary_distribution_supported')
        self.assertEqual(resolved['gross_cash_per_share'], 0.0)
        self.assertEqual(resolved['share_credit_date'], '2024-06-14')

    def test_rule_a_dual_publication_mixed_share_cash_event_upgrades(self):
        # P-20261015-011: BJSE vendors publish one mixed share+cash event under two
        # implementation-publication dates; the resolver must collapse the rights
        # (Rule A) and settle both legs on the vendor credit (listing) date.
        a = {'ts_code': '920046.BJ', 'div_proc': '实施', 'ex_date': '20240531',
             'record_date': '20240530', 'pay_date': '20240531', 'imp_ann_date': '20240523',
             'div_listdate': '20240531', 'stk_div': '0.4', 'cash_div_tax': '0.1', 'end_date': '20231231'}
        b = dict(a, imp_ann_date='20240524')
        base = dividend_terms([a, b], DAYS)
        self.assertEqual(len(base), 1)
        self.assertEqual(base[0]['status'], 'share_distribution_requires_settlement_adapter')
        resolved = resolve_action_terms(base, [a, b], [])[0]
        self.assertEqual(resolved['status'], 'ordinary_distribution_supported')
        self.assertEqual(resolved['share_credit_date'], '2024-05-31')
        self.assertEqual(resolved['gross_cash_per_share'], 0.1)

    def test_rule_a_dual_publication_pure_share_event_upgrades(self):
        a = {'ts_code': '920363.BJ', 'div_proc': '实施', 'ex_date': '20250527',
             'record_date': '20250526', 'pay_date': None, 'imp_ann_date': '20250519',
             'div_listdate': '20250527', 'stk_div': '0.2', 'cash_div_tax': None}
        b = dict(a, cash_div_tax='0.0', imp_ann_date='20250520')
        base = dividend_terms([a, b], DAYS)
        resolved = resolve_action_terms(base, [a, b], [])[0]
        self.assertEqual(resolved['status'], 'ordinary_distribution_supported')
        self.assertEqual(resolved['share_credit_date'], '2025-05-27')
        self.assertEqual(resolved['gross_cash_per_share'], 0.0)

    def test_missing_credit_date_stays_unsupported(self):
        # 002309.SZ 2024-12-24: the vendor row really has no div_listdate; the
        # resolver must not invent a settlement date and must leave the term
        # share_distribution_requires_settlement_adapter.
        a = {'ts_code': '002309.SZ', 'div_proc': '实施', 'ex_date': '20241224',
             'record_date': '20241223', 'pay_date': None, 'imp_ann_date': '20241218',
             'div_listdate': None, 'stk_div': '2.45', 'cash_div_tax': None}
        base = dividend_terms([a], DAYS)
        self.assertEqual(base[0]['status'], 'share_distribution_requires_settlement_adapter')
        resolved = resolve_action_terms(base, [a], [])[0]
        self.assertEqual(resolved['status'], 'share_distribution_requires_settlement_adapter')
        self.assertIsNone(resolved.get('share_credit_date'))

    def test_rule_c2_vendor_sum_row_dropped_and_relaxed_aggregate_corroborated(self):
        base = {'ts_code': '920003.BJ', 'div_proc': '实施', 'ex_date': '20241107',
                'record_date': '20241106', 'pay_date': '20241107', 'imp_ann_date': '20241031',
                'div_listdate': None, 'stk_div': '0'}
        rows = [dict(base, end_date='20241011', cash_div_tax='1.0'),
                dict(base, end_date='20240630', cash_div_tax='0.4'),
                dict(base, end_date='20240630', cash_div_tax='1.4')]
        term = dividend_terms(rows, DAYS)[0]
        self.assertEqual(term['status'], 'cash_terms_supported')
        self.assertEqual(term['gross_cash_per_share'], 1.4)
        self.assertTrue(term['aggregation_corroborated_by_vendor_sum_row'])
        self.assertEqual(term['vendor_row_resolution_audit']['vendor_sum_row_removed'], 1.4)

    def test_rule_c3_latest_plan_publication_supersedes_within_one_period(self):
        base = {'ts_code': '920004.BJ', 'div_proc': '实施', 'ex_date': '20240724',
                'record_date': '20240723', 'pay_date': '20240724', 'imp_ann_date': '20240718',
                'div_listdate': None, 'stk_div': '0', 'end_date': '20231231'}
        rows = [dict(base, cash_div_tax='0.3158', ann_date='20240622'),
                dict(base, cash_div_tax='0.265', ann_date='20240322')]
        term = dividend_terms(rows, DAYS)[0]
        self.assertEqual(term['status'], 'cash_terms_supported')
        self.assertEqual(term['gross_cash_per_share'], 0.3158)
        self.assertEqual(term['vendor_row_resolution_audit']['cash_plan_variants'], [0.265, 0.3158])

    def test_unresolvable_amounts_stay_conflicting(self):
        base = {'ts_code': '920005.BJ', 'div_proc': '实施', 'ex_date': '20250521',
                'record_date': '20250520', 'pay_date': '20250521', 'imp_ann_date': '20250515',
                'div_listdate': None, 'stk_div': '0'}
        rows = [dict(base, end_date='20250318', cash_div_tax='0.35', ann_date='20250430'),
                dict(base, end_date='20241231', cash_div_tax='0.98169', ann_date='20250430')]
        term = dividend_terms(rows, DAYS)[0]
        self.assertEqual(term['status'], 'conflicting_dividend_rows')

    def test_strict_quarter_aggregate_unchanged_without_corroboration(self):
        base = {'ts_code': '920006.BJ', 'div_proc': '实施', 'ex_date': '20260601',
                'record_date': '20260529', 'pay_date': '20260601', 'imp_ann_date': '20260526',
                'div_listdate': None, 'stk_div': '0'}
        a = dict(base, end_date='20260331', cash_div_tax='0.18')
        bad = dict(base, end_date='20251011', cash_div_tax='0.132')
        self.assertEqual(dividend_terms([a, bad], DAYS)[0]['status'], 'conflicting_dividend_rows')
        ok = dividend_terms([a, dict(bad, end_date='20251231')], DAYS)[0]
        self.assertEqual(ok['status'], 'cash_terms_supported')
        self.assertEqual(ok['gross_cash_per_share'], 0.312)


class ActionTermExemptionTests(unittest.TestCase):
    def _registry(self, tmp, entries):
        path = Path(tmp) / 'exemptions.json'
        registry = {'schema_version': 'fuxi.action_term_exemptions.1',
                    'declared_at': '2026-10-15T20:30:00+08:00', 'entries': entries}
        path.write_text(json.dumps(registry, ensure_ascii=False), encoding='utf-8')
        return {'path': str(path), 'sha256': file_hash(path)}

    def test_binds_matching_terms_and_requires_full_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            terms = [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03', 'status': 'conflicting_dividend_rows'}]
            spec = self._registry(tmp, [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03',
                                         'status': 'conflicting_dividend_rows', 'reason': 'r',
                                         'declared_at': '2026-10-15'}])
            out = apply_action_term_exemptions(terms, spec)
            self.assertEqual(out[0]['exemption']['reason'], 'r')
            # registry entry that binds no term must raise
            spec2 = self._registry(tmp, [{'instrument': 'B.XSHE', 'ex_date': '2024-01-03',
                                          'status': 'conflicting_dividend_rows', 'reason': 'r',
                                          'declared_at': '2026-10-15'}])
            with self.assertRaises(ContractError):
                apply_action_term_exemptions(terms, spec2)

    def test_status_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            terms = [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03', 'status': 'cash_terms_supported'}]
            spec = self._registry(tmp, [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03',
                                         'status': 'conflicting_dividend_rows', 'reason': 'r',
                                         'declared_at': '2026-10-15'}])
            with self.assertRaises(ContractError):
                apply_action_term_exemptions(terms, spec)

    def test_tampered_registry_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            terms = [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03', 'status': 'conflicting_dividend_rows'}]
            spec = self._registry(tmp, [{'instrument': 'A.XSHE', 'ex_date': '2024-01-03',
                                         'status': 'conflicting_dividend_rows', 'reason': 'r',
                                         'declared_at': '2026-10-15'}])
            Path(spec['path']).write_text('{"tampered": true}', encoding='utf-8')
            with self.assertRaises(ContractError):
                apply_action_term_exemptions(terms, spec)


class CertificationExemptionTests(unittest.TestCase):
    def _snapshot(self):
        return {'sessions': ['2024-01-02', '2024-01-03'],
                'universe': {'2024-01-02': ['A.XSHE'], '2024-01-03': ['A.XSHE']}}

    def _market(self, term):
        bars = [{'session': d, 'instrument': 'A.XSHE', 'corporate_action_known': True,
                 'event_coverage': 'published_corporate_action'}
                for d in self._snapshot()['sessions']]
        return {'bars': bars, 'action_terms': [term]}

    def test_exempted_term_is_declared_residual_not_blocking(self):
        term = {'instrument': 'A.XSHE', 'ex_date': '2024-01-03', 'status': 'conflicting_dividend_rows',
                'exemption': {'reason': 'r', 'declared_at': '2026-10-15', 'registry_hash': 'h'}}
        flag = certify_account_events(self._snapshot(), self._market(term))
        self.assertTrue(flag['value'])
        self.assertEqual(flag['blocking_gaps'], [])
        self.assertTrue(any('declared_action_term_exemptions:1' in g for g in flag['residual_gaps']))
        self.assertEqual(flag['basis']['exempted_terms'], 1)

    def test_unexempted_term_still_blocks(self):
        term = {'instrument': 'A.XSHE', 'ex_date': '2024-01-03', 'status': 'conflicting_dividend_rows'}
        flag = certify_account_events(self._snapshot(), self._market(term))
        self.assertFalse(flag['value'])
        self.assertTrue(any('unsupported_action_terms' in g for g in flag['blocking_gaps']))


if __name__ == '__main__':
    unittest.main()
