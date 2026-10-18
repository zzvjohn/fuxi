"""Columnar certification (P-20261015-006): same derivation, vectorised arrays.

The fixtures mirror tests.test_columnar.fixture but keep clean cells so each
flag can be driven to True, then flipped one gap at a time.
"""
from datetime import date, timedelta
from pathlib import Path
import unittest

import numpy as np

from fuxi_v1.columnar import array_reference
from fuxi_v1.certification import certify, DEFAULT_DECLARATION
from fuxi_v1.certification_columnar import certify_columnar
from fuxi_v1.contracts import digest, MARKET_CONTENT_EXCLUSIONS
from fuxi_v1.datasets import execution_profile
from fuxi_v1.pit_universe import ALL_A_FRAME_PREDICATE
from fuxi_v1.tests.test_kernel import scratch_directory


def _fixture_payload(directory, *, codes=None, drop_member_code=None, unsupported_term=False,
                     exempted_term=False, missing_evidence_cell=False, frozen_instrument_extra=None):
    root = Path(directory)
    days = []
    d = date(2024, 1, 2)
    while len(days) < 24:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    codes = codes or ['000001.XSHE', '000002.XSHE', '000003.XSHE', '000004.XSHE',
                      '600001.XSHG', '600002.XSHG', '600003.XSHG', '600004.XSHG',
                      '600005.XSHG', '600006.XSHG', '600007.XSHG', '600008.XSHG']
    shape = (len(days), len(codes))
    membership = np.ones(shape, dtype=bool)
    membership[7:12, 5] = False
    close = np.array([[10.0 + j + i * 0.01 for j in range(len(codes))] for i in range(len(days))])
    turnover = np.full(shape, 0.5)
    barriers = np.full(shape, -1, dtype=np.int32)
    fields = {'close': {'unit': 'price', 'adjustment': 'raw', 'frequency': 'daily'},
              'turnover_rate': {'unit': 'ratio', 'adjustment': 'not_applicable', 'frequency': 'daily'}}

    def ref(name, array):
        path = root / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        return array_reference(path)

    members = [{'ts_code': c.replace('.XSHE', '.SZ').replace('.XSHG', '.SH'),
                'list_date': '20000101', 'delist_date': None, 'list_status': 'L'}
               for c in codes if c != drop_member_code]
    cohort = {'frame_predicate': ALL_A_FRAME_PREDICATE, 'members': members,
              'posthoc_exclusion': False, 'rule': 'all common A-share lifecycle members',
              'rule_kind': 'all_a_share_census', 'salt': 'columnar-cert-fixture',
              'sampling_frame': 'full_lifecycle_table', 'sampling_key_fields': ['ts_code']}
    snapshot = {'schema_version': 'fuxi.columnar_snapshot.1', 'source_id': 'columnar_cert_fixture',
                'exposure': 'research_exposed', 'sessions': days, 'instruments': list(codes),
                'evaluation_start': days[5],
                'decision_times': {d: d + 'T18:30:00+08:00' for d in days}, 'fields': fields,
                'availability_times': {f: [d + 'T18:00:00+08:00' for d in days] for f in fields},
                'cohort_declaration': cohort,
                'arrays': {'fields': {'close': ref('cert_close', close), 'turnover_rate': ref('cert_turn', turnover)},
                           'membership': ref('cert_members', membership), 'barriers': ref('cert_barriers', barriers)},
                'quality': {'pit_certified': False}, 'provenance': {'fixture': True}}
    numeric = {'close': close, 'open': close * 0.999, 'high': close * 1.01, 'low': close * 0.99,
               'volume': np.full(shape, 1000000.0), 'up_limit': close * 1.1, 'down_limit': close * 0.9,
               'sizing_reference_close': np.full(shape, np.nan)}
    event_coverage = np.ones(shape, dtype=np.uint8)
    if missing_evidence_cell:
        event_coverage[5, 3] = 0  # 'missing_evidence' in this fixture's category table
    execution = dict(execution_profile(),
                     sell_tax_schedule=[['0001-01-01', '0.001'], ['2023-08-28', '0.0005']])
    market = {'schema_version': 'fuxi.columnar_market.1', 'sessions': days, 'instruments': list(codes),
              'execution': execution, 'actions': [], 'action_terms': [], 'provenance': {'fixture': True},
              'event_coverage_certified': False, 'tradability_certified': False, 'certification_hash': None,
              'evidence_namespace': 'columnar_cert_market', 'sparse_bars': {},
              'arrays': {'numeric': {f: ref('cert_bar_' + f, a) for f, a in numeric.items()},
                         'status': {'paused': ref('cert_paused', np.zeros(shape, dtype=np.int8)),
                                    'corporate_action_known': ref('cert_known', np.ones(shape, dtype=np.int8))},
                         'categorical': {'event_coverage': ref('cert_ec', event_coverage),
                                         'mark_source': ref('cert_ms', np.zeros(shape, dtype=np.uint8))}},
              'category_values': {'event_coverage': {'0': 'missing_evidence', '1': 'unchanged_reference_and_factor'},
                                  'mark_source': {'0': 'observed_close'}}}
    if unsupported_term:
        market['action_terms'] = [{'instrument': codes[0], 'ex_date': days[6], 'record_date': days[5],
                                   'pay_date': days[8], 'ann_date': days[4], 'gross_cash_per_share': 0.1,
                                   'share_distribution': 0.5, 'status': 'conflicting_dividend_rows',
                                   'evidence_id': digest(['conflict', codes[0], days[6]])}]
    if exempted_term:
        market['action_terms'] = [{'instrument': codes[0], 'ex_date': days[6], 'record_date': days[5],
                                   'pay_date': days[8], 'ann_date': days[4], 'gross_cash_per_share': 0.1,
                                   'share_distribution': 0.5, 'status': 'conflicting_dividend_rows',
                                   'exemption': {'reason': 'r', 'declared_at': '2026-10-15',
                                                 'registry_hash': 'h' * 64},
                                   'evidence_id': digest(['conflict', codes[0], days[6]])}]
    market['market_hash'] = digest({k: v for k, v in market.items() if k != 'market_hash'})
    return {'snapshot': snapshot, 'market': market}


def _life():
    return {c: {'list_date': '20000101', 'delist_date': None, 'list_status': 'L'}
            for c in ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ',
                      '600001.SH', '600002.SH', '600003.SH', '600004.SH',
                      '600005.SH', '600006.SH', '600007.SH', '600008.SH']}


def _declaration(fields=None):
    merged = dict(DEFAULT_DECLARATION)
    if fields is not None:
        merged['fields'] = fields
    return merged


class ColumnarCertificationTests(unittest.TestCase):
    def setUp(self):
        self.scratch = scratch_directory()
        self.path = self.scratch.__enter__()

    def tearDown(self):
        self.scratch.__exit__(None, None, None)

    def test_certify_dispatches_columnar_payload(self):
        payload = _fixture_payload(self.path)
        cert = certify(payload, declaration=_declaration(fields=['close', 'turnover_rate']), lifecycle=_life())
        self.assertEqual(set(cert['flags']),
                         {'pit_certified', 'universe_certified', 'market_representativeness_certified',
                          'account_events_certified', 'event_coverage_certified',
                          'tradability_certified', 'execution_rules_certified'})
        self.assertTrue(cert['flags']['pit_certified']['value'])
        self.assertTrue(cert['flags']['universe_certified']['value'])
        self.assertTrue(cert['flags']['market_representativeness_certified']['value'])
        self.assertTrue(cert['flags']['account_events_certified']['value'])
        self.assertTrue(cert['flags']['event_coverage_certified']['value'])
        self.assertTrue(cert['flags']['tradability_certified']['value'])
        self.assertTrue(cert['flags']['execution_rules_certified']['value'])
        self.assertEqual(cert['blocking_gaps'], [])
        self.assertEqual(cert['schema_version'], 'fuxi.data_certification.1')
        self.assertEqual(cert['snapshot_hash'], digest(payload['snapshot']))
        self.assertEqual(cert['certification_hash'], digest({k: v for k, v in cert.items()
                                                             if k not in ('certification_hash', 'certified_at')}))

    def test_cohort_member_mismatch_refuses_universe(self):
        payload = _fixture_payload(self.path, drop_member_code='600008.XSHG')
        cert = certify_columnar(payload, declaration=_declaration(fields=['close', 'turnover_rate']),
                                lifecycle=_life())
        self.assertFalse(cert['flags']['universe_certified']['value'])
        self.assertIn('cohort_not_reproducible_from_declared_rule',
                      cert['flags']['universe_certified']['blocking_gaps'])
        self.assertIn('cohort_not_reproducible_from_declared_rule', cert['blocking_gaps'])

    def test_unsupported_action_terms_refuse_account_events(self):
        payload = _fixture_payload(self.path, unsupported_term=True)
        cert = certify_columnar(payload, declaration=_declaration(fields=['close', 'turnover_rate']),
                                lifecycle=_life())
        flag = cert['flags']['account_events_certified']
        self.assertFalse(flag['value'])
        self.assertTrue(any(g.startswith('unsupported_action_terms:') for g in flag['blocking_gaps']))

    def test_exempted_action_term_is_declared_residual_not_blocking(self):
        # P-009 (a3): the columnar refusal must mirror the row-form rule -- a
        # hash-bound exemption turns the unsupported term into a residual gap.
        payload = _fixture_payload(self.path, exempted_term=True)
        cert = certify_columnar(payload, declaration=_declaration(fields=['close', 'turnover_rate']),
                                lifecycle=_life())
        flag = cert['flags']['account_events_certified']
        self.assertTrue(flag['value'])
        self.assertEqual(flag['blocking_gaps'], [])
        self.assertTrue(any('declared_action_term_exemptions:1' in g for g in flag['residual_gaps']))
        self.assertEqual(flag['basis']['exempted_terms'], 1)
        self.assertNotIn('unsupported_action_terms', '|'.join(cert['blocking_gaps']))

    def test_missing_event_evidence_refuses_event_coverage(self):
        payload = _fixture_payload(self.path, missing_evidence_cell=True)
        cert = certify_columnar(payload, declaration=_declaration(fields=['close', 'turnover_rate']),
                                lifecycle=_life())
        flag = cert['flags']['event_coverage_certified']
        self.assertFalse(flag['value'])
        self.assertTrue(any(g.startswith('unspecified_event_coverage_cells:') for g in flag['blocking_gaps']))

    def test_lifecycle_missing_refuses_universe_flag(self):
        payload = _fixture_payload(self.path)
        cert = certify_columnar(payload, declaration=_declaration(fields=['close', 'turnover_rate']),
                                lifecycle=None)
        self.assertFalse(cert['flags']['universe_certified']['value'])
        self.assertIn('lifecycle_table_not_supplied', cert['flags']['universe_certified']['blocking_gaps'])

    def test_row_form_path_unchanged(self):
        from fuxi_v1.tests.test_certification import SESSIONS
        bars = [{'session': '2024-01-02', 'instrument': '000001.XSHE', 'open': 10.0, 'close': 10.1,
                 'high': 10.2, 'low': 9.9, 'volume': 1000.0, 'up_limit': 11.0, 'down_limit': 9.0,
                 'corporate_action_known': True, 'event_coverage': True, 'paused': False}]
        payload = {'snapshot': {'schema_version': 'fuxi.snapshot.1', 'sessions': list(SESSIONS),
                                'decision_times': {d: d + 'T18:30:00+08:00' for d in SESSIONS},
                                'fields': {'close': {'frequency': 'daily', 'adjustment': 'raw'}},
                                'evaluation_start': list(SESSIONS)[0],
                                'universe': {d: ['000001.XSHE'] for d in SESSIONS},
                                'instruments': ['000001.XSHE'],
                                'columns': {'close': [[10.1] for _ in SESSIONS]}},
                   'market': {'bars': bars, 'action_terms': [], 'actions': [],
                              'execution': execution_profile(),
                              'market_hash': 'x' * 64}}
        cert = certify(payload, declaration=_declaration(fields=['close']), lifecycle=_life())
        self.assertEqual(cert['snapshot_hash'], digest(payload['snapshot']))

    def test_certification_writeback_and_assurance_bind_produce_formal_evidence(self):
        # P-20261015-012: certification annotation write-back and the assurance
        # binding must not break the market-content digest, and a bound columnar
        # market must yield formal (certified) account evidence via run_account.
        from fuxi_v1.columnar import ColumnarMarket
        from fuxi_v1.simulation import run_account
        payload = _fixture_payload(self.path)
        cert = certify(payload, declaration=_declaration(fields=['close', 'turnover_rate']), lifecycle=_life())
        self.assertEqual(cert['blocking_gaps'], [])
        # annotation write-back
        mkt = dict(payload['market'])
        mkt['event_coverage_certified'] = cert['flags']['event_coverage_certified']['value']
        mkt['tradability_certified'] = cert['flags']['tradability_certified']['value']
        mkt['certification_hash'] = cert['certification_hash']
        mkt['certification_snapshot_hash'] = cert['snapshot_hash']
        mkt['market_hash'] = digest({k: v for k, v in mkt.items() if k != 'market_hash'})
        self.assertEqual(digest({k: v for k, v in mkt.items() if k not in MARKET_CONTENT_EXCLUSIONS}),
                         cert['market_digest'])
        ColumnarMarket(mkt).close()
        # assurance bind
        mkt['investment_assurance'] = {
            'schema_version': 'fuxi.market_assurance.1',
            'snapshot_hash': cert['snapshot_hash'],
            'market_digest': cert['market_digest'],
            'receipt_id': 'p012-fixture-receipt',
            'assurance_hash': digest({'review': 'p012-fixture'}),
        }
        mkt['market_hash'] = digest({k: v for k, v in mkt.items() if k != 'market_hash'})
        bound = ColumnarMarket(mkt)
        ir = {'execution_hash': digest(bound['execution']), 'snapshot_hash': cert['snapshot_hash'],
              'decisions': []}
        ir['ir_hash'] = digest({k: v for k, v in ir.items() if k != 'ir_hash'})
        account = run_account(ir, bound)
        self.assertEqual(account['eval_status'], 'valid')
        self.assertTrue(account['formal_evidence'])
        # a mismatched snapshot binding must refuse formal evidence
        mkt['investment_assurance'] = dict(mkt['investment_assurance'],
                                           snapshot_hash=digest({'other': 'snapshot'}))
        mkt['market_hash'] = digest({k: v for k, v in mkt.items() if k != 'market_hash'})
        tampered = ColumnarMarket(mkt)
        self.assertFalse(run_account(ir, tampered)['formal_evidence'])
        tampered.close()
        bound.close()


if __name__ == '__main__':
    unittest.main()
