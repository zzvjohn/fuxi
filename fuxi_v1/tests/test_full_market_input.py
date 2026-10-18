"""Lifecycle retention and native raw-price/action contracts of the full-market loader."""
from collections import defaultdict
from datetime import date, timedelta
import unittest

import numpy as np

from fuxi_v1.full_market_input import derive_market_arrays, CLASSIFICATIONS, MARK_SOURCES
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.history import jq_code, adjustment_class


def fixture():
    sessions = [(date(2024, 1, 2) + timedelta(days=i)).isoformat() for i in range(16)]
    codes = ['%06d.SZ' % i for i in range(1, 12)] + ['600001.SH', '688001.SH', '920001.BJ']
    members = [dict(ts_code=code, list_date='20000101', delist_date=None, list_status='L') for code in codes]
    members[-1]['list_date'] = sessions[3].replace('-', '')
    members[3].update(delist_date=sessions[12].replace('-', ''), list_status='D')
    shape = (len(sessions), len(codes))
    close = np.array([[10 + j + i * .1 for j in range(len(codes))] for i in range(len(sessions))])
    pre_close = np.vstack([close[0], close[:-1]])
    numbers = {f: close.copy() for f in ('open', 'high', 'low', 'close')}
    numbers.update(pre_close=pre_close, vol=np.full(shape, 1000.), amount=np.full(shape, 1500.),
                   adj_factor=np.ones(shape), up_limit=close * 1.1, down_limit=close * .9,
                   turnover_rate=np.full(shape, 1.), total_share=np.full(shape, 100.),
                   float_share=np.full(shape, 80.), free_share=np.full(shape, 50.))
    names = {code: [dict(name='普通证券', start_date='20000101', end_date=None, ann_date='20000101')] for code in codes}
    masks = {api: np.ones(shape, bool) for api in ('daily', 'adj_factor', 'stk_limit', 'daily_basic')}
    return sessions, codes, members, numbers, masks, names


class FullMarketInputTests(unittest.TestCase):
    def test_observed_resume_record_keeps_native_false_even_when_volume_is_zero(self):
        sessions, codes, members, numbers, masks, names = fixture()
        i, j = 7, 2
        numbers['vol'][i, j] = 0.
        records = {(codes[j], sessions[i]): [dict(ts_code=codes[j], trade_date=sessions[i].replace('-', ''),
                                                 suspend_type='R', suspend_timing=None)]}
        result = derive_market_arrays(sessions, codes, members, numbers, masks, names, [], records,
                                      listing_age_calendar_days=0)
        self.assertEqual(result['status']['paused'][i, j], 0)
        self.assertTrue(result['membership'][i, j])

    def test_full_board_lifecycle_keeps_members_and_blocks_bj_prelisting_and_retirement(self):
        sessions, codes, members, numbers, masks, names = fixture()
        result = derive_market_arrays(sessions, codes, members, numbers, masks, names, [], {}, listing_age_calendar_days=0)
        self.assertEqual(result['columns']['close'].shape, (16, 14))
        self.assertTrue(np.isnan(result['columns']['close'][:3, -1]).all())
        self.assertFalse(result['membership'][:3, -1].any())
        self.assertTrue(result['membership'][3:, -1].all())
        self.assertFalse(result['membership'][12:, 3].any())
        self.assertFalse(result['status']['corporate_action_known'][12:, 3].any())
        self.assertTrue(result['membership'].sum(axis=1).min() >= 10)
        self.assertEqual(result['columns']['volume'][0, 0], 100000.)
        self.assertEqual(result['columns']['amount'][0, 0], 1500000.)
        self.assertEqual(result['columns']['turnover_rate'][0, 0], .01)

    def test_halted_cash_action_is_discounted_and_raw_history_stays_blocked(self):
        sessions, codes, members, numbers, masks, names = fixture()
        j, i = 1, 5
        previous = float(numbers['close'][i - 1, j])
        for field in ('open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount'):
            numbers[field][i, j] = np.nan
        numbers['pre_close'][i + 1, j] = previous - .5
        term = dict(instrument=jq_code(codes[j]), ex_date=sessions[i], status='cash_terms_supported',
                    gross_cash_per_share=.5, share_distribution=0, evidence_id='cash_fixture')
        halts = {(codes[j], sessions[i]): [dict(ts_code=codes[j], trade_date=sessions[i].replace('-', ''),
                                               suspend_type='S', suspend_timing=None)]}
        result = derive_market_arrays(sessions, codes, members, numbers, masks, names, [term], halts,
                                      listing_age_calendar_days=0)
        self.assertTrue(np.isnan(result['columns']['close'][i, j]))
        self.assertAlmostEqual(result['numeric']['close'][i, j], previous - .5)
        self.assertEqual(result['categorical']['mark_source'][i, j], MARK_SOURCES.index('declared_ex_cash_reference_on_vendor_halt'))
        self.assertEqual(result['status']['paused'][i, j], 1)
        self.assertTrue(result['status']['corporate_action_known'][i, j])
        self.assertFalse(result['membership'][i, j])
        self.assertEqual(result['barriers'][i, j], i)
        payload = dict(schema_version='fuxi.historical_snapshot.1', source_id='fixture', exposure='research_exposed',
                       sessions=sessions, evaluation_start=sessions[0], instruments=[jq_code(c) for c in codes],
                       universe={day: [jq_code(c) for k, c in enumerate(codes) if result['membership'][n, k]]
                                 for n, day in enumerate(sessions)},
                       decision_times={d: d + 'T18:30:00+08:00' for d in sessions},
                       fields={f: {} for f in result['columns']},
                       columns={f: [[None if np.isnan(v) else float(v) for v in row] for row in values]
                                for f, values in result['columns'].items()}, barriers=result['barriers'].tolist())
        snapshot = HistoricalSnapshot(payload)
        code = jq_code(codes[j])
        self.assertEqual(snapshot.value(i - 1, code, 'close', snapshot.clocks[sessions[i - 1]]), previous)
        self.assertIsNone(snapshot.value(i - 1, code, 'close', snapshot.clocks[sessions[i]]))
        self.assertEqual(result['categorical']['event_coverage'][i + 1, j], CLASSIFICATIONS.index('unchanged_reference_and_factor'))

    def test_partial_halt_observed_mark_does_not_grant_orders_or_fill_unknown_rights(self):
        sessions, codes, members, numbers, masks, names = fixture()
        day, code = sessions[7], codes[2]
        halts = {(code, day): [dict(ts_code=code, trade_date=day.replace('-', ''), suspend_type='S',
                                    suspend_timing='09:30-10:00')]}
        unsupported = dict(instrument=jq_code(codes[4]), ex_date=sessions[9],
                           status='share_distribution_requires_settlement_adapter',
                           gross_cash_per_share=0., share_distribution=.1, evidence_id='unresolved_fixture')
        result = derive_market_arrays(sessions, codes, members, numbers, masks, names, [unsupported], halts,
                                      listing_age_calendar_days=0,
                                      source_hashes={'suspend_d.csv': 'suspend_source', 'daily_prices.csv': 'price_source'})
        self.assertEqual(result['categorical']['mark_source'][7, 2], MARK_SOURCES.index('observed_close_on_vendor_partial_halt'))
        self.assertFalse(result['membership'][7, 2])
        self.assertEqual(result['status']['paused'][7, 2], 1)
        proof = result['sparse_bars'][day + '|' + jq_code(code)]['partial_halt_mark']
        self.assertEqual(proof['observed_close'], float(numbers['close'][7, 2]))
        self.assertFalse(result['status']['corporate_action_known'][9, 4])
        self.assertFalse(result['membership'][9, 4])
        self.assertEqual(result['barriers'][9, 4], 9)

    def test_regular_classifications_match_native_scalar_contract(self):
        sessions, codes, members, numbers, masks, names = fixture()
        numbers['adj_factor'][4, 5] += .000001
        numbers['adj_factor'][5:, 5] += .000001
        numbers['pre_close'][8, 6] -= .1
        result = derive_market_arrays(sessions, codes, members, numbers, masks, names, [], {}, listing_age_calendar_days=0)
        compared = 0
        for j in range(10):
            previous_factor = numbers['adj_factor'][0, j]
            previous_price = numbers['close'][0, j]
            for i in range(1, 12):
                expected = adjustment_class(float(previous_factor), float(numbers['adj_factor'][i, j]),
                                            float(previous_price), float(numbers['pre_close'][i, j]))
                self.assertEqual(CLASSIFICATIONS[result['categorical']['event_coverage'][i, j]], expected)
                previous_factor = numbers['adj_factor'][i, j]
                previous_price = numbers['close'][i, j]
                compared += 1
        self.assertEqual(compared, 110)


if __name__ == '__main__':
    unittest.main()
