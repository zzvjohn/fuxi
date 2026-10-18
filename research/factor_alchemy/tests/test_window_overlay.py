"""Contract tests for isolated overlay evidence and finite-prefix certification."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from fuxi_window_replay_20260923 import Book, canonical, merge_overlay, compare_reference
from portfolio_execution_ledger import CorporateAction, run_ledger


class WindowOverlayTests(unittest.TestCase):
    def test_historical_identity_is_date_bound(self):
        self.assertEqual(canonical('2025-10-08', '839680.BJ'), '920680.BJ')
        self.assertEqual(canonical('2025-01-01', '920680.BJ'), '920680.BJ')
        with self.assertRaisesRegex(ValueError, 'after_official'):
            canonical('2025-10-09', '839680.BJ')

    def test_conflicting_overlay_is_not_tradable(self):
        key = ('2025-01-02', '000001.SZ')
        conflicts = []
        limits = merge_overlay({key: {'up_limit': 11}}, {key: {'up_limit': 12}}, 'limits', conflicts)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['original'], {'up_limit': 11})
        book = Book({key: {'open': 10, 'close': 10}}, limits,
                    {key: {'action_check_available': True}}, set(), set(), [key])
        self.assertEqual(book.get(key)['state'], 'unknown')
        self.assertFalse(book.get(key)['action_check_available'])

    def test_identity_halt_and_quote_conflict(self):
        key = ('2025-04-01', '920680.BJ')
        book = Book({}, {}, {}, {key}, set())
        self.assertEqual(book.get((key[0], '839680.BJ'))['state'], 'halted')
        book.quotes[key] = {'open': 10, 'close': 10}
        self.assertEqual(book.get(key)['state'], 'unknown')

    def test_delisting_cannot_be_sold_at_stale_price(self):
        key = ('2025-02-10', '600898.SH')
        book = Book({key: {'open': 10, 'close': 10}}, {}, {}, set(), set())
        self.assertEqual(book.get(key)['state'], 'unknown')

    def test_distribution_lock_and_independent_reference(self):
        dates = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08']
        code = '000001.SZ'
        quotes = {(d, code): {'open': 10 if d < '2025-01-06' else 5,
                              'close': 10 if d < '2025-01-06' else 5,
                              'pre_close': 10 if d < '2025-01-06' else 5} for d in dates}
        limits = {k: {'up_limit': 20, 'down_limit': 1} for k in quotes}
        adjustments = {k: {'action_check_available': True,
                           'action_required': k[0] == '2025-01-06'} for k in quotes}
        targets = {'2025-01-02': {code: 1}, '2025-01-03': {}}
        action = CorporateAction('split', code, '2025-01-03', '2025-01-06', '2025-01-06',
                                 share_multiplier=2, new_shares_date='2025-01-08', evidence='fixture')
        result = run_ledger(dates, Book(quotes, limits, adjustments, set(), set()), targets,
                            initial_cash=10000000, cost_bps=15, actions=[action])
        self.assertEqual(result.status, 'complete_under_declared_assumptions')
        held = result.days[1]['positions'][code]['shares']
        self.assertEqual(result.days[2]['positions'][code]['shares'], held)
        self.assertEqual(result.days[2]['positions'][code]['locked'], held)
        self.assertEqual(result.days[-1]['positions'], {})
        from dataclasses import asdict
        ref = compare_reference(result, targets, quotes, limits, set(), [asdict(action)])
        self.assertEqual(ref['days'], 5)
        self.assertEqual(ref['first_supplied_share_event'], '2025-01-06')

    def test_decimal_reference_hand_calculation(self):
        from fuxi_window_reference_20260923 import simulate
        dates = ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07']
        code = '000001.SZ'
        quotes = {(d, code): {'open': 10 if d < '2025-01-06' else 5,
                              'close': 10 if d < '2025-01-06' else 5} for d in dates}
        limits = {k: {'up_limit': 20, 'down_limit': 1} for k in quotes}
        action = {'event_id': 'share', 'code': code, 'record_date': '2025-01-03',
                  'ex_date': '2025-01-06', 'pay_date': '2025-01-07',
                  'cash_per_share_net': 0.2, 'share_multiplier': 2,
                  'new_shares_date': '2025-01-07'}
        days, fills = simulate(dates, {'2025-01-02': {code: 1}, '2025-01-03': {}},
                               quotes, limits, set(), [action], initial=1000, bps=0)
        self.assertEqual([(r['date'], r['quantity']) for r in fills],
                         [('2025-01-03', 100), ('2025-01-06', -100), ('2025-01-07', -100)])
        self.assertEqual(days[2]['cash'], 500)
        self.assertEqual(days[2]['receivable'], 20)
        self.assertEqual(days[2]['locked'], {code: 100})
        self.assertEqual(days[2]['nav'], 1020)
        self.assertEqual(days[-1]['cash'], 1020)
        self.assertEqual(days[-1]['positions'], {})
        action['share_multiplier'] = 1.003
        with self.assertRaisesRegex(ValueError, 'fractional_entitlement'):
            simulate(dates, {'2025-01-02': {code: 1}}, quotes, limits, set(), [action], initial=1000, bps=0)

    def test_duplicate_identity_is_localized(self):
        import pandas as pd
        from fuxi_window_replay_20260923 import keyed
        frame = pd.DataFrame([{'trade_date': '2025-01-02', 'ts_code': c, 'close': v}
                              for c, v in [('839680.BJ', 10), ('920680.BJ', 11), ('000001.SZ', 12)]])
        conflicts = []
        quotes = keyed(frame, ['close'], conflicts)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(quotes[('2025-01-02', '000001.SZ')]['close'], 12)


if __name__ == '__main__':
    unittest.main()
