"""Hand-computed checks for the independent cash-only reference."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from fuxi_reference_cash_20260923 import simulate


class ReferenceCashTests(unittest.TestCase):
    def fixtures(self):
        days = ['2025-01-02', '2025-01-03', '2025-01-06']
        quotes = {(d, 'A'): {'open': 10, 'close': 10} for d in days}
        limits = {(d, 'A'): {'up_limit': 11, 'down_limit': 9} for d in days}
        return days, quotes, limits

    def test_cash_dividend_receivable_then_payment(self):
        days, quotes, limits = self.fixtures()
        quotes[(days[1], 'A')] = {'open': 9, 'close': 9}
        quotes[(days[2], 'A')] = {'open': 9, 'close': 9}
        quotes[('2024-12-31', 'A')] = {'close': 10}
        actions = [{'event_id': 'one', 'code': 'A', 'record_date': days[0],
                    'ex_date': days[1], 'pay_date': days[2], 'share_multiplier': 1,
                    'cash_per_share_net': 1}]
        rows, fills = simulate(days, {'2024-12-31': {'A': 1}}, quotes, limits, set(), actions,
                               initial=10000, bps=0)
        self.assertEqual(len(fills), 1)
        self.assertEqual([r['nav'] for r in rows], [10000, 10000, 10000])
        self.assertEqual([r['receivable'] for r in rows], [0, 1000, 0])
        self.assertEqual(rows[-1]['cash'], 1000)

    def test_expired_signal_cancels_delayed_limit_order(self):
        days, quotes, limits = self.fixtures()
        quotes[(days[1], 'A')]['open'] = 11
        rows, fills = simulate(days, {days[0]: {'A': 1}, days[1]: None}, quotes, limits, set(), [],
                               initial=10000, bps=0)
        self.assertEqual(fills, [])
        self.assertEqual(rows[-1]['nav'], 10000)

    def test_halted_exit_keeps_shares_and_last_mark(self):
        days, quotes, limits = self.fixtures()
        quotes[('2024-12-31', 'A')] = {'close': 10}
        quotes.pop((days[2], 'A'))
        rows, fills = simulate(days, {'2024-12-31': {'A': 1}, days[1]: {}}, quotes, limits,
                               {(days[2], 'A')}, [], initial=10000, bps=15)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]['quantity'], 900)
        self.assertEqual(rows[-1]['positions'], {'A': 900})
        self.assertEqual(rows[-1]['cash'], 986.5)
        self.assertEqual(rows[-1]['nav'], 9986.5)


if __name__ == '__main__':
    unittest.main()
