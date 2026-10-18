import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('cash_event_evidence', ROOT / 'scripts/fuxi_event_evidence_20260923.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class CashEventEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(ts_code='600116.SH', div_proc='实施', ex_date='20250108',
                        record_date='20250107', pay_date='20250108', imp_ann_date='20250102',
                        cash_div_tax=.05, cash_div=.05, stk_div=0.)

    def check(self):
        return evidence.checked_cash_action(self.row, '600116.SH', '2025-01-08', 10., 9.95, 'fixture-hash')

    def test_implemented_cash_event_reconciles_reference(self):
        action = self.check()
        self.assertEqual(action.cash_per_share_net, .05)
        self.assertEqual(action.record_date, '2025-01-07')

    def test_preliminary_approval_is_not_implementation(self):
        self.row['div_proc'] = '股东大会通过'
        with self.assertRaises(ValueError):
            self.check()

    def test_late_announcement_is_not_point_in_time_evidence(self):
        self.row['imp_ann_date'] = '20250109'
        with self.assertRaises(ValueError):
            self.check()

    def test_cash_amount_cannot_override_mismatching_price_reference(self):
        self.row['cash_div_tax'] = .2
        with self.assertRaises(ValueError):
            self.check()

    def test_bonus_or_invalid_net_requires_separate_contract(self):
        for key, value in [('stk_div', .1), ('cash_div', .06), ('cash_div', float('nan'))]:
            old = self.row[key]
            self.row[key] = value
            with self.assertRaises(ValueError):
                self.check()
            self.row[key] = old


if __name__ == '__main__':
    unittest.main()
