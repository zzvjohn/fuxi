import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from fuxi_distribution_evidence_20260923 import checked_distribution, normalize_issuer_event


class DistributionEvidenceTests(unittest.TestCase):
    def row(self):
        return dict(ts_code='A',div_proc='实施',ex_date='20250107',record_date='20250106',
                    imp_ann_date='20250102',pay_date='20250109',div_listdate='20250110',
                    cash_div_tax=1.,cash_div=1.,stk_div=1.,stk_bo_rate=0.,stk_co_rate=1.)

    def check(self, row, reference=4.5):
        return checked_distribution(row,'A','2025-01-07',10.,reference,'source-sha')

    def test_cash_and_stock_reference_and_lock_date(self):
        action=self.check(self.row())
        self.assertEqual(action.share_multiplier,2)
        self.assertEqual(action.new_shares_date,'2025-01-10')
        self.assertEqual(action.pay_date,'2025-01-09')

    def test_missing_stock_listing_date_is_rejected(self):
        r=self.row();r['div_listdate']=None
        with self.assertRaises(ValueError):self.check(r)

    def test_pure_stock_zero_payment_marker_not_inferred_cash(self):
        r=self.row();r.update(cash_div_tax=0.,cash_div=0.,pay_date=None)
        with self.assertRaises(ValueError):self.check(r,5.)
        a=checked_distribution(r,'A','2025-01-07',10.,5.,'source-sha',
                               allocation_evidence='synthetic issuer record-holder allocation')
        self.assertEqual(a.cash_per_share_net,0)
        self.assertEqual(a.pay_date,a.ex_date)

    def test_wrong_reference_or_late_announcement_rejected(self):
        with self.assertRaises(ValueError):self.check(self.row(),9.)
        r=self.row();r['imp_ann_date']='20250108'
        with self.assertRaises(ValueError):self.check(r)

    def test_missing_or_nonfinite_cash_not_silently_zeroed(self):
        r=self.row();r['cash_div']=float('nan')
        with self.assertRaises(ValueError):self.check(r)

    def test_unknown_allocation_components_not_inferred(self):
        r=self.row();r['stk_co_rate']=None
        with self.assertRaises((ValueError, TypeError)):self.check(r)
        r=self.row();r['stk_co_rate']=.5
        with self.assertRaises(ValueError):self.check(r)

    def test_issuer_normalization_requires_exact_terms(self):
        r=self.row();r['stk_bo_rate']=None
        proof=dict(code='A',ex_date='2025-01-07',record_date='2025-01-06',
                   pay_date='2025-01-09',div_listdate='2025-01-10',imp_ann_date='2025-01-02',
                   cash_gross_per_share=1.,capital_transfer_per_share=1.,bonus_share_per_share=0.,
                   normalization={'field':'stk_bo_rate'})
        self.assertEqual(normalize_issuer_event(r,proof)['stk_bo_rate'],0.)
        for changed in ({'ts_code':'B'},{'record_date':'20250103'},
                        {'cash_div_tax':.5},{'stk_co_rate':.5},{'stk_bo_rate':.1},
                        {'stk_bo_rate':float('inf')},{'stk_bo_rate':float('-inf')}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                normalize_issuer_event({**r,**changed},proof)


if __name__=='__main__':unittest.main()
