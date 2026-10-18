import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from fuxi_window_followup_evidence_20260923 import unique_contract, differential_cash


class FollowupContractTests(unittest.TestCase):
    def row(self):
        return dict(ts_code='X',div_proc='实施',end_date='20241231',ann_date='20250101',
                    imp_ann_date='20250102',record_date='20250106',ex_date='20250107',
                    pay_date='20250107',stk_div=0.,cash_div_tax=2.,cash_div=2.,
                    stk_bo_rate=None,stk_co_rate=None,div_listdate=None)

    def proof(self):
        return dict(vendor_contract=self.row(),gross_cash_per_share=2.,total_shares=100,
                    eligible_shares=90,repurchased_shares=10,total_cash=180.,
                    ex_reference_deduction_per_share=1.8)

    def test_same_implementation_contract_keeps_null_and_uses_implementation_time(self):
        r=self.row();v,dates=unique_contract([r,{**r,'ann_date':'20241220'}])
        self.assertEqual(v['ann_date'],'20250102')
        self.assertIsNone(v['stk_bo_rate'])
        self.assertEqual(dates,['20241220','20250101'])

    def test_cash_period_or_implementation_conflict_not_collapsed(self):
        r=self.row()
        for changed in [{'cash_div':1.},{'end_date':'20240630'},{'imp_ann_date':'20250103'}]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                unique_contract([r,{**r,**changed}])

    def test_post_implementation_announcement_not_collapsed(self):
        r=self.row()
        with self.assertRaises(ValueError):unique_contract([r,{**r,'ann_date':'20250105'}])

    def test_differential_reference_preserves_holder_cash(self):
        action=differential_cash(self.row(),self.proof(),10.,8.2,'issuer-proof')
        self.assertEqual(action.cash_per_share_net,2.)
        self.assertEqual(action.share_multiplier,1.)

    def test_unproven_formula_or_wrong_raw_reference_rejected(self):
        p=self.proof()
        for changed in [{'total_cash':200.},{'repurchased_shares':9},
                        {'ex_reference_deduction_per_share':2.},{'gross_cash_per_share':1.}]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                differential_cash(self.row(),{**p,**changed},10.,8.2,'issuer-proof')
        with self.assertRaises(ValueError):
            differential_cash(self.row(),p,10.,8.,'issuer-proof')


if __name__=='__main__':unittest.main()
