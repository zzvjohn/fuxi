import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from fuxi_issuer_overlay_20260923 import issuer_share


class IssuerShareTests(unittest.TestCase):
    def fixture(self):
        r=dict(ts_code='A',ex_date='20250107',record_date='20250106',pay_date='20250107',
               div_listdate='20250107',imp_ann_date='20250102',div_proc='实施',cash_div_tax=1.,cash_div=1.,
               stk_div=.1,stk_bo_rate=.1,stk_co_rate=None)
        p=dict(code='A',ex_date='2025-01-07',record_date='2025-01-06',pay_date='2025-01-07',
               div_listdate='2025-01-07',imp_ann_date='2025-01-02',cash_gross_per_share=1.,
               bonus_share_per_share=.1,capital_transfer_per_share=0.,normalization={'field':'stk_co_rate','to':0})
        return r,p

    def test_capital_component_zero_only_for_issuer_event(self):
        r,p=self.fixture();a=issuer_share(r,p,12.,10.,'proof')
        self.assertEqual(a.share_multiplier,1.1)
        for changed in [{'ts_code':'B'},{'stk_co_rate':.1},{'stk_co_rate':float('inf')},{'stk_bo_rate':None}]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):issuer_share({**r,**changed},p,12.,10.,'proof')

    def test_differential_factors_do_not_reduce_actual_entitlement(self):
        r,p=self.fixture()
        p['differential_ex_reference']=dict(required=True,total_shares=100,participating_shares=90,
                treasury_excluded_shares=10,cash_total=90.,transfer_shares_total=9.,
                reference_cash_per_share=.9,reference_share_multiplier=1.09)
        a=issuer_share(r,p,12.,(12.-.9)/1.09,'proof')
        self.assertEqual(a.cash_per_share_net,1.)
        self.assertEqual(a.share_multiplier,1.1)
        p['differential_ex_reference']['transfer_shares_total']=10
        with self.assertRaises(ValueError):issuer_share(r,p,12.,(12.-.9)/1.09,'proof')

    def test_wrong_reference_and_late_listing_rejected(self):
        r,p=self.fixture()
        with self.assertRaises(ValueError):issuer_share(r,p,12.,11.,'proof')
        r['div_listdate']='20250106';p['div_listdate']='2025-01-06'
        with self.assertRaises(ValueError):issuer_share(r,p,12.,10.,'proof')


if __name__=='__main__':unittest.main()
