import unittest
from fuxi_v1.public_notices import extract_ordinary


class ApprovedZeroCashTests(unittest.TestCase):
    def fixture(self):
        text=('证券代码300192。2025年年度权益分派实施公告。'
            '一、股东会审议通过利润分配及资本公积金转增股本方案的情况'
            '2025年度资本公积金转增股本方案为:以公司2025年12月31日的总股本329,143,329股为基数,'
            '向全体股东每10股转增4股,不派发现金红利,不送红股。'
            '自上述分配方案披露至本公告日,公司股本总额未发生变化。'
            '二、本次实施的权益分派方案公司2025年年度权益分派方案为:以公司现有总股本329,143,329股为基数,'
            '向全体股东每10股转增4股。'
            '三、股权登记日与除权除息日本次权益分派股权登记日为2026年7月7日,除权除息日为2026年7月8日。'
            '五、权益分派方法本次所转股于2026年7月8日直接记入股东证券账户。'
            '本次所转的无限售条件流通股的起始交易日为2026年7月8日。')
        return text,{'ts_code':'300192.SZ','ann_date':'20260630','title':'2025年年度权益分派实施公告'},            {'instrument':'300192.XSHE','ex_date':'2026-07-08','evidence_id':'event'},            [{'record_date':'20260707','stk_div':'.4','cash_div_tax':None}],{'sha256':'fixture'}

    def test_explicit_approved_zero_same_base_and_rate_links_to_implementation(self):
        r=extract_ordinary(*self.fixture());self.assertEqual(r['status'],'ordinary_terms_proposed')
        self.assertEqual(r['adjudication']['holder_cash'],0);self.assertEqual(r['adjudication']['holder_shares'],.4)
        self.assertIsNone(r['adjudication']['pay_date']);self.assertEqual(r['adjudication']['share_credit_date'],'2026-07-08')
        self.assertIn('unchanged share base',r['adjudication']['cash_zero_basis'])

    def test_missing_or_prior_or_changed_cash_plan_never_supplies_zero(self):
        text,*args=self.fixture()
        changed=[text.replace('不派发现金红利,',''),
            text.replace('公司2025年年度权益分派方案','公司2024年年度权益分派方案'),
            text.replace('股本总额未发生变化','股本总额已发生变化'),
            text.replace('现有总股本329,143,329股','现有总股本330,143,329股'),
            text.replace('向全体股东每10股转增4股。三、','向全体股东每10股转增5股。三、'),
            text.replace('向全体股东每10股转增4股。三、','向全体股东每10股转增4股,每10股派2元(含税)。三、')]
        for t in changed:
            with self.subTest(text=t):self.assertEqual(extract_ordinary(t,*args)['status'],'unresolved')

    def test_pdf_page_number_only_between_explicit_negative_cash_words(self):
        text,*args=self.fixture()
        page_break=text.replace('不派发现金红利','不\n\n\n2\n派发现金红利')
        self.assertEqual(extract_ordinary(page_break,*args)['status'],'ordinary_terms_proposed')
        self.assertEqual(extract_ordinary(text.replace('不派发现金红利','不2派发现金红利'),*args)['status'],'unresolved')


if __name__=='__main__':unittest.main()
