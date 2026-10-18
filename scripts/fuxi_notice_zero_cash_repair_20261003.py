"""Drafted while rc9 runs; apply only after its ten results are sealed."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
assert json.loads((OUT/'ten_rounds_v2/OUTCOME.json').read_text(encoding='utf-8'))['round_count']==10
assert not (ROOT/'data/fuxi_runtime.lock').exists(),'another manual writer still owns the workspace'
backup=OUT/'before_zero_cash_repair';backup.mkdir()
for name in ('public_notices.py','__init__.py'):
    with (backup/name).open('xb') as h:h.write((ROOT/'fuxi_v1'/name).read_bytes())
p=ROOT/'fuxi_v1/public_notices.py';s=p.read_bytes().decode('utf-8');assert '\r\n' not in s
helper='''def _approved_zero_cash_transfer(compact, implemented):
    """Link explicit approved zero cash only to an unchanged same-year transfer.

    Both sections must name the same share base and transfer ratio. A missing
    implementation cash amount alone, prior-year zero, or a changed plan never
    implies zero. These remain source-bound research terms, not assurance.
    """
    if implemented==compact or any(w in implemented for w in ('现金','红利','股利','分红金额','分配金额','分红比例')) or re.search(r'每10股.{0,18}?派.{0,18}?[0-9]+(?:\\.[0-9]+)?元',implemented):return False
    fiscal=re.search(r'(20\\d{2})(?:年年度|年度|年)权益分派实施公告',compact[:700])
    prior=re.search(r'一、(?:股东大会|股东会)审议通过.{0,50}?方案.{0,20}?情况(?P<body>.*?)二、',compact)
    if not fiscal or not prior:return False
    approved=prior.group('body');year=fiscal.group(1)
    annual=year+r'(?:年年度|年度)(?:资本公积金转增股本方案|利润分配暨资本公积金转增股本方案|权益分派方案)'
    named=re.search(annual,approved)
    if not named or not re.search(annual,implemented):return False
    approved_plan=approved[named.start():].split('。')[0]
    if not any(w in approved_plan for w in ('不派发现金红利','不派发现金股利')):return False
    if not any(w in approved for w in ('自上述分配方案披露至本公告日,公司股本总额未发生变化',
            '自本次利润分配预案披露至本公告披露日,公司股本总额未发生变化')):return False
    rates=lambda t:{Decimal(v) for v in re.findall(r'每10股.{0,18}?转增([0-9]+(?:\\.[0-9]+)?)股',t)}
    a,b=rates(approved_plan),rates(implemented)
    if len(a)!=1 or a!=b or next(iter(a))<=0:return False
    if any(re.search(r'每10股.{0,18}?送(?:红股)?[0-9]+(?:\\.[0-9]+)?股',t) for t in (approved_plan,implemented)):return False
    bases=lambda t:{int(v.replace(',','')) for v in re.findall(r'总股本(?:为)?([0-9,]+)股',t)}
    a,b=bases(approved_plan),bases(implemented)
    return len(a)==1 and a==b


'''
assert s.count('def extract_ordinary(')==1;s=s.replace('def extract_ordinary(',helper+'def extract_ordinary(')
anchor="    cash=set()"
assert s.count(anchor)==1;s=s.replace(anchor,"    linked_approved_zero=not zero_cash and _approved_zero_cash_transfer(compact,amount_text)\n    zero_cash=zero_cash or linked_approved_zero\n"+anchor)
anchor="    if aggregate:judgment['vendor_cash_match']="
assert s.count(anchor)==1;s=s.replace(anchor,"    if linked_approved_zero:judgment['cash_zero_basis']='explicit approved same-year zero cash; implemented transfer ratio and unchanged share base matched'\n"+anchor)
compile(s,str(p),'exec');p.write_bytes(s.encode('utf-8'))
p=ROOT/'fuxi_v1/__init__.py';p.write_bytes(p.read_bytes().replace(b'1.0.0-rc9',b'1.0.0-rc10'))
tests='''import unittest
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
        return text,{'ts_code':'300192.SZ','ann_date':'20260630','title':'2025年年度权益分派实施公告'},\
            {'instrument':'300192.XSHE','ex_date':'2026-07-08','evidence_id':'event'},\
            [{'record_date':'20260707','stk_div':'.4','cash_div_tax':None}],{'sha256':'fixture'}

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


if __name__=='__main__':unittest.main()
'''
compile(tests,'test_approved_zero_cash.py','exec')
with (ROOT/'fuxi_v1/tests/test_approved_zero_cash.py').open('xb') as h:h.write(tests.encode('utf-8'))
for oldname,newname in [('fuxi_notice_tests_v2_20261003.py','fuxi_notice_tests_v3_20261003.py'),
                        ('fuxi_notice_ten_rounds_v2_20261003.py','fuxi_notice_ten_rounds_v3_20261003.py'),
                        ('fuxi_notice_adjudicate_20261003.py','fuxi_notice_adjudicate_v2_20261003.py')]:
    content=(ROOT/'scripts'/oldname).read_bytes().decode('utf-8')
    if 'tests_v3' in newname:content=content.replace('tests_v2','tests_v3')
    elif 'ten_rounds_v3' in newname:
        content=content.replace('ten_rounds_v2','ten_rounds_v3').replace('tests_v2','tests_v3')
        content=content.replace('PUBLIC_IMPLEMENTATION_BUNDLE.json','PUBLIC_IMPLEMENTATION_BUNDLE_v2.json')
    else:content=content.replace('PUBLIC_IMPLEMENTATION_BUNDLE.json','PUBLIC_IMPLEMENTATION_BUNDLE_v2.json')
    with (ROOT/'scripts'/newname).open('xb') as h:h.write(content.encode('utf-8'))
print('rc10 approved-zero chapter linkage applied; rc9 ten-round artifacts unchanged')
