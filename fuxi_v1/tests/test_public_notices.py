import unittest
import json
from pathlib import Path
from fuxi_v1.datasets import file_hash
from fuxi_v1.contracts import ContractError
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.public_notices import extract_ordinary
from fuxi_v1.history import jq_code


class PublicNoticeTests(unittest.TestCase):
    def fixture(self):
        original={'instrument':jq_code('000100.SZ'),'ex_date':'2023-05-08','evidence_id':'event'}
        index={'ts_code':'000100.SZ','ann_date':'20230425','title':'权益分派实施公告'}
        rows=[{'record_date':'20230505','stk_div':'.1','cash_div_tax':None}]
        source={'sha256':'PDF'}
        text=('证券代码000100。本年度不进行现金分红，不送红股。向全体股东每10股转增1股。'
              '股权登记日2023年5月5日，除权除息日为2023年5月8日。'
              '本次所转增股份于2023年5月8日直接计入证券账户。起始交易日为2023年5月8日。')
        return text,index,original,rows,source

    def test_explicit_pure_capitalization_not_missing_cash_zero(self):
        args=self.fixture();result=extract_ordinary(*args)
        self.assertEqual(result['status'],'ordinary_terms_proposed')
        self.assertEqual(result['adjudication']['holder_cash'],0)
        self.assertEqual(result['adjudication']['holder_shares'],.1)
        self.assertIsNone(result['adjudication']['pay_date'])
        text=args[0].replace('本年度不进行现金分红，不送红股。','')
        self.assertEqual(extract_ordinary(text,*args[1:])['reason'],'explicit_gross_cash_rate_missing_or_ambiguous')

    def test_aggregate_rate_or_unobserved_credit_stays_unknown(self):
        args=self.fixture()
        text=args[0]+'按总股本每10股转增0.98股。'
        self.assertEqual(extract_ordinary(text,*args[1:])['status'],'unresolved')
        text=args[0].replace('本次所转增股份于2023年5月8日直接计入证券账户。','')
        self.assertEqual(extract_ordinary(text,*args[1:])['reason'],'separate_credit_and_tradability_dates_not_matched')

    def test_restructuring_and_wrong_issuer_are_not_ordinary(self):
        args=self.fixture()
        self.assertEqual(extract_ordinary('重整计划'+args[0],*args[1:])['reason'],'restructuring_requires_holder_specific_rights_review')
        self.assertEqual(extract_ordinary(args[0].replace('000100','999999'),*args[1:])['reason'],'issuer_identity_not_matched')

    def test_cash_rate_requires_payment_and_record_crosscheck(self):
        text,index,original,rows,source=self.fixture()
        rows=[{'record_date':'20230505','stk_div':'0','cash_div_tax':'.2'}]
        text=('证券代码000100。每10股派发现金红利2元(含税)，不送红股，不以资本公积金转增股本。'
              '股权登记日2023年5月5日，除息日2023年5月8日，现金红利发放日2023年5月8日。')
        result=extract_ordinary(text,index,original,rows,source)
        self.assertEqual(result['status'],'ordinary_terms_proposed')
        partial=[{'record_date':'20230505','stk_div':None,'cash_div_tax':None}]+rows
        self.assertEqual(extract_ordinary(text,index,original,partial,source)['status'],'ordinary_terms_proposed')
        rows[0]['record_date']='20230504'
        self.assertEqual(extract_ordinary(text,index,original,rows,source)['reason'],'no_matching_vendor_entitlement_component')

    def test_equal_bonus_and_capitalization_are_added(self):
        text,index,original,rows,source=self.fixture()
        text=text.replace('不送红股。','').replace('每10股转增1股。','每10股转增1股。每10股送红股1股。')
        rows[0]['stk_div']='.2'
        self.assertEqual(extract_ordinary(text,index,original,rows,source)['adjudication']['holder_shares'],.2)

    def test_section_heading_does_not_assign_record_date_to_ex_date(self):
        text,index,original,rows,source=self.fixture()
        text=text.replace('股权登记日2023', '三、股权登记日与除权除息日本次权益分派股权登记日2023')
        self.assertEqual(extract_ordinary(text,index,original,rows,source)['status'],'ordinary_terms_proposed')

    def test_archived_credit_phrase_variants_preserve_same_terms(self):
        text,index,original,rows,source=self.fixture()
        for phrase in ('本次所转股于','本次所送（转）股于','本次所转增的股份将于','本次转增的股份将于'):
            with self.subTest(phrase=phrase):
                v=extract_ordinary(text.replace('本次所转增股份于',phrase),index,original,rows,source)
                self.assertEqual(v['status'],'ordinary_terms_proposed')
                self.assertEqual(v['adjudication']['share_credit_date'],'2023-05-08')

    def test_generic_record_reference_cannot_capture_financial_year_end(self):
        text,index,original,rows,source=self.fixture()
        text='公司实施方案时股权登记日的总股本为基数，报告期末为2022年12月31日。'+text
        self.assertEqual(extract_ordinary(text,index,original,rows,source)['status'],'ordinary_terms_proposed')

    def test_actual_combined_section_and_payment_clock(self):
        text,index,original,rows,source=self.fixture()
        rows=[{'record_date':'20230505','stk_div':'0','cash_div_tax':v} for v in ('.4','1.0')]
        text=('证券代码000100。中期每10股派4元人民币现金(含税)，特别每10股派10元人民币现金(含税)。'
              '二、权益分派方案合计向全体股东每10股派14元人民币现金(含税)，不进行公积金转增股本，不送红股。'
              '三、股权登记日股权登记日2023年5月5日，除息日2023年5月8日。'
              '现金红利将于2023年5月8日通过托管机构直接划入资金账户。')
        v=extract_ordinary(text,index,original,rows,source)
        self.assertEqual(v['status'],'ordinary_terms_proposed')
        self.assertEqual(v['adjudication']['holder_cash'],1.4)
        self.assertEqual(extract_ordinary(text.replace('合计',''),index,original,rows,source)['status'],'unresolved')

    def test_capitalization_credit_and_reference_ratio_are_separate(self):
        text,index,original,rows,source=self.fixture()
        text=('证券代码000100。按总股本折算的每10股转增股本比例=0.980000股。'
              '二、本次实施的资本公积金转增股本方案向全体股东每10股转增1股，不派发现金红利，不送红股。'
              '三、股权登记日股权登记日2023年5月5日，除权日（除息日）2023年5月8日。'
              '新增可流通股份上市日2023年5月8日。'
              '本公司此次权益分派转增的股份将于2023年5月8日直接记入股东证券账户。')
        v=extract_ordinary(text,index,original,rows,source)
        self.assertEqual(v['status'],'ordinary_terms_proposed')
        self.assertEqual(v['adjudication']['holder_shares'],.1)

    def test_dated_BSE_identity_and_vendor_index_alias_are_distinct(self):
        text,index,original,rows,source=self.fixture()
        text=text.replace('000100','834001');index['ts_code']='834001.BJ';original['instrument']='920001.BJ'
        with scratch_directory() as temp:
            catalog=Path(temp)/'catalog.json'
            catalog.write_text(json.dumps({'identity_aliases':{'834001.BJ':{'stable_code':'920001.BJ',
                'effective_session':'2025-10-09','mapping_evidence_id':'mapping'}}}),encoding='utf-8')
            source.update(identity_mapping_catalog_path=str(catalog),identity_mapping_catalog_sha256=file_hash(catalog))
            self.assertEqual(extract_ordinary(text,index,original,rows,source)['status'],'ordinary_terms_proposed')
            text=text.replace('2023','2026');index['ann_date']='20260425';original['ex_date']='2026-05-08';rows[0]['record_date']='20260505'
            self.assertEqual(extract_ordinary(text,index,original,rows,source)['reason'],'issuer_identity_not_matched')
            self.assertEqual(extract_ordinary(text.replace('834001','920001'),index,original,rows,source)['status'],'ordinary_terms_proposed')

    def test_BSE_total_rate_and_taxable_reserve_breakdown(self):
        text,index,original,rows,source=self.fixture()
        rows=[{'record_date':'20230505','stk_div':'.2','cash_div_tax':'.13'}]
        text=('证券代码000100。一、权益分派方案1、以公司现有总股本为基数，向全体股东每10股转增2股，'
              '（其中以股票发行溢价形成的资本公积金每10股转增2股，不需要纳税；以其他资本公积每10股转增0股，需要纳税），'
              '每10股派人民币现金1.30元。2、扣税说明个人股东适用股息红利差别化个人所得税政策，持股一个月每10股补缴税款0.26元。'
              '二、权益登记日与除权除息日本次权益分派权益登记日为2023年5月5日，除权除息日为2023年5月8日。'
              '本次所送（转）股于2023年5月8日直接记入股东证券账户。现金红利将于2023年5月8日划入资金账户。起始交易日为2023年5月8日。')
        v=extract_ordinary(text,index,original,rows,source)
        self.assertEqual(v['status'],'ordinary_terms_proposed')
        self.assertEqual(v['adjudication']['holder_shares'],.2)
        self.assertEqual(v['adjudication']['holder_cash'],.13)
        reordered=text.replace('派人民币现金1.30元','派1.30元人民币现金')
        self.assertEqual(extract_ordinary(reordered,index,original,rows,source)['adjudication']['holder_cash'],.13)
        self.assertEqual(extract_ordinary(text.replace('转增0股','转增1股'),index,original,rows,source)['status'],'unresolved')
        self.assertEqual(extract_ordinary(text.replace('2、扣税说明','税率'),index,original,rows,source)['status'],'unresolved')

    def test_explicit_zero_in_confirmed_unchanged_plan(self):
        text,index,original,rows,source=self.fixture()
        text=('证券代码000100。本年度不派发现金股利，不送红股，每10股转增1股。'
              '本次实施的分派方案与股东大会审议通过的分配预案及其调整原则是一致的。'
              '二、本次权益分派方案每10股转增1股。三、股权登记日2023年5月5日，除权除息日为2023年5月8日。'
              '本次权益分派所转股将于2023年5月8日直接记入股东账户。起始交易日为2023年5月8日。')
        self.assertEqual(extract_ordinary(text,index,original,rows,source)['status'],'ordinary_terms_proposed')
        self.assertEqual(extract_ordinary(text.replace('与股东大会审议通过的分配预案及其调整原则是一致的','另有修订'),index,original,rows,source)['status'],'unresolved')

    def test_import_reproduces_terms_and_refuses_changed_sources(self):
        text,index,original,rows,source=self.fixture()
        original.update(status='share_distribution_requires_settlement_adapter',
            record_date='2023-05-05',pay_date=None,ann_date='2023-04-25',gross_cash_per_share=None,share_distribution=.1)
        rows[0].update(ts_code='000100.SZ',ex_date='20230508',div_proc='实施')
        with scratch_directory() as temp:
            pdf=Path(temp)/'source.pdf';txt=Path(temp)/'source.txt'
            pdf.write_bytes(b'%PDF-fixture');txt.write_text(text,encoding='utf-8')
            source.update(url='https://static.cninfo.com.cn/fixture.pdf',path=str(pdf),sha256=file_hash(pdf),
                          text_path=str(txt),text_sha256=file_hash(txt))
            judgment=extract_ordinary(text,index,original,rows,source)['adjudication']
            self.assertEqual(resolve_action_terms([original],rows,[judgment])[0]['gross_cash_per_share'],0)
            altered=json.loads(json.dumps(judgment));altered['holder_shares']=.2
            with self.assertRaises(ContractError):resolve_action_terms([original],rows,[altered])
            txt.write_text('modified',encoding='utf-8')
            with self.assertRaises(ContractError):resolve_action_terms([original],rows,[judgment])
