"""Actual frozen issuer originals: company capital is not stockholder entitlement."""
from copy import deepcopy
from pathlib import Path
import re
import shutil
import unittest
from fuxi_v1.account import Account
from fuxi_v1.action_accounting import post_entitlements
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.contracts import ContractError,load_json
from fuxi_v1.datasets import file_hash
from fuxi_v1.historical_adapter import dividend_terms
from fuxi_v1.public_notices import extract_restructuring,verified_restructuring_source
from fuxi_v1.tests.test_kernel import scratch_directory


class PublicRestructuringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[2]/'reports/fuxi_v1_full_landing_20261009/restructuring_000430'
        record=load_json(cls.root/'NATIVE_REPARSE_RESULT.json')
        cls.original=record['original_term'];cls.rows=record['vendor_rows']
        cls.judgment=load_json(cls.root/'ADJUDICATION.json')
        cls.source=cls.judgment['sources'][0]
        cls.text=verified_restructuring_source(cls.source)

    def parse(self,text=None,rows=None,index=None,original=None,*,allow_late=False):
        return extract_restructuring(self.text if text is None else text,
            self.source['publication_index'] if index is None else index,
            self.original if original is None else original,self.rows if rows is None else rows,
            self.source,allow_late=allow_late)

    def test_actual_original_bytes_native_term_and_two_publication_clocks(self):
        base=dividend_terms(self.rows,['2025-12-26','2025-12-29'])
        self.assertEqual(len(base),1)
        self.assertEqual(base[0]['evidence_id'],'673c5541c7fd0400aef691f02af213d32bc64352a02e3612837bb8f2add94f92')
        resolved=resolve_action_terms(base,self.rows,[self.judgment])[0]
        self.assertEqual(resolved['status'],'restructuring_no_holder_distribution')
        self.assertEqual((resolved['share_distribution'],resolved['gross_cash_per_share']),(0,0))
        self.assertEqual(self.judgment['company_share_distribution'],1)
        self.assertEqual(self.judgment['company_new_shares'],404817686)
        self.assertEqual(self.judgment['investor_new_shares']+self.judgment['creditor_new_shares'],404817686)
        self.assertEqual((resolved['record_date'],resolved['ex_date']),('2025-12-26','2025-12-29'))
        self.assertEqual(resolved['ann_date'],'2025-12-25')
        self.assertEqual(self.judgment['available_at_assumption'],'2025-12-26T00:00:00+08:00')
        self.assertFalse(self.judgment['publication_hour_verified'])
        self.assertFalse(self.judgment['independent_assurance'])
        self.assertIn('no broker or registration receipt',resolved['settlement_scope'])
        for source in self.judgment['sources']+self.judgment['corroborating_sources']:
            self.assertGreater(len(verified_restructuring_source(source)),1000)

    def test_account_old_shares_cash_and_claims_do_not_double(self):
        term=resolve_action_terms([self.original],self.rows,[self.judgment])[0]
        account=Account('2000')
        account.apply({'event_id':'buy','kind':'fill','timestamp':'2025-12-25T09:30:00+08:00','evidence_id':'fixture',
            'payload':{'instrument':term['instrument'],'side':'buy','quantity':100,'price':'10','fee':'0',
                'sellable_at':'2025-12-26T09:00:00+08:00'}})
        cash=account.cash;gaps=[];claims={}
        post_entitlements(account,[term],term['ex_date'],{term['evidence_id']:100},claims,'fixture',{},gaps)
        self.assertEqual(sum(p['quantity'] for p in account.positions[term['instrument']]),100)
        self.assertEqual(account.cash,cash)
        self.assertFalse(account.receivables);self.assertFalse(account.share_receivables)
        self.assertFalse(claims);self.assertFalse(gaps)

    def test_late_completion_cannot_be_primary_or_backdate_index(self):
        source=self.judgment['corroborating_sources'][0]
        text=verified_restructuring_source(source)
        args=(text,source['publication_index'],self.original,self.rows,source)
        self.assertEqual(extract_restructuring(*args)['reason'],'publication_after_record_date')
        self.assertEqual(extract_restructuring(*args,allow_late=True)['status'],'restructuring_terms_proposed')
        index=dict(source['publication_index'],ann_date='20251225')
        self.assertEqual(extract_restructuring(text,index,self.original,self.rows,source)['reason'],
            'publication_date_not_reproduced')
        changed=deepcopy(self.judgment);changed['sources']=[source]
        with self.assertRaisesRegex(ContractError,'primary terms no longer reproduce'):
            resolve_action_terms([self.original],self.rows,[changed])

    def test_explicit_zero_cash_and_no_original_share_distribution_required(self):
        text=re.sub(r'\s+','',self.text).replace('，',',')
        missing=text.replace('由于不涉及现金红利、股票红利及配股,公式中现金红利为0','')
        self.assertEqual(self.parse(missing)['reason'],'explicit_no_cash_holder_distribution_missing')
        missing=text.replace('不向原股东进行分配','向原股东进行分配')
        self.assertEqual(self.parse(missing)['reason'],'all_new_shares_destination_not_explicit')
        conflicting=text+'向原股东派发现金。'
        # A footer not at the end also rejects unauthenticated appended terms.
        self.assertEqual(self.parse(conflicting)['status'],'unresolved')

    def test_share_destination_sum_and_company_base_must_reconcile(self):
        text=re.sub(r'\s+','',self.text)
        changed=text.replace('38,817,686股用于偿付债务','38,817,685股用于偿付债务')
        self.assertNotEqual(changed,text)
        self.assertEqual(self.parse(changed)['reason'],'new_share_allocation_does_not_reconcile')
        changed=text.replace('现有总股本404,817,686股','现有总股本404,817,685股')
        self.assertNotEqual(changed,text)
        self.assertEqual(self.parse(changed)['reason'],'capital_base_and_conversion_do_not_reconcile')

    def test_wrong_identity_dates_and_any_conflicting_vendor_row_rejected(self):
        self.assertEqual(self.parse(index=dict(self.source['publication_index'],ts_code='000431.SZ'))['reason'],
            'issuer_identity_not_matched')
        self.assertEqual(self.parse(original=dict(self.original,ex_date='2025-12-30'))['reason'],
            'record_or_listing_date_ambiguous')
        for changed in ('0.9','NaN',None):
            with self.subTest(changed=changed):
                rows=self.rows+[dict(self.rows[0],stk_div=changed)]
                self.assertEqual(self.parse(rows=rows)['reason'],'issuer_vendor_capital_event_mismatch')

    def test_mutated_claim_or_bound_vendor_hash_cannot_be_imported(self):
        for field,value in (('holder_shares',1.),('holder_cash',1.),('company_share_distribution',.9),
                ('available_at_assumption','2025-12-25T00:00:00+08:00'),('vendor_evidence_id','wrong')):
            changed=deepcopy(self.judgment);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ContractError):
                resolve_action_terms([self.original],self.rows,[changed])

    def test_changed_archived_text_rejected_even_after_claimed_rehash(self):
        with scratch_directory() as temp:
            source=deepcopy(self.source);path=Path(temp)/'canonical.txt'
            path.write_bytes(Path(source['text_path']).read_bytes()+b'\nchanged')
            source['text_path']=str(path)
            with self.assertRaisesRegex(ContractError,'source changed'):verified_restructuring_source(source)
            source['text_sha256']=file_hash(path)
            with self.assertRaisesRegex(ContractError,'PDF and archived restructuring text differ'):
                verified_restructuring_source(source)

    def test_changed_raw_source_and_unapproved_host_rejected(self):
        with scratch_directory() as temp:
            source=deepcopy(self.source);path=Path(temp)/'original.pdf'
            shutil.copyfile(source['path'],path)
            with path.open('ab') as stream:stream.write(b'changed')
            source['path']=str(path)
            with self.assertRaisesRegex(ContractError,'source changed'):verified_restructuring_source(source)
        source=dict(self.source,url='https://example.org/terms.pdf')
        with self.assertRaisesRegex(ContractError,'publication host'):verified_restructuring_source(source)

    def test_contradictory_later_terms_and_fake_independent_assurance_rejected(self):
        changed=deepcopy(self.judgment)
        changed['corroborating_sources'][0]['publication_index']['ann_date']='20251225'
        with self.assertRaisesRegex(ContractError,'publication order invalid'):
            resolve_action_terms([self.original],self.rows,[changed])
        changed=deepcopy(self.judgment);changed['independent_assurance']=True
        with self.assertRaisesRegex(ContractError,'not independent assurance'):
            resolve_action_terms([self.original],self.rows,[changed])


if __name__=='__main__':unittest.main()
