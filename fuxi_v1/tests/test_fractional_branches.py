from copy import deepcopy
from decimal import Decimal
import unittest
from fuxi_v1.contracts import digest,ContractError
from fuxi_v1.action_accounting import post_entitlements
from fuxi_v1.account import Account
from fuxi_v1.fractional_branches import enumerate_accounts,compare_branches
from fuxi_v1.simulation import run_account,ResearchAccountContext
from fuxi_v1.tests import test_system


class FractionalBranchTests(unittest.TestCase):
    def test_pdf_table_date_boundaries_identity_and_ratio(self):
        from fuxi_v1.fractional_sources import clause_for_term
        term={'instrument':'688215.XSHG','ann_date':'2026-06-03','record_date':'2026-06-09',
              'ex_date':'2026-06-10','share_credit_date':'2026-06-10','share_distribution':.45}
        text='证券代码：688215\n每股转增0.45股\n2026/6/9\n2026/6/10\n不足1股的零碎股份按顺序排列，依次分配1股。'
        self.assertIsNotNone(clause_for_term(text,term,{'ann_date':'20260603'}))
        self.assertIsNone(clause_for_term(text.replace('688215','688557'),term,{'ann_date':'20260603'}))
        self.assertIsNone(clause_for_term(text.replace('0.45','0.4'),term,{'ann_date':'20260603'}))
        self.assertIsNone(clause_for_term(text,term,{'ann_date':'20260611'}))

    def test_each_event_chooses_its_own_integer_and_default_still_blocks(self):
        terms=[dict(instrument='s',status='ordinary_distribution_supported',ex_date=d,record_date='2024-01-01',
                    gross_cash_per_share=0,share_distribution='.1',share_credit_date=d,evidence_id=k,tax_fraction='.2')
               for k,d in [('first','2024-01-03'),('second','2024-01-04')]]
        a=Account('100');post_entitlements(a,terms,'2024-01-03',{'first':105},{},'ir',{},[])
        self.assertTrue(a.unresolved)
        b=Account('100')
        resolve=lambda t,q,e:int(e.to_integral_value(rounding='ROUND_FLOOR' if t['evidence_id']=='first' else 'ROUND_CEILING'))
        for day in ['2024-01-03','2024-01-04']:post_entitlements(b,terms,day,{'first':105,'second':105},{},'ir',{},[],fractional_resolver=resolve)
        self.assertEqual(sum(l['quantity'] for l in b.positions['s']),21)
        with self.assertRaisesRegex(ContractError,'bounds'):
            post_entitlements(Account('100'),terms,'2024-01-03',{'first':105},{},'ir',{},[],fractional_resolver=lambda *args:99)

    def test_reusable_research_context_matches_uncached_execution(self):
        test_system.PortfolioExecutionTests.setUpClass();f=test_system.PortfolioExecutionTests();market=deepcopy(f.payload['market'])
        for k in ['investment_assurance','certification_hash','event_coverage_certified','tradability_certified']:market.pop(k,None)
        ir=f.ir();ir['execution_hash']=digest(market['execution']);ir['ir_hash']=digest({k:v for k,v in ir.items() if k!='ir_hash'})
        a=run_account(ir,market);b=run_account(ir,market,research_context=ResearchAccountContext(market))
        self.assertEqual(a,b)
        with self.assertRaisesRegex(ContractError,'formal'):
            ResearchAccountContext(dict(market,investment_assurance={'receipt_id':'invented'}))

    def test_dynamic_mixed_paths_and_overflow_remain_honest(self):
        from unittest.mock import patch
        terms=[{'evidence_id':'one','instrument':'s','ex_date':'2024-01-03'},
               {'evidence_id':'two','instrument':'s','ex_date':'2024-01-04'}]
        rules={t['evidence_id']:{'term_hash':digest(t),'outcomes':'floor_or_ceil','source_scope':'archived_primary_clause'} for t in terms}
        def account(ir,market,*,fractional_resolver,**kwargs):
            q=fractional_resolver(terms[0],105,Decimal('10.5'))
            # Later entitlement depends on this path's changed holding.
            held=105+q;e=Decimal(held)*Decimal('.1')
            second=int(e) if e==int(e) else fractional_resolver(terms[1],held,e)
            return {'eval_status':'valid','formal_evidence':False,'quantity':held+second}
        with patch('fuxi_v1.fractional_branches.run_account',account):
            r=enumerate_accounts({}, {},rules,max_paths=4)
            self.assertTrue(r['enumeration_complete']);self.assertEqual(r['leaf_count'],4)
            self.assertEqual({tuple(p['choices'][t['evidence_id']] for t in terms) for p in r['paths']},
                             {('floor','floor'),('floor','ceil'),('ceil','floor'),('ceil','ceil')})
            self.assertEqual({e['record_holdings'] for e in r['splits'] if e['evidence_id']=='two'},{115,116})
            r=enumerate_accounts({}, {},rules,max_paths=3)
            self.assertFalse(r['enumeration_complete']);self.assertFalse(r['all_paths_identifiable'])
            self.assertIsNone(compare_branches(r,r,{})['lower'])
            wrong=deepcopy(rules);wrong['one']['term_hash']='different'
            with self.assertRaisesRegex(ContractError,'other terms'):enumerate_accounts({}, {},wrong)

    def test_real_executor_recomputes_later_entitlements_on_each_path(self):
        from fuxi_v1.action_accounting import reconcile_account
        test_system.PortfolioExecutionTests.setUpClass();fixture=test_system.PortfolioExecutionTests()
        market=deepcopy(fixture.payload['market']);ir=fixture.ir()
        for k in ['investment_assurance','certification_hash','event_coverage_certified','tradability_certified']:market.pop(k,None)
        old=run_account(ir,market);trade=ir['decisions'][len(ir['decisions'])//2]['execution_session']
        i=market['sessions'].index(trade);record=next(r for r in old['daily'] if r['session']==trade)
        code=next(c for c,q in record['holdings'].items() if q>0)
        terms=[]
        for number,offset in enumerate((0,2)):
            r,x=market['sessions'][i+offset],market['sessions'][i+offset+1]
            terms.append({'instrument':code,'record_date':r,'ex_date':x,'pay_date':None,'status':'ordinary_distribution_supported',
                'gross_cash_per_share':0,'share_distribution':.000137,'share_credit_date':x,'evidence_id':'path'+str(number),
                'tax_fraction':'.2'})
        market['action_terms']=terms
        rules={t['evidence_id']:{'term_hash':digest(t),'outcomes':'floor_or_ceil','source_scope':'archived_primary_clause'} for t in terms}
        result=enumerate_accounts(ir,market,rules,max_paths=4,research_context=ResearchAccountContext(market))
        self.assertTrue(result['enumeration_complete']);self.assertTrue(result['all_paths_identifiable'])
        self.assertEqual(result['leaf_count'],4)
        self.assertEqual(len({s['record_holdings'] for s in result['splits'] if s['evidence_id']=='path1'}),2)
        for leaf in result['paths']:
            self.assertFalse(leaf['account']['formal_evidence'])
            self.assertTrue(reconcile_account(leaf['account'],market['execution']['initial_cash'],market)['nav_reconciled'])


if __name__=='__main__':unittest.main()
