"""Conditional source bounds are not actual shareholder allocation receipts."""
import unittest
from copy import deepcopy
from pathlib import Path
from fuxi_v1.contracts import digest,ContractError
from fuxi_v1.datasets import file_hash
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests import test_system
from fuxi_v1.simulation import run_account
from fuxi_v1.restricted_assets import freeze_restricted_policy
from fuxi_v1.conditional_rights import freeze_rules,replay


class ConditionalRightsTests(unittest.TestCase):
    def test_primary_clause_hash_and_missing_clause(self):
        with scratch_directory() as temp:
            root=Path(temp);pdf=root/'test.pdf';text=root/'test.txt'
            import fitz
            with fitz.open() as doc:
                page=doc.new_page();page.insert_text((30,50),'证券代码：001331\n每10股转增4.5股\n2025年1月2日\n2025年1月3日\n不足1股部分按尾数排序，依次派发1股',fontname='china-s',fontsize=10)
                doc.save(pdf)
            with fitz.open(pdf) as doc:text.write_text('\n\n'.join(p.get_text() for p in doc),'utf8')
            source={'path':str(pdf),'sha256':file_hash(pdf),'text_path':str(text),'text_sha256':file_hash(text),
                    'url':'https://static.cninfo.com.cn/fixture.pdf','publication_index':{'ann_date':'20250101'}}
            term={'evidence_id':'e','instrument':'001331.XSHE','ann_date':'2025-01-01','record_date':'2025-01-02',
                  'ex_date':'2025-01-03','share_credit_date':'2025-01-03','status':'ordinary_distribution_supported',
                  'share_distribution':.45,'issuer_evidence':{'sources':[source]}}
            market={'action_terms':[term]};rules=freeze_rules(market)
            self.assertEqual(len(rules['rules']),1);self.assertFalse(rules['formal_eligibility'])
            text.write_text('no fractional clause','utf8')
            with self.assertRaises(ContractError):freeze_rules(market)
            self.assertFalse(freeze_rules({'action_terms':[]})['rules'])

    def test_nonempty_native_fractional_paths_with_restricted_accounts(self):
        test_system.PortfolioExecutionTests.setUpClass();fixture=test_system.PortfolioExecutionTests()
        market=deepcopy(fixture.payload['market']);ir=fixture.ir()
        for key in ['investment_assurance','certification_hash','event_coverage_certified','tradability_certified']:market.pop(key,None)
        before=run_account(ir,market);day=ir['decisions'][len(ir['decisions'])//2]['execution_session']
        index=market['sessions'].index(day);code=next(c for c,q in next(r for r in before['daily'] if r['session']==day)['holdings'].items() if q>0)
        ex=market['sessions'][index+1]
        term={'instrument':code,'record_date':day,'ex_date':ex,'pay_date':None,'status':'ordinary_distribution_supported',
              'gross_cash_per_share':0,'share_distribution':.000137,'share_credit_date':ex,'evidence_id':'bound',
              'tax_fraction':'.2'}
        market['action_terms']=[term]
        rules={'schema_version':'fuxi.conditional_rights_rules.1','market_digest':digest(market),
          'rules':{'bound':{'term_hash':digest(term),'outcomes':'floor_or_ceil','source_scope':'archived_primary_clause'}},
          'max_paths_per_account_per_scenario':8,'formal_eligibility':False}
        rules['rules_hash']=digest(rules)
        with scratch_directory() as temp:
            source=Path(temp)/'declared.txt';source.write_text('fixture source','utf8')
            # Restriction after the event retains real quotes and leaves entitlement branch independent.
            terminal=market['sessions'][index+3]
            policy=freeze_restricted_policy(market,[{'instrument':code,'restrict_from':terminal,
                'known_at':terminal+'T00:00:00+08:00','source_paths':[str(source)],'known_at_basis':'fixture'}],horizon_end=market['sessions'][-1])
            result=replay(ir,ir,market,policy,rules,start=market['sessions'][0],end=market['sessions'][-1])
            self.assertEqual(len(result['scenarios']),3)
            for scenario in result['scenarios']:
                self.assertEqual(len(scenario['pairs']),4)
                self.assertTrue(all(p['n_pairs']>0 for p in scenario['pairs']))
                self.assertTrue(all(not r['mismatches'] for refs in scenario['strict_references'].values() for r in refs))
            self.assertFalse(result['actual_holder_allocation_observed'])


if __name__=='__main__':unittest.main()
