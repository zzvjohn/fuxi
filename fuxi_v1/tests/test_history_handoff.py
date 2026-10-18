"""Operator reachability and lineage fixtures; never certify market data."""
from copy import deepcopy
from pathlib import Path
import unittest
from fuxi_v1.contracts import digest, canonical, evaluation, ContractError
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.tests.test_kernel import scratch_directory


class HistoryHandoffTests(unittest.TestCase):
    def test_generation_retraction_not_hidden_by_formula_cache(self):
        from fuxi_v1.historical_study import memory_state_usable
        with scratch_directory() as directory:
            store=EvidenceStore(Path(directory)/'fixture.sqlite')
            try:
                source=store.append('fixture',{'kind':'shared_formula'})
                origin=store.append('fixture',{'kind':'actual_parent_generation'})
                store.register_consumer('formula',[source],'rebuild')
                store.register_consumer('edge',[origin],'rebuild')
                candidate={'candidate_id':'child','formula_hash':'h'}
                cards={'h':{'state_id':'formula'}};edges={'child':'edge'}
                self.assertTrue(memory_state_usable(store,candidate,cards,edges))
                store.invalidate(origin)
                self.assertTrue(store.state_usable('formula'))
                self.assertFalse(memory_state_usable(store,candidate,cards,edges))
                self.assertTrue(memory_state_usable(store,candidate,cards,{}))
            finally:store.close()

    def test_non_cash_action_does_not_require_a_cash_payment_date(self):
        from fuxi_v1.historical_adapter import dividend_terms
        row={'ts_code':'000100.SZ','ex_date':'20230508','record_date':'20230505','div_proc':'实施',
             'cash_div_tax':None,'stk_div':'0.1','pay_date':None,'imp_ann_date':'20230426'}
        term=dividend_terms([row],['2023-01-01','2023-12-31'])[0]
        self.assertEqual(term['status'],'share_distribution_requires_settlement_adapter')
        self.assertIsNone(term['gross_cash_per_share'])
        self.assertIsNone(term['pay_date'])

    def test_jq_historical_tax_switch_at_effective_date(self):
        from fuxi_v1.jq_strategy import RUNTIME
        calls=[]
        namespace={'FROZEN':{'execution':{'sell_tax':'.0005','buy_commission':'.0003',
                    'sell_commission':'.0003','minimum_commission':'5',
                    'sell_tax_schedule':[['0001-01-01','.001'],['2023-08-28','.0005']]}},
                   'OrderCost':lambda **kw:kw,'set_order_cost':lambda cost,**kw:calls.append(cost)}
        exec(RUNTIME,namespace)
        for day in ('2021-01-04','2023-08-25','2023-08-28','2026-09-24'):
            namespace['_set_execution_cost'](day)
        self.assertEqual([c['close_tax'] for c in calls],[.001,.001,.0005,.0005])

    def test_new_policy_official_route_requires_frozen_context_and_data_receipt(self):
        from fuxi_v1.operations import operate
        from fuxi_v1.artifacts import publish
        from fuxi_v1.research_protocol import policy_from_plan
        class Snapshot:hash=digest('engineering_fixture_snapshot')
        plan={'window':{'research_start':'2022-01-12','end':'2026-09-24'},'reevaluate_formula_count':60,
              'protocol':{'id':'history_research_v2','minimum_detectable_ic':.03,'alpha':.05,'power':.8,
                          'block_grid':[6,12,24],'bootstrap_replicates':3999,'screen_min_sessions':120}}
        baseline={'diagnostics':{'sections':[{'rank_ic':.1 if i%2 else -.09} for i in range(217)]}}
        policy=policy_from_plan(plan,Snapshot(),baseline)
        frozen={'plan_hash':digest(plan),'code_hashes':{'fixture.py':digest('fixture')}}
        result={'candidate_id':'engineering_fixture','snapshot_hash':Snapshot.hash,'policy_hash':policy['policy_hash'],
                'strategy_ir':{'ir_hash':'fixture'},'source_scope':'contract_fixture',
                'evaluations':[evaluation(s,'valid','test_only',decision='pass') for s in ('E0','E1','E2','E3','E4')]}
        outcome={'freeze_hash':digest(frozen),'snapshot_hash':Snapshot.hash,'policy_hash':policy['policy_hash']}
        files={'evaluation.json':result,'policy.json':policy,'history_plan.json':plan,
               'history_freeze.json':frozen,'history_outcome.json':outcome}
        with scratch_directory() as directory:
            root=Path(directory);store=EvidenceStore(root/'fixture_authority.sqlite')
            try:
                bundle=publish(root/'bundles',files,{'engineering_fixture_only':True})
                blocked=operate(store,'admit-research',{'bundle_path':str(bundle)})
                # E0-E4 passing now grants JQ candidacy WITHOUT an investment data contract;
                # the contract is deferred to the live/QMT signal stage (investment_signoff).
                self.assertTrue(blocked['jq_candidate_allowed'])
                self.assertEqual(blocked['blockers'],[])
                self.assertEqual(blocked['investment_certification']['status'],'pending')
                self.assertEqual(blocked['investment_certification']['reviewer_role'],'account_owner_manual_signoff')
                # This fixture receipt is local to a temporary test authority, never the real study.
                receipt=store.receipt('data_contract',{'snapshot_hash':Snapshot.hash,
                      'schema_version':'fuxi.independent_data_assurance.1','investment_eligibility_certified':True,**{k:True for k in
                      ('pit_certified','universe_certified','account_events_certified','execution_rules_certified')}})
                passed=operate(store,'admit-research',{'bundle_path':str(bundle),'data_receipt_id':receipt})
                self.assertTrue(passed['jq_candidate_allowed'])
                self.assertFalse(passed['E5_required_for_first_JQ'])
                broken=deepcopy(files);broken['history_plan.json']['protocol']['alpha']=.2
                bad_bundle=publish(root/'bundles',broken,{'engineering_fixture_only':True})
                with self.assertRaisesRegex(ContractError,'plan'):
                    operate(store,'admit-research',{'bundle_path':str(bad_bundle),'data_receipt_id':receipt})
            finally:store.close()

    def test_historical_study_admission_threads_data_receipt(self):
        from fuxi_v1.historical_study import admit_evaluation
        with scratch_directory() as directory:
            store=EvidenceStore(Path(directory)/'fixture.sqlite')
            try:
                result={'candidate_id':'engineering_fixture','snapshot_hash':digest('engineering_fixture_snapshot'),
                        'policy_hash':digest('fixture_policy'),
                        'strategy_ir':{'ir_hash':'fixture'},'source_scope':'contract_fixture',
                        'evaluations':[evaluation(s,'valid','test_only',decision='pass') for s in ('E0','E1','E2','E3','E4')]}
                evaluation_id=store.append('evaluation',result)
                blocked=admit_evaluation(store,evaluation_id)
                self.assertTrue(blocked['jq_candidate_allowed'])
                self.assertEqual(blocked['blockers'],[])
                self.assertEqual(blocked['investment_certification']['status'],'pending')
                receipt=store.receipt('data_contract',{'snapshot_hash':result['snapshot_hash'],
                      'schema_version':'fuxi.independent_data_assurance.1','investment_eligibility_certified':True,**{k:True for k in
                      ('pit_certified','universe_certified','account_events_certified','execution_rules_certified')}})
                passed=admit_evaluation(store,evaluation_id,receipt)
                self.assertTrue(passed['jq_candidate_allowed'])
                self.assertFalse(passed['E5_required_for_first_JQ'])
            finally:store.close()


if __name__=='__main__':unittest.main()
