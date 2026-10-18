"""Learning, admission and checkpoint contracts; fixture passes grant no market qualification."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.tests.test_certification import reviewed_certification_fixture
from fuxi_v1.contracts import canonical, digest, evaluation, ContractError, timestamp
from fuxi_v1.learning import (stage_feedback, export_memory, valid_imports, refresh_import_dependencies,
                             behavioral_signature, checkpoint, paired_parent, prompt_cards, feedback_dependencies)
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.campaign import run_campaign
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.acceptance import AcceptanceService
from fuxi_v1.operations import operate
from fuxi_v1.artifacts import publish
from fuxi_v1.search import generate
from fuxi_v1.data import Snapshot, Labels


class LearningTests(unittest.TestCase):
    def test_llm_evidence_links_local_and_imported_state(self):
        card={'candidate_id':'old','evaluation_id':'ev','state_id':'state'}
        candidate={'parent_ids':[],'source_evidence_ids':['ev']}
        token={'source_path':'source','source_event':'event'}
        dependencies=feedback_dependencies(candidate,[card],[{'records':[{'card':card,'token':token}]}])
        self.assertEqual(set(dependencies),{'state','import_state_'+digest(token)})
        self.assertEqual(feedback_dependencies({'parent_ids':[]},[card],[]),[])

    def test_same_candidate_other_source_does_not_mask_revocation(self):
        with scratch_directory() as directory:
            root=Path(directory);a=EvidenceStore(root/'a.sqlite');b=EvidenceStore(root/'b.sqlite');t=EvidenceStore(root/'target.sqlite')
            try:
                manifests=[];events=[]
                for db in (a,b):
                    db.campaign('c',{'learning_compatibility':'k','budget_units':1})
                    event=db.append('evaluation',{'evidence':'valid'})
                    slot=['decision','candidate','capability','window','baseline']
                    reward=db.reward(slot,'capability',1.,event);events.append((slot,reward,event))
                    db.register_consumer('s',[reward,event],'rebuild')
                    db.append('feedback_v2',{'state_id':'s','candidate_id':'same','evaluation_id':event})
                    manifests.append(export_memory(db.path,'k'))
                refresh_import_dependencies(t,manifests)
                states=['import_state_'+digest(m['records'][0]['token']) for m in manifests]
                t.register_consumer('llm_child',[states[0]],'rebuild')
                slot,reward,event=events[0];a.reward(slot,'capability',None,event,supersedes=reward)
                self.assertEqual(len(refresh_import_dependencies(t,manifests)),1)
                self.assertFalse(t.state_usable(states[0]));self.assertTrue(t.state_usable(states[1]))
                self.assertFalse(t.state_usable('llm_child'))
            finally:a.close();b.close();t.close()

    def test_power_shortfall_is_not_economic_failure(self):
        e1=evaluation('E1','valid','old_gate',decision='reject',reject_class='statistical',
                      diagnostics={'valid_section_count':563,'power_required_sections':7075,
                                   'time_uncertainty':{'lower':.01},'tail_resolution_adequate':False})
        e3=evaluation('E3','data_insufficient','missing_event')
        result=stage_feedback({'evaluations':[e1,e3]})
        self.assertIn('power_plan_sample_shortfall',result[0]['causes'])
        self.assertNotIn('economic_reject',str(result))
        self.assertEqual(result[1]['causes'],['data_insufficient'])

    def test_ordinal_behavior_collapses_monotonic_duplicates(self):
        a={'d':{'a':1,'b':2,'c':None}}
        b={'d':{'a':11,'b':21,'c':None}}
        self.assertEqual(behavioral_signature(a,'d','d'),behavioral_signature(b,'d','d'))
        b['d']['c']=0
        self.assertNotEqual(behavioral_signature(a,'d','d'),behavioral_signature(b,'d','d'))

    def test_parent_comparison_asserts_shared_support(self):
        p=synthetic_campaign(130,6);s=Snapshot(p['snapshot']);labels=Labels(p['labels'],s)
        days=s.sessions[90:95];codes=s.universe[days[0]]
        a={d:{code:float(i) for i,code in enumerate(codes)} for d in days}
        b=deepcopy(a)
        for d in days:a[d][codes[0]]=None
        result=paired_parent(a,b,s,labels,days[0],days[-1],timestamp(p['research_clock']))
        self.assertEqual(result['n_pairs'],25)
        self.assertEqual(result['mean_ic_increment'],0)

    def test_checkpoint_sterility_and_errors(self):
        history=[{}]*5
        first=checkpoint(history,[],[],5)
        self.assertFalse(first['stop'])
        second=checkpoint([{}]*10,[],[],5,first)
        self.assertTrue(second['stop'])
        self.assertEqual(second['decision'],'stop_two_sterile_blocks')

    def test_memory_revision_revokes_descendants(self):
        with scratch_directory() as directory:
            path=Path(directory)/'source.sqlite';s=EvidenceStore(path);t=EvidenceStore(Path(directory)/'target.sqlite')
            try:
                s.campaign('c',{'learning_compatibility':'k','budget_units':1})
                ev=s.append('evaluation',{'value':'observed'})
                slot=['decision','candidate','capability','window','baseline']
                reward=s.reward(slot,'capability',1.,ev)
                s.register_consumer('s',[reward,ev],'rebuild')
                s.append('feedback_v2',{'state_id':'s','candidate_id':'c','value':1})
                manifest=export_memory(path,'k')
                self.assertEqual(len(valid_imports([manifest])),1)
                refresh_import_dependencies(t,[manifest])
                state='import_state_'+digest(manifest['records'][0]['token'])
                t.register_consumer('child',[state],'rebuild')
                t.register_consumer('grandchild',['child'],'rebuild')
                s.reward(slot,'capability',None,ev,supersedes=reward)
                self.assertEqual(refresh_import_dependencies(t,[manifest]),[])
                self.assertFalse(t.state_usable('grandchild'))
                with self.assertRaisesRegex(ContractError,'incompatible'):export_memory(path,'different_cost_or_window')
            finally:s.close();t.close()

    def test_official_operator_path_and_tail_guard(self):
        with scratch_directory() as directory:
            root=Path(directory);store=EvidenceStore(root/'isolated_fixture_authority.sqlite')
            try:
                spec=reviewed_certification_fixture(root)
                registered=operate(store,'register-data',spec)
                self.assertIsNone(registered['receipt_id'])
                receipt=registered['validation_receipt_id']
                policy={'status':'calibrated_research_profile','scope':'fixture_only'}
                policy['policy_hash']=digest(policy)
                result={'candidate_id':'fixture','snapshot_hash':spec['snapshot_hash'],'policy_hash':policy['policy_hash'],
                        'strategy_ir':{'ir_hash':'fixture'},'source_scope':'contract_fixture',
                        'evaluations':[evaluation(s,'valid','test_only',decision='pass') for s in ('E0','E1','E2','E3','E4')]}
                bundle=publish(root/'bundles',{'evaluation.json':result,'policy.json':policy},{'test_only':True})
                admitted=operate(store,'admit-research',{'bundle_path':str(bundle)})
                self.assertTrue(admitted['jq_candidate_allowed'])
                self.assertEqual(admitted['investment_certification']['status'],'pending')
                result['evaluations'][1]['diagnostics']['tail_resolution_adequate']=False
                rejected=AcceptanceService(store).research(store.append('evaluation',result))
                self.assertFalse(rejected['jq_candidate_allowed'])
                self.assertIn('bootstrap_tail_resolution_inadequate',rejected['blockers'])
                store.revoke(receipt,'fixture correction')
                with self.assertRaises(ContractError):store.consume_receipt(receipt,'data_validation')
            finally:store.close()

    def test_scheduled_llm_feedback_and_resume_no_double_charge(self):
        with scratch_directory() as directory:
            root=Path(directory);p=synthetic_campaign(130,6);source=root/'input.json';source.write_bytes(canonical(p))
            protocol={'schema_version':'fuxi.controlled_study.1','arm_schedule':['random','llm','refine','llm'],
                      'max_trials':8,'checkpoint_every':2,'memory_sources':[]}
            kwargs=dict(rounds=4,candidates_per_round=1,model_algorithms=(),llm_enabled=True,
                        campaign_id='campaign_learning',study_protocol=protocol)
            def llm(fields,cards,**kw):
                self.assertTrue(cards)
                self.assertIn('net_increment',cards[0]['text'])
                return generate(fields,arm='random',count=1,seed=999+len(cards)),{'attempted_calls':1,'status':'completed'}
            with patch('fuxi_v1.campaign.llm_candidates',side_effect=llm) as api:
                _,partial=run_campaign(source,root/'runs',pause_after_round=2,**kwargs)
                self.assertEqual(partial['status'],'checkpoint_paused')
                _,final=run_campaign(source,root/'runs',**kwargs)
                self.assertEqual(final['llm_calls'],2)
                _,again=run_campaign(source,root/'runs',**kwargs)
                self.assertEqual(api.call_count,2)
                self.assertEqual(final['attempted_candidates'],again['attempted_candidates'])

    def test_no_memory_never_sends_feedback(self):
        with scratch_directory() as directory:
            root=Path(directory);p=synthetic_campaign(130,6);source=root/'input.json';source.write_bytes(canonical(p))
            protocol={'schema_version':'fuxi.controlled_study.1','arm_schedule':['random','llm'],
                      'max_trials':4,'checkpoint_every':2,'memory_sources':[]}
            def llm(fields,cards,**kw):
                self.assertEqual(cards,[])
                return generate(fields,arm='random',count=1,seed=993),{'attempted_calls':1,'status':'completed'}
            with patch('fuxi_v1.campaign.llm_candidates',side_effect=llm):
                _,report=run_campaign(source,root/'runs',rounds=2,candidates_per_round=1,mode='no_memory',
                          llm_enabled=True,model_algorithms=(),study_protocol=protocol)
            self.assertEqual(report['llm_calls'],1)


if __name__=='__main__':unittest.main()
