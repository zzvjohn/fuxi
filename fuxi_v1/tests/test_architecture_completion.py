from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch
from fuxi_v1.contracts import digest, ContractError, load_json
from fuxi_v1.campaign import write_new
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.search import factor_spec,node
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.architecture_completion import paired_edit_trial, ranking_trial, source_ablation, file_hash
from fuxi_v1.operational_monitor import forward_plan, monitor
from fuxi_v1.manual_handoff import ingest, prepare
from fuxi_v1.evidence import EvidenceStore


class PairedResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=synthetic_campaign(150,8)
        cls.snapshot=Snapshot(cls.payload['snapshot'])
        cls.labels=Labels(cls.payload['labels'],cls.snapshot)
        days=cls.snapshot.sessions
        cls.base=factor_spec(node('rank',node('return',{'field':'close'},window=20)),cls.snapshot.payload['fields'],'persistence')
        cls.scope={'baseline':cls.base,'request':{'research_start':days[80],'research_end':days[-1]},
                   'memory_train':[days[80],days[114]],'memory_test':[days[115],days[-1]],
                   'ranking_windows':[[days[80],days[101]],[days[102],days[123]],[days[124],days[-1]]],
                   'top_k':1,'protocol_hash':'scope'}

    def test_memory_time_split_and_no_duplicate_formula_learning(self):
        result=paired_edit_trial(self.scope,self.snapshot,self.labels)
        self.assertEqual(result['eval_status'],'valid')
        self.assertEqual(len({r['child']['formula_hash'] for r in result['rows']}),len(result['rows']))
        self.assertLessEqual(result['training_direction']['latest_mature_at'],self.snapshot.clocks[self.scope['request']['research_start']].isoformat())
        self.assertFalse(result['economic_claim_allowed'])
        self.assertGreater(result['n_pairs'],0)
        self.assertEqual(result['LLM_requests'],0)

    def test_ranking_uses_intersection_and_keeps_missing_candidates(self):
        other=factor_spec(node('rank',node('return',{'field':'close'},window=5)),self.snapshot.payload['fields'],'persistence')
        result=ranking_trial(self.scope,self.snapshot,self.labels,[self.base,other])
        self.assertEqual(result['n_candidates'],2)
        self.assertEqual(len(result['redundancy']),1)
        self.assertTrue(all(set(r['oriented_rank_ic'])=={self.base['candidate_id'],other['candidate_id']} for r in result['windows']))
        self.assertGreater(sum(r['n_pairs'] for r in result['windows']),0)
        self.assertFalse(result['formal_selection_enabled'])

    def test_sparse_source_not_zero_filled_or_formally_admitted(self):
        panel=evaluate_ast(self.base['canonical_ast'],self.snapshot)
        sparse={d:{s:v if i%2==0 else None for i,(s,v) in enumerate(cross.items())} for d,cross in panel.items()}
        result=source_ablation(self.scope,self.snapshot,self.labels,{'panel':sparse,'stop_reason':'clock_unknown'})
        self.assertFalse(result['formal_use_allowed'])
        self.assertTrue(result['single_channel_stripping_executed'])
        self.assertEqual(result['eval_status'],'data_insufficient') # four codes, minimum five
        singleton={d:{s:v if i==0 else None for i,(s,v) in enumerate(cross.items())} for d,cross in panel.items()}
        self.assertEqual(source_ablation(self.scope,self.snapshot,self.labels,{'panel':singleton})['n_pairs'],0)

    def test_completion_freeze_resume_and_changed_inputs_are_rejected(self):
        from fuxi_v1.architecture_study import freeze as original_freeze, run as original_run
        from fuxi_v1.architecture_completion import freeze,run
        with scratch_directory() as directory:
            root=Path(directory);input_path=root/'input.json'
            write_new(input_path,synthetic_campaign(145,8))
            original_freeze(input_path,root/'old')
            original_run(root/'old')
            freeze(root/'old',root/'new')
            first=run(root/'new');second=run(root/'new')
            self.assertEqual(first,second)
            self.assertFalse(first['all_scientific_acceptance_complete'])
            input_path.write_bytes(input_path.read_bytes()+b' ')
            with self.assertRaises(ContractError):run(root/'new')


class MonitoringTests(unittest.TestCase):
    def spec(self):
        return {'schema_version':'fuxi.operational_observation.1','observation_clock':'2030-01-03T18:00:00+08:00',
                'expected_dependencies':{'code':'a','data':'b'},'actual_dependencies':{'code':'a','data':'b'},
                'data_channels':[{'name':'price','latest_available_at':'2030-01-03T17:00:00+08:00','expected_latest_at':'2030-01-03T17:00:00+08:00'}],
                'signals':[{'instrument':'000001.XSHE','decision_time':'2030-01-02T18:30:00+08:00',
                            'available_at':'2030-01-02T18:00:00+08:00','execution_time':'2030-01-03T09:30:00+08:00'}],
                'orders':[],'reconciliation':{'unknown_orders':0,'account_matched':True,'cash':'100','nav':'100','receivables':'0','holdings':{}},
                'registered_looks':['2030-02-01T18:00:00+08:00']}

    def test_clock_dependency_unknown_order_and_account_each_stop(self):
        clean=self.spec()
        self.assertEqual(monitor(clean)['recommendation'],'continue_manual_observation')
        for change in [lambda s:s['actual_dependencies'].update(code='changed'),
                       lambda s:s['signals'][0].update(available_at='2030-01-03T18:00:00+08:00'),
                       lambda s:s['orders'].append({'status':'unknown','order_id':'x'}),
                       lambda s:s['reconciliation'].update(nav=None)]:
            candidate=deepcopy(clean);change(candidate)
            self.assertEqual(monitor(candidate)['recommendation'],'halt_new_submissions_and_reconcile')

    def test_monitor_does_not_run_statistics_between_looks(self):
        result=monitor(self.spec())
        self.assertFalse(result['statistical_evaluation_due'])
        self.assertFalse(result['E5_evaluation_performed'])
        self.assertFalse(result['automatic_resubmit'])

    def test_real_calendar_hash_and_unmature_looks_remain_pending(self):
        with scratch_directory() as directory:
            directory=Path(directory)
            path=directory/'calendar.json';write_new(path,{'sessions':['2030-01-03']})
            spec={'sessions':['2030-01-03'],'decision_times':{'2030-01-03':'2030-01-03T18:00:00+08:00'},
                  'frozen_at':'2030-01-02T18:00:00+08:00','calendar_evidence_path':str(path),
                  'calendar_evidence_sha256':file_hash(path)}
            result=forward_plan(spec)
            self.assertEqual(result['status'],'pending')
            self.assertEqual(result['looks'][0]['status'],'calendar_missing')
            self.assertFalse(result['activation_performed'])
            spec['calendar_evidence_sha256']='f'*64
            with self.assertRaises(ContractError):forward_plan(spec)

    def test_revoked_receipt_blocks_operational_continue(self):
        with scratch_directory() as directory:
            directory=Path(directory)
            store=EvidenceStore(directory/'authority.sqlite')
            try:
                rid=store.receipt('research_acceptance',{'ir_hash':'x'})
                store.revoke(rid,'changed source')
                spec=self.spec();spec['required_receipts']={rid:'research_acceptance'}
                self.assertEqual(monitor(spec,store)['invalid_receipts'],[rid])
            finally:store.close()


class ManualReturnTests(unittest.TestCase):
    def setup_return(self,directory,store):
        receipt=store.receipt('research_acceptance',{'candidate_id':'candidate','ir_hash':'ir','jq_candidate_allowed':True})
        handoff={'research_receipt':receipt,'ir_hash':'ir','candidate_id':'candidate','policy_hash':'policy',
                 'snapshot_hash':'snapshot','source_evaluation_hash':'evaluation'}
        observed={'ir_hash':'ir','candidate_id':'candidate','policy_hash':'policy','snapshot_hash':'snapshot',
                  'observed_on_jq':True,'jq_run_id':'real_test_fixture','summary':{'net_return':0.1}}
        hp,op=directory/'handoff.json',directory/'observed.json'
        write_new(hp,handoff);write_new(op,observed)
        return {'schema_version':'fuxi.manual_jq_return.1','reviewer':'test_operator','reviewed_at':'2030-01-03T18:00:00+08:00',
                'evidence_files':{str(hp):file_hash(hp),str(op):file_hash(op)},'handoff_path':str(hp),'observed_path':str(op)},observed,op

    def test_summary_records_feedback_without_parity_or_qualification(self):
        with scratch_directory() as directory:
            directory=Path(directory)
            store=EvidenceStore(directory/'authority.sqlite')
            try:
                spec,_,_=self.setup_return(directory,store)
                result=ingest(store,spec)
                self.assertTrue(result['summary_recorded'])
                self.assertIsNone(result['receipt_id'])
                self.assertFalse(result['accepted'])
                self.assertEqual(result['n_pairs'],0)
            finally:store.close()

    def test_wrong_strategy_or_changed_raw_return_is_rejected(self):
        with scratch_directory() as directory:
            directory=Path(directory)
            store=EvidenceStore(directory/'authority.sqlite')
            try:
                spec,observed,path=self.setup_return(directory,store)
                observed['ir_hash']='other';path.write_text(__import__('json').dumps(observed))
                with self.assertRaises(ContractError):ingest(store,spec)
                spec['evidence_files'][str(path)]=file_hash(path)
                with self.assertRaises(ContractError):ingest(store,spec)
            finally:store.close()

    def test_first_JQ_needs_research_acceptance_and_no_E5(self):
        with scratch_directory() as directory:
            directory=Path(directory)
            store=EvidenceStore(directory/'authority.sqlite')
            try:
                with self.assertRaises(ContractError):prepare(store,{'bundle_path':str(directory),'output_root':str(directory/'out')})
                from fuxi_v1.artifacts import publish
                from fuxi_v1.incremental_study import account_for
                payload=synthetic_campaign(130,8);snap=Snapshot(payload['snapshot'])
                candidate=factor_spec(node('rank',node('return',{'field':'close'},window=20)),snap.payload['fields'],'persistence')
                scores=evaluate_ast(candidate['canonical_ast'],snap)
                ir,account=account_for(scores,snap,payload['market'],{'research_start':snap.sessions[80],'research_end':snap.sessions[-1]},identity='test')
                result={'strategy_ir':ir,'candidate_account':account}
                source=publish(directory/'source',{'evaluation.json':result},{'scope':'synthetic_test_only'})
                rid=store.receipt('research_acceptance',{'candidate_id':'test','ir_hash':ir['ir_hash'],
                    'jq_candidate_allowed':True,'policy_hash':'test','snapshot_hash':snap.hash})
                result=prepare(store,{'bundle_path':str(source),'output_root':str(directory/'out'),'research_receipt':rid})
                self.assertFalse(result['first_JQ_requires_E5'])
                self.assertTrue(result['backtest_orders_allowed'])
                self.assertTrue((Path(result['handoff_path'])/'RETURN_TEMPLATE.json').is_file())
            finally:store.close()

    def test_full_account_return_matches_and_unknown_NAV_cannot_pass(self):
        with scratch_directory() as directory:
            directory=Path(directory);store=EvidenceStore(directory/'authority.sqlite')
            try:
                spec,observed,op=self.setup_return(directory,store)
                row={'key':'2030-01-03','signal':{},'target_weights':{},'orders':[],
                     'holdings':{},'cash':'100','receivables':'0','nav':'100'}
                expected={k:v for k,v in observed.items() if k in ('candidate_id','ir_hash','policy_hash','snapshot_hash')}
                expected.update(source_evaluation_hash='evaluation',rows=[deepcopy(row)])
                observed['rows']=[deepcopy(row)]
                ep=directory/'expected.json'
                write_new(ep,expected)
                op.write_bytes(__import__('json').dumps(observed).encode())
                spec['expected_path']=str(ep)
                spec['evidence_files'].update({str(ep):file_hash(ep),str(op):file_hash(op)})
                self.assertTrue(ingest(store,spec)['accepted'])
                observed['rows'][0]['nav']=None
                op.write_bytes(__import__('json').dumps(observed).encode())
                spec['evidence_files'][str(op)]=file_hash(op)
                self.assertFalse(ingest(store,spec)['accepted'])
            finally:store.close()
