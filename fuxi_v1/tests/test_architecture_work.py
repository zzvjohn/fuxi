from copy import deepcopy
from pathlib import Path
import json
import unittest
from unittest.mock import patch
from fuxi_v1.contracts import ContractError, digest, canonical
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.search import factor_spec, node, generate
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.scheduler import choose, counters
from fuxi_v1.edits import refine_local, apply_patch, parse_llm_patches
from fuxi_v1.dependency_audit import dependency_scope, pit_styles
from fuxi_v1.factor_cards import card, query
from fuxi_v1.incremental_study import common_support, fold_predictions, account_for
from fuxi_v1.account_reference import reconcile
from fuxi_v1.diagnostic_parity import compare
from fuxi_v1.announcement_feature import channel
from fuxi_v1.architecture_study import freeze, run
from fuxi_v1.tests.test_kernel import scratch_directory


class SchedulerWorkTests(unittest.TestCase):
    def test_empty_selections_increase_visits_and_all_arms_stop(self):
        history = []
        for i in range(15):
            arm, info = choose(i,'adaptive',history,False,context_hash='same')
            if arm is None:
                break
            history.append({'round':i,'arm':arm,'generated_count':0,'evaluated_count':0,
                            'valid_e1_count':0,'elapsed_seconds':.01,'generation_context_hash':'same'})
        self.assertIsNone(arm)
        self.assertEqual(len(history),8)
        self.assertEqual(sum(counters(history,a)['selection_count'] for a in info['counters']),8)
        self.assertTrue(all(counters(history,a)['evaluated_count']==0 for a in info['counters']))
        revived,_ = choose(10,'adaptive',history,False,context_hash='new_parent')
        self.assertIsNotNone(revived)

    def test_resume_and_provider_failure(self):
        row = {'round':0,'arm':'llm','generated_count':0,'evaluated_count':0,'valid_e1_count':0,
               'elapsed_seconds':40,'generator_status':'error','generation_context_hash':'a'}
        a,aa = choose(6,'adaptive',[row],True,context_hash='a')
        b,bb = choose(6,'adaptive',json.loads(json.dumps([row])),True,context_hash='a')
        self.assertEqual((a,aa),(b,bb))
        self.assertEqual(aa['inactive']['llm'],'provider_failure_no_automatic_retry')


class EditsWorkTests(unittest.TestCase):
    def setUp(self):
        self.fields = {'close':{'unit':'price'},'open':{'unit':'price'},'volume':{'unit':'shares'}}
        self.parent = factor_spec(node('rank',node('return',{'field':'close'},window=20)),self.fields,'persistence')

    def test_deterministic_nearest_neighbors_without_fallback(self):
        a,r = refine_local(self.fields,[self.parent],10)
        b,_ = refine_local(self.fields,[self.parent],10)
        self.assertEqual(a,b)
        self.assertEqual([c['canonical_ast']['args'][0]['window'] for c in a],[15,10])
        self.assertEqual(refine_local(self.fields,[],2)[0],[])
        self.assertEqual(refine_local(self.fields,[self.parent],10,seen=[c['formula_hash'] for c in a])[0],[])
        self.assertEqual(r['attempts'],2)

    def test_patch_all_six_kinds_and_invalid_or_noop(self):
        edits = [('window',[0],15),('field',[0,0],'open'),('operator',[0],'delta'),('wrap',[],'neg'),
                 ('unwrap',[],None),('binary',[],{'op':'sub','field':'volume'})]
        for kind,path,value in edits:
            row = {'parent_id':self.parent['candidate_id'],'kind':kind,'path':path,'value':value}
            child = apply_patch(self.parent,row,self.fields)
            self.assertEqual(child['parent_ids'],[self.parent['candidate_id']])
            self.assertTrue(child['mutation_patch']['motifs'])
            self.assertEqual(apply_patch(self.parent,row,self.fields),child)
        for row in [{'parent_id':'other','kind':'wrap','path':[],'value':'neg'},
                    {'parent_id':self.parent['candidate_id'],'kind':'window','path':[0],'value':20},
                    {'parent_id':self.parent['candidate_id'],'kind':'wrap','path':[],'value':'eval'},
                    {'parent_id':self.parent['candidate_id'],'kind':'field','path':[0,9],'value':'open'}]:
            with self.assertRaises(ContractError):apply_patch(self.parent,row,self.fields)

    def test_llm_patch_response_contract_and_request_counter(self):
        parsed = {'patches':[{'parent_id':self.parent['candidate_id'],'kind':'window','path':[0],'value':15}]}
        proposals,rejected,n = parse_llm_patches(parsed,[self.parent],self.fields,[],1)
        self.assertEqual((len(proposals),rejected,n),(1,[],1))
        with self.assertRaises(ContractError):parse_llm_patches({'patches':[],'code':'bad'},[self.parent],self.fields,[],1)
        counts = {}
        generated = generate(self.fields,arm='fixed',count=2,seed=1,counters=counts)
        self.assertEqual(len(generated),2)
        self.assertGreater(counts['generation_attempts'],2)


class EvidenceWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = synthetic_campaign(145,8)

    def test_dependency_closure_keeps_departed_names_and_issued_rights(self):
        payload = deepcopy(self.payload)
        lost = payload['snapshot']['universe'][payload['snapshot']['sessions'][100]][0]
        for d in payload['snapshot']['sessions'][110:]:payload['snapshot']['universe'][d].remove(lost)
        payload['market']['actions'] = [{'payload':{'instrument':lost,'target_instrument':'NEW.XSHG'}}]
        snap = Snapshot(payload['snapshot'])
        c = factor_spec(node('rank',node('return',{'field':'close'},window=5)),snap.payload['fields'],'p')
        scope = dependency_scope(snap,[c],snap.sessions[100],snap.sessions[-1],payload['market'])
        self.assertEqual(scope['fields'],['close'])
        self.assertIn(lost,scope['holdings_and_rights_closure'])
        self.assertIn('NEW.XSHG',scope['holdings_and_rights_closure'])
        self.assertFalse(scope['investment_eligibility_certified'])
        self.assertGreater(scope['gaps']['potential_holding_mark_missing'],0)

    def test_PIT_styles_do_not_backfill_late_revision(self):
        snap = Snapshot(self.payload['snapshot'])
        code,first,late = snap.universe[snap.sessions[20]][0],snap.sessions[20],snap.sessions[60]
        rows = []
        for style,value in [('industry','A'),('float_market_cap',100.)]:
            rows.append({'style':style,'instrument':code,'effective_session':first,'value':value,
                         'published_at':first+'T18:00:00+08:00','available_at':first+'T18:00:00+08:00',
                         'ingested_at':first+'T18:00:00+08:00','evidence_id':style,'revision_id':'r1'})
        rows.append(dict(rows[-1],value=999.,available_at=late+'T18:00:00+08:00',ingested_at=late+'T18:00:00+08:00',revision_id='r2'))
        spec = {'schema_version':'fuxi.pit_styles.1','source_id':'fixture','primary_evidence_sha256':'fixture',
                'style_names':['industry','float_market_cap'],'rows':rows}
        panels,source = pit_styles(spec,snap)
        self.assertEqual(panels['float_market_cap'][snap.sessions[59]][code],100.)
        self.assertEqual(panels['float_market_cap'][late][code],999.)
        self.assertFalse(source['investment_neutralization_certified'])

    def test_cards_index_readonly_and_detect_tampering(self):
        c = factor_spec(node('rank',{'field':'close'}),self.payload['snapshot']['fields'],'p')
        from fuxi_v1.contracts import evaluation
        e = evaluation('E1','data_insufficient','missing')
        value = card(c,e,{'window':{'start':'2023-01-03','end':'2023-08-01'},'exposure':'synthetic'},{'max_trials':3})
        with scratch_directory() as folder:
            p = Path(folder)/'cards'
            p.mkdir()
            f = p/'a.json'
            f.write_bytes(canonical(value))
            before = f.stat().st_mtime_ns
            self.assertEqual(len(query(folder,status='data_insufficient')),1)
            self.assertEqual(f.stat().st_mtime_ns,before)
            self.assertFalse(value['jq_candidate_allowed'])
            changed = dict(value,live_execution_allowed=True)
            f.write_bytes(canonical(changed))
            with self.assertRaises(ContractError):query(folder)

    def test_announcement_missing_zero_and_postclose_lag_are_distinct(self):
        snap = Snapshot(self.payload['snapshot'])
        day = snap.sessions[60]
        code = snap.universe[day][0]
        ts_code = code.replace('.XSHG','.SH')
        with scratch_directory() as folder:
            p = Path(folder)/'source.csv'
            p.write_text('ts_code,div_proc,imp_ann_date,cash_div_tax\n'+ts_code+',实施,'+day.replace('-','')+',0\n',encoding='utf8')
            panel,manifest = channel(p,snap)
            self.assertIsNone(panel[day][code])
            self.assertEqual(panel[snap.sessions[61]][code],0.)
            self.assertIsNone(panel[snap.sessions[81]][code])
            self.assertTrue(all(panel[snap.sessions[61]][s] is None for s in snap.universe[day] if s!=code))
            self.assertFalse(manifest['formal_use_allowed'])
            with p.open('a',encoding='utf8') as f:f.write(ts_code+',实施,'+day.replace('-','')+',1\n')
            conflicted,_ = channel(p,snap)
            self.assertIsNone(conflicted[snap.sessions[61]][code])

    def test_reference_cash_holdings_rights_and_unknown(self):
        p = deepcopy(self.payload)
        snapshot = Snapshot(p['snapshot'])
        scores = evaluate_ast(node('rank',{'field':'close'}),snapshot)
        request = {'research_start':snapshot.sessions[100],'research_end':snapshot.sessions[-1]}
        _,account = account_for(scores,snapshot,p['market'],request,identity='probe')
        result = reconcile(account,p['market'])
        self.assertEqual(result['eval_status'],'valid')
        self.assertGreater(result['n_pairs'],0)
        self.assertFalse(result['formal_certification'])
        changed = deepcopy(account)
        changed['daily'][-1]['cash'] = '1'
        self.assertEqual(reconcile(changed,p['market'])['eval_status'],'error')
        code = snapshot.universe[snapshot.sessions[125]][0]
        p['market']['actions'].append({'event_id':'unknown','kind':'unknown','timestamp':snapshot.sessions[125]+'T09:00:00+08:00',
                                     'evidence_id':'unknown','payload':{'instrument':code,'reason':'unknown'}})
        p['market']['market_hash'] = digest({k:v for k,v in p['market'].items() if k!='market_hash'})
        _,account = account_for(scores,snapshot,p['market'],request,identity='unknown_probe')
        result = reconcile(account,p['market'])
        self.assertEqual(result['eval_status'],'valid')
        self.assertGreater(result['unidentifiable_rows'],0)


class IncrementAndPlatformWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = synthetic_campaign(145,8)

    def test_all_model_methods_have_same_support_and_future_labels_cannot_select(self):
        p = self.payload
        snap = Snapshot(p['snapshot'])
        labels = Labels(p['labels'],snap)
        base = evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),snap)
        candidates = {'a':evaluate_ast(node('rank',{'field':'volume'}),snap),
                      'b':evaluate_ast(node('rank',node('return',{'field':'close'},window=5)),snap)}
        candidates['a'][snap.sessions[101]][snap.universe[snap.sessions[101]][0]] = None
        request = {'research_start':snap.sessions[100],'research_end':snap.sessions[-1]}
        a = fold_predictions(base,candidates,snap,labels,request,method='screened')
        b = fold_predictions(base,candidates,snap,labels,request,method='ordinary')
        self.assertEqual(a['support_hash'],b['support_hash'])
        changed = deepcopy(p['labels'])
        for row in changed['rows']:
            if row['mature_at'] > snap.clocks[snap.sessions[100]].isoformat() and row['value'] is not None:row['value'] *= -100
        later = fold_predictions(base,candidates,snap,Labels(changed,snap),request,method='screened')
        self.assertEqual(a['folds'][0]['selected_feature'],later['folds'][0]['selected_feature'])
        self.assertEqual({d:a['scores'][d] for d in snap.sessions[100:140]}, {d:later['scores'][d] for d in snap.sessions[100:140]})
        self.assertTrue(all(r['changed_design_axis']=='one_added_feature' for r in a['folds']))

    def test_parity_zero_pairs_null_time_and_real_export_identity(self):
        e = {'protocol_hash':'p','rows':[{'signal_session':'2026-01-02','instrument':'600000.XSHG',
             'factor_value':1.,'target_weight':.1,'decision_time':'18:30','execution_session':'2026-01-05'}]}
        o = {'schema_version':'fuxi.jq_observed_diagnostic.1','observed_on_jq':True,'raw_export_sha256':'hash',**deepcopy(e)}
        self.assertTrue(compare(e,o)['diagnostic_match'])
        o['rows'][0]['factor_value'] = None
        self.assertFalse(compare(e,o)['diagnostic_match'])
        o['rows'][0]['target_weight'] = None
        with self.assertRaisesRegex(ContractError,'zero numeric'):compare(e,o)
        o['observed_on_jq'] = False
        with self.assertRaisesRegex(ContractError,'real platform'):compare(e,o)

    def test_frozen_protocol_refuses_input_or_code_change(self):
        with scratch_directory() as folder:
            src, out = Path(folder)/'input.json',Path(folder)/'study'
            src.write_bytes(canonical(self.payload))
            scope = freeze(src,out)
            with patch('fuxi_v1.architecture_study.code_manifest',return_value={'changed':'hash'}):
                with self.assertRaisesRegex(ContractError,'code changed'):run(out)
            src.write_bytes(canonical(dict(self.payload,research_clock='changed')))
            with self.assertRaisesRegex(ContractError,'input changed'):run(out)
            self.assertFalse(scope['live_execution_allowed'])


if __name__=='__main__':unittest.main()
