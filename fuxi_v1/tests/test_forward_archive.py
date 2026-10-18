"""Clock-patched synthetic contracts are software tests, never real future evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import unittest
from unittest.mock import patch
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.contracts import ContractError, canonical, digest, load_json
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.search import node
from fuxi_v1.portfolio import strategy_spec
from fuxi_v1.models import fit_model, predict_frozen_model, capabilities, walk_forward
from fuxi_v1.online import daily_decision, daily_model_decision
from fuxi_v1.forward_archive import freeze_archive, capture_day, record_account_pair, build_observation, status
from fuxi_v1.lifecycle_status import build_requests, evidence_state
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.acceptance import AcceptanceService
from fuxi_v1.policy import register_confirmatory_protocol
from fuxi_v1.operations import operate


def saved(path, value):
    path.write_bytes(canonical(value)+b'\n')
    return str(path),hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenModelTests(unittest.TestCase):
    def test_all_available_serialized_backends_reproduce_nonempty_predictions(self):
        import numpy as np
        x=[[i/10,(i%7)/3] for i in range(80)];y=[.02*a-.04*b+(i%3)*.001 for i,(a,b) in enumerate(x)]
        for name,supported in capabilities().items():
            if supported!='supported':continue
            with self.subTest(backend=name):
                state,predict=fit_model(name,x,y,seed=23)
                actual=predict_frozen_model(state,x[3:29])
                self.assertEqual(len(actual),26)
                np.testing.assert_allclose(actual,predict(x[3:29]),rtol=1e-6,atol=1e-7)

    def test_corrupted_empty_and_nonfinite_model_input_rejected(self):
        state,_=fit_model('ridge',[[i,i%3] for i in range(20)],[i/100 for i in range(20)])
        bad=deepcopy(state);bad['coef'][0]+=1
        with self.assertRaises(ContractError):predict_frozen_model(bad,[[1,2]])
        with self.assertRaises(ContractError):predict_frozen_model(state,[])
        with self.assertRaises(ContractError):predict_frozen_model(state,[[float('nan'),2]])

    def test_row_snapshot_daily_decision_is_causal_and_model_state_is_consumed(self):
        p=synthetic_campaign(40,6)['snapshot'];day=p['sessions'][25];spec=strategy_spec('daily')
        ast=node('rank',{'field':'close'})
        first=daily_decision(ast,Snapshot(p),day,p['decision_times'][day],spec)
        q=deepcopy(p)
        for row in q['observations']:
            if row['session']>day:row['value']*=100
        after=daily_decision(ast,Snapshot(q),day,p['decision_times'][day],spec)
        self.assertEqual(first['decision_id'],after['decision_id'])
        state,predict=fit_model('ridge',[[i] for i in range(20)],[i*.1 for i in range(20)])
        result=daily_model_decision({'close':{'field':'close'}},state,Snapshot(p),day,p['decision_times'][day],spec)
        self.assertEqual(len(result['scores']),6)
        values=[next(r['value'] for r in p['observations'] if r['session']==day and r['instrument']==c
                     and r['field']=='close') for c in p['universe'][day]]
        import numpy as np
        np.testing.assert_allclose(list(result['scores'].values()),predict([[v] for v in values]))

    def test_native_walk_forward_state_feature_order_and_predictions_reproduce(self):
        p=synthetic_campaign(80,6);snapshot=Snapshot(p['snapshot']);labels=Labels(p['labels'],snapshot)
        ast={'field':'close'};features={'close':evaluate_ast(ast,snapshot)}
        fitted=walk_forward(features,snapshot,labels,predict_start=snapshot.sessions[45],min_train_sessions=20)
        self.assertGreater(fitted['trained_model_count'],0)
        state=fitted['models'][-1];day=snapshot.sessions[-1]
        result=daily_model_decision({'close':ast},state,snapshot,day,snapshot.payload['decision_times'][day],strategy_spec('frozen'))
        import numpy as np
        np.testing.assert_allclose(list(result['scores'].values()),list(fitted['scores'][day].values()))
        with self.assertRaises(ContractError):daily_model_decision({'renamed':ast},state,snapshot,day,
            snapshot.payload['decision_times'][day],strategy_spec('frozen'))
        bad=deepcopy(state);bad['training_cutoff']='2099-01-01T00:00:00+00:00'
        with self.assertRaises(ContractError):predict_frozen_model(bad,[[1]])

    def test_nonselected_candidate_missingness_is_in_frozen_common_support(self):
        p=synthetic_campaign(40,6)['snapshot'];day=p['sessions'][-1];code=p['universe'][day][0]
        for row in p['observations']:
            if row['session']==day and row['instrument']==code and row['field']=='volume':row['value']=None
        state,_=fit_model('ridge',[[i] for i in range(20)],[i*.1 for i in range(20)])
        spec=strategy_spec('support');snapshot=Snapshot(p)
        full=daily_model_decision({'close':{'field':'close'}},state,snapshot,day,p['decision_times'][day],spec)
        masked=daily_model_decision({'close':{'field':'close'}},state,snapshot,day,p['decision_times'][day],spec,
            support_asts=[{'field':'close'},{'field':'volume'}])
        self.assertIsNotNone(full['scores'][code]);self.assertIsNone(masked['scores'][code])
        self.assertNotEqual(full['decision_id'],masked['decision_id'])
        self.assertTrue(masked['support_hash'])
        self.assertEqual(sum(v is not None for v in masked['scores'].values()),5)


class ForwardArchiveTests(unittest.TestCase):
    def setup_archive(self, directory, *, horizon=2):
        directory=Path(directory);p=synthetic_campaign(40,6);days=p['snapshot']['sessions']
        calendar_path,sha=saved(directory/'calendar.json',{'sessions':days})
        strategy=strategy_spec('future',top_n=2,max_weight=.6,rebalance_sessions=1)
        spec={'schema_version':'fuxi.forward_archive_request.1','mode':'research_observation',
            'candidate_id':'fixture_candidate','ir_hash':'fixture_parent_ir','baseline_hash':'fixture_baseline',
            'cost_profile_hash':digest(p['market']['execution']),
            'candidate':{'ast':node('rank',{'field':'close'}),'strategy':strategy},
            'baseline':{'ast':node('neg',node('rank',{'field':'close'})),
                        'strategy':strategy_spec('baseline',top_n=2,max_weight=.6,rebalance_sessions=1)},
            'calendar':{'sessions':days[20:],'decision_times':{d:p['snapshot']['decision_times'][d] for d in days[20:]},
                        'evidence_path':calendar_path,'evidence_sha256':sha},
            'look_sessions':[4,8],'horizon_sessions':horizon,'minimum_sections':4}
        freeze_clock=days[19]+'T19:00:00+08:00';root=directory/'archive'
        with patch('fuxi_v1.forward_archive.clock',return_value=freeze_clock):frozen=freeze_archive(spec,root)
        return p,spec,root,frozen

    def through(self, p, day):
        q=deepcopy(p);s=q['snapshot'];s['sessions']=[d for d in s['sessions'] if d<=day]
        s['decision_times']={d:t for d,t in s['decision_times'].items() if d<=day}
        s['universe']={d:v for d,v in s['universe'].items() if d<=day}
        s['observations']=[r for r in s['observations'] if r['session']<=day]
        q['market']['sessions']=[d for d in q['market']['sessions'] if d<=day]
        q['market']['bars']=[r for r in q['market']['bars'] if r['session']<=day]
        return q

    def capture(self, directory,p,root,day):
        path,sha=saved(Path(directory)/('source_'+day+'.json'),self.through(p,day))
        request={'session':day,'snapshot_path':path,'snapshot_sha256':sha}
        with patch('fuxi_v1.forward_archive.clock',return_value=day+'T19:00:00+08:00'):
            return capture_day(root,request),request

    def test_real_runtime_freeze_status_and_not_due_do_not_mint_acceptance(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d)
            with patch('fuxi_v1.forward_archive.clock',return_value=frozen['frozen_at']):
                result=status(root);pending=build_observation(root,{'look_sessions':4})
                again=freeze_archive(spec,root)
            self.assertTrue(again['idempotent']);self.assertEqual(result['captured_sections'],0)
            self.assertEqual(pending['status'],'not_due');self.assertFalse(pending['future_acceptance_granted'])

    def test_calendar_source_membership_and_formal_receipt_are_enforced(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d)
            q=deepcopy(spec);q['calendar']['sessions'].append('2099-01-01')
            q['calendar']['decision_times']['2099-01-01']='2099-01-01T18:30:00+08:00'
            with self.assertRaises(ContractError):freeze_archive(q,Path(d)/'bad')
            q=deepcopy(spec);q['mode']='formal_acceptance'
            with self.assertRaises(ContractError):freeze_archive(q,Path(d)/'formal')
            q=deepcopy(spec);q['frozen_at']='1999-01-01T00:00:00+00:00'
            with self.assertRaises(ContractError):freeze_archive(q,Path(d)/'forged')

    def test_actual_project_csv_calendar_with_BOM_is_consumed_by_freeze(self):
        import csv
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d)
            source=Path('data/raw/trade_calendar.csv').read_bytes()
            text=source.decode('utf-8-sig');rows=list(csv.DictReader(text.splitlines()))
            key=next(k for k in ('cal_date','trade_date','date','session') if k in rows[0])
            days=sorted({str(r[key])[:4]+'-'+str(r[key])[4:6]+'-'+str(r[key])[6:8]
                if len(str(r[key]))==8 else str(r[key]) for r in rows if str(r.get('is_open','1'))=='1'})
            future=[day for day in days if day>'2026-10-09'][:20]
            self.assertGreaterEqual(len(future),10)
            calendar=Path(d)/'project_calendar_with_bom.csv';calendar.write_bytes(b'\xef\xbb\xbf'+text.encode('utf8'))
            spec['calendar']={'sessions':future,'decision_times':{day:day+'T18:30:00+08:00' for day in future},
                'evidence_path':str(calendar),'evidence_sha256':hashlib.sha256(calendar.read_bytes()).hexdigest()}
            with patch('fuxi_v1.forward_archive.clock',return_value='2026-10-09T00:00:00+00:00'):
                actual=freeze_archive(spec,Path(d)/'csv_archive')
            self.assertEqual(actual['future_sessions'],future)
            self.assertEqual(actual['looks'][0]['status'],'calendar_resolved')

    def test_nonzero_daily_capture_is_immutable_and_retry_is_idempotent(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d);day=p['snapshot']['sessions'][20]
            row,request=self.capture(d,p,root,day)
            self.assertEqual(len(row['decisions']['candidate']['scores']),6)
            with patch('fuxi_v1.forward_archive.clock',return_value=day+'T20:00:00+08:00'):
                self.assertTrue(capture_day(root,request)['idempotent'])
                q=deepcopy(request);q['candidate_prior']={'600000.XSHG':.1}
                with self.assertRaises(ContractError):capture_day(root,q)
            q=deepcopy(request);q['observation_clock']='2099-01-01T00:00:00+00:00'
            with self.assertRaises(ContractError):capture_day(root,q)

    def test_future_decision_and_future_snapshot_are_rejected(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d);day=p['snapshot']['sessions'][20]
            path,sha=saved(Path(d)/'whole.json',p)
            req={'session':day,'snapshot_path':path,'snapshot_sha256':sha}
            with patch('fuxi_v1.forward_archive.clock',return_value=frozen['frozen_at']):
                with self.assertRaises(ContractError):capture_day(root,req)
            with patch('fuxi_v1.forward_archive.clock',return_value=day+'T19:00:00+08:00'):
                with self.assertRaises(ContractError):capture_day(root,req)

    def test_stale_new_decision_rejected_but_existing_capture_can_resume_later(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d);days=p['snapshot']['sessions'][20:22]
            row,request=self.capture(d,p,root,days[0])
            path,sha=saved(Path(d)/'stale.json',self.through(p,days[1]))
            with patch('fuxi_v1.forward_archive.clock',return_value=p['snapshot']['sessions'][25]+'T19:00:00+08:00'):
                self.assertTrue(capture_day(root,request)['idempotent'])
                with self.assertRaises(ContractError):capture_day(root,{'session':days[1],
                    'snapshot_path':path,'snapshot_sha256':sha})
    def test_paired_replay_uses_archived_scores_and_reference_and_builds_mature_rows(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d)
            days=p['snapshot']['sessions'][20:27]
            for day in days:self.capture(d,p,root,day)
            path,sha=saved(Path(d)/'pair.json',self.through(p,days[-1]))
            req={'input_path':path,'input_sha256':sha,'through_session':days[-1]}
            with patch('fuxi_v1.forward_archive.clock',return_value=days[-1]+'T19:00:00+08:00'):
                pair=record_account_pair(root,req);again=record_account_pair(root,req)
                observation=build_observation(root,{'look_sessions':4})
                repeated=build_observation(root,{'look_sessions':4})
            self.assertTrue(again['idempotent']);self.assertTrue(pair['account_identifiable'])
            for side in ('candidate','baseline'):
                self.assertGreater(pair['reference'][side]['n_pairs'],0)
                self.assertEqual(pair['reference'][side]['mismatches'],[])
            self.assertEqual(observation['n_pairs'],4)
            self.assertTrue(all(r['paired_net_increment'] is not None for r in observation['observation']['rows']))
            self.assertFalse(observation['future_acceptance_granted'])
            self.assertTrue(repeated['idempotent'])

    def test_missing_capture_and_cost_mismatch_cannot_be_hidden(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d);days=p['snapshot']['sessions'][20:23]
            self.capture(d,p,root,days[0]);self.capture(d,p,root,days[2])
            q=self.through(p,days[-1]);path,sha=saved(Path(d)/'pair.json',q)
            req={'input_path':path,'input_sha256':sha,'through_session':days[-1]}
            with patch('fuxi_v1.forward_archive.clock',return_value=days[-1]+'T19:00:00+08:00'):
                with self.assertRaises(ContractError):record_account_pair(root,req)
            q['market']['execution']['slippage_bps']=99
            path,sha=saved(Path(d)/'cost.json',q);req.update(input_path=path,input_sha256=sha)
            with self.assertRaises(ContractError):record_account_pair(root,req)

    def test_published_capture_tampering_rejected(self):
        with scratch_directory() as d:
            p,spec,root,frozen=self.setup_archive(d);day=p['snapshot']['sessions'][20]
            row,_=self.capture(d,p,root,day)
            (Path(row['bundle_path'])/'SNAPSHOT_SOURCE.json').write_bytes(b'{}')
            with self.assertRaises(ContractError):status(root)


class LifecycleRequestTests(unittest.TestCase):
    def test_requests_ready_waiting_and_retry_keep_templates_from_becoming_evidence(self):
        with scratch_directory() as d:
            spec={'schema_version':'fuxi.lifecycle_requests.1','software_checks':{'contracts':True},
                  'candidate_id':'still_unqualified'}
            result=build_requests(spec,d);again=build_requests(spec,d)
            self.assertTrue(again['idempotent']);self.assertFalse(result['qualification_granted'])
            state=load_json(Path(d)/'LIFECYCLE_STATUS.json')
            self.assertEqual(state['status'],'ready_awaiting_evidence')
            self.assertEqual(state['next_action'],'admit-research')
            self.assertFalse(load_json(Path(d)/'REGISTER_PROTOCOL_TEMPLATE.json')['calibration']['verified'])

    def test_receipt_authority_is_required_for_active_state(self):
        state=evidence_state({'research_receipt':'invented'},software_checks={'contracts':True})
        self.assertEqual(state['receipts']['research_receipt']['state'],'authority_not_opened')
        self.assertFalse(state['JQ_backtest_observed'])

    def test_interrupted_request_package_can_resume_without_overwriting(self):
        with scratch_directory() as d:
            spec={'schema_version':'fuxi.lifecycle_requests.1','software_checks':{'contracts':True}}
            root=Path(d)
            state=evidence_state({},software_checks=spec['software_checks'])
            saved(root/'LIFECYCLE_STATUS.json',state)
            result=build_requests(spec,root)
            self.assertTrue((root/'REQUEST_INDEX.json').is_file())
            self.assertFalse(result['qualification_granted'])


class ForwardAcceptancePreflightTests(unittest.TestCase):
    def test_foreign_scope_does_not_spend_look_and_valid_retry_can_be_evaluated(self):
        with scratch_directory() as d:
            store=EvidenceStore(Path(d)/'authority.sqlite')
            try:
                research=store.receipt('research_acceptance',{'candidate_id':'test','ir_hash':'ir','policy_hash':'p'})
                parity=store.receipt('jq_parity',{'ir_hash':'ir','accepted':True,'n_pairs':2,'observed_on_jq':True})
                scope={'sources':['synthetic_fixture'],'instruments':['*'],'resolved':True,
                    'intervals':[['2030-01-01T00:00:00+00:00','2030-01-31T23:59:59+00:00']], 'label_intervals':[]}
                calibration=store.append('calibration',{'kind':'confirmatory_calibration','verified':True,'policy_hash':'p'})
                design={'schema_version':'fuxi.confirmatory_protocol.1','policy_hash':'p','candidate_ids':['test'],
                    'baseline_hash':'baseline','scope':scope,'frozen_at':'2029-12-01T00:00:00+00:00',
                    'minimum_sections':4,'alpha':.05,'block_length':2,'bootstrap_replicates':399,
                    'look_schedule':['2030-03-01T00:00:00+00:00'],'cost_profile_hash':'cost',
                    'minimum_net_increment_daily':0,'calibration_evidence_id':calibration}
                protocol=register_confirmatory_protocol(store,design);service=AcceptanceService(store)
                pending=service.freeze_forward(research,protocol,parity)
                rows=[{'decision_time':'2030-01-'+str(i+1).zfill(2)+'T00:00:00+00:00',
                    'mature_at':'2030-01-31T00:00:00+00:00','paired_net_increment':.01+i*.00001} for i in range(8)]
                observed={'ir_hash':'ir','baseline_hash':'baseline','cost_profile_hash':'cost','scope':scope,
                    'rows':rows,'mature_at':'2030-01-31T00:00:00+00:00','account_identifiable':True}
                bad=deepcopy(observed);bad['rows'][0]['decision_time']='2030-02-01T00:00:00+00:00'
                bad['rows'][0]['mature_at']='2030-02-02T00:00:00+00:00'
                identity=store.append('synthetic_observation',bad)
                with self.assertRaises(ContractError):service.forward(pending,identity,design['look_schedule'][0])
                self.assertEqual(len(list(store.facts('acceptance_look'))),0)
                good=store.append('synthetic_observation',observed)
                result=service.forward(pending,good,design['look_schedule'][0])
                self.assertEqual(result['n_pairs'],8)
                self.assertEqual(len(list(store.facts('acceptance_look'))),1)
            finally:store.close()

    def test_real_operator_rejects_future_clock_before_any_observation_fact(self):
        with scratch_directory() as d:
            store=EvidenceStore(Path(d)/'authority.sqlite')
            try:
                with self.assertRaises(ContractError):operate(store,'observe-forward',
                    {'observation_clock':'2099-01-01T00:00:00+00:00'})
                self.assertEqual(len(list(store.facts('forward_observation'))),0)
            finally:store.close()


if __name__=='__main__':unittest.main()
