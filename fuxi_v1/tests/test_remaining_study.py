import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot, snapshot_from_payload, Labels
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.contracts import digest, ContractError,load_json
from fuxi_v1.campaign import write_new
from fuxi_v1.market_catalog import verify_frozen_sources
from fuxi_v1.datasets import file_hash
from fuxi_v1.remaining_study import freeze,run,proxy_styles,ledger_attribution,broad_comparison
from fuxi_v1.attribution import style_attribution
from fuxi_v1.search import node
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.incremental_study import compare_methods,account_for


def compact(payload):
    snap=Snapshot(payload['snapshot']);fields=snap.payload['fields'];codes=sorted(set.union(*[set(v) for v in snap.universe.values()]))
    p={'schema_version':'fuxi.historical_snapshot.1','exposure':'research_exposed','source_id':'test_compact',
       'sessions':snap.sessions,'evaluation_start':snap.sessions[snap.start],'universe':snap.universe,
       'instruments':codes,'decision_times':snap.payload['decision_times'],'fields':fields,
       'columns':{f:[[snap.value(i,s,f,snap.clocks[d]) for s in codes] for i,d in enumerate(snap.sessions)] for f in fields},
       'barriers':[[-1]*len(codes) for d in snap.sessions]}
    result=deepcopy(payload);result['snapshot']=p
    result['labels']['feature_snapshot_hash']=digest(p)
    return result


class RemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=compact(synthetic_campaign(140,8));cls.snapshot=snapshot_from_payload(cls.payload['snapshot'])
        d=cls.snapshot.sessions
        cls.request={'research_start':d[85],'research_end':d[-1],'alpha':.05,'max_trials':3,'block_length':6,
                     'bootstrap_replicates':99,'minimum_net_increment_daily':0.,'cost_stress_multiplier':2.,'capacity_stress_multiplier':.5}

    def test_schema_dispatch_keeps_action_barrier(self):
        p=deepcopy(self.payload['snapshot']);p['barriers'][60][0]=60;s=snapshot_from_payload(p)
        self.assertIsInstance(s,HistoricalSnapshot)
        self.assertIsNone(s.value(59,p['instruments'][0],'close',s.clocks[p['sessions'][60]]))
        self.assertIsNotNone(s.value(59,p['instruments'][0],'close',s.clocks[p['sessions'][59]]))
        self.assertIsInstance(snapshot_from_payload(synthetic_campaign(140,8)['snapshot']),Snapshot)

    def test_compact_matched_fold_study_keeps_data_gate(self):
        s=self.snapshot;base=evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),s)
        add=evaluate_ast(node('rank',node('return',{'field':'close'},window=5)),s)
        result=compare_methods(self.payload,base,{'return5':add},self.request)
        self.assertGreater(result['comparisons']['screened']['n_pairs'],0)
        self.assertFalse(result['comparisons']['screened']['economic_reward_eligible'])
        self.assertEqual(result['comparisons']['screened']['eval_status'],'data_insufficient')
        self.assertEqual(len({v['support_hash'] for v in result['outputs'].values()}),1)
        for fold in result['outputs']['screened']['folds']:
            for row in fold.get('provenance',[]):
                if row['status']=='valid':self.assertLessEqual(row['latest_training_label_mature_at'],row['training_cutoff'])

    def test_declared_fold_end_prevents_future_training_and_predictions(self):
        from fuxi_v1.incremental_study import fold_predictions
        s=self.snapshot;labels=Labels(self.payload['labels'],s)
        panel=evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),s)
        request=dict(self.request,research_end=s.sessions[109])
        result=fold_predictions(panel,{'same':panel},s,labels,request,method='screened')
        self.assertTrue(result['folds'])
        self.assertTrue(all(f['prediction_end']<=request['research_end'] for f in result['folds']))
        self.assertTrue(all(r['session']<=request['research_end'] for f in result['folds'] for r in f['provenance']))
        self.assertTrue(all(v is None for d,x in result['scores'].items() if d>request['research_end'] for v in x.values()))

    def test_frozen_catalog_survives_changed_origin_but_rejects_partition(self):
        with scratch_directory() as temp:
            root=Path(temp);origin=root/'origin.csv';origin.write_bytes(b'old');sealed=root/'part.bin';sealed.write_bytes(b'sealed')
            source={'path':str(origin),'sha256':file_hash(origin)}
            p={'tables':{'adj_factor':[{'file':'part.bin','source_hash':source['sha256'],'sha256':file_hash(sealed)}]},'sources':{'adj_factor':source}}
            p['manifest_hash']=digest(p);origin.write_bytes(b'new')
            result=verify_frozen_sources(root,p)
            self.assertFalse(result['origins'][0]['current_raw_matches'])
            self.assertFalse(result['origins'][0]['raw_file_consumed'])
            sealed.write_bytes(b'tampered')
            with self.assertRaises(ContractError):verify_frozen_sources(root,p)

    def test_style_proxy_keeps_overlaps_unknown_and_free_cap_semantics(self):
        p=deepcopy(self.payload);s=self.snapshot;codes=s.payload['instruments'];days=s.sessions
        p['snapshot']['fields']['free_shares']={'unit':'shares','frequency':'daily','adjustment':'raw'}
        p['snapshot']['columns']['free_shares']=[[10000. for c in codes] for d in days]
        queries=[]
        for code in codes:
            vendor=code[:6]+'.SH';rows=[{'ts_code':vendor,'l1_code':'801010.SI','in_date':days[0].replace('-',''),'out_date':None}]
            if code==codes[0]:rows.append(dict(rows[0],l1_code='801020.SI'))
            queries.append({'code':vendor,'rows':rows})
        panels,provenance=proxy_styles(p,{'queries':queries,'captured_at':'2030-01-01T00:00:00+08:00'})
        self.assertFalse(provenance['PIT_industry_certified'])
        self.assertFalse(provenance['historical_ingested_at_invented'])
        self.assertGreater(provenance['conflicting_industry_cells'],0)
        self.assertIsNone(panels['industry_known'][days[-1]][codes[0]])
        from math import log
        self.assertAlmostEqual(panels['log_free_float_market_cap'][days[-1]][codes[1]],log(p['snapshot']['columns']['close'][-1][1]*10000))

    def test_same_support_explained_residual_has_explicit_PIT_status(self):
        s=self.snapshot;labels=Labels(self.payload['labels'],s);scores=evaluate_ast(node('return',{'field':'close'},window=20),s)
        style=evaluate_ast(node('rank',{'field':'close'}),s)
        result=style_attribution(scores,{'price':style},labels,s,s.clocks[s.sessions[-1]].isoformat())
        valid=[r for r in result['sections'] if r['status']=='valid']
        self.assertTrue(valid);self.assertLess(max(r['decomposition_max_error'] for r in valid),1e-10)
        self.assertIn('PIT_market_cap',result['missing_styles'])

    def test_unknown_mark_not_zero_filled_in_ledger_or_broad(self):
        s=self.snapshot;scores=evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),s)
        _,a=account_for(scores,s,self.payload['market'],self.request,identity='test')
        bridge=ledger_attribution(a,self.payload['market'])
        self.assertGreater(bridge['identified_positions'],0)
        self.assertLess(max(abs(r['arithmetic_bridge_error']) for r in bridge['daily'] if r['arithmetic_bridge_error'] is not None),1e-8)
        unknown=deepcopy(a);unknown['daily'][5]['nav']=None
        observed=ledger_attribution(unknown,self.payload['market'])
        self.assertIsNone(observed['daily'][4]['nav_delta'])
        index={'rows':[{'trade_date':r['session'].replace('-',''),'close':100.} for r in a['daily']]}
        result=broad_comparison(unknown,a,index)
        self.assertEqual(result['n_pairs'],len(a['daily'])-3)
        self.assertFalse(result['causal_style_alpha_identified'])

    def test_unfilled_shares_are_not_labelled_as_money(self):
        account={'daily':[{'session':'2025-01-02','nav':'1000','holdings':{}},
                          {'session':'2025-01-03','nav':'1000','holdings':{}}],
                 'orders':[{'session':'2025-01-03','instrument':'000001.XSHE','requested':100,'filled':0,'reason':'limit_up'}]}
        market={'execution':{'initial_cash':'1000'},'bars':[{'session':'2025-01-03','instrument':'000001.XSHE','open':10.}]}
        result=ledger_attribution(account,market)['daily'][0]
        self.assertEqual(result['unfilled_share_quantities_by_instrument'],{'000001.XSHE':100})
        self.assertEqual(result['unfilled_notional_at_observed_open'],'1000.0')
        market['bars'][0]['open']=None
        self.assertIsNone(ledger_attribution(account,market)['daily'][0]['unfilled_notional_at_observed_open'])

    def test_immutable_stage_resume_and_input_change_refused(self):
        with scratch_directory() as temp:
            root=Path(temp);source=root/'input.json';write_new(source,self.payload)
            scope=freeze(source,root/'run',self.request)
            result=run(root/'run');self.assertEqual(run(root/'run'),result)
            self.assertEqual(result['formal_qualification_count'],0);self.assertEqual(result['future_observations'],0)
            source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaises(ContractError):run(root/'run')
