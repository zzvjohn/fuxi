"""New array path versus legacy scalar execution, with nonempty real comparisons."""
from copy import deepcopy
from datetime import date, timedelta
import json
import math
from pathlib import Path
import time
import unittest

import numpy as np

from fuxi_v1.columnar import (ColumnarSnapshot, ColumnarLabels, ColumnarMarket, ScoreGrid,
    array_reference, evaluate_columnar_ast, rank_vector, selected_by_dates,
    factor_statistics_columnar, paired_scores_columnar, paired_parent_columnar,
    mix_scores_columnar, common_support_grid, save_score_grid, load_score_grid,
    style_attribution_columnar, account_quotes, populate_research_context, window_reduction,walk_forward_columnar)
from fuxi_v1.contracts import ContractError, digest, timestamp
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.data import Labels
from fuxi_v1.dsl import evaluate_ast, ranks
from fuxi_v1.evaluation import DEFAULT_POLICY, factor_statistics
from fuxi_v1.research_evaluation import paired_scores
from fuxi_v1.learning import paired_parent
from fuxi_v1.attribution import style_attribution
from fuxi_v1.portfolio import mix_scores, compile_strategy, strategy_spec
from fuxi_v1.datasets import execution_profile
from fuxi_v1.simulation import run_account, ResearchAccountContext
from fuxi_v1.account import Account
from fuxi_v1.tests.test_kernel import scratch_directory


def fixture(directory):
    root=Path(directory);days=[];d=date(2024,1,2)
    while len(days)<34:
        if d.weekday()<5:days.append(d.isoformat())
        d+=timedelta(days=1)
    codes=['000001.XSHE','000002.XSHE','600001.XSHG','600002.XSHG','600003.XSHG','600004.XSHG']
    shape=(len(days),len(codes));membership=np.ones(shape,dtype=bool);membership[7:12,5]=False
    close=np.array([[10+j+math.sin(i*.2+j)*.3+i*.01*(j-2) for j in range(len(codes))] for i in range(len(days))])
    close[6,1]=np.nan;turnover=np.array([[.5+j*.1+math.cos(i*.2+j)*.02 for j in range(len(codes))] for i in range(len(days))])
    barriers=np.full(shape,-1,dtype=np.int32);barriers[13:,2]=13
    fields={'close':{'unit':'price','adjustment':'raw','frequency':'daily'},
            'turnover_rate':{'unit':'ratio','adjustment':'not_applicable','frequency':'daily'}}
    def ref(name,array):
        path=root/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        return array_reference(path)
    snapshot={'schema_version':'fuxi.columnar_snapshot.1','source_id':'synthetic_array_equivalence',
        'exposure':'research_exposed','sessions':days,'instruments':codes,'evaluation_start':days[5],
        'decision_times':{d:d+'T18:30:00+08:00' for d in days},'fields':fields,
        'availability_times':{f:[d+'T18:00:00+08:00' for d in days] for f in fields},
        'arrays':{'fields':{'close':ref('feature_close',close),'turnover_rate':ref('turnover',turnover)},
                  'membership':ref('members',membership),'barriers':ref('barriers',barriers)},
        'quality':{'pit_certified':False},'provenance':{'fixture':True}}
    native={'schema_version':'fuxi.historical_snapshot.1','source_id':snapshot['source_id'],'exposure':'research_exposed',
        'sessions':days,'instruments':codes,'evaluation_start':days[5],'decision_times':snapshot['decision_times'],
        'fields':fields,'universe':{d:[c for j,c in enumerate(codes) if membership[i,j]] for i,d in enumerate(days)},
        'columns':{f:[[float(v) if math.isfinite(v) else None for v in row] for row in value] for f,value in [('close',close),('turnover_rate',turnover)]},
        'barriers':barriers.tolist()}
    cs=ColumnarSnapshot(snapshot);ns=HistoricalSnapshot(native)
    horizon=2;values=np.full(shape,np.nan);status=np.zeros(shape,dtype=np.uint8);mature=[];rows=[]
    for i,day in enumerate(days):
        if i+1+horizon>=len(days):mature.append(None);continue
        end=days[i+1+horizon];mature.append(end+'T18:00:00+08:00')
        for j,code in enumerate(codes):
            if not membership[i,j]:continue
            value=close[i+1+horizon,j]/close[i+1,j]-1
            values[i,j]=value if math.isfinite(value) else np.nan;status[i,j]=1 if math.isfinite(value) else 2
            rows.append({'session':day,'instrument':code,'decision_time':snapshot['decision_times'][day],
                'first_eligible_execution_time':days[i+1]+'T09:30:00+08:00','label_start':days[i+1]+'T09:30:00+08:00',
                'label_end':end+'T15:00:00+08:00','mature_at':mature[-1],
                'value':float(value) if math.isfinite(value) else None,'status':'valid' if math.isfinite(value) else 'data_insufficient',
                'evidence_id':'fixture_'+day+code})
    spec={'definition':'synthetic next-open to 2-session raw close','horizon_sessions':horizon,
          'execution_contract_hash':digest('fixture_execution'),'kind':'prediction_proxy','price_basis':'declared_in_definition'}
    labels={'schema_version':'fuxi.columnar_labels.1','feature_snapshot_hash':cs.hash,'label_spec':spec,
        'sessions':days,'instruments':codes,'mature_at':mature,'evidence_namespace':'fixture_labels',
        'status_codes':{'absent':0,'valid':1,'data_insufficient':2,'not_mature':3,'invalid':4},
        'arrays':{'values':ref('label_values',values),'status':ref('label_status',status)}}
    native_labels={'schema_version':'fuxi.labels.1','feature_snapshot_hash':ns.hash,'label_spec':spec,'rows':rows}
    numeric={'close':close,'open':close*.999,'high':close*1.01,'low':close*.99,
             'volume':np.full(shape,1000000.),'up_limit':close*1.1,'down_limit':close*.9,
             'sizing_reference_close':np.full(shape,np.nan)}
    market={'schema_version':'fuxi.columnar_market.1','sessions':days,'instruments':codes,
        'execution':execution_profile(),'actions':[],'action_terms':[],'provenance':{'fixture':True},
        'event_coverage_certified':False,'tradability_certified':False,'certification_hash':None,
        'evidence_namespace':'fixture_market','sparse_bars':{},
        'arrays':{'numeric':{f:ref('bar_'+f,a) for f,a in numeric.items()},
            'status':{'paused':ref('paused',np.zeros(shape,dtype=np.int8)),
                      'corporate_action_known':ref('events_known',np.ones(shape,dtype=bool))},
            'categorical':{'event_coverage':ref('event_coverage',np.zeros(shape,dtype=np.uint8)),
                           'mark_source':ref('mark_source',np.zeros(shape,dtype=np.uint8))}},
        'category_values':{'event_coverage':{'0':'synthetic_evidenced'},'mark_source':{'0':'actual_raw_close'}}}
    market['market_hash']=digest(market)
    return cs,ns,ColumnarLabels(labels,cs),Labels(native_labels,ns),ColumnarMarket(market)


class ColumnarTests(unittest.TestCase):
    def setUp(self):
        self.scratch=scratch_directory();self.path=self.scratch.__enter__()
        self.cs,self.ns,self.cl,self.nl,self.market=fixture(self.path)
    def tearDown(self):
        self.cs.close();self.cl.close();self.market.close()
        self.scratch.__exit__(None,None,None)

    def assertPanelsClose(self,left,right,places=12):
        self.assertEqual(set(left),set(right));count=0
        for d in left:
            self.assertEqual(set(left[d]),set(right[d]))
            for c in left[d]:
                count+=1;a,b=left[d][c],right[d][c]
                if a is None or b is None:self.assertIs(a,b)
                else:self.assertAlmostEqual(a,b,places=places)
        self.assertGreater(count,0)

    def test_all_dsl_operators_full_cross_section_match_scalar(self):
        field={'field':'close'};ratio={'field':'turnover_rate'}
        asts=[field,ratio]
        asts += [{'op':op,'window':4,'args':[field]} for op in ('lag','return','delta','mean','std','min','max')]
        asts += [{'op':op,'args':[ratio]} for op in ('neg','abs','square','log','rank','zscore')]
        asts += [{'op':op,'args':[{'op':'rank','args':[field]},{'op':'rank','args':[ratio]}]} for op in ('add','sub','mul','div')]
        for ast in asts:
            self.assertPanelsClose(evaluate_columnar_ast(ast,self.cs),evaluate_ast(ast,self.ns))

    def test_average_ties_use_entire_universe_and_constant_missing(self):
        vector=np.array([1.,2.,1.,5.,np.nan,2.])
        expected=ranks(dict(enumerate([None if np.isnan(v) else float(v) for v in vector])))
        actual=rank_vector(vector)
        for j,v in expected.items():
            if v is None:self.assertTrue(np.isnan(actual[j]))
            else:self.assertEqual(v,float(actual[j]))
        self.assertEqual(actual[0],.2)
        self.assertTrue(np.isnan(rank_vector(np.array([np.nan,1.]))).all())
        self.assertTrue(np.array_equal(rank_vector(np.ones(6)),np.full(6,.5)))

    def test_availability_and_barriers_missing_are_not_zero(self):
        code=self.cs.instruments[2];asof=self.cs.clocks[self.cs.sessions[15]]
        self.assertIsNone(self.cs.value(12,code,'close',asof))
        self.assertIsNotNone(self.cs.value(13,code,'close',asof))
        self.assertIsNone(self.cs.value(15,code,'close',self.cs.clocks[self.cs.sessions[14]]))
        self.assertIsNone(self.cs.value(6,self.cs.instruments[1],'close',asof))
        late=deepcopy(self.cs.payload);late['availability_times']['turnover_rate'][15]=self.cs.sessions[16]+'T18:00:00+08:00'
        s=ColumnarSnapshot(late)
        try:
            score=evaluate_columnar_ast({'field':'turnover_rate'},s)
            self.assertTrue(all(v is None for v in score[self.cs.sessions[15]].values()))
        finally:s.close()

    def test_labels_maturity_null_and_lookup_match_legacy(self):
        count=0
        for day in self.cs.sessions:
            for code in self.cs.universe[day]:
                for clock in (self.cs.clocks[day],self.cs.clocks[self.cs.sessions[-1]]):
                    self.assertEqual(self.cl.value(day,code,clock),self.nl.value(day,code,clock));count+=1
        self.assertGreater(count,0)
        self.assertEqual(len(self.cl.rows),len(self.nl.rows))
        d,c=next(iter(self.cl.rows));self.assertEqual(self.cl.rows[(d,c)]['label_end'],self.nl.rows[(d,c)]['label_end'])

    def test_statistics_pairs_and_parent_match_native_metrics(self):
        ast={'op':'rank','args':[{'op':'return','window':3,'args':[{'field':'close'}]}]}
        child=evaluate_columnar_ast(ast,self.cs);native=evaluate_ast(ast,self.ns)
        parent=evaluate_columnar_ast({'op':'rank','args':[{'field':'turnover_rate'}]},self.cs)
        native_parent=evaluate_ast({'op':'rank','args':[{'field':'turnover_rate'}]},self.ns)
        clock=self.cs.sessions[-1]+'T20:00:00+08:00';a=factor_statistics_columnar(child,self.cs,self.cl,DEFAULT_POLICY,clock)
        b=factor_statistics(native,self.ns,self.nl,DEFAULT_POLICY,clock)
        self.assertEqual(a['n_pairs'],b['n_pairs']);self.assertGreater(a['n_pairs'],0)
        self.assertAlmostEqual(a['diagnostics']['spearman']['mean'],b['diagnostics']['spearman']['mean'],places=12)
        start,end=self.cs.sessions[6],self.cs.sessions[-1]
        a=paired_scores_columnar(child,parent,start,end);b=paired_scores(native,native_parent,start,end)
        self.assertEqual(a['n_pairs'],b['n_pairs']);self.assertAlmostEqual(a['mean_abs_correlation'],b['mean_abs_correlation'],places=12)
        a=paired_parent_columnar(child,parent,self.cs,self.cl,start,end,timestamp(clock))
        b=paired_parent(native,native_parent,self.ns,self.nl,start,end,timestamp(clock))
        self.assertEqual(a['n_pairs'],b['n_pairs']);self.assertGreater(a['n_sections'],0)
        self.assertAlmostEqual(a['mean_ic_increment'],b['mean_ic_increment'],places=12)

    def test_selection_mix_and_common_support_preserve_denominator(self):
        a=evaluate_columnar_ast({'field':'close'},self.cs);b=evaluate_columnar_ast({'field':'turnover_rate'},self.cs)
        start,end=self.cs.sessions[8],self.cs.sessions[15]
        selected=selected_by_dates(a,start,end)
        self.assertEqual(len(selected),8);self.assertEqual(set(selected),set(self.cs.sessions[8:16]))
        self.assertPanelsClose(mix_scores_columnar(a,b),mix_scores({d:dict(r) for d,r in a.items()},{d:dict(r) for d,r in b.items()}))
        common=common_support_grid(b,a)
        self.assertEqual(set(common[self.cs.sessions[6]]),set(self.cs.universe[self.cs.sessions[6]]))
        self.assertIsNone(common[self.cs.sessions[6]][self.cs.instruments[1]])
        selected_dict={d:r for d,r in a.items() if start<=d<=end}
        result=factor_statistics_columnar(selected_dict,self.cs,self.cl,DEFAULT_POLICY,self.cs.sessions[-1]+'T20:00:00+08:00')
        self.assertEqual(result['diagnostics']['requested_section_count'],8)

    def test_source_and_score_sidecar_tampering_fail_closed(self):
        scores=evaluate_columnar_ast({'field':'close'},self.cs);ref=save_score_grid(scores,Path(self.path)/'transport','scores')
        restored=load_score_grid(ref)
        try:self.assertPanelsClose(scores,restored)
        finally:restored.close()
        path=Path(ref['values']['path']);content=path.read_bytes();changed=Path(self.path)/'altered_scores.npy'
        changed.write_bytes(content[:-1]+bytes([content[-1]^1]));altered=deepcopy(ref);altered['values']['path']=str(changed.resolve())
        altered['reference_hash']=digest({k:v for k,v in altered.items() if k!='reference_hash'})
        with self.assertRaises(ContractError):load_score_grid(altered)
        source=self.cs.payload['arrays']['fields']['close']['path'];raw=Path(source).read_bytes();changed=Path(self.path)/'altered_feature.npy'
        changed.write_bytes(raw+b'x');altered=deepcopy(self.cs.payload);altered['arrays']['fields']['close']['path']=str(changed.resolve())
        with self.assertRaises(ContractError):ColumnarSnapshot(altered)

    def test_compact_style_preserves_real_metric_sections(self):
        scores=evaluate_columnar_ast({'op':'return','window':3,'args':[{'field':'close'}]},self.cs)
        styles={'turnover':evaluate_columnar_ast({'field':'turnover_rate'},self.cs)}
        clock=self.cs.sessions[-1]+'T20:00:00+08:00'
        a=style_attribution_columnar(scores,styles,self.cl,self.cs,clock)
        b=style_attribution({d:dict(r) for d,r in scores.items()},{n:{d:dict(r) for d,r in p.items()} for n,p in styles.items()},self.nl,self.ns,clock)
        self.assertEqual(a['n_pairs'],b['n_pairs']);self.assertGreater(a['n_pairs'],0)
        for x,y in zip(a['sections'],b['sections']):
            self.assertEqual(x['n_pairs'],y['n_pairs'])
            if x['status']=='valid':
                self.assertEqual(x['support_hash'],y['support_hash'])
                for key in ('raw_pearson_ic','residual_pearson_ic','explained_pearson_ic'):
                    self.assertAlmostEqual(x[key],y[key],places=11)
        self.assertTrue(all('raw_signal' not in r for r in a['sections']))

    def test_market_manifest_iteration_and_lazy_rows_match(self):
        self.assertEqual(dict(self.market),self.market.manifest_payload)
        self.assertEqual(len(self.market['bars']),len(self.cs.sessions)*len(self.cs.instruments))
        d,c=self.cs.sessions[0],self.cs.instruments[0];row=self.market.bar_index[(d,c)]
        self.assertEqual(row['session'],d);self.assertEqual(row['instrument'],c)
        self.assertIs(row['paused'],False);self.assertIs(row['corporate_action_known'],True)
        self.assertEqual(row['evidence_id'],digest(['fixture_market',d,c]))
        self.assertEqual(row['action_ids'],[])

    def test_account_quote_domain_and_complete_small_replay_equivalence(self):
        account=Account('1000000');code=self.cs.instruments[0]
        account.positions[code]=[{'quantity':100,'sellable_at':timestamp(self.cs.sessions[0]+'T09:30:00+08:00')}]
        pending=self.cs.instruments[2];account.share_receivables['claim']={'instrument':pending,'quantity':20}
        day=self.cs.sessions[4];when=day+'T18:00:00+08:00'
        all_quotes={c:{'price':str(b['close']),'asof':when,'evidence_id':b['evidence_id']} for c,b in self.market.by_day[day].items() if b.get('close') is not None and b['close']>0}
        selective=account_quotes(self.market,account,day)
        self.assertEqual(set(selective),{code,pending})
        self.assertEqual(account.mark(when,all_quotes),account.mark(when,selective))
        scores=evaluate_columnar_ast({'op':'rank','args':[{'field':'close'}]},self.cs)
        expanded={d:dict(r) for d,r in scores.items()}
        ir=compile_strategy(strategy_spec('same_ir',top_n=2,max_weight=.5,rebalance_sessions=2),expanded,self.cs,digest(self.market['execution']))
        legacy=dict(self.market);legacy['bars']=[dict(r) for r in self.market['bars']]
        context=object.__new__(ResearchAccountContext);populate_research_context(context,self.market)
        a=run_account(ir,legacy,start=self.cs.sessions[5],end=self.cs.sessions[-1])
        b=run_account(ir,self.market,start=self.cs.sessions[5],end=self.cs.sessions[-1],research_context=context)
        self.assertEqual(a,b);self.assertGreater(len(a['daily']),0);self.assertGreater(len(a['orders']),0)

    def test_complete_window_stability_large_values_and_ties(self):
        window=np.array([[1e150,4.,2.,np.nan],[1e150,4.,3.,2.],[1e150,4.,4.,3.]])
        self.assertEqual(window_reduction(window,'std')[0],0.)
        self.assertEqual(window_reduction(window,'std')[1],0.)
        self.assertAlmostEqual(window_reduction(window,'std')[2],1.,places=14)
        self.assertTrue(np.isnan(window_reduction(window,'mean')[3]))
        repeated=np.tile(np.array([1.,2.,3.,4.])[:,None],(1,16))
        self.assertTrue(np.array_equal(rank_vector(window_reduction(repeated,'std')),np.full(16,.5)))
        from statistics import stdev
        tiny_variations=np.array([1e150,1e150+1e135,1e150+2e135])
        exact=stdev(tiny_variations.tolist());actual=window_reduction(tiny_variations[:,None],'std')[0]
        self.assertLess(abs(actual/exact-1),1e-12)

    def test_walk_forward_fit_rows_maturity_and_predictions_match_native(self):
        from fuxi_v1.models import walk_forward
        features={'close_rank':evaluate_columnar_ast({'op':'rank','args':[{'field':'close'}]},self.cs),
                  'turnover':evaluate_columnar_ast({'field':'turnover_rate'},self.cs)}
        native={n:{d:dict(r) for d,r in grid.items()} for n,grid in features.items()}
        kw={'algorithm':'ridge','min_train_sessions':5,'lookback_sessions':15,'retrain_every':6,
            'gap_sessions':1,'seed':17,'predict_start':self.cs.sessions[16]}
        a=walk_forward_columnar(features,self.cs,self.cl,**kw);b=walk_forward(native,self.ns,self.nl,**kw)
        self.assertGreater(a['trained_model_count'],0);self.assertEqual(a['trained_model_count'],b['trained_model_count'])
        self.assertPanelsClose(a['scores'],b['scores'])
        for left,right in zip(a['models'],b['models']):
            excluded={'feature_snapshot_hash','labels_hash','bundle_hash'}
            self.assertEqual({k:v for k,v in left.items() if k not in excluded},{k:v for k,v in right.items() if k not in excluded})
            self.assertLessEqual(timestamp(left['latest_training_label_mature_at']),timestamp(left['training_cutoff']))
        self.assertEqual([{k:v for k,v in r.items() if k!='model_hash'} for r in a['prediction_provenance']],
                         [{k:v for k,v in r.items() if k!='model_hash'} for r in b['prediction_provenance']])


if __name__=='__main__':unittest.main()
