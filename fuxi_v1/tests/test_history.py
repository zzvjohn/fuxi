from copy import deepcopy
from decimal import Decimal
import unittest
import numpy as np
from fuxi_v1.history import name_state, adjustment_class, select_cohort, mainboard
from fuxi_v1.history import normalize_price_frame
from fuxi_v1.historical_adapter import HistoricalSnapshot, dividend_terms
from fuxi_v1.research_protocol import block_inference_fast, hac_planning, screen
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.contracts import digest, ContractError
from fuxi_v1.portfolio import strategy_spec,compile_strategy
from fuxi_v1.simulation import run_account


def _two_exchange_campaign(sessions=65, instruments=6):
    """synthetic_campaign 生成全 XSHG；把后一半 instrument 改成 XSHE 构造沪深两市。"""
    from fuxi_v1.datasets import materialize_labels
    p = synthetic_campaign(sessions, instruments)
    snap = p['snapshot']
    codes = [str(600000 + j) + '.XSHG' for j in range(instruments)]
    remap = {c: ('00000' + str(j) + '.XSHE' if j >= instruments // 2 else c) for j, c in enumerate(codes)}
    snap['universe'] = {d: [remap[c] for c in row] for d, row in snap['universe'].items()}
    for o in snap['observations']:
        if o['instrument'] in remap:
            o['instrument'] = remap[o['instrument']]
    for b in p['market']['bars']:
        if b['instrument'] in remap:
            b['instrument'] = remap[b['instrument']]
    p['market']['market_hash'] = digest({k: v for k, v in p['market'].items() if k != 'market_hash'})
    p['labels'] = materialize_labels(Snapshot(snap), p['market'], 5)
    return p


class HistoryTests(unittest.TestCase):
    def test_price_rotation_requires_two_identities_and_units(self):
        import pandas as pd
        native={'ts_code':'000001.SZ','trade_date':'20260910','open':10.,'high':11.,'low':9.,'close':10.1,
                'pre_close':10.,'pct_chg':1.,'change':.1,'vol':1000.,'amount':1000.}
        rotated=dict(native,pct_chg=.1,vol=1.,amount=1000.,change=1000.)
        invalid=dict(rotated,change=-1000.)
        f=normalize_price_frame(pd.DataFrame([native,rotated,invalid]))
        self.assertEqual(f.field_contract_status.tolist(),['header_layout_verified','vendor_order_rotation_verified','unresolved_field_contract'])
        self.assertEqual(f.vol.iloc[1],1000.);self.assertEqual(f.amount.iloc[1],1000.)
        self.assertTrue(pd.isna(f.vol.iloc[2]));self.assertTrue(pd.isna(f.close.iloc[2]))
        no_change=normalize_price_frame(pd.DataFrame([native]).drop(columns='change'))
        self.assertEqual(no_change.field_contract_status.iloc[0],'header_layout_verified')

    def test_active_collector_reorders_or_fails_without_rewriting(self):
        import ast,os
        import pandas as pd
        from pathlib import Path
        from fuxi_v1.tests.test_kernel import scratch_directory
        source=Path(__file__).resolve().parents[2]/'scripts/stage0_daily_update_v3.py'
        tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='smart_append_csv')
        namespace={'pd':pd,'os':os,'normalize_date':lambda value:value,
                   'get_line_count':lambda p:max(0,len(Path(p).read_text(encoding='utf-8-sig').splitlines())-1) if Path(p).exists() else 0}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        append=namespace['smart_append_csv']
        with scratch_directory() as directory:
            path=Path(directory)/'daily.csv';path.write_text('ts_code,trade_date,vol,amount\nA,20240101,1000,2000\n',encoding='utf-8')
            append(str(path),pd.DataFrame([{'amount':2200,'vol':1100,'trade_date':'20240102','ts_code':'A'}]),['ts_code','trade_date'])
            self.assertEqual(pd.read_csv(path).vol.tolist(),[1000,1100]);before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'schema changed'):
                append(str(path),pd.DataFrame([{'amount':2200,'trade_date':'20240103','ts_code':'A'}]),['ts_code','trade_date'])
            self.assertEqual(before,path.read_bytes())

    def test_mainboard_includes_sme_and_all_sh_prefixes_not_star_or_b(self):
        for c in ['000001.SZ','001001.SZ','002001.SZ','003001.SZ','600001.SH','601001.SH','603001.SH','605001.SH']:self.assertTrue(mainboard(c))
        for c in ['300001.SZ','688001.SH','900901.SH','200001.SZ','920001.BJ']:self.assertFalse(mainboard(c))

    def test_cohort_keeps_delisted_and_is_order_independent(self):
        master=[{'ts_code':c,'list_date':'20000101','delist_date':'20250101','name':'退市','list_status':'D'}
                for c in ['000001.SZ','000002.SZ','600001.SH','600002.SH']]
        a=select_cohort(master,'2021-01-04','2026-09-24',1)
        b=select_cohort(list(reversed(master)),'2021-01-04','2026-09-24',1)
        self.assertEqual(a,b);self.assertEqual(len(a['members']),2)

    def test_st_unknown_never_backfilled_from_future_removal(self):
        records=[{'name':'正常名称','start_date':'20230101','end_date':None,'ann_date':'20221231'}]
        self.assertEqual(name_state(records,'2021-03-01')[0],'unknown')
        self.assertEqual(name_state(records,'2023-01-02')[0],'normal')
        records[0]['name']='*ST测试';self.assertEqual(name_state(records,'2023-01-02')[0],'st')
        records[0]['ann_date']='20230103';self.assertEqual(name_state(records,'2023-01-02')[0],'not_yet_published')

    def test_small_drift_requires_reference_and_not_event(self):
        self.assertEqual(adjustment_class(125.0496,125.0493,10.,10.),'tiny_drift_consistent_reference_research_only')
        self.assertEqual(adjustment_class(125.0496,125.0493,10.,9.9),'factor_or_reference_conflict')
        self.assertEqual(adjustment_class(125.0496,125.0493,10.,10.,True),'published_corporate_action')
        self.assertEqual(adjustment_class(None,125.,10.,10.),'missing_evidence')

    def test_vendor_revision_drift_up_to_one_percent_is_research_sensitive(self):
        # P-20261015-011: vendor adj revisions (e.g. 300295.SZ 3.869 -> 3.8695 on
        # 2024-06-25) drift up to ~0.76% relative on days whose pre_close chain is
        # continuous. Those cells are research-consistent, not corporate actions.
        self.assertEqual(adjustment_class(3.869, 3.8695, 8.47, 8.47),
                         'tiny_drift_consistent_reference_research_only')
        # 0.76% relative drift (603081.SH 1.0617 -> 1.0698) with continuous chain.
        self.assertEqual(adjustment_class(1.0617, 1.0698, 9.05, 9.05),
                         'tiny_drift_consistent_reference_research_only')
        # A broken price chain is a real event even when the adj change is small.
        self.assertEqual(adjustment_class(1.0625, 1.0622, 30.31, 31.29),
                         'factor_or_reference_conflict')
        # Above the 1% sanity bound the cell stays flagged for human attention.
        self.assertEqual(adjustment_class(1.0, 1.012, 10., 10.), 'factor_or_reference_conflict')

    def test_missing_dividend_pay_or_stock_terms_not_invented(self):
        row={'ts_code':'600001.SH','ex_date':'20240103','record_date':'20240102','div_proc':'实施',
             'cash_div_tax':'0.1','stk_div':'0','pay_date':'20240105','imp_ann_date':'20231228'}
        days=['2024-01-02','2024-01-03','2024-01-04','2024-01-05']
        self.assertEqual(dividend_terms([row,row],days)[0]['status'],'cash_terms_supported')
        self.assertEqual(dividend_terms([dict(row,pay_date=None)],days)[0]['status'],'dividend_terms_missing')
        self.assertEqual(dividend_terms([dict(row,stk_div='0.1')],days)[0]['status'],'share_distribution_requires_settlement_adapter')
        self.assertEqual(dividend_terms([row,dict(row,cash_div_tax='0.2')],days)[0]['status'],'conflicting_dividend_rows')

    def test_columnar_matches_native_and_blocks_cross_event_history(self):
        p=synthetic_campaign(28,3)['snapshot'];old=Snapshot(p);codes=p['universe'][p['sessions'][0]]
        packed={'schema_version':'fuxi.historical_snapshot.1','exposure':'research_exposed','source_id':'fixture',
                'sessions':p['sessions'],'evaluation_start':p['evaluation_start'],'universe':p['universe'],
                'decision_times':p['decision_times'],'fields':p['fields'],'instruments':codes,
                'columns':{f:[[old.value(i,c,f,old.clocks[d]) for c in codes] for i,d in enumerate(old.sessions)] for f in p['fields']},
                'barriers':[[-1]*3 for _ in p['sessions']]}
        new=HistoricalSnapshot(packed);ast={'op':'return','args':[{'field':'close'}],'window':5}
        self.assertEqual(evaluate_ast(ast,new),evaluate_ast(ast,old))
        new.payload['barriers'][23][0]=22
        self.assertIsNone(new.value(20,codes[0],'close',new.clocks[new.sessions[23]]))
        self.assertIsNone(new.value(24,codes[0],'close',new.clocks[new.sessions[23]]))
        packed['columns']['close'][0][0]=float('nan')
        with self.assertRaises(ContractError):HistoricalSnapshot(packed)

    def test_bootstrap_exact_manual_draws_and_gaps(self):
        values=[.1,.2,None,-.3,.4,.2,.1,.05,.02,.03,.04,.08,.09]
        got=block_inference_fast(values,block_length=3,replicates=199,seed=9)
        rng=np.random.default_rng(9);draws=[]
        for size in [128,71]:
            starts=rng.integers(0,len(values),size=(size,5))
            for starts_row in starts:
                sample=[values[(int(s)+j)%len(values)] for s in starts_row for j in range(3)][:len(values)]
                sample=[x for x in sample if x is not None];draws.append(sum(sample)/len(sample))
        self.assertAlmostEqual(got['lower'],float(np.quantile(draws,.025)),12)
        self.assertEqual(got['n_pairs'],12);self.assertEqual(got['calendar_slots'],13)

    def test_hac_dependency_and_power_are_advisory(self):
        rng=np.random.default_rng(7);noise=rng.normal(size=2000);ar=[];x=0
        for v in noise:x=.8*x+v;ar.append(float(x))
        iid=hac_planning(noise.tolist());dep=hac_planning(ar)
        self.assertGreater(dep['estimates'][-1]['dependence_ratio'],iid['estimates'][-1]['dependence_ratio']*3)
        self.assertFalse(dep['is_admission_gate'])

    def test_screen_stability_distinct_from_data_failure(self):
        rules={'screen_min_pairs':10,'screen_min_sessions':120,'screen_pair_coverage':.8,
               'screen_positive_year_fraction':.6,'screen_positive_exchange_count':2}
        request={'screen_rules':rules,'block_grid':[6,12,24],'alpha':.05,'bootstrap_replicates':999}
        base={'diagnostics':{'sections':[{'pair_count':20,'universe_count':20,'rank_ic':.1} for _ in range(150)]},
              'eval_status':'valid','metrics':{},'decision':'not_applicable','reject_class':None}
        periods={'2022':{'n':150,'mean_rank_ic':.1},'2023':{'n':150,'mean_rank_ic':.1}}
        ex={'SH':{'n':150,'mean_rank_ic':.1},'SZ':{'n':150,'mean_rank_ic':.1}}
        passed=screen(deepcopy(base),request,periods,ex)
        self.assertEqual(passed['decision'],'pass');self.assertIsNone(passed['diagnostics']['power_required_sections'])
        ex['SZ']['mean_rank_ic']=-.1
        self.assertEqual(screen(deepcopy(base),request,periods,ex)['reject_class'],'statistical')
        base['diagnostics']['sections']=base['diagnostics']['sections'][:20]
        self.assertEqual(screen(base,request,periods,ex)['eval_status'],'data_insufficient')

    def test_dividend_record_holdings_and_later_cash(self):
        p=synthetic_campaign(32,3);s=Snapshot(p['snapshot']);days=s.sessions;codes=s.universe[days[0]]
        market=p['market'];execution=market['execution'];execution['participation']='1'
        scores={d:{c:(1. if c==codes[0] and i<23 else 2. if c==codes[1] and i>=23 else 0.) for c in codes} for i,d in enumerate(days) if i>=s.start}
        term={'instrument':codes[0],'record_date':days[22],'ex_date':days[23],'pay_date':days[27],
              'status':'cash_terms_supported','gross_cash_per_share':.1,'tax_fraction':'.20','evidence_id':'cash_term'}
        market['action_terms']=[term]
        for bar in market['bars']:
            if bar['session']==days[23] and bar['instrument']==codes[0]:
                bar['sizing_reference_close']=bar['close'];bar['sizing_reference_evidence']='reference'
        market['market_hash']=digest({k:v for k,v in market.items() if k!='market_hash'})
        spec=strategy_spec('dividend',top_n=1,max_weight=1.,rebalance_sessions=1,turnover_budget=1.)
        ir=compile_strategy(spec,scores,s,digest(execution));run=run_account(ir,market,start=days[20],end=days[-1])
        record=next(r for r in run['daily'] if r['session']==days[22])
        q=record['holdings'][codes[0]];receipts=[e for e in run['journal'] if e['kind']=='receivable']
        self.assertEqual(len(receipts),1);self.assertEqual(Decimal(receipts[0]['payload']['amount']),Decimal(q)*Decimal('.08'))
        self.assertEqual([e['timestamp'][:10] for e in run['journal'] if e['kind']=='payment'],[days[27]])

    def test_baseline_cache_preserves_readouts(self):
        from fuxi_v1.research_evaluation import evaluate_candidate
        from fuxi_v1.search import generate,node
        from fuxi_v1.data import Labels
        p=_two_exchange_campaign(65,6);s=Snapshot(p['snapshot']);labels=Labels(p['labels'],s)
        candidate=generate(s.payload['fields'],arm='fixed',count=1,seed=1)[0]
        scores=evaluate_ast(candidate['canonical_ast'],s)
        baseline=evaluate_ast(node('rank',node('return',{'field':'close'},window=20)),s)
        request={'research_start':s.sessions[30],'research_end':s.sessions[-1],'block_length':6,'alpha':.05,
                 'max_trials':3,'bootstrap_replicates':99,'minimum_net_increment_daily':0.,
                 'cost_stress_multiplier':2.,'capacity_stress_multiplier':.5,
                 'research_gate_version':'history_research_v2',
                 'block_grid':[6,12,24],
                 'screen_rules':{'screen_min_pairs':10,'screen_min_sessions':120,'screen_pair_coverage':.8,
                                 'screen_positive_year_fraction':.6,'screen_positive_exchange_count':2}}
        policy={'request':request,'minimum_effective_sections':2,'policy_hash':'test'};cache={}
        a=evaluate_candidate(candidate,scores,baseline,s,labels,p['market'],policy,p['research_clock'],baseline_cache=cache)
        b=evaluate_candidate(candidate,scores,baseline,s,labels,p['market'],policy,p['research_clock'],baseline_cache=cache)
        self.assertEqual(a,b);self.assertEqual(len(cache),3)


if __name__=='__main__':unittest.main()
