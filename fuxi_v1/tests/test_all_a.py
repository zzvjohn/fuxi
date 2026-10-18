from copy import deepcopy
import json
import tempfile
import unittest
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.a_share import (board, rule_for, instrument_rule, buy_quantity, sell_quantity,
                            desired_shares, exclude_new_entries, delivery_verdict,identity_aliases,venue_code_at)
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.tushare_source import TushareSource, VendorFailure
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.historical_adapter import HistoricalSnapshot
from fuxi_v1.portfolio import strategy_spec, compile_strategy
from fuxi_v1.simulation import run_account
from fuxi_v1.action_accounting import reconcile_account
from fuxi_v1.pit_universe import lifecycle_from_rows,reproduce_cohort,ALL_A_FRAME_PREDICATE


class AllATests(unittest.TestCase):
    def test_types_and_aliases(self):
        for c,b in [('600000.XSHG','main_sh'),('000001.XSHE','main_sz'),('300001.SZ','chinext'),
                    ('688001.SH','star'),('920001.BJ','bj'),('830001.BJ','bj')]: self.assertEqual(board(c),b)
        for c in ('900001.SH','200001.SZ','689009.SH','000300.SH'): self.assertIsNone(board(c))

    def test_quantity_boundaries(self):
        star, bj, main = rule_for('688001.SH'), rule_for('920001.BJ'), rule_for('600000.SH')
        self.assertEqual([buy_quantity(q,star) for q in (199,200,201,100001)], [0,200,201,100000])
        self.assertEqual([buy_quantity(q,bj) for q in (99,100,101,1000001)], [0,100,101,1000000])
        self.assertEqual([buy_quantity(q,main) for q in (99,100,199,201)], [0,100,100,200])
        self.assertEqual(sell_quantity(199,199,199,199,star),199)
        self.assertEqual(sell_quantity(199,300,300,1000,star),0)
        self.assertEqual(sell_quantity(201,201,0,1000,star),0)
        self.assertEqual(sell_quantity(350,350,350,350,main),350)
        self.assertEqual(sell_quantity(99,350,350,350,main),0)
        chinext=rule_for('300001.SZ')
        self.assertEqual(chinext['max_market_order'],150000)
        self.assertEqual(buy_quantity(1000000,chinext),300000)
        self.assertEqual(sell_quantity(1000000,1000000,1000000,1000000,chinext),300000)
        self.assertEqual(buy_quantity(1000000,chinext,order_type='market'),150000)
        self.assertEqual(buy_quantity(100001,star,order_type='market'),50000)
        self.assertEqual(sell_quantity(100001,100001,100001,100001,star,order_type='market'),50000)

    def test_explicit_board_rules_not_fallback(self):
        legacy={'buy_lot':100,'sell_lot':100,'settlement_sessions':1}
        with self.assertRaises(ContractError): instrument_rule(legacy,'688001.SH')
        explicit={**legacy,'instrument_rules':{'688001.SH':rule_for('688001.SH')}}
        self.assertEqual(instrument_rule(explicit,'688001.SH')['buy_step'],1)
        explicit['instrument_rules']['688001.SH']['min_buy']=100
        with self.assertRaises(ContractError): instrument_rule(explicit,'688001.SH')

    def test_exclusion_only_after_evidence(self):
        row={'instrument':'600000.SH','reason':'transfer_entitlement_missing','evidence_ids':['e'],
             'observed_at':'2026-10-03T09:00:00+08:00','effective_at':'2026-10-03T09:00:00+08:00',
             'review_after':'2026-10-10T09:00:00+08:00'}
        codes=['600000.XSHG','688001.SH']
        self.assertEqual(exclude_new_entries(codes,[row],'2026-10-02T18:30:00+08:00'),codes)
        self.assertEqual(exclude_new_entries(codes,[row],'2026-10-03T18:30:00+08:00'),['688001.SH'])
        row['effective_at']='2026-10-01T09:00:00+08:00'
        with self.assertRaises(ContractError): exclude_new_entries(codes,[row],'2026-10-03T18:30:00+08:00')

    def test_unknown_assurance_never_delivery(self):
        verdict=delivery_verdict(True,True,False,False,False)
        self.assertFalse(verdict['research_delivery_ready']);self.assertFalse(verdict['live_delivery_ready'])

    def test_bse_venue_code_changes_do_not_change_stable_identity(self):
        aliases=identity_aliases([{'o_code':'833819.BJ','n_code':'920819.BJ'},
                                 {'o_code':'839729.BJ','n_code':'920729.BJ'}])
        self.assertEqual(venue_code_at('920819.BJ','2025-05-05',aliases),'833819.BJ')
        self.assertEqual(venue_code_at('920819.BJ','2025-05-06',aliases),'920819.BJ')
        self.assertEqual(venue_code_at('920729.BJ','2025-10-08',aliases),'839729.BJ')
        self.assertEqual(venue_code_at('920729.BJ','2025-10-09',aliases),'920729.BJ')
        with self.assertRaises(ContractError):identity_aliases([{'o_code':'833819.BJ','n_code':'920819.BJ'}]*2)

    def test_excluded_holdings_retain_weight_without_automatic_exit(self):
        from fuxi_v1.portfolio import target_weights
        prior={'600000.SH':.4,'688001.SH':.2}
        spec=strategy_spec('frozen',top_n=1,max_weight=1.,turnover_budget=.2)
        target=target_weights({'688001.SH':1.},spec,prior=prior,frozen_prior=['600000.SH'])
        self.assertEqual(target['600000.SH'],.4)
        self.assertLessEqual(sum(target.values()),1.)
        self.assertLessEqual(sum(abs(target.get(c,0)-prior.get(c,0)) for c in set(target)|set(prior)),.2+1e-12)

    def test_all_entry_exclusions_hold_account_and_future_resolution_does_not_rewrite_decision(self):
        from fuxi_v1.online import daily_decision
        p=synthetic_campaign(25,3)['snapshot'];base=SnapshotForTest(p)
        day=base.sessions[base.start];clock=base.payload['decision_times'][day]
        rows=[{'instrument':c,'reason':'evidence_missing','evidence_ids':['r'],
            'observed_at':clock,'effective_at':clock,'review_after':'2027-01-01T00:00:00+08:00'} for c in base.universe[day]]
        spec=strategy_spec('empty');ast={'field':'close'}
        decision=daily_decision(ast,base,day,clock,spec,exclusions=rows)
        self.assertTrue(decision['missing_signal_hold']);self.assertIsNone(decision['target_weights'])
        rows[0].update(resolved_at='2027-01-02T00:00:00+08:00',resolution_evidence_ids=['future'])
        self.assertEqual(decision['decision_id'],daily_decision(ast,base,day,clock,spec,exclusions=rows)['decision_id'])

    def test_vendor_pagination_cache_and_failure(self):
        calls=[]
        def transport(p):
            calls.append(p)
            rows=[[1],[2]] if p['params']['offset']==0 else [[3]]
            return json.dumps({'code':0,'data':{'fields':['x'],'items':rows}}).encode()
        with scratch_directory() as temp:
            source=TushareSource(temp,'fake',interval=0,transport=transport)
            self.assertEqual(len(source.query('fixture',{},'x',page_size=2)['rows']),3)
            source.query('fixture',{},'x',page_size=2);self.assertEqual(len(calls),2)
        with scratch_directory() as temp:
            source=TushareSource(temp,'secret',interval=0,transport=lambda p:json.dumps({'code':-1,'msg':'token secret 无效'}).encode())
            with self.assertRaises(VendorFailure):source.query('fixture',{},'x')
            from pathlib import Path
            self.assertNotIn('secret',next(Path(temp).glob('failure*')).read_text())

    def test_duplicate_page_refused(self):
        with scratch_directory() as temp:
            source=TushareSource(temp,'fake',interval=0,transport=lambda p:json.dumps({'code':0,'data':{'fields':['x'],'items':[[1],[2]]}}).encode())
            with self.assertRaises(ContractError):source.query('fixture',{},'x',page_size=2)

    def test_sizing(self):
        self.assertEqual(desired_shares('2010','10',rule_for('688001.SH')),201)
        self.assertEqual(desired_shares('2010','10',rule_for('600000.SH')),200)

    def test_census_has_all_boards_and_delisted_members(self):
        codes=['600000.SH','000001.SZ','300001.SZ','688001.SH','920001.BJ']
        life=lifecycle_from_rows([{'ts_code':c,'list_date':'20210101','delist_date':'20260901' if c=='600000.SH' else None} for c in codes])
        declaration={'rule_kind':'all_a_share_census','salt':'scope','frame_predicate':ALL_A_FRAME_PREDICATE}
        got,basis=reproduce_cohort(life,declaration,['2021-01-04','2026-09-30'])
        self.assertEqual(got,sorted(codes));self.assertTrue(basis['permutation_invariant'])

    def test_native_accounts_across_all_boards(self):
        from fuxi_v1.data import Snapshot
        for replacement in ('000001.XSHE','300001.XSHE','688001.XSHG','920001.BJ'):
            with self.subTest(code=replacement):
                p=synthetic_campaign(25,3);old=p['snapshot']['universe'][p['snapshot']['sessions'][0]][0]
                for row in p['snapshot']['observations']+p['market']['bars']:
                    if row['instrument']==old:row['instrument']=replacement
                for names in p['snapshot']['universe'].values(): names[names.index(old)]=replacement
                s=Snapshot(p['snapshot']);market=p['market'];execution=market['execution']
                execution['instrument_rules']={c.replace('.XSHE','.SZ').replace('.XSHG','.SH'):rule_for(c) for c in s.universe[s.sessions[0]]}
                execution['odd_lot_exit']='whole_residual'
                scores={d:{c:(1. if c==replacement else 0.) for c in s.universe[d]} for d in s.sessions[s.start:]}
                ir=compile_strategy(strategy_spec('board',top_n=1,max_weight=1.),scores,s,digest(execution))
                result=run_account(ir,market)
                self.assertEqual(result['eval_status'],'valid')
                self.assertGreater(sum(o['filled'] for o in result['orders']),0)
                audit=reconcile_account(result,execution['initial_cash'],market)
                self.assertTrue(audit['cash_and_share_reconciled']);self.assertTrue(audit['nav_reconciled'])

    def test_native_market_order_uses_market_cap_for_star(self):
        from fuxi_v1.data import Snapshot
        p=synthetic_campaign(25,3);old=p['snapshot']['universe'][p['snapshot']['sessions'][0]][0]
        star='688001.XSHG'
        for row in p['snapshot']['observations']+p['market']['bars']:
            if row['instrument']==old:row['instrument']=star
        for names in p['snapshot']['universe'].values():names[names.index(old)]=star
        snapshot=Snapshot(p['snapshot']);market=p['market'];execution=market['execution']
        execution['instrument_rules']={c.replace('.XSHG','.SH'):rule_for(c) for c in snapshot.universe[snapshot.sessions[0]]}
        execution['order_type']='market'
        for bar in market['bars']:
            if bar['instrument']==star:
                bar.update(open=.1,high=.2,low=.05,close=.1,volume=1000000000,up_limit=1.,down_limit=.01)
        scores={d:{c:1. if c==star else 0. for c in snapshot.universe[d]} for d in snapshot.sessions[snapshot.start:]}
        ir=compile_strategy(strategy_spec('market_cap',top_n=1,max_weight=1.),scores,snapshot,digest(execution))
        result=run_account(ir,market)
        filled=[o['filled'] for o in result['orders'] if o['instrument']==star and o['side']=='buy' and o['filled']]
        self.assertTrue(filled);self.assertEqual(max(filled),50000)
        self.assertEqual(result['eval_status'],'valid')

    def test_catalog_overlay_empty_query_and_conflicting_duplicates(self):
        from pathlib import Path
        import pandas as pd
        from fuxi_v1.market_catalog import load_table
        from fuxi_v1.datasets import file_hash
        with scratch_directory() as temp:
            root=Path(temp);part=root/'basic.parquet'
            pd.DataFrame([{'ts_code':'000001.SZ','trade_date':'20260928','turnover_rate':'1'},
                          {'ts_code':'000001.SZ','trade_date':'20260928','turnover_rate':'2'},
                          {'ts_code':'000001.SZ','trade_date':'20260929','turnover_rate':'1'}]).to_parquet(part,index=False)
            query=root/'empty.json';query.write_text(json.dumps({'rows':[],'rows_hash':digest([])}),encoding='utf-8')
            manifest={'tables':{'daily_basic':[{'file':'basic.parquet','sha256':file_hash(part)}]},
              'fields':{'daily_basic':'ts_code,trade_date,turnover_rate'},
              'queries':{'daily_basic':[{'path':str(query),'sha256':file_hash(query),'params':{'trade_date':'20260929'}}]}}
            manifest['manifest_hash']=digest(manifest);(root/'CATALOG.json').write_text(json.dumps(manifest),encoding='utf-8')
            loaded=load_table(root,'daily_basic.csv',['000001.SZ'],{'20260928','20260929'})
            self.assertEqual(len(loaded),1);self.assertIsNone(loaded.iloc[0]['turnover_rate'])
            self.assertEqual(loaded.iloc[0]['field_contract_status'],'conflicting_duplicate_unavailable')

    def test_audit_duplicate_conflicts_across_chunks_and_explicit_empty_replacement(self):
        import pandas as pd
        from fuxi_v1.panel_audit import accumulate_panel
        code='000001.SZ';days=['20260928','20260929']
        def frame(day,value):return pd.DataFrame([{'ts_code':code,'trade_date':day,'adj_factor':value}])
        result=accumulate_panel('adj_factor',[(frame(days[0],'1'),()),(frame(days[0],'1.0'),()),
            (frame(days[1],'2'),()),(frame(days[1],'3'),())],days,[code],keep=['adj_factor'])
        self.assertTrue(result['valid'][0,0]);self.assertFalse(result['valid'][1,0])
        self.assertTrue(result['conflict'][1,0])
        empty=frame(days[0],'1').iloc[:0]
        result=accumulate_panel('adj_factor',[(frame(days[0],'1'),()),(empty,[days[0]])],days,[code])
        self.assertFalse(result['observed'][0,0])

    def test_partial_basic_observation_is_not_a_conflicting_value_and_limits_need_actual_bounds(self):
        import pandas as pd
        from fuxi_v1.panel_audit import accumulate_panel,numeric_contract
        common={'ts_code':'000001.SZ','trade_date':'20260928','turnover_rate':'1'}
        partial=pd.DataFrame([dict(common,total_share=None,float_share=None,free_share=None)])
        complete=pd.DataFrame([dict(common,total_share='100',float_share='90',free_share='70')])
        result=accumulate_panel('daily_basic',[(partial,()),(complete,())],['20260928'],['000001.SZ'],keep=['float_share'])
        self.assertTrue(result['valid'][0,0]);self.assertFalse(result['conflict'][0,0])
        self.assertEqual(result['values']['float_share'][0,0],90)
        changed=complete.copy();changed['turnover_rate']='2'
        bad=accumulate_panel('daily_basic',[(partial,()),(changed,())],['20260928'],['000001.SZ'])
        self.assertFalse(bad['valid'][0,0]);self.assertTrue(bad['conflict'][0,0])
        limit=pd.DataFrame([dict(common,up_limit='11',down_limit='9')])
        self.assertTrue(numeric_contract('stk_limit',limit)[2][0])
        limit['pre_close']='-1';self.assertFalse(numeric_contract('stk_limit',limit)[2][0])

    def test_stock_range_empty_overlay_preserves_other_stock_and_date(self):
        import pandas as pd
        from pathlib import Path
        from fuxi_v1.panel_audit import accumulate_panel
        from fuxi_v1.market_catalog import load_table
        from fuxi_v1.datasets import file_hash
        days=['20260928','20260929'];codes=['000001.SZ','000002.SZ']
        frame=pd.DataFrame([{'ts_code':c,'trade_date':d,'adj_factor':'1'} for d in days for c in codes])
        replacement={'codes':[codes[0]],'start_date':days[0],'end_date':days[0]}
        result=accumulate_panel('adj_factor',[(frame,()),(frame.iloc[:0],replacement)],days,codes)
        self.assertEqual(result['valid'].tolist(),[[False,True],[True,True]])
        with scratch_directory() as temp:
            root=Path(temp);part=root/'base.parquet';frame.to_parquet(part,index=False)
            query=root/'query.json';query.write_text(json.dumps({'rows':[],'rows_hash':digest([])}),encoding='utf-8')
            manifest={'fields':{'adj_factor':','.join(frame.columns)},'tables':{'adj_factor':[{'file':part.name,'sha256':file_hash(part)}]},
                'queries':{'adj_factor':[{'path':str(query),'sha256':file_hash(query),'params':{
                    'ts_code':codes[0],'start_date':days[0],'end_date':days[0]}}]}}
            manifest['manifest_hash']=digest(manifest);(root/'CATALOG.json').write_text(json.dumps(manifest),encoding='utf-8')
            got=load_table(root,'adj_factor_daily.csv',codes)
            self.assertEqual(set(zip(got.ts_code,got.trade_date)),{(codes[1],days[0]),(codes[0],days[1]),(codes[1],days[1])})

    def test_simultaneous_cash_distributions_require_distinct_unambiguous_periods(self):
        from fuxi_v1.historical_adapter import dividend_terms
        base={'ts_code':'002043.SZ','div_proc':'实施','ex_date':'20260601','record_date':'20260529',
              'pay_date':'20260601','imp_ann_date':'20260526','div_listdate':None,'stk_div':'0'}
        a=dict(base,end_date='20260331',cash_div_tax='.18')
        b=dict(base,end_date='20251231',cash_div_tax='.132',stk_div='0.0')
        term=dividend_terms([a,b],['2026-01-01','2026-09-30'])[0]
        self.assertEqual(term['status'],'cash_terms_supported');self.assertEqual(term['gross_cash_per_share'],.312)
        self.assertEqual(term['cash_component_count'],2)
        for bad in (dict(b,end_date='20260331'),dict(b,end_date='20251011'),dict(b,stk_div='.1')):
            self.assertEqual(dividend_terms([a,bad],['2026-01-01','2026-09-30'])[0]['status'],'conflicting_dividend_rows')
        duplicate=dict(a,stk_div='0.0',cash_div_tax='0.1800')
        self.assertEqual(dividend_terms([a,duplicate],['2026-01-01','2026-09-30'])[0]['gross_cash_per_share'],.18)

    def test_full_catalog_audit_distinguishes_complete_proxy_from_formal_assurance(self):
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from fuxi_v1.panel_audit import audit_catalog
        from fuxi_v1.datasets import file_hash
        for issue in ('none','future_name','halt_quote_conflict','factor_precision','factor_changed','cross_quote_conflict','declared_share_event'):
            future_name=issue=='future_name'
            with self.subTest(issue=issue),scratch_directory() as temp:
                root=Path(temp)/'catalog';out=Path(temp)/'out';root.mkdir();out.mkdir()
                days=['20260928','20260929','20260930'];code='600000.SH';tables={};fields={}
                records={
                    'daily':{'open':'10','high':'10.2','low':'9.8','close':'10','pre_close':'10',
                             'change':'0','pct_chg':'0','vol':'100','amount':'100'},
                    'adj_factor':{'adj_factor':'1'},'stk_limit':{'pre_close':'10','up_limit':'11','down_limit':'9'},
                    'daily_basic':{'turnover_rate':'1','total_share':'100','float_share':'90','free_share':'70'}}
                for api,values in records.items():
                    frame=pd.DataFrame([dict(values,ts_code=code,trade_date=d) for d in days]);path=root/(api+'.parquet')
                    if issue.startswith('factor_') or issue in ('cross_quote_conflict','declared_share_event'):
                        if api=='adj_factor':frame.loc[frame.trade_date!=days[0],'adj_factor']='1.0002'
                    frame.to_parquet(path,index=False);tables[api]=[{'file':path.name,'sha256':file_hash(path)}]
                    fields[api]=','.join(frame.columns)
                halt_rows=[{'ts_code':code,'trade_date':days[0],'suspend_type':'S','suspend_timing':None}] if issue=='halt_quote_conflict' else []
                query=root/'halt.json';query.write_text(json.dumps({'rows':halt_rows,'rows_hash':digest(halt_rows)}),encoding='utf-8')
                manifest={'tables':tables,'fields':fields,'identity_aliases':{},'queries':{'suspend_d':[
                    {'path':str(query),'sha256':file_hash(query),'params':{'start_date':days[0],'end_date':days[-1]}}]}}
                if issue.startswith('factor_') or issue in ('cross_quote_conflict','declared_share_event'):
                    parts=[]
                    for index,day in enumerate(days):
                        params={'trade_date':day}
                        rows=[{'ts_code':code,'trade_date':day,'adj_factor':2 if issue=='factor_changed' and index else 1,
                            'close':11 if issue=='cross_quote_conflict' and index else 10,'pre_close':10}]
                        path=root/('cross_'+day+'.json');path.write_text(json.dumps({'identity':{'api':'stk_factor','params':params},
                            'rows':rows,'rows_hash':digest(rows)}),encoding='utf-8')
                        parts.append({'path':str(path),'sha256':file_hash(path),'params':params})
                    manifest['queries']['stk_factor']=parts
                manifest['manifest_hash']=digest(manifest);(root/'CATALOG.json').write_text(json.dumps(manifest),encoding='utf-8')
                pd.DataFrame([{'ts_code':code,'name':'浦发银行','start_date':'20200101','end_date':None,
                    'ann_date':'20261001' if future_name else '20200101'}]).to_csv(root/'namechange.csv',index=False)
                dividends=[{'ts_code':code,'div_proc':'实施','ex_date':days[1],'record_date':days[0],'pay_date':days[1],
                    'stk_div':'.1','cash_div_tax':'0','div_listdate':days[2],'imp_ann_date':days[0]}] if issue=='declared_share_event' else []
                pd.DataFrame(dividends,columns=['ts_code','div_proc','ex_date','record_date','pay_date','stk_div','cash_div_tax','div_listdate','imp_ann_date']).to_csv(root/'dividend.csv',index=False)
                metadata={'manifest_hash':manifest['manifest_hash'],'files':{f:file_hash(root/f) for f in ('namechange.csv','dividend.csv')}}
                metadata['metadata_hash']=digest(metadata);(root/'CATALOG_METADATA.json').write_text(json.dumps(metadata),encoding='utf-8')
                np.savez(out/'LOCAL_MASKS.npz',active=np.ones((3,1),bool))
                result=audit_catalog(root,{'sessions':days,'codes':[code],'master_rows':[{'ts_code':code}]},out)
                self.assertEqual(result['dense_core_complete_instruments'],1)
                # P-009 (a1): vendor-dated pure share events now settle under the
                # vendor proxy (share leg credited on the vendor credit date), so a
                # declared share event keeps vendor-proxy completeness; it is still
                # never formal data assurance (no broker receipt).
                self.assertEqual(result['data_complete_under_vendor_proxy_instruments'],
                                 1 if issue in ('none','factor_precision','declared_share_event') else 0)
                self.assertEqual(result['formal_data_assured_instruments'],0)
                self.assertEqual(result['gap_counts'].get('name_missing_or_conflicting',0),3 if future_name else 0)
                self.assertEqual(result['gap_counts'].get('halt_quote_conflict',0),1 if issue=='halt_quote_conflict' else 0)
                self.assertEqual(result['factor_corroboration']['accepted_cells'],1 if issue=='factor_precision' else 0)

    def test_jq_compiles_same_star_quantity_helpers(self):
        from fuxi_v1.data import Snapshot
        from fuxi_v1.jq_strategy import compile_jq_strategy
        p=synthetic_campaign(25,3);old=p['snapshot']['universe'][p['snapshot']['sessions'][0]][0];star='688001.XSHG'
        for row in p['snapshot']['observations']+p['market']['bars']:
            if row['instrument']==old:row['instrument']=star
        for names in p['snapshot']['universe'].values():names[names.index(old)]=star
        snapshot=Snapshot(p['snapshot']);market=p['market']
        market['execution']['instrument_rules']={c.replace('.XSHG','.SH'):rule_for(c) for c in snapshot.universe[snapshot.sessions[0]]}
        market['execution']['order_type']='market'
        scores={d:{c:1. for c in snapshot.universe[d]} for d in snapshot.sessions[snapshot.start:]}
        ir=compile_strategy(strategy_spec('mapped'),scores,snapshot,digest(market['execution']))
        script,metadata=compile_jq_strategy(ir,market)
        compile(script,'generated.py','exec')
        self.assertIn(b'board_sell_quantity=sell_quantity',script)
        self.assertFalse(metadata['observed_on_jq'])

    def test_invalid_observed_price_bounds_block_account(self):
        from fuxi_v1.data import Snapshot
        for up,down in ((11,0),(11,-1),(9,11),(True,9)):
            with self.subTest(up=up,down=down):
                p=synthetic_campaign(25,3);snapshot=Snapshot(p['snapshot']);market=p['market']
                for bar in market['bars']:bar.update(up_limit=up,down_limit=down)
                scores={d:{c:1. for c in snapshot.universe[d]} for d in snapshot.sessions[snapshot.start:]}
                ir=compile_strategy(strategy_spec('bad_bounds'),scores,snapshot,digest(market['execution']))
                result=run_account(ir,market)
                self.assertEqual(result['eval_status'],'data_insufficient')
                self.assertFalse(any(o['filled'] for o in result['orders']))
                self.assertTrue(any(g['reason']=='market_rule_or_quote_invalid' for g in result['gaps']))

    def test_missing_limits_never_imply_unbounded_prices(self):
        from fuxi_v1.data import Snapshot
        for evidence in (None,'exchange_first_listing_clause'):
            with self.subTest(evidence=evidence):
                p=synthetic_campaign(25,3);s=Snapshot(p['snapshot']);market=p['market']
                market['execution']['instrument_rules']={c.replace('.XSHG','.SH'):rule_for(c) for c in s.universe[s.sessions[0]]}
                for bar in market['bars']:
                    bar.update(up_limit=None,down_limit=None,limit_status='unbounded_evidenced',limit_evidence_id=evidence)
                scores={d:{c:1. for c in s.universe[d]} for d in s.sessions[s.start:]}
                ir=compile_strategy(strategy_spec('limit',top_n=1,max_weight=1.),scores,s,digest(market['execution']))
                result=run_account(ir,market)
                self.assertEqual(result['eval_status'],'valid' if evidence else 'data_insufficient')
                self.assertEqual(any(o['filled']>0 for o in result['orders']),bool(evidence))

    def test_catalog_native_history_reads_bound_external_sources_and_all_boards(self):
        import pandas as pd
        from pathlib import Path
        from fuxi_v1.historical_adapter import build_history_input
        from fuxi_v1.market_catalog import TABLE_APIS
        from fuxi_v1.datasets import file_hash
        from fuxi_v1.pit_universe import ALL_A_FRAME_PREDICATE
        codes=['600000.SH','600001.SH','600002.SH','000001.SZ','000002.SZ','000003.SZ','002001.SZ',
               '300001.SZ','688001.SH','920001.BJ','920002.BJ']
        days=pd.bdate_range('2026-09-01',periods=15).strftime('%Y%m%d').tolist()
        iso=lambda day:day[:4]+'-'+day[4:6]+'-'+day[6:]
        with scratch_directory() as temp:
            root=Path(temp)/'catalog';raw=Path(temp)/'raw';root.mkdir();raw.mkdir()
            values={'daily':{'open':'10','high':'10.2','low':'9.8','close':'10','pre_close':'10',
                'change':None,'pct_chg':'0','vol':'100','amount':'100'},
                'adj_factor':{'adj_factor':'1'},'stk_limit':{'pre_close':'10','up_limit':'11','down_limit':'9'},
                'daily_basic':{'turnover_rate':'1','total_share':'100','float_share':'90','free_share':'70'},
                'suspend_d':{'suspend_timing':None,'suspend_type':'S'}}
            tables={};fields={};sources={}
            for filename,api in TABLE_APIS.items():
                records=[] if api=='suspend_d' else [dict(values[api],ts_code=c,trade_date=d) for d in days for c in codes]
                frame=pd.DataFrame(records,columns=['ts_code','trade_date']+list(values[api]))
                frame.to_csv(raw/filename,index=False);path=root/(api+'.parquet');frame.to_parquet(path,index=False)
                tables[api]=[{'file':path.name,'sha256':file_hash(path)}];fields[api]=','.join(frame.columns)
                sources[api]={'path':str(raw/filename),'sha256':file_hash(raw/filename)}
            manifest={'tables':tables,'fields':fields,'sources':sources,'identity_aliases':{},'queries':{},'vintage':'test_current_proxy'}
            manifest['manifest_hash']=digest(manifest);(root/'CATALOG.json').write_text(json.dumps(manifest),encoding='utf-8')
            pd.DataFrame([{'exchange':'SSE','cal_date':d,'is_open':1} for d in days]).to_csv(root/'trade_calendar.csv',index=False)
            members=[{'ts_code':c,'list_date':'20200101','delist_date':None} for c in codes]
            pd.DataFrame(members).to_csv(root/'delist_dates.csv',index=False)
            pd.DataFrame([{'ts_code':c,'name':'样例','start_date':'20200101','end_date':None,'ann_date':'20200101'} for c in codes]).to_csv(root/'namechange.csv',index=False)
            pd.DataFrame(columns=['ts_code','div_proc','ex_date']).to_csv(root/'dividend.csv',index=False)
            metadata={'manifest_hash':manifest['manifest_hash'],'files':{f:file_hash(root/f) for f in ('trade_calendar.csv','delist_dates.csv','namechange.csv','dividend.csv')}}
            metadata['metadata_hash']=digest(metadata);(root/'CATALOG_METADATA.json').write_text(json.dumps(metadata),encoding='utf-8')
            plan={'window':{'start':iso(days[0]),'end':iso(days[-1]),'calibration_end':iso(days[2]),
                'research_start':iso(days[6]),'horizon_sessions':2,'warmup_sessions':3},
                'cohort':{'members':members,'rule_kind':'all_a_share_census','selection':'census','frame_predicate':ALL_A_FRAME_PREDICATE},
                'universe':{'listing_age_calendar_days':90},'raw_sources':{name:sources[api] for name,api in TABLE_APIS.items()},
                'catalog_manifest_hash':manifest['manifest_hash'],'catalog_metadata_hash':metadata['metadata_hash']}
            result=build_history_input(root,plan,root,Path(temp)/'audit')
            self.assertEqual(result['audit']['selected_instruments'],11)
            self.assertEqual(result['audit']['eligible_min'],11)
            self.assertEqual(len(result['market']['execution']['instrument_rules']),11)
            self.assertFalse(result['market']['tradability_certified'])
            self.assertEqual(result['snapshot']['provenance']['catalog_metadata_hash'],metadata['metadata_hash'])
            self.assertEqual(result['snapshot']['columns']['turnover_rate'][0][0],.01)
            self.assertEqual(result['snapshot']['columns']['float_shares'][0][0],900000)
            self.assertEqual(result['snapshot']['fields']['turnover_rate']['unit'],'ratio')
            self.assertEqual(result['snapshot']['fields']['float_shares']['unit'],'shares')
            # The numeric audit accepts the same native row with an absent optional change.
            from fuxi_v1.panel_audit import numeric_contract
            self.assertTrue(numeric_contract('daily',pd.DataFrame([dict(values['daily'],ts_code=codes[0],trade_date=days[0])]))[2][0])
            members[0]['delist_date']=days[10]
            pd.DataFrame(members).to_csv(root/'delist_dates.csv',index=False)
            metadata['files']['delist_dates.csv']=file_hash(root/'delist_dates.csv')
            metadata['metadata_hash']=digest({k:v for k,v in metadata.items() if k!='metadata_hash'})
            (root/'CATALOG_METADATA.json').write_text(json.dumps(metadata),encoding='utf-8')
            plan['catalog_metadata_hash']=metadata['metadata_hash']
            retired=build_history_input(root,plan,root,Path(temp)/'delisting_audit')
            self.assertEqual(retired['audit']['eligible_min'],10)
            after=[r for r in retired['market']['bars'] if r['instrument']=='600000.XSHG' and r['session']>=iso(days[10])]
            self.assertTrue(after);self.assertTrue(all(not r['corporate_action_known'] for r in after))


def SnapshotForTest(payload):
    from fuxi_v1.data import Snapshot
    original=Snapshot(payload);codes=payload['universe'][payload['sessions'][0]]
    packed={'schema_version':'fuxi.historical_snapshot.1','exposure':'research_exposed','source_id':'fixture',
        'sessions':payload['sessions'],'evaluation_start':payload['sessions'][original.start],
        'instruments':codes,'universe':payload['universe'],'decision_times':payload['decision_times'],
        'fields':payload['fields'],'columns':{f:[[original.value(i,c,f,original.clocks[d]) for c in codes]
            for i,d in enumerate(original.sessions)] for f in payload['fields']},
        'barriers':[[-1]*len(codes) for d in original.sessions]}
    return HistoricalSnapshot(packed)


if __name__=='__main__':unittest.main()
