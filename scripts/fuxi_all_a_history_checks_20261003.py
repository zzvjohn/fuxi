"""Ten exhaustive real-vendor-history account diagnostics, never investment admission."""
from pathlib import Path
import sys
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import digest,require
from fuxi_v1.datasets import file_hash,execution_profile
from fuxi_v1.a_share import rule_for
from fuxi_v1.history import jq_code
from fuxi_v1.portfolio import strategy_spec
from fuxi_v1.simulation import run_account
from fuxi_v1.action_accounting import reconcile_account
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_all_a_20261003'


def run():
    import numpy as np
    import pandas as pd
    from fuxi_v1.pit_universe import canonical_code
    coverage=json.loads((OUT/'LOCAL_COVERAGE.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    require(all(file_hash(ROOT/'fuxi_v1'/name)==expected for name,expected in audit['code_hashes'].items()),'audited code changed')
    from fuxi_all_a_20261003 import active_catalog
    manifest=json.loads((active_catalog()[0]/'CATALOG.json').read_text(encoding='utf-8'))
    require(manifest['manifest_hash']==audit['manifest_hash'],'history/audit catalog mismatch')
    expected=audit['artifact_hashes']['AUDITED_PANELS.npz']
    require(file_hash(OUT/'AUDITED_PANELS.npz')==expected,'audited panels changed')
    # NPZ indexing inflates an entire array on every access. Materialize once;
    # per-stock/per-bar indexing must not decompress the census hundreds of thousands of times.
    keys=['active','dense_core_complete','event_known_vendor_proxy','halt','partial_halt',
          'vol','open','high','low','close','up_limit','down_limit']
    with np.load(OUT/'AUDITED_PANELS.npz') as archive:panels={key:archive[key] for key in keys}
    codes=coverage['codes'];sessions=coverage['sessions']
    eligible=(panels['active'] & panels['dense_core_complete'] & panels['event_known_vendor_proxy'] &
              ~panels['halt'] & ~panels['partial_halt'] & (panels['vol']>0))
    # These short diagnostics start in cash and do not claim event settlement.
    # A supported action must not be silently omitted from a chosen price window.
    term_path=OUT/'ACTION_TERMS.csv'
    require(file_hash(term_path)==audit['artifact_hashes']['ACTION_TERMS.csv'],'action terms changed')
    ci={c:i for i,c in enumerate(codes)};di={d:i for i,d in enumerate(sessions)}
    for term in pd.read_csv(term_path).to_dict('records'):
        code=canonical_code(term['instrument']);day=term['ex_date'].replace('-','')
        if code in ci and day in di:eligible[di[day],ci[code]]=False
    triples=eligible[:-2] & eligible[1:-1] & eligible[2:]
    candidates=[np.flatnonzero(triples[:,j]) for j in range(len(codes))]
    source=digest({'audit':file_hash(OUT/'DATA_AUDIT.json'),'panels':expected,
        'runner_sha256':file_hash(Path(__file__)),
        'code_hashes':{p.name:file_hash(p) for p in (ROOT/'fuxi_v1').glob('*.py')},
        'rules_hash':digest({c:rule_for(c) for c in codes})})
    rounds=[]
    for number in range(1,11):
        path=OUT/('HISTORY_ACCOUNT_ROUND_%02d.json'%number)
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'));require(saved['source_hash']==source and saved['rows_hash']==digest(saved['rows']),'history round resume mismatch')
            rounds.append(saved['summary']);continue
        rows=[];counts=Counter();fills=0
        for j,code in enumerate(codes):
            options=candidates[j]
            if not len(options):
                rows.append({'ts_code':code,'status':'no_complete_three_session_market_window','formal_evidence':False})
                counts['data_insufficient']+=1;continue
            # Fixed coverage stratum, no return/IC/PNL ranking. This is a diagnostic
            # existence probe, not an outcome-independent research sample.
            first=int(options[(number-1)*(len(options)-1)//9]);indices=range(first,first+3)
            dates=[sessions[i][:4]+'-'+sessions[i][4:6]+'-'+sessions[i][6:] for i in indices]
            instrument=jq_code(code);bars=[]
            for i,day in zip(indices,dates):
                bars.append({'session':day,'instrument':instrument,
                    **{f:float(panels[f][i,j]) for f in ('open','high','low','close','up_limit','down_limit')},
                    'volume':float(panels['vol'][i,j])*100,'paused':False,'corporate_action_known':True,
                    'evidence_id':digest([source,code,day]),'event_coverage':'vendor_daily_reference_proxy',
                    'mark_source':'observed_close'})
            execution=execution_profile();execution.update(instrument_rules={code:rule_for(code)},odd_lot_exit='whole_residual',
                order_type='market',
                identity_aliases=manifest.get('identity_aliases',{}),
                sell_tax_schedule=[['0001-01-01','0.001'],['2023-08-28','0.0005']])
            market={'sessions':dates,'bars':bars,'actions':[],'execution':execution,
                    'event_coverage_certified':False,'tradability_certified':False}
            market['market_hash']=digest(market)
            ir={'schema_version':'fuxi.strategy_ir.1','strategy':strategy_spec('full_census_diagnostic',top_n=1,max_weight=1.),
                'snapshot_hash':digest([source,code,dates]),'execution_hash':digest(execution),'model_hash':None,'score_origin':'factor',
                'decisions':[{'decision_time':dates[0]+'T18:30:00+08:00','signal_session':dates[0],
                    'execution_session':dates[1],'first_order_time':dates[1]+'T09:30:00+08:00',
                    'scores':{instrument:1.},'universe':[instrument]}]}
            ir['ir_hash']=digest(ir)
            cost_multiplier=2. if number%3==2 else 1.
            participation_multiplier=.5 if number%3==0 else 1.
            result=run_account(ir,market,cost_multiplier=cost_multiplier,participation_multiplier=participation_multiplier)
            reconcile=reconcile_account(result,execution['initial_cash'],market)
            require(not result['formal_evidence'],'diagnostic account cannot become formal')
            require(reconcile['cash_and_share_reconciled'] and reconcile['nav_reconciled'],'native account mismatch')
            counts[result['eval_status']]+=1;filled=sum(o['filled']>0 for o in result['orders']);fills+=filled
            rows.append({'ts_code':code,'status':result['eval_status'],'window':dates,'filled_orders':filled,
                'order_reject_reasons':dict(Counter(o['reason'] for o in result['orders'] if o['reason'])),
                'gaps':result['gaps'],'journal_hash':digest(result['journal']),'account_hash':digest(result),
                'cost_multiplier':cost_multiplier,'participation_multiplier':participation_multiplier,
                'reconciliation':reconcile,'formal_evidence':False})
        require(len(rows)==len(codes) and counts['valid']+counts['data_insufficient']==len(codes),'silent cohort omission')
        require(counts['valid']>0 and fills>0,'zero actual accounts or fills')
        summary={'round':number,'instrument_count':len(codes),'status_counts':dict(counts),'filled_orders':fills,
            'formal_accounts':0,'all_instruments_have_diagnostic_window':counts['data_insufficient']==0}
        body={'source_hash':source,'summary':summary,'rows':rows,'rows_hash':digest(rows),
              'scope':'three-session event-free actual-vendor-history account arithmetic; current market-order quantity rules and declared daily proxies',
              'selection':'fixed time strata of data-complete event-free windows; diagnostic only; no formal factor or account admission'}
        with path.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False)
        rounds.append(summary);print(json.dumps(summary),flush=True)
    result={'schema_version':'fuxi.full_census_history_checks.1','rounds':rounds,'source_hash':source,
        'runner_sha256':file_hash(Path(__file__)),
        'instrument_rounds':sum(r['instrument_count'] for r in rounds),
        'formal_accounts':0,'actual_platform_runs':0,'live_orders':0,
        'round_artifact_hashes':{str(n):file_hash(OUT/('HISTORY_ACCOUNT_ROUND_%02d.json'%n)) for n in range(1,11)}}
    path=OUT/'TEN_ROUND_HISTORY_CHECKS.json'
    with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)


if __name__=='__main__':
    with controlled_writer(OUT):run()
