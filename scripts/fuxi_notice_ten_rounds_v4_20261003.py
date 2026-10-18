"""Ten fixed all-A acceptance fixtures after public-notice supplementation.

Full-window account identification and economic diagnostics are separate. If
preflight blocks expanded search, these ten preregistered acceptance fixtures
still exercise and report the blocked path; no adaptive search is started.
"""
import argparse
import gc
import json
import sys
import time
import shutil
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,digest,ContractError
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.runtime import controlled_writer,authority_location
from fuxi_v1.historical_study import _write_gzip,_load_gzip
from fuxi_v1.search import node,factor_spec

OUT=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v4'
PARENT=ROOT/'reports/fuxi_v1_all_a_20261003'
BUNDLE=OUT.parent/'PUBLIC_IMPLEMENTATION_BUNDLE_v3.json'
SCENARIOS=[['normal',1.,1.],['double_cost',2.,1.],['half_capacity',1.,.5]]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,separators=(',',':'))


def definitions(fields):
    f=lambda name:{'field':name}
    rank=lambda x:node('rank',x)
    ret=lambda w:node('return',f('close'),window=w)
    avg=lambda name,w:node('mean',f(name),window=w)
    asts=[
        ('reversal5',rank(node('neg',ret(5)))),
        ('low_price_cv20',rank(node('neg',node('div',node('std',f('close'),window=20),avg('close',20))))),
        ('intraday_reversal',rank(node('div',node('sub',f('open'),f('close')),f('open')))),
        ('low_range',rank(node('neg',node('div',node('sub',f('high'),f('low')),f('close'))))),
        ('low_turnover_surprise',rank(node('neg',node('div',f('turnover_rate'),avg('turnover_rate',20))))),
        ('amihud_proxy_low',rank(node('neg',node('div',node('abs',ret(1)),f('amount'))))),
        ('close_location_reversal',rank(node('div',node('sub',node('sub',f('high'),f('close')),node('sub',f('close'),f('low'))),node('sub',f('high'),f('low'))))),
        ('reversal_volume5',rank(node('mul',node('neg',ret(5)),node('div',f('volume'),avg('volume',20))))),
        ('momentum60',rank(ret(60))),
        ('liquidity_adjusted_momentum20',rank(node('div',ret(20),avg('turnover_rate',20)))),
    ]
    return [dict(factor_spec(ast,fields,name),round_number=i+1,probe_name=name) for i,(name,ast) in enumerate(asts)]


def prepare():
    from fuxi_v1.pit_universe import lifecycle_from_rows,build_pit_cohort
    pointer=load_json(PARENT/'ACTIVE_CATALOG.json');parent=load_json(pointer['full_plan_path'])
    require(file_hash(pointer['full_plan_path'])==pointer['full_plan_sha256'],'parent full plan changed')
    require(file_hash(Path(pointer['path'])/'CATALOG.json')==pointer['catalog_sha256'],'parent catalog changed')
    require(BUNDLE.exists(),'public implementation bundle incomplete')
    tested=load_json(OUT.parent/'tests_v5/TEST_RESULT.json')
    require(tested['successful'] and tested['code_hashes']==code_manifest(),'freeze requires successful tests of the exact code')
    coverage=load_json(PARENT/'LOCAL_COVERAGE.json')
    sessions=[d[:4]+'-'+d[4:6]+'-'+d[6:] for d in coverage['sessions'] if '20250102'<=d<='20260930']
    master=load_json(PARENT/'SECURITY_MASTER.json')['rows'];members={r['ts_code']:r for r in master}
    codes,declaration,neutrality=build_pit_cohort(lifecycle_from_rows(master),sessions,
        salt='notice-acceptance-census-20261003',rule_kind='all_a_share_census')
    protocol=load_json(ROOT/'reports/fuxi_v1_exclusion_20260930/PLAN.json')['protocol']
    plan=dict(parent,batch='paid_notice_ten_fixed_acceptance_rounds_20261003',
        window={'start':sessions[0],'end':sessions[-1],'calibration_end':'2025-05-30',
                'research_start':'2025-06-11','horizon_sessions':5,'warmup_sessions':60},
        cohort=dict(declaration,selection=declaration['rule'],members=[members[c] for c in codes]),
        neutrality=neutrality,protocol=protocol,reevaluate_formula_count=10,
        search={'max_candidate_requests':0,'external_LLM_calls':0},
        fixture_budget=10,expanded_search_requires_preflight=True,
        data_validation_scope='source-bound ordinary terms and all-census account diagnostics; existing false formal quality flags retained; independent assurance not asserted',
        issuer_action_evidence={'path':str(BUNDLE),'sha256':file_hash(BUNDLE)},
        issuer_halt_evidence={'path':str(OUT.parent/'postflight_halt_interval_688766/BUNDLE.json'),
                              'sha256':file_hash(OUT.parent/'postflight_halt_interval_688766/BUNDLE.json')},
        scope='all ordinary A-share lifecycle members in frozen window; ten fixed acceptance fixtures; research exposed; no adaptive search or unseen holdout',
        parent_full_history_plan_sha256=file_hash(pointer['full_plan_path']))
    fields={name:{'unit':'shares' if name=='volume' or name.endswith('_shares') else 'currency' if name=='amount' else 'ratio' if name=='turnover_rate' else 'price'}
        for name in ('open','high','low','close','volume','amount','turnover_rate','total_shares','float_shares','free_shares')}
    candidates=definitions(fields)
    frozen={'plan_hash':digest(plan),'candidate_hash':digest(candidates),'catalog':pointer,
            'code_hashes':code_manifest(),'runner_sha256':file_hash(Path(__file__)),'scenarios':SCENARIOS,
            'authority_path':str(authority_location(OUT)),
            'ten_unique_formula_hashes':len({c['formula_hash'] for c in candidates}),
            'online_parity_sessions_per_round':3,'all_attempts_retained':True}
    if (OUT/'FREEZE.json').exists():
        require(load_json(OUT/'FREEZE.json')==frozen,'ten-round freeze changed')
    else:
        save(OUT/'PLAN.json',plan);save(OUT/'CANDIDATES.json',candidates);save(OUT/'FREEZE.json',frozen)
        for directory in (ROOT/'fuxi_v1',ROOT/'fuxi_v1/tests'):
            dest=OUT/('code/tests' if directory.name=='tests' else 'code');dest.mkdir(parents=True,exist_ok=True)
            for path in directory.glob('*.py'):shutil.copy2(path,dest/path.name)
        shutil.copy2(Path(__file__),OUT/'runner.py')
    print(json.dumps({'prepared':True,'instruments':len(codes),'sessions':len(sessions),
        'research_sessions':sum(d>=plan['window']['research_start'] for d in sessions),
        'retained_window_delistings':neutrality['cohort_names_delisting_inside_window'],'formula_count':len(candidates)},ensure_ascii=False),flush=True)


def check_freeze():
    frozen=load_json(OUT/'FREEZE.json');require(code_manifest()==frozen['code_hashes'],'frozen core changed')
    require(file_hash(Path(__file__))==frozen['runner_sha256'],'frozen runner changed')
    plan=load_json(OUT/'PLAN.json');require(digest(plan)==frozen['plan_hash'],'frozen plan changed')
    return frozen,plan


def snapshot():
    from fuxi_v1.historical_adapter import build_history_input
    frozen,plan=check_freeze();path=OUT/'market_input.json'
    if path.exists():
        receipt=load_json(OUT/'INPUT_RECEIPT.json');require(file_hash(path)==receipt['sha256'],'history input changed');return
    source=frozen['catalog']['path'];started=time.monotonic()
    payload=build_history_input(source,plan,source,OUT/'input_audit')
    save(path,payload);save(OUT/'INPUT_RECEIPT.json',{'path':str(path),'sha256':file_hash(path),
        'bytes':path.stat().st_size,'elapsed_seconds':time.monotonic()-started,'audit':payload['audit'],'freeze_hash':digest(frozen)})
    print(json.dumps({'snapshot_completed':True,'bytes':path.stat().st_size,'audit':payload['audit']},ensure_ascii=False),flush=True)


def preflight():
    from fuxi_v1.remediation import preflight as account_preflight
    check_freeze()
    if (OUT/'preflight/DECISION.json').exists():return
    decision=account_preflight(OUT/'market_input.json',OUT/'PLAN.json',OUT/'preflight')
    print(json.dumps({k:decision[k] for k in ('probe_count','search_allowed','formal_account_evidence_certified')},ensure_ascii=False),flush=True)


def rounds():
    from fuxi_v1.historical_adapter import HistoricalSnapshot
    from fuxi_v1.data import Labels
    from fuxi_v1.dsl import evaluate_ast
    from fuxi_v1.evaluation import factor_statistics,DEFAULT_POLICY
    from fuxi_v1.research_protocol import policy_from_plan
    from fuxi_v1.research_evaluation import evaluate_candidate,paired_accounts
    from fuxi_v1.simulation import run_account
    from fuxi_v1.action_accounting import reconcile_account
    from fuxi_v1.evidence import EvidenceStore
    from fuxi_v1.acceptance import AcceptanceService
    from fuxi_v1.online import daily_decision
    from fuxi_v1.jq_strategy import compile_jq_strategy
    from fuxi_v1.tests.test_engineering import fault_checks
    from fuxi_v1.readiness import assess
    frozen,plan=check_freeze();candidates=load_json(OUT/'CANDIDATES.json')
    require(len(candidates)==10 and digest(candidates)==frozen['candidate_hash'],'candidate fixture mismatch')
    receipt=load_json(OUT/'INPUT_RECEIPT.json');require(file_hash(OUT/'market_input.json')==receipt['sha256'],'input changed')
    gate=load_json(OUT/'preflight/DECISION.json');require(gate['probe_count']==12,'missing preflight')
    payload=load_json(OUT/'market_input.json');s=HistoricalSnapshot(payload['snapshot']);labels=Labels(payload['labels'],s);market=payload['market']
    base_ast=node('rank',node('return',{'field':'close'},window=20));base=evaluate_ast(base_ast,s)
    cal=factor_statistics({d:v for d,v in base.items() if d<=plan['window']['calibration_end']},s,labels,DEFAULT_POLICY,
        plan['window']['calibration_end']+'T23:59:59+08:00')
    policy=policy_from_plan(plan,s,cal);request=policy['request']
    if (OUT/'POLICY.json').exists():require(load_json(OUT/'POLICY.json')==policy,'calibration changed')
    else:save(OUT/'POLICY.json',policy)
    from fuxi_v1.datasets import scope_of
    authority=EvidenceStore(authority_location(OUT))
    authority.expose('notice_acceptance_'+digest(frozen),scope_of(s),'research');authority.close()
    store=EvidenceStore(OUT/'evidence.sqlite');baselines={};rows=[]
    for candidate in candidates:
        number=candidate['round_number'];path=OUT/'rounds'/('%02d.json.gz'%number);meta=OUT/'rounds'/('%02d.receipt.json'%number)
        if meta.exists():
            row=load_json(meta);require(file_hash(path)==row['sha256'] and row['freeze_hash']==digest(frozen),'round cache changed');rows.append(row);continue
        require(not path.exists(),'unsealed partial round requires recorded recovery')
        started=time.monotonic();scores=evaluate_ast(candidate['canonical_ast'],s)
        result=evaluate_candidate(candidate,scores,base,s,labels,market,policy,payload['research_clock'],top_n=10,baseline_cache=baselines)
        ir=result['strategy_ir'];accounts={};comparisons={};baseline_accounts={}
        for name,cost,cap in SCENARIOS:
            account=result['candidate_account'] if name=='normal' else run_account(ir,market,start=request['research_start'],end=request['research_end'],cost_multiplier=cost,participation_multiplier=cap)
            replay=reconcile_account(account,market['execution']['initial_cash'],market)
            if name not in baselines:
                baselines[name]=result['baseline_account'] if name=='normal' else run_account(result['baseline_ir'],market,start=request['research_start'],end=request['research_end'],cost_multiplier=cost,participation_multiplier=cap)
            comparisons[name]=paired_accounts(account,baselines[name],request)
            accounts[name]={'account':account,'reconciliation':replay}
            baseline_accounts[name]={'account':baselines[name],
                'reconciliation':reconcile_account(baselines[name],market['execution']['initial_cash'],market)}
        decisions=[d for d in ir['decisions'] if request['research_start']<=d['signal_session']<=request['research_end']]
        require(len(decisions)>=3,'insufficient daily/batch comparisons')
        picked=[decisions[i] for i in (0,len(decisions)//2,len(decisions)-1)];parity=[]
        for decision in picked:
            day=decision['signal_session'];online=daily_decision(candidate['canonical_ast'],s,day,s.payload['decision_times'][day],ir['strategy'],baseline_ast=base_ast)
            require(online['scores']==decision['scores'],'daily/batch score mismatch')
            parity.append({'session':day,'decision_id':online['decision_id'],'instruments':len(online['scores']),'degenerate':online['ranking_degenerate']})
        day=picked[1]['signal_session'];index=s.sessions.index(day)
        before=daily_decision(candidate['canonical_ast'],s,day,s.payload['decision_times'][day],ir['strategy'],baseline_ast=base_ast)
        tails={k:v[index+1:] for k,v in s.payload['columns'].items()}
        try:
            for k,v in s.payload['columns'].items():v[index+1:]=[[None]*len(s.payload['instruments']) for _ in tails[k]]
            after=daily_decision(candidate['canonical_ast'],s,day,s.payload['decision_times'][day],ir['strategy'],baseline_ast=base_ast)
            require(before==after,'future perturbation altered current signal')
        finally:
            for k,v in tails.items():s.payload['columns'][k][index+1:]=v
        faults=fault_checks(OUT/'rounds'/('%02d_fixture.sqlite'%number))
        event=store.append('evaluation',result);admission=AcceptanceService(store).research(event)
        jq={'status':'not_attempted','observed_on_platform':False}
        try:
            code,jqmeta=compile_jq_strategy(ir,market)
            jq.update(status='diagnostic_code_compiled',**jqmeta)
            diagnostic=OUT/'rounds'/('%02d_jq_diagnostic.py'%number)
            with diagnostic.open('xb') as handle:handle.write(code)
        except ValueError as exc:
            jq.update(status='delivery_blocked',reason=str(exc))
        result.update(scenario_accounts=accounts,scenario_baseline_accounts=baseline_accounts,baseline_comparisons=comparisons,online_parity=parity,
            future_invariant=True,execution_fault_checks=faults,admission=admission,jq_delivery=jq)
        _write_gzip(path,result)
        e1=result['evaluations'][1]
        bounds={name:(value.get('metrics') or value.get('diagnostics') or {}).get('inference',{}).get('lower') for name,value in comparisons.items()}
        means={name:(value.get('metrics') or value.get('diagnostics') or {}).get('paired_mean_increment') for name,value in comparisons.items()}
        account_statuses={name:v['account']['eval_status'] for name,v in accounts.items()}
        row={'round':number,'name':candidate['probe_name'],'candidate_id':candidate['candidate_id'],'formula_hash':candidate['formula_hash'],
            'sha256':file_hash(path),'freeze_hash':digest(frozen),'elapsed_seconds':time.monotonic()-started,
            'stage_readouts':[{k:e.get(k) for k in ('stage','eval_status','decision','reason_code','reject_class')} for e in result['evaluations']],
            'e1_status':e1['eval_status'],'e1_decision':e1['decision'],'rank_ic':e1['diagnostics']['spearman']['mean'],
            'valid_ic_sections':e1['diagnostics']['valid_section_count'],'pair_coverage':e1['diagnostics'].get('research_pair_coverage'),
            'block_grid_lower':[x.get('lower') for x in e1['diagnostics'].get('block_sensitivity',[])],
            'positive_year_fraction':e1['diagnostics'].get('positive_year_fraction'),
            'account_statuses':account_statuses,'identified_scenarios':sum(v=='valid' for v in account_statuses.values()),
            'baseline_account_statuses':{name:baselines[name]['eval_status'] for name,_,_ in SCENARIOS},
            'paired_identified_scenarios':sum(v['n_pairs']>0 for v in comparisons.values()),
            'account_gaps':{name:dict(Counter(g['reason'] for g in a['account']['gaps'])) for name,a in accounts.items()},
            'baseline_account_gaps':{name:dict(Counter(g['reason'] for g in a['account']['gaps'])) for name,a in baseline_accounts.items()},
            'root_gap_instruments':sorted({g['instrument'] for a in list(accounts.values())+list(baseline_accounts.values()) for g in a['account']['gaps'] if 'instrument' in g}),
            'net_increment_lower':bounds,'paired_mean_net_increment':means,
            'three_scenario_lower_positive':all(v is not None and v>0 for v in bounds.values()),
            'software_invariants_passed':all(faults['checks'].values()) and len(parity)==3,
            'daily_batch_comparisons':len(parity),'jq_candidate_allowed':admission['jq_candidate_allowed'],
            'admission_blockers':admission['blockers'],'jq_delivery':jq,
            'expanded_search_started':False,'preflight_search_allowed':gate['search_allowed']}
        save(meta,row);rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        keep={digest(base_ast),digest(node('div',node('std',{'field':'close'},window=20),node('mean',{'field':'close'},window=20)))}
        s._score_cache={k:v for k,v in s._score_cache.items() if k in keep};gc.collect()
    require(len(rows)==10 and len({r['formula_hash'] for r in rows})==10,'ten distinct rounds incomplete')
    software_checks={str(r['round'])+'_native_invariants':r['software_invariants_passed'] for r in rows}
    software_checks.update({str(r['round'])+'_JQ_delivery':r['jq_delivery']['status']=='diagnostic_code_compiled' for r in rows})
    software_checks['manual_forward_simulation_adapter_implemented']=False
    readiness=assess(store,software_checks)
    outcome={'freeze_hash':digest(frozen),'rows':rows,'round_count':10,'scope':plan['scope'],'readiness':readiness,
        'instruments':len(plan['cohort']['members']),'window':plan['window'],
        'preflight_search_allowed':gate['search_allowed'],'expanded_search_started':False,
        'software_invariants_passed':sum(r['software_invariants_passed'] for r in rows),
        'identified_scenarios':sum(r['identified_scenarios'] for r in rows),
        'paired_identified_scenarios':sum(r['paired_identified_scenarios'] for r in rows),
        'e1_pass':sum(r['e1_decision']=='pass' for r in rows),
        'economic_identified_rounds':sum(r['paired_identified_scenarios']==3 for r in rows),
        'e1_and_three_scenario_increment_positive':sum(r['e1_decision']=='pass' and r['three_scenario_lower_positive'] for r in rows),
        'formal_JQ_qualifications':sum(r['jq_candidate_allowed'] for r in rows),
        'actual_platform_runs':0,'live_eligible':0,'data_formal_assurance':False}
    target=OUT/'OUTCOME.json'
    if target.exists():require(load_json(target)==outcome,'sealed outcome changed')
    else:save(target,outcome)
    store.close();print(json.dumps({k:v for k,v in outcome.items() if k!='rows'},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','snapshot','preflight','rounds'))
    args=parser.parse_args()
    with controlled_writer(OUT):globals()[args.action]()


if __name__=='__main__':main()
