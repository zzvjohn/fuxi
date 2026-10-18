"""Eight frozen concentration arms, conditional accounts and honest data-gap outcomes."""
import argparse,sys,json,time,gc,shutil
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.historical_study import _load_gzip,_write_gzip
from fuxi_v1.simulation import ResearchAccountContext
from fuxi_v1.fractional_branches import enumerate_accounts,compare_branches
from fuxi_v1.action_accounting import reconcile_account
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
PRIOR=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v4'
SCENARIOS=[['normal',1.,1.],['double_cost',2.,1.],['half_capacity',1.,.5]]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def freeze():
    old=load_json(PRIOR/'OUTCOME.json');candidates=[r for r in old['rows'] if r['e1_decision']=='pass']
    require(len(candidates)==4,'candidate basis changed')
    rules=OUT/'FRACTIONAL_RULES_v2.json';tested=load_json(OUT/'tests/TEST_RESULT.json')
    require(tested['successful'] and tested['code_hashes']==code_manifest(),'tested code differs')
    plan={'schema_version':'fuxi.concentration_plan.1','registered_at':datetime.now(timezone.utc).isoformat(),
        'basis_outcome_sha256':file_hash(PRIOR/'OUTCOME.json'),'input_path':str(PRIOR/'market_input.json'),
        'input_sha256':load_json(PRIOR/'INPUT_RECEIPT.json')['sha256'],
        'rules_path':str(rules),'rules_sha256':file_hash(rules),'code_hashes':code_manifest(),
        'runner_sha256':file_hash(Path(__file__)),'scenarios':SCENARIOS,'window':old['window'],
        'arms':[{'name':r['name'],'round':r['round'],'formula_hash':r['formula_hash'],'old_record_sha256':r['sha256'],
                 'top_n':n,'max_weight':1/n} for r in candidates for n in (30,50)],
        'control_top_n':10,'max_new_portfolio_arms':8,'max_paths_per_role_scenario':128,
        'external_LLM_requests':0,'threshold_relaxation_allowed':False,'source_assurance_granted':False,
        'input_is_research_exposed':True,'formal_JQ':0,'live':0,
        'source_retry_budget':'at most one source-only freeze revision; identities/window/costs retained; repeated executions not new arms',
        'fallback_policy':'if acquired primary sources still fail, prospective entry exclusions; optional separate posthoc diagnostic book excludes data-gap union at most 20 instruments, never edits original accounts',
        'comparison':'candidate/baseline matched concentration; full source-supported conditional path Cartesian outer bound',
        'prior_E1':'same full-universe formulas; old screening readouts retained as hypothesis basis; not a new independent confirmation'}
    require(len(plan['arms'])==8,'wrong arm budget')
    save(OUT/'PLAN.json',plan)
    code=OUT/'code';code.mkdir()
    for name in plan['code_hashes']:shutil.copyfile(ROOT/'fuxi_v1'/name,code/name)
    shutil.copyfile(Path(__file__),OUT/'runner.py')
    print(json.dumps({'frozen':True,'arms':8,'formula_names':[r['name'] for r in candidates]},ensure_ascii=False),flush=True)


def resize(ir,n,excluded=()):
    from copy import deepcopy
    value=deepcopy(ir);value['strategy'].update(top_n=n,max_weight=1/n)
    require({k for k in value['strategy'] if value['strategy'][k]!=ir['strategy'][k]}<= {'top_n','max_weight'},'more than concentration axis changed')
    if excluded:
        banned=set(excluded)
        for d in value['decisions']:
            d['scores']={s:v for s,v in d['scores'].items() if s not in banned}
            d['universe']=[s for s in d['universe'] if s not in banned]
            require(d['universe'],'empty filtered diagnostic universe')
    value['ir_hash']=digest({k:v for k,v in value.items() if k!='ir_hash'});return value


def execute(phase='attempt1',rule_path=None,excluded=()):
    plan=load_json(OUT/'PLAN.json');require(code_manifest()==plan['code_hashes'],'code changed after freeze')
    require(file_hash(Path(__file__))==plan['runner_sha256'],'runner changed after freeze')
    directory=OUT/phase;directory.mkdir(parents=True,exist_ok=True)
    rule_path=Path(rule_path or plan['rules_path']);rules_receipt=load_json(rule_path);rules=rules_receipt['rules']
    phase_plan={'plan_hash':digest(plan),'rules_path':str(rule_path),'rules_sha256':file_hash(rule_path),
        'excluded':sorted(excluded),'posthoc_diagnostic_only':bool(excluded),'formal_JQ':0,'live':0}
    if (directory/'FREEZE.json').exists():require(load_json(directory/'FREEZE.json')==phase_plan,'phase changed')
    else:save(directory/'FREEZE.json',phase_plan)
    require(file_hash(plan['input_path'])==plan['input_sha256'],'historical input changed')
    payload=load_json(plan['input_path']);market=payload['market'];del payload;gc.collect()
    context=ResearchAccountContext(market);request=load_json(PRIOR/'POLICY.json')['request']
    baselines={};rows=[];focus={};root_gaps=[];controls=[]
    def account(ir,n,name,cost,capacity,role):
        key=digest([ir['ir_hash'],cost,capacity])
        if role=='baseline' and key in baselines:return baselines[key]
        result=enumerate_accounts(ir,market,rules,max_paths=128,research_context=context,
            start=request['research_start'],end=request['research_end'],cost_multiplier=cost,participation_multiplier=capacity)
        # Every completed path gets a separate cash/share/NAV arithmetic replay.
        for leaf in result['paths']:
            leaf['reconciliation']=reconcile_account(leaf['account'],market['execution']['initial_cash'],market)
        if role=='baseline':baselines[key]=result
        return result
    def trial(arm,control=False):
        started=time.monotonic();old=_load_gzip(PRIOR/'rounds'/f"{arm['round']:02d}.json.gz")
        require(file_hash(PRIOR/'rounds'/f"{arm['round']:02d}.json.gz")==arm['old_record_sha256'],'old round changed')
        n=10 if control else arm['top_n'];identity=arm['name']+'_'+str(n);path=directory/'accounts'/(identity+'.json.gz')
        receipt=directory/'accounts'/(identity+'.receipt.json')
        if receipt.exists():
            value=load_json(receipt);require(file_hash(path)==value['sha256'],'trial cache changed');return value
        require(not path.exists(),'partial trial requires recorded recovery')
        ir=resize(old['strategy_ir'],n,excluded);base=resize(old['baseline_ir'],n,excluded)
        full={};summary=[]
        for scenario,cost,capacity in SCENARIOS:
            a=account(ir,n,scenario,cost,capacity,'candidate');b=account(base,n,scenario,cost,capacity,'baseline')
            comparison=compare_branches(a,b,request)
            full[scenario]={'candidate':a,'baseline':b,'comparison':comparison}
            gaps=[]
            for role,result in [('candidate',a),('baseline',b)]:
                for leaf in result['paths']:
                    for gap in leaf['account']['gaps']:
                        if gap.get('instrument'):gaps.append(dict(gap,role=role,path_id=leaf['path_id']))
            summary.append({'scenario':scenario,'candidate_leaves':a['leaf_count'],'baseline_leaves':b['leaf_count'],
                'candidate_complete':a['enumeration_complete'],'baseline_complete':b['enumeration_complete'],
                'candidate_identifiable':a['all_paths_identifiable'],'baseline_identifiable':b['all_paths_identifiable'],
                'comparison':{k:v for k,v in comparison.items() if k!='path_pairs'},
                'gap_counts':dict(Counter(g['reason'] for g in gaps)),
                'root_gaps':list({digest(g):g for g in gaps}.values())})
        _write_gzip(path,{'identity':identity,'ir':ir,'baseline_ir':base,'phase_freeze':phase_plan,'scenarios':full,
            'prior_E1':old['evaluations'][1],'prior_E2':old['evaluations'][2],'formal_JQ':0,'live':0})
        result={'name':arm['name'],'top_n':n,'is_control':control,'formula_hash':arm['formula_hash'],
            'scenarios':summary,'sha256':file_hash(path),'path':str(path),'formal_JQ':0,'live':0,
            'three_scenario_robust_positive':all(s['comparison']['lower'] is not None and s['comparison']['lower']>0 for s in summary),
            'elapsed_seconds':time.monotonic()-started,'posthoc_diagnostic_only':bool(excluded)}
        save(receipt,result);print(json.dumps({k:v for k,v in result.items() if k!='scenarios'},ensure_ascii=False),flush=True)
        return result
    for arm in plan['arms']:
        if arm['name'] not in focus:focus[arm['name']]=arm
    for arm in focus.values():controls.append(trial(arm,True))
    for arm in plan['arms']:rows.append(trial(arm))
    for r in controls+rows:
        for s in r['scenarios']:
            root_gaps.extend(dict(g,name=r['name'],top_n=r['top_n'],scenario=s['scenario']) for g in s['root_gaps'])
    outcome={'phase':phase,'arms_completed':len(rows),'controls_completed':len(controls),'rows':rows,'controls':controls,
        'identified_arm_scenarios':sum(s['candidate_identifiable'] and s['baseline_identifiable'] for r in rows for s in r['scenarios']),
        'robust_positive_arms':sum(r['three_scenario_robust_positive'] for r in rows),'root_gaps':root_gaps,
        'root_instruments':sorted({g['instrument'] for g in root_gaps}),'rules_sha256':file_hash(rule_path),
        'formal_JQ':0,'live':0,'posthoc_diagnostic_only':bool(excluded),'excluded':sorted(excluded),
        'research_exposed':True,'actual_platform_runs':0,'phase_freeze_hash':digest(phase_plan)}
    save(directory/'OUTCOME.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ('rows','controls','root_gaps')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--phase',default='attempt1')
    parser.add_argument('--rules');parser.add_argument('--exclude',nargs='*',default=[]);args=parser.parse_args()
    with controlled_writer(OUT):
        if args.action=='freeze':freeze()
        else:execute(args.phase,args.rules,args.exclude)


if __name__=='__main__':main()
