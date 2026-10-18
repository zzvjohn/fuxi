"""Independent artifact/count/path/source and selected fixed-seed CI verification."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,digest
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.historical_study import _load_gzip
from fuxi_v1.statistics import performance
from fuxi_v1.research_protocol import inference
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
PRIOR=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v4'


def verify(phase):
    directory=OUT/phase;result=load_json(directory/'OUTCOME.json');freeze=load_json(directory/'FREEZE.json');plan=load_json(OUT/'PLAN.json')
    require(result['arms_completed']==len(result['rows'])==8 and result['controls_completed']==len(result['controls'])==4,'arm budget/count mismatch')
    require(code_manifest()==plan['code_hashes'],'installed code differs from freeze')
    require(result['phase_freeze_hash']==digest(freeze) and freeze['plan_hash']==digest(plan),'freeze binding mismatch')
    require(file_hash(freeze['rules_path'])==freeze['rules_sha256']==result['rules_sha256'],'rules changed')
    rules=load_json(freeze['rules_path'])['rules'];terms={t['evidence_id']:t for t in load_json(OUT/'ACTION_TERM_INVENTORY.json')}
    for key,rule in rules.items():
        require(rule['term_hash']==digest(terms[key]),'rule term changed')
        source=rule['source'];require(file_hash(source['path'])==source['sha256'] and file_hash(source['text_path'])==source['text_sha256'],'source archive changed')
    request=load_json(PRIOR/'POLICY.json')['request'];rows=[];identified=0;leaves=0;replays=0;branches=0;comparisons=0;ci_checks=0
    for row in result['controls']+result['rows']:
        path=Path(row['path']);require(file_hash(path)==row['sha256'],'account artifact changed');body=_load_gzip(path)
        require(body['phase_freeze']==freeze and body['formal_JQ']==0 and body['live']==0,'account scope changed')
        require(body['ir']['strategy']['top_n']==body['baseline_ir']['strategy']['top_n']==row['top_n'],'unmatched concentration')
        require(abs(body['ir']['strategy']['max_weight']-1/row['top_n'])<1e-12,'wrong concentration weight')
        for summary in row['scenarios']:
            actual=body['scenarios'][summary['scenario']];a,b=actual['candidate'],actual['baseline'];ci=actual['comparison']
            require(summary['comparison']=={k:v for k,v in ci.items() if k!='path_pairs'},'summary CI differs')
            require(summary['candidate_leaves']==len(a['paths'])==a['leaf_count'] and summary['baseline_leaves']==len(b['paths'])==b['leaf_count'],'leaf counts differ')
            for tree in (a,b):
                leaves+=len(tree['paths']);replays+=tree['attempt_count'];branches+=len(tree['splits'])
                for leaf in tree['paths']:
                    check=leaf['reconciliation'];require(check['sessions']>0 and check['nav_reconciled'] and check['cash_and_share_reconciled'],'account replay missing')
                    require(not leaf['account']['formal_evidence'],'conditional path claims formal evidence')
                    for event in leaf['observed_choices']:
                        require(event['rule_hash']==digest(rules[event['event']]),'choice source not bound')
            if a['all_paths_identifiable'] and b['all_paths_identifiable']:
                require(ci['pairs']==len(ci['path_pairs'])==len(a['paths'])*len(b['paths'])>0,'zero/wrong path comparisons')
                require(ci['lower']==min(p['lower'] for p in ci['path_pairs']) and ci['upper']==max(p['upper'] for p in ci['path_pairs']),'outer bound differs')
                pair=ci['path_pairs'][0]
                aa=next(p['account'] for p in a['paths'] if p['path_id']==pair['candidate_path'])
                bb=next(p['account'] for p in b['paths'] if p['path_id']==pair['baseline_path'])
                ar,br=performance(aa['daily'])['returns'],performance(bb['daily'])['returns'];require(len(ar)==len(br)==pair['n_pairs']>0,'date pair count changed')
                recomputed=inference([x-y for x,y in zip(ar,br)],request,731)
                require(recomputed['lower']==pair['lower'] and recomputed['upper']==pair['upper'],'fixed CI does not reproduce')
                ci_checks+=1;comparisons+=ci['pairs']
                if not row['is_control']:identified+=1
            else:require(ci['lower'] is None and ci['pairs']==0,'unknown path given numeric performance')
        robust=all(s['comparison']['lower'] is not None and s['comparison']['lower']>0 for s in row['scenarios'])
        require(row['three_scenario_robust_positive']==robust,'robust count mismatch')
        rows.append({'name':row['name'],'top_n':row['top_n'],'sha256':row['sha256'],'verified':True})
    require(identified==result['identified_arm_scenarios'],'identified count mismatch')
    require(sum(r['three_scenario_robust_positive'] for r in result['rows'])==result['robust_positive_arms'],'positive count mismatch')
    value={'verified':True,'phase':phase,'account_artifacts':len(rows),'arm_scenarios_identified':identified,
        'scenario_role_leaves':leaves,'scenario_role_replays':replays,'scenario_role_splits':branches,
        'conditional_comparisons':comparisons,'selected_CI_rechecks':ci_checks,'source_rules_verified':len(rules),
        'outcome_sha256':file_hash(directory/'OUTCOME.json'),'runner_sha256':file_hash(Path(__file__)),
        'rows':rows,'formal_JQ':0,'actual_platform_runs':0,'live':0,'scope':'artifact arithmetic and conditional research only'}
    with (directory/'INDEPENDENT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in value.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');args=p.parse_args()
    with controlled_writer(OUT):verify(args.phase)
