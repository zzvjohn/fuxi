"""Independent reconstruction of reported counts and paired daily return means."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json, require, digest
from fuxi_v1.datasets import file_hash
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.historical_study import _load_gzip
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_announcements_20261003';TEN=OUT/'ten_rounds_v4'


def main():
    final=load_json(OUT/'FINAL_VERIFICATION.json');outcome=load_json(TEN/'OUTCOME.json')
    tests=load_json(OUT/'tests_v5/TEST_RESULT.json');freeze=load_json(TEN/'FREEZE.json')
    require(final['core_hashes']==tests['code_hashes']==freeze['code_hashes']==code_manifest(),'core changed after tests/run')
    require(file_hash(OUT/'REPORT.md')==final['report_sha256']==file_hash(ROOT/'research/fuxi_v1_20261003/PAID_ANNOUNCEMENTS_AND_TEN_ROUNDS.md'),'report mirror changed')
    for name,expected in final['ledger_hashes'].items():require(file_hash(ROOT/name)==expected,'ledger changed '+name)
    require(file_hash(OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')==final['bundle_sha256'],'source bundle mismatch')
    require(file_hash(OUT/'postflight_halt_interval_688766/BUNDLE.json')==final['halt_bundle_sha256'],'halt bundle mismatch')
    probe_path=ROOT/'deliverables/fuxi_v1_20261004/PROBE_RECEIPT_v2.json'
    probe=load_json(probe_path)
    require(file_hash(probe_path)==final['manual_probe_receipt_sha256'] and file_hash(probe['path'])==final['manual_probe_sha256']==probe['sha256'],'manual probe changed')
    require(probe['actual_platform_runs']==0 and len(probe['local_fixture_checks'])==4,'manual probe scope changed')
    require(file_hash(TEN/'OUTCOME.json')==final['outcome_sha256'] and outcome['freeze_hash']==digest(freeze),'outcome mismatch')
    old=load_json(OUT/'ten_rounds_v2/OUTCOME.json')
    require(file_hash(OUT/'ten_rounds_v2/OUTCOME.json')==final['first_ten_outcome_sha256'] and old['round_count']==10,'first ten changed')
    require([r['formula_hash'] for r in old['rows']]==[r['formula_hash'] for r in outcome['rows']],'repair changed candidate identities')
    require(load_json(OUT/'ten_rounds_v2/FREEZE.json')['scenarios']==freeze['scenarios'],'repair changed cost scenarios')
    require(old['window']==outcome['window'] and old['instruments']==outcome['instruments'],'repair changed window/cohort')
    counts={'e1_pass':0,'identified_scenarios':0,'paired_identified_scenarios':0,'software_invariants_passed':0,
            'e1_and_three_scenario_increment_positive':0,'formal_JQ_qualifications':0}
    mean_checks=[];pairs=0
    for row in outcome['rows']:
        path=TEN/'rounds'/f"{row['round']:02d}.json.gz";require(file_hash(path)==row['sha256'],'round changed')
        result=_load_gzip(path)
        e1=result['evaluations'][1];require(e1['decision']==row['e1_decision'],'E1 summary mismatch')
        counts['e1_pass']+=e1['decision']=='pass';counts['software_invariants_passed']+=row['software_invariants_passed']
        counts['formal_JQ_qualifications']+=row['jq_candidate_allowed']
        require(len(result['online_parity'])==3 and result['future_invariant'],'missing native daily/future checks')
        pairs+=len(result['online_parity']);lowers=[]
        for name,_,_ in freeze['scenarios']:
            a=result['scenario_accounts'][name]['account'];b=result['scenario_baseline_accounts'][name]['account']
            identified=a['eval_status']=='valid';counts['identified_scenarios']+=identified
            require(row['account_statuses'][name]==a['eval_status'],'account summary mismatch')
            compared=result['baseline_comparisons'][name]
            paired=identified and b['eval_status']=='valid' and compared['n_pairs']>0
            counts['paired_identified_scenarios']+=paired
            if paired:
                require([d['session'] for d in a['daily']]==[d['session'] for d in b['daily']],'date mismatch')
                nav_a=[float(d['nav']) for d in a['daily']];nav_b=[float(d['nav']) for d in b['daily']]
                delta=[nav_a[i]/nav_a[i-1]-nav_b[i]/nav_b[i-1] for i in range(1,len(nav_a))]
                require(delta and len(delta)==compared['n_pairs'],'zero/wrong daily pairs')
                mean=math.fsum(delta)/len(delta)
                require(math.isclose(mean,row['paired_mean_net_increment'][name],rel_tol=1e-10,abs_tol=1e-12),'paired return mean mismatch')
                info=compared.get('metrics') or compared['diagnostics']
                require(info['inference']['lower']==row['net_increment_lower'][name],'CI summary mismatch')
                require(math.isclose(mean,info['paired_mean_increment'],rel_tol=1e-10,abs_tol=1e-12),'paired diagnostic mismatch')
                mean_checks.append({'round':row['round'],'scenario':name,'pairs':len(delta),'mean':mean})
                lowers.append(info['inference']['lower'])
            else:
                require(row['net_increment_lower'][name] is None and row['paired_mean_net_increment'][name] is None,'unidentified NAV got numeric conclusion')
                lowers.append(None)
        counts['e1_and_three_scenario_increment_positive']+=e1['decision']=='pass' and all(x is not None and x>0 for x in lowers)
    require(all(outcome[k]==v for k,v in counts.items()),'aggregate report mismatch')
    require(pairs==30 and len({r['formula_hash'] for r in outcome['rows']})==10,'wrong fixture/parity denominator')
    require(not outcome['data_formal_assurance'] and outcome['actual_platform_runs']==outcome['live_eligible']==0,'unsupported formal claim')
    policy=load_json(ROOT/'codex_automations/fuxi_runtime_policy.json')
    require(policy['current_mode']=='controlled_manual' and policy['autonomous_writes_enabled'] is False,'runtime policy changed')
    record={'verified_at':datetime.now(timezone.utc).isoformat(),'verified':True,'count_reconstruction':counts,
        'scope':'arithmetic and report reconstruction; not independent market-source assurance or investment certification',
        'daily_batch_comparisons':pairs,'paired_return_mean_checks':mean_checks,'two_sets_same_formula_window_and_costs':True,
        'software_release':outcome['readiness']['software_ready'],'paper_release':outcome['readiness']['paper_release_ready'],
        'report_sha256':final['report_sha256'],'runner_sha256':file_hash(Path(__file__)),'version':final['version']}
    with (OUT/'INDEPENDENT_REPORT_VERIFICATION.json').open('x',encoding='utf-8') as handle:json.dump(record,handle,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='paired_return_mean_checks'},ensure_ascii=False))


if __name__=='__main__':
    with controlled_writer(OUT):main()
