"""Run causal entry/holding/integrity checks on the concrete exclusion consumer."""
import json,sys,importlib.util
from pathlib import Path
from datetime import date,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,ContractError
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_all_a import SnapshotForTest
from fuxi_v1.portfolio import strategy_spec
from fuxi_v1.online import daily_decision
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
DELIVERY=ROOT/'deliverables/fuxi_concentration_20261004'


def main():
    module_spec=importlib.util.spec_from_file_location('reviewed_decision',DELIVERY/'decision_with_exclusions.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    p=synthetic_campaign(26,6)['snapshot'];old_days=p['sessions'];new_days=[];d=date(2026,9,1)
    while len(new_days)<len(old_days):
        if d.weekday()<5:new_days.append(d.isoformat())
        d+=timedelta(days=1)
    mapping=dict(zip(old_days,new_days));names=['001211.XSHE','688215.XSHG','688557.XSHG','920699.BJ','600077.XSHG','600000.XSHG']
    codes=dict(zip(p['universe'][old_days[0]],names))
    p['sessions']=new_days;p['evaluation_start']=mapping[p['evaluation_start']]
    p['decision_times']={d:d+'T18:30:00+08:00' for d in new_days}
    p['universe']={mapping[d]:[codes[c] for c in u] for d,u in p['universe'].items()}
    for row in p['observations']:
        row['session']=mapping[row['session']];row['available_at']=row['session']+'T18:00:00+08:00';row['instrument']=codes[row['instrument']]
    snapshot=SnapshotForTest(p);ast={'field':'close'};spec=strategy_spec('reviewed_exclusions',top_n=1,max_weight=1.,turnover_budget=1.)
    prior={'688215.XSHG':.2,'600000.XSHG':.1};checks=[]
    early=new_days[20];clock=p['decision_times'][early]
    original=daily_decision(ast,snapshot,early,clock,spec,prior=prior)
    before=module.decision_with_reviewed_exclusions(ast,snapshot,early,clock,spec,prior=prior)
    # 600077 was observed Oct3, other exclusions Oct4; this early decision must be byte-equivalent.
    require(before['decision']==original,'bundle rewrites an earlier decision');checks.append('before_observation_decision_identical')
    late=new_days[-1];clock=p['decision_times'][late]
    result=module.decision_with_reviewed_exclusions(ast,snapshot,late,clock,spec,prior=prior)
    decision=result['decision'];policy=load_json(DELIVERY/'EXCLUSIONS.json');banned={r['instrument'].split('.')[0] for r in policy['exclusions']}
    require(all(c.split('.')[0] not in banned for c in decision['scores']),'excluded new entry scored')
    require(decision['target_weights']['688215.XSHG']==.2 and '688215.XSHG' in decision['retained_excluded_holdings'],'prior holding erased')
    require(not result['live_execution_allowed'] and not result['formal_JQ_granted'],'consumer falsely permits live');checks.append('new_entries_blocked_prior_weight_retained')
    try:module.decision_with_reviewed_exclusions(ast,snapshot,late,clock,spec,exclusions=[])
    except ContractError:checks.append('policy_override_rejected')
    else:raise AssertionError('override accepted')
    value={'verified':True,'checks':checks,'fixture_scope':'synthetic observations with actual exclusion bundle and native consumer; not actual platform',
        'bundle_sha256':file_hash(DELIVERY/'EXCLUSIONS.json'),'consumer_sha256':file_hash(DELIVERY/'decision_with_exclusions.py'),
        'runner_sha256':file_hash(Path(__file__)),'sample_receipt':result,'actual_platform_runs':0}
    with (OUT/'EXCLUSION_DELIVERY_CHECK.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'verified':True,'checks':checks,'actual_platform_runs':0},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
