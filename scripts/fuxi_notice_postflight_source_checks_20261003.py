"""Bounded paid/vendor source checks for exact completed-round account gaps.

Supplemental evidence never replaces the frozen ten-round metrics or authorises
retrospective exclusion. Vendor observations remain current-vintage research.
"""
import argparse
from collections import Counter
from datetime import date,timedelta,datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,digest
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.historical_study import _load_gzip
from fuxi_v1.history import jq_code
from fuxi_v1.tushare_source import TushareSource,VendorFailure
from scripts.fuxi_all_a_20261003 import FIELDS
TEN=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v2'
OUT=TEN.parent/'postflight_sources'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,ensure_ascii=False,indent=2)


def freeze():
    outcome=load_json(TEN/'OUTCOME.json');require(outcome['round_count']==10,'source follow-up requires completed ten rounds')
    if (OUT/'PLAN.json').exists():return load_json(OUT/'PLAN.json')
    gaps={}
    for row in outcome['rows']:
        result=_load_gzip(TEN/'rounds'/('%02d.json.gz'%row['round']))
        for key in ('scenario_accounts','scenario_baseline_accounts'):
            for scenario,value in result[key].items():
                for gap in value['account']['gaps']:
                    if gap.get('instrument'):gaps[digest(gap)]=gap
    gate=load_json(TEN/'preflight/DECISION.json')
    for r in gate['readouts']:
        for gap in r['root_gaps']:
            if gap.get('instrument'):gaps[digest(gap)]=gap
    observed=sorted(gaps.values(),key=lambda r:(r['session'],r['instrument'],r['reason']))
    mapping_path=Path(load_json(TEN/'FREEZE.json')['catalog']['path'])/'CATALOG.json'
    aliases=load_json(mapping_path).get('identity_aliases',{})
    requests={}
    for g in observed:
        if g['reason'] not in ('corporate_action_coverage_unknown','market_rule_or_quote_missing','market_rule_or_quote_invalid'):continue
        code=g['instrument'].replace('.XSHG','.SH').replace('.XSHE','.SZ')
        day=date.fromisoformat(g['session']);start=(day-timedelta(days=15)).strftime('%Y%m%d')
        end=min(date(2026,9,30),day+timedelta(days=15)).strftime('%Y%m%d')
        for api in ('daily','adj_factor','stk_factor','stk_limit'):
            req={'api':api,'params':{'ts_code':code,'start_date':start,'end_date':end},'fields':FIELDS[api],
                'root_date':g['session'],'root_instrument':g['instrument']}
            requests[digest(req)]=req
        req={'api':'anns_d','params':{'ts_code':code,'start_date':start,'end_date':end},
             'fields':'ann_date,ts_code,name,title,url,rec_time','root_date':g['session'],'root_instrument':g['instrument']}
        requests[digest(req)]=req
    original_requests=list(requests.values())
    for req in original_requests:
        stable=req['params']['ts_code']
        for old,row in aliases.items():
            if row['stable_code']==stable:
                extra=dict(req,params=dict(req['params'],ts_code=old))
                requests[digest(extra)]=extra
    plan={'mapping_path':str(mapping_path),'mapping_sha256':file_hash(mapping_path),'frozen_at':datetime.now(timezone.utc).isoformat(),'outcome_sha256':file_hash(TEN/'OUTCOME.json'),
        'gaps':observed,'requests':list(requests.values()),'request_budget':1000,'max_pages':5,'page_size':2000,
        'scope':'only exact completed account/preflight source gaps; 15-day flanks; no performance-based new universe',
        'may_replace_frozen_metrics':False,'formal_assurance':False}
    require(len(requests)<=plan['request_budget']//plan['max_pages'],'bounded source queue exceeds budget')
    save(OUT/'PLAN.json',plan)
    print(json.dumps({'source_plan_frozen':True,'unique_root_gaps':len(observed),'queries':len(requests),'source_scope':plan['scope']},ensure_ascii=False),flush=True)
    return plan


def collect():
    plan=freeze();require(file_hash(TEN/'OUTCOME.json')==plan['outcome_sha256'],'ten-round result changed')
    if (OUT/'RESULT.json').exists():return
    require(file_hash(plan['mapping_path'])==plan['mapping_sha256'],'dated identity mapping changed')
    aliases=load_json(plan['mapping_path']).get('identity_aliases',{})
    from credentials import get_tushare_token
    source=TushareSource(ROOT/'data/fuxi_v1/announcements/20261003/postflight_responses',get_tushare_token(),max_calls=plan['request_budget'])
    rows=[]
    for req in plan['requests']:
        p=OUT/'queries'/(digest(req)+'.json')
        if p.exists():r=load_json(p)
        else:
            try:
                result=source.query(req['api'],req['params'],req['fields'],page_size=plan['page_size'],max_pages=plan['max_pages'])
                if not all(jq_code(aliases.get(r.get('ts_code'),{}).get('stable_code',r.get('ts_code','')))==req['root_instrument'] and req['params']['start_date']<=r.get('trade_date',r.get('ann_date',''))<=req['params']['end_date'] for r in result['rows']):
                    raise VendorFailure('response_scope_mismatch',req['api'])
                on_day=[r for r in result['rows'] if r.get('trade_date',r.get('ann_date'))==req['root_date'].replace('-','')]
                r={'request':req,'status':'vendor_observed','result':result,'root_date_rows':on_day,
                   'readout':'new source observations; do not infer missing quotes, zero cash or completed account from empty responses'}
            except VendorFailure as e:r={'request':req,'status':e.category,'api':e.api,'vendor_code':e.code}
            save(p,r)
        rows.append({'path':str(p),'sha256':file_hash(p),'api':req['api'],'instrument':req['root_instrument'],
            'date':req['root_date'],'status':r['status'],'rows':len(r.get('result',{}).get('rows',[])),
            'root_date_rows':r.get('root_date_rows',[])})
        print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('path','sha256','root_date_rows')},ensure_ascii=False),flush=True)
    summary={'runner_sha256':file_hash(Path(__file__)),'plan_sha256':file_hash(OUT/'PLAN.json'),'physical_calls':source.calls,'queries':len(rows),
        'statuses':dict(Counter(r['status'] for r in rows)),'readouts':rows,'formal_assurance':False,
        'frozen_ten_round_results_unchanged':file_hash(TEN/'OUTCOME.json')==plan['outcome_sha256'],
        'admitted_to_production':False,'source_findings_require_separate_versioned_consumer_validation':True}
    save(OUT/'RESULT.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='readouts'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','collect'));args=parser.parse_args()
    with controlled_writer(OUT):globals()[args.action]()
