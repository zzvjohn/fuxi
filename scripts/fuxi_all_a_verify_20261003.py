"""Verify the delivered census evidence without granting investment or live status."""
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fuxi_v1.artifacts import code_manifest
from fuxi_v1.contracts import digest,load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer

OUT=ROOT/'reports/fuxi_v1_all_a_20261003'


def run():
    from fuxi_all_a_20261003 import active_catalog
    code=code_manifest();tests=load_json(OUT/'TEST_RESULT_FINAL.json')
    require(tests['code_hashes']==code and tests['exit_code']==0 and tests['failures']==0 and tests['errors']==0,'test evidence mismatch')
    require(tests['test_hashes']=={p.name:file_hash(p) for p in (ROOT/'fuxi_v1/tests').glob('test_*.py')},'test source changed')
    coverage=load_json(OUT/'LOCAL_COVERAGE.json');codes=set(coverage['codes'])
    audit=load_json(OUT/'DATA_AUDIT.json');summary=load_json(OUT/'DELIVERY_SUMMARY.json')
    require(audit['code_hashes']==code,'audit source changed')
    for name,expected in audit['artifact_hashes'].items():
        require(file_hash(OUT/name)==expected,'audited artifact changed: '+name)
    catalog,plan_path=active_catalog();manifest=load_json(catalog/'CATALOG.json');metadata=load_json(catalog/'CATALOG_METADATA.json')
    require(manifest['manifest_hash']==digest({k:v for k,v in manifest.items() if k!='manifest_hash'}),'catalog identity mismatch')
    require(metadata['metadata_hash']==digest({k:v for k,v in metadata.items() if k!='metadata_hash'}),'metadata identity mismatch')
    require(audit['manifest_hash']==manifest['manifest_hash'],'audit catalog changed')
    plan=load_json(plan_path)
    require(plan['cohort']['selection']==plan['cohort']['rule'],'consumer selection differs from declaration')
    parent=load_json(OUT/'FULL_A_RESEARCH_PLAN_V3.json');unchanged=dict(plan)
    unchanged.pop('consumer_schema_repair');unchanged['cohort']=dict(unchanged['cohort']);unchanged['cohort'].pop('selection')
    require(unchanged==parent,'consumer schema repair changed scope')
    raw=load_json(OUT/'RAW_RECHECK.json')
    for name,record in manifest['sources'].items():
        require(raw[name]['expected']==raw[name]['actual']==record['sha256']==file_hash(Path(record['path'])),'raw source changed: '+name)
    contracts=load_json(OUT/'contracts_v3/TEN_ROUND_CONTRACTS.json')
    require(contracts['code_hashes']==code and len(contracts['rounds'])==10,'contract scope mismatch')
    require(contracts['runner_sha256']==file_hash(ROOT/'scripts/fuxi_all_a_contracts_20261003.py'),'contract runner changed')
    history=load_json(OUT/'TEN_ROUND_HISTORY_CHECKS.json');valid=Counter();filled=Counter();windows={c:set() for c in codes}
    require(len(history['rounds'])==10 and history['runner_sha256']==file_hash(ROOT/'scripts/fuxi_all_a_history_checks_20261003.py'),'history scope mismatch')
    for n in range(1,11):
        cpath=OUT/'contracts_v3'/('CONTRACT_ROUND_%02d.json'%n)
        require(file_hash(cpath)==contracts['round_artifact_hashes'][str(n)],'contract round changed')
        c=load_json(cpath);require(c['summary']['rows_hash']==digest(c['rows']) and {r['instrument'] for r in c['rows']}==codes and
            len(c['rows'])==len(codes) and c['summary']['passed']==len(codes) and c['summary']['failures']==0,'contract denominator mismatch')
        hpath=OUT/('HISTORY_ACCOUNT_ROUND_%02d.json'%n)
        require(file_hash(hpath)==history['round_artifact_hashes'][str(n)],'history round changed')
        h=load_json(hpath);require(h['source_hash']==history['source_hash'] and h['rows_hash']==digest(h['rows']) and
            {r['ts_code'] for r in h['rows']}==codes and len(h['rows'])==len(codes),'history denominator mismatch')
        require(h['summary']['status_counts'].get('valid',0)>0 and h['summary']['filled_orders']>0,'empty history comparison')
        for r in h['rows']:
            require(r['formal_evidence'] is False,'diagnostic granted formal evidence')
            if r['status']=='valid':
                require(r['reconciliation']['cash_and_share_reconciled'] and r['reconciliation']['nav_reconciled'],'account reconciliation failed')
                valid[r['ts_code']]+=1;windows[r['ts_code']].add(tuple(r['window']))
            if r.get('filled_orders',0)>0:filled[r['ts_code']]+=1
    require(file_hash(OUT/'DELIVERY_STATUS.csv')==summary['registry_sha256'],'delivery registry changed')
    with (OUT/'DELIVERY_STATUS.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    require(len(rows)==len(codes) and {r['ts_code'] for r in rows}==codes,'delivery registry omitted identity')
    for r in rows:
        c=r['ts_code'];require(int(r['native_history_valid_rounds'])==valid[c] and
            int(r['native_history_rounds_with_actual_vendor_bar_fill'])==filled[c] and
            int(r['distinct_history_windows'])==len(windows[c]),'per-stock diagnostic readout mismatch')
    require(sum(r['research_delivery_ready']=='True' for r in rows)==summary['formal_research_delivery_ready']==0 and
        sum(r['live_delivery_ready']=='True' for r in rows)==summary['live_delivery_ready']==0,'unwarranted admission')
    require(file_hash(OUT/'DIAGNOSTIC_WINDOW_EXCEPTIONS.json')==summary['diagnostic_window_exceptions_sha256'],'diagnostic exception ledger changed')
    smoke=load_json(OUT/'smoke/preflight/DECISION.json')
    require(smoke['code_hashes']==code and smoke['probe_count']==12 and len(smoke['readouts'])==12,'smoke scope mismatch')
    require(smoke['plan_hash']==digest(load_json(OUT/'FULL_A_SMOKE_PLAN.json')) and
        smoke['input_sha256']==file_hash(OUT/'smoke/market_input.json'),'smoke input changed')
    for r in smoke['readouts']:
        stem=r['name']+'_'+r['scenario'];require(load_json(OUT/'smoke/preflight'/(stem+'.receipt.json'))==r and
            file_hash(OUT/'smoke/preflight'/(stem+'.json.gz'))==r['sha256'],'smoke result changed')
    require(smoke['formal_account_evidence_certified'] is False,'native smoke granted platform evidence')
    for path,expected in load_json(OUT/'DELIVERED_CODE.json').items():require(file_hash(ROOT/path)==expected,'delivered source changed: '+path)
    requirement=load_json(OUT/'DATA_REQUIREMENTS.json')
    require(file_hash(Path(requirement['concrete_instrument_requests_csv']))==requirement['source_requests_sha256'],'source request queue changed')
    registry=load_json(ROOT/'data/version_registry.json');last=registry['versions'][-1]
    require(any(p['proposal_id']=='P-20261003-001' for p in last['proposals']),'implementation not versioned')
    tracked=['CHANGELOG.md','data/version_registry.json','data/stage1_optimization_proposals.json',
             '.workbuddy/memory/_fuxi_landing_details.md','.workbuddy/memory/2026-10-03.md']
    line_endings={}
    for name in tracked:
        blob=(ROOT/name).read_bytes();bare=blob.count(b'\n')-blob.count(b'\r\n')
        require(bare==0,'new bare LF in ledger: '+name)
        line_endings[name]={'crlf':blob.count(b'\r\n'),'bare_lf':bare,'sha256':file_hash(ROOT/name)}
    policy=load_json(ROOT/'codex_automations/fuxi_runtime_policy.json')
    require(policy['autonomous_writes_enabled'] is False,'automatic writes unexpectedly enabled')
    artefacts=['REPORT.md','DELIVERY_SUMMARY.json','DATA_AUDIT.json','TEST_RESULT_FINAL.json','COST_PROFILE.json',
              'DELIVERED_CODE.json','RAW_RECHECK.json','DATA_REQUIREMENTS.json','EXTERNAL_SOURCE_REQUESTS.csv',
              'EXCLUSIONS.json','DIAGNOSTIC_WINDOW_EXCEPTIONS.json','contracts_v3/TEN_ROUND_CONTRACTS.json',
              'TEN_ROUND_HISTORY_CHECKS.json','smoke/preflight/DECISION.json','smoke_attempt1/FAILURE.json']
    result={'schema_version':'fuxi.all_a_delivery_verification.1','verified_at':datetime.now(timezone.utc).isoformat(),
        'verification_complete':True,'runner_sha256':file_hash(Path(__file__)),'version':last['version'],
        'code_hashes':code,'artifact_hashes':{n:file_hash(OUT/n) for n in artefacts},'ledger_line_endings':line_endings,
        'tests':tests['tests'],'instruments':len(codes),'contract_rounds':10,'actual_history_diagnostic_rounds':10,
        'history_instruments_valid_in_all_rounds':sum(v==10 for v in valid.values()),
        'history_instruments_filled_in_all_rounds':sum(v==10 for v in filled.values()),
        'history_instruments_ten_distinct_windows':sum(len(v)==10 for v in windows.values()),
        'downstream_probe_statuses':dict(Counter(r['status'] for r in smoke['readouts'])),
        'expanded_search_allowed':smoke['search_allowed'],'formal_research_ready':0,'actual_platform_runs':0,'live_orders':0,
        'scope':'evidence identity, complete denominators and native arithmetic; external platform and primary-data assurance remain unverified'}
    with (OUT/'FINAL_VERIFICATION.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('verification_complete','version','tests','instruments','downstream_probe_statuses','formal_research_ready','live_orders')}),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):run()
