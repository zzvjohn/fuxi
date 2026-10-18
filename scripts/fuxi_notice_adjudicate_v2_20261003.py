"""Source-bound conservative ordinary distributions; retain every unresolved event."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import require,digest,load_json
from fuxi_v1.datasets import file_hash
from fuxi_v1.historical_adapter import dividend_terms
from fuxi_v1.history import jq_code,iso
from fuxi_v1.public_notices import extract_ordinary
from fuxi_v1.action_evidence import resolve_action_terms
from fuxi_v1.runtime import controlled_writer

OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
COLLECTION=OUT/'collection_v6/COLLECTION_RESULT.json'
CATALOG=ROOT/'data/fuxi_v1/supplements/20261003/catalog_v3'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def main():
    require(not (OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v2.json').exists(),'ordinary implementation bundle already sealed')
    collected=load_json(COLLECTION)
    require(collected['event_count']==1085 and len(collected['events'])==1085,'incomplete event collection')
    records=list(csv.DictReader((CATALOG/'dividend.csv').open(encoding='utf-8-sig')))
    records=[{k:None if v in ('','nan','None') else v for k,v in r.items()} for r in records]
    master_path=ROOT/'reports/fuxi_v1_all_a_20261003/SECURITY_MASTER.json'
    coverage_path=ROOT/'reports/fuxi_v1_all_a_20261003/LOCAL_COVERAGE.json'
    selected={jq_code(code) for code in load_json(coverage_path)['codes']}
    require(len(selected)==5909,'ordinary A-share master scope changed')
    base=[t for t in dividend_terms(records,['2021-01-04','2026-09-30']) if t['instrument'] in selected]
    byid={r['evidence_id']:r for r in base}
    groups=defaultdict(list)
    for r in records:
        if r['div_proc']=='实施':groups[(jq_code(r['ts_code']),iso(r.get('ex_date')))].append(r)
    adjudications=[];readouts=[];index_records={}
    for row in collected['events']:
        require(file_hash(row['path'])==row['sha256'],'announcement event receipt changed')
        event=load_json(row['path']);original=byid[row['event_id']]
        for record in event.get('result',{}).get('rows',[]):index_records[digest(record)]=record
        proposals=[];docs=[]
        for doc in event.get('documents',[]):
            if doc['state']!='downloaded':continue
            require(file_hash(doc['path'])==doc['sha256'] and file_hash(doc['text_path'])==doc['text_sha256'],'archived notice changed')
            source={k:doc[k] for k in ('url','path','sha256','text_path','text_sha256')}
            if original['instrument'].endswith('.BJ'):
                source.update(identity_mapping_catalog_path=str(CATALOG/'CATALOG.json'),
                              identity_mapping_catalog_sha256=file_hash(CATALOG/'CATALOG.json'))
            result=extract_ordinary(Path(doc['text_path']).read_text(encoding='utf-8'),doc['index_record'],
                                    original,groups[(original['instrument'],original['ex_date'])],source)
            docs.append(result)
            if result['status']=='ordinary_terms_proposed':proposals.append(result['adjudication'])
        terms={digest({k:p[k] for k in ('instrument','ex_date','record_date','pay_date','holder_cash','holder_shares','share_credit_date')}):p for p in proposals}
        reason='no_unambiguous_ordinary_implementation'
        if len(terms)==1:
            chosen=max(proposals,key=lambda p:(p['ann_date'],p['sources'][0]['sha256']))
            resolved=resolve_action_terms([original],groups[(original['instrument'],original['ex_date'])],[chosen])
            require(len(resolved)==1 and resolved[0]['status'] in ('cash_terms_supported','ordinary_distribution_supported'),'implementation import failed')
            adjudications.append(chosen);reason='source_bound_ordinary_terms'
        elif len(terms)>1:reason='conflicting_public_implementation_terms'
        readouts.append({'event_id':row['event_id'],'instrument':original['instrument'],'ex_date':original['ex_date'],
            'original_status':original['status'],'resolved':len(terms)==1,'reason':reason,
            'document_checks':[{k:v for k,v in x.items() if k!='adjudication'} for x in docs]})
    resolved=resolve_action_terms(base,records,adjudications)
    clocks=Counter()
    for record in index_records.values():
        ann=iso(record.get('ann_date'));raw_clock=record.get('rec_time')
        rec=raw_clock[:10] if isinstance(raw_clock,str) else ''
        clocks['unique_full_index_records']+=1
        if not rec:clocks['rec_time_missing']+=1
        if raw_clock is not None and not isinstance(raw_clock,str):clocks['rec_time_non_text_unparsed']+=1
        elif ann and rec>ann:clocks['rec_time_date_after_ann_date']+=1
    bundle={'schema_version':'fuxi.public_implementation_bundle.1','adjudications':adjudications,
        'readouts':readouts,'independent_assurance':False,'formal_qualification_granted':False,
        'collection_sha256':file_hash(COLLECTION),'vendor_dividend_sha256':file_hash(CATALOG/'dividend.csv'),
        'ordinary_A_share_master_sha256':file_hash(master_path),'ordinary_A_share_master_size':len(selected),
        'ordinary_A_share_scope_path':str(coverage_path),'ordinary_A_share_scope_sha256':file_hash(coverage_path),
        'parser_sha256':file_hash(ROOT/'fuxi_v1/public_notices.py'),'runner_sha256':file_hash(Path(__file__)),
        'publication_index_clock_diagnostics':dict(clocks),
        'resolved_existing_gap_events':len(adjudications),'original_status_counts':dict(Counter(t['status'] for t in base)),
        'current_status_counts':dict(Counter(t['status'] for t in resolved)),
        'unresolved_event_reasons':dict(Counter(r['reason'] for r in readouts if not r['resolved'])),
        'unresolved_document_reasons':dict(Counter(x.get('reason') for r in readouts for x in r['document_checks'] if x['status']=='unresolved')),
        'settlement_scope':'hypothetical record-close entitlement; integral shares only; declared 20% tax; no actual investor allocation receipt'}
    save(OUT/'PUBLIC_IMPLEMENTATION_BUNDLE_v2.json',bundle)
    print(json.dumps({k:v for k,v in bundle.items() if k not in ('adjudications','readouts')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
