"""Resolve conditional rules; missing clauses remain unsupported."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,digest
from fuxi_v1.fractional_sources import verified_rule
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.datasets import file_hash
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'


def main():
    inventory=load_json(OUT/'ACTION_TERM_INVENTORY.json');sources=load_json(OUT/'fractional_sources/RESULT.json')['rows']
    rules={};audit=[]
    for term in inventory:
        if term['status']!='ordinary_distribution_supported' or term['share_distribution']<=0:continue
        documents=list((term.get('issuer_evidence') or {}).get('sources',[]))
        for r in sources:
            prefix=term['instrument'].split('.')[0]
            if r['code'].split('.')[0]==prefix and r['ex_date']==term['ex_date']:
                documents.extend(d for d in r['documents'] if d['state']=='downloaded')
        found=[]
        for doc in documents:
            rule=verified_rule(term,doc)
            if rule:found.append(rule)
        if found:rules[term['evidence_id']]=found[0]
        audit.append({'instrument':term['instrument'],'ex_date':term['ex_date'],'evidence_id':term['evidence_id'],
                      'documents_reviewed':len(documents),'status':'source_supported_floor_or_ceil' if found else 'primary_fractional_clause_not_established'})
    value={'rules':rules,'audit':audit,'formal_assurance':False,'inventory_sha256':file_hash(OUT/'ACTION_TERM_INVENTORY.json'),
           'source_result_sha256':file_hash(OUT/'fractional_sources/RESULT.json'),'runner_sha256':file_hash(Path(__file__))}
    with (OUT/'FRACTIONAL_RULES_v2.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'supported_rules':len(rules),'reviewed_ordinary_share_events':len(audit),
        'focus':[r for r in audit if r['instrument'] in {'001211.XSHE','688215.XSHG','688557.XSHG','920699.BJ'}]},ensure_ascii=False))


if __name__=='__main__':
    with controlled_writer(OUT):main()
