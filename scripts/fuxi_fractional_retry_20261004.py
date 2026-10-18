"""One bounded source-only revision selected by observed account gaps."""
import json,sys,importlib.util
from pathlib import Path
from datetime import date,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource
from fuxi_v1.fractional_sources import verified_rule
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
DATA=ROOT/'data/fuxi_v1/fractional_sources/20261004/revision1'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    from credentials import get_tushare_token
    from fuxi_v1.a_share import canonical_code
    outcome=load_json(OUT/'attempt1/OUTCOME.json');inventory=load_json(OUT/'ACTION_TERM_INVENTORY.json')
    require(outcome['arms_completed']==8,'complete frozen arms before source revision')
    already={(r['code'],r['ex_date']) for r in load_json(OUT/'fractional_sources/RESULT.json')['rows']}
    events=sorted({(canonical_code(g['instrument']),g['session']) for g in outcome['root_gaps']
        if g['reason']=='fractional_allotment_not_observed'}-already)
    require(len({c for c,d in events})<=20 and len(events)<=40,'source revision exceeds frozen stock/event budget')
    frozen={'events':events,'observed_outcome_sha256':file_hash(OUT/'attempt1/OUTCOME.json'),
        'selection':'data gaps only; no return selection','max_physical_vendor_calls':90,'revision_number':1,
        'runner_sha256':file_hash(Path(__file__)),'formal_assurance':False}
    base=OUT/'source_revision1';save(base/'FREEZE.json',frozen)
    aliases=load_json(ROOT/'data/fuxi_v1/supplements/20261003/catalog_v3/CATALOG.json')['identity_aliases']
    source=TushareSource(DATA/'responses',get_tushare_token(),max_calls=90)
    spec=importlib.util.spec_from_file_location('notice_source_helper',ROOT/'scripts/fuxi_notice_collection_v6_20261003.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.DATA=DATA
    import requests
    session=requests.Session();rows=[];terms=inventory;rules=load_json(OUT/'FRACTIONAL_RULES_v2.json')['rules'].copy()
    for code,day in events:
        codes=[code]+[old for old,v in aliases.items() if v['stable_code']==code];notices=[];queries=[];d=date.fromisoformat(day)
        for alias in codes:
            query=source.query('anns_d',{'ts_code':alias,'start_date':(d-timedelta(days=30)).strftime('%Y%m%d'),
                'end_date':min(date(2026,9,30),d+timedelta(days=7)).strftime('%Y%m%d')},module.FIELDS,page_size=2000,max_pages=3)
            queries+=query['evidence'];notices+=query['rows']
        docs=[];seen=set();checks=[]
        for n in notices:
            title=n.get('title') or ''
            if '实施' not in title or not any(w in title for w in ('权益分派','利润分配','资本公积','转增')):continue
            if n.get('url') in seen:continue
            seen.add(n.get('url'));docs.append(module.download(n,session))
        for term in terms:
            if canonical_code(term['instrument'])!=code or term['ex_date']!=day:continue
            for doc in docs:
                try:
                    rule=verified_rule(term,doc)
                    if rule:rules[term['evidence_id']]=rule
                    checks.append({'event':term['evidence_id'],'supported':bool(rule)})
                except Exception as exc:checks.append({'event':term['evidence_id'],'supported':False,'reason':str(exc)})
        row={'code':code,'ex_date':day,'queries':queries,'documents':docs,'rule_checks':checks}
        save(base/(digest([code,day])+'.json'),row);rows.append(row)
        print(json.dumps({'code':code,'ex_date':day,'documents':len(docs),'supported':sum(c['supported'] for c in checks)},ensure_ascii=False),flush=True)
    save(base/'RESULT.json',{'rows':rows,'physical_calls':source.calls,'formal_assurance':False})
    save(OUT/'FRACTIONAL_RULES_v3.json',{'rules':rules,'base_sha256':file_hash(OUT/'FRACTIONAL_RULES_v2.json'),
        'source_revision_sha256':file_hash(base/'RESULT.json'),'formal_assurance':False})


if __name__=='__main__':
    with controlled_writer(OUT):main()
