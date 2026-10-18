"""Bounded primary-source reads for previously observed fractional account gaps."""
import json,sys,importlib.util
from pathlib import Path
from datetime import date,timedelta,datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource,VendorFailure
OUT=ROOT/'reports/fuxi_v1_concentration_20261004'
DATA=ROOT/'data/fuxi_v1/fractional_sources/20261004'
EVENTS=[('001211.SZ','2026-06-04'),('688215.SH','2025-07-11'),('688215.SH','2026-06-10'),('688557.SH','2026-07-09'),('920699.BJ','2026-07-10')]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    from credentials import get_tushare_token
    aliases=load_json(ROOT/'data/fuxi_v1/supplements/20261003/catalog_v3/CATALOG.json')['identity_aliases']
    source=TushareSource(DATA/'responses',get_tushare_token(),max_calls=45)
    spec=importlib.util.spec_from_file_location('notice_source_helper',ROOT/'scripts/fuxi_notice_collection_v6_20261003.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.DATA=DATA
    import requests
    session=requests.Session();rows=[]
    frozen={'events':EVENTS,'start_offset_days':30,'end_offset_days':7,'max_physical_vendor_calls':45,
            'runner_sha256':file_hash(Path(__file__)),'selection':'actual previous account gaps only; no return selection','formal_assurance':False}
    path=OUT/'fractional_sources/FREEZE.json'
    if path.exists():require(load_json(path)==json.loads(json.dumps(frozen)),'fractional query freeze changed')
    else:save(path,frozen)
    for code,day in EVENTS:
        codes=[code]+[old for old,v in aliases.items() if v['stable_code']==code]
        notices=[];queries=[];d=date.fromisoformat(day)
        for alias in codes:
            query=source.query('anns_d',{'ts_code':alias,'start_date':(d-timedelta(days=30)).strftime('%Y%m%d'),
                'end_date':min(date(2026,9,30),d+timedelta(days=7)).strftime('%Y%m%d')},module.FIELDS,page_size=2000,max_pages=3)
            queries+=query['evidence'];notices+=query['rows']
        docs=[];seen=set()
        for n in notices:
            title=n.get('title') or ''
            if '实施' not in title or not any(w in title for w in ('权益分派','利润分配','资本公积','转增')):continue
            if n.get('url') in seen:continue
            seen.add(n.get('url'));docs.append(module.download(n,session))
        row={'code':code,'ex_date':day,'index_count':len(notices),'queries':queries,'documents':docs}
        receipt=OUT/'fractional_sources'/(digest([code,day])+'.json')
        if receipt.exists():require(load_json(receipt)==row,'fractional receipt changed')
        else:save(receipt,row)
        rows.append(row);print(json.dumps({'code':code,'ex_date':day,'notices':len(notices),'documents':len(docs)},ensure_ascii=False),flush=True)
    result={'rows':rows,'physical_calls':source.calls,'formal_assurance':False,'runner_sha256':file_hash(Path(__file__))}
    result_path=OUT/'fractional_sources/RESULT.json'
    if result_path.exists():require(load_json(result_path)['rows']==rows,'source result changed')
    else:save(result_path,result)


if __name__=='__main__':
    with controlled_writer(OUT):main()
