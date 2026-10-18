"""Bounded BSE historical-identifier diagnosis with the user's existing entitlement."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require,digest
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource,VendorFailure
OUT=ROOT/'reports/fuxi_v1_announcements_20261003'


def main():
    from credentials import get_tushare_token
    target=OUT/'BSE_IDENTIFIER_PROBE.json';require(not target.exists(),'BSE identifier probe already sealed')
    aliases=load_json(ROOT/'data/fuxi_v1/supplements/20261003/catalog_v3/CATALOG.json')['identity_aliases']
    old=next(k for k,v in aliases.items() if v['stable_code']=='920000.BJ')
    source=TushareSource(ROOT/'data/fuxi_v1/announcements/20261003/bse_identifier_probe',get_tushare_token(),max_calls=20)
    rows=[]
    for code in (old,old.replace('.BJ','.NQ'),old.split('.')[0]):
        query={'ts_code':code,'start_date':'20240406','end_date':'20240612'}
        row={'query':query,'stable_code':'920000.BJ','dated_mapping_evidence':aliases[old]}
        try:
            result=source.query('anns_d',query,'ann_date,ts_code,name,title,url,rec_time',page_size=2000,max_pages=2)
            row.update(status='observed_rows' if result['rows'] else 'successful_empty',rows=len(result['rows']),
                       implementation_notices=[r for r in result['rows'] if '实施' in r.get('title','') or '权益分派' in r.get('title','')],result=result)
        except VendorFailure as exc:row.update(status='vendor_failure',category=exc.category,code=exc.code)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('result','dated_mapping_evidence')},ensure_ascii=False),flush=True)
    record={'queries':rows,'runner_sha256':file_hash(Path(__file__)),'new_http_calls':source.calls,'qualification_granted':False}
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    with controlled_writer(OUT):main()
