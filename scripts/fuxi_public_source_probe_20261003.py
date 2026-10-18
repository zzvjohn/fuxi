"""Small, isolated permission probes; never alter production data or qualification gates."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource,VendorFailure

OUT=ROOT/'reports/fuxi_manual_handoff_20261003'
DATA=ROOT/'data/fuxi_v1/public_source_probe/20261003'
QUERIES=[
    {'api':'anns_d','params':{'ts_code':'001225.SZ','start_date':'20260901','end_date':'20260930'},
     'fields':'ann_date,ts_code,name,title,url,rec_time','page_size':2000,
     'claim':'original notice index and declared publication clock; no account ownership claim'},
    {'api':'income','params':{'ts_code':'600000.SH','start_date':'20210101','end_date':'20260930','report_type':'5'},
     'fields':'ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,total_revenue,net_profit','page_size':2000,
     'claim':'availability of retained pre-adjustment financial statements; one stock only, not complete PIT archive'},
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'TUSHARE_SOURCE_PROBE.json'
    require(not target.exists(),'permission probe already sealed')
    from credentials import get_tushare_token
    source=TushareSource(DATA/'responses',get_tushare_token(),max_calls=12)
    receipts=[]
    for query in QUERIES:
        identity={k:query[k] for k in ('api','params','fields','page_size')}
        row={**identity,'claim':query['claim'],'decision':'not_observed'}
        try:
            result=source.query(query['api'],query['params'],query['fields'],page_size=query['page_size'],max_pages=2)
            path=DATA/(digest(identity)+'.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
            row.update(decision='observed_rows' if result['rows'] else 'successful_empty_response',rows=len(result['rows']),
                query_path=str(path),query_sha256=file_hash(path),independent_assurance=False)
            if query['api']=='anns_d':
                row.update(rows_with_pdf_url=sum(bool(r.get('url')) for r in result['rows']),
                    rows_with_publication_clock=sum(bool(r.get('rec_time')) for r in result['rows']))
            else:
                row.update(report_types=sorted({str(r.get('report_type')) for r in result['rows']}),
                    update_flags=sorted({str(r.get('update_flag')) for r in result['rows']}))
        except VendorFailure as exc:
            row.update(decision='vendor_failure',failure_category=exc.category,vendor_code=exc.code)
        receipts.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    record={'schema_version':'fuxi.public_source_probe.1','observed_at':datetime.now(timezone.utc).isoformat(),
        'runner_sha256':file_hash(Path(__file__)),'queries':receipts,'new_http_calls':source.calls,
        'scope':'bounded existing-credential public-data permission probe; no purchases, platform submissions, trading, or production catalog changes',
        'formal_qualification_granted':False}
    with target.open('x',encoding='utf-8') as handle:json.dump(record,handle,ensure_ascii=False,indent=2)


if __name__=='__main__':
    with controlled_writer(OUT):main()
