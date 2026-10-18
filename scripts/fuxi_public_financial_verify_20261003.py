"""Verify documented income columns; preserve earlier misspelled-field probes."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource,VendorFailure
OUT=ROOT/'reports/fuxi_manual_handoff_20261003'


def main():
    path=OUT/'FINANCIAL_COLUMN_VERIFICATION.json';require(not path.exists(),'financial probe already sealed')
    from credentials import get_tushare_token
    source=TushareSource(ROOT/'data/fuxi_v1/public_source_probe/20261003/financial_columns',get_tushare_token(),max_calls=12)
    fields='ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,total_revenue,n_income'
    rows=[]
    for report_type in ('1','5'):
        params={'ts_code':'600000.SH','start_date':'20210101','end_date':'20260930','report_type':report_type}
        observation={'api':'income','params':params,'fields':fields,'independent_assurance':False}
        try:
            response=source.query('income',params,fields,page_size=2000,max_pages=2)
            destination=OUT/('INCOME_TYPE'+report_type+'_DOCUMENTED_COLUMNS.json')
            with destination.open('x',encoding='utf-8') as handle:json.dump(response,handle,ensure_ascii=False,indent=2)
            observation.update(state='observed_rows' if response['rows'] else 'successful_empty_response',rows=len(response['rows']),
                query_path=str(destination),query_sha256=file_hash(destination),
                report_types=sorted({str(r['report_type']) for r in response['rows']}),
                update_flags=sorted({str(r['update_flag']) for r in response['rows']}))
        except VendorFailure as exc:observation.update(state='vendor_failure',failure_category=exc.category,vendor_code=exc.code)
        rows.append(observation);print(json.dumps(observation,ensure_ascii=False),flush=True)
    body={'schema_version':'fuxi.financial_schema_verification.1','observed_at':datetime.now(timezone.utc).isoformat(),
        'runner_sha256':file_hash(Path(__file__)),'queries':rows,'new_http_calls':source.calls,
        'earlier_probe_correction':'income net profit field is n_income, not net_profit; earlier contract failures are client column errors, not proof that Tushare lacks financial history',
        'source_document':'https://tushare.pro/document/2?doc_id=33',
        'scope':'one-stock field and retained-report availability; not complete historical revision or publication-time certification',
        'production_catalog_changed':False,'qualification_granted':False}
    with path.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False,indent=2)


if __name__=='__main__':
    with controlled_writer(OUT):main()
