"""Bounded financial-schema diagnosis and one public issuer notice; evidence only."""
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import digest,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource

OUT=ROOT/'reports/fuxi_manual_handoff_20261003'
PDF_URL='https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESZ_STOCK/2026/2026-9/2026-09-22/12616031.PDF'


def main():
    import requests
    target=OUT/'PUBLIC_SOURCE_FOLLOWUP.json';require(not target.exists(),'follow-up already sealed')
    from credentials import get_tushare_token
    source=TushareSource(ROOT/'data/fuxi_v1/public_source_probe/20261003/schema_followup',get_tushare_token(),max_calls=2)
    fields='ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,total_revenue,net_profit'
    request={'api':'income','params':{'ts_code':'600000.SH','start_date':'20210101','end_date':'20260930','report_type':'1'},'fields':fields}
    finance={'request':request,'formal_assurance':False}
    try:
        raw=source._http({'api_name':request['api'],'token':source.token,'params':dict(request['params'],limit=2000,offset=0),'fields':fields})
        result=json.loads(raw);data=result.get('data') or {};names=data.get('fields');items=data.get('items')
        finance.update(vendor_code=result.get('code'),response_sha256=hashlib.sha256(raw).hexdigest(),
            rows=len(items) if isinstance(items,list) else None,
            matching_requested_fields=sorted(set(names)&set(fields.split(','))) if isinstance(names,list) else [],
            field_schema_observed=isinstance(names,list) and set(fields.split(','))<=set(names))
        if result.get('code')==0 and finance['field_schema_observed'] and isinstance(items,list) and all(len(r)==len(names) for r in items):
            rows=[{k:r[names.index(k)] for k in fields.split(',')} for r in items]
            path=OUT/'INCOME_REPORT_TYPE1_SAMPLE.json'
            with path.open('x',encoding='utf-8') as handle:json.dump({'request':request,'rows':rows,'rows_hash':digest(rows)},handle,ensure_ascii=False,indent=2)
            finance.update(state='observed_rows' if rows else 'successful_empty_response',query_path=str(path),query_sha256=file_hash(path),
                report_types=sorted({str(r['report_type']) for r in rows}),update_flags=sorted({str(r['update_flag']) for r in rows}))
        else:finance['state']='schema_unusable_or_empty_without_field_schema'
    except Exception as exc:finance.update(state='transport_or_parse_failure',exception_type=type(exc).__name__)
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    notice={'url':PDF_URL,'issuer':'湖北祥源新材科技股份有限公司','instrument':'300980.SZ','notice_number':'2026-054',
        'notice_date':'2026-09-22','scope':'issuer implementation terms; public mirror download; not holder-specific settlement',
        'formal_assurance':False}
    try:
        with requests.get(PDF_URL,timeout=(10,30),stream=True) as response:
            response.raise_for_status();blob=response.raw.read(8000001,decode_content=True)
        require(len(blob)<=8000000 and blob.startswith(b'%PDF-'),'issuer PDF byte contract')
        path=evidence/'300980_20260922_2026-054.pdf'
        with path.open('xb') as handle:handle.write(blob)
        notice.update(state='downloaded_issuer_pdf',path=str(path),sha256=file_hash(path),bytes=len(blob),
            observed_public_terms={'gross_cash_per_share':'0','cash_payment_date':'not_applicable_pure_capitalization',
                'shares_per_existing_holder_share':'0.4','record_date':'2026-09-29','ex_date':'2026-09-30',
                'share_credit_date':'2026-09-30','new_tradable_share_date':'2026-09-30',
                'total_company_reference_share_ratio':'0.3901769',
                'share_ratio_difference_basis':'repurchase account excluded from distribution; aggregate ex-rights ratio differs from eligible-holder ratio',
                'fractional_allocation':'rank fractional remainders; random order for equal remainders; holder outcome not determined by the public notice'},
            terms_checked_via='web PDF text, three pages, pages 1-2 implementation sections; no production adjudication imported')
    except Exception as exc:notice.update(state='pdf_download_or_contract_failure',exception_type=type(exc).__name__)
    body={'schema_version':'fuxi.public_source_followup.1','observed_at':datetime.now(timezone.utc).isoformat(),
        'runner_sha256':file_hash(Path(__file__)),'parent_probe_sha256':file_hash(OUT/'TUSHARE_SOURCE_PROBE.json'),
        'financial_sample':finance,'issuer_notice':notice,'production_catalog_changed':False,'qualification_granted':False,
        'scope':'read-only evidence acquisition and financial schema diagnosis; no strategy submission or account connection'}
    with target.open('x',encoding='utf-8') as handle:json.dump(body,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'finance':finance,'notice':notice},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
