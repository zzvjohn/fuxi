"""Collect a frozen, outcome-independent existing corporate-action gap queue."""
import csv
import json
import sys
import time
from collections import Counter
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fuxi_v1.contracts import digest, require, load_json
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource, VendorFailure

OUT = ROOT / 'reports/fuxi_v1_announcements_20261003'
DATA = ROOT / 'data/fuxi_v1/announcements/20261003'
REQUESTS = ROOT / 'reports/fuxi_v1_all_a_20261003/EXTERNAL_SOURCE_REQUESTS.csv'
FIELDS = 'ann_date,ts_code,name,title,url,rec_time'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def freeze():
    rows = [r for r in csv.DictReader(REQUESTS.open(encoding='utf-8-sig'))
            if r['source_kind'] == 'issuer_and_register_holder_rights']
    require(len(rows) == 1085, 'source event queue changed')
    queries = []
    for row in rows:
        ex = date.fromisoformat(row['date_start'])
        queries.append({'event': row, 'api': 'anns_d', 'params': {
            'ts_code': row['ts_code'], 'start_date': (ex-timedelta(days=60)).strftime('%Y%m%d'),
            'end_date': min(date(2026,9,30),ex+timedelta(days=7)).strftime('%Y%m%d')},
            'fields': FIELDS, 'page_size': 2000, 'max_pages': 5})
    frozen = {'schema_version': 'fuxi.notice_gap_collection.1', 'queue': queries,
        'source_request_sha256': file_hash(REQUESTS), 'runner_sha256': file_hash(Path(__file__)),
        'http_budget': 6000, 'max_pdf_count': 6000, 'query_scope': 'existing audit gaps only',
        'selection_uses_returns': False, 'original_panels_immutable': True,
        'formal_assurance': False, 'publication_time_is_vendor_claim': True}
    path = OUT/'COLLECTION_FREEZE.json'
    if path.exists():
        require(load_json(path) == frozen, 'frozen collection changed')
    else:
        save(path, frozen)
        (OUT/'notice_collection_runner.py').write_bytes(Path(__file__).read_bytes())
    return frozen


def download(notice, session):
    url = notice.get('url') or ''
    parsed = urlparse(url)
    allowed = ('cninfo.com.cn','sse.com.cn','szse.cn','bse.cn')
    require(parsed.scheme in ('http','https') and parsed.hostname and
            any(parsed.hostname == h or parsed.hostname.endswith('.'+h) for h in allowed),
            'unrecognized public document host')
    url = url.replace('http://','https://',1)
    key = digest({'url':url})
    meta = DATA/'documents'/(key+'.receipt.json')
    if meta.exists():
        row = load_json(meta)
        if row['state'] == 'downloaded':
            require(file_hash(row['path'])==row['sha256'] and
                    file_hash(row['text_path'])==row['text_sha256'], 'notice cache changed')
        return row
    row = {'url':url,'index_record':notice,'retrieved_at':datetime.now(timezone.utc).isoformat(),
           'independent_assurance':False}
    for attempt in range(3):
        try:
            with session.get(url, timeout=(10,30), stream=True) as response:
                response.raise_for_status()
                blob = response.raw.read(8000001, decode_content=True)
            require(len(blob)<=8000000 and blob.startswith(b'%PDF-'), 'PDF byte contract')
            import fitz
            with fitz.open(stream=blob, filetype='pdf') as doc:
                require(len(doc)<=250, 'notice page budget')
                text = '\n\n'.join(page.get_text() for page in doc)
                pages = len(doc)
            pdf = DATA/'documents'/(key+'.pdf'); txt = DATA/'documents'/(key+'.txt')
            pdf.parent.mkdir(parents=True,exist_ok=True)
            with pdf.open('xb') as handle: handle.write(blob)
            with txt.open('x',encoding='utf-8',newline='\n') as handle: handle.write(text)
            row.update(state='downloaded', path=str(pdf),sha256=file_hash(pdf),bytes=len(blob),
                       text_path=str(txt),text_sha256=file_hash(txt),pages=pages,
                       searchable_text=bool(text.strip()))
            break
        except Exception as exc:
            row.update(state='download_failed',exception_type=type(exc).__name__)
            if attempt<2:time.sleep(1+attempt)
    save(meta,row)
    return row


def run():
    import requests
    from credentials import get_tushare_token
    frozen = freeze()
    source = TushareSource(DATA/'gap_responses', get_tushare_token(), max_calls=frozen['http_budget'])
    session = requests.Session(); receipts=[]; documents={}
    for number, query in enumerate(frozen['queue'],1):
        key = digest(query); target=DATA/'events'/(key+'.json')
        if target.exists():
            receipt=load_json(target)
            require(receipt['query']==query, 'event receipt changed')
            for doc in receipt.get('documents',[]):
                if doc['state']=='downloaded':
                    require(file_hash(doc['path'])==doc['sha256'] and file_hash(doc['text_path'])==doc['text_sha256'],
                            'existing document changed')
        else:
            receipt={'query':query,'freeze_hash':digest(frozen),'state':'not_observed'}
            try:
                result=source.query(query['api'],query['params'],query['fields'],
                                    page_size=query['page_size'],max_pages=query['max_pages'])
                require(all(r['ts_code']==query['params']['ts_code'] and
                    query['params']['start_date']<=r['ann_date']<=query['params']['end_date'] for r in result['rows']),
                    'announcement identity/date mismatch')
                matches=[r for r in result['rows'] if any(w in (r.get('title') or '')
                    for w in ('实施','权益分派','除权','除息','重整计划')) and
                    not any(w in (r.get('title') or '') for w in ('意见书','核查意见','股东大会','董事会决议'))]
                docs=[]
                for notice in matches:
                    require(len(documents)<frozen['max_pdf_count'], 'PDF queue budget exhausted')
                    try:doc=download(notice,session)
                    except Exception as exc:doc={'state':'document_contract_failure','exception_type':type(exc).__name__,'index_record':notice}
                    docs.append(doc); documents[doc.get('url') or digest(notice)]=doc
                receipt.update(state='index_observed',result=result,index_rows=len(result['rows']),
                    matching_notices=len(matches),documents=docs,
                    unresolved_reason=None if any(d['state']=='downloaded' for d in docs) else 'no_downloaded_matching_original')
            except VendorFailure as exc:
                receipt.update(state='vendor_failure',failure_category=exc.category,vendor_code=exc.code)
            except Exception as exc:
                receipt.update(state='contract_failure',exception_type=type(exc).__name__)
            save(target,receipt)
        receipts.append({'event_id':query['event']['current_evidence_id'],
            'ts_code':query['event']['ts_code'],'ex_date':query['event']['date_start'],
            'path':str(target),'sha256':file_hash(target),'state':receipt['state'],
            'index_rows':receipt.get('index_rows',0),'matching_notices':receipt.get('matching_notices',0),
            'downloaded_notices':sum(d['state']=='downloaded' for d in receipt.get('documents',[])),
            'unresolved_reason':receipt.get('unresolved_reason')})
        for doc in receipt.get('documents',[]):documents[doc.get('url') or digest(doc)]=doc
        if number%10==0 or number==len(frozen['queue']):
            print(json.dumps({'completed':number,'total':len(frozen['queue']),'new_http_calls':source.calls,
                'downloaded_documents':sum(d['state']=='downloaded' for d in documents.values()),
                'events_with_original':sum(r['downloaded_notices']>0 for r in receipts)},ensure_ascii=False),flush=True)
    summary={'freeze_hash':digest(frozen),'events':receipts,'event_count':len(receipts),
       'states':dict(Counter(r['state'] for r in receipts)),
       'events_with_original':sum(r['downloaded_notices']>0 for r in receipts),
       'unique_documents':len(documents),'downloaded_documents':sum(d['state']=='downloaded' for d in documents.values()),
       'document_states':dict(Counter(d['state'] for d in documents.values())),
       'new_http_calls_this_invocation':source.calls,'formal_assurance':False,'qualification_granted':False}
    target=OUT/'COLLECTION_RESULT.json'
    if target.exists():
        old=load_json(target)
        require(old['events']==summary['events'] and old['freeze_hash']==summary['freeze_hash'],'completed collection changed')
    else:save(target,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):run()
