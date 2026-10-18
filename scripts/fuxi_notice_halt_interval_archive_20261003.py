"""Preserve rc10 partial attempt; archive actual issuer resumption proof."""
import json,re,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_announcements_20261003';DEST=OUT/'postflight_halt_interval_688766'
URL='https://static.cninfo.com.cn/finalpage/2025-12-09/1224858827.PDF'


def main():
    old=OUT/'ten_rounds_v3';require(not (old/'OUTCOME.json').exists(),'rc10 unexpectedly finished')
    require(not list((old/'rounds').glob('*.receipt.json')),'sealed rounds exist; require explicit recovery')
    stop={'stopped_at':datetime.now(timezone.utc).isoformat(),'reason':'single-day issuer halt proof moved held-path coverage gap from 2025-11-26 to 2025-11-27; closed halt interval requires separate source adjudication',
        'completed_candidate_rounds':0,'preflight_sha256':file_hash(old/'preflight/DECISION.json'),
        'freeze_sha256':file_hash(old/'FREEZE.json'),'input_receipt_sha256':file_hash(old/'INPUT_RECEIPT.json'),
        'frozen_input_and_code_retained':True,'next_attempt':'ten_rounds_v4'}
    with (old/'STOP_FOR_INTERVAL_CORRECTION.json').open('x',encoding='utf-8') as h:json.dump(stop,h,ensure_ascii=False,indent=2)
    import requests
    from pypdf import PdfReader
    from io import BytesIO
    DEST.mkdir();parent=OUT/'postflight_sources/queries/a085e943e70ee1fb703661ee49f6bfe5c4c05b1c8321c0e7f977e2300a1e9ac4.json'
    index=next(r for r in load_json(parent)['result']['rows'] if r['url'].replace('http://','https://')==URL)
    response=requests.get(URL,timeout=(10,30));response.raise_for_status();raw=response.content
    require(raw.startswith(b'%PDF') and len(raw)<16000000,'invalid resumption PDF')
    text='\n'.join(p.extract_text() or '' for p in PdfReader(BytesIO(raw)).pages)
    compact=re.sub(r'\s+','',text).replace('（','(').replace('）',')')
    require('证券代码：688766' in compact[:700] and '将于2025年12月9日(星期二)开市起复牌' in compact,'explicit resumption not matched')
    pdf=DEST/'resumption.pdf';txt=DEST/'resumption.txt'
    with pdf.open('xb') as h:h.write(raw)
    with txt.open('x',encoding='utf-8',newline='\n') as h:h.write(text)
    start=load_json(OUT/'postflight_halt_688766/BUNDLE.json')['rows'][0]
    resumption={'index_record':index,'url':URL,'path':str(pdf),'sha256':file_hash(pdf),'text_path':str(txt),'text_sha256':file_hash(txt),
        'first_observed_at':datetime.now(timezone.utc).isoformat(),'vendor_index_sha256':file_hash(parent)}
    bundle={'schema_version':'fuxi.public_closed_halt_bundle.1','independent_assurance':False,
        'scope':'intersection of actual issuer start and explicitly recalled start-to-resumption interval; research valuation only; source start-date conflict retained',
        'rows':[{'instrument':'688766.XSHG','start_source':start,'resumption_source':resumption}]}
    with (DEST/'BUNDLE.json').open('x',encoding='utf-8') as h:json.dump(bundle,h,ensure_ascii=False,indent=2)
    print(json.dumps({'archived':True,'pdf_sha256':file_hash(pdf),'bundle_sha256':file_hash(DEST/'BUNDLE.json'),'stopped_completed_rounds':0},ensure_ascii=False))


if __name__=='__main__':
    with controlled_writer(DEST):main()
