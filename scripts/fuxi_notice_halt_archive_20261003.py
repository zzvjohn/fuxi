"""Archive the paid-index primary proof for one ambiguous full-halt date."""
import hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.contracts import load_json,require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_announcements_20261003/postflight_halt_688766'
URL='https://static.cninfo.com.cn/finalpage/2025-11-26/1224826077.PDF'


def main():
    import requests
    from pypdf import PdfReader
    from io import BytesIO
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'BUNDLE.json').exists(),'full-halt source already sealed')
    parent=ROOT/'reports/fuxi_v1_announcements_20261003/postflight_sources/queries/a085e943e70ee1fb703661ee49f6bfe5c4c05b1c8321c0e7f977e2300a1e9ac4.json'
    rows=load_json(parent)['result']['rows']
    index=next(r for r in rows if r['ann_date']=='20251126' and r['url'].replace('http://','https://')==URL)
    response=requests.get(URL,timeout=(10,30));response.raise_for_status();raw=response.content
    require(raw.startswith(b'%PDF') and len(raw)<16000000,'invalid primary PDF response')
    text='\n'.join(p.extract_text() or '' for p in PdfReader(BytesIO(raw)).pages)
    compact=re.sub(r'\s+','',text).replace('（','(').replace('）',')')
    require('688766' in compact[:700] and '自2025年11月26日(星期三)开市起开始停牌' in compact,'primary halt clause not matched')
    pdf=OUT/'issuer.pdf';txt=OUT/'issuer.txt'
    with pdf.open('xb') as h:h.write(raw)
    with txt.open('x',encoding='utf-8',newline='\n') as h:h.write(text)
    bundle={'schema_version':'fuxi.public_full_halt_bundle.1','independent_assurance':False,
        'scope':'one explicitly documented start date only; no inference from expected halt duration',
        'rows':[{'instrument':'688766.XSHG','session':'2025-11-26','index_record':index,
            'url':URL,'path':str(pdf),'sha256':file_hash(pdf),'text_path':str(txt),'text_sha256':file_hash(txt),
            'first_observed_at':datetime.now(timezone.utc).isoformat(),'vendor_index_sha256':file_hash(parent)}]}
    with (OUT/'BUNDLE.json').open('x',encoding='utf-8') as h:json.dump(bundle,h,ensure_ascii=False,indent=2)
    print(json.dumps({'archived':True,'session':'2025-11-26','pdf_sha256':file_hash(pdf),'bundle_sha256':file_hash(OUT/'BUNDLE.json'),'independent_assurance':False}),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
