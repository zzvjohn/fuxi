"""Archive one directly located official Shanghai registration guide."""
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fuxi_v1.runtime import controlled_writer
OUT=ROOT/'reports/fuxi_v1_concentration_20261004/settlement_guide'
URL='https://www.chinaclear.cn/zdjs/dsh/202503/48f86699863d4b47bed4aa200ccae43e/files/中国证券登记结算有限责任公司上海分公司证券发行人业务指南.pdf'


def main():
    import requests,fitz
    OUT.mkdir(parents=True,exist_ok=True)
    row={'url':URL,'retrieved_at':datetime.now(timezone.utc).isoformat(),'formal_assurance':False}
    try:
        response=requests.get(URL,timeout=(10,30));response.raise_for_status();blob=response.content
        if not blob.startswith(b'%PDF-') or len(blob)>8000000:raise ValueError('primary_pdf_unavailable')
        with fitz.open(stream=blob,filetype='pdf') as doc:text='\n\n'.join(p.get_text() for p in doc)
        pdf=OUT/'SH_GUIDE_202503.pdf';txt=OUT/'SH_GUIDE_202503.txt'
        pdf.write_bytes(blob);txt.write_text(text,encoding='utf-8',newline='\n')
        row.update(status='downloaded',path=str(pdf),text_path=str(txt),sha256=hashlib.sha256(blob).hexdigest(),
                   text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    except Exception as exc:row.update(status='primary_guide_not_available',exception_type=type(exc).__name__)
    with (OUT/'RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(row,f,ensure_ascii=False,indent=2)
    print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with controlled_writer(OUT):main()
