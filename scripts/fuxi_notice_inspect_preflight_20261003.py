import gzip
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/fuxi_v1_announcements_20261003/ten_rounds_v3/preflight'
for name in ('reversal5_normal','low_price_cv20_normal'):
    path=P/(name+'.json.gz')
    if not path.exists():continue
    with gzip.open(path,'rt',encoding='utf-8') as handle:r=json.load(handle)
    a=r.get('account',r)
    print(name,list(r),json.dumps([g for g in a.get('gaps',[]) if g.get('instrument')],ensure_ascii=False))
