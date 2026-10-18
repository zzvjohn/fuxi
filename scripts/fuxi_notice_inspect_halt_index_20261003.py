import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/fuxi_v1_announcements_20261003'
r=json.loads((OUT/'postflight_sources/RESULT.json').read_text(encoding='utf-8'))
for item in r['readouts']:
    p=Path(item['path']);raw=json.loads(p.read_text(encoding='utf-8'))
    if '688766' in json.dumps(raw.get('request',{})) and 'anns_d' in json.dumps(raw.get('request',{})):
        print(json.dumps(raw,ensure_ascii=False))
print('readout keys',[(x.get('endpoint'),x.get('instrument'),x['path']) for x in r['readouts']])
