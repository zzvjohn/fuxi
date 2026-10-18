"""Fresh announcement entitlement probe; immutable receipts and bounded reads."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fuxi_v1.contracts import digest, require
from fuxi_v1.datasets import file_hash
from fuxi_v1.runtime import controlled_writer
from fuxi_v1.tushare_source import TushareSource, VendorFailure

OUT = ROOT / 'reports/fuxi_v1_announcements_20261003'
DATA = ROOT / 'data/fuxi_v1/announcements/20261003'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def probe():
    target = OUT / 'ENTITLEMENT_PROBE.json'
    require(not target.exists(), 'fresh probe already sealed')
    from credentials import get_tushare_token
    source = TushareSource(DATA / 'responses', get_tushare_token(), max_calls=12)
    query = {'api': 'anns_d', 'params': {'ts_code': '300980.SZ',
             'start_date': '20260901', 'end_date': '20260930'},
             'fields': 'ann_date,ts_code,name,title,url', 'page_size': 2000}
    row = dict(query)
    try:
        result = source.query(query['api'], query['params'], query['fields'],
                              page_size=2000, max_pages=3)
        raw = DATA / ('query_' + digest(query) + '.json')
        save(raw, result)
        row.update(decision='observed_rows' if result['rows'] else 'successful_empty_response',
                   rows=len(result['rows']), query_path=str(raw), query_sha256=file_hash(raw),
                   rows_with_url=sum(bool(r.get('url')) for r in result['rows']))
        print(json.dumps({'decision': row['decision'], 'rows': len(result['rows']),
              'matching_implementation_notices': [r for r in result['rows']
              if any(x in r.get('title', '') for x in ('实施', '权益分派'))]},
              ensure_ascii=False), flush=True)
    except VendorFailure as exc:
        row.update(decision='vendor_failure', failure_category=exc.category, vendor_code=exc.code)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    save(target, {'schema_version': 'fuxi.announcement_entitlement.1',
         'observed_at': datetime.now(timezone.utc).isoformat(), 'query': row,
         'new_http_calls': source.calls, 'runner_sha256': file_hash(Path(__file__)),
         'independent_assurance': False, 'formal_qualification_granted': False})


if __name__ == '__main__':
    with controlled_writer(OUT):
        probe()
