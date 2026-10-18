# -*- coding: utf-8 -*-
"""Collect the vendor's authoritative delisting table.

The local security master (`stock_list.csv`) is a point-in-time snapshot that
still carries names which have since left the market, so "absent from the
master" cannot be used as a delisting signal on its own. This batch adds the
vendor's own `delist_date` as an explicit channel.

Writes `data/raw/delist_dates.csv` (utf-8-sig, CRLF, matching stock_list.csv).
Idempotent: an existing file is only replaced when the content actually differs,
and the previous content is kept next to it.
"""
import io
import os
import sys
import csv
import json
import time
import hashlib
import datetime

sys.path.insert(0, 'e:/quant')

ROOT = 'e:/quant'
RAW = os.path.join(ROOT, 'data', 'raw')
TARGET = os.path.join(RAW, 'delist_dates.csv')
COLUMNS = ['ts_code', 'name', 'list_status', 'list_date', 'delist_date', 'market']
STATUSES = ('L', 'D', 'P')
FIELDS = 'ts_code,name,list_status,list_date,delist_date,market'


def sha256(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(pro):
    rows = []
    for status in STATUSES:
        frame = pro.stock_basic(list_status=status, fields=FIELDS)
        if frame is None:
            continue
        for record in frame.to_dict('records'):
            rows.append({column: ('' if record.get(column) is None
                                  else str(record.get(column)).strip())
                         for column in COLUMNS})
        time.sleep(0.4)
    return rows


def serialise(rows):
    """CRLF + BOM, single writer, deterministic order."""
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator='\r\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r['ts_code']):
        writer.writerow(row)
    text = buffer.getvalue()
    assert '\r\n' in text and text.count('\n') == text.count('\r\n'), 'not pure CRLF'
    return text.encode('utf-8-sig')


def main():
    from credentials import get_tushare_token
    import tushare as ts

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    pro.__timeout = 120

    rows = fetch(pro)
    by_status = {}
    for row in rows:
        by_status[row['list_status']] = by_status.get(row['list_status'], 0) + 1
    print('fetched %d rows; by status %s' % (len(rows), by_status), flush=True)

    listed = [r for r in rows if r['list_status'] == 'L']
    delisted = [r for r in rows if r['list_status'] == 'D']
    assert listed and delisted, 'vendor returned an implausible master'
    missing_dates = [r['ts_code'] for r in delisted if not r['delist_date']]
    assert not missing_dates, 'delisted rows without a delist_date: %s' % missing_dates[:5]

    payload = serialise(rows)
    previous = io.open(TARGET, 'rb').read() if os.path.exists(TARGET) else None
    changed = previous != payload
    if changed:
        if previous is not None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            keep = TARGET + '.bak_' + stamp
            io.open(keep, 'wb').write(previous)
            print('previous content kept at %s' % os.path.basename(keep), flush=True)
        io.open(TARGET, 'wb').write(payload)
        print('wrote %s (%d bytes)' % (TARGET, len(payload)), flush=True)
    else:
        print('content unchanged; file left untouched', flush=True)

    check = io.open(TARGET, 'rb').read()
    assert check == payload, 'written file does not read back'
    assert check[:3] == b'\xef\xbb\xbf', 'missing BOM'
    assert check.count(b'\n') - check.count(b'\r\n') == 0, 'bare LF introduced'
    delist_dates = sorted(r['delist_date'] for r in delisted)
    after_window_start = [d for d in delist_dates if d >= '20241230']
    manifest = {
        'file': 'data/raw/delist_dates.csv',
        'sha256': sha256(TARGET),
        'rows': len(rows),
        'by_status': by_status,
        'bom': True, 'line_ending': 'CRLF',
        'vendor_fields': FIELDS,
        'delisted_rows_without_date': 0,
        'delist_dates_total': len(delist_dates),
        'delist_dates_since_20241230': len(after_window_start),
        'delist_dates_since_20241230_list': after_window_start,
        'collected_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    out = os.path.join(ROOT, 'reports', 'delist_table_20260924')
    os.makedirs(out, exist_ok=True)
    io.open(os.path.join(out, 'collection_manifest.json'), 'w',
            encoding='utf-8', newline='').write(
        json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
