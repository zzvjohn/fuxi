# -*- coding: utf-8 -*-
"""Extend `data/raw/suspend_d.csv` to a common end date.

Root cause this addresses: `strategies/final/collect_daily_full.py` hard-codes
the trading-calendar end as 20260430, so the suspension table stops there. A
table that stops early is worse than a table that is absent: the freeze
derivation reads contiguous coverage to the *table* edge as coverage to the
*evidence* edge, so it emits a freeze date equal to the first session after the
last stored row. That is a fabricated date dressed as a derived one.

Append-only, idempotent, backup first. Existing rows are never rewritten. New
rows mirror the stored format: utf-8-sig, CRLF, `ts_code,trade_date,
suspend_timing,suspend_type` with plain YYYYMMDD dates.
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
SUSPEND = os.path.join(RAW, 'suspend_d.csv')
COVERAGE = os.path.join(ROOT, 'reports', 'suspend_coverage_20260924')
COLUMNS = ['ts_code', 'trade_date', 'suspend_timing', 'suspend_type']
FIELDS = 'ts_code,trade_date,suspend_timing,suspend_type'
TARGET_END = '20260923'


def sha256(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_existing():
    with io.open(SUSPEND, 'rb') as handle:
        payload = handle.read()
    assert payload[:3] == b'\xef\xbb\xbf', 'suspend_d.csv lost its BOM'
    assert payload.count(b'\n') - payload.count(b'\r\n') == 0, 'suspend_d.csv has bare LF'
    text = payload[3:].decode('utf-8')
    reader = csv.reader(io.StringIO(text, newline=''))
    header = next(reader)
    assert header == COLUMNS, 'unexpected suspend_d header: %s' % header
    rows, dates = [], {}
    for row in reader:
        if not row or not row[0]:
            continue
        assert len(row) == len(COLUMNS), row
        rows.append(row)
        dates.setdefault(row[1], 0)
        dates[row[1]] += 1
    return payload, rows, dates


def sessions_between(pro, start, end):
    cal = pro.trade_cal(exchange='SSE', start_date=start, end_date=end)
    days = sorted(cal[cal.is_open == 1].cal_date.astype(str).tolist())
    return days


def main():
    from credentials import get_tushare_token
    import tushare as ts

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    pro.__timeout = 120

    payload, rows, dates = read_existing()
    have = sorted(dates)
    print('suspend_d.csv: %d rows, %d dated days, %s .. %s'
          % (len(rows), len(dates), have[0], have[-1]), flush=True)
    assert have[-1] <= TARGET_END, 'table already runs past the target end'

    wanted = sessions_between(pro, have[-1], TARGET_END)
    missing = [d for d in wanted if d not in dates]
    print('sessions %s..%s: %d; already dated %d; to collect %d'
          % (have[-1], TARGET_END, len(wanted), len(wanted) - len(missing),
             len(missing)), flush=True)

    coverage = {}
    new_rows = []
    empty_days, failed = [], []
    for day in missing:
        time.sleep(0.35)
        frame = None
        for attempt in range(3):
            try:
                frame = pro.suspend_d(trade_date=day, fields=FIELDS)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    failed.append({'day': day, 'error': repr(exc)})
                time.sleep(4 * (attempt + 1))
        if frame is None:
            continue
        records = frame.to_dict('records')
        coverage[day] = len(records)
        if not records:
            empty_days.append(day)
            continue
        for record in records:
            new_rows.append([str(record.get('ts_code') or '').strip(),
                             str(record.get('trade_date') or day).strip(),
                             '' if record.get('suspend_timing') is None
                             else str(record['suspend_timing']).strip(),
                             '' if record.get('suspend_type') is None
                             else str(record['suspend_type']).strip()])
    print('collected %d rows over %d days; %d days empty; %d failed'
          % (len(new_rows), len(coverage), len(empty_days), len(failed)), flush=True)
    assert not failed, 'collection failed on %s' % failed[:3]

    known = set(zip([r[0] for r in rows], [r[1] for r in rows]))
    incoming = [r for r in new_rows if (r[0], r[1]) not in known]
    duplicated = len(new_rows) - len(incoming)
    assert duplicated == 0, '%d incoming rows duplicate stored (code,date)' % duplicated
    for row in incoming:
        assert row[1] in missing, 'row not in the requested gap: %s' % row
        assert row[3] in ('S', 'R'), 'unexpected suspend_type: %s' % row

    if not incoming:
        print('nothing new to append', flush=True)
    else:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        keep = SUSPEND + '.bak_' + stamp
        io.open(keep, 'wb').write(payload)
        print('backup written to %s' % os.path.basename(keep), flush=True)
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer, lineterminator='\r\n')
        for row in incoming:
            writer.writerow(row)
        addition = buffer.getvalue()
        assert addition.count('\n') == addition.count('\r\n'), 'addition is not pure CRLF'
        io.open(SUSPEND, 'ab').write(addition.encode('utf-8'))
        print('appended %d rows (%d bytes)' % (len(incoming), len(addition)), flush=True)

    check, rows_after, dates_after = read_existing()
    assert len(rows_after) == len(rows) + len(incoming), 'row count did not grow as expected'
    assert sorted(dates_after)[-1] == TARGET_END, sorted(dates_after)[-1]
    for day in missing:
        assert day in dates_after or day in empty_days, 'day %s was never settled' % day
    assert check.count(b'\n') - check.count(b'\r\n') == 0, 'bare LF after append'

    os.makedirs(COVERAGE, exist_ok=True)
    manifest = {
        'file': 'data/raw/suspend_d.csv',
        'sha256_before': hashlib.sha256(payload).hexdigest(),
        'sha256_after': sha256(SUSPEND),
        'rows_before': len(rows),
        'rows_after': len(rows_after),
        'rows_appended': len(incoming),
        'dated_days_before': len(dates),
        'dated_days_after': len(dates_after),
        'max_date_before': have[-1],
        'max_date_after': sorted(dates_after)[-1],
        'target_end': TARGET_END,
        'sessions_requested': len(missing),
        'sessions_with_rows': len(coverage),
        'sessions_empty': empty_days,
        'per_day_rows': coverage,
        'root_cause': ('strategies/final/collect_daily_full.py line 74 pins the '
                       'trading calendar end_date to 20260430'),
        'format': {'encoding': 'utf-8-sig', 'line_ending': 'CRLF',
                   'date_format': 'YYYYMMDD', 'columns': COLUMNS},
        'collected_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    io.open(os.path.join(COVERAGE, 'coverage_manifest.json'), 'w',
            encoding='utf-8', newline='').write(
        json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: manifest[k] for k in
                      ('rows_appended', 'max_date_before', 'max_date_after',
                       'sessions_requested', 'sessions_with_rows',
                       'sha256_after')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
