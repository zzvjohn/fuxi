# -*- coding: utf-8 -*-
"""P1 of the drought follow-up: make the block-trade family's coverage complete.

Why this file exists
--------------------
`data/raw/block_trade_full.csv` is NOT a daily history. Audited in this batch, inside
the evaluation window (2024-12-01 .. 2026-09-01, 426 SSE sessions) it holds only 155
sessions, in blocks of 8-10 sessions separated by 8-28 session gaps, and every January
of 2020-2024 is absent outright. The IC-ceiling batch (P0) measured the family on
37/87 sections and had to return CEILING_UNCLEAR for exactly that reason: the sample
was too thin to rule the family in or out.

So before any hypothesis about block trades is worth testing, the coverage has to be
repaired. This collector pulls the vendor's own per-session table (trade_date == the
session, which is the documented call shape) over every SSE open day in the requested
range and writes a complete file at a NEW path. The old file is left untouched: it is
a merged artefact with its own repair history, and overwriting it would destroy the
only evidence of what P0 actually read.

Contracts honoured
------------------
* range is derived, not pinned: the default end is the last SSE open day carried by the
  local price panel, so this file does not silently stop advancing the way
  `collect_daily_full.py` did;
* utf-8-sig + CRLF, matching `block_trade_full.csv` and `delist_dates.csv`;
* idempotent: identical content leaves the file alone; changed content keeps the
  previous bytes next to it;
* byte-level read-back assertions, and a manifest with a SHA256.

Usage
-----
  python fuxi_collect_blocktrade_20260924.py --dry-run
  python fuxi_collect_blocktrade_20260924.py
Env overrides: FUXI_BT_CAL_START, FUXI_BT_CAL_END (both YYYYMMDD).
"""
import argparse
import csv
import datetime
import hashlib
import io
import os
import sys
import time

sys.path.insert(0, 'e:/quant')

ROOT = 'e:/quant'
RAW = os.path.join(ROOT, 'data', 'raw')
CALENDAR = os.path.join(RAW, 'trade_calendar.csv')
PRICES = os.path.join(RAW, 'daily_prices.csv')
TARGET = os.path.join(RAW, 'block_trade_daily.csv')
MANIFEST = os.path.join(RAW, 'block_trade_daily.manifest.json')
COLUMNS = ['ts_code', 'trade_date', 'price', 'vol', 'amount', 'buyer', 'seller']
DEFAULT_START = '20241201'
MAX_RETRY = 4
SLEEP = 0.35


def sha256(path):
    digest = hashlib.sha256()
    with io.open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def open_days(lo, hi):
    days = []
    with io.open(CALENDAR, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('exchange') != 'SSE' or str(row.get('is_open')).strip() != '1':
                continue
            day = str(row['cal_date']).strip().replace('-', '')[:8]
            if lo <= day <= hi:
                days.append(day)
    return sorted(set(days))


def price_ceiling():
    last = None
    with io.open(PRICES, encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = header.index('trade_date')
        for row in reader:
            value = str(row[index]).strip().replace('-', '')[:8]
            if last is None or value > last:
                last = value
    return last


def fetch_day(pro, day):
    """One vendor call, with bounded retries. Raises on persistent failure."""
    last_error = None
    for attempt in range(MAX_RETRY):
        try:
            frame = pro.block_trade(trade_date=day)
            return frame
        except Exception as exc:  # noqa: BLE001 - the vendor raises many shapes
            last_error = exc
            wait = 1.5 * (attempt + 1)
            print('    retry %d/%d on %s: %s' % (attempt + 1, MAX_RETRY, day, exc), flush=True)
            time.sleep(wait)
    raise RuntimeError('%s failed after %d attempts: %s' % (day, MAX_RETRY, last_error))


def serialise(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator='\r\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    text = buffer.getvalue()
    assert '\r\n' in text and text.count('\n') == text.count('\r\n'), 'not pure CRLF'
    return text.encode('utf-8-sig')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='list the sessions that would be pulled, then stop')
    args = parser.parse_args()

    start = os.environ.get('FUXI_BT_CAL_START', DEFAULT_START).strip()
    end = os.environ.get('FUXI_BT_CAL_END', '').strip() or price_ceiling()
    assert end is None or len(end) == 8, 'unusable price ceiling: %r' % end
    assert start <= end, 'start %s after end %s' % (start, end)
    days = open_days(start, end)
    print('[collect] block_trade  %s .. %s  ->  %d SSE sessions' % (start, end, len(days)),
          flush=True)
    if args.dry_run:
        print('[dry-run] first %s  last %s' % (days[:5], days[-5:]))
        return

    from credentials import get_tushare_token
    import tushare as ts

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()

    rows, per_day, failed, empty = [], {}, [], []
    for position, day in enumerate(days, 1):
        frame = fetch_day(pro, day)
        count = 0
        if frame is not None and len(frame):
            for record in frame.to_dict('records'):
                rows.append({column: ('' if record.get(column) is None
                                      else str(record.get(column)).strip())
                             for column in COLUMNS})
                count += 1
        per_day[day] = count
        if count == 0:
            empty.append(day)
        if position % 50 == 0 or position == len(days):
            print('  %4d/%d  %s  rows so far %d  empty days %d'
                  % (position, len(days), day, len(rows), len(empty)), flush=True)
        time.sleep(SLEEP)

    total = len(days)
    covered = total - len(empty)
    print('[collect] rows=%d  sessions_with_trades=%d/%d  empty=%d  failed=%d'
          % (len(rows), covered, total, len(empty), len(failed)), flush=True)
    # A vendor table that answers on almost no session is a permission or outage
    # problem, not a finding; refuse to write it out as if it were data.
    assert covered >= 0.80 * total, \
        'only %d/%d sessions answered; refusing to write a partial pull' % (covered, total)
    assert rows, 'no rows at all'

    # The vendor emits one row per counterparty leg. Measured in this batch: 6-17 groups
    # per session share (code, price, vol) with different buyer/seller, plus 13-26 rows
    # that repeat on ALL seven fields. Those two are different things:
    #   * a leg with a different counterparty is a real, additional print -> KEEP, so
    #     `bt_amount` sums to the true block turnover;
    #   * a row identical on all seven fields is a feed artefact -> FOLD.
    # `block_trade_repair.py`, which produced the file this one replaces, folded on
    # (ts_code, trade_date, price, vol) instead, which silently deleted real legs and
    # biased the old file's bt_share downwards. The old file is left untouched as
    # evidence; see reports/blocktrade_p1_20260924/defect_round1_subset_and_codes/.
    key_of = lambda r: (r['ts_code'], r['trade_date'], r['price'], r['vol'],
                        r['amount'], r['buyer'], r['seller'])
    seen, deduped, folded = set(), [], 0
    for row in rows:
        key = key_of(row)
        if key in seen:
            folded += 1
            continue
        seen.add(key)
        deduped.append(row)
    legs_kept = len(deduped) - len({(r['ts_code'], r['trade_date'], r['price'], r['vol'])
                                    for r in deduped})
    print('[collect] vendor rows %d -> written %d (folded %d identical, kept %d extra '
          'counterparty legs)' % (len(rows), len(deduped), folded, legs_kept), flush=True)

    ordered = sorted(deduped, key=lambda r: (r['trade_date'], r['ts_code']))
    payload = serialise(ordered)

    previous = io.open(TARGET, 'rb').read() if os.path.exists(TARGET) else None
    if previous == payload:
        print('[collect] content unchanged; file left untouched', flush=True)
    else:
        if previous is not None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            keep = TARGET + '.bak_' + stamp
            io.open(keep, 'wb').write(previous)
            print('[collect] previous content kept at %s' % os.path.basename(keep), flush=True)
        io.open(TARGET, 'wb').write(payload)
        print('[collect] wrote %s (%d bytes)' % (TARGET, len(payload)), flush=True)
    unique_rows = len(ordered)

    check = io.open(TARGET, 'rb').read()
    assert check == payload, 'written file does not read back'
    assert check[:3] == b'\xef\xbb\xbf', 'missing BOM'
    assert check.count(b'\n') - check.count(b'\r\n') == 0, 'bare LF introduced'

    manifest = {
        'file': 'data/raw/block_trade_daily.csv',
        'sha256': sha256(TARGET),
        'bytes': len(check),
        'api': 'block_trade(trade_date=...)',
        'range': [days[0], days[-1]],
        'sessions_requested': total,
        'sessions_with_trades': covered,
        'sessions_without_trades': empty,
        'rows_returned_by_vendor': len(rows),
        'rows_written': unique_rows,
        'rows_folded_identical_on_all_seven_fields': folded,
        'extra_counterparty_legs_kept': legs_kept,
        'dedup_key': 'all seven columns (ts_code,trade_date,price,vol,amount,buyer,seller)',
        'leg_policy': 'the vendor emits one row per counterparty leg; only rows identical '
                      'on all seven fields are folded, so bt_amount sums the real legs. '
                      'This differs from block_trade_repair.py, which folded on '
                      '(ts_code,trade_date,price,vol) and therefore understated the old '
                      "file's block turnover; see defect_round1_subset_and_codes/.",
        'code_convention': 'ts_code as returned by the vendor today, i.e. Northbound '
                           '(BJ) names use the 920xxx codes that data/raw/daily_prices.csv '
                           'also uses. The old file carries pre-migration codes such as '
                           '836247.BJ that no longer join the price panel.',
        'per_day_rows': per_day,
        'prior_source': 'data/raw/block_trade_full.csv (partial; left untouched)',
        'collected_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    import json
    MANIFEST_TEXT = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    io.open(MANIFEST, 'wb').write(MANIFEST_TEXT.encode('utf-8'))
    print('[collect] rows written %d (folded %d identical, kept %d extra legs)  manifest -> %s'
          % (unique_rows, folded, legs_kept, MANIFEST), flush=True)


if __name__ == '__main__':
    main()
