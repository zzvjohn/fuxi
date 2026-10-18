# -*- coding: utf-8 -*-
"""Probe why 61 old name-days are absent from the new pull, and whether the vendor's
per-session call truncates.

Read-only. Three questions:
  1. does the vendor still return the .BJ names the old file carried?
  2. does the endpoint cap rows per call?
  3. do the 4 cells where new < old look like revisions?
"""
import sys
import time

sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_token  # noqa: E402
import tushare as ts  # noqa: E402

ts.set_token(get_tushare_token())
pro = ts.pro_api()

for day in ('20250220', '20250221', '20260123', '20250411'):
    frame = pro.block_trade(trade_date=day)
    if frame is None:
        print('=== %s  None' % day)
        continue
    codes = frame.ts_code.astype(str)
    suffix = codes.str.slice(-3)
    print('=== %s rows=%d  by suffix %s' % (day, len(frame), suffix.value_counts().to_dict()))
    bj = codes[suffix == '.BJ']
    print('    .BJ names: %s' % (list(bj[:8]) if len(bj) else 'NONE'))
    if day == '20250220':
        print('    836247.BJ present: %s' % bool((codes == '836247.BJ').any()))
        one = pro.block_trade(ts_code='836247.BJ', start_date='20250201', end_date='20250228')
        print('    per-code pull 836247.BJ: %s rows'
              % (0 if one is None else len(one)))
        if one is not None and len(one):
            print(one.head(5).to_string())
    time.sleep(0.6)

# row cap probe: a session with many prints, pulled two ways
day = '20260923'
a = pro.block_trade(trade_date=day)
print('=== cap probe %s: rows=%d' % (day, 0 if a is None else len(a)))
if a is not None and len(a):
    # paginate explicitly through offset to see whether the default call was capped
    total = []
    for offset in range(0, 600, 200):
        chunk = pro.block_trade(trade_date=day, offset=offset, limit=200)
        n = 0 if chunk is None else len(chunk)
        total.append(n)
        print('    offset=%d limit=200 -> %d rows' % (offset, n))
        time.sleep(0.4)
    print('    paginated total %d vs single call %d' % (sum(total), len(a)))
