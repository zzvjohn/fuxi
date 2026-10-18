# -*- coding: utf-8 -*-
"""Diagnose the duplicate shape before changing the collector's dedup key.

Hypothesis: one block print can be matched against several counterparties, so
(code, date, price, vol) repeats legitimately while (buyer, seller) differs.
Read-only: pulls a few sessions and reports the multiplicity structure.
"""
import sys
import time
from collections import Counter

sys.path.insert(0, 'e:/quant')
from credentials import get_tushare_token  # noqa: E402
import tushare as ts  # noqa: E402

ts.set_token(get_tushare_token())
pro = ts.pro_api()

for day in ('20250303', '20250715', '20260923'):
    frame = pro.block_trade(trade_date=day)
    print('=== %s rows=%d' % (day, 0 if frame is None else len(frame)), flush=True)
    if frame is None or not len(frame):
        continue
    df = frame.astype(str)
    full = df.duplicated(subset=['ts_code', 'price', 'vol', 'buyer', 'seller'],
                         keep=False).sum()
    print('   exact-duplicate on (code,price,vol,buyer,seller): %d' % full)
    key = ['ts_code', 'price', 'vol']
    multi = df.groupby(key, dropna=False).size()
    multi = multi[multi > 1]
    print('   (code,price,vol) groups with >1 row: %d covering %d rows'
          % (len(multi), int(multi.sum())))
    if len(multi):
        g = df.set_index(key).loc[multi.index[0]]
        print('   example of one such group:')
        print(g.to_string())
    amount_note = df.groupby(['ts_code'], dropna=False).size()
    print('   names on the session: %d, rows/name max %d'
          % (len(amount_note), int(amount_note.max())))
    time.sleep(0.4)
