# -*- coding: utf-8 -*-
"""Probe the vendor's block_trade interface: permission, columns, per-day volume.

Read-only. Answers two questions before we build a collector:
  1. Can we pull one trade_date at a time (the documented call shape)?
  2. How many rows does a typical day return, and does the local table match?
"""
import sys
import time

sys.path.insert(0, 'e:/quant')

from credentials import get_tushare_token  # noqa: E402
import tushare as ts  # noqa: E402

ts.set_token(get_tushare_token())
pro = ts.pro_api()
print('[probe] pro acquired', flush=True)

for day in ('20250102', '20250303', '20260817', '20260818'):
    try:
        frame = pro.block_trade(trade_date=day)
    except Exception as exc:  # noqa: BLE001
        print('[probe] %s  ERROR %s: %s' % (day, type(exc).__name__, exc), flush=True)
        continue
    if frame is None:
        print('[probe] %s  None' % day, flush=True)
    else:
        print('[probe] %s  rows=%d  cols=%s' % (day, len(frame), list(frame.columns)), flush=True)
        if len(frame):
            print('        sample: %s' % frame.head(2).to_dict('records'), flush=True)
    time.sleep(0.5)
