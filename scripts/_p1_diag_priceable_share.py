# -*- coding: utf-8 -*-
"""What share of block prints is priceable in the FULL window (read-only)?

The P1 swap-audit asserts >= 0.995. A 3-session stub failed that at 0.68, which may mean
either the stub is unrepresentative or the threshold was invented. Measure the real
number, and look at WHY the remainder is unpriceable, before touching either.
"""
import sys
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
sys.path.insert(0, 'e:/quant/research/factor_alchemy')
from fuxi_execution_lab_20260923 import load_panel  # noqa: E402
import fuxi_ic_ceiling_20260924 as ic  # noqa: E402
import fuxi_blocktrade_p1_20260924 as p1  # noqa: E402

price, _ = load_panel('data/raw/daily_prices.csv',
                      ['ts_code', 'trade_date', 'open', 'close', 'amount'], lower='20241201')
adj, _ = load_panel('data/raw/adj_factor_daily.csv',
                    ['ts_code', 'trade_date', 'adj_factor'], lower='20241201')
merged = price.merge(adj, on=['trade_date', 'ts_code'], validate='one_to_one', how='left')
byday = {d: g.set_index('ts_code') for d, g in merged.groupby('trade_date')}
print('price panel days %d' % len(byday))

stamps = {d for d in set(ic._daykey(pd.Series(byday.keys())))}
# use the protocol's 87 stamps
import json
proto = json.load(open('e:/quant/reports/ic_ceiling_20260924/protocol.json', encoding='utf-8'))
stamps = set(s.replace('-', '') for s in proto['signal_dates'])
print('signal stamps %d' % len(stamps))

for label, path in (('new', p1.BLOCK_DAILY), ('old', p1.BLOCK_OLD)):
    frame = p1.load_block_rows(path, stamps)
    if not len(frame):
        print('[%s] no rows in window' % label)
        continue
    pairs = p1.priceable_index(byday)
    keys = list(zip(frame.day.map(ic._iso), frame.ts_code))
    mask = [k in pairs for k in keys]
    share = sum(mask) / len(mask)
    print('\n[%s] rows=%d priceable=%d share=%.4f' % (label, len(frame), sum(mask), share))
    bad = frame[[not m for m in mask]]
    if len(bad):
        print('  unpriceable by suffix: %s'
              % bad.ts_code.str.slice(-3).value_counts().to_dict())
        # is the name absent from the panel entirely, or only from that session?
        all_codes = set()
        for table in byday.values():
            all_codes.update(table.index)
        absent = bad[~bad.ts_code.isin(all_codes)]
        print('  unpriceable AND absent from panel in every session: %d (%.4f of bad)'
              % (len(absent), len(absent) / len(bad)))
        print('  unpriceable but present on some session (halt/new listing?): %d'
              % (len(bad) - len(absent)))
        worst = bad.groupby('day').size().sort_values(ascending=False).head(8)
        print('  worst days by unpriceable rows: %s' % worst.to_dict())
        ex = bad.groupby('ts_code').size().sort_values(ascending=False).head(8)
        print('  worst codes: %s' % ex.to_dict())
