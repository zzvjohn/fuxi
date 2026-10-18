# -*- coding: utf-8 -*-
"""Diagnose why the source swap is NOT a pure coverage change (read-only).

The reconcile assertion in fuxi_blocktrade_p1_20260924 failed three ways:
  * name sets differ on shared days (old 23 vs new 68 on 20250718; old 3 vs new 79)
  * bt_share differs where name sets match (20250411: 0.0072)

Both point at the same suspicion: the old file is partial in TWO dimensions (whole days
missing, and truncated name sets on the days it does carry), and both files collapse the
vendor's per-counterparty rows, so bt_amount understates the true block turnover.

This script measures exactly that at the name-day level, so the correction is derived
from evidence rather than guessed.
"""
import csv
import os
import sys
from collections import defaultdict

ROOT = 'e:/quant'
RAW = os.path.join(ROOT, 'data', 'raw')

import pandas as pd  # noqa: E402


def load(path, label):
    df = pd.read_csv(path, dtype=str)
    df['day'] = df.trade_date.astype(str).str.replace('-', '', regex=False).str.slice(0, 8)
    for column in ('price', 'vol', 'amount'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df[df.day.str.startswith(('2025', '2026'))]
    print('[%s] rows=%d days=%d names=%d' % (label, len(df), df.day.nunique(),
                                             df.ts_code.nunique()))
    return df


old = load(os.path.join(RAW, 'block_trade_full.csv'), 'old')
new = load(os.path.join(RAW, 'block_trade_daily.csv'), 'new')

old_days = set(old.day)
new_days = set(new.day)
shared = sorted(old_days & new_days)
print('\nold days %d  new days %d  shared %d  new-only %d  old-only %d'
      % (len(old_days), len(new_days), len(shared), len(new_days - old_days),
         len(old_days - new_days)))

# vendor legs per (day, code, price, vol) in the new file
new_legs = (new.groupby(['day', 'ts_code', 'price', 'vol'], dropna=False)
            .size().rename('legs').reset_index())
multi = new_legs[new_legs.legs > 1]
print('new: (day,code,price,vol) groups with >1 leg: %d covering %d rows; max legs %s'
      % (len(multi), int(multi.legs.sum()), int(new_legs.legs.max())))

old_keys = set(zip(old.day, old.ts_code))
new_keys = set(zip(new.day, new.ts_code))
print('\nname-day: old %d  new %d  shared %d  new-only %d  OLD-ONLY %d'
      % (len(old_keys), len(new_keys), len(old_keys & new_keys),
         len(new_keys - old_keys), len(old_keys - new_keys)))
if old_keys - new_keys:
    missing = sorted(old_keys - new_keys)
    print('  old-only examples (a vendor revision would be a real problem): %s' % missing[:10])

# name-day amount comparison on shared days
old_amt = old.groupby(['day', 'ts_code']).amount.sum()
new_amt = new.groupby(['day', 'ts_code']).amount.sum()
both = old_amt.index.intersection(new_amt.index)
cmp = pd.DataFrame({'old': old_amt.loc[both], 'new': new_amt.loc[both]})
cmp['new_minus_old'] = cmp['new'] - cmp['old']
cmp['new_over_old'] = cmp['new'] / cmp['old'].replace(0, pd.NA)
print('\nname-day amount, shared %d cells:' % len(cmp))
print('  new < old  : %d' % int((cmp.new_minus_old < -1e-9).sum()))
print('  new == old : %d' % int((cmp.new_minus_old.abs() <= 1e-9).sum()))
print('  new > old  : %d' % int((cmp.new_minus_old > 1e-9).sum()))
print('  recovered amount on those cells: %.2f (old total %.2f, new total %.2f)'
      % (cmp[cmp.new_minus_old > 0].new_minus_old.sum(), cmp.old.sum(), cmp['new'].sum()))

# per-day name counts on shared days
oc = old[old.day.isin(shared)].groupby('day').ts_code.nunique()
nc = new[new.day.isin(shared)].groupby('day').ts_code.nunique()
comp = pd.DataFrame({'old': oc, 'new': nc}).dropna()
comp['delta'] = comp['new'] - comp['old']
print('\nshared days %d; names gained per day: min %d  median %d  max %d'
      % (len(comp), comp.delta.min(), comp.delta.median(), comp.delta.max()))
print('  days where new < old: %d' % int((comp.delta < 0).sum()))
print(comp.sort_values('delta').head(8).to_string())
print('  worst/typical days:')
print(comp.sort_values('delta', ascending=False).head(4).to_string())

# the 20250411 case: bt_share mismatch with matching name sets
day = '20250411'
if day in old_days and day in new_days:
    o = old[old.day == day].groupby('ts_code').amount.sum()
    n = new[new.day == day].groupby('ts_code').amount.sum()
    j = pd.DataFrame({'old': o, 'new': n}).dropna()
    j['diff'] = j['new'] - j['old']
    print('\n=== %s: name-days %d, amount differs on %d, max diff %.4f'
          % (day, len(j), int((j['diff'].abs() > 1e-9).sum()), j['diff'].abs().max()))
    print('  rows in old for that day: %d ; in new: %d' % (
        len(old[old.day == day]), len(new[new.day == day])))
    print(j[j['diff'].abs() > 1e-9].sort_values('diff', ascending=False).head(5).to_string())
