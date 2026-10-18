#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neutralize C-grade factors via Ridge regression
"""

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
FACTOR_DIR = BASE / 'research' / 'factor_alchemy' / 'output'

print('Loading style factors...')
weekly = pd.read_parquet(BASE / 'output/ap_batch/cache/weekly_prices.parquet')
weekly['trade_date'] = pd.to_datetime(weekly['trade_date'])
weekly = weekly[['stock_code', 'trade_date', 'momentum_12_1', 'vol_60', 'log_size']]
print(f'  Weekly: {len(weekly):,} rows, {weekly["trade_date"].nunique()} periods')

def neutralize_factor(factor_name, input_path):
    df = pd.read_csv(input_path)
    if str(df.columns[0]).startswith('Unnamed'):
        df.rename(columns={df.columns[0]: 'trade_date'}, inplace=True)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    df = df.melt(id_vars='trade_date', var_name='stock_code', value_name=factor_name)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.dropna(subset=[factor_name])

    merged = df.merge(weekly, on=['trade_date', 'stock_code'], how='inner')
    merged = merged.dropna(subset=['momentum_12_1', 'vol_60', 'log_size'])

    dates = sorted(merged['trade_date'].unique())
    neutralized = []

    for date in dates:
        sub = merged[merged['trade_date'] == date].copy()
        if len(sub) < 100:
            continue

        vals = sub[factor_name].values
        X = sub[['log_size', 'vol_60', 'momentum_12_1']].values

        # MAD
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        lo, hi = median - 3 * mad, median + 3 * mad
        vals = np.clip(vals, lo, hi)

        # Ridge
        ss = StandardScaler()
        Xs = ss.fit_transform(X)
        model = Ridge(alpha=1.0).fit(Xs, vals)
        y_hat = model.predict(Xs)
        resid = vals - y_hat

        # Zscore
        m, s = resid.mean(), resid.std()
        if s > 0:
            resid = (resid - m) / s
        else:
            resid = np.zeros_like(resid)

        sub['val'] = resid
        neutralized.append(sub[['trade_date', 'stock_code', 'val']])

    if not neutralized:
        return None

    result = pd.concat(neutralized, ignore_index=True)
    result = result.rename(columns={'val': f'{factor_name}_neut'})

    # long -> wide
    wide = result.pivot_table(
        values=f'{factor_name}_neut', index='trade_date', columns='stock_code'
    )

    out_path = FACTOR_DIR / f'factor_{factor_name}_neut.csv'
    wide.to_csv(out_path, encoding='utf-8-sig')

    return {
        'factor': factor_name,
        'weeks': len(dates),
        'path': str(out_path),
    }

TARGETS = [
    'dollar_vol_20d', 'davol_20', 'abnormal_turnover',
    'bp', 'atr_14', 'vol_1m', 'ret_3m', 'avg_turnover_1m',
    'vol_3m', 'max_ret_1m', 'downside_vol',
]

print(f'\nNeutralizing {len(TARGETS)} factors...')
results = []
for fac in TARGETS:
    csv_path = FACTOR_DIR / f'factor_{fac}.csv'
    if not csv_path.exists():
        csv_path = BASE / 'research/factor_alchemy/output/icir_biased' / f'factor_{fac}.csv'
    if not csv_path.exists():
        print(f'  {fac}: file not found, skip')
        continue

    print(f'  {fac}...', end=' ', flush=True)
    r = neutralize_factor(fac, str(csv_path))
    if r:
        results.append(r)
        print(f'{r["weeks"]} weeks OK')
    else:
        print('FAILED')

print(f'\nDone: {len(results)}/{len(TARGETS)} factors neutralized')

# Register in config
print('\nRegistering in config.py...')
config_path = BASE / 'research/factor_alchemy/config.py'
with open(config_path, 'r', encoding='utf-8') as f:
    config = f.read()

insert_pos = config.find("# neutralized")
if insert_pos < 0:
    insert_pos = config.find("# jqfactor")

neut_entries = "\n    # ---- Ridge-neutralized C-grade factors ----\n"
for r in results:
    fac = r['factor']
    neut_entries += f"    '{fac}_neut': {{'category': 'neutralized', 'label': '{fac}_neut'}},\n"
neut_entries += "\n"

if insert_pos > 0:
    config = config[:insert_pos] + neut_entries + config[insert_pos:]

with open(config_path, 'w', encoding='utf-8') as f:
    f.write(config)

print(f'Added {len(results)} neutralized factors to FACTOR_DEFS')
names = [r['factor'] + '_neut' for r in results]
print(f'Factors: {names}')
