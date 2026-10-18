"""Quick check of data file date ranges"""
import pandas as pd
from pathlib import Path

raw = Path('e:/quant/data/raw')

for fname in ['daily_prices.csv', 'adj_factor.csv', 'daily_basic.csv', 'index_000300_daily.csv']:
    fpath = raw / fname
    if not fpath.exists():
        print(f'{fname}: NOT FOUND')
        continue
    
    size_mb = fpath.stat().st_size / 1024 / 1024
    # Read first row
    first_chunk = pd.read_csv(fpath, nrows=1)
    first_date = first_chunk['trade_date'].iloc[0]
    
    # Read last row efficiently
    with open(fpath, 'rb') as f:
        try:
            f.seek(-4096, 2)  # Go to last 4KB
        except OSError:
            f.seek(0)
        tail_lines = f.read().decode('utf-8', errors='replace').strip().split('\n')
    
    last_line = tail_lines[-1]
    last_date = last_line.split(',')[1] if len(last_line.split(',')) > 1 else 'unknown'
    
    print(f'{fname}: {size_mb:.0f}MB, first={first_date}, last={last_date}')
