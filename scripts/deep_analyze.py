import pandas as pd, numpy as np

COL_NAMES = ['日期','品种','标的','多空','数量','可用数量','收盘价',
             '市值','逐笔浮盈','开仓均价','持仓均价','保证金',
             '当日盈亏','今手数','盈亏占比','净值','仓位占比']

def read_positions(path):
    df = pd.read_csv(path, encoding='gbk', header=0, names=COL_NAMES, skiprows=1, dtype=str)
    df = df[df['品种'] == '股票'].copy()
    for col in ['收盘价','市值','逐笔浮盈','当日盈亏','净值']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    return df.dropna(subset=['日期'])

print("=" * 70)
print(">>> Deep Analysis of position_none.csv (real strategy) <<<")
print("=" * 70)

s = read_positions('E:/position2/position_none.csv')
s['code'] = s['标的'].str.extract(r'\((\d+)\.')[0]
s['pnl_pct'] = s['逐笔浮盈'] / s['净值'].shift(1).where(s['日期'] == s['日期'].shift(1), np.nan)

# --- 1. Price characteristics of winners vs losers ---
s['daily_pl'] = s['逐笔浮盈']
stock_stats = s.groupby('code').agg(
    total_pl=('daily_pl','sum'),
    days=('日期','nunique'),
    avg_price=('收盘价','mean'),
    max_price=('收盘价','max'),
    min_price=('收盘价','min')
)

# Split into winners and losers
winners = stock_stats[stock_stats['total_pl'] > 50000].copy()
losers = stock_stats[stock_stats['total_pl'] < -50000].copy()

winners['price_range'] = winners['max_price'] - winners['min_price']
winners['volatility'] = winners['price_range'] / winners['avg_price']
losers['price_range'] = losers['max_price'] - losers['min_price']
losers['volatility'] = losers['price_range'] / losers['avg_price']

print("\n--- Price Characteristics ---")
print(f"Winners (n={len(winners)}): avg_price={winners['avg_price'].mean():.1f}, "
      f"volatility={winners['volatility'].mean():.2f}, days={winners['days'].mean():.1f}")
print(f"Losers  (n={len(losers)}): avg_price={losers['avg_price'].mean():.1f}, "
      f"volatility={losers['volatility'].mean():.2f}, days={losers['days'].mean():.1f}")

# Price distribution of winners
print("\nWinner price distribution:")
for rng, label in [((0,10),'<10'), ((10,20),'10-20'), ((20,50),'20-50'), ((50,100),'50-100'), ((100,999),'100+')]:
    cnt = len(winners[(winners['avg_price'] >= rng[0]) & (winners['avg_price'] < rng[1])])
    print(f"  Y{label}: {cnt} ({cnt/max(len(winners),1)*100:.0f}%)")

# --- 2. Top gainers summary ---
print("\n--- Top 30 Individual Winners ---")
top30 = stock_stats.nlargest(30, 'total_pl')
for i, (code, row) in enumerate(top30.iterrows()):
    print(f"  {i+1:2d}. {code}: +{int(row['total_pl']):>10,} ({int(row['days'])}d, Y{row['avg_price']:.1f})")

# --- 3. Stage analysis: profit vs market phase ---
s['yr'] = s['日期'].dt.year
yr_stats = s.groupby('yr').agg(
    total_pl=('daily_pl','sum'),
    days=('日期','nunique'),
    stocks=('code','nunique'),
    avg_price=('收盘价','mean')
)
print("\n--- Yearly Performance ---")
for yr, row in yr_stats.iterrows():
    print(f"  {yr}: P&L={row['total_pl']:>12,.0f}  days={int(row['days'])}  stocks={int(row['stocks'])}  avg_price={row['avg_price']:.1f}")

# --- 4. Holding duration vs return ---
print("\n--- Hold Duration vs Return ---")
for days_range, label in [(1,'1d'), (2,'2d'), (3,'3-4d'), (5,'5-7d'), (8,'8-14d'), (15,'15-30d'), (31,'31d+')]:
    sub = stock_stats[(stock_stats['days'] >= days_range[0]) & (stock_stats['days'] <= (days_range[1] if isinstance(days_range,tuple) else 999))]
    print(f"  {label:>6s}: n={len(sub):4d}, avg_pl={sub['total_pl'].mean():>10,.0f}, med_pl={sub['total_pl'].median():>10,.0f}")

# --- 5. fakewin analysis ---
print("\n\n" + "=" * 70)
print(">>> Deep Analysis of position_fakewin.csv <<<")
print("=" * 70)

f = read_positions('E:/position2/position_fakewin.csv')
f['code'] = f['标的'].str.extract(r'\((\d+)\.')[0]

f_stats = f.groupby('code').agg(
    total_pl=('逐笔浮盈','sum'),
    days=('日期','nunique'),
    avg_price=('收盘价','mean'),
    max_price=('收盘价','max'),
    min_price=('收盘价','min')
)

f_winners = f_stats[f_stats['total_pl'] > 1000000].copy()  # >1M
f_winners['price_range'] = f_winners['max_price'] - f_winners['min_price']
f_winners['volatility'] = f_winners['price_range'] / f_winners['avg_price']

print(f"\nFake-win super winners (PL>1M, n={len(f_winners)}):")
print(f"  avg_price={f_winners['avg_price'].mean():.1f}")
print(f"  avg_days={f_winners['days'].mean():.1f}")
print(f"  avg_volatility={f_winners['volatility'].mean():.2f}")

# Price distribution of fake-win mega winners
print("\nFake-win price distribution (PL>1M):")
for rng, label in [((0,5),'<5'), ((5,10),'5-10'), ((10,20),'10-20'), ((20,50),'20-50'), ((50,100),'50-100'), ((100,999),'100+')]:
    cnt = len(f_winners[(f_winners['avg_price'] >= rng[0]) & (f_winners['avg_price'] < rng[1])])
    print(f"  Y{label}: {cnt} ({cnt/max(len(f_winners),1)*100:.0f}%)")

# Top 15 mega winners
print("\nTop 15 Fake-win Mega Winners:")
top15f = f_stats.nlargest(15, 'total_pl')
for i, (code, row) in enumerate(top15f.iterrows()):
    print(f"  {i+1:2d}. {code}: +{int(row['total_pl']):>12,} ({int(row['days'])}d, Y{row['avg_price']:.1f}, vol={row['max_price']/row['min_price']-1:.1%})")

# --- 6. Cross-analysis: what fake-win picked that none didn't ---
f_codes = set(f['code'].unique())
s_codes = set(s['code'].unique())
fake_only = f_codes - s_codes
none_only = s_codes - f_codes

fake_only_stats = f_stats.loc[f_stats.index.isin(fake_only)].sort_values('total_pl', ascending=False)
print(f"\n--- Cross Analysis ---")
print(f"Fake-only stocks: {len(fake_only)}, total_pl={fake_only_stats['total_pl'].sum():,.0f}")
print(f"None-only stocks: {len(none_only)}")
print(f"\nTop 10 fake-only contributors:")
for i, (code, row) in enumerate(fake_only_stats.head(10).iterrows()):
    print(f"  {code}: +{int(row['total_pl']):>12,} ({int(row['days'])}d, Y{row['avg_price']:.1f})")
