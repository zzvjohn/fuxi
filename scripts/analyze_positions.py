import pandas as pd, numpy as np
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)

# ============= Common read function =============
# CSV has 17 data columns but only 16 headers; last 2 are 净值,仓位占比
COL_NAMES = ['日期','品种','标的','多空','数量','可用数量','收盘价',
             '市值','逐笔浮盈','开仓均价','持仓均价','保证金',
             '当日盈亏','今手数','盈亏占比','净值','仓位占比']

def read_positions(path):
    df = pd.read_csv(path, encoding='gbk', header=0, names=COL_NAMES, skiprows=1, dtype=str)
    # Filter to stock rows only, exclude Cash rows
    df = df[df['品种'] == '股票'].copy()
    # Convert numeric cols
    for col in ['收盘价','市值','逐笔浮盈','当日盈亏','净值']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期'])
    return df

# ============= position_none.csv analysis =============
print("=" * 60)
print(">>> position_none.csv (VOL_CONFIRM_RATIO=None) <<<")
print("=" * 60)

stocks = read_positions('E:/position2/position_none.csv')

before = stocks[stocks['日期'] < '2022-06-01']
after = stocks[stocks['日期'] >= '2022-06-01']

print(f'\n--- Before June 2022 ---')
print(f'Trade days: {before["日期"].nunique()}')
print(f'Total P&L: {before["逐笔浮盈"].sum():,.0f}')
print(f'Unique stocks held: {before["标的"].nunique()}')
print(f'Average price: {before["收盘价"].mean():.1f}')

bp = before.groupby('标的').agg(
    total_pl=('逐笔浮盈', 'sum'),
    days=('日期', 'nunique'),
    avg_price=('收盘价', 'mean')
).sort_values('total_pl', ascending=False)

print('\nTop 10 winners:')
for i, (idx, row) in enumerate(bp.head(10).iterrows()):
    print(f'  {i+1}. {idx}: +{row.total_pl:,.0f} ({row.days}d, avg Y{row.avg_price:.1f})')

print('\nTop 10 losers:')
for i, (idx, row) in enumerate(bp.tail(10).iterrows()):
    print(f'  {i+1}. {idx}: {row.total_pl:,.0f} ({row.days}d, avg Y{row.avg_price:.1f})')

print(f'\n--- After June 2022 ---')
print(f'Trade days: {after["日期"].nunique()}')
print(f'Total P&L: {after["逐笔浮盈"].sum():,.0f}')
print(f'Unique stocks held: {after["标的"].nunique()}')
print(f'Average price: {after["收盘价"].mean():.1f}')

ap = after.groupby('标的').agg(
    total_pl=('逐笔浮盈', 'sum'),
    days=('日期', 'nunique'),
    avg_price=('收盘价', 'mean')
).sort_values('total_pl', ascending=False)

print('\nTop 10 winners:')
for i, (idx, row) in enumerate(ap.head(10).iterrows()):
    print(f'  {i+1}. {idx}: +{row.total_pl:,.0f} ({row.days}d, avg Y{row.avg_price:.1f})')

print('\nTop 10 losers:')
for i, (idx, row) in enumerate(ap.tail(10).iterrows()):
    print(f'  {i+1}. {idx}: {row.total_pl:,.0f} ({row.days}d, avg Y{row.avg_price:.1f})')

# Monthly breakdown
stocks['月份'] = stocks['日期'].dt.to_period('M')
monthly = stocks.groupby('月份')['逐笔浮盈'].sum()
print('\n--- Monthly P&L ---')
for m, pl in monthly.items():
    mark = ''
    if pl < -50000: mark = ' <<<<<<'
    elif pl > 50000: mark = ' >>>>>>'
    print(f'{m}: {pl:>12,.0f}{mark}')


# ============= position_fakewin.csv analysis =============
print("\n\n" + "=" * 60)
print(">>> position_fakewin.csv (fake-win version) <<<")
print("=" * 60)

stocks2 = read_positions('E:/position2/position_fakewin.csv')

print(f'\nTotal trade days: {stocks2["日期"].nunique()}')
print(f'Total P&L (all time): {stocks2["逐笔浮盈"].sum():,.0f}')
print(f'Unique stocks held: {stocks2["标的"].nunique()}')

# Overall winners/losers
all_profit = stocks2.groupby('标的').agg(
    total_pl=('逐笔浮盈', 'sum'),
    days=('日期', 'nunique'),
    avg_price=('收盘价', 'mean')
).sort_values('total_pl', ascending=False)

print('\nTop 20 ALL-TIME winners:')
for i, (idx, row) in enumerate(all_profit.head(20).iterrows()):
    days_held = row['days']
    daily_pl = row['total_pl'] / max(days_held, 1)
    print(f'  {i+1:2d}. {idx}: +{row.total_pl:>12,.0f} ({days_held:>3d}d, avg Y{row.avg_price:>7.1f}, {daily_pl:>8,.0f}/d)')

# What stock codes did fake-win pick?
codes = stocks2['标的'].str.extract(r'\((\d+)\.')[0]
code_counts = codes.value_counts()
print(f'\nMost held stock codes (prefix analysis):')
for prefix in ['000','002','300','600','601','603','605','688']:
    cnt = sum(1 for c in codes if str(c).startswith(prefix))
    print(f'  {prefix}xxx: {cnt} occurrences')

# Did fake-win hold stocks longer?
avg_hold_days = stocks2.groupby('标的')['日期'].nunique().mean()
print(f'\nAverage hold days per stock: {avg_hold_days:.1f}')

# Compare overlap
codes_none = set(stocks['标的'].str.extract(r'\((\d+)\.')[0])
codes_fake = set(stocks2['标的'].str.extract(r'\((\d+)\.')[0])
overlap = codes_none & codes_fake
print(f'\nOverlap: {len(overlap)} stocks appear in both portfolios')
print(f'None-only: {len(codes_none - codes_fake)}, Fake-only: {len(codes_fake - codes_none)}')

# Top overlapping stocks
if overlap:
    overlap_df = all_profit[all_profit.index.str.contains('|'.join(list(overlap)[:10]))]
    print('\nPerformance of overlapping stocks in fake-win:')
    for idx, row in overlap_df.head(10).iterrows():
        print(f'  {idx}: +{row.total_pl:,.0f}')
