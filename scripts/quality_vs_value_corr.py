# -*- coding: utf-8 -*-
"""ROE vs v5b 因子相关性"""
import pandas as pd, numpy as np, os
RAW_DIR = 'e:/quant/data/raw'

daily = pd.read_csv(os.path.join(RAW_DIR, 'daily_prices.csv'), dtype={'trade_date': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic = pd.read_csv(os.path.join(RAW_DIR, 'daily_basic.csv'), dtype={'trade_date': str})
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
fina = pd.read_csv(os.path.join(RAW_DIR, 'fina_indicator.csv'), dtype={'end_date': str})
fina['end_date'] = pd.to_datetime(fina['end_date'], format='mixed')

close_wide = daily.pivot_table('close', 'trade_date', 'ts_code', aggfunc='last')
vol_wide   = daily.pivot_table('vol', 'trade_date', 'ts_code', aggfunc='last')
mcap_wide  = basic.pivot_table('total_mv', 'trade_date', 'ts_code', aggfunc='last')
pb_wide    = basic.pivot_table('pb', 'trade_date', 'ts_code', aggfunc='last')
ps_wide    = basic.pivot_table('ps', 'trade_date', 'ts_code', aggfunc='last')
roe_wide   = fina.pivot_table('roe', 'end_date', 'ts_code', aggfunc='last')

# 交集
common = sorted(set(close_wide.columns) & set(mcap_wide.columns) & set(pb_wide.columns) & set(ps_wide.columns) & set(roe_wide.columns))
common = [c for c in common if not c.startswith('688') and not c.startswith('689')]
print(f"共同股票(价格+基本面+ROE): {len(common)}")

target = pd.Timestamp('2025-12-31')
di = close_wide.index.get_indexer([target], method='pad')[0]
close_arr = close_wide[common].values[di-24:di+1].astype(np.float64)
vol_arr   = vol_wide[common].values[di-24:di+1].astype(np.float64)
fi = mcap_wide.index.get_indexer([target], method='pad')[0]
ri = roe_wide.index.get_indexer([target], method='pad')[0]

mcap = mcap_wide[common].iloc[fi].values.astype(np.float64)
pb   = pb_wide[common].iloc[fi].values.astype(np.float64)
ps   = ps_wide[common].iloc[fi].values.astype(np.float64)
roe  = roe_wide[common].iloc[ri].values.astype(np.float64)

v = (~np.isnan(close_arr[-1]) & (mcap>0) & (pb>0) & (ps>0) & ~np.isnan(roe))
print(f"有效: {v.sum()}")

def zs(x):
    m=np.nanmedian(x); s=np.nanstd(x)
    return np.where(s>0, (x-m)/s, 0)

to20 = vol_arr[-21:-1,:][:,v]
f_size = zs(-np.log(mcap[v]))
f_ps   = zs(-np.log(ps[v]))
f_pb   = zs(-np.log(pb[v]))
f_str  = zs(-np.nanstd(to20,axis=0)/np.maximum(np.nanmean(to20,axis=0),1e-8))
f_roe  = zs(roe[v])

data = np.column_stack([f_size, f_ps, f_pb, f_str, f_roe])
data = data[~np.any(np.isnan(data), axis=1)]
corr = np.corrcoef(data.T)
names = ['size','ps','pb','str','roe']

print("\n相关系数矩阵:")
print("-" * 50)
for i, ni in enumerate(names):
    row = [f"{ni:>6s}"] + [f"{corr[i,j]:+.3f}" for j in range(len(names))]
    print(" ".join(row))

print("\nROE vs v5b:")
for i, n in enumerate(names[:-1]):
    print(f"  ROE vs {n:4s}  r = {corr[i,4]:+.3f}", end="")
    if abs(corr[i,4]) > 0.3: print("  <- 注意!")
    else: print()

avg = np.mean([abs(corr[i,4]) for i in range(4)])
print(f"\nROE vs v5b 平均 |r| = {avg:.3f}")
if avg < 0.15: print("结论: ROE 与价值因子高度独立, 可安全加入 ✓✓")
elif avg < 0.25: print("结论: ROE 与价值因子弱相关, 可加入 ✓")
else: print("结论: 中度相关, 需谨慎")
