# -*- coding: utf-8 -*-
"""
7因子相关系数矩阵分析
=====================================
检查 v5b(4因子) 和 腿B(3因子) 之间是否存在高度相关
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = 'e:/quant/data/raw'
SAMPLE_DATE = '2025-12-31'  # 取最近的一个截面看看

print("=" * 60)
print("7因子相关系数矩阵")
print("=" * 60)

# 加载数据
daily = pd.read_csv(os.path.join(RAW_DIR, 'daily_prices.csv'),
                    dtype={'trade_date': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')

basic = pd.read_csv(os.path.join(RAW_DIR, 'daily_basic.csv'),
                    dtype={'trade_date': str})
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

idx = pd.read_csv(os.path.join(RAW_DIR, 'index_000300_daily.csv'),
                  dtype={'trade_date': str})
idx['trade_date'] = pd.to_datetime(idx['trade_date'], format='mixed')
idx = idx.sort_values('trade_date')

# 获取目标日期前后数据
lookback = 130  # 足够覆盖 120 天窗口
target_dt = pd.Timestamp(SAMPLE_DATE)
daily_sub = daily[daily['trade_date'] <= target_dt].tail(130 * 6000)  # 粗略筛选

# 只取有效股票（排除688/689，有足够数据的）
close_wide = daily.pivot_table('close', 'trade_date', 'ts_code', aggfunc='last')
vol_wide = daily.pivot_table('vol', 'trade_date', 'ts_code', aggfunc='last')
mcap_wide = basic.pivot_table('total_mv', 'trade_date', 'ts_code', aggfunc='last')
pb_wide = basic.pivot_table('pb', 'trade_date', 'ts_code', aggfunc='last')
ps_wide = basic.pivot_table('ps', 'trade_date', 'ts_code', aggfunc='last')

# 对齐代码
common = (set(close_wide.columns) & set(mcap_wide.columns) &
          set(pb_wide.columns) & set(ps_wide.columns))
common = [c for c in common if not c.startswith('688') and not c.startswith('689')]
print(f"共同股票: {len(common)}")

# 取目标日期所在行
date_idx = close_wide.index.get_indexer([target_dt], method='pad')[0]
if date_idx < lookback + 2:
    date_idx = close_wide.index.get_indexer([target_dt], method='bfill')[0]

print(f"目标日期: {close_wide.index[date_idx]}, 回看 {lookback} 天")

close_arr = close_wide[common].values[date_idx-lookback:date_idx+1].astype(np.float64)
vol_arr = vol_wide[common].values[date_idx-lookback:date_idx+1].astype(np.float64)

# 指数数据
idx_date_idx = np.searchsorted(idx['trade_date'].values, target_dt)
if idx_date_idx >= len(idx):
    idx_date_idx = len(idx) - 1
idx_close = idx['close'].values[idx_date_idx-lookback:idx_date_idx+1].astype(np.float64)

# 基本面 (取最近的)
fund_dates = mcap_wide.index
fund_idx = fund_dates.get_indexer([target_dt], method='pad')[0]
mcap_row = mcap_wide[common].iloc[fund_idx].values.astype(np.float64)
pb_row = pb_wide[common].iloc[fund_idx].values.astype(np.float64)
ps_row = ps_wide[common].iloc[fund_idx].values.astype(np.float64)

# 过滤有效股票
valid_mask = (~np.isnan(close_arr[-1]) &
              ~np.isnan(vol_arr[-1]) &
              (mcap_row > 0) &
              (pb_row > 0) &
              (ps_row > 0) &
              (close_arr.shape[0] - np.argmax(~np.isnan(close_arr[::-1]), axis=0) >= lookback - 2))

c = close_arr[:, valid_mask]
v = vol_arr[:, valid_mask]
mcap = mcap_row[valid_mask]
pb = pb_row[valid_mask]
ps = ps_row[valid_mask]
n_stocks = c.shape[1]
print(f"有效股票: {n_stocks}")

# ==================== 计算因子 ====================

def zscore(x):
    m = np.nanmedian(x)
    s = np.nanstd(x)
    return np.where(s > 0, (x - m) / s, 0)

# v5b 因子
f_size = zscore(-np.log(mcap))
f_ps   = zscore(-np.log(ps))
f_pb   = zscore(-np.log(pb))
to20 = v[-21:-1]
to_mean = np.nanmean(to20, axis=0)
to_std = np.nanstd(to20, axis=0)
f_str = zscore(-to_std / np.maximum(to_mean, 1e-8))

# legB 因子
RESID_MOM_WIN = 60
ILLIQ_WIN = 20
MOM_1M_WIN = 20

# resid_mom
ret64 = c[-RESID_MOM_WIN-1:] / c[-RESID_MOM_WIN-2:-1] - 1
idx_ret = idx_close[-RESID_MOM_WIN-1:] / idx_close[-RESID_MOM_WIN-2:-1] - 1
idx_var = np.var(idx_ret)
idx_mean = np.mean(idx_ret)
stock_mean = np.nanmean(ret64, axis=0)
if idx_var > 1e-10:
    idx_ctr = idx_ret - idx_mean
    stock_ctr = ret64 - stock_mean[np.newaxis, :]
    cov_sm = np.nansum(idx_ctr[:, np.newaxis] * stock_ctr, axis=0) / RESID_MOM_WIN
    beta = cov_sm / idx_var
else:
    beta = np.ones(n_stocks)
alpha = stock_mean - beta * idx_mean
f_resid_mom = zscore(-alpha)

# illiq
ret21 = c[-ILLIQ_WIN-1:] / c[-ILLIQ_WIN-2:-1] - 1
vol21 = v[-ILLIQ_WIN-1:]
abs_ret = np.abs(ret21)
illiq_daily = np.where(vol21 > 0, abs_ret / vol21, 0)
illiq = np.nanmean(illiq_daily, axis=0)
f_illiq = zscore(-illiq)

# mom_1m
mom_val = c[-2] / c[-MOM_1M_WIN-2] - 1
f_mom = zscore(-mom_val)

# ==================== 相关系数矩阵 ====================

factor_names = ['size', 'ps', 'pb', 'str', 'resid_mom', 'illiq', 'mom_1m']
factor_data = np.column_stack([f_size, f_ps, f_pb, f_str,
                                f_resid_mom, f_illiq, f_mom])

# 删掉含有NaN的行
valid_rows = ~np.any(np.isnan(factor_data), axis=1)
factor_data = factor_data[valid_rows]

corr = np.corrcoef(factor_data.T)
corr_df = pd.DataFrame(corr, index=factor_names, columns=factor_names)

print(f"\n7因子相关系数矩阵 ({SAMPLE_DATE}截面, {factor_data.shape[0]}只股票):")
print("=" * 72)
print(corr_df.round(3).to_string())

# 高亮 |r| > 0.3 的配对
print("\n\n|r| > 0.3 的因子对:")
high_corr = []
for i in range(len(factor_names)):
    for j in range(i+1, len(factor_names)):
        if abs(corr[i, j]) > 0.3:
            high_corr.append((factor_names[i], factor_names[j], round(corr[i, j], 3)))

if high_corr:
    for n1, n2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
        print(f"  {n1:12s} vs {n2:12s}  r = {r:+.3f}")
else:
    print("  无")

# 跨策略相关性 (v5b vs legB)
print("\n\n跨策略因子对 (v5b因子 vs legB因子):")
v5b_factors = ['size', 'ps', 'pb', 'str']
legb_factors = ['resid_mom', 'illiq', 'mom_1m']
for vf in v5b_factors:
    vi = factor_names.index(vf)
    for lf in legb_factors:
        li = factor_names.index(lf)
        r = corr[vi, li]
        flag = " ⚠️" if abs(r) > 0.3 else ""
        print(f"  {vf:12s} vs {lf:12s}  r = {r:+.3f}{flag}")

# 平均跨策略相关性
cross_corrs = []
for vf in v5b_factors:
    for lf in legb_factors:
        cross_corrs.append(abs(corr[factor_names.index(vf), factor_names.index(lf)]))
print(f"\n跨策略平均 |r| = {np.mean(cross_corrs):.3f}")

# 结论
print("\n" + "=" * 60)
if np.mean(cross_corrs) < 0.2:
    print("结论: 两套因子高度独立, 可以直接统一建模 ✓")
elif np.mean(cross_corrs) < 0.35:
    print("结论: 两套因子有轻度关联但可接受, 可以直接统一建模")
else:
    print("结论: 两套因子存在显著关联, 建议:")
    print("  1) 从中筛选最独立的子集")
    print("  2) 或使用PCA降维取代线性加权")
    print("  3) 或做正交化处理(对legB因子做v5b因子正交)")
print("=" * 60)
