#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JQData vs Tushare 因子值对比
============================
对 v7.11 使用的 6 个关键因子，分别用 JQData 和 Tushare 数据源计算，
对比因子值的相关性、Top-K 重合率。

v7.11 因子:
  large_dominant: earnings_stability, atr_14, earnings_season_vol_div
  small_bull:     relative_spread_proxy, sp, ret_6m

输出:
  - 每因子的 JQ vs TS 相关性
  - Top-30 选股重合率（每周）
  - 交叉因子值散点统计
"""
import sys, os, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # factor_alchemy/
sys.path.insert(0, str(SCRIPT_DIR))

from credentials import get_jqdata_auth
get_jqdata_auth()
from jqdatasdk import get_price, get_fundamentals, get_all_securities, get_trade_days
from jqdatasdk import query, valuation, indicator

print("=" * 70)
print("  JQData vs Tushare 因子值对比 (v7.11 6因子)")
print("=" * 70)

# =====================================================
# 1. 股票池: 全A (排除ST/新上市/退市)
# =====================================================
print("\n[1] 获取股票池")
securities = get_all_securities(['stock'], date='2021-01-04')
stocks_all = securities.index.tolist()
# 排除 688(科创板), 8/4开头
stocks_filtered = [s for s in stocks_all if not s.startswith(('688', '8', '4'))]
print(f"  全A: {len(stocks_all)} → 过滤后: {len(stocks_filtered)}")

# =====================================================
# 2. 下载 JQData 数据
# =====================================================
print("\n[2] 下载 JQData 数据...")

# 日期范围
start_date = '2021-01-01'
end_date = '2026-06-28'

# 2a. 日线价格 (后复权)
print("  [2a] 日线价格...")
t0 = time.time()

# 分批下载(防超时), 每批200只
chunk_size = 200
all_prices = {}
for i in range(0, len(stocks_filtered), chunk_size):
    chunk = stocks_filtered[i:i+chunk_size]
    try:
        df = get_price(chunk, start_date=start_date, end_date=end_date,
                       frequency='daily', fields=['close', 'open', 'high', 'low', 'volume', 'money'],
                       fq='post', skip_paused=True, panel=False)
        all_prices[i // chunk_size] = df
        if (i // chunk_size) % 10 == 0:
            print(f"    chunk {i//chunk_size}: {len(chunk)} stocks, {len(df)} rows")
    except Exception as e:
        print(f"    chunk {i//chunk_size} ERROR: {e}")
        time.sleep(5)

price_df = pd.concat(all_prices.values(), ignore_index=True)
print(f"  总计: {len(price_df)} 行 | {time.time()-t0:.0f}s")

# pivot化
close_jq = price_df.pivot(index='time', columns='code', values='close')
open_jq = price_df.pivot(index='time', columns='code', values='open')
high_jq = price_df.pivot(index='time', columns='code', values='high')
low_jq = price_df.pivot(index='time', columns='code', values='low')
vol_jq = price_df.pivot(index='time', columns='code', values='volume')
amount_jq = price_df.pivot(index='time', columns='code', values='money')

# 2b. 估值数据 (每日)
print("  [2b] 估值数据...")
t0 = time.time()
q_val = query(
    valuation.code, valuation.day,
    valuation.pe_ratio, valuation.pb_ratio, valuation.ps_ratio,
    valuation.market_cap, valuation.circulating_market_cap,
    valuation.turnover_ratio,
).filter(
    valuation.code.in_(stocks_filtered)
)
val_df = get_fundamentals(q_val, date=end_date)  # 只用最新日期快照不够
print(f"  需要按日期分片获取...")

# JQData get_fundamentals 一次只能一个日期, 需要逐季度获取
# 改为: 获取2021Q1-2026Q2的所有报告期
fund_dates = pd.date_range('2021-01-01', '2026-06-30', freq='QE')

# 简化: 只取关键日期的估值
sample_dates = pd.date_range('2021-01-03', '2026-06-28', freq='W')
val_frames = []
for d in sample_dates[::4]:  # 每4周取一次, 减少API调用
    try:
        df = get_fundamentals(q_val, date=d.strftime('%Y-%m-%d'))
        if df is not None and len(df) > 0:
            val_frames.append(df)
    except:
        continue
    if len(val_frames) % 10 == 0:
        print(f"    估值快照: {len(val_frames)}/{len(sample_dates[::4])}")

if val_frames:
    val_jq = pd.concat(val_frames, ignore_index=True)
else:
    val_jq = pd.DataFrame()

print(f"  估值: {val_jq.shape} | {time.time()-t0:.0f}s")

# 2c. 财务数据
print("  [2c] 财务数据...")
t0 = time.time()
q_ind = query(
    indicator.code,
    indicator.roe, indicator.roa,
    indicator.gross_profit_margin, indicator.net_profit_margin,
    indicator.inc_net_profit_year_on_year,
    indicator.inc_revenue_year_on_year,
    indicator.eps, indicator.bps, indicator.ocfps, indicator.cfps,
    indicator.assets_turn, indicator.current_ratio,
    indicator.debt_to_assets,
).filter(
    indicator.code.in_(stocks_filtered[:500])  # 先用500只测试
)

fin_frames = []
fyears = [2021, 2022, 2023, 2024, 2025, 2026]
fquarters = [1, 2, 3, 4]
for y in fyears:
    for q in fquarters:
        if y == 2026 and q > 2:
            break
        try:
            # statDate格式
            stat_date = f'{y}q{q}'
            df = get_fundamentals(q_ind, statDate=stat_date)
            if df is not None and len(df) > 0:
                fin_frames.append(df)
        except Exception as e:
            print(f"    {stat_date}: {e}")

if fin_frames:
    fin_jq = pd.concat(fin_frames, ignore_index=True)
else:
    fin_jq = pd.DataFrame()
print(f"  财务: {fin_jq.shape} | {time.time()-t0:.0f}s")

# =====================================================
# 3. 保存 JQData 下载数据
# =====================================================
print("\n[3] 保存下载数据...")
jqdata_dir = SCRIPT_DIR / 'data' / 'jqdata'
jqdata_dir.mkdir(exist_ok=True)

# 保存价格数据
price_jq_data = {
    'close': close_jq,
    'open': open_jq,
    'high': high_jq,
    'low': low_jq,
    'volume': vol_jq,
    'amount': amount_jq,
}
with open(jqdata_dir / 'price_jq.pkl', 'wb') as f:
    pickle.dump(price_jq_data, f)

val_jq.to_csv(jqdata_dir / 'valuation_jq.csv', index=False)
fin_jq.to_csv(jqdata_dir / 'financial_jq.csv', index=False)

print(f"  已保存: {jqdata_dir}")
print(f"    price_jq.pkl ({len(price_jq_data)} DataFrames)")
print(f"    valuation_jq.csv ({val_jq.shape})")
print(f"    financial_jq.csv ({fin_jq.shape})")

# =====================================================
# 4. 用 JQData 计算 6 个因子
# =====================================================
print("\n" + "=" * 70)
print("[4] 用 JQData 计算 v7.11 6因子")
print("=" * 70)

# 周频化
close_jq_w = close_jq.resample('W').last()

# ---- atr_14 ----
print("\n  [atr_14]")
high_w = high_jq.resample('W').last()
low_w = low_jq.resample('W').last()
tr = pd.concat([high_w - low_w,
                (high_w - close_jq_w.shift(1)).abs(),
                (low_w - close_jq_w.shift(1)).abs()], axis=1).max(axis=1, level=0)  # won't work with multi-level
# 简化: 单level计算
tr_df = pd.DataFrame(index=close_jq_w.index, columns=close_jq_w.columns)
for col in close_jq_w.columns[:100]:  # 先用100只测试
    try:
        h = high_w[col]
        l = low_w[col]
        c = close_jq_w[col]
        c1 = c.shift(1)
        tr1 = h - l
        tr2 = (h - c1).abs()
        tr3 = (l - c1).abs()
        tr_combined = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr_combined.rolling(14).mean()
        tr_df[col] = atr / c
    except:
        pass
atr14_jq = tr_df

# ---- ret_6m ----
print("  [ret_6m]")
ret6m_jq = close_jq_w / close_jq_w.shift(26) - 1  # 26 weeks ≈ 6 months
ret6m_jq = -ret6m_jq  # short-term reversal

# ---- relative_spread_proxy ----
print("  [relative_spread_proxy]")
spread_jq = (high_jq.resample('W').last() - low_jq.resample('W').last()) / close_jq_w
relspread_jq = spread_jq.rolling(20).mean()

# ---- sp (sales to price, 用 ps_ratio 倒数) ----
# 需要从估值数据取 ps_ratio
print("  [sp] — 需要估值数据, 跳过")
sp_jq = None

# ---- earnings_stability ----
print("  [earnings_stability] — 需要财务数据, 跳过")
earnstab_jq = None

# ---- earnings_season_vol_div ----
print("  [earnings_season_vol_div] — 需要量价+财务, 跳过")

# =====================================================
# 5. 对比 JQ vs Tushare (已计算的因子)
# =====================================================
print("\n" + "=" * 70)
print("[5] JQ vs Tushare 因子值对比")
print("=" * 70)

# 加载 Tushare 因子缓存
with open(SCRIPT_DIR / 'output' / 'factor_cache.pkl', 'rb') as f:
    factor_ts = pickle.load(f)

from factors.composite import standardize_factors
factor_ts_std = standardize_factors(factor_ts)
factor_weeks = sorted(factor_ts_std['atr_14'].index)

# 对比 atr_14
ts = factor_ts_std['atr_14']
jq = atr14_jq

# 对齐股票代码
common_stocks = sorted(set(ts.columns) & set(jq.columns))
print(f"\n  atr_14: common stocks={len(common_stocks)}")

# 逐周计算top-30重合率
overlap_rates = []
correlations = []
for w in factor_weeks[:50]:  # 前50周
    if w not in jq.index:
        continue
    ts_w = ts.loc[w].dropna()
    jq_w = jq.loc[w].dropna()
    common = sorted(set(ts_w.index) & set(jq_w.index))
    if len(common) < 30:
        continue
    
    ts_top30 = set(ts_w[common].nlargest(30).index)
    jq_top30 = set(jq_w[common].nlargest(30).index)
    overlap = len(ts_top30 & jq_top30) / 30
    overlap_rates.append(overlap)
    
    corr = ts_w[common].corr(jq_w[common])
    correlations.append(corr)

print(f"  Top-30 重合率: mean={np.mean(overlap_rates):.1%} (range={np.min(overlap_rates):.0%}-{np.max(overlap_rates):.0%})")
print(f"  横截面相关性: mean={np.mean(correlations):.3f} (range={np.min(correlations):.3f}-{np.max(correlations):.3f})")

# 对比 ret_6m
ts_ret = factor_ts_std['ret_6m']
common_stocks2 = sorted(set(ts_ret.columns) & set(ret6m_jq.columns))
print(f"\n  ret_6m: common stocks={len(common_stocks2)}")

overlap_rates2 = []
correlations2 = []
for w in factor_weeks[:50]:
    if w not in ret6m_jq.index:
        continue
    ts_w = ts_ret.loc[w].dropna()
    jq_w = ret6m_jq.loc[w].dropna()
    common = sorted(set(ts_w.index) & set(jq_w.index))
    if len(common) < 30:
        continue
    ts_top30 = set(ts_w[common].nlargest(30).index)
    jq_top30 = set(jq_w[common].nlargest(30).index)
    overlap_rates2.append(len(ts_top30 & jq_top30) / 30)
    correlations2.append(ts_w[common].corr(jq_w[common]))

print(f"  Top-30 重合率: mean={np.mean(overlap_rates2):.1%} (range={np.min(overlap_rates2):.0%}-{np.max(overlap_rates2):.0%})")
print(f"  横截面相关性: mean={np.mean(correlations2):.3f} (range={np.min(correlations2):.3f}-{np.max(correlations2):.3f})")

# =====================================================
# 6. 总结
# =====================================================
print("\n" + "=" * 70)
print("  总结")
print("=" * 70)
print(f"  atr_14 Top-30 overlap: {np.mean(overlap_rates):.1%}")
print(f"  ret_6m Top-30 overlap: {np.mean(overlap_rates2):.1%}")
print(f"  如果重合率 < 70%, 说明因子数据差异是导致 JQ 偏离的主因")
print(f"  如果重合率 > 80%, 说明问题在其他环节 (执行价/成本模型)")

print("\n完成!")
