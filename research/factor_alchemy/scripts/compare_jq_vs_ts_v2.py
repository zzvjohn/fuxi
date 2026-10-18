#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JQData vs Tushare 因子对比 v2 (用免费版1年窗口)
"""
import sys, os, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent; ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR)); sys.path.insert(0, str(ROOT_DIR))

from credentials import get_jqdata_auth
get_jqdata_auth()
from jqdatasdk import get_price, get_all_securities, get_trade_days

print("=" * 60)
print("  JQData vs Tushare 因子对比 (1年窗口)")
print("=" * 60)

# ===== 股票池: 取前200只沪深300成分股 (减少下载量) =====
print("\n[1] 股票池...")
hs300 = pd.read_csv(ROOT_DIR / 'data' / 'raw' / 'index_000300_daily.csv')
# 直接用Tushare缓存中的常见股票
with open(SCRIPT_DIR / 'output' / 'factor_cache.pkl', 'rb') as f:
    fts = pickle.load(f)
from factors.composite import standardize_factors
fts_s = standardize_factors(fts)

# 取在Tushare中出现频率最高的200只股票, 转为JQ格式
stock_freq = pd.Series(0, index=fts_s['atr_14'].columns)
for fname in fts_s:
    valid = fts_s[fname].notna().sum()
    stock_freq = stock_freq.add(valid, fill_value=0)
top_stocks_ts = stock_freq.nlargest(200).index.tolist()

# 转换 Tushare格式 → JQData格式: 000001.SZ → 000001.XSHE, 600000.SH → 600000.XSHG
def ts2jq(code):
    if code.endswith('.SZ'):
        return code.replace('.SZ', '.XSHE')
    elif code.endswith('.SH'):
        return code.replace('.SH', '.XSHG')
    return code
top_stocks = [ts2jq(c) for c in top_stocks_ts]
code_map = {jq: ts for jq, ts in zip(top_stocks, top_stocks_ts)}  # JQ→TS 映射
print(f"  Top-200 stocks, converted to JQ format")

# ===== 下载 JQData 价格 =====
print("\n[2] 下载 JQData 价格...")
# JQData免费版: 约2025-04-11 到 2026-04-18
jd_start = '2025-04-11'
jd_end = '2026-04-18'

frames = []
for i in range(0, len(top_stocks), 50):
    chunk = top_stocks[i:i+50]
    try:
        df = get_price(chunk, start_date=jd_start, end_date=jd_end,
                       frequency='daily', fields=['close','open','high','low'],
                       fq='post', skip_paused=True, panel=False)
        if df is not None and len(df) > 0:
            frames.append(df)
    except Exception as e:
        if i == 0:
            print(f"  ERROR sample: {e}")
    if i % 50 == 0:
        print(f"  {i}/{len(top_stocks)}")

price = pd.concat(frames, ignore_index=True)
price['time'] = pd.to_datetime(price['time'])
print(f"  {len(price)} rows, {price['code'].nunique()} stocks, {price['time'].min().date()}~{price['time'].max().date()}")

# ===== 周频化 =====, 同时转换 JQ→TS 代码
print("\n[3] 周频化 + 代码映射...")
price_w = price.set_index('time').groupby('code').resample('W').last().reset_index(level=0)

close_w = price_w.pivot(columns='code', values='close')
high_w = price_w.pivot(columns='code', values='high')
low_w = price_w.pivot(columns='code', values='low')

# 映射回TS代码以便对比
def jq2ts(jq_code):
    return code_map.get(jq_code, jq_code)
close_w = close_w.rename(columns=jq2ts)
high_w = high_w.rename(columns=jq2ts)
low_w = low_w.rename(columns=jq2ts)
print(f"  {close_w.shape[0]} 周 x {close_w.shape[1]} 股")

# ===== 计算 JQ 版因子 =====
print("\n[4] 计算 JQ 版因子...")
# atr_14
c1 = close_w.shift(1)
tr = pd.concat([high_w - low_w, (high_w - c1).abs(), (low_w - c1).abs()]).groupby(level=0).max()
atr14_jq = tr.rolling(14).mean() / close_w

# ret_6m
ret6m_jq = -(close_w / close_w.shift(26) - 1)

# relative_spread_proxy: avg of (high-low)/close over 20 weeks
relspread_jq = ((high_w - low_w) / close_w).rolling(20).mean()

# ===== 对比 =====
print("\n" + "=" * 60)
print("[5] 逐周对比")
print("=" * 60)

jq_factors = {'atr_14': atr14_jq, 'ret_6m': ret6m_jq, 'relative_spread_proxy': relspread_jq}

for fname, jq_df in jq_factors.items():
    ts_df = fts_s.get(fname)
    if ts_df is None:
        print(f"\n  {fname}: TS缓存中无此因子, 跳过")
        continue
    
    common_weeks = sorted(set(jq_df.dropna(how='all').index) & set(ts_df.dropna(how='all').index))
    if len(common_weeks) < 4:
        print(f"\n  {fname}: 共同周={len(common_weeks)} < 4, 跳过")
        continue
    
    overlaps, cors, rank_cors = [], [], []
    from scipy.stats import spearmanr
    
    for w in common_weeks:
        jq_w = jq_df.loc[w].dropna()
        ts_w = ts_df.loc[w].dropna()
        common = sorted(set(jq_w.index) & set(ts_w.index))
        if len(common) < 30:
            continue
        
        jq_v, ts_v = jq_w[common], ts_w[common]
        cors.append(jq_v.corr(ts_v))
        rc, _ = spearmanr(jq_v, ts_v)
        rank_cors.append(rc)
        
        n = min(30, len(common))
        jq_top = set(jq_v.nlargest(n).index)
        ts_top = set(ts_v.nlargest(n).index)
        overlaps.append(len(jq_top & ts_top) / n)
    
    if not cors:
        print(f"\n  {fname}: 无足够数据")
        continue
    
    avg_corr = np.mean(cors)
    avg_rank = np.mean(rank_cors)
    avg_overlap = np.mean(overlaps)
    
    verdict = "🟢" if avg_corr > 0.8 else ("🟡" if avg_corr > 0.5 else "🔴")
    print(f"\n  {fname} ({len(cors)}周):")
    print(f"    Pearson:   {avg_corr:.4f} [{np.min(cors):.3f}, {np.max(cors):.3f}]")
    print(f"    Spearman:  {avg_rank:.4f} [{np.min(rank_cors):.3f}, {np.max(rank_cors):.3f}]")
    print(f"    Top-30重合: {avg_overlap:.1%} [{np.min(overlaps):.0%}, {np.max(overlaps):.0%}]")
    print(f"    {verdict}")

print("\n完成!")
