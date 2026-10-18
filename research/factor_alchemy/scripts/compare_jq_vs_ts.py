#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JQData vs Tushare 因子值差异诊断
=================================
计算 atr_14 + ret_6m (纯量价) 两个因子, 对比 JQData vs Tushare.
"""

import sys, os, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SCRIPT_DIR.parent.parent  # e:/quant
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from credentials import get_jqdata_auth
get_jqdata_auth()
from jqdatasdk import get_price, get_all_securities

print("=" * 60)
print("  JQData vs Tushare: atr_14 & ret_6m 因子值对比")
print("=" * 60)

# ===== 加载 Tushare 因子缓存 =====
print("\n[1] 加载 Tushare 因子缓存...")
with open(SCRIPT_DIR / 'output' / 'factor_cache.pkl', 'rb') as f:
    fts = pickle.load(f)
from factors.composite import standardize_factors
fts_s = standardize_factors(fts)

# ===== 获取股票池 =====
print("\n[2] 获取 JQData 股票池...")
secs = get_all_securities(['stock'], date='2021-01-04')
all_codes = [s for s in secs.index if not s.startswith(('688','8','4'))]
print(f"  {len(all_codes)} stocks")

# ===== 下载 JQData 价格 (后复权) =====
print("\n[3] 下载价格数据...")
t0 = time.time()

# 分批: 每批100只, 合并
frames = []
for i in range(0, min(1000, len(all_codes)), 100):
    chunk = all_codes[i:i+100]
    try:
        df = get_price(chunk, start_date='2021-01-01', end_date='2026-06-28',
                       frequency='daily', fields=['close','open','high','low'],
                       fq='post', skip_paused=True, panel=False)
        if len(df) > 0:
            frames.append(df)
    except Exception as e:
        print(f"  chunk {i}: {e}")
        time.sleep(1)
    if i % 300 == 0:
        print(f"  {i}/{min(1000,len(all_codes))}... ({time.time()-t0:.0f}s)")

price = pd.concat(frames, ignore_index=True)
print(f"  总计: {len(price)} rows, {price['code'].nunique()} stocks | {time.time()-t0:.0f}s")

# ===== 周频化 =====
print("\n[4] 周频化...")
price['time'] = pd.to_datetime(price['time'])
price_w = price.set_index('time').groupby('code').resample('W').last().reset_index(level=0)

close_w = price_w.pivot(columns='code', values='close')
high_w = price_w.pivot(columns='code', values='high')
low_w = price_w.pivot(columns='code', values='low')
open_w = price_w.pivot(columns='code', values='open')

print(f"  周频: {close_w.shape}")

# ===== 计算 JQData 版因子 =====
print("\n[5] 计算 JQData 版因子...")

# --- atr_14 ---
# True Range: max(H-L, |H-C1|, |L-C1|)
c1 = close_w.shift(1)
tr1 = high_w - low_w
tr2 = (high_w - c1).abs()
tr3 = (low_w - c1).abs()
tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
atr14_jq = tr.rolling(14).mean() / close_w
print(f"  atr_14: {atr14_jq.shape}")

# --- ret_6m ---
ret6m_jq = -(close_w / close_w.shift(26) - 1)  # 反转
print(f"  ret_6m: {ret6m_jq.shape}")

# ===== 对比 =====
print("\n" + "=" * 60)
print("[6] 逐周对比")
print("=" * 60)

for fname, jq_df in [('atr_14', atr14_jq), ('ret_6m', ret6m_jq)]:
    ts_df = fts_s[fname]
    
    # 对齐
    jq_wks = set(jq_df.dropna(how='all').index)
    ts_wks = set(ts_df.dropna(how='all').index)
    common_weeks = sorted(jq_wks & ts_wks)
    
    if len(common_weeks) < 10:
        print(f"\n  {fname}: 共同周 < 10, 跳过")
        continue
    
    # 取每季第一周作为快照 (减少计算量)
    sample_weeks = common_weeks[::13]  # ~每季度
    print(f"\n  {fname}: {len(common_weeks)} 共同周, 抽样 {len(sample_weeks)} 周")

    overlaps = []
    cors = []
    rank_cors = []
    
    for w in sample_weeks:
        jq_w = jq_df.loc[w].dropna()
        ts_w = ts_df.loc[w].dropna()
        common = sorted(set(jq_w.index) & set(ts_w.index))
        if len(common) < 50:
            continue
        
        jq_v = jq_w[common]
        ts_v = ts_w[common]
        
        # Pearson corr
        corr = jq_v.corr(ts_v)
        cors.append(corr)
        
        # Spearman rank corr
        from scipy.stats import spearmanr
        rc, _ = spearmanr(jq_v, ts_v)
        rank_cors.append(rc)
        
        # Top-30 重合
        n = min(30, len(common))
        jq_top = set(jq_v.nlargest(n).index)
        ts_top = set(ts_v.nlargest(n).index)
        overlaps.append(len(jq_top & ts_top) / n)
    
    print(f"  Pearson corr:   mean={np.mean(cors):.4f}  (min={np.min(cors):.3f}, max={np.max(cors):.3f})")
    print(f"  Spearman corr:  mean={np.mean(rank_cors):.4f}  (min={np.min(rank_cors):.3f}, max={np.max(rank_cors):.3f})")
    print(f"  Top-30 overlap: mean={np.mean(overlaps):.1%}  (min={np.min(overlaps):.0%}, max={np.max(overlaps):.0%})")
    
    # 判定
    avg_overlap = np.mean(overlaps)
    avg_corr = np.mean(cors)
    if avg_corr > 0.8:
        verdict = "🟢 高度一致 — JQ/TS 因子值差异很小"
    elif avg_corr > 0.5:
        verdict = "🟡 中度差异 — 部分数据源差异导致选股不同"
    else:
        verdict = "🔴 显著差异 — 因子数据源是 JQ 偏离的主因"
    print(f"  → 判定: {verdict}")

print("\n完成!")
