#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精确复刻 regime_lasso.py Phase 7 的 composite/index 逻辑, 打印分 regime GROSS/NET,
用于验证 small_bull 到底是正还是负 (与 Phase7 原本打印的 +15.9% 对照)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from data.loader import load_all_data
from factors.composite import standardize_factors

print("加载因子缓存...")
with open(SCRIPT_DIR/'output'/'factor_cache.pkl','rb') as f:
    factor_dfs = pickle.load(f)
factor_dfs_std = standardize_factors(factor_dfs)

print("加载价格...")
data = load_all_data('2021-01-01','2026-06-25')
close = data['price_data']['close']
close_weekly = close.resample('W').last(); close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1)/close_weekly - 1
print(f"  forward_returns: {forward_returns.shape}")

# regime 对齐 (完全照抄 Phase 4)
regime_df = pd.read_csv(SCRIPT_DIR/'data'/'regime_labels.csv', parse_dates=['date'])
regime_df = regime_df.sort_values('date').reset_index(drop=True)
regime_df['regime_merged'] = regime_df['regime'].replace('neutral',np.nan).ffill().fillna('small_bull')
regime_series = regime_df.set_index('date')['regime_merged']
fr_index = pd.DatetimeIndex(forward_returns.index)
_aligned = regime_series.reindex(fr_index, method='nearest', tolerance=pd.Timedelta(days=10))
_aligned = _aligned.ffill().bfill()
regime_map = {d:_aligned.loc[d] for d in fr_index if pd.notna(_aligned.loc[d])}

with open(SCRIPT_DIR/'output'/'regime_lasso_result_v710.json',encoding='utf-8') as f:
    res = json.load(f)
all_strat = {r:res['regimes'][r] for r in res['regimes'] if 'weights' in res['regimes'][r]}

TOP_N = 30
QMT = 0.0072/5

# 复刻 Phase 7 全样本回测 (包括分 regime)
all_weeks = sorted(regime_map.keys())
rows=[]
for week in all_weeks:
    if week not in forward_returns.index: continue
    regime = regime_map[week]
    strat = all_strat.get(regime,{})
    if 'weights' not in strat: continue
    weights = strat['weights']
    composite = pd.Series(0.0, index=forward_returns.columns)
    for fname,w in weights.items():
        fdf = factor_dfs_std.get(fname)
        if fdf is not None and week in fdf.index:
            composite += w*fdf.loc[week].reindex(composite.index).fillna(0)
    composite = composite.dropna()
    if len(composite)<TOP_N: continue
    top = composite.nlargest(TOP_N).index
    fr = forward_returns.loc[week]
    valid = [s for s in top if s in fr.index and pd.notna(fr[s])]
    if len(valid)<TOP_N//2: continue
    port_ret = fr[valid].mean()
    rows.append({'week':week,'regime':regime,'gross':port_ret,'net':port_ret-QMT})

bt = pd.DataFrame(rows).set_index('week')
print(f"\n全样本周数: {len(bt)} | 合并 NET 年化: {bt['net'].mean()*52:+.1%} (Phase7 JSON: -6.5%)")
print("\n分 regime (精确复刻 Phase7):")
for r,g in bt.groupby('regime'):
    ann_g = g['gross'].mean()*52
    ann_n = g['net'].mean()*52
    sd_n = g['net'].std()*np.sqrt(52)
    sh = ann_n/sd_n if sd_n>0 else 0
    print(f"  {r:18s} {len(g):>3}周 | GROSS年化 {ann_g:+.1%} | NET年化 {ann_n:+.1%} | Sharpe {sh:.2f}")
