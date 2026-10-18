#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证: v710 分支的 GROSS (无成本/无换手限制) top-30 收益, 隔离成本拖累"""
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
weeks = sorted(factor_dfs_std[list(factor_dfs_std)[0]].index)
cols = factor_dfs_std[list(factor_dfs_std)[0]].columns

print("加载价格...")
data = load_all_data('2021-01-01','2026-06-28')
close = data['price_data']['close']
close_w = close.resample('W').last(); close_w.index = pd.to_datetime(close_w.index)
close_w = close_w.reindex(columns=cols)
fwd = close_w.shift(-1)/close_w - 1

# regime
regime_df = pd.read_csv(SCRIPT_DIR/'data'/'regime_labels.csv', parse_dates=['date'])
regime_df['regime_merged'] = regime_df['regime'].replace('neutral',np.nan).ffill().fillna('small_bull')
rs = regime_df.set_index('date')['regime_merged']
regime_map = rs.reindex(pd.DatetimeIndex(weeks), method='nearest', tolerance=pd.Timedelta(days=10)).ffill().bfill().to_dict()

with open(SCRIPT_DIR/'output'/'regime_lasso_result_v710.json',encoding='utf-8') as f:
    res = json.load(f)
weights = {r:res['regimes'][r]['weights'] for r in res['regimes'] if 'weights' in res['regimes'][r]}

QMT_WEEKLY = 0.0072/5  # Phase7 用的 flat 成本

def gross_branch(name, w):
    comp = pd.DataFrame(0.0, index=weeks, columns=cols)
    for fn,wt in w.items():
        comp = comp.add(wt*factor_dfs_std[fn].reindex(index=weeks,columns=cols).fillna(0))
    rw = [d for d,r in regime_map.items() if r==name]
    rets=[]
    for d in rw:
        c = comp.loc[d].dropna()
        if len(c)<30: continue
        top = c.nlargest(30).index
        fr = fwd.loc[d, top]
        fr = fr[fr.notna()]
        if len(fr)<15: continue
        rets.append(fr.mean())
    rets = pd.Series(rets)
    gross_ann = rets.mean()*52
    net7_ann = (rets.mean()-QMT_WEEKLY)*52
    print(f"\n{name}: {len(rets)}周")
    print(f"  GROSS 年化: {gross_ann:+.1%} | Phase7-flat成本 年化: {net7_ann:+.1%} | 周均gross: {rets.mean():+.3%}")
    return gross_ann, net7_ann

print("="*60)
g_l, n_l = gross_branch('large_dominant', weights['large_dominant'])
g_s, n_s = gross_branch('small_bull', weights['small_bull'])
print("\n"+"="*60)
print("对照 (校准后 PortfolioSimulator 净收益):")
print(f"  large_dominant: GROSS {g_l:+.1%} -> Phase7flat {n_l:+.1%} -> 校准净 +0.8%")
print(f"  small_bull:     GROSS {g_s:+.1%} -> Phase7flat {n_s:+.1%} -> 校准净 -31.1%")
print("\n若 GROSS 为正但校准净为负 => 成本/换手拖累 (真实口径);")
print("若 GROSS 已为负 => 因子本身无效 (需查方向).")
