# -*- coding: utf-8 -*-
"""恢复 composite_v1 选中的 5 个复合因子的带符号 ICIR (确定 flip 方向)"""
import sys, os, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_composer import _fast_standardize, _fast_icir_vectorized

BASE = os.path.dirname(os.path.abspath(__file__))

# 加载因子缓存 + 前向收益
with open(os.path.join(BASE, 'factor_cache.pkl'), 'rb') as f:
    factor_dfs = pickle.load(f)
with open(os.path.join(BASE, 'forward_returns.pkl'), 'rb') as f:
    forward_returns = pickle.load(f)
if isinstance(forward_returns, dict):
    forward_returns = forward_returns.get('forward_returns', list(forward_returns.values())[0])

pairs = {
    'comp_dollar_vol_20d_x_turnover_std': ('dollar_vol_20d', 'turnover_std'),
    'comp_overnight_5d_x_tvma_20': ('overnight_5d', 'tvma_20'),
    'comp_money_flow_20_x_value_momentum': ('money_flow_20', 'value_momentum'),
    'comp_ret_open_2d_proxy_x_skewness_20': ('ret_open_2d_proxy', 'skewness_20'),
    'comp_gap_up_x_relative_spread_proxy': ('gap_up', 'relative_spread_proxy'),
}

# value_momentum 等 6 个交叉复合因子不在缓存, 需现算
COMPOSITE_DEFS = {
    'value_momentum': ('bp', 'ret_3m'),
}
for cname, (fa, fb) in COMPOSITE_DEFS.items():
    if cname not in factor_dfs and fa in factor_dfs and fb in factor_dfs:
        dfa, dfb = factor_dfs[fa], factor_dfs[fb]
        cd = dfa.index.intersection(dfb.index)
        uc = dfa.columns.union(dfb.columns)
        ra = dfa.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
        rb = dfb.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
        factor_dfs[cname] = ra * rb
        print(f'现算 composite base: {cname}')

weeks = list(forward_returns.index)
out = {}
for comp_name, (fa, fb) in pairs.items():
    if fa not in factor_dfs or fb not in factor_dfs:
        print(f'[MISS] {comp_name}: {fa} or {fb} 不在缓存')
        continue
    dfa, dfb = factor_dfs[fa], factor_dfs[fb]
    cd = dfa.index.intersection(dfb.index)
    uc = dfa.columns.union(dfb.columns)
    ra = dfa.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
    rb = dfb.loc[cd].reindex(columns=uc).rank(pct=True, axis=1, na_option='keep')
    comp = ra * rb
    z = _fast_standardize(comp)
    res = _fast_icir_vectorized(z, forward_returns, weeks, min_samples=20)
    signed = res['icir']
    out[comp_name] = {
        'pair': [fa, fb],
        'signed_icir': round(float(signed), 4),
        'flip': bool(signed < 0),
        'direction': 'NEGATIVE (值越小越好, 用时取负)' if signed < 0 else 'POSITIVE (值越大越好)',
    }
    print(f"{comp_name:45s} signed_ICIR={signed:+.4f} flip={signed<0}")

with open(os.path.join(BASE, 'composite_v1_flip_signs.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('已保存 composite_v1_flip_signs.json')
