#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机制验证: large_dominant 分支在 不同配置 下的净收益, 确认 mcap 过滤 / 成本模型 的作用."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from data.loader import load_all_data
from factors.composite import standardize_factors
from portfolio.simulator import PortfolioSimulator

with open(SCRIPT_DIR/'output'/'factor_cache.pkl','rb') as f:
    factor_dfs = pickle.load(f)
factor_dfs_std = standardize_factors(factor_dfs)
weeks = sorted(factor_dfs_std[list(factor_dfs_std)[0]].index)
cols = factor_dfs_std[list(factor_dfs_std)[0]].columns

data = load_all_data('2021-01-01','2026-06-28')
price_daily = data['price_data']['close']
price_weekly = price_daily.resample('W').last(); price_weekly.index = pd.to_datetime(price_weekly.index)
price_weekly = price_weekly.reindex(columns=cols)
val = data['valuation_data']
mcap_daily = val.pivot_table(index='trade_date', columns='code', values='circ_mv')
mcap_daily.index = pd.to_datetime(mcap_daily.index)
mcap_weekly = (mcap_daily.resample('W').last().ffill(limit=5)/1e5).reindex(columns=cols)

regime_df = pd.read_csv(SCRIPT_DIR/'data'/'regime_labels.csv', parse_dates=['date'])
regime_df['regime_merged'] = regime_df['regime'].replace('neutral',np.nan).ffill().fillna('small_bull')
rs = regime_df.set_index('date')['regime_merged']
regime_map = rs.reindex(pd.DatetimeIndex(weeks), method='nearest', tolerance=pd.Timedelta(days=10)).ffill().bfill().to_dict()
large_weeks = [d for d,r in regime_map.items() if r=='large_dominant']

with open(SCRIPT_DIR/'output'/'regime_lasso_result_v710.json',encoding='utf-8') as f:
    res = json.load(f)
w = res['regimes']['large_dominant']['weights']

comp = pd.DataFrame(0.0, index=weeks, columns=cols)
for fn,wt in w.items():
    comp = comp.add(wt*factor_dfs_std[fn].reindex(index=weeks,columns=cols).fillna(0))
comp_only = comp.loc[large_weeks]

def run(label, mcap_df, mcap_min, cost, cap):
    sim = PortfolioSimulator(factor_df=comp_only, price_df=price_weekly, top_n=30,
        max_turnover=cap, commission=cost[0], stamp_tax=cost[1], slippage=cost[2],
        min_commission=5, mcap_df=mcap_df, mcap_min=mcap_min)
    s = sim.run()['stats']
    if 'error' in s: print(f"  {label}: ERROR {s['error']}"); return
    print(f"  {label:42s} 年化 {s['cagr']:+.1%} | SR {s['sharpe']:.2f} | MDD {s['max_drawdown']:.1%} | 换手 {s['avg_turnover']:.2%}")

print("large_dominant 分支 — 配置对比 (验证 mcap 过滤 / 成本机制):")
print(f"  周数={len(large_weeks)} | 因子: {', '.join(w.keys())}")
run("① 校准全口径 (mcap≥5亿+成本+cap0.6)", mcap_weekly, 5, (0.0001,0.001,0.003), 0.60)
run("② 无mcap过滤 (成本+cap0.6)", None, None, (0.0001,0.001,0.003), 0.60)
run("③ 无mcap+无成本+无cap (纯选股gross)", None, None, (0,0,0), 1.0)
run("④ 无mcap+成本+无cap (gross-成本)", None, None, (0.0001,0.001,0.003), 1.0)
