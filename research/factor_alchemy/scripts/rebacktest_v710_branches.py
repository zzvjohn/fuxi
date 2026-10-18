#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v7.10 ensemble 分支 — 校准后 PortfolioSimulator 重回测
======================================================
用校准过 (corr=0.79 vs JQ) 的 PortfolioSimulator 重跑两个 regime 分支,
替代 regime_lasso.py Phase 7 的简化 Top30 等权回测 (无换手限制/无市值过滤/无存活偏差修复).

口径对齐 JQ 文件 (JQGenerator 默认):
  - top_n = 30
  - max_turnover = 0.60
  - mcap_min = 5  (亿, 流通市值; 由 circ_mv 千元 / 1e5 转换)
  - 成本: commission 万1 + 印花千1(仅卖) + 滑点 0.3% (QMT 标准双边)
  - 周频 W-SUN 对齐 (与 factor_cache 一致)
"""
import sys, os, time, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # factor_alchemy/
sys.path.insert(0, str(SCRIPT_DIR))

from data.loader import load_all_data
from factors.composite import standardize_factors
from portfolio.simulator import PortfolioSimulator

print("=" * 70)
print("  v7.10 ensemble 分支 — 校准后 PortfolioSimulator 重回测")
print("=" * 70)

# =====================================================
# 1. 加载因子缓存 (W-SUN)
# =====================================================
print("\n[1] 加载因子缓存")
t0 = time.time()
with open(SCRIPT_DIR / 'output' / 'factor_cache.pkl', 'rb') as f:
    factor_dfs = pickle.load(f)
print(f"  {len(factor_dfs)} 个因子 | {time.time()-t0:.1f}s")

factor_dfs_std = standardize_factors(factor_dfs)
factor_weeks = sorted(factor_dfs_std[list(factor_dfs_std.keys())[0]].index)
print(f"  因子周: {factor_weeks[0].strftime('%Y-%m-%d')} ~ {factor_weeks[-1].strftime('%Y-%m-%d')} ({len(factor_weeks)}周)")

# =====================================================
# 2. 价格 (hfq, 周频 W-SUN) + 市值 (流通, 亿元, 周频 W-SUN)
# =====================================================
print("\n[2] 价格 + 市值周频化")
t0 = time.time()
data = load_all_data('2021-01-01', '2026-06-28')
price_daily = data['price_data']['close']          # hfq 日线
val = data['valuation_data']                         # 含 circ_mv (千元)
print(f"  价格: {price_daily.shape} | 估值: {val.shape} | {time.time()-t0:.1f}s")

price_weekly = price_daily.resample('W').last()
price_weekly.index = pd.to_datetime(price_weekly.index)

# circ_mv 千元 -> 亿元, 周频 W-SUN, ffill 补缺
mcap_daily = val.pivot_table(index='trade_date', columns='code', values='circ_mv')
mcap_daily.index = pd.to_datetime(mcap_daily.index)
mcap_weekly = mcap_daily.resample('W').last().ffill(limit=5) / 1e5   # 亿元
mcap_weekly.index = pd.to_datetime(mcap_weekly.index)
print(f"  市值(亿): {mcap_weekly.shape} | 样例中位数={mcap_weekly.stack().median():.1f}亿")

# 对齐列到因子列
common_cols = factor_dfs_std[list(factor_dfs_std.keys())[0]].columns
price_weekly = price_weekly.reindex(columns=common_cols)
mcap_weekly = mcap_weekly.reindex(columns=common_cols)

# =====================================================
# 3. Regime 标签对齐 (W-FRI -> W-SUN, 同 regime_lasso Phase 4)
# =====================================================
print("\n[3] Regime 标签对齐")
regime_df = pd.read_csv(SCRIPT_DIR / 'data' / 'regime_labels.csv', parse_dates=['date'])
regime_df = regime_df.sort_values('date').reset_index(drop=True)
regime_df['regime_merged'] = regime_df['regime'].replace('neutral', np.nan).ffill()
regime_df['regime_merged'] = regime_df['regime_merged'].fillna('small_bull')
regime_series = regime_df.set_index('date')['regime_merged']

fr_index = pd.DatetimeIndex(factor_weeks)
_aligned = regime_series.reindex(fr_index, method='nearest', tolerance=pd.Timedelta(days=10))
_aligned = _aligned.ffill().bfill()
regime_map = {d: _aligned.loc[d] for d in fr_index if pd.notna(_aligned.loc[d])}
print(f"  对齐 {len(regime_map)}/{len(factor_weeks)} 周")
from collections import Counter
for r, c in Counter(regime_map.values()).most_common():
    print(f"    {r}: {c}周 ({100*c/len(regime_map):.0f}%)")

# =====================================================
# 4. 加载 v710 权重
# =====================================================
print("\n[4] 加载 v710 权重")
with open(SCRIPT_DIR / 'output' / 'regime_lasso_result_v710.json', encoding='utf-8') as f:
    res = json.load(f)
weights = {r: res['regimes'][r]['weights'] for r in res['regimes'] if 'weights' in res['regimes'][r]}
for r, w in weights.items():
    print(f"  {r}: {len(w)} 因子 -> {', '.join(f'{k}:{v:.2f}' for k,v in sorted(w.items(), key=lambda x:-x[1]))}")

# =====================================================
# 5. 构建合成因子 (W-SUN x stocks)
# =====================================================
print("\n[5] 构建合成因子")
def build_composite(w_dict):
    comp = pd.DataFrame(0.0, index=factor_weeks, columns=common_cols)
    for fname, w in w_dict.items():
        if fname not in factor_dfs_std:
            print(f"    [WARN] {fname} 不在因子缓存, 跳过")
            continue
        f = factor_dfs_std[fname].reindex(index=factor_weeks, columns=common_cols).fillna(0)
        comp = comp.add(w * f)
    return comp

comp_large = build_composite(weights.get('large_dominant', {}))
comp_small = build_composite(weights.get('small_bull', {}))

# 合并 regime-aware: 每周用对应 regime 的合成值
regime_arr = pd.Series(regime_map)
comp_combined = comp_large.copy()
is_small = regime_arr == 'small_bull'
comp_combined[is_small.values] = comp_small[is_small.values]

# 分支独立: 仅保留该 regime 的周 (用于独立口径评估)
large_weeks = sorted([d for d, r in regime_map.items() if r == 'large_dominant'])
small_weeks = sorted([d for d, r in regime_map.items() if r == 'small_bull'])
comp_large_only = comp_large.loc[large_weeks]
comp_small_only = comp_small.loc[small_weeks]

# =====================================================
# 6. 校准后 PortfolioSimulator 回测
# =====================================================
print("\n" + "=" * 70)
print("[6] 校准后 PortfolioSimulator 回测")
print("=" * 70)

SIM_KW = dict(top_n=30, weighting='equal', max_turnover=0.60,
              commission=0.0001, stamp_tax=0.001, slippage=0.003,
              min_commission=5, mcap_df=mcap_weekly, mcap_min=5, debug=False)

def run_branch(name, comp, price, weeks_note):
    sim = PortfolioSimulator(factor_df=comp, price_df=price, **SIM_KW)
    out = sim.run()
    s = out['stats']
    print(f"\n--- {name} ({weeks_note}) ---")
    if 'error' in s:
        print(f"  ERROR: {s['error']}")
        return s
    print(f"  周数: {s['n_weeks']} | 年化收益: {s['cagr']:+.1%} | Sharpe: {s['sharpe']:.2f}")
    print(f"  MaxDD: {s['max_drawdown']:.1%} | Calmar: {s['calmar']:.2f} | 胜率: {s['win_rate']:.1%}")
    print(f"  年化换手: {s['avg_turnover']:.2%} (单边/周均)")
    return s

stats = {}
stats['large_dominant'] = run_branch('large_dominant 分支', comp_large_only, price_weekly,
                                      f"独立, {len(large_weeks)}周")
stats['small_bull'] = run_branch('small_bull 分支', comp_small_only, price_weekly,
                                 f"独立, {len(small_weeks)}周")
stats['combined'] = run_branch('regime-aware 合并', comp_combined, price_weekly,
                               f"全样本 {len(factor_weeks)}周")

# 汇总对比
print("\n" + "=" * 70)
print("  汇总对比 (校准后 PortfolioSimulator vs Phase7 简化回测)")
print("=" * 70)
phase7 = res.get('backtest', {})
print(f"{'分支':<20s} {'年化':>10s} {'Sharpe':>8s} {'MaxDD':>9s} {'Calmar':>7s}")
print(f"{'large_dominant(校)':<20s} {stats['large_dominant'].get('cagr',float('nan')):>+10.1%} "
      f"{stats['large_dominant'].get('sharpe',0):>8.2f} {stats['large_dominant'].get('max_drawdown',0):>9.1%} "
      f"{stats['large_dominant'].get('calmar',0):>7.2f}")
print(f"{'small_bull(校)':<20s} {stats['small_bull'].get('cagr',float('nan')):>+10.1%} "
      f"{stats['small_bull'].get('sharpe',0):>8.2f} {stats['small_bull'].get('max_drawdown',0):>9.1%} "
      f"{stats['small_bull'].get('calmar',0):>7.2f}")
print(f"{'合并(校)':<20s} {stats['combined'].get('cagr',float('nan')):>+10.1%} "
      f"{stats['combined'].get('sharpe',0):>8.2f} {stats['combined'].get('max_drawdown',0):>9.1%} "
      f"{stats['combined'].get('calmar',0):>7.2f}")
print(f"{'合并(Phase7)':<20s} {phase7.get('annual_return_net',0):>+10.1%} "
      f"{phase7.get('sharpe_net',0):>8.2f} {phase7.get('max_dd_net',0):>9.1%} {'-':>7s}")

# 保存
out = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'method': 'PortfolioSimulator_calibrated',
    'params': {'top_n': 30, 'max_turnover': 0.60, 'mcap_min': 5, 'cost': 'QMT_std'},
    'stats': {k: {kk: vv for kk, vv in v.items() if kk != 'error'} for k, v in stats.items() if 'error' not in v},
    'phase7_backtest': phase7,
}
with open(SCRIPT_DIR / 'output' / 'rebacktest_v710_simulator.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  保存: {SCRIPT_DIR / 'output' / 'rebacktest_v710_simulator.json'}")
print("\n完成!")
