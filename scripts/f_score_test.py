#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
f_score 强制入选实验
=====================
用实际价格回测, 对比:
  A) 当前Q↑ = earnings_growth_yoy + turnover_cv_20d + dollar_vol_20d + gross_margin
  B) f_score + 当前Q↑ 最优3个
  C) f_score + 其他高ICIR 3个

方法: 每周等权Top30, 周频换仓, 2021-2026
"""

import pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations
import sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(__file__).parent.parent
FACTOR_DIR = BASE / 'research' / 'factor_alchemy' / 'output'

# 当前 Q↑
CURRENT_Q = ['earnings_growth_yoy', 'turnover_cv_20d', 'dollar_vol_20d', 'gross_margin']

# ============================================================
# 加载数据
# ============================================================
print('加载数据...')

# 周频价格 (复用 ap_batch 的缓存)
weekly = pd.read_parquet(BASE / 'output/ap_batch/cache/weekly_prices.parquet')
weekly['trade_date'] = pd.to_datetime(weekly['trade_date'])
weekly = weekly.sort_values(['stock_code', 'trade_date'])
# fwd_ret 已算好
print(f'  价格: {len(weekly):,} 条, {weekly["trade_date"].nunique()} 周')

# 加载因子
factor_dfs = {}
all_factors = ['f_score', 'turnover_cv_20d', 'dollar_vol_20d', 'earnings_growth_yoy', 'gross_margin']
for fac in all_factors:
    csv_path = FACTOR_DIR / f'factor_{fac}.csv'
    if not csv_path.exists():
        csv_path = BASE / 'research/factor_alchemy/output/icir_biased' / f'factor_{fac}.csv'
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    if str(df.columns[0]).startswith('Unnamed'):
        df.rename(columns={df.columns[0]: 'trade_date'}, inplace=True)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    df = df.melt(id_vars='trade_date', var_name='stock_code', value_name=fac)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.dropna(subset=[fac])
    
    # 季频因子: forward fill到周频 (财务数据发布后持续有效)
    df = df.sort_values(['stock_code', 'trade_date'])
    df[fac] = df.groupby('stock_code')[fac].ffill()
    df = df.dropna(subset=[fac])
    
    factor_dfs[fac] = df
    print(f'  {fac}: {len(df):,} 条')

# 合并所有因子 - 使用外层join在日期维度
merged = list(factor_dfs.values())[0]
for fac, df in list(factor_dfs.items())[1:]:
    merged = merged.merge(df, on=['trade_date', 'stock_code'], how='outer')

# ffill across dates for each stock (季频因子填补)
merged = merged.sort_values(['stock_code', 'trade_date'])
for col in all_factors:
    if col in merged.columns:
        merged[col] = merged.groupby('stock_code')[col].ffill()

# 现在合并价格 (inner join)
merged = merged.merge(
    weekly[['stock_code', 'trade_date', 'fwd_ret']],
    on=['trade_date', 'stock_code'], how='inner'
)
dates = sorted(merged['trade_date'].unique())
print(f'  合并: {len(merged):,} 条, {len(dates)} 周\n')

# ============================================================
# 回测函数
# ============================================================
def backtest_top30(factors, merged, dates, n_top=30, name=''):
    """等权Top30周频回测"""
    available = [f for f in factors if f in merged.columns]
    if len(available) < 2:
        return None
    
    # 计算等权rank
    merged_c = merged.copy()
    merged_c['score'] = 0
    for f in available:
        direction = 1  # 默认正向
        # 负向因子
        if f in ['gross_margin']:
            direction = -1
        merged_c['score'] += merged_c.groupby('trade_date')[f].rank(pct=True) * direction
    
    merged_c['score'] = merged_c['score'] / len(available)
    
    returns = []
    for i, date in enumerate(dates[:-1]):
        day = merged_c[merged_c['trade_date'] == date].dropna(subset=['fwd_ret'])
        if len(day) < n_top * 2:
            continue
        top = day.nlargest(n_top, 'score')
        ret = top['fwd_ret'].mean()
        returns.append(ret)
    
    if len(returns) < 20:
        return None
    
    returns = np.array(returns)
    total = np.prod(1 + returns) - 1
    ann_ret = (1 + total) ** (52 / len(returns)) - 1
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
    max_dd = 0
    cum = np.cumprod(1 + returns)
    peak = cum[0]
    for v in cum:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    return {
        'name': name, 'factors': available,
        'total_ret': total, 'ann_ret': ann_ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'n_periods': len(returns)
    }

# ============================================================
# 组合测试
# ============================================================

# A) 当前Q↑
print('=' * 70)
print('A) 当前 Q↑')
print('=' * 70)
r_a = backtest_top30(CURRENT_Q, merged, dates, name='Q↑当前')
if r_a:
    print(f'  因子: {r_a["factors"]}')
    print(f'  总收益: {r_a["total_ret"]:+.2%}  年化: {r_a["ann_ret"]:+.2%}')
    print(f'  Sharpe: {r_a["sharpe"]:+.2f}  MaxDD: {r_a["max_dd"]:+.2%}')

# B) f_score 替换 gross_margin
print('\n' + '=' * 70)
print("B) f_score 替换 gross_margin (earnings_growth_yoy + tcv + dollar_vol)")
print('=' * 70)
r_b = backtest_top30(['f_score', 'earnings_growth_yoy', 'turnover_cv_20d', 'dollar_vol_20d'], 
                      merged, dates, name='f_score替换gross_margin')
if r_b:
    print(f'  因子: {r_b["factors"]}')
    print(f'  总收益: {r_b["total_ret"]:+.2%}  年化: {r_b["ann_ret"]:+.2%}')
    print(f'  Sharpe: {r_b["sharpe"]:+.2f}  MaxDD: {r_b["max_dd"]:+.2%}')
    if r_a:
        print(f'  vs Q↑: Sharpe {r_b["sharpe"] - r_a["sharpe"]:+.2f}, '
              f'总收益 {r_b["total_ret"] - r_a["total_ret"]:+.2%}')

# C) f_score + earnings_growth_yoy + turnover_cv_20d (3因子简化)
print('\n' + '=' * 70)
print("C) f_score + earnings_growth_yoy + turnover_cv_20d (三因子)")
print('=' * 70)
r_c = backtest_top30(['f_score', 'earnings_growth_yoy', 'turnover_cv_20d'],
                      merged, dates, name='f_score三因子')
if r_c:
    print(f'  因子: {r_c["factors"]}')
    print(f'  总收益: {r_c["total_ret"]:+.2%}  年化: {r_c["ann_ret"]:+.2%}')
    print(f'  Sharpe: {r_c["sharpe"]:+.2f}  MaxDD: {r_c["max_dd"]:+.2%}')

# D) 纯质量: f_score + earnings_growth_yoy (无流动性因子)
print('\n' + '=' * 70)
print("D) 纯质量: f_score + earnings_growth_yoy (无流动性因子)")
print('=' * 70)
r_d = backtest_top30(['f_score', 'earnings_growth_yoy'],
                      merged, dates, name='纯质量')
if r_d:
    print(f'  因子: {r_d["factors"]}')
    print(f'  总收益: {r_d["total_ret"]:+.2%}  年化: {r_d["ann_ret"]:+.2%}')
    print(f'  Sharpe: {r_d["sharpe"]:+.2f}  MaxDD: {r_d["max_dd"]:+.2%}')

# 汇总
print('\n' + '=' * 70)
print('汇总对比')
print('=' * 70)
print(f'{"策略":<35} {"总收益":>8} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8}')
print('-' * 70)
for r in [r_a, r_b, r_c, r_d]:
    if r:
        print(f'{r["name"]:<35} {r["total_ret"]:>+7.2%} {r["ann_ret"]:>+7.2%} '
              f'{r["sharpe"]:>+6.2f} {r["max_dd"]:>+7.2%}')
