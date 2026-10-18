"""
因子炼金术 v7 FAST — 精简流水线
跳过 Phase 3 重计算 (decile/double_sort), 直接用 ICIR 数据 + 预计算IC序列
快速进入 NSGA-II 三目标优化
"""
import sys, time, os
from pathlib import Path
from datetime import datetime

# Encoding fix
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except: pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

t_total = time.time()

from config import FACTOR_DEFS, OOS_SPLIT_DATE, MAX_FACTORS, ICIR_THRESHOLD

print("=" * 70)
print("  因子炼金术 v7 FAST - 三目标NSGA-II (精简)")
print(f"  MAX_FACTORS={MAX_FACTORS} | ICIR: clip | OOS={OOS_SPLIT_DATE}")
print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ===== Phase 1: Data =====
print("\n[Phase 1] 加载数据...")
from data.loader import load_all_data
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
print(f"  耗时: {time.time()-t1:.0f}s")

price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']

close = price_data.get('close')
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1

mcap_df = None
if valuation_data is not None and hasattr(valuation_data, 'columns'):
    if 'market_cap' in valuation_data.columns:
        mcap_raw = valuation_data[['code', 'trade_date', 'market_cap']].copy()
        mcap_raw['trade_date'] = pd.to_datetime(mcap_raw['trade_date'])
        mcap_df = mcap_raw.pivot(index='trade_date', columns='code', values='market_cap')
        mcap_df = mcap_df.resample('W').last()
        mcap_df = np.log(mcap_df.clip(lower=1))

# ===== Phase 2: Factor computation =====
print("\n[Phase 2] 计算因子...")
from factors import ALL_FACTORS
factor_dfs = {}
t2 = time.time()

for name in FACTOR_DEFS:
    if name not in ALL_FACTORS:
        continue
    factor_obj = ALL_FACTORS[name]()
    try:
        df = factor_obj.compute(price_data=price_data, financial_data=financial_data,
                                valuation_data=valuation_data)
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df_weekly = df.resample('W').last()
            factor_def = FACTOR_DEFS[name]
            is_financial = df_weekly.shape[0] <= 10 and factor_def['category'] in \
                ('profitability', 'value', 'growth', 'quality', 'size')
            if is_financial:
                df_weekly = df_weekly.reindex(close_weekly.index).ffill()
            factor_dfs[name] = df_weekly
    except Exception as e:
        pass

print(f"  成功: {len(factor_dfs)}/{len(FACTOR_DEFS)} ({time.time()-t2:.0f}s)")

# ===== Phase 3: IC/ICIR only =====
print("\n[Phase 3] IC/ICIR 快速评估...")
from evaluation.ic_analysis import compute_ic_summary, compute_ic_icir
t3 = time.time()

ic_summary = compute_ic_summary(factor_dfs, forward_returns)

print(f"\n  Top 20 ICIR:")
print(f"  {'因子':28s} {'ICIR':>8s} {'IC_mean':>8s} {'+IC%':>7s}")
print(f"  {'-'*54}")
for name in ic_summary.index[:20]:
    row = ic_summary.loc[name]
    print(f"  {name:28s} {row['ICIR']:+7.3f} {row['IC_mean']:+7.4f} {row['+IC%']:6.1%}")

# ===== Phase 3.4: Precompute IC series =====
print("\n  预计算 IC 序列...")
factor_ic_series = {}
for name in factor_dfs:
    if name not in ic_summary.index:
        continue
    try:
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_s = icr.get('ic_series', pd.Series(dtype=float))
        if len(ic_s) > 50:
            factor_ic_series[name] = ic_s
    except: pass
print(f"  完成: {len(factor_ic_series)}/{len(factor_dfs)}")

# ===== Phase 3.4b: Standardize factors =====
print("\n  预标准化因子...")
from factors.composite import standardize_factors
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  完成: {len(factor_dfs_std)}")

# ===== Phase 4: NSGA-II =====
print("\n" + "=" * 70)
print("[Phase 4] NSGA-II 三目标优化 (40pop x 5gen)")
print("=" * 70)

valid_names = [n for n in factor_dfs if n in ic_summary.index
               and not np.isnan(ic_summary.loc[n, 'ICIR'])]
n_factors = len(valid_names)
print(f"  有效因子: {n_factors}")

# Correlation matrix
from evaluation.correlation import factor_correlation_matrix
try:
    corr_matrix_full = factor_correlation_matrix(
        {n: factor_dfs[n] for n in valid_names if n in factor_dfs})
except:
    corr_matrix_full = pd.DataFrame(np.eye(n_factors), index=valid_names, columns=valid_names)

# OOS split
fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]
print(f"  训练集: {train_fr.shape[0]} 周")

# Train IC series
oos_dt = pd.Timestamp(OOS_SPLIT_DATE)
train_ic = {}
for name, ic_series in factor_ic_series.items():
    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
        ic_train = ic_series[ic_series.index < oos_dt].dropna()
        if len(ic_train) > 10:
            train_ic[name] = ic_train

# ICIR biased init
from ga.chromosome import icir_biased_chromosome
icir_list = []
for name in valid_names:
    if name in ic_summary.index:
        icir_list.append(abs(ic_summary.loc[name, 'ICIR']))
    else:
        icir_list.append(0.0)
icir_arr = np.array(icir_list)
print(f"\n  ICIR偏置初始化, ICIR范围: [{icir_arr.min():.3f}, {icir_arr.max():.3f}]")
print(f"  Top 5: {sorted(zip(valid_names, icir_arr), key=lambda x: -x[1])[:5]}")

# Run NSGA-II
from ga.nsga2 import nsga2_run
t4 = time.time()
pop_size, n_gen = 40, 5

pop = [icir_biased_chromosome(n_factors, icir_arr, bias_strength=1.0) for _ in range(pop_size)]

final_population, final_objectives, pareto_front = nsga2_run(
    pop, valid_names, factor_dfs,
    train_fr, mcap_df,
    n_generations=n_gen, pop_size=pop_size,
    crossover_prob=0.7, mutation_prob=0.2, mutation_sigma=0.1,
    corr_matrix=corr_matrix_full,
    ic_summary=ic_summary,
    factor_ic_series=train_ic,
    max_factors=MAX_FACTORS,
    verbose=True,
)
print(f"\n  NSGA-II 耗时: {time.time()-t4:.0f}s")

# ===== Decode Pareto Front =====
print(f"\n  解码帕累托前沿...")
from factors.composite import combine_factors_vectorized

pareto_solutions = []
for i, (obj, w) in enumerate(zip(pareto_front['objectives'], pareto_front['weights'])):
    try:
        sel = {n: factor_dfs_std[n] for n in w if n in factor_dfs_std}
        comp = combine_factors_vectorized(sel, w)
        if comp.empty or comp.shape[0] < 10:
            continue
        icir_train = abs(compute_ic_icir(comp, train_fr)['icir'])
        from ga.fitness import _estimate_weekly_turnover
        est_to = _estimate_weekly_turnover(comp, top_n=30, buffer_mult=1.8)
        pareto_solutions.append({
            'idx': i, 'quality_obj': float(obj[0]), 'stability_obj': float(obj[1]),
            'sharpe_obj': float(obj[2]), 'icir_train': icir_train,
            'weights': w, 'estimated_to': est_to, 'composite': comp,
        })
    except Exception as e:
        pass

print(f"\n  === 帕累托前沿 ({len(pareto_solutions)} 个解) [训练集] ===")
print(f"  {'#':>3s} {'质量(Q)':>8s} {'稳定(S)':>8s} {'Sharpe(C)':>10s} {'ICIR_train':>10s} {'Est_TO':>8s} {'N':>3s}")
for ps in sorted(pareto_solutions, key=lambda x: -x['quality_obj']):
    print(f"  {ps['idx']:3d} {ps['quality_obj']:8.3f} {ps['stability_obj']:8.3f} "
          f"{ps['sharpe_obj']:10.3f} {ps['icir_train']:10.3f} "
          f"{ps['estimated_to']:7.1%} {len(ps['weights']):3d}")

# ===== Phase 4.5: Select Strategies =====
print("\n" + "=" * 70)
print("[Phase 4.5] 3D帕累托选点")
print("=" * 70)

if pareto_solutions:
    high_quality = max(pareto_solutions, key=lambda x: x['quality_obj'])
    high_stable  = max(pareto_solutions, key=lambda x: x['stability_obj'])
    high_sharpe  = max(pareto_solutions, key=lambda x: x['sharpe_obj'])
    balanced = max(pareto_solutions, key=lambda x:
                   (max(x['quality_obj'], 0.001) ** 0.4) *
                   (max(x['stability_obj'], 0.001) ** 0.3) *
                   (max(x['sharpe_obj'], 0.001) ** 0.3))

    strategies = [
        ('high_sharpe',  high_sharpe),
        ('high_quality', high_quality),
        ('balanced',     balanced),
        ('high_stable',  high_stable),
    ]

    print(f"\n  前沿策略对比 (3D v7 FAST):")
    print(f"  {'策略':>12s} {'质量(Q)':>8s} {'稳定(S)':>8s} {'Sharpe(C)':>10s} {'Est_TO':>7s} {'N因子':>5s} {'因子列表'}")
    print(f"  {'-'*90}")
    for label, ps in strategies:
        factors_str = ', '.join(list(ps['weights'].keys())[:5])
        print(f"  {label:>12s} {ps['quality_obj']:8.3f} {ps['stability_obj']:8.3f} "
              f"{ps['sharpe_obj']:10.3f} {ps['estimated_to']:6.1%} {len(ps['weights']):5d} "
              f"[{factors_str}]")

# ===== OOS Verification =====
print("\n" + "=" * 70)
print("[Phase 5.5] OOS 验证")
print("=" * 70)

fr_index = pd.DatetimeIndex(forward_returns.index)
oos_mask = fr_index >= pd.Timestamp(OOS_SPLIT_DATE)
oos_fr = forward_returns.loc[oos_mask]
print(f"  OOS: {oos_fr.shape[0]} 周 ({OOS_SPLIT_DATE} ~ {forward_returns.index[-1].strftime('%Y-%m-%d')})")

for label, ps in strategies:
    try:
        w = ps['weights']
        sel = {n: factor_dfs_std[n] for n in w if n in factor_dfs_std}
        comp = combine_factors_vectorized(sel, w)
        oos_icir = abs(compute_ic_icir(comp, oos_fr)['icir'])
        print(f"  {label:>12s}: OOS ICIR={oos_icir:+.3f} | 训练 ICIR={ps['icir_train']:+.3f}")
        print(f"    因子: {dict(sorted(w.items(), key=lambda x: -x[1]))}")
    except Exception as e:
        print(f"  {label:>12s}: OOS验证失败 ({str(e)[:40]})")

print(f"\n  总耗时: {time.time()-t_total:.0f}s")
print("=" * 70)
print("  因子炼金术 v7 FAST 完成")
print("=" * 70)
