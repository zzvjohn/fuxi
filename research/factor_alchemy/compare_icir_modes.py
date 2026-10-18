"""
ICIR Obj 模式对比: raw vs tanh vs clip
======================================
独立对比脚本: 复用 Phase 1-3 的数据，只跑 NSGA-II (Phase 4) 三次，
比较三种 ICIR 打分模式下的帕累托前沿区分度。

用法: python compare_icir_modes.py
"""
import sys, time, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

# ── 加载数据 (复用 run_fa 的 Phase 1-3) ──
from config import (FACTOR_DEFS, OOS_SPLIT_DATE, MAX_FACTORS, N_OBJECTIVES,
                    CONCENTRATION_GROUPS, MAX_CATEGORY_WEIGHT,
                    CATEGORY_CONCENTRATION_PENALTY_MULT)
from factors import ALL_FACTORS
from data.loader import load_all_data
from evaluation.ic_analysis import compute_ic_icir
from evaluation.correlation import factor_correlation_matrix
from ga.chromosome import random_chromosome
from ga.nsga2 import nsga2_run
from ga.fitness import icir_to_objective

print("=" * 70)
print("  ICIR Obj 模式对比: raw vs tanh vs clip")
print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ▸ Phase 1
print("\n[Phase 1] 加载数据...")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-04-30')
price_data = data['price_data']
financial_data = data['financial_data']
valuation_data = data['valuation_data']
close = price_data.get('close')
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
mcap_df = None
if valuation_data is not None and hasattr(valuation_data, 'columns'):
    mcap_df = valuation_data.get('market_cap', None)
print(f"  完成 ({time.time()-t1:.0f}s)")

# ▸ Phase 2: 因子计算
print("\n[Phase 2] 计算因子...")
t2 = time.time()
factor_dfs = {}
is_size_map = {}
for name, cfg in FACTOR_DEFS.items():
    if name not in ALL_FACTORS:
        continue
    try:
        obj = ALL_FACTORS[name]()
        df = obj.compute(price_data, financial_data, valuation_data)
        if df is not None and not df.empty:
            df_weekly = df.resample('W').last() if hasattr(df, 'resample') else df
            factor_dfs[name] = df_weekly
            is_size_map[name] = (cfg.get('category') == 'size')
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")
print(f"  计算完成: {len(factor_dfs)} 个因子 ({time.time()-t2:.0f}s)")

# ▸ Phase 3: 单因子 IC 评估
print("\n[Phase 3] 单因子 IC 评估...")
t3 = time.time()
ic_records = []
for name in factor_dfs:
    try:
        result = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_records.append({
            'name': name,
            'IC_mean': result['ic_mean'],
            'ICIR': result['icir'],
            'IC_std': result.get('ic_std', 0),
        })
    except Exception:
        pass
ic_summary = pd.DataFrame(ic_records).set_index('name')
print(f"  评估完成: {len(ic_summary)} 个因子 ({time.time()-t3:.0f}s)")

# 预计算 IC 序列
print("  预计算 IC 序列...")
factor_ic_series = {}
for name in factor_dfs:
    if name not in ic_summary.index:
        continue
    try:
        icr = compute_ic_icir(factor_dfs[name], forward_returns)
        ic_s = icr.get('ic_series', pd.Series(dtype=float))
        if len(ic_s) > 50:
            factor_ic_series[name] = ic_s
    except Exception:
        pass
print(f"  IC 序列: {len(factor_ic_series)}/{len(factor_dfs)} 个")

# 相关性矩阵
valid_names = [n for n in factor_dfs if n in ic_summary.index
               and not np.isnan(ic_summary.loc[n, 'ICIR'])]
try:
    corr_matrix_full = factor_correlation_matrix(
        {n: factor_dfs[n] for n in valid_names if n in factor_dfs}
    )
except Exception:
    corr_matrix_full = pd.DataFrame(np.eye(len(valid_names)),
                                    index=valid_names, columns=valid_names)

# OOS 分割
fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]
oos_dt = pd.Timestamp(OOS_SPLIT_DATE)
train_ic = {}
for name, ic_series in factor_ic_series.items():
    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
        ic_train = ic_series[ic_series.index < oos_dt].dropna()
        if len(ic_train) > 10:
            train_ic[name] = ic_train
print(f"  训练集 IC: {len(train_ic)} 个 | 训练周: {train_fr.shape[0]}")

# ▸▸▸ 三次 NSGA-II 运行
import config  # 用于 monkey-patch ICIR_OBJ_MODE
n_factors = len(valid_names)
pop_size, n_gen = 40, 5

modes = ['raw', 'tanh', 'clip']
mode_labels = {
    'raw': 'raw ICIR',
    'tanh': 'tanh(ICIR/0.5)',
    'clip': 'clip(ICIR/1.5)',
}
results = {}

for mode in modes:
    print(f"\n{'='*70}")
    print(f"[Phase 4] NSGA-II ({mode_labels[mode]})")
    print(f"{'='*70}")

    # Monkey-patch 模式
    config.ICIR_OBJ_MODE = mode

    seed = 42
    np.random.seed(seed)
    pop = [random_chromosome(n_factors) for _ in range(pop_size)]

    t4 = time.time()
    final_pop, final_obj, pareto = nsga2_run(
        pop, valid_names, factor_dfs,
        train_fr, mcap_df,
        n_generations=n_gen, pop_size=pop_size,
        crossover_prob=0.7, mutation_prob=0.3, mutation_sigma=0.3,
        corr_matrix=corr_matrix_full,
        ic_summary=ic_summary,
        factor_ic_series=train_ic,
        verbose=True,
    )
    elapsed = time.time() - t4

    pareto_obj = pareto['objectives']
    pareto_icir_raw = pareto_obj[:, 0]
    pareto_stab = pareto_obj[:, 1]
    pareto_w = pareto['weights']

    results[mode] = {
        'pareto_obj': pareto_obj,
        'icir_raw': pareto_icir_raw,
        'stab': pareto_stab,
        'weights': pareto_w,
        'n_solutions': len(pareto_w),
        'elapsed': elapsed,
    }

    print(f"\n  帕累托前沿: {len(pareto_w)} 个解 ({elapsed:.0f}s)")
    print(f"  ICIR 范围: [{pareto_icir_raw.min():.3f}, {pareto_icir_raw.max():.3f}]")
    print(f"  Stab 范围: [{pareto_stab.min():.3f}, {pareto_stab.max():.3f}]")

# ▸▸▸ 对比分析
print("\n" + "=" * 70)
print("  对比结果")
print("=" * 70)

# 1. 区分度指标
print(f"\n  区分度对比:")
print(f"  {'模式':>12s} {'#解':>4s} {'ICIR_min':>10s} {'ICIR_max':>10s}"
      f" {'ICIR_range':>10s} {'Stab_range':>10s} {'Pareto面积≈':>12s}")
print(f"  {'-'*68}")

for mode in modes:
    r = results[mode]
    icir_min = r['icir_raw'].min()
    icir_max = r['icir_raw'].max()
    icir_range = icir_max - icir_min
    stab_range = r['stab'].max() - r['stab'].min()
    area = icir_range * stab_range  # 2D Pareto 超体积近似
    print(f"  {mode_labels[mode]:>12s} {r['n_solutions']:4d} "
          f"{icir_min:10.3f} {icir_max:10.3f} {icir_range:10.3f} "
          f"{stab_range:10.3f} {area:12.4f}")

# 2. 超体积 (Volumetric) 对比
print(f"\n  前沿多样性:")
best_area = max(r['icir_raw'].max() - r['icir_raw'].min()
                 for r in results.values())
for mode in modes:
    r = results[mode]
    icir_range = r['icir_raw'].max() - r['icir_raw'].min()
    ratio = icir_range / best_area if best_area > 0 else 0
    print(f"  {mode_labels[mode]:>12s}: {'#' * int(ratio * 40)} {icir_range:.3f}")

# 3. Obj 值的 Kendall 一致性
print(f"\n  目标值与原始 ICIR 的秩一致性:")
for mode in modes:
    r = results[mode]
    obj_vals = r['pareto_obj'][:, 0]
    icir_raw = r['icir_raw']
    
    # 计算 Spearman rank correlation between obj and raw ICIR
    from scipy.stats import spearmanr
    if len(obj_vals) > 2:
        corr, p = spearmanr(obj_vals, icir_raw)
        print(f"  {mode_labels[mode]:>12s}: ρ={corr:+.4f}  (p={p:.4f})")

# 4. 覆盖率: 每个模式发现了几个不同的因子
print(f"\n  因子发现多样性:")
for mode in modes:
    r = results[mode]
    all_factors = set()
    for w in r['weights']:
        all_factors.update(w.keys())
    print(f"  {mode_labels[mode]:>12s}: {len(all_factors)} 个不同因子")

# 5. 具体显示各模式的前沿 (Top 5)
print(f"\n  {'='*70}")
print(f"  各模式帕累托前沿 — Top 5 个解")
print(f"  {'='*70}")

for mode in modes:
    r = results[mode]
    pareto_obj = r['pareto_obj']
    pareto_w = r['weights']
    
    print(f"\n  【{mode_labels[mode]}】")
    print(f"  {'#':>3s} {'Obj1(ICIR)':>12s} {'Obj2(Stab)':>12s} {'因子数':>6s} {'因子组合':>50s}")
    print(f"  {'-'*85}")
    
    # 按 Obj1 排序
    order = np.argsort(-pareto_obj[:, 0])
    for rank, idx in enumerate(order[:5]):
        obj1 = pareto_obj[idx, 0]
        obj2 = pareto_obj[idx, 1]
        factors = pareto_w[idx]
        factor_str = '+'.join(list(factors.keys())[:5])
        if len(factors) > 5:
            factor_str += f'...({len(factors)}个)'
        print(f"  {rank+1:3d} {obj1:12.4f} {obj2:12.4f} {len(factors):6d} {factor_str:50s}")

print(f"\n{'='*70}")
print(f"  对比完成 — 区分度 = ICIR_range 越大越好")
print(f"{'='*70}")
