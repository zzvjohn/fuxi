"""
快速隔离测试: 直接调用 compute_multi_objective
诊断 Obj1/Obj2/Obj3 全部为0的根因
"""
import sys, os
from pathlib import Path

# 确保路径正确
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import time

from config import *
from factors.composite import standardize_factors, combine_factors_vectorized, weights_from_chromosome
from ga.fitness import compute_multi_objective, icir_to_objective, _estimate_weekly_turnover
from ga.chromosome import random_chromosome
from evaluation.ic_analysis import compute_ic_icir

# === 加载数据 (与run_fa.py相同) ===
DATA_DIR = ROOT.parent.parent / 'data'

print("=" * 60)
print("加载数据...")
print("=" * 60)

# 加载价格
price_path = DATA_DIR / 'daily_prices.csv'
prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
print(f"  价格: {prices.shape}")

# 加载复权因子
adj_path = DATA_DIR / 'adj_factor.csv'
adj = pd.read_csv(adj_path, index_col=0, parse_dates=True)
adj = adj.fillna(0)
print(f"  复权因子: {adj.shape}")

# 后复权价格
hfq = prices * adj.reindex_like(prices)
print(f"  后复权: {hfq.shape}")

# 加载估值
basic_path = DATA_DIR / 'daily_basic.csv'
daily_basic = pd.read_csv(basic_path, index_col=0, parse_dates=True)
print(f"  估值数据: {daily_basic.shape}")

# 加载财务
fina_path = DATA_DIR / 'fina_indicator_rich.csv'
fina = pd.read_csv(fina_path, index_col=0, parse_dates=True)
print(f"  财务数据: {fina.shape}")

print("\n计算周频收益...")
t0 = time.time()
weekly_prices = hfq.resample('W').last()
weekly_ret = weekly_prices.pct_change()
# 对齐日期: 确保从2021年开始
start_mask = weekly_ret.index >= pd.Timestamp('2021-01-01')
weekly_ret = weekly_ret[start_mask]
print(f"  周频收益: {weekly_ret.shape}, 日期范围: {weekly_ret.index[0]} ~ {weekly_ret.index[-1]}")

# 前向收益 (下周收益)
forward_returns_full = weekly_ret.shift(-1)

# 市值
mcap_raw = daily_basic[['total_mv']].copy()
mcap_raw = mcap_raw.resample('W').last()
mcap_df = np.log(mcap_raw.clip(lower=1e6))

t1 = time.time()
print(f"  数据准备完成 ({t1-t0:.0f}s)")

# === OOS 分割 ===
oos_split = pd.Timestamp(OOS_SPLIT_DATE)
train_mask = forward_returns_full.index < oos_split
train_fr = forward_returns_full[train_mask].copy()
print(f"\n训练集: {train_fr.shape[0]} 周 ({train_fr.index[0]} ~ {train_fr.index[-1]})")

# === 构建因子池 (从 config.FACTOR_DEFS) ===
print("\n构建因子池...")
from factors.factor_computer import compute_all_factors
factor_dfs = compute_all_factors(
    hfq, daily_basic, fina, weekly_ret, mcap_df,
    factor_names=list(FACTOR_DEFS.keys())
)
print(f"  有效因子: {len(factor_dfs)}")

# === 预标准化 ===
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  标准化: {len(factor_dfs_std)}")

# === 预计算相关性矩阵 ===
from evaluation.ic_analysis import compute_correlation_matrix
corr_matrix_full = compute_correlation_matrix(factor_dfs_std)
print(f"  相关性矩阵: {corr_matrix_full.shape}")

# === 测试: 直接调用 compute_multi_objective ===
print("\n" + "=" * 60)
print("测试 compute_multi_objective")
print("=" * 60)

factor_names = list(factor_dfs_std.keys())
n_factors = len(factor_names)

# 用 ICIR 最高的因子初始化 (模拟 icir_biased)
from run_fa import _compute_single_factor_icir
icir_values = {}
for name in factor_names[:20]:  # 只测前20个加快速度
    try:
        icir = _compute_single_factor_icir(factor_dfs_std[name], train_fr)
        icir_values[name] = icir
    except Exception as e:
        icir_values[name] = 0.0
        print(f"  [WARN] ICIR {name}: {e}")

print(f"  ICIR计算完成: {len(icir_values)} 因子")

# 测试单个染色体
from ga.chromosome import Chromosome

# 手动构造一个包含 top4 因子的染色体
top4 = sorted(icir_values.items(), key=lambda x: -abs(x[1]))[:4]
print(f"  Top4: {[(n, round(v,3)) for n,v in top4]}")

# 手动构造染色体 (高权重给 top4)
chromo = np.random.randn(n_factors) * 0.1
for i, (name, icir) in enumerate(top4):
    idx = factor_names.index(name)
    chromo[idx] = 2.0  # 高权重

print(f"\n  染色体: {len(chromo)} 维")

# ★ 调用 compute_multi_objective
print("\n  >>> 调用 compute_multi_objective <<<")
try:
    result = compute_multi_objective(
        chromo, factor_names, factor_dfs,
        train_fr, mcap_df,
        corr_matrix=corr_matrix_full,
        factor_dfs_std=factor_dfs_std,
        debug_obj3=True,  # 强制 Obj3 诊断输出
    )
    print(f"\n  ✅ 成功! 结果: Obj1={result[0]:.3f} Obj2={result[1]:.3f} Obj3={result[2]:.3f}")
except Exception as e:
    print(f"\n  ❌ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试不同 factor combo")
print("=" * 60)

# 测试2: 只用 plrc_12 (最高ICIR)
print("\n--- 单因子: plrc_12 ---")
w_single = {}
for name in factor_names:
    if name == 'plrc_12':
        w_single[name] = 1.0
    else:
        w_single[name] = 0.0

chromo_single = np.zeros(n_factors)
idx = factor_names.index('plrc_12')
chromo_single[idx] = 1.0

try:
    result = compute_multi_objective(
        chromo_single, factor_names, factor_dfs,
        train_fr, mcap_df,
        corr_matrix=corr_matrix_full,
        factor_dfs_std=factor_dfs_std,
        debug_obj3=True,
    )
    print(f"  结果: Obj1={result[0]:.3f} Obj2={result[1]:.3f} Obj3={result[2]:.3f}")
except Exception as e:
    print(f"  失败: {e}")

# 测试3: 验证 composite 是否有预测力
print("\n--- 验证单因子 composite ICIR ---")
sel = {n: factor_dfs_std[n] for n in factor_names[:1]}
try:
    comp = combine_factors_vectorized(sel, {factor_names[0]: 1.0})
    icir_r = compute_ic_icir(comp, train_fr)
    print(f"  {factor_names[0]}: ICIR={icir_r['icir']:.3f}, IC_series前5={icir_r.get('ic_series', [])[:5]}")
except Exception as e:
    print(f"  失败: {e}")

# 测试4: 验证 plrc_12 composite ICIR
print("\n--- 验证 plrc_12 composite ICIR ---")
try:
    sel = {n: factor_dfs_std[n] for n in ['plrc_12']}
    comp = combine_factors_vectorized(sel, {'plrc_12': 1.0})
    icir_r = compute_ic_icir(comp, train_fr)
    print(f"  plrc_12 ICIR={icir_r['icir']:.3f}, ic_series类型={type(icir_r.get('ic_series'))}")
    ic_s = icir_r.get('ic_series')
    if ic_s is not None:
        ic_list = list(ic_s) if not isinstance(ic_s, list) else ic_s
        print(f"  ic_series 前10: {[round(x,4) for x in ic_list[:10]]}")
        print(f"  ic_series 后10: {[round(x,4) for x in ic_list[-10:]]}")
        
        # 计算 rank autocorr
        import pandas as pd
        s = pd.Series(ic_list)
        rank_s = s.rank()
        print(f"  rank 前10: {list(rank_s.values[:10])}")
        ac = rank_s.autocorr(lag=1)
        print(f"  rank_autocorr(lag=1): {ac}")
except Exception as e:
    print(f"  失败: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 诊断完成")
