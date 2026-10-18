# -*- coding: utf-8 -*-
"""
因果发现模块 — 简约 Double ML 筛选有因果效应的因子
====================================================

核心思路:
  传统 ICIR 找的是「相关性」—— regime 切换时可能消失。
  Double ML 找的是「因果效应」(ATE) —— 控制混杂后,
  因子对收益的独立因果影响。因果结构比相关性更稳定。

方法 (不依赖 econml/numba, 纯 sklearn + scipy):
  1. 加载 86 因子 (winsorize + rank 标准化)
  2. 对每个因子, 三阶段 Double ML 估计 ATE:
     a) Orthogonalize treatment: regress factor on confounders → residuals
     b) Orthogonalize outcome: regress return on confounders → residuals  
     c) Regress return residuals on factor residuals → ATE estimate
  3. Bootstrap 95% CI + p-value
  4. 筛选显著 + 方向稳定的因果因子
  5. 对比 ICIR 筛选结果 + 因果因子 ElasticNetCV 组合
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from data.loader import load_all_data
from factors import ALL_FACTORS
from config import FACTOR_DEFS
from factors.composite import standardize_factors
from factors.preprocess import winsorize_price_data, winsorize_financial_data, winsorize_valuation_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from scipy import stats

# =====================================================
# Config
# =====================================================
OUTPUT_DIR = SCRIPT_DIR / 'output'
FACTOR_CACHE = OUTPUT_DIR / 'factor_cache_winsorized.pkl'
RESULT_JSON = OUTPUT_DIR / 'causal_discovery_result.json'

N_CONFOUNDERS = 8       # 用 top-K ICIR 因子控制混杂
P_THRESHOLD = 0.05      # 显著性阈值
ATE_MIN = 0.005         # 最小因果效应 (周收益绝对幅度)
N_BOOTSTRAP = 200       # 残差上 Bootstrap (近零成本, 200 次得紧 CI)
MAX_ROWS = None         # 全精度: 不降采样, 用全量 130万行 (交叉拟合+GPU 使全量可行)

print("=" * 70)
print("  因果发现 — Double ML (sklearn 实现)")
print(f"  混杂控制: Top {N_CONFOUNDERS} ICIR 因子")
print("=" * 70)

# =====================================================
# Phase 1: 数据加载
# =====================================================
print("\n[Phase 1] 数据加载")
t1 = time.time()
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
close = price_data.get('close')
print(f"  价格矩阵: {close.shape} | {time.time()-t1:.0f}s")

# forward returns (target) — 原始价格
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape}")

# Winsorize
price_data = winsorize_price_data(price_data)

# =====================================================
# Phase 2: 因子加载/计算
# =====================================================
if FACTOR_CACHE.exists():
    print(f"\n[Phase 2] 因子 [CACHE: {FACTOR_CACHE.name}]")
    with open(FACTOR_CACHE, 'rb') as f:
        factor_dfs = pickle.load(f)
    print(f"  {len(factor_dfs)} 因子")
else:
    print(f"\n[Phase 2] 因子计算 (~7min)...")
    financial_data = winsorize_financial_data(data.get('financial_data'))
    valuation_data = winsorize_valuation_data(data.get('valuation_data'))
    industry_map = data.get('industry_map', {})
    
    factor_dfs = {}
    for name in FACTOR_DEFS:
        if name not in ALL_FACTORS or name.endswith('_neut'):
            continue
        factor_obj = ALL_FACTORS[name]()
        try:
            df = factor_obj.compute(price_data=price_data, financial_data=financial_data,
                                     valuation_data=valuation_data, industry_map=industry_map)
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df_weekly = df.resample('W').last()
                # 稀疏性检测 (修复数据坍缩根因):
                #   季度财报因子 resample('W').last() 后仍有 ~286 行, 但绝大多数为 NaN
                #   (仅披露周有值)。原 is_fin = shape<=10 对此失效, 导致 ffill 分支永不触发,
                #   13 个财报因子仅 21 周有效 → Phase 5 交集坍缩到 ~19 周。
                #   改为按「NaN 占比」判断低频因子, 触发时间轴前向填充 (用最近一次有效财报值, 无前视)。
                frac_nan = df_weekly.isna().mean().mean()
                n_weeks = df_weekly.shape[0]
                is_sparse = (n_weeks <= 30) or (frac_nan > 0.5)
                df_weekly = df_weekly.reindex(close_weekly.index)
                if is_sparse:
                    # 低频/财报因子: 时间轴前向填充 (严格 no-lookahead, 不 bfill 避免前视)
                    factor_dfs[name] = df_weekly.ffill()
                else:
                    # 日频因子: 本身密集, 仅对齐周频索引 (不 ffill, 保留其真实 NaN)
                    factor_dfs[name] = df_weekly
        except Exception as e:
            pass
    print(f"  {len(factor_dfs)} 因子 → 缓存: {FACTOR_CACHE}")
    with open(FACTOR_CACHE, 'wb') as f:
        pickle.dump(factor_dfs, f)

# =====================================================
# Phase 3: Rank 标准化
# =====================================================
print("\n[Phase 3] Rank 标准化")
factor_dfs_std = standardize_factors(factor_dfs)
print(f"  {len(factor_dfs_std)} 因子")

# =====================================================
# Phase 4: ICIR 基线
# =====================================================
from evaluation.ic_analysis import compute_ic_icir

print(f"\n[Phase 4] ICIR 基线")
t4 = time.time()
factor_icir = {}
for fname, fdf in factor_dfs_std.items():
    try:
        ic_ir = compute_ic_icir(fdf, forward_returns, min_samples=10)
        if ic_ir and 'icir' in ic_ir and pd.notna(ic_ir['icir']):
            factor_icir[fname] = ic_ir['icir']
    except:
        pass

icir_sorted = sorted(factor_icir.items(), key=lambda x: -abs(x[1]))
print(f"  {len(factor_icir)} 因子 | {time.time()-t4:.0f}s")
print(f"  Top 10 ICIR:")
for name, val in icir_sorted[:10]:
    print(f"    {name}: {val:+.4f}")

# Control factors = top ICIR (excluding the current treatment)
control_factors = [name for name, _ in icir_sorted[:N_CONFOUNDERS]]

# =====================================================
# Phase 5: 构建矩阵
# =====================================================
print(f"\n[Phase 5] 构建矩阵")
t5 = time.time()

# Stack all weeks into long format
# 完整性要求: 仅要求「混杂因子(control_factors) + 收益 y」非空即纳入。
# 治疗因子(单个)的 NaN 由 process_factor 逐因子屏蔽 (np.isfinite(Ti))。
# 不再要求全部 93 因子同时非空 —— 那会把交集窗口砍到 ~25 周 (基本面因子稀疏),
# 严重损失样本。改为混杂完整即可, 恢复全量 130万+ 行。
all_factor_names = sorted(factor_dfs_std.keys())
X_parts, y_parts = [], []
valid_weeks = []
for week in forward_returns.index:
    row_dict = {}
    for fn in all_factor_names:
        fdf = factor_dfs_std[fn]
        if week in fdf.index:
            row_dict[fn] = fdf.loc[week]
    # 至少要有全部混杂因子当周可用
    if not all(cf in row_dict for cf in control_factors):
        continue
    week_df = pd.DataFrame(row_dict)
    week_y = forward_returns.loc[week].reindex(week_df.index)
    # 只要求混杂因子 + y 非空 (治疗因子 NaN 逐因子处理)
    mask = week_df[control_factors].notna().all(axis=1) & week_y.notna()
    if mask.sum() < 30:
        continue
    X_parts.append(week_df[mask])
    y_parts.append(week_y[mask])
    valid_weeks.append(week)

X_all = pd.concat(X_parts, ignore_index=True)
y_all = pd.concat(y_parts, ignore_index=True)
print(f"  X={X_all.shape}, y={y_all.shape}, 周数={len(valid_weeks)} | {time.time()-t5:.0f}s")

# 降采样: 全量 130万行对 Double ML 过密, 随机抽 20万行即可 (统计功效足够, 提速 ~6.5x)
if MAX_ROWS and len(X_all) > MAX_ROWS:
    rng_sub = np.random.default_rng(20240722)
    sub_idx = rng_sub.choice(len(X_all), size=MAX_ROWS, replace=False)
    X_all = X_all.iloc[sub_idx].reset_index(drop=True)
    y_all = y_all.iloc[sub_idx].reset_index(drop=True)
    print(f"  [降采样] → X={X_all.shape} (用于因果估计)")

# =====================================================
# Phase 6: Double ML 因果效应估计
# =====================================================
print(f"\n[Phase 6] Double ML ({N_BOOTSTRAP} bootstrap) — XGBoost CUDA 并行")
print(f"  Confounders: {control_factors}")
t6 = time.time()

# Confounders matrix (always available)
W = X_all[control_factors].values.astype(np.float32)
Y = y_all.values.astype(np.float32)

all_factors = [c for c in X_all.columns if c in factor_dfs_std and c not in control_factors]
n_factors = len(all_factors)

# Phase 5 之后不再需要原始因子矩阵, 释放显存/RAM (省 ~2GB, 降低 OOM 风险)
del factor_dfs, factor_dfs_std
gc.collect()

# 单进程顺序执行 (主进程内 XGBoost CUDA 可用, 且 process_factor 的 try/except 兜住单因子异常,
# 避免多进程的 BrokenProcessPool 整池崩溃)。交叉拟合使单因子仅 ~10 次 GBT 拟合, 约 5s/因子。
from causal_worker import process_factor

print(f"  顺序执行: 主进程 XGBoost device=cuda | {n_factors} 因子...")

# 每个因子只取 T 列 (float32)
T_cols = {f: X_all[f].values.astype(np.float32) for f in all_factors}

# 检查点续跑: 若上次进程在 Phase 6 中途崩溃 (850 次 XGBoost CUDA fit 累积吃满显存),
# 载入已完成因子跳过, 不再重算, 且每算完一个因子即落盘, 进程再死也不丢进度。
PARTIAL = OUTPUT_DIR / 'causal_partial.json'
if PARTIAL.exists():
    try:
        with open(PARTIAL, encoding='utf-8') as f:
            causal_results = json.load(f)
        print(f"  [续跑] 载入已完成的 {len(causal_results)} 个因子, 跳过")
    except Exception:
        causal_results = {}
else:
    causal_results = {}

done = 0
for fname in all_factors:
    if fname in causal_results and 'ate' in causal_results.get(fname, {}):
        done += 1
        continue  # 已完成, 续跑跳过
    args = (fname, T_cols[fname], Y, W, factor_icir, P_THRESHOLD, N_BOOTSTRAP)
    fn, res = process_factor(args)
    causal_results[fn] = res
    del T_cols[fname]   # 释放该因子列, 降低内存压力
    done += 1
    # 检查点: 每算完一个因子即落盘
    with open(PARTIAL, 'w', encoding='utf-8') as f:
        json.dump(causal_results, f, ensure_ascii=False)
    if done % 5 == 0:
        print(f"    {done}/{n_factors}  done | GPU活跃 | {time.time()-t6:.0f}s", flush=True)

elapsed = time.time() - t6
print(f"  Done: {n_factors} 因子 | {elapsed:.0f}s ({elapsed/n_factors:.1f}s/因子)", flush=True)
PARTIAL.unlink(missing_ok=True)   # Phase 6 成功, 清检查点

# 防御性: Phase 6 一结束就落盘核心因果结果, 避免后续 Phase 出错丢失
with open(RESULT_JSON, 'w', encoding='utf-8') as f:
    json.dump({'causal_results': causal_results,
               'control_factors': control_factors,
               'n_bootstrap': N_BOOTSTRAP,
               'phase': 'phase6_done'}, f, ensure_ascii=False, indent=2)
print(f"  [已落盘核心结果] {RESULT_JSON}", flush=True)

# =====================================================
# Phase 7: 结果分析
# =====================================================
print("\n" + "=" * 70)
print("[Phase 7] 因果发现结果")
print("=" * 70)

sorted_ate = sorted(causal_results.items(), key=lambda x: -abs(x[1]['ate']))

print(f"\n  Top 20 因果因子 (|ATE| 降序):")
print(f"  {'因子':<35s} {'ATE':>8s}  {'p_val':>8s}  {'显著':>4s}  {'ICIR':>8s}  {'ICIR一致':>8s}")
print(f"  {'-'*85}")
for name, r in sorted_ate[:20]:
    ate_str = f"{r['ate']:+.5f}"
    p_str = f"{r['p_value']:.4f}" if r['p_value'] > 0.001 else f"{r['p_value']:.1e}"
    sig = '✅' if r['significant'] else '  '
    icir = r['icir']
    consistent = '✅' if r['ate'] * icir > 0 else '⚠️'
    print(f"  {name:<35s} {ate_str:>8s}  {p_str:>8s}  {sig:>4s}  {icir:>+.4f}  {consistent:>8s}")

n_sig = sum(1 for r in causal_results.values() if r['significant'])
n_ate = sum(1 for r in causal_results.values() if abs(r['ate']) > ATE_MIN)
n_both = sum(1 for r in causal_results.values() if r['significant'] and abs(r['ate']) > ATE_MIN)

print(f"\n  p < {P_THRESHOLD}: {n_sig}/{len(causal_results)}")
print(f"  |ATE| > {ATE_MIN}: {n_ate}")
print(f"  两者满足: {n_both}")

# Overlap analysis
causal_sig = set(name for name, r in causal_results.items() if r['significant'])
icir_top20 = set(name for name, _ in icir_sorted[:20])
causal_top20 = set(name for name, _ in sorted_ate[:20])

overlap = causal_sig & icir_top20
causal_only = causal_top20 - icir_top20
icir_only = icir_top20 - causal_top20

print(f"\n  ICIR Top20 ∩ Causal Top20: {len(overlap)}")
if causal_only:
    print(f"  因果独有 (低ICIR高因果): {sorted(causal_only)[:10]}")
if icir_only:
    print(f"  ICIR独有 (高ICIR低因果): {sorted(icir_only)[:10]}")

# =====================================================
# Phase 8: 因果因子组合
# =====================================================
print("\n[Phase 8] 因果因子 ElasticNetCV")
causal_factors = [name for name, r in sorted_ate
                  if r['significant'] and abs(r['ate']) > ATE_MIN]

if len(causal_factors) >= 3:
    c_idx = [all_factors.index(f) for f in causal_factors]
    c_X = X_all[causal_factors].values
    en = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                      cv=5, max_iter=5000, n_jobs=1, random_state=42)
    en.fit(c_X, Y)
    coefs = pd.Series(en.coef_, index=causal_factors)
    nonzero = coefs[coefs != 0]
    print(f"  {len(causal_factors)}→{len(nonzero)} 因子, alpha={en.alpha_:.6f}, l1_ratio={en.l1_ratio_:.3f}")
    for fn, coef in nonzero.sort_values(key=abs, ascending=False).items():
        ic = factor_icir.get(fn, 0)
        at = causal_results[fn]['ate']
        print(f"    {fn}: w={coef:+.4f}  ATE={at:+.5f}  ICIR={ic:+.4f}")
else:
    print(f"  [SKIP] 因果因子={len(causal_factors)} < 3")

# =====================================================
# Phase 9: 对比 state_2 基线因子
# =====================================================
state2_factors = ['asset_growth', 'turnover_cv_20d', 'volume_stability', 'high_low_range', 'bp']
print(f"\n{'='*70}")
print("[Phase 9] state_2 基线因子因果分析")
print(f"{'='*70}")
print(f"  {'因子':<25s} {'ATE':>8s}  {'p_val':>8s}  {'显著':>4s}  {'ICIR':>8s}")
for fn in state2_factors:
    if fn in causal_results:
        r = causal_results[fn]
        sig = '✅' if r['significant'] else '  '
        print(f"  {fn:<25s} {r['ate']:>+.5f}  {r['p_value']:>.4f}  {sig:>4s}  {r['icir']:>+.4f}")
    else:
        print(f"  {fn:<25s} [NOT IN POOL]")

# =====================================================
# 保存
# =====================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_data = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'method': 'double_ml_sklearn',
    'n_confounders': N_CONFOUNDERS,
    'control_factors': control_factors,
    'n_factors': len(causal_results),
    'n_significant': n_sig,
    'n_ate_positive': n_ate,
    'n_both': n_both,
    'causal_factors': causal_factors,
    'causal_results': causal_results,
    'icir_top20': [n for n, _ in icir_sorted[:20]],
    'overlap_icir_causal': len(overlap),
    'state2_analysis': {fn: causal_results.get(fn, {}) for fn in state2_factors},
}
with open(RESULT_JSON, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)
print(f"\n  结果: {RESULT_JSON}")
print(f"  总耗时: {time.time()-t1:.0f}s")
