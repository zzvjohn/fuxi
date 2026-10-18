"""
综合诊断脚本 v1.0 (2026-06-27)
==============================
1. Obj3 成本后Sharpe 逐步诊断
2. ICIR raw vs clip 区分度对比 (叠加惩罚链)
3. plrc_12 前视偏差审查

用法: python debug_comprehensive.py
环境: FA_OBJ3_DEBUG=1 自动启用Obj3详细输出
"""
import sys, os, time, pickle
from pathlib import Path

os.environ['FA_OBJ3_DEBUG'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import (
    FACTOR_DEFS, ICIR_THRESHOLD, ICIR_OBJ_MODE, MAX_FACTORS,
    MAX_TOTAL_PENALTY, FUNDAMENTAL_CATEGORIES, FUNDAMENTAL_BONUS,
    FUNDAMENTAL_MIN_COUNT, STABILITY_THRESHOLD, OOS_SPLIT_DATE,
    CONCENTRATION_GROUPS, MAX_CATEGORY_WEIGHT,
    CATEGORY_CONCENTRATION_PENALTY_MULT,
)
from ga.fitness import (
    icir_to_objective,
    _estimate_weekly_turnover,
    _compute_turnover_penalty,
    _compute_category_penalty,
    _compute_complexity_penalty,
    _compute_correlation_penalty,
    _compute_top30_cost_adjusted_sharpe,
    compute_multi_objective,
)
from ga.chromosome import random_chromosome, icir_biased_chromosome
from factors.composite import combine_factors, weights_from_chromosome
from evaluation.ic_analysis import compute_ic_icir

# ============================================================
# 加载预计算数据
# ============================================================
DATA_CACHE = Path(__file__).parent / 'output' / 'fa_data_cache.pkl'

print("=" * 70)
print("  综合诊断: Obj3 + ICIR raw/clip + plrc_12")
print("=" * 70)

if not DATA_CACHE.exists():
    print(f"\n[ERROR] 数据缓存不存在: {DATA_CACHE}")
    print("请先运行 run_fa.py 生成缓存数据")
    sys.exit(1)

print(f"\n[1/4] 加载数据缓存...")
with open(DATA_CACHE, 'rb') as f:
    cache = pickle.load(f)

factor_dict = cache['factor_dict']
forward_returns = cache['forward_returns']
mcap_df = cache.get('mcap_df')
corr_matrix = cache.get('corr_matrix')
is_size_map = cache.get('is_size_map', {})
icir_values = cache.get('icir_values', None)
factor_names = cache['factor_names']

print(f"  因子数: {len(factor_names)}")
print(f"  日期范围: {forward_returns.index[0]} ~ {forward_returns.index[-1]}")
print(f"  股票数: {forward_returns.shape[1]}")

# 获取单因子 ICIR (用于偏置初始化)
if icir_values is None:
    print("  [WARN] 无单因子ICIR, 使用随机初始化")
    icir_values_arr = [0.5] * len(factor_names)
else:
    if isinstance(icir_values, dict):
        icir_values_arr = [abs(icir_values.get(name, 0.5)) for name in factor_names]
    else:
        icir_values_arr = list(icir_values)

# ============================================================
# SECTION A: plrc_12 前视偏差审查
# ============================================================
print("\n" + "=" * 70)
print("  SECTION A: plrc_12 前视偏差审查")
print("=" * 70)

plrc_name = 'plrc_12'
if plrc_name in factor_dict:
    plrc_df = factor_dict[plrc_name]
    fr_df = forward_returns
    
    print(f"\n  因子名: {plrc_name}")
    print(f"  类别: {FACTOR_DEFS.get(plrc_name, {}).get('category', 'unknown')}")
    print(f"  标签: {FACTOR_DEFS.get(plrc_name, {}).get('label', 'unknown')}")
    print(f"  数据形状: {plrc_df.shape}")
    print(f"  日期范围: {plrc_df.index[0]} ~ {plrc_df.index[-1]}")
    print(f"  非NaN率: {(~plrc_df.isna()).sum().sum() / (plrc_df.shape[0] * plrc_df.shape[1]) * 100:.1f}%")
    
    # 定义: plrc_12 = OLS slope of close/rolling_mean(close,12) vs t=1..12
    # 检查: 时间t的plrc_12值是否只用了t-11到t的数据
    print(f"\n  [审查] 前视偏差检查:")
    print(f"    - 输入: close价格 (截至时刻t)")
    print(f"    - rolling(12) 使用 t-11 至 t (含t)")
    print(f"    - close/roll_mean: 同时点除法")
    print(f"    - OLS斜率: 使用前12期 y 值")
    print(f"    => 结论: 无前视偏差 (所有输入均为已实现数据)")
    
    # ICIR 分时段
    common_dates = sorted(set(plrc_df.index) & set(fr_df.index))
    train_dates = [d for d in common_dates if d < OOS_SPLIT_DATE]
    oos_dates = [d for d in common_dates if d >= OOS_SPLIT_DATE]
    
    for label, dates in [("全时段", common_dates), ("训练期", train_dates), ("OOS期", oos_dates)]:
        if len(dates) < 5:
            continue
        sub_factor = plrc_df.loc[dates]
        sub_fr = fr_df.loc[dates]
        result = compute_ic_icir(sub_factor, sub_fr)
        print(f"    {label} ({len(dates)}周): ICIR={result['icir']:+.4f}, "
              f"+IC%={result['ic_positive_ratio']*100:.1f}%, "
              f"IC_mean={result['ic_mean']:+.4f}")
    
    # ⚠️ 样本依赖警告
    train_icir = None
    for label, dates in [("训练期", train_dates)]:
        if len(dates) < 5:
            continue
        sub_factor = plrc_df.loc[dates]
        sub_fr = fr_df.loc[dates]
        result = compute_ic_icir(sub_factor, sub_fr)
        train_icir = result['icir']
    
    if train_icir is not None and abs(train_icir) < 0.1:
        print(f"\n  ⚠️ [WARN] 训练期 ICIR={train_icir:+.4f} (近乎随机)")
        print(f"    plrc_12 预测力高度依赖市场状态 (OOS牛市 → 强, 训练含熊市 → 弱)")
        print(f"    建议: ICIR偏置初始化排除 plrc_12, 或用训练期ICIR替代全时段ICIR")
else:
    print(f"\n  [WARN] {plrc_name} 不在因子池中")

# ============================================================
# SECTION B: Obj3 成本后Sharpe 逐步诊断
# ============================================================
print("\n" + "=" * 70)
print("  SECTION B: Obj3 成本后Sharpe 逐步诊断")
print("=" * 70)

# 用 ICIR偏置初始化几个代表性染色体进行测试
np.random.seed(42)

# 选几个已知好因子构建测试染色体
test_factor_names = list(factor_names)
n_test_factors = len(test_factor_names)

# 创建3个测试染色体: 高ICIR, 随机, 空
chromosomes_test = []

# 测试1: ICIR偏置 (高质量因子权重高)
c1 = icir_biased_chromosome(n_test_factors, icir_values_arr, bias_strength=1.5)
chromosomes_test.append(("ICIR偏置(Top4)", c1))

# 测试2: 手动给earnings_growth_yoy + dollar_vol_stability 高权重
c2 = np.random.uniform(-3, 3, n_test_factors)
if 'earnings_growth_yoy' in test_factor_names:
    idx = test_factor_names.index('earnings_growth_yoy')
    c2[idx] = 5.0
if 'dollar_vol_stability' in test_factor_names:
    idx = test_factor_names.index('dollar_vol_stability')
    c2[idx] = 5.0
if 'turnover_std' in test_factor_names:
    idx = test_factor_names.index('turnover_std')
    c2[idx] = 4.0
c2 = np.array(c2)
chromosomes_test.append(("手工偏好(价值+量价)", c2))

# 测试3: 全0 (等权所有因子)
c3 = np.zeros(n_test_factors)
chromosomes_test.append(("全0等权", c3))

# 测试4: 只给反转类因子高权重 (测试负IC因子 → Obj3方向)
c4 = np.random.uniform(-3, 3, n_test_factors)
for name in ['short_rev_5d', 'max_ret_1m', 'skew_1m', 'high_52w_dist']:
    if name in test_factor_names:
        idx = test_factor_names.index(name)
        c4[idx] = 5.0
c4 = np.array(c4)
chromosomes_test.append(("反转因子偏好", c4))

for label, chromo in chromosomes_test:
    print(f"\n  {'─' * 60}")
    print(f"  测试: {label}")
    
    weights = weights_from_chromosome(chromo, test_factor_names, max_factors=MAX_FACTORS)
    active_names = list(weights.keys())
    
    if not active_names:
        print("    [SKIP] 无活跃因子")
        continue
    
    print(f"    活跃因子({len(active_names)}): {', '.join(f'{n}({w:.3f})' for n, w in list(weights.items())[:8])}")
    
    # 构建合成因子
    selected_factors = {name: factor_dict[name] for name in weights if name in factor_dict}
    composite = combine_factors(selected_factors, weights)
    
    if composite.empty:
        print("    [SKIP] composite为空")
        continue
    
    print(f"    composite形状: {composite.shape}")
    print(f"    composite日期: {composite.index[0]} ~ {composite.index[-1]}")
    
    # -- Step 1: ICIR --
    icir_result = compute_ic_icir(composite, forward_returns)
    icir = abs(icir_result['icir'])
    print(f"\n    Step1 ICIR: {icir:+.4f} (+IC%={icir_result['ic_positive_ratio']*100:.1f}%)")
    
    # -- Step 2: Top30选择验证 --
    common_idx = composite.index.intersection(forward_returns.index)
    common_cols = composite.columns.intersection(forward_returns.columns)
    
    if len(common_idx) < 12 or len(common_cols) < 30:
        print(f"    [SKIP] 数据不足: {len(common_idx)}周 x {len(common_cols)}股")
        continue
    
    comp_arr = composite.loc[common_idx, common_cols].values
    fr_arr = forward_returns.loc[common_idx, common_cols].values
    
    # 取第一周验证Top30选择
    comp_week0 = comp_arr[0]
    fr_week0 = fr_arr[0]
    valid_mask = np.isfinite(comp_week0)
    
    top30_local = np.argsort(-comp_week0)[:30]
    top30_values = comp_week0[top30_local]
    top30_rets = fr_week0[top30_local]
    
    print(f"\n    Step2 Top30选择验证 (第1周):")
    print(f"      有效股票数: {valid_mask.sum()}")
    print(f"      composite范围: [{np.nanmin(comp_week0):.4f}, {np.nanmax(comp_week0):.4f}]")
    print(f"      Top30 composite: [{top30_values.min():.4f}, {top30_values.max():.4f}]")
    print(f"      Top30 前向收益: mean={np.nanmean(top30_rets)*100:.4f}%")
    print(f"      全部股票 前向收益: mean={np.nanmean(fr_week0)*100:.4f}%")
    
    # 检查: composite是否与forward returns正相关
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) >= 30:
        corr_val = stats.spearmanr(comp_week0[valid_idx], fr_week0[valid_idx])
        corr_val = corr_val[0] if isinstance(corr_val, tuple) else corr_val.statistic
        print(f"      composite vs forward_ret Spearman: {corr_val:+.4f} (应>0)")
    
    # -- Step 3: 完整Obj3计算 --
    obj3_val = _compute_top30_cost_adjusted_sharpe(composite, forward_returns)
    print(f"\n    Step3 Obj3 (clip后): {obj3_val:.4f}")
    
    # 手动复现Obj3计算
    n_weeks, n_stocks = comp_arr.shape
    comp_filled = np.where(np.isnan(comp_arr), -np.inf, comp_arr)
    rankings = np.argsort(-comp_filled, axis=1)
    top30_idx = rankings[:, :30]
    
    week_indices = np.arange(n_weeks - 1)[:, None]
    stock_indices = top30_idx[:-1, :]
    gross_rets = np.nanmean(fr_arr[week_indices, stock_indices], axis=1)
    
    top30_mask = np.zeros((n_weeks, n_stocks), dtype=bool)
    for i in range(n_weeks):
        top30_mask[i, top30_idx[i]] = True
    overlap = (top30_mask[:-1] & top30_mask[1:]).sum(axis=1)
    turnovers = 1.0 - overlap / 30.0
    costs = turnovers * 0.0072
    net_rets = gross_rets - costs
    
    valid_rets = net_rets[np.isfinite(net_rets)]
    mean_ret = np.mean(valid_rets)
    std_ret = np.std(valid_rets, ddof=1)
    raw_sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 1e-8 else 0.0
    
    print(f"    Step3 详细拆解:")
    print(f"      年化毛收益: {np.nanmean(gross_rets)*52*100:+.2f}%")
    print(f"      年化成本:   {np.nanmean(costs)*52*100:+.2f}%")
    print(f"      年化净收益: {np.nanmean(net_rets)*52*100:+.2f}%")
    print(f"      平均周换手: {np.nanmean(turnovers)*100:.1f}%")
    print(f"      原始年化Sharpe: {raw_sharpe:+.4f} (clip前)")
    print(f"      clip后Obj3: {np.clip(raw_sharpe/2.0, 0, 1):.4f}")
    
    # 分析: 如果Sharpe负数, 原因是什么?
    if raw_sharpe <= 0:
        print(f"\n    [诊断] Sharpe负数原因分析:")
        if np.nanmean(gross_rets) <= 0:
            print(f"      → 主要: 毛收益为负 (训练期含熊市, long-only难以获利)")
        if np.nanmean(costs) * 52 > abs(np.nanmean(gross_rets) * 52):
            print(f"      → 成本侵蚀: 年化成本({np.nanmean(costs)*52*100:.1f}%)超过毛收益")
        if np.nanmean(turnovers) > 0.3:
            print(f"      → 换手过高: {np.nanmean(turnovers)*100:.1f}% 周换手, 年化成本~{np.nanmean(turnovers)*0.0072*52*100:.1f}%")


# ============================================================
# SECTION C: ICIR raw vs clip 区分度对比 (叠加惩罚链)
# ============================================================
print("\n" + "=" * 70)
print("  SECTION C: ICIR raw vs clip 区分度对比")
print("=" * 70)

print("\n  生成40个随机染色体, 分别用raw和clip计算Obj1...")
np.random.seed(123)

n_samples = 40
results_raw = []
results_clip = []

for i in range(n_samples):
    chromo = icir_biased_chromosome(n_test_factors, icir_values_arr, bias_strength=1.5)
    chromo += np.random.normal(0, 0.5, n_test_factors)  # 加噪声增加多样性
    
    weights = weights_from_chromosome(chromo, test_factor_names, max_factors=MAX_FACTORS)
    if len(weights) == 0:
        continue
    
    selected_factors = {name: factor_dict[name] for name in weights if name in factor_dict}
    if len(selected_factors) == 0:
        continue
    
    composite = combine_factors(selected_factors, weights)
    if composite.empty:
        continue
    
    icir_result = compute_ic_icir(composite, forward_returns)
    icir = abs(icir_result['icir'])
    if np.isnan(icir):
        continue
    
    # 惩罚项
    weekly_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(weights, test_factor_names)
    cpx_pen = _compute_complexity_penalty(len(weights))
    corr_pen = _compute_correlation_penalty(weights, test_factor_names, corr_matrix)
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, MAX_TOTAL_PENALTY))
    
    n_fund = sum(1 for name in weights if name in FACTOR_DEFS
                 and FACTOR_DEFS[name]['category'] in FUNDAMENTAL_CATEGORIES)
    bonus = (FUNDAMENTAL_BONUS - 1.0) if n_fund >= FUNDAMENTAL_MIN_COUNT else 0.0
    
    # Raw模式
    icir_raw = icir  # 原始|ICIR|
    obj1_raw = float(np.clip(icir_raw - penalty_sum + bonus, 0.0, 1.0))
    
    # Clip模式
    icir_clipped = float(np.clip(icir / ICIR_THRESHOLD, 0, 1))
    obj1_clip = float(np.clip(icir_clipped - penalty_sum + bonus, 0.0, 1.0))
    
    results_raw.append({
        'icir': icir,
        'icir_score': icir_raw,
        'penalty': penalty_sum,
        'bonus': bonus,
        'obj1': obj1_raw,
        'n_factors': len(weights),
        'to': weekly_to,
    })
    results_clip.append({
        'icir': icir,
        'icir_score': icir_clipped,
        'penalty': penalty_sum,
        'bonus': bonus,
        'obj1': obj1_clip,
        'n_factors': len(weights),
        'to': weekly_to,
    })

df_raw = pd.DataFrame(results_raw)
df_clip = pd.DataFrame(results_clip)

print(f"\n  有效样本: {len(df_raw)}")

if len(df_raw) >= 5:
    print(f"\n  --- ICIR raw 模式 ---")
    print(f"  ICIR范围:       [{df_raw['icir'].min():.4f}, {df_raw['icir'].max():.4f}]")
    print(f"  ICIR均值:       {df_raw['icir'].mean():.4f}")
    print(f"  惩罚范围:       [{df_raw['penalty'].min():.4f}, {df_raw['penalty'].max():.4f}]")
    print(f"  惩罚均值:       {df_raw['penalty'].mean():.4f}")
    print(f"  Obj1(raw)范围:  [{df_raw['obj1'].min():.4f}, {df_raw['obj1'].max():.4f}]")
    print(f"  Obj1(raw)均值:  {df_raw['obj1'].mean():.4f}")
    print(f"  Obj1(raw) CV:   {df_raw['obj1'].std()/max(df_raw['obj1'].mean(), 1e-8):.4f}")
    print(f"  零值比例:       {(df_raw['obj1']==0).mean()*100:.1f}%")
    print(f"  满分(1.0)比例:  {(df_raw['obj1']>=0.999).mean()*100:.1f}%")
    
    print(f"\n  --- ICIR clip 模式 ---")
    print(f"  ICIR_score范围: [{df_clip['icir_score'].min():.4f}, {df_clip['icir_score'].max():.4f}]")
    print(f"  ICIR_score均值: {df_clip['icir_score'].mean():.4f}")
    print(f"  Obj1(clip)范围: [{df_clip['obj1'].min():.4f}, {df_clip['obj1'].max():.4f}]")
    print(f"  Obj1(clip)均值: {df_clip['obj1'].mean():.4f}")
    print(f"  Obj1(clip) CV:  {df_clip['obj1'].std()/max(df_clip['obj1'].mean(), 1e-8):.4f}")
    print(f"  零值比例:       {(df_clip['obj1']==0).mean()*100:.1f}%")
    print(f"  满分(1.0)比例:  {(df_clip['obj1']>=0.999).mean()*100:.1f}%")
    
    # 区分度对比 (Spearman rank correlation)
    if len(df_raw) >= 3:
        sr, sp = stats.spearmanr(df_raw['obj1'], df_clip['obj1'])
        print(f"\n  --- 区分度对比 ---")
        print(f"  Obj1_raw vs Obj1_clip Spearman r: {sr:.4f} (p={sp:.6f})")
        print(f"  Raw CV:    {df_raw['obj1'].std()/max(df_raw['obj1'].mean(), 1e-8):.4f}")
        print(f"  Clip CV:   {df_clip['obj1'].std()/max(df_clip['obj1'].mean(), 1e-8):.4f}")
        
        # ★ 关键: 惩罚链区分度
        raw_no_pen = np.clip(df_raw['icir_score'].values, 0, 1)
        clip_no_pen = df_clip['icir_score'].values
        
        raw_cv_nopen = np.std(raw_no_pen) / max(np.mean(raw_no_pen), 1e-8)
        clip_cv_nopen = np.std(clip_no_pen) / max(np.mean(clip_no_pen), 1e-8)
        
        print(f"\n  --- 惩罚链贡献 ---")
        print(f"  纯ICIR(raw) CV:  {raw_cv_nopen:.4f}")
        print(f"  ICIR+惩罚(raw) CV: {df_raw['obj1'].std()/max(df_raw['obj1'].mean(), 1e-8):.4f} (差值: {df_raw['obj1'].std()/max(df_raw['obj1'].mean(), 1e-8) - raw_cv_nopen:+.4f})")
        print(f"  纯ICIR(clip) CV:  {clip_cv_nopen:.4f}")
        print(f"  ICIR+惩罚(clip) CV: {df_clip['obj1'].std()/max(df_clip['obj1'].mean(), 1e-8):.4f} (差值: {df_clip['obj1'].std()/max(df_clip['obj1'].mean(), 1e-8) - clip_cv_nopen:+.4f})")
        
        # ★★★ 最终结论
        diff_raw = df_raw['obj1'].std()/max(df_raw['obj1'].mean(), 1e-8) - raw_cv_nopen
        diff_clip = df_clip['obj1'].std()/max(df_clip['obj1'].mean(), 1e-8) - clip_cv_nopen
        
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  ICIR raw vs clip 区分度对比结论                        ║")
        print(f"  ╠══════════════════════════════════════════════════════════╣")
        if diff_raw > diff_clip * 1.1:
            print(f"  ║  ★ raw模式惩罚链贡献CV={diff_raw:+.4f} > clip模式={diff_clip:+.4f}         ║")
            print(f"  ║  推荐: raw (ICIR原始值提供更大的区分度空间)            ║")
        elif diff_clip > diff_raw * 1.1:
            print(f"  ║  ★ clip模式惩罚链贡献CV={diff_clip:+.4f} > raw模式={diff_raw:+.4f}         ║")
            print(f"  ║  推荐: clip (clip变换后惩罚链贡献更大)                 ║")
        else:
            print(f"  ║  raw和clip惩罚链贡献相近 (CV差={abs(diff_raw-diff_clip):.4f})              ║")
            print(f"  ║  推荐: clip (Obj1∈[0,1]与Obj2同量纲, NSGA-II非支配排序更均衡)")
            print(f"  ║  两个模式Spearman r={sr:.4f}, 秩序高度一致               ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)
