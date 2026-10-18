"""
Step 5 standalone: ICIR raw模式 vs clip模式 惩罚链区分度对比
避免emoji崩溃, 输出到文件
"""
import sys
sys.path.insert(0, 'E:/quant')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from research.factor_alchemy.config import FACTOR_DEFS, OOS_SPLIT_DATE, MAX_FACTORS, \
    ICIR_THRESHOLD, MAX_TOTAL_PENALTY, FUNDAMENTAL_CATEGORIES, FUNDAMENTAL_BONUS, FUNDAMENTAL_MIN_COUNT
from research.factor_alchemy.factors import ALL_FACTORS
from research.factor_alchemy.evaluation.ic_analysis import compute_ic_icir, compute_ic_summary
from research.factor_alchemy.evaluation.correlation import factor_correlation_matrix
from research.factor_alchemy.factors.composite import combine_factors, weights_from_chromosome
from research.factor_alchemy.ga.fitness import (
    _estimate_weekly_turnover, _compute_turnover_penalty,
    _compute_category_penalty, _compute_complexity_penalty,
    _compute_correlation_penalty, icir_to_objective,
)
from research.factor_alchemy.ga.chromosome import random_chromosome, icir_biased_chromosome
from data.loader import load_all_data
from scipy import stats

print("[Step 5] ICIR raw模式 + 惩罚链区分度分析")
print("=" * 70)

# Load data
print("\nLoading data...")
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
close = price_data.get('close')

close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1

fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]
print(f"Train FR: {train_fr.shape}")

# Compute factors
print("Computing factors...")
factor_dfs = {}
for name in FACTOR_DEFS:
    if name not in ALL_FACTORS:
        continue
    try:
        factor_obj = ALL_FACTORS[name]()
        df = factor_obj.compute(
            price_data=price_data,
            financial_data=data.get('financial_data'),
            valuation_data=data.get('valuation_data'),
            industry_map=data.get('industry_map', {}),
        )
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df_weekly = df.resample('W').last()
            is_financial = df_weekly.shape[0] <= 10 and FACTOR_DEFS[name]['category'] in \
                ('profitability', 'value', 'growth')
            if is_financial:
                df_weekly = df_weekly.reindex(close_weekly.index).ffill().bfill()
            else:
                common_idx = close_weekly.index.intersection(df_weekly.index)
                if len(common_idx) > 0:
                    df_weekly = df_weekly.reindex(common_idx)
            factor_dfs[name] = df_weekly
    except Exception:
        pass

print(f"Valid factors: {len(factor_dfs)}")

# IC summary
ic_summary = compute_ic_summary(factor_dfs, forward_returns)
valid_names = [n for n in factor_dfs if n in ic_summary.index
               and not np.isnan(ic_summary.loc[n, 'ICIR'])]
n_factors = len(valid_names)
print(f"Factors with ICIR: {n_factors}")

# Correlation matrix
try:
    corr_matrix = factor_correlation_matrix(
        {n: factor_dfs[n] for n in valid_names if n in factor_dfs}
    )
except Exception:
    corr_matrix = None

# Generate test combinations (200 random + biased)
n_test = 200
test_chromosomes = []
for _ in range(n_test):
    if np.random.random() < 0.5:
        icir_list = [abs(ic_summary.loc[n, 'ICIR']) if n in ic_summary.index and
                     not np.isnan(ic_summary.loc[n, 'ICIR']) else 0.0
                     for n in valid_names]
        c = icir_biased_chromosome(n_factors, np.array(icir_list), bias_strength=1.0)
    else:
        c = random_chromosome(n_factors)
    test_chromosomes.append(c)

# Evaluate all in raw and clip modes
results_raw = []
results_clip = []
all_icir = []
all_penalty = []

print(f"\nEvaluating {n_test} combos (raw vs clip)...")
for chromo in test_chromosomes:
    weights = weights_from_chromosome(chromo, valid_names, max_factors=MAX_FACTORS)
    if len(weights) == 0:
        continue
    
    selected = {name: factor_dfs[name] for name in weights if name in factor_dfs}
    if len(selected) == 0:
        continue
    
    composite = combine_factors(selected, weights)
    if composite.empty:
        continue
    
    icir_result = compute_ic_icir(composite, train_fr)
    icir = abs(icir_result['icir'])
    if np.isnan(icir):
        continue
    
    all_icir.append(icir)
    
    # Penalty chain
    weekly_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(weights, valid_names)
    cpx_pen = _compute_complexity_penalty(len(weights))
    corr_pen = _compute_correlation_penalty(weights, valid_names, corr_matrix)
    
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, MAX_TOTAL_PENALTY))
    all_penalty.append(penalty_sum)
    
    # Bonus
    n_fund = sum(1 for name in weights if name in FACTOR_DEFS
                 and FACTOR_DEFS[name]['category'] in FUNDAMENTAL_CATEGORIES)
    bonus = (FUNDAMENTAL_BONUS - 1.0) if n_fund >= FUNDAMENTAL_MIN_COUNT else 0.0
    
    # Raw mode
    icir_raw = icir_to_objective(icir, mode='raw')
    obj1_raw = float(np.clip(icir_raw * (1.0 - penalty_sum) * (1.0 + bonus), 0.0, 10.0))
    
    # Clip mode
    icir_clip = icir_to_objective(icir, mode='clip', clip_threshold=ICIR_THRESHOLD)
    obj1_clip = float(np.clip(icir_clip * (1.0 - penalty_sum) * (1.0 + bonus), 0.0, 1.0))
    
    results_raw.append({
        'icir': icir, 'penalty': penalty_sum, 'bonus': bonus,
        'obj1': obj1_raw, 'n_factors': len(weights),
        'to_pen': to_pen, 'cat_pen': cat_pen, 'cpx_pen': cpx_pen, 'corr_pen': corr_pen
    })
    results_clip.append({
        'icir': icir, 'penalty': penalty_sum, 'bonus': bonus,
        'obj1': obj1_clip, 'n_factors': len(weights)
    })

n_valid = len(results_raw)
print(f"Valid solutions: {n_valid}")

# ===== ICIR distribution =====
icir_arr = np.array(all_icir)
print(f"\n{'='*60}")
print(f"ICIR Distribution")
print(f"{'='*60}")
print(f"  mean={icir_arr.mean():.4f}  std={icir_arr.std():.4f}  "
      f"min={icir_arr.min():.4f}  max={icir_arr.max():.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(icir_arr, p):.4f}")

# ===== Penalty distribution =====
pen_arr = np.array(all_penalty)
print(f"\n{'='*60}")
print(f"Penalty Distribution (addition chain)")
print(f"{'='*60}")
print(f"  mean={pen_arr.mean():.4f}  std={pen_arr.std():.4f}  "
      f"min={pen_arr.min():.4f}  max={pen_arr.max():.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(pen_arr, p):.4f}")

# Penalty breakdown
to_pens = np.array([r['to_pen'] for r in results_raw])
cat_pens = np.array([r['cat_pen'] for r in results_raw])
cpx_pens = np.array([r['cpx_pen'] for r in results_raw])
corr_pens = np.array([r['corr_pen'] for r in results_raw])
print(f"\n  Penalty Breakdown (mean):")
print(f"    Turnover:  {to_pens.mean():.4f}")
print(f"    Category:  {cat_pens.mean():.4f}")
print(f"    Complexity:{cpx_pens.mean():.4f}")
print(f"    Correlation:{corr_pens.mean():.4f}")

# ===== Obj1 Raw =====
obj1_raw_arr = np.array([r['obj1'] for r in results_raw])
print(f"\n{'='*60}")
print(f"Obj1 (RAW mode) Discrimination")
print(f"{'='*60}")
print(f"  mean={obj1_raw_arr.mean():.4f}  std={obj1_raw_arr.std():.4f}  "
      f"min={obj1_raw_arr.min():.4f}  max={obj1_raw_arr.max():.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(obj1_raw_arr, p):.4f}")

raw_p90 = np.percentile(obj1_raw_arr, 90)
raw_p10 = np.percentile(obj1_raw_arr, 10)
raw_spread = raw_p90 - raw_p10
raw_disc = raw_spread / max(obj1_raw_arr.max(), 1e-8)
print(f"  P90-P10 spread: {raw_spread:.4f}")
print(f"  Normalized disc: {raw_disc:.4f}")
print(f"  COV: {obj1_raw_arr.std()/obj1_raw_arr.mean():.4f}")

# ===== Obj1 Clip =====
obj1_clip_arr = np.array([r['obj1'] for r in results_clip])
print(f"\n{'='*60}")
print(f"Obj1 (CLIP mode) Discrimination")
print(f"{'='*60}")
print(f"  mean={obj1_clip_arr.mean():.4f}  std={obj1_clip_arr.std():.4f}  "
      f"min={obj1_clip_arr.min():.4f}  max={obj1_clip_arr.max():.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(obj1_clip_arr, p):.4f}")

clip_p90 = np.percentile(obj1_clip_arr, 90)
clip_p10 = np.percentile(obj1_clip_arr, 10)
clip_spread = clip_p90 - clip_p10
clip_disc = clip_spread / max(obj1_clip_arr.max(), 1e-8)
print(f"  P90-P10 spread: {clip_spread:.4f}")
print(f"  Normalized disc: {clip_disc:.4f}")
print(f"  COV: {obj1_clip_arr.std()/obj1_clip_arr.mean():.4f}")

# ===== Rank correlation =====
icir_rank = stats.rankdata(icir_arr)
obj1_raw_rank = stats.rankdata(obj1_raw_arr)
obj1_clip_rank = stats.rankdata(obj1_clip_arr)

raw_rho, raw_p = stats.spearmanr(icir_rank, obj1_raw_rank)
clip_rho, clip_p = stats.spearmanr(icir_rank, obj1_clip_rank)
penalty_rank = stats.rankdata(pen_arr)
pen_rho, pen_p = stats.spearmanr(icir_rank, penalty_rank)

print(f"\n{'='*60}")
print(f"Rank Consistency Analysis")
print(f"{'='*60}")
print(f"  ICIR vs Obj1_raw rank corr: rho={raw_rho:.4f} (p={raw_p:.6f})")
print(f"  ICIR vs Obj1_clip rank corr: rho={clip_rho:.4f} (p={clip_p:.6f})")
print(f"  ICIR vs Penalty rank corr:   rho={pen_rho:.4f} (p={pen_p:.6f})")
print(f"  Penalty alters ranking: {'YES' if raw_rho < 0.95 else 'NO'}")

# ===== Top 10 RAW =====
print(f"\n{'='*60}")
print(f"Top 10 RAW mode (highest Obj1)")
print(f"{'='*60}")
print(f"  {'Rank':<5s} {'Obj1':>8s} {'ICIR':>8s} {'Penalty':>8s} {'Bonus':>6s} {'N':>3s}")
sorted_raw = sorted(results_raw, key=lambda x: -x['obj1'])
for i, r in enumerate(sorted_raw[:10]):
    print(f"  {i+1:<5d} {r['obj1']:>8.4f} {r['icir']:>8.4f} {r['penalty']:>8.4f} {r['bonus']:>6.4f} {r['n_factors']:>3d}")

# ===== Top 10 CLIP =====
print(f"\n{'='*60}")
print(f"Top 10 CLIP mode (highest Obj1)")
print(f"{'='*60}")
sorted_clip = sorted(results_clip, key=lambda x: -x['obj1'])
for i, r in enumerate(sorted_clip[:10]):
    print(f"  {i+1:<5d} {r['obj1']:>8.4f} {r['icir']:>8.4f} {r['penalty']:>8.4f} {r['bonus']:>6.4f} {r['n_factors']:>3d}")

# ===== Penalty by n_factors =====
print(f"\n{'='*60}")
print(f"Penalty by N Factors")
print(f"{'='*60}")
for nf in sorted(set(r['n_factors'] for r in results_raw)):
    sub_pen = [r['penalty'] for r in results_raw if r['n_factors'] == nf]
    sub_icir = [r['icir'] for r in results_raw if r['n_factors'] == nf]
    if sub_pen:
        print(f"  n_factors={nf}: penalty mean={np.mean(sub_pen):.4f} std={np.std(sub_pen):.4f} "
              f"icir mean={np.mean(sub_icir):.4f} count={len(sub_pen)}")

# ===== FINAL VERDICT =====
print(f"\n{'='*60}")
print(f"VERDICT: ICIR raw + penalty chain discrimination")
print(f"{'='*60}")
print(f"")
print(f"  Raw mode:")
print(f"    Spread (P90-P10): {raw_spread:.4f}")
print(f"    Normalized disc:  {raw_disc:.4f}")
print(f"    COV:              {obj1_raw_arr.std()/obj1_raw_arr.mean():.4f}")
print(f"")
print(f"  Clip mode:")
print(f"    Spread (P90-P10): {clip_spread:.4f}")
print(f"    Normalized disc:  {clip_disc:.4f}")
print(f"    COV:              {obj1_clip_arr.std()/obj1_clip_arr.mean():.4f}")
print(f"")
print(f"  Penalty std: {pen_arr.std():.4f} (needs > 0 to create spread)")
print(f"  ICIR-Obj1_raw rank rho: {raw_rho:.4f} (1.0 = penalty didn't change ranking)")
print(f"")
if raw_spread > clip_spread * 0.8 and pen_arr.std() > 0.01:
    print(f"  [PASS] Raw mode + penalty chain has sufficient discrimination.")
    print(f"         Penalty chain creates meaningful spread between solutions.")
elif raw_rho >= 0.99:
    print(f"  [WARN] Penalty chain barely alters ICIR ranking.")
    print(f"         Consider increasing penalty scale or adding more penalty dimensions.")
else:
    print(f"  [NEUTRAL] Raw mode discrimination is adequate.")
    if pen_arr.std() < 0.01:
        print(f"            But penalty spread is low — may not differentiate enough.")
