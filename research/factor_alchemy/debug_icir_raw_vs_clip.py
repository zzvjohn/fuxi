"""
ICIR raw vs clip 区分度对比 (v7.1)
===================================
最小化脚本, 仅做 Step 5
"""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data.loader import load_all_data
from config import FACTOR_DEFS, OOS_SPLIT_DATE, MAX_FACTORS, ICIR_THRESHOLD
from factors import ALL_FACTORS
from evaluation.ic_analysis import compute_ic_icir
from factors.composite import combine_factors, weights_from_chromosome
from ga.fitness import (
    _estimate_weekly_turnover, _compute_turnover_penalty,
    _compute_category_penalty, _compute_complexity_penalty,
    _compute_correlation_penalty, icir_to_objective,
)
from ga.chromosome import random_chromosome, icir_biased_chromosome
import config

# =====================================================
# Step 1: Load data
# =====================================================
t0 = time.time()
print("=" * 70)
print("[ICIR raw vs clip] Step 1: Loading data...")
print("=" * 70)

raw_data = load_all_data()
close = raw_data['close']
adj = raw_data['adj_factor']
daily_basic = raw_data['daily_basic']
fina_indicator = raw_data.get('fina_indicator')
print(f"  Price: {close.shape}, adj: {adj.shape}, basic: {daily_basic.shape} ({time.time()-t0:.0f}s)")

# Build weekly forward returns
from data.loader import build_weekly_forward_returns
forward_returns = build_weekly_forward_returns(close, adj)
print(f"  Forward returns: {forward_returns.shape}")
print(f"  Date range: {forward_returns.index[0]} ~ {forward_returns.index[-1]}")

# Market cap
mcap_df = None
if 'market_cap' in daily_basic.columns:
    mcap_df = daily_basic.pivot_table(
        index='trade_date', columns='ts_code', values='market_cap', aggfunc='first'
    )
    mcap_df.index = pd.to_datetime(mcap_df.index)
    mcap_df = np.log(mcap_df.clip(lower=1e8))
    print(f"  Market cap (log): {mcap_df.shape}")

# OOS split
oos_date = pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns[forward_returns.index < oos_date].copy()
oos_fr = forward_returns[forward_returns.index >= oos_date].copy()
print(f"  Train: {train_fr.shape[0]} weeks, OOS: {oos_fr.shape[0]} weeks")

# =====================================================
# Step 2: Compute factors
# =====================================================
t1 = time.time()
print("\n" + "=" * 70)
print("[ICIR raw vs clip] Step 2: Computing factors...")
print("=" * 70)

factor_dfs = {}
for name, factor in ALL_FACTORS.items():
    try:
        df = factor(close, daily_basic, fina_indicator, mcap=mcap_df)
        if isinstance(df, pd.DataFrame) and not df.empty:
            factor_dfs[name] = df
    except Exception:
        pass

print(f"  Valid factors: {len(factor_dfs)} ({time.time()-t1:.0f}s)")

# Compute IC summary
print("  Computing IC summary...")
ic_summary_data = []
for name, factor_df in factor_dfs.items():
    try:
        result = compute_ic_icir(factor_df, train_fr)
        ic_summary_data.append({
            'factor': name,
            'ICIR': result['icir'],
            'IC_mean': result.get('ic_mean', np.nan),
            'IC_std': result.get('ic_std', np.nan),
            '+IC%': result.get('pos_ic_ratio', np.nan),
        })
    except Exception:
        pass

ic_summary = pd.DataFrame(ic_summary_data).set_index('factor').sort_values('ICIR', ascending=False)
valid_factor_names = ic_summary[ic_summary['ICIR'].notna()].index.tolist()
n_factors = len(valid_factor_names)
print(f"  Valid factors (with ICIR): {n_factors}")

# =====================================================
# Step 3: ICIR raw vs clip discrimination test
# =====================================================
print("\n" + "=" * 70)
print("[ICIR raw vs clip] Step 3: Discrimination test")
print("=" * 70)

np.random.seed(42)

# Generate random portfolios
n_test = 200
test_chromosomes = []
for _ in range(n_test):
    if np.random.random() < 0.5:
        icir_list = [abs(ic_summary.loc[n, 'ICIR']) if n in ic_summary.index and
                     not np.isnan(ic_summary.loc[n, 'ICIR']) else 0.0
                     for n in valid_factor_names]
        c = icir_biased_chromosome(n_factors, np.array(icir_list), bias_strength=1.0)
    else:
        c = random_chromosome(n_factors)
    test_chromosomes.append(c)

# Precompute correlation matrix
from evaluation.correlation import factor_correlation_matrix
try:
    corr_matrix_test = factor_correlation_matrix(
        {n: factor_dfs[n] for n in valid_factor_names if n in factor_dfs}
    )
    print(f"  Correlation matrix: {corr_matrix_test.shape}")
except Exception:
    corr_matrix_test = None
    print("  [WARN] Correlation matrix failed, skipping corr penalty")

# Collect results
results_raw = []
results_clip = []
icir_values = []
penalty_values = []

print(f"\n  Evaluating {n_test} portfolios...")
for i, chromo in enumerate(test_chromosomes):
    weights = weights_from_chromosome(chromo, valid_factor_names, max_factors=MAX_FACTORS)
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
    
    icir_values.append(icir)
    
    # Compute penalties
    weekly_to = _estimate_weekly_turnover(composite, top_n=30, buffer_mult=1.8)
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(weights, valid_factor_names)
    cpx_pen = _compute_complexity_penalty(len(weights))
    corr_pen = _compute_correlation_penalty(weights, valid_factor_names, corr_matrix_test)
    
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, config.MAX_TOTAL_PENALTY))
    penalty_values.append(penalty_sum)
    
    # Bonus
    n_fund = sum(1 for name in weights if name in FACTOR_DEFS
                 and FACTOR_DEFS[name]['category'] in config.FUNDAMENTAL_CATEGORIES)
    bonus = (config.FUNDAMENTAL_BONUS - 1.0) if n_fund >= config.FUNDAMENTAL_MIN_COUNT else 0.0
    
    # Raw mode
    icir_raw = icir_to_objective(icir, mode='raw')
    obj1_raw = float(np.clip(icir_raw * (1.0 - penalty_sum) * (1.0 + bonus), 0.0, 10.0))
    
    # Clip mode
    icir_clip = icir_to_objective(icir, mode='clip', clip_threshold=ICIR_THRESHOLD)
    obj1_clip = float(np.clip(icir_clip * (1.0 - penalty_sum) * (1.0 + bonus), 0.0, 1.0))
    
    results_raw.append({
        'icir': icir, 'penalty': penalty_sum, 'bonus': bonus,
        'obj1': obj1_raw, 'n_factors': len(weights)
    })
    results_clip.append({
        'icir': icir, 'penalty': penalty_sum, 'bonus': bonus,
        'obj1': obj1_clip, 'n_factors': len(weights)
    })

n_valid = len(results_raw)
print(f"  Valid portfolios: {n_valid}/{n_test}")

# =====================================================
# Discrimination analysis
# =====================================================
print("\n" + "=" * 70)
print("[RESULTS] Discrimination Analysis")
print("=" * 70)

# ICIR distribution
icir_arr = np.array(icir_values)
print(f"\n--- ICIR Distribution ---")
print(f"  mean={icir_arr.mean():.4f}, std={icir_arr.std():.4f}, "
      f"min={icir_arr.min():.4f}, max={icir_arr.max():.4f}")
print(f"  P10={np.percentile(icir_arr,10):.4f}, P25={np.percentile(icir_arr,25):.4f}, "
      f"P50={np.percentile(icir_arr,50):.4f}, P75={np.percentile(icir_arr,75):.4f}, "
      f"P90={np.percentile(icir_arr,90):.4f}")

# Penalty distribution
pen_arr = np.array(penalty_values)
print(f"\n--- Penalty Distribution ---")
print(f"  mean={pen_arr.mean():.4f}, std={pen_arr.std():.4f}, "
      f"min={pen_arr.min():.4f}, max={pen_arr.max():.4f}")
print(f"  P10={np.percentile(pen_arr,10):.4f}, P25={np.percentile(pen_arr,25):.4f}, "
      f"P50={np.percentile(pen_arr,50):.4f}, P75={np.percentile(pen_arr,75):.4f}, "
      f"P90={np.percentile(pen_arr,90):.4f}")

# === RAW MODE ===
print(f"\n--- Obj1 (RAW mode) ---")
obj1_raw_arr = np.array([r['obj1'] for r in results_raw])
print(f"  mean={obj1_raw_arr.mean():.4f}, std={obj1_raw_arr.std():.4f}, "
      f"min={obj1_raw_arr.min():.4f}, max={obj1_raw_arr.max():.4f}")
print(f"  P10={np.percentile(obj1_raw_arr,10):.4f}, P25={np.percentile(obj1_raw_arr,25):.4f}, "
      f"P50={np.percentile(obj1_raw_arr,50):.4f}, P75={np.percentile(obj1_raw_arr,75):.4f}, "
      f"P90={np.percentile(obj1_raw_arr,90):.4f}")
raw_spread = np.percentile(obj1_raw_arr, 90) - np.percentile(obj1_raw_arr, 10)
raw_discrimination = raw_spread / max(obj1_raw_arr.max(), 1e-8)
print(f"  Spread (P90-P10): {raw_spread:.4f}")
print(f"  Normalized discrimination: {raw_discrimination:.4f}")
print(f"  Unique values: {len(np.unique(np.round(obj1_raw_arr, 6)))} / {n_valid}")

# === CLIP MODE ===
print(f"\n--- Obj1 (CLIP mode) ---")
obj1_clip_arr = np.array([r['obj1'] for r in results_clip])
print(f"  mean={obj1_clip_arr.mean():.4f}, std={obj1_clip_arr.std():.4f}, "
      f"min={obj1_clip_arr.min():.4f}, max={obj1_clip_arr.max():.4f}")
print(f"  P10={np.percentile(obj1_clip_arr,10):.4f}, P25={np.percentile(obj1_clip_arr,25):.4f}, "
      f"P50={np.percentile(obj1_clip_arr,50):.4f}, P75={np.percentile(obj1_clip_arr,75):.4f}, "
      f"P90={np.percentile(obj1_clip_arr,90):.4f}")
clip_spread = np.percentile(obj1_clip_arr, 90) - np.percentile(obj1_clip_arr, 10)
clip_discrimination = clip_spread / max(obj1_clip_arr.max(), 1e-8)
print(f"  Spread (P90-P10): {clip_spread:.4f}")
print(f"  Normalized discrimination: {clip_discrimination:.4f}")
print(f"  Unique values: {len(np.unique(np.round(obj1_clip_arr, 6)))} / {n_valid}")

# === COMPARISON ===
print(f"\n--- COMPARISON: Raw vs Clip ---")
print(f"  Raw:  spread={raw_spread:.4f}, disc={raw_discrimination:.4f}, "
      f"unique={len(np.unique(np.round(obj1_raw_arr,6)))}")
print(f"  Clip: spread={clip_spread:.4f}, disc={clip_discrimination:.4f}, "
      f"unique={len(np.unique(np.round(obj1_clip_arr,6)))}")
if raw_spread > clip_spread:
    print(f"  => RAW mode has {raw_spread/clip_spread:.1f}x better discrimination")
else:
    print(f"  => CLIP mode has {clip_spread/raw_spread:.1f}x better discrimination")

# Top 10 Raw
print(f"\n--- Top 10 RAW ---")
sorted_raw = sorted(results_raw, key=lambda x: -x['obj1'])
for i, r in enumerate(sorted_raw[:10]):
    print(f"  {i+1}. Obj1={r['obj1']:.4f}  ICIR={r['icir']:.4f}  "
          f"penalty={r['penalty']:.4f}  bonus={r['bonus']:.4f}  "
          f"n_factors={r['n_factors']}")

# Top 10 Clip
print(f"\n--- Top 10 CLIP ---")
sorted_clip = sorted(results_clip, key=lambda x: -x['obj1'])
for i, r in enumerate(sorted_clip[:10]):
    print(f"  {i+1}. Obj1={r['obj1']:.4f}  ICIR={r['icir']:.4f}  "
          f"penalty={r['penalty']:.4f}  bonus={r['bonus']:.4f}  "
          f"n_factors={r['n_factors']}")

# Spearman rank correlation
from scipy import stats
icir_rank = stats.rankdata(icir_arr)
obj1_raw_rank = stats.rankdata(obj1_raw_arr)
obj1_clip_rank = stats.rankdata(obj1_clip_arr)

raw_rho, raw_p = stats.spearmanr(icir_rank, obj1_raw_rank)
clip_rho, clip_p = stats.spearmanr(icir_rank, obj1_clip_rank)
print(f"\n--- ICIR vs Obj1 Rank Correlation ---")
print(f"  ICIR vs Obj1_raw: rho={raw_rho:.4f} (p={raw_p:.6f})")
print(f"  ICIR vs Obj1_clip: rho={clip_rho:.4f} (p={clip_p:.6f})")
print(f"  Does penalty chain change ranking in raw mode? "
      f"{'Yes' if raw_rho < 0.99 else 'No'} (rho={raw_rho:.4f})")

# Factor count vs penalty
print(f"\n--- Factor Count vs Penalty ---")
for nf in sorted(set(r['n_factors'] for r in results_raw)):
    sub_pen = [r['penalty'] for r in results_raw if r['n_factors'] == nf]
    if sub_pen:
        print(f"  n={nf}: penalty mean={np.mean(sub_pen):.4f} std={np.std(sub_pen):.4f} count={len(sub_pen)}")

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"[DONE] Total: {elapsed:.0f}s")
print(f"{'='*70}")
