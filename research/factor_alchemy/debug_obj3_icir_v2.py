"""
ICIR Raw vs Clip 多因子区分度测试 (精简版)
跳过已完成的 Obj3 诊断
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import FACTOR_DEFS, OOS_SPLIT_DATE, MAX_FACTORS, MAX_TOTAL_PENALTY

print("=" * 70)
print("  ICIR Raw vs Clip 多因子区分度测试")
print(f"  MAX_FACTORS={MAX_FACTORS} | OOS={OOS_SPLIT_DATE}")
print("=" * 70)

# ===== Load data =====
print("\n[1] Loading data...")
from data.loader import load_all_data
data = load_all_data('2021-01-01', '2026-06-25')

price_data = data['price_data']
close = price_data.get('close')
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1

# ===== Compute factors =====
print("[2] Computing factors...")
t0 = time.time()
from factors import ALL_FACTORS
factor_dfs = {}
for name in FACTOR_DEFS:
    if name not in ALL_FACTORS:
        continue
    factor_obj = ALL_FACTORS[name]()
    try:
        df = factor_obj.compute(price_data=price_data, financial_data=data['financial_data'],
                                valuation_data=data['valuation_data'])
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
print(f"  {len(factor_dfs)} factors ({time.time()-t0:.0f}s)")

# ===== IC/ICIR =====
print("[3] IC/ICIR...")
from evaluation.ic_analysis import compute_ic_summary
ic_summary = compute_ic_summary(factor_dfs, forward_returns)

fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]

from factors.composite import standardize_factors, combine_factors
factor_dfs_std = standardize_factors(factor_dfs)

valid_names = [n for n in factor_dfs if n in ic_summary.index 
               and not np.isnan(ic_summary.loc[n, 'ICIR'])]
top_names = sorted(valid_names, key=lambda n: -abs(ic_summary.loc[n, 'ICIR']))[:20]

# Correlation matrix
from evaluation.correlation import factor_correlation_matrix
try:
    corr_matrix_full = factor_correlation_matrix({n: factor_dfs[n] for n in valid_names if n in factor_dfs})
except:
    corr_matrix_full = pd.DataFrame(np.eye(len(valid_names)), index=valid_names, columns=valid_names)

from evaluation.ic_analysis import compute_ic_icir
from ga.fitness import (_estimate_weekly_turnover, _compute_turnover_penalty,
                         _compute_category_penalty, _compute_complexity_penalty,
                         _compute_correlation_penalty)
from scipy.stats import spearmanr

# ===== Test 1: Top N incremental =====
print("\n" + "=" * 70)
print("[Test 1] 递增因子数 (Top ICIR)")
print("=" * 70)
print(f"  {'N':>3s} {'ICIR_train':>10s} {'TO':>7s} {'P_TO':>6s} {'P_cat':>6s} "
      f"{'P_cpx':>6s} {'P_corr':>6s} {'P_sum':>7s} {'Q_clip':>8s} {'Q_raw':>8s}")

inc_results = []
for n in range(1, min(MAX_FACTORS + 3, len(top_names))):
    sel_names = top_names[:n]
    sel = {n: factor_dfs_std[n] for n in sel_names if n in factor_dfs_std}
    if len(sel) < 1:
        continue
    w = {n: 1.0/len(sel) for n in sel}
    comp = combine_factors(sel, w)
    
    icir_r = compute_ic_icir(comp, train_fr)
    icir_train = abs(icir_r['icir'])
    weekly_to = _estimate_weekly_turnover(comp, top_n=30, buffer_mult=1.8)
    
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(w, list(sel.keys()))
    cpx_pen = _compute_complexity_penalty(len(w))
    sub_names = [n for n in sel if n in corr_matrix_full.index]
    sub_corr = corr_matrix_full.loc[sub_names, sub_names] if len(sub_names) > 1 else None
    corr_pen = _compute_correlation_penalty(w, list(sel.keys()), sub_corr) if sub_corr is not None else 0.0
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, MAX_TOTAL_PENALTY))
    
    q_clip = np.clip(np.clip(icir_train/1.5, 0, 1) * (1.0 - penalty_sum), 0, 1.0)
    q_raw = np.clip(icir_train * (1.0 - penalty_sum), 0, 10.0)
    
    print(f"  {len(w):3d} {icir_train:10.4f} {weekly_to:6.1%} {to_pen:6.4f} {cat_pen:6.4f} "
          f"{cpx_pen:6.4f} {corr_pen:6.4f} {penalty_sum:7.4f} {q_clip:8.4f} {q_raw:8.4f}")
    
    inc_results.append({
        'n': len(w), 'icir_train': icir_train, 'q_clip': q_clip, 'q_raw': q_raw,
        'penalty': penalty_sum,
    })

# ===== Test 2: Random multi-factor =====
print(f"\n[Test 2] 随机多因子组合 (30 samples)")
print(f"  {'样本':>4s} {'N':>3s} {'ICIR':>8s} {'TO':>7s} {'Penalty':>8s} {'Q_clip':>8s} {'Q_raw':>8s}")

np.random.seed(42)
rand_results = []
for i in range(30):
    n_f = np.random.randint(1, min(MAX_FACTORS + 2, 7))
    sel_names = list(np.random.choice(valid_names, n_f, replace=False))
    sel = {n: factor_dfs_std[n] for n in sel_names if n in factor_dfs_std}
    if len(sel) < 1:
        continue
    w = {n: 1.0/len(sel) for n in sel}
    comp = combine_factors(sel, w)
    
    icir_r = compute_ic_icir(comp, train_fr)
    icir_train = abs(icir_r['icir'])
    weekly_to = _estimate_weekly_turnover(comp, top_n=30, buffer_mult=1.8)
    
    to_pen = _compute_turnover_penalty(weekly_to)
    cat_pen = _compute_category_penalty(w, list(sel.keys()))
    cpx_pen = _compute_complexity_penalty(len(w))
    sub_names = [n for n in sel if n in corr_matrix_full.index]
    sub_corr = corr_matrix_full.loc[sub_names, sub_names] if len(sub_names) > 1 else None
    corr_pen = _compute_correlation_penalty(w, list(sel.keys()), sub_corr) if sub_corr is not None else 0.0
    penalty_sum = float(np.clip(to_pen + cat_pen + cpx_pen + corr_pen, 0.0, MAX_TOTAL_PENALTY))
    
    q_clip = np.clip(np.clip(icir_train/1.5, 0, 1) * (1.0 - penalty_sum), 0, 1.0)
    q_raw = np.clip(icir_train * (1.0 - penalty_sum), 0, 10.0)
    
    print(f"  {i:4d} {len(w):3d} {icir_train:8.4f} {weekly_to:6.1%} {penalty_sum:8.4f} {q_clip:8.4f} {q_raw:8.4f}")
    
    rand_results.append({
        'icir_train': icir_train, 'q_clip': q_clip, 'q_raw': q_raw,
        'penalty': penalty_sum, 'n': len(w),
    })

# ===== Analysis =====
print("\n" + "=" * 70)
print("  区分度分析")
print("=" * 70)

for label, data in [("递增因子", inc_results), ("随机组合", rand_results)]:
    if len(data) < 2:
        continue
    qc = np.array([r['q_clip'] for r in data])
    qr = np.array([r['q_raw'] for r in data])
    ic = np.array([r['icir_train'] for r in data])
    pn = np.array([r['penalty'] for r in data])
    
    cv_clip = qc.std() / max(qc.mean(), 1e-8)
    cv_raw = qr.std() / max(qr.mean(), 1e-8)
    
    print(f"\n  [{label}] n={len(data)}")
    print(f"  {'':20s} {'Clip':>12s} {'Raw':>12s} {'Diff':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'Q范围':20s} [{qc.min():.4f},{qc.max():.4f}]  [{qr.min():.4f},{qr.max():.4f}]")
    print(f"  {'Q均值':20s} {qc.mean():12.4f} {qr.mean():12.4f} {(qr.mean()-qc.mean()):+12.4f}")
    print(f"  {'Q变异系数(CV)':20s} {cv_clip:12.4f} {cv_raw:12.4f} {(cv_raw-cv_clip):+12.4f}")
    
    if len(qc) >= 5:
        top3 = np.sort(qc)[-3:].mean() / max(np.sort(qc)[:3].mean(), 1e-8)
        top3r = np.sort(qr)[-3:].mean() / max(np.sort(qr)[:3].mean(), 1e-8)
        print(f"  {'Top3/Bottom3':20s} {top3:12.2f} {top3r:12.2f}")
    
    clip_rho, _ = spearmanr(ic, qc)
    raw_rho, _ = spearmanr(ic, qr)
    print(f"  {'Q vs ICIR rho':20s} {clip_rho:12.3f} {raw_rho:12.3f}")
    
    # Check: does penalty chain improve discrimination?
    # Compare: raw with penalty vs raw without penalty (just ICIR)
    raw_no_pen = ic  # raw mode without penalty = just ICIR
    cv_raw_no_pen = raw_no_pen.std() / max(raw_no_pen.mean(), 1e-8)
    cv_raw_with_pen = cv_raw
    print(f"  {'CV(no_pen) vs CV':20s} {cv_raw_no_pen:12.4f} {cv_raw_with_pen:12.4f} {(cv_raw_with_pen-cv_raw_no_pen):+12.4f}")
    
    penalized_idx = pn > 0.01
    if penalized_idx.sum() >= 3:
        qc_pen = qc[penalized_idx]
        qc_nopen = qc[~penalized_idx]
        qr_pen = qr[penalized_idx]
        qr_nopen = qr[~penalized_idx]
        print(f"  {'Q(pen>0) vs Q(pen=0)':20s} {qc_pen.mean():.4f}/{qc_nopen.mean():.4f}  {qr_pen.mean():.4f}/{qr_nopen.mean():.4f}")

# ===== Final summary =====
print("\n" + "=" * 70)
print("  === 最终结论 ===")
print("=" * 70)

if inc_results and rand_results:
    qc_all = np.array([r['q_clip'] for r in inc_results + rand_results])
    qr_all = np.array([r['q_raw'] for r in inc_results + rand_results])
    cv_c = qc_all.std() / max(qc_all.mean(), 1e-8)
    cv_r = qr_all.std() / max(qr_all.mean(), 1e-8)
    
    print(f"")
    print(f"  ICIR discrimination (all tests):")
    print(f"    Clip CV = {cv_c:.4f}")
    print(f"    Raw  CV = {cv_r:.4f}")
    print(f"    Delta  = {cv_r - cv_c:+.4f}")
    print(f"")
    if abs(cv_r - cv_c) < 0.01:
        print(f"  >> 区分度基本无差异")
        print(f"  >> 原因: raw/clip是单调变换, 惩罚链乘性叠加不改变排序")
    elif cv_r > cv_c:
        print(f"  >> Raw模式区分度略高 ({cv_r:.3f} vs {cv_c:.3f})")
    else:
        print(f"  >> Clip模式区分度略高 ({cv_c:.3f} vs {cv_r:.3f})")
    
    print(f"")
    print(f"  建议:")
    print(f"    1. 如果追求NSGA-II目标尺度一致(Obj1/2/3同量纲): 用clip")
    print(f"    2. 如果追求ICIR原始信号强度保留: 用raw (但需注意Obj2[0,1]尺度不匹配)")
    print(f"    3. 对惩罚链区分度: 关键在MAX_TOTAL_PENALTY和各项惩罚乘数,")
    print(f"       raw/clip选择对区分度影响可忽略")

print(f"\nDone.")
