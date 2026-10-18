"""
Obj3 成本后Sharpe 诊断脚本
===========================
独立验证 _compute_top30_cost_adjusted_sharpe 的计算逻辑,
对选定的代表性因子进行完整成本拆解。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Fix encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

from data.loader import load_all_data

print("=" * 70)
print("  Obj3 成本后Sharpe 诊断 (v7.2)")
print("=" * 70)

# Load data
print("\n[1] 加载数据...")
data = load_all_data('2021-01-01', '2026-06-25')
price_data = data['price_data']
close = price_data.get('close')

# Build weekly returns
close_weekly = close.resample('W').last()
close_weekly.index = pd.to_datetime(close_weekly.index)
forward_returns = close_weekly.shift(-1) / close_weekly - 1
print(f"  周频收益: {forward_returns.shape}")

# Split train/OOS
OOS_SPLIT_DATE = '2025-01-01'
fr_index = pd.DatetimeIndex(forward_returns.index)
train_mask = fr_index < pd.Timestamp(OOS_SPLIT_DATE)
train_fr = forward_returns.loc[train_mask]
print(f"  训练集: {train_fr.index[0].strftime('%Y-%m-%d')} ~ {train_fr.index[-1].strftime('%Y-%m-%d')} ({train_fr.shape[0]}周)")
oos_fr = forward_returns.loc[~train_mask]
print(f"  OOS集: {oos_fr.index[0].strftime('%Y-%m-%d')} ~ {oos_fr.index[-1].strftime('%Y-%m-%d')} ({oos_fr.shape[0]}周)")

# Compute a few test factors
from config import FACTOR_DEFS
from factors import ALL_FACTORS

print("\n[2] 计算测试因子...")
factor_dfs = {}
for name in ['plrc_12', 'dollar_vol_stability', 'turnover_std', 'earnings_growth_yoy', 'ret_1m']:
    if name not in ALL_FACTORS:
        print(f"  {name}: 未注册, 跳过")
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
            # Align with forward_returns
            common_idx = train_fr.index.intersection(df_weekly.index)
            if len(common_idx) > 0:
                df_weekly = df_weekly.reindex(common_idx)
                factor_dfs[name] = df_weekly
                print(f"  {name}: OK ({df_weekly.shape[0]}周 x {df_weekly.shape[1]}股)")
    except Exception as e:
        print(f"  {name}: FAIL ({e})")

# Import the Obj3 function
from ga.fitness import _compute_top30_cost_adjusted_sharpe

print("\n[3] Obj3 成本拆解 (训练集)...")
print(f"  {'因子':28s} {'年化总收益':>10s} {'年化成本':>10s} {'年化净收益':>10s} "
      f"{'周换手':>8s} {'年化Sharpe':>10s} {'Obj3得分':>8s}")
print(f"  {'-'*90}")

for name, df in factor_dfs.items():
    # Align with train_fr
    common_idx = df.index.intersection(train_fr.index)
    common_cols = df.columns.intersection(train_fr.columns)
    if len(common_idx) < 12 or len(common_cols) < 30:
        print(f"  {name:28s} 数据不足")
        continue
    
    comp = df.loc[common_idx, common_cols]
    fr = train_fr.loc[common_idx, common_cols]
    
    # === Manual deep dive into Obj3 ===
    ROUND_TRIP_COST = 0.0072
    comp_arr = comp.values
    fr_arr = fr.values
    n_weeks, n_stocks = comp_arr.shape
    
    # Top30 selection
    comp_filled = np.where(np.isnan(comp_arr), -np.inf, comp_arr)
    rankings = np.argsort(-comp_filled, axis=1)
    top30_idx = rankings[:, :30]
    
    # Returns
    week_indices = np.arange(n_weeks - 1)[:, None]
    stock_indices = top30_idx[:-1, :]
    gross_rets = np.nanmean(fr_arr[week_indices, stock_indices], axis=1)
    
    # Turnover
    top30_mask = np.zeros((n_weeks, n_stocks), dtype=bool)
    for i in range(n_weeks):
        top30_mask[i, top30_idx[i]] = True
    overlap = (top30_mask[:-1] & top30_mask[1:]).sum(axis=1)
    turnovers = 1.0 - overlap / 30.0
    
    # Costs
    costs = turnovers * ROUND_TRIP_COST
    net_rets = gross_rets - costs
    
    valid_rets = net_rets[np.isfinite(net_rets)]
    if len(valid_rets) >= 10:
        mean_ret = np.mean(valid_rets)
        std_ret = np.std(valid_rets, ddof=1)
        annual_sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 1e-8 else 0.0
    else:
        annual_sharpe = 0.0
    
    # Display
    annual_gross = np.nanmean(gross_rets) * 52
    annual_cost = np.nanmean(costs) * 52
    annual_net = np.nanmean(net_rets) * 52
    mean_to = np.nanmean(turnovers)
    
    # Also compute ICIR for reference
    from evaluation.ic_analysis import compute_ic_icir
    icir_val = compute_ic_icir(comp, fr)['icir']
    
    obj3_score = float(np.clip(annual_sharpe / 2.0, 0.0, 1.0))
    
    print(f"  {name:28s} {annual_gross:+10.4f} {annual_cost:10.4f} {annual_net:+10.4f} "
          f"{mean_to:7.1%} {annual_sharpe:+10.3f} {obj3_score:8.3f}  |ICIR|={abs(icir_val):.3f}")

# Check: is the direction correct?
print("\n[4] 方向验证: 多因子组合 (plrc_12 + turnover_std)...")
if 'plrc_12' in factor_dfs and 'turnover_std' in factor_dfs:
    from factors.composite import combine_factors, standardize_factors
    std_factors = standardize_factors({k: factor_dfs[k] for k in ['plrc_12', 'turnover_std']})
    weights = {'plrc_12': 0.6, 'turnover_std': 0.4}
    composite = combine_factors(std_factors, weights)
    
    common_idx = composite.index.intersection(train_fr.index)
    common_cols = composite.columns.intersection(train_fr.columns)
    comp = composite.loc[common_idx, common_cols]
    fr = train_fr.loc[common_idx, common_cols]
    
    obj3 = _compute_top30_cost_adjusted_sharpe(comp, fr)
    icir = compute_ic_icir(comp, fr)['icir']
    print(f"  组合 Obj3={obj3:.3f} |ICIR|={abs(icir):.3f}")

# Check OOS too
print("\n[5] OOS 验证 (plrc_12)...")
if 'plrc_12' in factor_dfs:
    df_plrc = factor_dfs['plrc_12']
    common_idx = df_plrc.index.intersection(oos_fr.index)
    common_cols = df_plrc.columns.intersection(oos_fr.columns)
    if len(common_idx) >= 12 and len(common_cols) >= 30:
        comp = df_plrc.loc[common_idx, common_cols]
        fr = oos_fr.loc[common_idx, common_cols]
        
        comp_arr = comp.values
        fr_arr = fr.values
        n_weeks = comp_arr.shape[0]
        
        comp_filled = np.where(np.isnan(comp_arr), -np.inf, comp_arr)
        rankings = np.argsort(-comp_filled, axis=1)
        top30_idx = rankings[:, :30]
        
        week_indices = np.arange(n_weeks - 1)[:, None]
        stock_indices = top30_idx[:-1, :]
        gross_rets = np.nanmean(fr_arr[week_indices, stock_indices], axis=1)
        
        top30_mask = np.zeros((n_weeks, comp_arr.shape[1]), dtype=bool)
        for i in range(n_weeks):
            top30_mask[i, top30_idx[i]] = True
        overlap = (top30_mask[:-1] & top30_mask[1:]).sum(axis=1)
        turnovers = 1.0 - overlap / 30.0
        
        costs = turnovers * 0.0072
        net_rets = gross_rets - costs
        
        valid_rets = net_rets[np.isfinite(net_rets)]
        annual_sharpe = np.mean(valid_rets) / np.std(valid_rets, ddof=1) * np.sqrt(52) if np.std(valid_rets) > 1e-8 else 0.0
        
        print(f"  plrc_12 OOS: 年化总收益={np.nanmean(gross_rets)*52:+.4f} "
              f"年化成本={np.nanmean(costs)*52:.4f} "
              f"年化Sharpe={annual_sharpe:+.3f}")
    else:
        print(f"  OOS数据不足")

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)
