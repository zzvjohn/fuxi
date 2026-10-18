"""
Rank IC / ICIR 分析
====================
计算因子的截面 Rank IC 和 ICIR (Information Coefficient / IC-IR Ratio)

v4 (2026-06-24): 添加 compute_ic_icir_fast — numpy向量化Spearman,
    避免 scipy.stats.spearmanr 的逐日开销, Phase 3.5 Beam Search 加速 5-10x.
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_ic_icir(factor_df, forward_returns, min_samples=30):
    """
    计算截面 Rank IC 和 ICIR
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子值, index=date, columns=stocks
    forward_returns : pd.DataFrame
        前向收益 (从当期到下一调仓日)
    min_samples : int
        每次IC计算的最低股票数
    
    Returns
    -------
    dict
        {
            'ic_series': pd.Series (每个截面的IC),
            'ic_mean': float,
            'ic_std': float,
            'icir': float,
            'ic_positive_ratio': float (IC>0的比例),
            'ic_t_stat': float,
            'ic_p_value': float,
        }
    """
    # 对齐日期
    common_dates = sorted(set(factor_df.index) & set(forward_returns.index))
    
    ic_list = []
    ic_dates = []
    
    for date in common_dates:
        f_row = factor_df.loc[date].dropna()
        r_row = forward_returns.loc[date].dropna()
        
        common_stocks = list(set(f_row.index) & set(r_row.index))
        if len(common_stocks) < min_samples:
            continue
        
        f_vals = f_row[common_stocks]
        r_vals = r_row[common_stocks]
        
        # 排名IC (Spearman rank correlation)
        try:
            ic, pval = stats.spearmanr(f_vals, r_vals)
            if not np.isnan(ic) and np.isfinite(ic):
                ic_list.append(ic)
                ic_dates.append(date)
        except:
            continue
    
    if len(ic_list) < 5:
        return {
            'ic_series': pd.Series(dtype=float),
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan,
            'ic_t_stat': np.nan,
            'ic_p_value': np.nan,
        }
    
    ic_series = pd.Series(ic_list, index=ic_dates)
    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    pos_ratio = np.mean([1 if ic > 0 else 0 for ic in ic_list])
    
    # t检验: H0: IC=0
    t_stat, p_value = stats.ttest_1samp(ic_list, 0)
    
    return {
        'ic_series': ic_series,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': pos_ratio,
        'ic_t_stat': t_stat,
        'ic_p_value': p_value,
    }


def compute_ic_icir_fast(factor_df, forward_returns, min_samples=30):
    """
    快速截面 Rank IC 和 ICIR — numpy向量化版本
    
    v4: 用 numpy rankdata + Spearman公式替代 scipy.stats.spearmanr,
        避免 p-value计算开销。适合 Phase 3.5 Beam Search 等高频调用场景。
    
    Spearman公式: r_s = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))
    其中 d_i = rank(x_i) - rank(y_i)
    
    Parameters 同 compute_ic_icir
    """
    common_dates = factor_df.index.intersection(forward_returns.index)
    
    ic_list = []
    ic_dates = []
    
    for date in common_dates:
        f_row = factor_df.loc[date].dropna()
        r_row = forward_returns.loc[date].dropna()
        
        common_stocks = f_row.index.intersection(r_row.index)
        n = len(common_stocks)
        if n < min_samples:
            continue
        
        f_vals = f_row[common_stocks].values
        r_vals = r_row[common_stocks].values
        
        try:
            # numpy rankdata: 比 scipy.spearmanr 快 5-10x (不计算p-value)
            f_rank = stats.rankdata(f_vals)
            r_rank = stats.rankdata(r_vals)
            
            # Spearman 公式
            d2 = np.sum((f_rank - r_rank) ** 2)
            ic = 1.0 - 6.0 * d2 / (n * (n * n - 1))
            
            if np.isfinite(ic):
                ic_list.append(ic)
                ic_dates.append(date)
        except Exception:
            continue
    
    if len(ic_list) < 5:
        return {
            'ic_series': pd.Series(dtype=float),
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan,
            'ic_t_stat': np.nan,
            'ic_p_value': np.nan,
        }
    
    ic_series = pd.Series(ic_list, index=ic_dates)
    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    pos_ratio = np.mean([1 if ic > 0 else 0 for ic in ic_list])
    t_stat, p_value = stats.ttest_1samp(ic_list, 0)
    
    return {
        'ic_series': ic_series,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': pos_ratio,
        'ic_t_stat': t_stat,
        'ic_p_value': p_value,
    }


def compute_ic_summary(factor_dict, forward_returns, min_samples=30):
    """
    批量计算多个因子的 IC/ICIR
    
    Parameters
    ----------
    factor_dict : dict
        {factor_name: pd.DataFrame}
    forward_returns : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        行=因子, 列=ICIR/IC_mean/IC_std/+IC%/t-stat/p-value
    """
    results = []
    for name, df in factor_dict.items():
        ic_result = compute_ic_icir_fast(df, forward_returns, min_samples)
        results.append({
            'factor': name,
            'ICIR': ic_result['icir'],
            'IC_mean': ic_result['ic_mean'],
            'IC_std': ic_result['ic_std'],
            '+IC%': ic_result['ic_positive_ratio'],
            't_stat': ic_result['ic_t_stat'],
            'p_value': ic_result['ic_p_value'],
            'n_periods': len(ic_result['ic_series']),
        })
    
    return pd.DataFrame(results).set_index('factor').sort_values('ICIR', key=abs, ascending=False)
