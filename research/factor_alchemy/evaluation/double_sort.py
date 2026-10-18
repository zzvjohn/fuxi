"""
独立双重排序法 (Independent Double Sorting)
=============================================

非规模因子:
  市值5组 × 目标因子5组 = 25个投资组合
  计算因子预期收益率 = 高因子组(跨市值)均值 - 低因子组(跨市值)均值
  t检验 H0: factor_return <= 0, H1: factor_return > 0 (单尾)

规模因子:
  仅市值单变量排序, 10分位 → 10个投资组合
  预期收益率 = 最小市值组 - 最大市值组
  t检验
"""
import numpy as np
import pandas as pd
from scipy import stats


def independent_double_sort(factor_df, forward_returns, mcap_df,
                             n_size=5, n_factor=5):
    """
    独立双重排序
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        目标因子值, index=date, columns=stocks
    forward_returns : pd.DataFrame
        前向收益
    mcap_df : pd.DataFrame
        市值数据, index=date, columns=stocks
    n_size : int
        市值分组数 (默认5)
    n_factor : int
        因子分组数 (默认5)
    
    Returns
    -------
    dict
        {
            'portfolio_returns': pd.DataFrame,  # 25组合 × 时间序列
            'factor_expected_return': float,    # 因子预期收益率
            't_stat': float,                   # t统计量 (单尾)
            'p_value': float,                  # p值 (单尾)
            'passed': bool,                    # 是否通过检验
            'factor_quintile_returns': dict,   # 5个因子分位组的平均收益
            'factor_quintile_spread': float,   # Top - Bottom因子分组收益
        }
    """
    # 对齐三个 DataFrame
    common_dates = sorted(set(factor_df.index) & set(forward_returns.index) & set(mcap_df.index))
    
    if len(common_dates) < 20:
        return _empty_double_sort_result()
    
    # Step 1: 每个截面独立双重排序
    factor_quintile_rets = {i: [] for i in range(n_factor)}
    
    for date in common_dates:
        f_row = factor_df.loc[date].dropna()
        r_row = forward_returns.loc[date].dropna()
        m_row = mcap_df.loc[date].dropna()
        
        common_stocks = list(set(f_row.index) & set(r_row.index) & set(m_row.index))
        if len(common_stocks) < n_size * n_factor * 3:
            continue
        
        f_vals = f_row[common_stocks].values
        r_vals = r_row[common_stocks].values
        m_vals = m_row[common_stocks].values
        
        n_stocks = len(common_stocks)
        
        # 市值排序 → 5组
        m_ranks = np.argsort(np.argsort(m_vals))
        m_group_size = n_stocks // n_size
        
        # 因子排序 → 5组
        f_ranks = np.argsort(np.argsort(f_vals))
        f_group_size = n_stocks // n_factor
        
        # 对每个市值组, 因子独立排序
        # 先分市值组
        for s in range(n_size):
            s_start = s * m_group_size
            s_end = (s + 1) * m_group_size if s < n_size - 1 else n_stocks
            s_mask = (m_ranks >= s_start) & (m_ranks < s_end)
            
            s_f_vals = f_vals[s_mask]
            s_r_vals = r_vals[s_mask]
            
            if len(s_f_vals) < n_factor:
                continue
            
            # 在该市值组内对因子排序 → 5组
            s_f_ranks = np.argsort(np.argsort(s_f_vals))
            s_n = len(s_f_ranks)
            s_f_size = s_n // n_factor
            
            for f_idx in range(n_factor):
                f_start = f_idx * s_f_size
                f_end = (f_idx + 1) * s_f_size if f_idx < n_factor - 1 else s_n
                f_mask = (s_f_ranks >= f_start) & (s_f_ranks < f_end)
                
                if f_mask.sum() > 0:
                    port_ret = np.mean(s_r_vals[f_mask])
                    factor_quintile_rets[f_idx].append(port_ret)
    
    # 检查数据
    n_obs = min(len(v) for v in factor_quintile_rets.values())
    if n_obs < 10:
        return _empty_double_sort_result()
    
    # 对齐长度
    for k in factor_quintile_rets:
        factor_quintile_rets[k] = factor_quintile_rets[k][:n_obs]
    
    # 计算每个因子分位组的平均收益 (跨市值组平均)
    quintile_means = {}
    for i in range(n_factor):
        quintile_means[i] = np.mean(factor_quintile_rets[i])
    
    # 因子预期收益率 = Top因子组 - Bottom因子组
    top_rets = np.array(factor_quintile_rets[n_factor - 1])  # 最高因子分组
    bot_rets = np.array(factor_quintile_rets[0])             # 最低因子分组
    
    spread_rets = top_rets - bot_rets
    factor_expected_return = np.mean(spread_rets)
    
    # t检验: H0: factor_return <= 0, H1: factor_return > 0 (单尾)
    if len(spread_rets) > 5:
        t_stat, p_val_two_tail = stats.ttest_1samp(spread_rets, 0)
        # 转换为单尾 (H1: > 0)
        if t_stat > 0:
            p_val = p_val_two_tail / 2
        else:
            p_val = 1 - p_val_two_tail / 2
        
        passed = (p_val < 0.05) and (factor_expected_return > 0)
    else:
        t_stat, p_val = np.nan, np.nan
        passed = False
    
    return {
        'factor_quintile_returns': quintile_means,
        'factor_quintile_spread': factor_expected_return,
        't_stat': t_stat,
        'p_value': p_val,
        'passed': passed,
        'spread_series': pd.Series(spread_rets),
    }


def size_single_sort(mcap_df, forward_returns, n_groups=10):
    """
    规模因子单变量排序
    
    Parameters
    ----------
    mcap_df : pd.DataFrame
        市值, index=date, columns=stocks
    forward_returns : pd.DataFrame
        前向收益
    n_groups : int
        分组数 (默认10)
    
    Returns
    -------
    dict
        {
            'group_returns': dict,    # {group: mean_return}
            'size_premium': float,    # 最小 - 最大
            't_stat': float,
            'p_value': float,
            'passed': bool,
        }
    """
    common_dates = sorted(set(mcap_df.index) & set(forward_returns.index))
    
    group_rets = {i: [] for i in range(n_groups)}
    
    for date in common_dates:
        m_row = mcap_df.loc[date].dropna()
        r_row = forward_returns.loc[date].dropna()
        
        common_stocks = list(set(m_row.index) & set(r_row.index))
        if len(common_stocks) < n_groups * 3:
            continue
        
        m_vals = m_row[common_stocks].values
        r_vals = r_row[common_stocks].values
        
        # 按市值从小到大排序
        sort_idx = np.argsort(m_vals)
        r_sorted = r_vals[sort_idx]
        n = len(sort_idx)
        group_size = n // n_groups
        
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else n
            group_rets[g].append(np.mean(r_sorted[start:end]))
    
    # 各组的平均收益
    group_means = {g: np.mean(rets) for g, rets in group_rets.items() if len(rets) > 0}
    
    # 规模溢价 = 最小 - 最大
    smallest = np.array(group_rets[0])
    largest = np.array(group_rets[n_groups - 1])
    min_len = min(len(smallest), len(largest))
    spread = smallest[:min_len] - largest[:min_len]
    
    size_premium = np.mean(spread)
    
    if len(spread) > 5:
        t_stat, p_val_two_tail = stats.ttest_1samp(spread, 0)
        if t_stat > 0:
            p_val = p_val_two_tail / 2
        else:
            p_val = 1 - p_val_two_tail / 2
        passed = (p_val < 0.05) and (size_premium > 0)
    else:
        t_stat, p_val = np.nan, np.nan
        passed = False
    
    return {
        'group_returns': group_means,
        'size_premium': size_premium,
        't_stat': t_stat,
        'p_value': p_val,
        'passed': passed,
    }


def _empty_double_sort_result():
    return {
        'factor_quintile_returns': {},
        'factor_quintile_spread': np.nan,
        't_stat': np.nan,
        'p_value': np.nan,
        'passed': False,
        'spread_series': pd.Series(dtype=float),
    }
