"""
十分位分组检验 (Decile Portfolio Test)
========================================
单因子排序 → 分成10组 → 等权计算每组收益 → 检验单调性
"""
import numpy as np
import pandas as pd
from scipy import stats


def decile_portfolio_test(factor_df, forward_returns, n_groups=10):
    """
    十分位分组检验
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子值, index=date, columns=stocks
    forward_returns : pd.DataFrame
        前向收益
    n_groups : int
        分组数 (默认10)
    
    Returns
    -------
    dict
        {
            'group_returns': pd.DataFrame (日期 × 分组 的收益矩阵),
            'cum_returns': pd.DataFrame (分组累计收益),
            'spread_return': pd.Series (Top-Bottom 收益序列),
            'monotonicity_score': float,
            'top_bottom_t_stat': float,
            'top_bottom_p_value': float,
        }
    """
    common_dates = sorted(set(factor_df.index) & set(forward_returns.index))
    
    group_returns = {i: [] for i in range(n_groups)}
    spread_returns = []
    spread_dates = []
    
    for date in common_dates:
        f_row = factor_df.loc[date].dropna()
        r_row = forward_returns.loc[date].dropna()
        
        common_stocks = list(set(f_row.index) & set(r_row.index))
        if len(common_stocks) < n_groups * 3:
            continue
        
        f_vals = f_row[common_stocks].values
        r_vals = r_row[common_stocks].values
        
        # 排序分组
        ranks = np.argsort(np.argsort(f_vals))
        n = len(ranks)
        group_size = n // n_groups
        
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else n
            group_ret = np.mean(r_vals[ranks[start:end]])
            group_returns[g].append(group_ret)
        
        # Top - Bottom
        top_idx = ranks[-group_size:]
        bot_idx = ranks[:group_size]
        spread = np.mean(r_vals[top_idx]) - np.mean(r_vals[bot_idx])
        spread_returns.append(spread)
        spread_dates.append(date)
    
    # 构建 DataFrame
    max_len = max(len(v) for v in group_returns.values())
    group_df = pd.DataFrame(index=common_dates[:max_len])
    for g in range(n_groups):
        group_df[f'G{g+1}'] = group_returns[g] + [np.nan] * (max_len - len(group_returns[g]))
    
    spread_series = pd.Series(spread_returns, index=spread_dates)
    
    # 累计收益
    cum_returns = (1 + group_df.fillna(0)).cumprod()
    
    # 单调性检验: 分组编号 vs 平均收益的 Spearman 相关
    avg_ret_by_group = group_df.mean()
    if avg_ret_by_group.notna().sum() >= 3:
        mono_corr, _ = stats.spearmanr(range(len(avg_ret_by_group)), avg_ret_by_group)
    else:
        mono_corr = np.nan
    
    # Top-Bottom t检验
    if len(spread_returns) > 5:
        t_stat, p_val = stats.ttest_1samp(spread_returns, 0)
    else:
        t_stat, p_val = np.nan, np.nan
    
    return {
        'group_returns': group_df,
        'cum_returns': cum_returns,
        'spread_return': spread_series,
        'monotonicity_score': mono_corr,
        'top_bottom_t_stat': t_stat,
        'top_bottom_p_value': p_val,
        'avg_group_returns': avg_ret_by_group,
    }


def test_monotonicity(decile_result, p_threshold=0.05):
    """
    检验单调性是否显著
    
    通过条件:
    1. Top-Bottom spread t-test p < threshold (双尾)
    2. Spearman corr between group# and avg return is significant
    3. G10 avg return > G1 avg return (或反向, 取决于因子方向)
    
    Returns
    -------
    dict
        {'passed': bool, 'p_value': float, 'mono_score': float, 'spread_mean': float}
    """
    spread_mean = decile_result['spread_return'].mean()
    p_val = decile_result['top_bottom_p_value']
    mono = decile_result['monotonicity_score']
    
    passed = (
        not np.isnan(p_val) and p_val < p_threshold and
        not np.isnan(mono) and abs(mono) > 0.3
    )
    
    return {
        'passed': passed,
        'p_value': p_val,
        'monotonicity_score': mono,
        'spread_mean': spread_mean,
    }
