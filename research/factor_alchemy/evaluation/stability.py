"""
时序稳定性评估 (Time-Series Stability Evaluation)
==================================================
基于 jqfactor_analyzer 灵感设计，评估因子排名在时间轴上的持久性。

5个子维度:
  1. 排名自相关 (Rank Autocorrelation) — 因子排名在连续周的相关性
  2. 高分位留存率 (Top Quantile Retention) — 最高分位股票在下期的留存比例
  3. IC 稳定性 (IC Stability) — IC序列的变异系数和滚动标准差
  4. IC 衰减 (IC Decay) — IC在不同前瞻期的衰减速度
  5. 平均换手率 (Mean Turnover) — 维持Top-N持仓所需的周均换手

综合 0-1 分数越高越好。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from collections import OrderedDict


def compute_rank_autocorrelation(factor_df, lags=(1, 2, 4), min_stocks=30):
    """
    因子排名自相关: 衡量因子值排名的时序稳定性
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子值, index=date, columns=stocks
    lags : tuple
        滞后周期数 (如 (1,2,4) 表示滞后1/2/4周)
    min_stocks : int
        每期最低股票数
    
    Returns
    -------
    dict
        {
            'lag_1': {'mean': float, 'std': float, 'series': pd.Series},
            'lag_2': {...},
            ...
            'avg_autocorr': float,       # 各滞后期均值
            'stability_score': float,    # 0-1, 基于lag_1均值
        }
    """
    result = OrderedDict()
    all_lag1 = []
    
    for lag in lags:
        corr_list = []
        dates = []
        
        dates_sorted = sorted(factor_df.index)
        for i in range(len(dates_sorted) - lag):
            t0 = dates_sorted[i]
            t1 = dates_sorted[i + lag]
            
            f0 = factor_df.loc[t0].dropna()
            f1 = factor_df.loc[t1].dropna()
            
            common = list(set(f0.index) & set(f1.index))
            if len(common) < min_stocks:
                continue
            
            # Spearman rank correlation of factor values
            try:
                r, _ = stats.spearmanr(f0[common], f1[common])
                if np.isfinite(r) and not np.isnan(r):
                    corr_list.append(r)
                    dates.append(t1)
            except:
                continue
        
        if len(corr_list) < 5:
            result[f'lag_{lag}'] = {
                'mean': np.nan, 'std': np.nan,
                'series': pd.Series(dtype=float),
                'n': 0
            }
        else:
            series = pd.Series(corr_list, index=dates)
            result[f'lag_{lag}'] = {
                'mean': np.mean(corr_list),
                'std': np.std(corr_list, ddof=1),
                'series': series,
                'n': len(corr_list),
            }
            if lag == 1:
                all_lag1 = corr_list
    
    # 综合指标
    all_means = [result[f'lag_{l}']['mean'] for l in lags
                 if not np.isnan(result[f'lag_{l}']['mean'])]
    avg_autocorr = np.mean(all_means) if all_means else np.nan
    
    # 稳定性分数: lag_1 均值作为主要指标 (经验上 >0.8=稳定, <0.3=极不稳定)
    lag1_mean = result['lag_1']['mean'] if not np.isnan(result['lag_1']['mean']) else 0
    stability_score = np.clip((lag1_mean - 0.3) / 0.5, 0, 1) if lag1_mean > 0 else 0
    
    result['avg_autocorr'] = avg_autocorr
    result['stability_score'] = stability_score
    
    return result


def compute_top_quantile_retention(factor_df, n_quantiles=5, top_fraction=0.2,
                                    min_stocks=30):
    """
    高分位留存率: 最高分位的股票在下期仍留在最高分位的比例
    
    Parameters
    ----------
    factor_df : pd.DataFrame
    n_quantiles : int
        分组数
    top_fraction : float
        高分位的股票比例 (默认20%)
    min_stocks : int
        每期最低股票数
    
    Returns
    -------
    dict
        {
            'mean_retention': float,    # 平均留存率
            'std_retention': float,
            'series': pd.Series,        # 每期留存率
            'stability_score': float,   # 0-1
        }
    """
    dates_sorted = sorted(factor_df.index)
    retention_list = []
    ret_dates = []
    
    for i in range(len(dates_sorted) - 1):
        t0 = dates_sorted[i]
        t1 = dates_sorted[i + 1]
        
        f0 = factor_df.loc[t0].dropna()
        f1 = factor_df.loc[t1].dropna()
        
        n = len(f0)
        if n < min_stocks:
            continue
        
        # 确定高分位阈值
        top_n = max(1, int(n * top_fraction))
        top_stocks = set(f0.nlargest(top_n).index)
        
        # 下期仍在前 top_n 的数量
        top_next = set(f1.nlargest(top_n).index)
        retained = len(top_stocks & top_next)
        
        retention = retained / len(top_stocks) if len(top_stocks) > 0 else 0
        retention_list.append(retention)
        ret_dates.append(t1)
    
    if len(retention_list) < 5:
        return {
            'mean_retention': np.nan, 'std_retention': np.nan,
            'series': pd.Series(dtype=float),
            'stability_score': np.nan,
        }
    
    series = pd.Series(retention_list, index=ret_dates)
    mean_ret = np.mean(retention_list)
    std_ret = np.std(retention_list, ddof=1)
    
    # 经验上: >60%=稳定, <20%=极不稳定
    stability_score = np.clip((mean_ret - 0.2) / 0.4, 0, 1)
    
    return {
        'mean_retention': mean_ret,
        'std_retention': std_ret,
        'series': series,
        'stability_score': stability_score,
    }


def compute_ic_stability(ic_series, window=12):
    """
    IC 稳定性: IC序列的变异系数和滚动标准差
    
    Parameters
    ----------
    ic_series : pd.Series
        IC时间序列 (来自 compute_ic_icir)
    window : int
        滚动窗口大小
    
    Returns
    -------
    dict
        {
            'cv_ic': float,             # IC变异系数 (越小越稳定)
            'mean_ic': float,
            'std_ic': float,
            'rolling_std_mean': float,  # 滚动标准差的均值
            'rolling_std_max': float,
            'rolling_std_series': pd.Series,
            'stability_score': float,   # 0-1
        }
    """
    if len(ic_series) < 10:
        return {
            'cv_ic': np.nan, 'mean_ic': np.nan, 'std_ic': np.nan,
            'rolling_std_mean': np.nan, 'rolling_std_max': np.nan,
            'rolling_std_series': pd.Series(dtype=float),
            'stability_score': np.nan,
        }
    
    mean_ic = ic_series.mean()
    std_ic = ic_series.std(ddof=1)
    cv_ic = std_ic / abs(mean_ic) if abs(mean_ic) > 0 else np.inf
    
    # 滚动标准差 (衡量IC波动随时间的变化)
    roll_std = ic_series.rolling(window=min(window, max(5, len(ic_series)//4))).std()
    roll_std_clean = roll_std.dropna()
    
    roll_std_mean = roll_std_clean.mean() if len(roll_std_clean) > 0 else np.nan
    roll_std_max = roll_std_clean.max() if len(roll_std_clean) > 0 else np.nan
    
    # 稳定性分数: CV < 1.0 非常稳定, CV > 5.0 极不稳定
    # 1/|ICIR| = CV, 所以 ICIR=0.5 → CV=2.0
    stability_score = np.clip(1.0 - (cv_ic - 1.0) / 4.0, 0, 1) if np.isfinite(cv_ic) else 0
    
    return {
        'cv_ic': cv_ic,
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'rolling_std_mean': roll_std_mean,
        'rolling_std_max': roll_std_max,
        'rolling_std_series': roll_std,
        'stability_score': stability_score,
    }


def compute_ic_decay(factor_df, forward_returns, lags=(1, 2, 4), min_stocks=30):
    """
    IC 衰减: IC在不同前瞻期的表现
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子值
    forward_returns : pd.DataFrame
        前向收益 (周频，已对齐)
    lags : tuple
        前瞻周期数
    min_stocks : int
    
    Returns
    -------
    dict
        {
            'lag_1': {'ic_mean': float, 'icir': float},
            'lag_2': {...},
            ...
            'decay_rate': float,        # 衰减率 (负值=加速衰减不好)
            'half_life': float,         # IC半衰期 (周)
            'stability_score': float,   # 0-1
        }
    """
    from .ic_analysis import compute_ic_icir
    
    result = OrderedDict()
    
    for lag in lags:
        # 构造滞后前向收益
        shifted = forward_returns.shift(-(lag - 1))
        ic_result = compute_ic_icir(factor_df, shifted, min_stocks)
        result[f'lag_{lag}'] = {
            'ic_mean': ic_result['ic_mean'],
            'icir': abs(ic_result['icir']),
            'ic_std': ic_result['ic_std'],
        }
    
    # 衰减分析
    ic_means = [result[f'lag_{l}']['ic_mean'] for l in lags]
    icirs = [result[f'lag_{l}']['icir'] for l in lags]
    
    # 线性衰减率 (log-scale)
    if all(np.isfinite(x) and x != 0 for x in ic_means) and len(lags) >= 2:
        from scipy.stats import linregress
        log_means = [np.log(abs(x)) if abs(x) > 0 else -10 for x in ic_means]
        slope, _, _, _, _ = linregress([0] + list(lags[:-1]), log_means)
        decay_rate = slope  # 越负衰减越快
        half_life = -np.log(2) / slope if slope < 0 else np.inf
    else:
        decay_rate = np.nan
        half_life = np.nan
    
    # 稳定性分数: IC半衰期越长越稳定 (>8周=稳定, <2周=极不稳定)
    if np.isfinite(half_life) and half_life > 0:
        stability_score = np.clip((half_life - 2) / 6, 0, 1)
    else:
        stability_score = 0.5  # 默认中性
    
    result['decay_rate'] = decay_rate
    result['half_life'] = half_life
    result['stability_score'] = stability_score
    
    return result


def compute_mean_turnover(factor_df, top_n=30, min_stocks=50):
    """
    平均换手率: 维持Top-N持仓所需的周均换手
    
    Parameters
    ----------
    factor_df : pd.DataFrame
    top_n : int
        持仓数量
    min_stocks : int
        每期最低股票数
    
    Returns
    -------
    dict
        {
            'mean_turnover': float,     # 平均单边周换手率
            'std_turnover': float,
            'series': pd.Series,
            'stability_score': float,   # 0-1
        }
    """
    dates_sorted = sorted(factor_df.index)
    turnover_list = []
    turn_dates = []
    
    for i in range(len(dates_sorted) - 1):
        t0 = dates_sorted[i]
        t1 = dates_sorted[i + 1]
        
        f0 = factor_df.loc[t0].dropna()
        
        n = len(f0)
        if n < min_stocks:
            continue
        
        actual_top = min(top_n, n)
        prev_top = set(f0.nlargest(actual_top).index)
        
        f1 = factor_df.loc[t1].dropna()
        new_top = set(f1.nlargest(actual_top).index)
        
        # 新增 = 不在上期top中的
        new_stocks = len(new_top - prev_top)
        turnover = new_stocks / actual_top if actual_top > 0 else 0
        turnover_list.append(turnover)
        turn_dates.append(t1)
    
    if len(turnover_list) < 5:
        return {
            'mean_turnover': np.nan, 'std_turnover': np.nan,
            'series': pd.Series(dtype=float),
            'stability_score': np.nan,
        }
    
    series = pd.Series(turnover_list, index=turn_dates)
    mean_to = np.mean(turnover_list)
    std_to = np.std(turnover_list, ddof=1)
    
    # 经验: <30%=非常稳定, >70%=极不稳定
    stability_score = np.clip(1.0 - mean_to / 0.7, 0, 1)
    
    return {
        'mean_turnover': mean_to,
        'std_turnover': std_to,
        'series': series,
        'stability_score': stability_score,
    }


def compute_stability_comprehensive(factor_df, forward_returns=None, ic_series=None,
                                     top_n=30, lags=(1, 2, 4), min_stocks=30):
    """
    综合时序稳定性评估 (5子维度)
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子值面板
    forward_returns : pd.DataFrame or None
        前向收益 (IC稳定性/衰减需要)
    ic_series : pd.Series or None
        已有IC序列 (可选, 避免重复计算)
    top_n : int
        换手率计算的持仓数
    lags : tuple
        自相关滞后周期
    min_stocks : int
        最低股票数
    
    Returns
    -------
    dict
        {
            'rank_autocorr': dict,
            'top_retention': dict,
            'ic_stability': dict,
            'ic_decay': dict,
            'mean_turnover': dict,
            'sub_scores': {5个子分数},
            'total_stability_score': float,   # 0-1 综合
            'passed': bool,
        }
    """
    # 1. 排名自相关
    autocorr = compute_rank_autocorrelation(factor_df, lags=lags, min_stocks=min_stocks)
    
    # 2. 高分位留存率
    retention = compute_top_quantile_retention(factor_df, min_stocks=min_stocks)
    
    # 3. IC稳定性
    if ic_series is None and forward_returns is not None:
        from .ic_analysis import compute_ic_icir
        ic_result = compute_ic_icir(factor_df, forward_returns, min_stocks)
        ic_series = ic_result.get('ic_series', pd.Series(dtype=float))
    elif ic_series is None:
        ic_series = pd.Series(dtype=float)
    
    ic_stab = compute_ic_stability(ic_series) if len(ic_series) >= 10 else {
        'stability_score': np.nan
    }
    
    # 4. IC衰减
    if forward_returns is not None:
        ic_decay = compute_ic_decay(factor_df, forward_returns, lags=lags, min_stocks=min_stocks)
    else:
        ic_decay = {'stability_score': np.nan}
    
    # 5. 平均换手率
    turnover = compute_mean_turnover(factor_df, top_n=top_n, min_stocks=min_stocks)
    
    # 综合分数
    sub_scores = {
        'rank_autocorr': autocorr.get('stability_score', np.nan),
        'top_retention': retention.get('stability_score', np.nan),
        'ic_stability': ic_stab.get('stability_score', np.nan),
        'ic_decay': ic_decay.get('stability_score', np.nan),
        'mean_turnover': turnover.get('stability_score', np.nan),
    }
    
    # 可用子维度求平均 (跳过NaN)
    valid_scores = [s for s in sub_scores.values() if not np.isnan(s)]
    total_score = np.mean(valid_scores) if valid_scores else 0.0
    
    passed = total_score >= 0.5  # 0.5 为综合通过阈值
    
    return {
        'rank_autocorr': autocorr,
        'top_retention': retention,
        'ic_stability': ic_stab,
        'ic_decay': ic_decay,
        'mean_turnover': turnover,
        'sub_scores': sub_scores,
        'total_stability_score': total_score,
        'passed': passed,
    }


def print_stability_report(stability_result):
    """打印稳定性评估报告"""
    sr = stability_result
    
    print(f"\n  {'='*60}")
    print(f"  时序稳定性评估报告")
    print(f"  {'='*60}")
    
    autocorr = sr['rank_autocorr']
    retention = sr['top_retention']
    ic_stab = sr['ic_stability']
    ic_decay = sr['ic_decay']
    turnover = sr['mean_turnover']
    
    # 排名自相关
    print(f"\n  [1/5] 排名自相关 (Rank Autocorrelation)")
    for lag in [1, 2, 4]:
        key = f'lag_{lag}'
        if key in autocorr and not np.isnan(autocorr[key].get('mean', np.nan)):
            m = autocorr[key]['mean']
            bar = '█' * int(max(0, min(20, m * 20)))
            print(f"    lag_{lag}: {m:+.3f}  {bar}")
    print(f"    分数: {autocorr.get('stability_score', 0):.3f}")
    
    # 高分位留存率
    print(f"\n  [2/5] 高分位留存率 (Top Quantile Retention)")
    mr = retention.get('mean_retention', 0)
    if not np.isnan(mr):
        bar = '█' * int(max(0, min(20, mr * 20 / 0.8)))
        print(f"    平均留存: {mr:.1%}  {bar}")
        print(f"    分数: {retention.get('stability_score', 0):.3f}")
    
    # IC稳定性
    print(f"\n  [3/5] IC稳定性 (IC Stability)")
    cv = ic_stab.get('cv_ic', np.nan)
    if not np.isnan(cv):
        print(f"    CV(IC) = {cv:.2f}  (= 1/|ICIR|)")
        print(f"    滚动σ均值: {ic_stab.get('rolling_std_mean', 0):.4f}")
        print(f"    分数: {ic_stab.get('stability_score', 0):.3f}")
    
    # IC衰减
    print(f"\n  [4/5] IC衰减 (IC Decay)")
    for lag in [1, 2, 4]:
        key = f'lag_{lag}'
        if key in ic_decay:
            d = ic_decay[key]
            if not np.isnan(d.get('ic_mean', np.nan)):
                print(f"    IC_{lag}w: mean={d['ic_mean']:+.4f}  ICIR={d['icir']:.3f}")
    hl = ic_decay.get('half_life', np.nan)
    if not np.isnan(hl):
        print(f"    IC半衰期: {hl:.1f}周  |  分数: {ic_decay.get('stability_score', 0):.3f}")
    
    # 平均换手率
    print(f"\n  [5/5] 平均换手率 (Mean Turnover)")
    mt = turnover.get('mean_turnover', 0)
    if not np.isnan(mt):
        bar = '█' * int(max(0, min(20, (1 - mt) * 20)))
        print(f"    周均换手: {mt:.1%}  {bar}")
        print(f"    分数: {turnover.get('stability_score', 0):.3f}")
    
    # 综合
    print(f"\n  {'─'*60}")
    print(f"  综合稳定性分数: {sr['total_stability_score']:.3f}  "
          f"{'PASS' if sr['passed'] else 'FAIL'}")
    print(f"  {'─'*60}")
