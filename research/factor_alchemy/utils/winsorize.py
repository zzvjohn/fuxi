"""
异常值缩尾处理
"""
import numpy as np
import pandas as pd


def winsorize_cross_section(df, lower=0.01, upper=0.99):
    """
    按截面缩尾: 每行(时间点)独立做 1%/99% 分位数缩尾
    
    Parameters
    ----------
    df : pd.DataFrame
        行为时间, 列为股票代码
    lower, upper : float
        缩尾分位数
    
    Returns
    -------
    pd.DataFrame
        缩尾后的 DataFrame
    """
    result = df.copy()
    for idx in df.index:
        row = df.loc[idx].dropna()
        if len(row) < 5:
            continue
        lo = np.percentile(row, lower * 100)
        hi = np.percentile(row, upper * 100)
        result.loc[idx] = df.loc[idx].clip(lo, hi)
    return result


def winsorize_series(series, lower=0.01, upper=0.99):
    """单序列缩尾"""
    s = series.dropna()
    if len(s) < 5:
        return series
    lo = np.percentile(s, lower * 100)
    hi = np.percentile(s, upper * 100)
    return series.clip(lo, hi)
