"""
因子计算基类
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseFactor(ABC):
    """所有因子的抽象基类"""
    
    def __init__(self, name, category, label):
        self.name = name
        self.category = category
        self.label = label
    
    @abstractmethod
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        """
        计算因子值
        
        Parameters
        ----------
        price_data : dict
            价格数据, 包含 close/volume/turnover 等 DataFrame
            key: field name, value: pd.DataFrame(index=date, columns=stocks)
        financial_data : pd.DataFrame
            财务数据
        valuation_data : pd.DataFrame
            日频估值数据
        
        Returns
        -------
        pd.DataFrame
            因子值, index=date, columns=stocks
        """
        pass
    
    def __repr__(self):
        return f"Factor({self.name}: {self.label})"


def rolling_apply(df, window, func, min_periods=None):
    """
    滚动窗口应用函数
    
    Parameters
    ----------
    df : pd.DataFrame
    window : int
    func : callable
    min_periods : int
    
    Returns
    -------
    pd.DataFrame
    """
    if min_periods is None:
        min_periods = max(window // 2, 5)
    
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < min_periods:
            continue
        rolled = s.rolling(window=window, min_periods=min_periods)
        result[col] = func(rolled)
    return result


def cross_sectional_zscore(df, method='median'):
    """
    截面标准化
    
    Parameters
    ----------
    df : pd.DataFrame
    method : str
        'median' 或 'mean'
    
    Returns
    -------
    pd.DataFrame
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for idx in df.index:
        row = df.loc[idx].dropna()
        if len(row) < 10:
            continue
        if method == 'median':
            center = row.median()
        else:
            center = row.mean()
        scale = row.std()
        if scale > 0:
            result.loc[idx] = (df.loc[idx] - center) / scale
    return result
