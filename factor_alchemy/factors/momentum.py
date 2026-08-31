"""
动量因子: 各周期收益 (1m/3m/6m/12m)、跳1月收益、最大单日收益
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class Ret1M(BaseFactor):
    """近1月收益"""
    def __init__(self):
        super().__init__('ret_1m', 'momentum', '1月收益')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        ret = close / close.shift(20) - 1  # ~1个月交易日
        
        # A股: 短期反转 → 取负号
        result = -ret
        return cross_sectional_zscore(result)


class Ret3M(BaseFactor):
    """近3月收益"""
    def __init__(self):
        super().__init__('ret_3m', 'momentum', '3月收益')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        ret = close / close.shift(60) - 1
        
        result = -ret
        return cross_sectional_zscore(result)


class Ret6M(BaseFactor):
    """近6月收益"""
    def __init__(self):
        super().__init__('ret_6m', 'momentum', '6月收益')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        ret = close / close.shift(120) - 1
        
        result = -ret
        return cross_sectional_zscore(result)


class Ret12M(BaseFactor):
    """近12月收益"""
    def __init__(self):
        super().__init__('ret_12m', 'momentum', '12月收益')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        ret = close / close.shift(240) - 1
        
        result = -ret
        return cross_sectional_zscore(result)


class Ret1MSkip1M(BaseFactor):
    """近1月收益(跳过最近1月) = (t-21)/(t-41) - 1"""
    def __init__(self):
        super().__init__('ret_1m_skip1m', 'momentum', '1月收益(跳1月)')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        ret = close.shift(20) / close.shift(40) - 1
        
        result = -ret
        return cross_sectional_zscore(result)


class MaxRet1M(BaseFactor):
    """
    月内最大单日收益 (Bali et al. 2011 - 彩票型特征)
    MAX > 高 → 彩票型股票 → 负向因子
    """
    def __init__(self):
        super().__init__('max_ret_1m', 'momentum', '月最大日收益')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        max_ret = daily_ret.rolling(window=20, min_periods=10).max()
        
        result = -max_ret
        return cross_sectional_zscore(result)
