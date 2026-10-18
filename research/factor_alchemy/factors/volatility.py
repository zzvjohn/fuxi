"""
波动率因子: 1月/3月波动、下行波动、Beta、特质波动
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class Vol1M(BaseFactor):
    """近1月日收益标准差"""
    def __init__(self):
        super().__init__('vol_1m', 'volatility', '1月波动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(window=20, min_periods=10).std()
        
        # 低波正向 → 取负号
        result = -vol
        return cross_sectional_zscore(result)


class Vol3M(BaseFactor):
    """近3月日收益标准差"""
    def __init__(self):
        super().__init__('vol_3m', 'volatility', '3月波动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(window=60, min_periods=30).std()
        
        result = -vol
        return cross_sectional_zscore(result)


class DownsideVol(BaseFactor):
    """下行波动 = 负收益的标准差"""
    def __init__(self):
        super().__init__('downside_vol', 'volatility', '下行波动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        if close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        # 仅保留负收益
        neg_ret = daily_ret.copy()
        neg_ret[neg_ret > 0] = np.nan
        
        dvol = neg_ret.rolling(window=60, min_periods=20).std()
        
        result = -dvol
        return cross_sectional_zscore(result)


class Beta(BaseFactor):
    """市场Beta (用沪深300做基准) — 向量化计算"""
    def __init__(self):
        super().__init__('beta', 'volatility', 'Beta')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        index_close = price_data.get('index_close')
        
        if close is None or index_close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        idx_ret = index_close.pct_change()
        idx_aligned = idx_ret.reindex(daily_ret.index)
        
        window = 60
        # 向量化滚动Beta:
        # Beta = cov(r_i, r_m) / var(r_m)
        # cov(X,Y) = E[XY] - E[X]E[Y]
        prod = daily_ret.multiply(idx_aligned, axis=0)
        roll_mean_prod = prod.rolling(window=window, min_periods=window).mean()
        roll_mean_stock = daily_ret.rolling(window=window, min_periods=window).mean()
        roll_mean_idx = idx_aligned.rolling(window=window, min_periods=window).mean()
        roll_cov = roll_mean_prod - roll_mean_stock.multiply(roll_mean_idx, axis=0)
        roll_var_idx = idx_aligned.rolling(window=window, min_periods=window).var()
        beta = roll_cov.div(roll_var_idx, axis=0)
        beta[roll_var_idx <= 0] = np.nan
        
        # 低Beta正向 (低风险异象) → 取负号
        beta = -beta
        return cross_sectional_zscore(beta)


class IdioVol(BaseFactor):
    """特质波动 = 残差标准差 — 向量化近似
    
    idio_vol = std(r_i) * sqrt(1 - corr(r_i, r_m)^2)
    等价于 CAPM 残差的标准差
    """
    def __init__(self):
        super().__init__('idio_vol', 'volatility', '特质波动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        index_close = price_data.get('index_close')
        
        if close is None or index_close is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        idx_ret = index_close.pct_change()
        idx_aligned = idx_ret.reindex(daily_ret.index)
        
        window = 60
        # 向量化: idio = std(r_i) * sqrt(1 - corr(r_i, r_m)^2)
        roll_std = daily_ret.rolling(window=window, min_periods=window).std()
        
        # Rolling correlation via vectorized cov/(std1*std2)
        prod = daily_ret.multiply(idx_aligned, axis=0)
        roll_mean_prod = prod.rolling(window=window, min_periods=window).mean()
        roll_mean_stock = daily_ret.rolling(window=window, min_periods=window).mean()
        roll_mean_idx = idx_aligned.rolling(window=window, min_periods=window).mean()
        roll_cov = roll_mean_prod - roll_mean_stock.multiply(roll_mean_idx, axis=0)
        roll_std_idx = idx_aligned.rolling(window=window, min_periods=window).std()
        
        roll_corr = roll_cov.div(roll_std.multiply(roll_std_idx, axis=0), axis=0)
        roll_corr = roll_corr.clip(-0.999, 0.999)
        
        idio_vol = roll_std * np.sqrt(np.maximum(1 - roll_corr**2, 0))
        
        # 高特质波动 → 负向信号 (A股特质波动异象) → 取负号
        return cross_sectional_zscore(-idio_vol)
