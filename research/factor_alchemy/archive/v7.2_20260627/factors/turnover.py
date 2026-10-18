"""
换手率因子: 月均换手/季均换手/换手波动/换手变化/异常换手
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class AvgTurnover1M(BaseFactor):
    """近1月日均换手率"""
    def __init__(self):
        super().__init__('avg_turnover_1m', 'turnover', '月均换手')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'turnover' not in price_data:
            return pd.DataFrame()
        
        to = price_data['turnover'].copy()  # index=date, columns=stocks
        
        # 20日滚动均值
        avg_20d = to.rolling(window=20, min_periods=10).mean()
        
        # 截面标准化 (换手高→负向, 取负号)
        result = -avg_20d
        return cross_sectional_zscore(result)


class AvgTurnover3M(BaseFactor):
    """近3月日均换手率"""
    def __init__(self):
        super().__init__('avg_turnover_3m', 'turnover', '季均换手')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'turnover' not in price_data:
            return pd.DataFrame()
        
        to = price_data['turnover'].copy()
        
        # 60日滚动均值 (约13周)
        avg_60d = to.rolling(window=60, min_periods=30).mean()
        
        result = -avg_60d
        return cross_sectional_zscore(result)


class TurnoverStd(BaseFactor):
    """换手率波动 = 20日换手率标准差"""
    def __init__(self):
        super().__init__('turnover_std', 'turnover', '换手波动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'turnover' not in price_data:
            return pd.DataFrame()
        
        to = price_data['turnover'].copy()
        
        # 20日滚动标准差
        std_20d = to.rolling(window=20, min_periods=10).std()
        
        result = -std_20d
        return cross_sectional_zscore(result)


class TurnoverChange(BaseFactor):
    """换手率变化 = 近5日均换手 / 近20日均换手 - 1"""
    def __init__(self):
        super().__init__('turnover_change', 'turnover', '换手变化')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'turnover' not in price_data:
            return pd.DataFrame()
        
        to = price_data['turnover'].copy()
        
        avg_5d = to.rolling(window=5, min_periods=3).mean()
        avg_20d = to.rolling(window=20, min_periods=10).mean()
        
        change = avg_5d / avg_20d.replace(0, np.nan) - 1
        
        # 换手率骤升 → 可能有信息/操纵 → 负向 (A股实证)
        result = -change
        return cross_sectional_zscore(result)


class AbnormalTurnover(BaseFactor):
    """异常换手 = 实际换手 - 行业均值换手"""
    def __init__(self):
        super().__init__('abnormal_turnover', 'turnover', '异常换手')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'turnover' not in price_data:
            return pd.DataFrame()
        
        to = price_data['turnover'].copy()
        
        industry_map = kwargs.get('industry_map', None)
        
        if industry_map is not None:
            avg_20d = to.rolling(window=20, min_periods=10).mean()
            
            # 按行业计算均值
            for idx in avg_20d.index:
                row = avg_20d.loc[idx]
                # 行业去均值
                for ind in set(industry_map.values()):
                    stocks_in_ind = [s for s in row.index if industry_map.get(s) == ind]
                    if len(stocks_in_ind) < 3:
                        continue
                    ind_mean = row[stocks_in_ind].mean()
                    avg_20d.loc[idx, stocks_in_ind] = avg_20d.loc[idx, stocks_in_ind] - ind_mean
            
            result = -avg_20d
        else:
            avg_20d = to.rolling(window=20, min_periods=10).mean()
            result = -avg_20d
        
        return cross_sectional_zscore(result)
