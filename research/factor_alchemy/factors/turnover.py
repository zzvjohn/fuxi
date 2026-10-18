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

        avg_20d = to.rolling(window=20, min_periods=10).mean()

        if industry_map is not None:
            # ★ 向量化行业去均值 (替代 O(N^3) 三重循环, v8.1 性能修复)
            # 将宽表 → 长表 (date, stock) → join 行业 → groupby transform
            stacked = avg_20d.stack()
            stacked.name = 'turnover'
            ind_series = pd.Series(industry_map, name='industry')
            df_long = pd.DataFrame(stacked).join(ind_series, on=stacked.index.names[1])

            # 按 (date, industry) 分组去均值
            ind_mean = df_long.groupby(
                [df_long.index.get_level_values(0), 'industry']
            )['turnover'].transform('mean')

            de_meaned = df_long['turnover'] - ind_mean
            result = -de_meaned.unstack()
        else:
            result = -avg_20d
        
        return cross_sectional_zscore(result)
