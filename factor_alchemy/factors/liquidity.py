"""
流动性因子: Amihud非流动性、日均成交额、换手/波动比
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class AmihudIlliq(BaseFactor):
    """
    Amihud (2002) 非流动性指标
    
    illiq = mean(|ret| / dollar_volume) over N days
    高illiq = 低流动性 = 流动性溢价 (正向因子)
    """
    def __init__(self):
        super().__init__('amihud_illiq', 'liquidity', 'Amihud非流动')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        
        if close is None or volume is None:
            return pd.DataFrame()
        
        # 日收益
        daily_ret = close.pct_change().abs()
        
        # 成交额 (简化: 用 volume 代理)
        # 更精确应该用 close * volume
        dollar_vol = close * volume
        
        # illiq = |ret| / dollar_vol
        illiq_daily = daily_ret / dollar_vol.replace(0, np.nan)
        
        # 20日均值
        illiq = illiq_daily.rolling(window=20, min_periods=10).mean()
        
        # 高 illiq = 低流动性 = 正向 (流动性溢价)
        return cross_sectional_zscore(illiq)


class DollarVol20D(BaseFactor):
    """近20日日均成交额"""
    def __init__(self):
        super().__init__('dollar_vol_20d', 'liquidity', '日均成交额')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        
        if close is None or volume is None:
            return pd.DataFrame()
        
        dollar_vol = close * volume
        avg_dv = dollar_vol.rolling(window=20, min_periods=10).mean()
        
        # 取对数
        result = np.log(avg_dv.replace(0, np.nan))
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 低成交额 → 流动性溢价 → 正向
        return cross_sectional_zscore(-result)


class DollarVolStability(BaseFactor):
    """
    成交额稳定性 (变异系数 CV)
    
    计算: -std(dollar_vol, 20d) / mean(dollar_vol, 20d)
    其中 dollar_vol = volume × close
    低CV → 成交额稳定 → 流动性供给稳定 → 预期收益高
    
    与 dollar_vol_20d 区分: 该因子衡量成交额水平(规模),
    DollarVolStability 衡量成交额稳定性(波动/均值比).
    """
    def __init__(self):
        super().__init__('dollar_vol_stability', 'liquidity', '成交额稳定性')
    
    def compute(self, price_data, financial_data=None, valuation_data=None, window=20, min_periods=10, **kwargs):
        close = price_data.get('close')
        volume = price_data.get('volume')
        
        if close is None or volume is None:
            return pd.DataFrame()
        
        dollar_vol = close * volume
        roll_std = dollar_vol.rolling(window=window, min_periods=min_periods).std()
        roll_mean = dollar_vol.rolling(window=window, min_periods=min_periods).mean()
        
        cv = roll_std / roll_mean.replace(0, np.nan)
        result = -cv  # 高CV(不稳定) → 负向; 低CV(稳定) → 正向
        result = result.replace([np.inf, -np.inf], np.nan)
        return result


class TurnoverCv20D(BaseFactor):
    """
    换手率变异系数 (CV of Turnover)
    
    计算: -std(turnover, 20d) / mean(turnover, 20d)
    高CV → 换手率波动大 → 流动性供给不稳定 → 负向收益
    低CV → 换手率稳定 → 流动性稳定 → 正向收益
    
    与 turnover_std 区分: turnover_std 衡量换手率的绝对波动幅度,
    TurnoverCv20D 衡量换手率的相对变异性(变异系数), 消除了规模效应。
    
    来源: 2026-06-27 日频因子实验, ICIR=+0.984, +IC%=84.5%
    """
    def __init__(self):
        super().__init__('turnover_cv_20d', 'liquidity', '换手率变异系数')
    
    def compute(self, price_data, financial_data=None, valuation_data=None, window=20, min_periods=10, **kwargs):
        turnover = price_data.get('turnover')
        
        if turnover is None:
            return pd.DataFrame()
        
        roll_std = turnover.rolling(window=window, min_periods=min_periods).std()
        roll_mean = turnover.rolling(window=window, min_periods=min_periods).mean()
        
        cv = roll_std / roll_mean.replace(0, np.nan)
        result = -cv  # 高CV(不稳定) → 负向; 低CV(稳定) → 正向
        result = result.replace([np.inf, -np.inf], np.nan)
        return result


class TurnoverToVol(BaseFactor):
    """换手率 / 波动率比 (流动性深度指标)"""
    def __init__(self):
        super().__init__('turnover_to_vol', 'liquidity', '换手/波动比')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        turnover = price_data.get('turnover')
        
        if close is None or turnover is None:
            return pd.DataFrame()
        
        daily_ret = close.pct_change()
        vol_20d = daily_ret.rolling(window=20, min_periods=10).std()
        to_avg_20d = turnover.rolling(window=20, min_periods=10).mean()
        
        ratio = to_avg_20d / vol_20d.replace(0, np.nan)
        
        # 低换手/波动比 → 流动性差 → 正向 (流动性溢价)
        return cross_sectional_zscore(-ratio)
