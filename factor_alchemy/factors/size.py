"""
规模因子
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class LnMarketCap(BaseFactor):
    """对数总市值"""
    def __init__(self):
        super().__init__('ln_mcap', 'size', '对数市值')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'market_cap' not in valuation_data.columns:
            return pd.DataFrame()
        
        mcap = valuation_data[['code', 'trade_date', 'market_cap']].copy()
        mcap_pivot = mcap.pivot(index='trade_date', columns='code', values='market_cap')
        mcap_pivot.index = pd.to_datetime(mcap_pivot.index)
        
        # 取对数, 负向 (小市值 → 高得分)
        result = -np.log(mcap_pivot)
        result = result.replace([np.inf, -np.inf], np.nan)
        
        return cross_sectional_zscore(result)


class LnCirculatingMarketCap(BaseFactor):
    """对数流通市值"""
    def __init__(self):
        super().__init__('ln_circulating_mcap', 'size', '对数流通市值')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'circ_mv' not in valuation_data.columns:
            return pd.DataFrame()
        
        cmcap = valuation_data[['code', 'trade_date', 'circ_mv']].copy()
        cmcap_pivot = cmcap.pivot(index='trade_date', columns='code', values='circ_mv')
        cmcap_pivot.index = pd.to_datetime(cmcap_pivot.index)
        
        result = -np.log(cmcap_pivot)
        result = result.replace([np.inf, -np.inf], np.nan)
        
        return cross_sectional_zscore(result)
