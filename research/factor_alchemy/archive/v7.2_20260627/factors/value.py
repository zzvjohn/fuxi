"""
价值因子: EP, BP, SP, CFP, DP
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class EP(BaseFactor):
    """Earnings-to-Price = 1/PE"""
    def __init__(self):
        super().__init__('ep', 'value', 'E/P')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'pe_ttm' not in valuation_data.columns:
            return pd.DataFrame()
        
        df = valuation_data[['code', 'trade_date', 'pe_ttm']].copy()
        # 剔除负PE
        df = df[df['pe_ttm'] > 0]
        df['ep'] = 1.0 / df['pe_ttm']
        
        pivot = df.pivot(index='trade_date', columns='code', values='ep')
        pivot.index = pd.to_datetime(pivot.index)
        
        result = pivot.replace([np.inf, -np.inf], np.nan)
        return cross_sectional_zscore(result)


class BP(BaseFactor):
    """Book-to-Price = 1/PB"""
    def __init__(self):
        super().__init__('bp', 'value', 'B/P')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'pb' not in valuation_data.columns:
            return pd.DataFrame()
        
        df = valuation_data[['code', 'trade_date', 'pb']].copy()
        df = df[df['pb'] > 0]
        df['bp'] = 1.0 / df['pb']
        
        pivot = df.pivot(index='trade_date', columns='code', values='bp')
        pivot.index = pd.to_datetime(pivot.index)
        
        result = pivot.replace([np.inf, -np.inf], np.nan)
        return cross_sectional_zscore(result)


class SP(BaseFactor):
    """Sales-to-Price = 1/PS"""
    def __init__(self):
        super().__init__('sp', 'value', 'S/P')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'ps_ttm' not in valuation_data.columns:
            return pd.DataFrame()
        
        df = valuation_data[['code', 'trade_date', 'ps_ttm']].copy()
        df = df[df['ps_ttm'] > 0]
        df['sp'] = 1.0 / df['ps_ttm']
        
        pivot = df.pivot(index='trade_date', columns='code', values='sp')
        pivot.index = pd.to_datetime(pivot.index)
        
        result = pivot.replace([np.inf, -np.inf], np.nan)
        return cross_sectional_zscore(result)


class CFP(BaseFactor):
    """Cash-Flow-to-Price = 每股现金流 / 股价
    
    数据源: fina_indicator_rich.cfps (每股现金流, TTM) + 后复权收盘价
    daily_basic 不含 pcf_ttm, 故直接从财务+价格推导
    """
    def __init__(self):
        super().__init__('cfp', 'value', 'CF/P')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        close = price_data.get('close')
        fn = financial_data
        
        # 优先使用 valuation_data 的 pcf_ttm (如果将来有)
        if valuation_data is not None and 'pcf_ttm' in valuation_data.columns:
            df = valuation_data[['code', 'trade_date', 'pcf_ttm']].copy()
            df = df[df['pcf_ttm'] > 0]
            df['cfp'] = 1.0 / df['pcf_ttm']
            pivot = df.pivot(index='trade_date', columns='code', values='cfp')
            pivot.index = pd.to_datetime(pivot.index)
            result = pivot.replace([np.inf, -np.inf], np.nan)
            return cross_sectional_zscore(result)
        
        # 回退方案: cfps / close_price
        if fn is None or close is None:
            return pd.DataFrame()
        
        if 'cfps' not in fn.columns:
            return pd.DataFrame()
        
        # 从财务数据取每股现金流
        cfps_df = fn[['ts_code', 'end_date', 'cfps']].dropna()
        cfps_df['end_date'] = pd.to_datetime(cfps_df['end_date'])
        cfps_pivot = cfps_df.pivot(index='end_date', columns='ts_code', values='cfps')
        
        # 对齐到周频, 前向填充
        weekly_close = close.resample('W').last()
        cfps_weekly = cfps_pivot.resample('W').last().reindex(weekly_close.index).ffill()
        
        # CF/P = cfps / close (cfps > 0)
        cfp = cfps_weekly / weekly_close
        cfp[cfps_weekly <= 0] = np.nan
        
        return cross_sectional_zscore(cfp)


class DP(BaseFactor):
    """Dividend Yield = 股息率"""
    def __init__(self):
        super().__init__('dp', 'value', '股息率')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if 'dv_ttm' not in valuation_data.columns:
            return pd.DataFrame()
        
        df = valuation_data[['code', 'trade_date', 'dv_ttm']].copy()
        df = df[df['dv_ttm'] > 0]
        
        pivot = df.pivot(index='trade_date', columns='code', values='dv_ttm')
        pivot.index = pd.to_datetime(pivot.index)
        
        result = pivot.replace([np.inf, -np.inf], np.nan)
        return cross_sectional_zscore(result)
