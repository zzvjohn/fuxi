"""
成长因子: 营收增长/盈利增长/资产增长
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


class RevGrowthYoY(BaseFactor):
    """营收同比增长率"""
    def __init__(self):
        super().__init__('rev_growth_yoy', 'growth', '营收增长')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None or 'or_yoy' not in fn.columns:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'or_yoy']].copy()
        df = df.dropna(subset=['or_yoy'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='or_yoy')
        return cross_sectional_zscore(pivot)


class EarningsGrowthYoY(BaseFactor):
    """盈利同比增长率"""
    def __init__(self):
        super().__init__('earnings_growth_yoy', 'growth', '盈利增长')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None:
            return pd.DataFrame()
        
        # 优先用 netprofit_yoy, 其次 profit_dedt
        col = None
        for c in ['netprofit_yoy', 'profit_dedt']:
            if c in fn.columns:
                col = c
                break
        
        if col is None:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', col]].copy()
        df = df.dropna(subset=[col])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values=col)
        return cross_sectional_zscore(pivot)


class AssetGrowth(BaseFactor):
    """总资产增长率 (Cooper et al. 2008 — 资产增长异象)"""
    def __init__(self):
        super().__init__('asset_growth', 'growth', '资产增长')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None:
            return pd.DataFrame()
        
        # 优先用 assets_yoy
        if 'assets_yoy' in fn.columns:
            df = fn[['ts_code', 'end_date', 'assets_yoy']].copy()
            df = df.dropna(subset=['assets_yoy'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            pivot = df.pivot(index='end_date', columns='ts_code', values='assets_yoy')
        else:
            # fallback: 手动计算
            asset_col = None
            for c in fn.columns:
                cl = c.lower()
                if 'total_assets' in cl or 'total_asset' in cl:
                    asset_col = c
                    break
            
            if asset_col is None:
                return pd.DataFrame()
            
            df = fn[['ts_code', 'end_date', asset_col]].copy()
            df = df.dropna()
            df['end_date'] = pd.to_datetime(df['end_date'])
            df = df.sort_values(['ts_code', 'end_date'])
            
            results = {}
            for code, group in df.groupby('ts_code'):
                group = group.sort_values('end_date')
                if len(group) >= 5:
                    latest = group.iloc[-1]
                    prev = group.iloc[-5]
                    if prev[asset_col] > 0:
                        growth = latest[asset_col] / prev[asset_col] - 1
                        results[code] = growth
            
            if not results:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(
                [(k, v) for k, v in results.items()],
                columns=['code', 'asset_growth']
            )
            result_df['trade_date'] = pd.to_datetime(df['end_date'].max())
            pivot = result_df.pivot(index='trade_date', columns='code', values='asset_growth')
        
        # 资产增长高 → 负向 (Cooper异象) → 取负号
        result = -pivot
        return cross_sectional_zscore(result)
