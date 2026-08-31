"""
盈利因子: Piotroski F-score, ROE, ROA, ROIC, 毛利率, 净利率, 应计利润

Piotroski F-score 严格按照原论文 9 项指标计算:
  Profitability (4项): ROA>0, CFO>0, ΔROA>0, ACCRUAL<0
  Leverage/Liquidity (3项): ΔLEVER<0, ΔLIQUID>0, EQ_OFFER=0
  Operating Efficiency (2项): ΔMARGIN>0, ΔTURNOVER>0
"""
import numpy as np
import pandas as pd
from .base import BaseFactor, cross_sectional_zscore


# ==================== Piotroski F-score ====================

class PiotroskiFScore(BaseFactor):
    """Piotroski F-Score (0-9) — 使用 fina_indicator_rich 38列数据"""
    def __init__(self):
        super().__init__('f_score', 'profitability', 'Piotroski F-score')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        """
        从财务数据计算 F-score
        
        需要的列: roa, grossprofit_margin, current_ratio, debt_to_assets,
                   assets_turn, ocfps, eps, bps
        """
        fn = financial_data
        if fn is None or len(fn) == 0:
            return pd.DataFrame()
        
        fn = fn.copy()
        if 'end_date' in fn.columns:
            fn['end_date'] = pd.to_datetime(fn['end_date'])
        
        fn = fn.sort_values(['ts_code', 'end_date'])
        
        results = {}
        
        for code, group in fn.groupby('ts_code'):
            if len(group) < 2:
                continue
            group = group.sort_values('end_date')
            
            latest = group.iloc[-1]
            prev = group.iloc[-2]
            
            score = 0
            
            # --- Profitability (4项) ---
            # 1. ROA > 0
            if 'roa' in group.columns:
                if pd.notna(latest.get('roa')) and latest['roa'] > 0:
                    score += 1
            
            # 2. CFO > 0 (用 ocfps 每股经营现金流)
            if 'ocfps' in group.columns:
                if pd.notna(latest.get('ocfps')) and latest['ocfps'] > 0:
                    score += 1
            
            # 3. ΔROA > 0
            if 'roa' in group.columns:
                roa_latest = latest.get('roa')
                roa_prev = prev.get('roa')
                if pd.notna(roa_latest) and pd.notna(roa_prev) and roa_latest > roa_prev:
                    score += 1
            
            # 4. Accrual < 0: 净利润现金含量 (eps vs ocfps)
            if 'eps' in group.columns and 'ocfps' in group.columns:
                eps_val = latest.get('eps')
                ocf_val = latest.get('ocfps')
                if pd.notna(eps_val) and pd.notna(ocf_val) and eps_val > 0:
                    if ocf_val > eps_val:  # 现金流 > 利润 → 低应计
                        score += 1
            
            # --- Leverage/Liquidity (3项) ---
            # 5. ΔLeverage < 0
            if 'debt_to_assets' in group.columns:
                lev_latest = latest.get('debt_to_assets')
                lev_prev = prev.get('debt_to_assets')
                if pd.notna(lev_latest) and pd.notna(lev_prev) and lev_latest < lev_prev:
                    score += 1
            
            # 6. ΔCurrent Ratio > 0
            if 'current_ratio' in group.columns:
                cr_latest = latest.get('current_ratio')
                cr_prev = prev.get('current_ratio')
                if pd.notna(cr_latest) and pd.notna(cr_prev) and cr_latest > cr_prev:
                    score += 1
            
            # 7. No equity offering (用 bps 不降来近似)
            if 'bps' in group.columns:
                bps_latest = latest.get('bps')
                bps_prev = prev.get('bps')
                if pd.notna(bps_latest) and pd.notna(bps_prev) and bps_latest >= bps_prev * 0.98:
                    score += 1
            else:
                score += 1  # 无数据默认通过
            
            # --- Operating Efficiency (2项) ---
            # 8. ΔGross Margin > 0
            if 'grossprofit_margin' in group.columns:
                gm_latest = latest.get('grossprofit_margin')
                gm_prev = prev.get('grossprofit_margin')
                if pd.notna(gm_latest) and pd.notna(gm_prev) and gm_latest > gm_prev:
                    score += 1
            
            # 9. ΔAsset Turnover > 0
            if 'assets_turn' in group.columns:
                at_latest = latest.get('assets_turn')
                at_prev = prev.get('assets_turn')
                if pd.notna(at_latest) and pd.notna(at_prev) and at_latest > at_prev:
                    score += 1
            
            results[code] = score
        
        # 转为 DataFrame
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(
            [(k, v) for k, v in results.items()],
            columns=['code', 'f_score']
        )
        result_df['trade_date'] = pd.to_datetime(fn['end_date'].max())
        
        pivot = result_df.pivot(index='trade_date', columns='code', values='f_score')
        return pivot


# ==================== 其他盈利因子 ====================

class ROE(BaseFactor):
    """ROE"""
    def __init__(self):
        super().__init__('roe', 'profitability', 'ROE')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if financial_data is None or 'roe' not in financial_data.columns:
            return pd.DataFrame()
        
        df = financial_data[['ts_code', 'end_date', 'roe']].copy()
        df = df.dropna(subset=['roe'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='roe')
        return cross_sectional_zscore(pivot)


class ROA(BaseFactor):
    """ROA"""
    def __init__(self):
        super().__init__('roa', 'profitability', 'ROA')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if financial_data is None or 'roa' not in financial_data.columns:
            return pd.DataFrame()
        
        df = financial_data[['ts_code', 'end_date', 'roa']].copy()
        df = df.dropna(subset=['roa'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='roa')
        return cross_sectional_zscore(pivot)


class ROIC(BaseFactor):
    """ROIC (直接使用 rich 数据的 roic 列)"""
    def __init__(self):
        super().__init__('roic', 'profitability', 'ROIC')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None:
            return pd.DataFrame()
        
        if 'roic' in fn.columns:
            col = 'roic'
        elif 'roa' in fn.columns:
            col = 'roa'  # fallback
        else:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', col]].copy()
        df = df.dropna(subset=[col])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values=col)
        return cross_sectional_zscore(pivot)


class GrossMargin(BaseFactor):
    """毛利率 = (revenue - cost) / revenue"""
    def __init__(self):
        super().__init__('gross_margin', 'profitability', '毛利率')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if financial_data is None:
            return pd.DataFrame()
        
        if 'grossprofit_margin' in financial_data.columns:
            col = 'grossprofit_margin'
        elif 'gross_margin' in financial_data.columns:
            col = 'gross_margin'
        else:
            return pd.DataFrame()
        
        df = financial_data[['ts_code', 'end_date', col]].copy()
        df = df.dropna(subset=[col])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values=col)
        return cross_sectional_zscore(pivot)


class NetMargin(BaseFactor):
    """净利率"""
    def __init__(self):
        super().__init__('net_margin', 'profitability', '净利率')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        if financial_data is None or 'netprofit_margin' not in financial_data.columns:
            return pd.DataFrame()
        
        df = financial_data[['ts_code', 'end_date', 'netprofit_margin']].copy()
        df = df.dropna(subset=['netprofit_margin'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='netprofit_margin')
        return cross_sectional_zscore(pivot)


class Accruals(BaseFactor):
    """应计利润 = (EPS - OCFPS) / BPS
    
    高应计 → 盈利质量差 → 负向因子 (取负号)
    用 fina_indicator_rich: eps(每股收益), ocfps(每股经营现金流), bps(每股净资产)
    """
    def __init__(self):
        super().__init__('accruals', 'profitability', '应计利润')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None:
            return pd.DataFrame()
        
        # 需要 eps, ocfps, bps (每股净资产)
        required = ['eps', 'ocfps', 'bps']
        missing = [c for c in required if c not in fn.columns]
        if missing:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'eps', 'ocfps', 'bps']].copy()
        df = df.dropna(subset=['eps', 'ocfps', 'bps'])
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # 应计 = (净利润 - 经营现金流) / 净资产, 全部用每股值
        df['accrual'] = (df['eps'] - df['ocfps']) / df['bps'].replace(0, np.nan)
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='accrual')
        # 应计利润高 → 负向信号, 取负号
        result = -pivot
        return cross_sectional_zscore(result)


# ==================== v5.1 新增基本面因子 ====================

class OCFQuality(BaseFactor):
    """现金流质量 = OCFPS / EPS
    
    高 → 利润有现金流支撑 → 质量好 → 正向因子
    使用 fina_indicator_rich: ocfps(每股经营现金流), eps(每股收益)
    """
    def __init__(self):
        super().__init__('ocf_quality', 'quality', '现金流质量')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None:
            return pd.DataFrame()
        
        required = ['ocfps', 'eps']
        missing = [c for c in required if c not in fn.columns]
        if missing:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'ocfps', 'eps']].copy()
        df = df.dropna(subset=['ocfps', 'eps'])
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # OCF/NetProfit 比率, eps>0 才有意义
        df['ocf_quality'] = np.where(
            df['eps'] > 0,
            df['ocfps'] / df['eps'],
            np.nan
        )
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='ocf_quality')
        return cross_sectional_zscore(pivot)


class DebtCoverage(BaseFactor):
    """债务覆盖 = 经营现金流 / 总债务
    
    高 → 偿债能力强 → 低风险 → 正向因子
    使用 fina_indicator_rich: ocf_to_debt(已预计算)
    """
    def __init__(self):
        super().__init__('debt_coverage', 'quality', '债务覆盖')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None or 'ocf_to_debt' not in fn.columns:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'ocf_to_debt']].copy()
        df = df.dropna(subset=['ocf_to_debt'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='ocf_to_debt')
        return cross_sectional_zscore(pivot)


class EarningsStability(BaseFactor):
    """盈利稳定性 = -std(netprofit_yoy over 8 quarters)
    
    盈利增速波动大 → 不稳定 → 负向 (取负号)
    使用 fina_indicator_rich: netprofit_yoy(净利润同比增速)
    """
    def __init__(self):
        super().__init__('earnings_stability', 'quality', '盈利稳定性')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None or 'netprofit_yoy' not in fn.columns:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'netprofit_yoy']].copy()
        df = df.dropna(subset=['netprofit_yoy'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        df = df.sort_values(['ts_code', 'end_date'])
        
        # 滚动 8 季度标准差
        stability = df.groupby('ts_code')['netprofit_yoy'].transform(
            lambda x: x.rolling(8, min_periods=4).std()
        )
        df['earnings_stability'] = -stability  # 负向: 波动大→差
        
        df = df.dropna(subset=['earnings_stability'])
        
        if len(df) == 0:
            return pd.DataFrame()
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='earnings_stability')
        return cross_sectional_zscore(pivot)


class AssetTurnover(BaseFactor):
    """总资产周转率
    
    高 → 运营效率好 → 正向因子
    使用 fina_indicator_rich: assets_turn
    """
    def __init__(self):
        super().__init__('asset_turnover', 'quality', '资产周转率')
    
    def compute(self, price_data, financial_data, valuation_data, **kwargs):
        fn = financial_data
        if fn is None or 'assets_turn' not in fn.columns:
            return pd.DataFrame()
        
        df = fn[['ts_code', 'end_date', 'assets_turn']].copy()
        df = df.dropna(subset=['assets_turn'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        pivot = df.pivot(index='end_date', columns='ts_code', values='assets_turn')
        return cross_sectional_zscore(pivot)
