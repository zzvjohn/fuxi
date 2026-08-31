"""
因子模块
"""
import pandas as pd
from pathlib import Path
from .base import BaseFactor, cross_sectional_zscore
from .size import LnMarketCap, LnCirculatingMarketCap
from .value import EP, BP, SP, CFP, DP
from .profitability import (PiotroskiFScore, ROE, ROA, ROIC, GrossMargin, NetMargin, Accruals,
                             OCFQuality, DebtCoverage, EarningsStability, AssetTurnover)
from .turnover import AvgTurnover1M, AvgTurnover3M, TurnoverStd, TurnoverChange, AbnormalTurnover
from .momentum import Ret1M, Ret3M, Ret6M, Ret12M, Ret1MSkip1M, MaxRet1M
from .volatility import Vol1M, Vol3M, DownsideVol, Beta, IdioVol
from .liquidity import AmihudIlliq, DollarVol20D, DollarVolStability, TurnoverCv20D, TurnoverToVol
from .growth import RevGrowthYoY, EarningsGrowthYoY, AssetGrowth
from .technical import (                              # ★ jqfactor 启发新因子 (7)
    Streak, RSI14, HighLowRange, VPT, ShortRev5D,
    BollPctB, VolumeRatio, Overnight5D, MinRet1M,
)
from .advanced_technical import (                       # ★ jqfactor 二次补充 (20 → 22 → 27)
    # 趋势延伸
    Bias20, ROC20, CCI14, PLRC12, Price1M,
    # 情绪因子
    VR26, VROC12, PSY12, MoneyFlow20, DAVOL20,
    # 收益分布
    Skewness20, Kurtosis20, Sharpe20, ATR14,
    # 形态因子
    BullPower, BearPower, High52WRank, Rank1M,
    # 成交量结构
    VMDiff, TVMA20,
    # Phase 2 独立因子
    High52WDist, Skew1M,
    # XQuant Ch9 通过因子 (第1-2轮)
    GapUp, PanicSelling,
    # 日频因子实验 v2 通过因子
    TrendPersistenceScore, AttentionDecay,
    # 吴先兴五维成长因子框架 (2026-06-20)
    EarningsQualityProxy, CashFlowMatchingProxy, CapitalEfficiencyProxy,
    OperationalEfficiencyProxy, BargainingPowerProxy,
    # 日频因子实验 v2 通过 (2026-06-21)
    VolumeStability,
    # 日频因子实验 v2 通过 (2026-06-21 round 2)
    VolumeClimaxReversal,
    # 日频因子实验 v2 破格注册 (2026-06-29)
    OpeningGapMomentum,
    # 日频因子实验 v2 通过 (2026-06-30)
    MaxDrawdownDuration,
    # 日频因子实验 v2 通过 (2026-07-01)
    AccrualQualityProxy,
    # 日频因子实验 v2 积压批量注册 (2026-07-03)
    IntradayReversal, RangeConsistency, VolatilityOfVolatility,
    RelativeSpreadProxy, TrendSmoothness,
    EarningsConsistencyProxy, RetOpen2DProxy,
    # 日频因子实验 v2 通过 (2026-07-13)
    EarningsSeasonVolDiv,
    # 日频因子实验 v2 通过 (2026-07-17)
    PostEarningsStability, EarningsVolumeDrift,
    # 日频因子实验 v2 通过 (2026-07-31)
    HarveySiddiqueCoskew,
)
# 因子注册表 (93个因子, 2026-07-17 +2个中报窗口因子)
ALL_FACTORS = {
    # 规模
    'ln_mcap': LnMarketCap,
    'ln_circulating_mcap': LnCirculatingMarketCap,
    # 价值
    'ep': EP, 'bp': BP, 'sp': SP, 'cfp': CFP, 'dp': DP,
    # 盈利
    'f_score': PiotroskiFScore,
    'roe': ROE, 'roa': ROA, 'roic': ROIC,
    'gross_margin': GrossMargin, 'net_margin': NetMargin, 'accruals': Accruals,
    # 换手
    'avg_turnover_1m': AvgTurnover1M, 'avg_turnover_3m': AvgTurnover3M,
    'turnover_std': TurnoverStd, 'turnover_change': TurnoverChange,
    'abnormal_turnover': AbnormalTurnover,
    # 动量
    'ret_1m': Ret1M, 'ret_3m': Ret3M, 'ret_6m': Ret6M, 'ret_12m': Ret12M,
    'ret_1m_skip1m': Ret1MSkip1M, 'max_ret_1m': MaxRet1M,
    # 波动
    'vol_1m': Vol1M, 'vol_3m': Vol3M, 'downside_vol': DownsideVol,
    'beta': Beta, 'idio_vol': IdioVol,
    # 流动性
    'amihud_illiq': AmihudIlliq, 'dollar_vol_20d': DollarVol20D,
    'dollar_vol_stability': DollarVolStability,
    'turnover_cv_20d': TurnoverCv20D,
    'turnover_to_vol': TurnoverToVol,
    # 成长
    'rev_growth_yoy': RevGrowthYoY, 'earnings_growth_yoy': EarningsGrowthYoY,
    'asset_growth': AssetGrowth,
    # ★ jqfactor 启发: 技术/量价因子 (7)
    'streak': Streak,
    'rsi_14': RSI14,
    'high_low_range': HighLowRange,
    'vpt': VPT,
    'short_rev_5d': ShortRev5D,
    'boll_pct_b': BollPctB,
    'volume_ratio': VolumeRatio,
    # ★ 跨周期稳健因子 (2026-07-26)
    'overnight_5d': Overnight5D,
    'min_ret_1m': MinRet1M,
    # ★ jqfactor 二次补充: 趋势延伸 (5)
    'bias_20': Bias20,
    'roc_20': ROC20,
    'cci_14': CCI14,
    'plrc_12': PLRC12,
    'price_1m': Price1M,
    # ★ jqfactor 二次补充: 情绪因子 (5)
    'vr_26': VR26,
    'vroc_12': VROC12,
    'psy_12': PSY12,
    'money_flow_20': MoneyFlow20,
    'davol_20': DAVOL20,
    # ★ jqfactor 二次补充: 收益分布 (4)
    'skewness_20': Skewness20,
    'kurtosis_20': Kurtosis20,
    'sharpe_20': Sharpe20,
    'atr_14': ATR14,
    # ★ jqfactor 二次补充: 形态因子 (4)
    'bull_power': BullPower,
    'bear_power': BearPower,
    'high_52w_rank': High52WRank,
    'rank_1m': Rank1M,
    # ★ jqfactor 二次补充: 成交量结构 (2)
    'vm_diff': VMDiff,
    'tvma_20': TVMA20,
    # ★ Phase 2 独立因子补充 (2)
    'high_52w_dist': High52WDist,
    'skew_1m': Skew1M,
    # ★ XQuant Ch9 通过因子 (第1-2轮, 2026-06-14)
    'gap_up': GapUp,
    'panic_selling': PanicSelling,
    # ★ 日频因子实验 v2 通过 (2026-06-15)
    'trend_persistence_score': TrendPersistenceScore,
    # ★ 日频因子实验 v2 通过 (2026-06-20)
    'attention_decay': AttentionDecay,
    # ★ 吴先兴五维成长因子框架 (2026-06-20)
    'earnings_quality_proxy': EarningsQualityProxy,
    'cashflow_matching_proxy': CashFlowMatchingProxy,
    'capital_efficiency_proxy': CapitalEfficiencyProxy,
    'operational_efficiency_proxy': OperationalEfficiencyProxy,
    'bargaining_power_proxy': BargainingPowerProxy,
    # ★ 日频因子实验 v2 通过 (2026-06-21)
    'volume_stability': VolumeStability,
    # ★ 日频因子实验 v2 通过 (2026-06-21 round 2)
    'volume_climax_reversal': VolumeClimaxReversal,
    # ★ 日频因子实验 v2 破格注册 (2026-06-29)
    'opening_gap_momentum': OpeningGapMomentum,
    # ★ 日频因子实验 v2 通过 (2026-06-30)
    'max_drawdown_duration': MaxDrawdownDuration,
    # ★ 日频因子实验 v2 通过 (2026-07-01)
    'accrual_quality_proxy': AccrualQualityProxy,
    # ★ v5.1 基本面 Alpha 因子 (4)
    'ocf_quality': OCFQuality,
    'debt_coverage': DebtCoverage,
    'earnings_stability': EarningsStability,
    'asset_turnover': AssetTurnover,
    # ★ 日频因子实验 v2 积压批量注册 (2026-07-03)
    'intraday_reversal': IntradayReversal,
    'range_consistency': RangeConsistency,
    'volatility_of_volatility': VolatilityOfVolatility,
    'relative_spread_proxy': RelativeSpreadProxy,
    'trend_smoothness': TrendSmoothness,
    'earnings_consistency_proxy': EarningsConsistencyProxy,
    'ret_open_2d_proxy': RetOpen2DProxy,
    # ★ 日频因子实验 v2 通过 (2026-07-13)
    'earnings_season_vol_div': EarningsSeasonVolDiv,
    # ★ 日频因子实验 v2 通过 (2026-07-17)
    'post_earnings_stability': PostEarningsStability,
    'earnings_volume_drift': EarningsVolumeDrift,
    # ★ Stage 2 通过 (2026-07-31)
    'harvey_siddique_coskew': HarveySiddiqueCoskew,
}

# ===== Ridge-neutralized factors (pre-computed CSVs) =====
class PrecomputedFactor(BaseFactor):
    """Wrapper for pre-computed factor CSVs (e.g., Ridge-neutralized)"""
    def __init__(self):
        super().__init__(name=self.name, category='neutralized', label=self.name)
    
    def compute(self, price_data=None, financial_data=None, valuation_data=None):
        csv_path = Path(__file__).parent.parent / 'output' / f'factor_{self.name}.csv'
        if not csv_path.exists():
            csv_path = Path(__file__).parent.parent / 'output' / 'icir_biased' / f'factor_{self.name}.csv'
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

NEUT_FACTORS = [
    'dollar_vol_20d_neut', 'davol_20_neut', 'abnormal_turnover_neut',
    'bp_neut', 'atr_14_neut', 'vol_1m_neut', 'ret_3m_neut',
    'avg_turnover_1m_neut', 'vol_3m_neut', 'max_ret_1m_neut', 'downside_vol_neut',
]
for nf in NEUT_FACTORS:
    # Dynamically create a subclass with the correct name
    cls = type(f'Precomputed_{nf}', (PrecomputedFactor,), {'name': nf})
    ALL_FACTORS[nf] = cls


def get_factor_instance(name):
    """获取因子实例"""
    if name not in ALL_FACTORS:
        raise ValueError(f"未知因子: {name}")
    return ALL_FACTORS[name]()
