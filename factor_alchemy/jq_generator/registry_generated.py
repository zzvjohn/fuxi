# -*- coding: utf-8 -*-
# Auto-generated JQ factor registry entries (placeholder)
# 90 factors with placeholder compute functions

REGISTRY_ADDITIONS = {
    'abnormal_turnover': {
        'name': 'abnormal_turnover',
        'local_source': 'factors/turnover.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'turnover',
        'jq_compute_function': '''
def compute_abnormal_turnover(stocks, price_data):
    """Auto-generated placeholder for abnormal_turnover."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'abnormal_turnover_neut': {
        'name': 'abnormal_turnover_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_abnormal_turnover_neut(stocks, price_data):
    """Auto-generated placeholder for abnormal_turnover_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'accruals': {
        'name': 'accruals',
        'local_source': 'factors/profitability.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'profitability',
        'jq_compute_function': '''
def compute_accruals(stocks, price_data):
    """Auto-generated placeholder for accruals."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'amihud_illiq': {
        'name': 'amihud_illiq',
        'local_source': 'factors/liquidity.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'liquidity',
        'jq_compute_function': '''
def compute_amihud_illiq(stocks, price_data):
    """Auto-generated placeholder for amihud_illiq."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'asset_growth': {
        'name': 'asset_growth',
        'local_source': 'factors/growth.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth',
        'jq_compute_function': '''
def compute_asset_growth(stocks, price_data):
    """Auto-generated placeholder for asset_growth."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'asset_turnover': {
        'name': 'asset_turnover',
        'local_source': 'factors/quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'quality',
        'jq_compute_function': '''
def compute_asset_turnover(stocks, price_data):
    """Auto-generated placeholder for asset_turnover."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'atr_14': {
        'name': 'atr_14',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close', 'high', 'low'],
        'jq_window': 30,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
                'jq_compute_function': '''
def compute_atr_14(stocks, price_data):
    """ATR(14) normalized by close, negated. Low ATR = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        if len(h) < 16 or len(l) < 16 or len(c) < 16:
            continue
        tr = np.zeros(len(c) - 1)
        for t in range(1, len(c)):
            tr[t - 1] = max(
                h[t] - l[t],
                abs(h[t] - c[t - 1]),
                abs(l[t] - c[t - 1]))
        if len(tr) < 14:
            continue
        alpha = 2.0 / 15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1 - alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'atr_14_neut': {
        'name': 'atr_14_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_atr_14_neut(stocks, price_data):
    """Auto-generated placeholder for atr_14_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'attention_decay': {
        'name': 'attention_decay',
        'local_source': 'factors/behavioral.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'behavioral',
        'jq_compute_function': '''
def compute_attention_decay(stocks, price_data):
    """Auto-generated placeholder for attention_decay."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'avg_turnover_1m': {
        'name': 'avg_turnover_1m',
        'local_source': 'factors/turnover.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'turnover',
        'jq_compute_function': '''
def compute_avg_turnover_1m(stocks, price_data):
    """Auto-generated placeholder for avg_turnover_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'avg_turnover_1m_neut': {
        'name': 'avg_turnover_1m_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_avg_turnover_1m_neut(stocks, price_data):
    """Auto-generated placeholder for avg_turnover_1m_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'avg_turnover_3m': {
        'name': 'avg_turnover_3m',
        'local_source': 'factors/turnover.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'turnover',
        'jq_compute_function': '''
def compute_avg_turnover_3m(stocks, price_data):
    """Auto-generated placeholder for avg_turnover_3m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'bargaining_power_proxy': {
        'name': 'bargaining_power_proxy',
        'local_source': 'factors/growth_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth_quality',
        'jq_compute_function': '''
def compute_bargaining_power_proxy(stocks, price_data):
    """Auto-generated placeholder for bargaining_power_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'bear_power': {
        'name': 'bear_power',
        'local_source': 'factors/pattern.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'pattern',
        'jq_compute_function': '''
def compute_bear_power(stocks, price_data):
    """Auto-generated placeholder for bear_power."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'beta': {
        'name': 'beta',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_beta(stocks, price_data):
    """Auto-generated placeholder for beta."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'bias_20': {
        'name': 'bias_20',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_bias_20(stocks, price_data):
    """Auto-generated placeholder for bias_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'boll_pct_b': {
        'name': 'boll_pct_b',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_boll_pct_b(stocks, price_data):
    """Auto-generated placeholder for boll_pct_b."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'bp': {
        'name': 'bp',
        'local_source': 'factors/value.py',
        'local_formula': 'unknown',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'value',
                'jq_compute_function': '''
def compute_bp(stocks, context):
    """Book-to-Price = 1/pb_ratio. Higher BP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pb_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pb = row.get('pb_ratio')
        if code in stock_idx and pb is not None and not np.isnan(pb) and pb > 0:
            arr[stock_idx[code]] = 1.0 / pb
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'bp_neut': {
        'name': 'bp_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_bp_neut(stocks, price_data):
    """Auto-generated placeholder for bp_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'bull_power': {
        'name': 'bull_power',
        'local_source': 'factors/pattern.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'pattern',
        'jq_compute_function': '''
def compute_bull_power(stocks, price_data):
    """Auto-generated placeholder for bull_power."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'capital_efficiency_proxy': {
        'name': 'capital_efficiency_proxy',
        'local_source': 'factors/growth_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth_quality',
        'jq_compute_function': '''
def compute_capital_efficiency_proxy(stocks, price_data):
    """Auto-generated placeholder for capital_efficiency_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'cashflow_matching_proxy': {
        'name': 'cashflow_matching_proxy',
        'local_source': 'factors/growth_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth_quality',
        'jq_compute_function': '''
def compute_cashflow_matching_proxy(stocks, price_data):
    """Auto-generated placeholder for cashflow_matching_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'cci_14': {
        'name': 'cci_14',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_cci_14(stocks, price_data):
    """Auto-generated placeholder for cci_14."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'cfp': {
        'name': 'cfp',
        'local_source': 'factors/value.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'value',
        'jq_compute_function': '''
def compute_cfp(stocks, price_data):
    """Auto-generated placeholder for cfp."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'davol_20': {
        'name': 'davol_20',
        'local_source': 'factors/sentiment.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'sentiment',
        'jq_compute_function': '''
def compute_davol_20(stocks, price_data):
    """Auto-generated placeholder for davol_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'davol_20_neut': {
        'name': 'davol_20_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_davol_20_neut(stocks, price_data):
    """Auto-generated placeholder for davol_20_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'debt_coverage': {
        'name': 'debt_coverage',
        'local_source': 'factors/quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'quality',
        'jq_compute_function': '''
def compute_debt_coverage(stocks, price_data):
    """Auto-generated placeholder for debt_coverage."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'dollar_vol_20d_neut': {
        'name': 'dollar_vol_20d_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_dollar_vol_20d_neut(stocks, price_data):
    """Auto-generated placeholder for dollar_vol_20d_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'dollar_vol_stability': {
        'name': 'dollar_vol_stability',
        'local_source': 'factors/liquidity.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'liquidity',
        'jq_compute_function': '''
def compute_dollar_vol_stability(stocks, price_data):
    """Auto-generated placeholder for dollar_vol_stability."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'downside_vol': {
        'name': 'downside_vol',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_downside_vol(stocks, price_data):
    """Auto-generated placeholder for downside_vol."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'downside_vol_neut': {
        'name': 'downside_vol_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_downside_vol_neut(stocks, price_data):
    """Auto-generated placeholder for downside_vol_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'dp': {
        'name': 'dp',
        'local_source': 'factors/value.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'value',
        'jq_compute_function': '''
def compute_dp(stocks, price_data):
    """Auto-generated placeholder for dp."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'earnings_consistency_proxy': {
        'name': 'earnings_consistency_proxy',
        'local_source': 'factors/fundamental_quality_proxy.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'fundamental_quality_proxy',
        'jq_compute_function': '''
def compute_earnings_consistency_proxy(stocks, price_data):
    """Auto-generated placeholder for earnings_consistency_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'earnings_quality_proxy': {
        'name': 'earnings_quality_proxy',
        'local_source': 'factors/growth_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth_quality',
        'jq_compute_function': '''
def compute_earnings_quality_proxy(stocks, price_data):
    """Auto-generated placeholder for earnings_quality_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'earnings_season_vol_div': {
        'name': 'earnings_season_vol_div',
        'local_source': 'factors/advanced_technical.py (EarningsSeasonVolDiv)',
        'local_formula': '5日窗口 价变vs量变 Spearman秩相关 取负, 5日平滑; 量价背离=正向',
        'jq_data': 'price',
        'jq_fields': ['close', 'volume'],
        'jq_window': 15,
        'direction': 'positive',
        'direction_note': '中报窗口量价背离→机构吸筹→正向(价涨量缩的背离度)',
        'verified': True,
        'category': 'earnings_season_divergence',
        'jq_compute_function': '''
def compute_earnings_season_vol_div(stocks, price_data):
    """中报窗口量价背离度: 价变与量变秩相关取负(背离→正向), 5日平滑.

    对齐本地 EarningsSeasonVolDiv:
      pct = close.pct_change(); vol = volume.pct_change()
      5日窗口内 (pct, vol) 的 Spearman秩相关取负 → 价涨量缩(背离)为正
      取最近5个窗口的均值作为截面信号
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    win = 5
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.asarray(close_d[s], dtype=float)
        v = np.asarray(vol_d[s], dtype=float)
        if len(c) < 15 or len(v) < 15:
            continue
        c = c[-20:]; v = v[-20:]
        pct = np.diff(c) / c[:-1]
        denom = np.where(v[:-1] != 0, v[:-1], np.nan)
        vc = np.diff(v) / denom
        pct = np.nan_to_num(pct)
        vc = np.nan_to_num(vc)
        L = len(pct)
        rho = np.full(L, np.nan)
        for t in range(win - 1, L):
            x = pct[t - win + 1:t + 1]
            y = vc[t - win + 1:t + 1]
            rx = x.argsort().argsort().astype(float)
            ry = y.argsort().argsort().astype(float)
            mr = rx.mean(); my = ry.mean()
            num = ((rx - mr) * (ry - my)).sum()
            den = np.sqrt(((rx - mr) ** 2).sum() * ((ry - my) ** 2).sum())
            rho[t] = (-num / den) if (den > 0 and np.isfinite(den)) else 0.0
        last5 = rho[-win:]
        val = np.nanmean(last5)
        if np.isfinite(val):
            arr[i] = val
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'earnings_stability': {
        'name': 'earnings_stability',
        'local_source': 'factors/profitability.py:330 (EarningsStability)',
        'local_formula': '-rolling(8q)std(netprofit_yoy) [稳定=低波动=正向, 本地取负]',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '盈利同比增速波动小→盈利稳定→正向. JQ维护inc_net_profit_year_on_year近40周历史, 取std取负对齐本地',
        'verified': True,
        'category': 'quality',
        'jq_compute_function': '''
def compute_earnings_stability(stocks, context):
    """盈利稳定性 = -std(inc_net_profit_year_on_year 历史). 波动小=稳定=正向.
    对齐本地 EarningsStability(-rolling(8q)std(netprofit_yoy)).
    预热: 首次调用用过去~10个季度末快照填充历史, 保证第1个交易日即有效."""
    import calendar
    n = len(stocks)
    arr = np.full(n, np.nan)
    if not hasattr(g, 'earnings_yoy_hist') or g.earnings_yoy_hist is None:
        g.earnings_yoy_hist = {}

    def _snap(dt):
        snap = {}
        try:
            end = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else dt
            q = query(indicator.code, indicator.inc_net_profit_year_on_year).filter(
                indicator.code.in_(stocks))
            df = get_fundamentals(q, date=end)
        except Exception as e:
            if g.debug:
                log.error('[earnings_stability] get_fundamentals FAILED: %s' % str(e))
            return snap
        if df is None or len(df) == 0:
            return snap
        for _, row in df.iterrows():
            code = row['code']
            val = row.get('inc_net_profit_year_on_year')
            if val is not None and np.isfinite(val):
                snap[code] = float(val)
        return snap

    # 首次预热: 过去~10个季度末快照
    if not getattr(g, 'earnings_yoy_warmed', False):
        end = context.previous_date
        y, m = end.year, end.month
        q = (m - 1) // 3
        qe_month = q * 3 + 3
        qe_y = y
        if qe_month > 12:
            qe_month -= 12; qe_y += 1
        ends = [datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1])]
        for _ in range(9):
            qe_month -= 3
            if qe_month < 1:
                qe_month += 12; qe_y -= 1
            ends.append(datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1]))
        for d in ends:
            s0 = _snap(d)
            for s in stocks:
                if s in s0:
                    h = g.earnings_yoy_hist.setdefault(s, [])
                    h.append(s0[s])
                    if len(h) > 40:
                        h[:] = h[-40:]
        g.earnings_yoy_warmed = True

    # 当前快照
    snap = _snap(context.previous_date)
    for s in stocks:
        if s in snap:
            h = g.earnings_yoy_hist.setdefault(s, [])
            h.append(snap[s])
            if len(h) > 40:
                h[:] = h[-40:]
    stock_idx = {s: i for i, s in enumerate(stocks)}
    valid = np.zeros(n, dtype=bool)
    for s in snap:
        if s in stock_idx:
            h = g.earnings_yoy_hist.get(s, [])
            hv = [x for x in h if x is not None and np.isfinite(x)]
            if len(hv) >= 4:
                arr[stock_idx[s]] = -np.std(hv, ddof=1)
                valid[stock_idx[s]] = True
    return arr, valid
        ''',
    },
    'earnings_volume_drift': {
        'name': 'earnings_volume_drift',
        'local_source': 'factors/advanced_technical.py (EarningsVolumeDrift)',
        'local_formula': 'ret_5d.clip(-0.3,0.3)*(-vol_chg_5d).clip(-1,2); only ret_5d>0; 缩量上涨=筹码锁定',
        'jq_data': 'price',
        'jq_fields': ['close', 'volume'],
        'jq_window': 15,
        'direction': 'positive',
        'direction_note': '缩量上涨→筹码锁定→正向(5日收益×量能变化取负, 仅保留上涨段)',
        'verified': True,
        'category': 'earnings_anomaly',
        'jq_compute_function': '''
def compute_earnings_volume_drift(stocks, price_data):
    """业绩窗量价漂移: 缩量上涨=筹码锁定→正向. 仅保留上涨段.

    对齐本地 EarningsVolumeDrift:
      ret_5d   = close/close.shift(5) - 1                (取最新截面)
      vol_chg  = MA5(volume)/MA5(volume).shift(5) - 1
      signal   = ret_5d.clip(-0.3,0.3) * (-vol_chg).clip(-1,2)
      signal   = signal.where(ret_5d > 0, 0)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.asarray(close_d[s], dtype=float)
        v = np.asarray(vol_d[s], dtype=float)
        if len(c) < 12 or len(v) < 12:
            continue
        c = c[-12:]; v = v[-12:]
        c0 = c[-6]
        if not np.isfinite(c0) or c0 == 0:
            continue
        r = c[-1] / c0 - 1.0
        if not np.isfinite(r):
            continue
        m5 = np.mean(v[-5:])
        m5p = np.mean(v[-11:-6]) if len(v) >= 11 else np.nan
        if not np.isfinite(m5p) or m5p == 0:
            vc = 0.0
        else:
            vc = m5 / m5p - 1.0
        sig = np.clip(r, -0.3, 0.3) * np.clip(-vc, -1.0, 2.0)
        if r <= 0:
            sig = 0.0
        arr[i] = sig
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ep': {
        'name': 'ep',
        'local_source': 'factors/value.py',
        'local_formula': 'unknown',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'value',
                'jq_compute_function': '''
def compute_ep(stocks, context):
    """Earnings-to-Price = 1/pe_ratio. Higher EP = value = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.pe_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        pe = row.get('pe_ratio')
        if code in stock_idx and pe is not None and not np.isnan(pe) and pe > 0:
            arr[stock_idx[code]] = 1.0 / pe
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'high_52w_dist': {
        'name': 'high_52w_dist',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
        'jq_compute_function': '''
def compute_high_52w_dist(stocks, price_data):
    """Auto-generated placeholder for high_52w_dist."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'high_52w_rank': {
        'name': 'high_52w_rank',
        'local_source': 'factors/pattern.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'pattern',
        'jq_compute_function': '''
def compute_high_52w_rank(stocks, price_data):
    """Auto-generated placeholder for high_52w_rank."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'high_low_range': {
        'name': 'high_low_range',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close', 'high', 'low'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
                'jq_compute_function': '''
def compute_high_low_range(stocks, price_data):
    """20d avg (H-L)/close, negated. Narrow range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        min_len = min(len(h), len(l), len(c))
        if min_len < 10:
            continue
        h_arr = h[-min_len:]
        l_arr = l[-min_len:]
        c_arr = c[-min_len:]
        daily_range = (h_arr - l_arr) / c_arr
        valid_range = daily_range[np.isfinite(daily_range) & (c_arr > 0)]
        if len(valid_range) < 10:
            continue
        roll = valid_range[-20:] if len(valid_range) >= 20 else valid_range
        avg_range = np.nanmean(roll)
        if np.isnan(avg_range):
            continue
        arr[i] = -avg_range
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'idio_vol': {
        'name': 'idio_vol',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_idio_vol(stocks, price_data):
    """Auto-generated placeholder for idio_vol."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'intraday_reversal': {
        'name': 'intraday_reversal',
        'local_source': 'factors/return_decomposition.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'return_decomposition',
        'jq_compute_function': '''
def compute_intraday_reversal(stocks, price_data):
    """Auto-generated placeholder for intraday_reversal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'kurtosis_20': {
        'name': 'kurtosis_20',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
        'jq_compute_function': '''
def compute_kurtosis_20(stocks, price_data):
    """Auto-generated placeholder for kurtosis_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ln_circulating_mcap': {
        'name': 'ln_circulating_mcap',
        'local_source': 'factors/size.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'size',
        'jq_compute_function': '''
def compute_ln_circulating_mcap(stocks, price_data):
    """Auto-generated placeholder for ln_circulating_mcap."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ln_mcap': {
        'name': 'ln_mcap',
        'local_source': 'factors/size.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'size',
        'jq_compute_function': '''
def compute_ln_mcap(stocks, price_data):
    """Auto-generated placeholder for ln_mcap."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'max_drawdown_duration': {
        'name': 'max_drawdown_duration',
        'local_source': 'factors/extreme_events.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'extreme_events',
        'jq_compute_function': '''
def compute_max_drawdown_duration(stocks, price_data):
    """Auto-generated placeholder for max_drawdown_duration."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'max_ret_1m': {
        'name': 'max_ret_1m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_max_ret_1m(stocks, price_data):
    """Auto-generated placeholder for max_ret_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'max_ret_1m_neut': {
        'name': 'max_ret_1m_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_max_ret_1m_neut(stocks, price_data):
    """Auto-generated placeholder for max_ret_1m_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ocf_quality': {
        'name': 'ocf_quality',
        'local_source': 'factors/quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'quality',
        'jq_compute_function': '''
def compute_ocf_quality(stocks, price_data):
    """Auto-generated placeholder for ocf_quality."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'opening_gap_momentum': {
        'name': 'opening_gap_momentum',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_opening_gap_momentum(stocks, price_data):
    """Auto-generated placeholder for opening_gap_momentum."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'operational_efficiency_proxy': {
        'name': 'operational_efficiency_proxy',
        'local_source': 'factors/growth_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'growth_quality',
        'jq_compute_function': '''
def compute_operational_efficiency_proxy(stocks, price_data):
    """Auto-generated placeholder for operational_efficiency_proxy."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'panic_selling': {
        'name': 'panic_selling',
        'local_source': 'factors/behavioral.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'behavioral',
        'jq_compute_function': '''
def compute_panic_selling(stocks, price_data):
    """Auto-generated placeholder for panic_selling."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'plrc_12': {
        'name': 'plrc_12',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_plrc_12(stocks, price_data):
    """Auto-generated placeholder for plrc_12."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'post_earnings_stability': {
        'name': 'post_earnings_stability',
        'local_source': 'factors/earnings_anomaly.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'earnings_anomaly',
        'jq_compute_function': '''
def compute_post_earnings_stability(stocks, price_data):
    """Auto-generated placeholder for post_earnings_stability."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'price_1m': {
        'name': 'price_1m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_price_1m(stocks, price_data):
    """Auto-generated placeholder for price_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'psy_12': {
        'name': 'psy_12',
        'local_source': 'factors/sentiment.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'sentiment',
        'jq_compute_function': '''
def compute_psy_12(stocks, price_data):
    """Auto-generated placeholder for psy_12."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'range_consistency': {
        'name': 'range_consistency',
        'local_source': 'factors/price_pattern.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['high', 'low'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'price_pattern',
                'jq_compute_function': '''
def compute_range_consistency(stocks, price_data):
    """Range CV over 20d, negated. Consistent range = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        min_len = min(len(h), len(l))
        if min_len < 10:
            continue
        rng = np.abs(h[-min_len:] - l[-min_len:])
        roll = rng[-20:] if len(rng) >= 20 else rng
        std_r = np.nanstd(roll)
        mean_r = np.nanmean(roll)
        if mean_r == 0:
            continue
        arr[i] = -std_r / mean_r
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'rank_1m': {
        'name': 'rank_1m',
        'local_source': 'factors/pattern.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'pattern',
        'jq_compute_function': '''
def compute_rank_1m(stocks, price_data):
    """Auto-generated placeholder for rank_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_12m': {
        'name': 'ret_12m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_ret_12m(stocks, price_data):
    """Auto-generated placeholder for ret_12m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_1m': {
        'name': 'ret_1m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_ret_1m(stocks, price_data):
    """Auto-generated placeholder for ret_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_1m_skip1m': {
        'name': 'ret_1m_skip1m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_ret_1m_skip1m(stocks, price_data):
    """Auto-generated placeholder for ret_1m_skip1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_3m': {
        'name': 'ret_3m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_ret_3m(stocks, price_data):
    """Auto-generated placeholder for ret_3m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_3m_neut': {
        'name': 'ret_3m_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_ret_3m_neut(stocks, price_data):
    """Auto-generated placeholder for ret_3m_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'ret_6m': {
        'name': 'ret_6m',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 135,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
                'jq_compute_function': '''
def compute_ret_6m(stocks, price_data):
    """6-month return reversal. Negative past return = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c) & (c > 0)]
        if len(c_clean) < 131:
            continue
        ret = c_clean[-1] / c_clean[-126] - 1.0
        arr[i] = -ret
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'rev_growth_yoy': {
        'name': 'rev_growth_yoy',
        'local_source': 'factors/growth.py (RevGrowthYoY)',
        'local_formula': 'or_yoy = 营业收入同比增长率 (indicator.inc_revenue_year_on_year)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '营收同比增长越高→成长越强→正向',
        'verified': True,
        'category': 'growth',
        'jq_compute_function': '''
def compute_rev_growth_yoy(stocks, context):
    """营收同比增长率 (or_yoy). 取最新财报快照, 值越高=成长越强=正向."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_revenue_year_on_year).filter(
            indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('inc_revenue_year_on_year')
        if code in stock_idx and val is not None and np.isfinite(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'roc_20': {
        'name': 'roc_20',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_roc_20(stocks, price_data):
    """Auto-generated placeholder for roc_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'roic': {
        'name': 'roic',
        'local_source': 'factors/profitability.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'profitability',
        'jq_compute_function': '''
def compute_roic(stocks, price_data):
    """Auto-generated placeholder for roic."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'rsi_14': {
        'name': 'rsi_14',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_rsi_14(stocks, price_data):
    """Auto-generated placeholder for rsi_14."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'scc_network_centrality': {
        'name': 'scc_network_centrality',
        'local_source': 'factors/network.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'network',
        'jq_compute_function': '''
def compute_scc_network_centrality(stocks, price_data):
    """Auto-generated placeholder for scc_network_centrality."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'sharpe_20': {
        'name': 'sharpe_20',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
        'jq_compute_function': '''
def compute_sharpe_20(stocks, price_data):
    """Auto-generated placeholder for sharpe_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'short_rev_5d': {
        'name': 'short_rev_5d',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_short_rev_5d(stocks, price_data):
    """Auto-generated placeholder for short_rev_5d."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'skew_1m': {
        'name': 'skew_1m',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
        'jq_compute_function': '''
def compute_skew_1m(stocks, price_data):
    """Auto-generated placeholder for skew_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'skewness_20': {
        'name': 'skewness_20',
        'local_source': 'factors/distribution.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'distribution',
        'jq_compute_function': '''
def compute_skewness_20(stocks, price_data):
    """Auto-generated placeholder for skewness_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'sp': {
        'name': 'sp',
        'local_source': 'factors/value.py (S/P)',
        'local_formula': '1 / valuation.ps_ratio  (市销率倒数=销售/价格比)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': 'S/P越高=越便宜=价值→正向',
        'verified': True,
        'category': 'value',
        'jq_compute_function': '''
def compute_sp(stocks, context):
    """S/P = 1/ps_ratio. 销售/价格比, 越高越便宜=价值正向信号."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.ps_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        ps = row.get('ps_ratio')
        if code in stock_idx and ps is not None and np.isfinite(ps) and ps > 0:
            arr[stock_idx[code]] = 1.0 / ps
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'streak': {
        'name': 'streak',
        'local_source': 'factors/momentum.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'momentum',
        'jq_compute_function': '''
def compute_streak(stocks, price_data):
    """Auto-generated placeholder for streak."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'trend_persistence_score': {
        'name': 'trend_persistence_score',
        'local_source': 'factors/trend_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'trend_quality',
        'jq_compute_function': '''
def compute_trend_persistence_score(stocks, price_data):
    """Auto-generated placeholder for trend_persistence_score."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'trend_smoothness': {
        'name': 'trend_smoothness',
        'local_source': 'factors/trend_quality.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'trend_quality',
        'jq_compute_function': '''
def compute_trend_smoothness(stocks, price_data):
    """Auto-generated placeholder for trend_smoothness."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'turnover_to_vol': {
        'name': 'turnover_to_vol',
        'local_source': 'factors/liquidity.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'liquidity',
        'jq_compute_function': '''
def compute_turnover_to_vol(stocks, price_data):
    """Auto-generated placeholder for turnover_to_vol."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'tvma_20': {
        'name': 'tvma_20',
        'local_source': 'factors/volume_structure.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volume_structure',
        'jq_compute_function': '''
def compute_tvma_20(stocks, price_data):
    """Auto-generated placeholder for tvma_20."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vol_1m': {
        'name': 'vol_1m',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_vol_1m(stocks, price_data):
    """Auto-generated placeholder for vol_1m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vol_1m_neut': {
        'name': 'vol_1m_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_vol_1m_neut(stocks, price_data):
    """Auto-generated placeholder for vol_1m_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vol_3m': {
        'name': 'vol_3m',
        'local_source': 'factors/volatility.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility',
        'jq_compute_function': '''
def compute_vol_3m(stocks, price_data):
    """Auto-generated placeholder for vol_3m."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vol_3m_neut': {
        'name': 'vol_3m_neut',
        'local_source': 'factors/neutralized.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'neutralized',
        'jq_compute_function': '''
def compute_vol_3m_neut(stocks, price_data):
    """Auto-generated placeholder for vol_3m_neut."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'volatility_of_volatility': {
        'name': 'volatility_of_volatility',
        'local_source': 'factors/volatility_structure.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volatility_structure',
        'jq_compute_function': '''
def compute_volatility_of_volatility(stocks, price_data):
    """Auto-generated placeholder for volatility_of_volatility."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'volume_ratio': {
        'name': 'volume_ratio',
        'local_source': 'factors/turnover.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'turnover',
        'jq_compute_function': '''
def compute_volume_ratio(stocks, price_data):
    """Auto-generated placeholder for volume_ratio."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'volume_stability': {
        'name': 'volume_stability',
        'local_source': 'factors/volume_structure.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['volume'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'volume_structure',
                'jq_compute_function': '''
def compute_volume_stability(stocks, price_data):
    """Volume CV over 20d, negated. Stable volume = positive signal."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd:
            continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 10:
            continue
        roll = v_clean[-20:] if len(v_clean) >= 20 else v_clean
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        arr[i] = -std_v / mean_v
    valid = np.isfinite(arr)
    return arr, valid
                ''',
    },
    'vpt': {
        'name': 'vpt',
        'local_source': 'factors/liquidity.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'liquidity',
        'jq_compute_function': '''
def compute_vpt(stocks, price_data):
    """Auto-generated placeholder for vpt."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vr_26': {
        'name': 'vr_26',
        'local_source': 'factors/sentiment.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'sentiment',
        'jq_compute_function': '''
def compute_vr_26(stocks, price_data):
    """Auto-generated placeholder for vr_26."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
    'vroc_12': {
        'name': 'vroc_12',
        'local_source': 'factors/sentiment.py',
        'local_formula': 'unknown',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 60,
        'direction': 'positive',
        'direction_note': 'auto-generated placeholder',
        'verified': False,
        'category': 'sentiment',
        'jq_compute_function': '''
def compute_vroc_12(stocks, price_data):
    """Auto-generated placeholder for vroc_12."""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    },
}
