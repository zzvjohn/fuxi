#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成 JQ_FACTOR_REGISTRY 缺失映射
====================================
自动扫描 config.FACTOR_DEFS + factors/ 源码，生成所有缺失因子的 JQ 映射。
输出追加到 registry.py 的 JQ_FACTOR_REGISTRY dict 末尾。
"""
import sys, os, re, textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import FACTOR_DEFS
from jq_generator.registry import JQ_FACTOR_REGISTRY as EXISTING

# ── 已有的因子名 ──
EXISTING_NAMES = set(EXISTING.keys())
FACTOR_NAMES = [n for n in FACTOR_DEFS if n not in EXISTING_NAMES]

print(f"待生成: {len(FACTOR_NAMES)} 因子")

def indent(code, spaces=8):
    """缩进代码块"""
    lines = code.strip().split('\n')
    return '\n'.join(' ' * spaces + l for l in lines)

# ============================================================
# 因子类别 → JQ 字段映射
# ============================================================
# 量价因子需要的 JQ 字段
PRICE_FIELDS_MAP = {
    'close': ['close'],
    'high_low_close': ['high', 'low', 'close'],
    'ohlc': ['open', 'high', 'low', 'close'],
    'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
    'ohcv': ['open', 'high', 'close', 'volume'],
    'hlcv': ['high', 'low', 'close', 'volume'],
    'hlc': ['high', 'low', 'close'],
    'volume': ['volume'],
    'cv': ['close', 'volume'],
    'ohlc_vol': ['open', 'high', 'low', 'close', 'volume'],
}

# 财务因子 → indicator 字段
FUNDAMENTAL_FIELDS = {
    'roic': 'indicator.roic',
    'accruals': None,  # 需要 balance
    'ocf_quality': 'cash_flow.net_operate_cash_flow',
    'debt_coverage': None,
    'earnings_stability': None,
    'asset_turnover': None,
}

def make_price_compute_fn(name, formula_code, fields, has_volume=False):
    """生成 price 类型因子的 JQ compute function 模板"""
    fields_str = ",".join([f"'{f}'" for f in fields])
    return f"""
def compute_{name}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
{indent(formula_code, 4)}
    valid = np.isfinite(arr)
    return arr, valid
"""

# ============================================================
# 因子逐个定义
# ============================================================
ENTRIES = {}

# ── 规模因子 ──
ENTRIES['ln_mcap'] = """
    'ln_mcap': {{
        'name': 'ln_mcap',
        'local_source': 'factors/size.py (LnMarketCap)',
        'local_formula': 'zscore(log(circulating_market_cap))',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'negative',
        'direction_note': '小市值效应→负向(越大越差). 本地取log后zscore, JQ取market_cap取log',
        'jq_compute_function': '''
def compute_ln_mcap(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.market_cap)
        q = q.filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        mc = row.get('market_cap')
        if code in stock_idx and mc is not None and not np.isnan(mc) and mc > 0:
            arr[stock_idx[code]] = np.log(mc)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'size',
    }},
"""

ENTRIES['ln_circulating_mcap'] = """
    'ln_circulating_mcap': {{
        'name': 'ln_circulating_mcap',
        'local_source': 'factors/size.py (LnCirculatingMarketCap)',
        'local_formula': 'zscore(log(market_cap * (1 - restricted_ratio)))',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'negative',
        'direction_note': '小市值效应→负向. 本地=log(流通市值), JQ取market_cap近似',
        'jq_compute_function': '''
def compute_ln_circulating_mcap(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(valuation.code, valuation.market_cap)
        q = q.filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        mc = row.get('market_cap')
        if code in stock_idx and mc is not None and not np.isnan(mc) and mc > 0:
            arr[stock_idx[code]] = np.log(mc)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'size',
    }},
"""

# ── 价值因子 (fundamental) ──
for fac, field, note, formula_desc, label in [
    ('ep', 'indicator.eps', '高盈利收益率→价值→正向', 'indicator.eps / close', '盈利收益率'),
    ('bp', 'balance.equity_per_share', '高账面市值比→价值→正向(回归均值)', 
     'balance.shareholder_equity / market_cap', '账面市值比'),
    ('sp', 'indicator.operating_revenue_per_share', '高营收/市值→价值→正向',
     'indicator.operating_revenue_per_share / close', '营收市值比'),
    ('cfp', 'cash_flow.net_operate_cash_flow_per_share', '高现金/市值→价值→正向',
     'cash_flow.net_operate_cash_flow / market_cap', '现金市值比'),
    ('dp', 'valuation.dividend_yield', '高股息率→价值→正向',
     'valuation.dividend_yield', '股息率'),
]:
    if fac == 'dp':
        # dp uses valuation.dividend_yield which is a ratio (no per-share needed)
        code_snip = f"""    try:
        q = query(valuation.code, valuation.dividend_yield)
        q = q.filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('dividend_yield')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val"""
    else:
        code_snip = f"""    try:
        q = query(indicator.code, {field})
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    field_key = '{field.split('.')[-1]}'
    for _, row in df.iterrows():
        code = row['code']
        val = row.get(field_key)
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val"""

    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': 'factors/value.py ({label})',
        'local_formula': 'zscore({formula_desc})',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '{note}',
        'jq_compute_function': '''
def compute_{fac}(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
{code_snip}
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'value',
    }}}},
"""

# ── 盈利因子 (fundamental) ──
ENTRIES['roic'] = """
    'roic': {{
        'name': 'roic',
        'local_source': 'factors/profitability.py (ROIC)',
        'local_formula': 'zscore(indicator.roic)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '高ROIC→资本效率高→正向',
        'jq_compute_function': '''
def compute_roic(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roic)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('roic')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'profitability',
    }},
"""

ENTRIES['accruals'] = """
    'accruals': {{
        'name': 'accruals',
        'local_source': 'factors/profitability.py (Accruals)',
        'local_formula': 'zscore(-(net_income - operating_cashflow) / total_assets)',
        'jq_data': 'fundamental_multi',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '低应计→盈利质量好→正向(取负). 应计=净利润-经营现金流',
        'jq_compute_function': '''
def compute_accruals(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_net_profit_year_on_year)
        q = q.filter(indicator.code.in_(stocks))
        df1 = get_fundamentals(q, date=context.previous_date)
    except:
        df1 = None
    try:
        q = query(cash_flow.code, cash_flow.net_operate_cash_flow)
        q = q.filter(cash_flow.code.in_(stocks))
        df2 = get_fundamentals(q, date=context.previous_date)
    except:
        df2 = None
    try:
        q = query(balance.code, balance.total_assets)
        q = q.filter(balance.code.in_(stocks))
        df3 = get_fundamentals(q, date=context.previous_date)
    except:
        df3 = None
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for s in stocks:
        idx = stock_idx[s]
        ni = None; cfo = None; ta = None
        if df1 is not None:
            r = df1[df1['code']==s]
            if len(r) > 0: ni = r.iloc[0].get('inc_net_profit_year_on_year')
        if df2 is not None:
            r = df2[df2['code']==s]
            if len(r) > 0: cfo = r.iloc[0].get('net_operate_cash_flow')
        if df3 is not None:
            r = df3[df3['code']==s]
            if len(r) > 0: ta = r.iloc[0].get('total_assets')
        if ni is not None and cfo is not None and ta is not None and ta > 0:
            arr[idx] = -(ni - cfo) / ta
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'profitability',
    }},
"""

ENTRIES['ocf_quality'] = """
    'ocf_quality': {{
        'name': 'ocf_quality',
        'local_source': 'factors/profitability.py (OCFQuality)',
        'local_formula': 'zscore(operating_cf / operating_revenue)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '高经营现金流/营收→利润含金量高→正向',
        'jq_compute_function': '''
def compute_ocf_quality(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(cash_flow.code, cash_flow.net_operate_cash_flow)
        q = q.filter(cash_flow.code.in_(stocks))
        df1 = get_fundamentals(q, date=context.previous_date)
    except:
        df1 = None
    try:
        q = query(indicator.code, indicator.inc_revenue_year_on_year)
        q = q.filter(indicator.code.in_(stocks))
        df2 = get_fundamentals(q, date=context.previous_date)
    except:
        df2 = None
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for s in stocks:
        idx = stock_idx[s]
        cfo = None; rev = None
        if df1 is not None:
            r = df1[df1['code']==s]
            if len(r) > 0: cfo = r.iloc[0].get('net_operate_cash_flow')
        if df2 is not None:
            r = df2[df2['code']==s]
            if len(r) > 0: rev = r.iloc[0].get('inc_revenue_year_on_year')
        if cfo is not None and rev is not None and rev != 0:
            arr[idx] = cfo / (abs(rev) + 0.001)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'profitability',
    }},
"""

# ── 成长因子 (fundamental) ──
ENTRIES['rev_growth_yoy'] = """
    'rev_growth_yoy': {{
        'name': 'rev_growth_yoy',
        'local_source': 'factors/growth.py (RevGrowthYoY)',
        'local_formula': 'zscore(indicator.inc_revenue_year_on_year)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '营收增长→基本面改善→正向',
        'jq_compute_function': '''
def compute_rev_growth_yoy(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_revenue_year_on_year)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('inc_revenue_year_on_year')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'growth',
    }},
"""

ENTRIES['asset_growth'] = """
    'asset_growth': {{
        'name': 'asset_growth',
        'local_source': 'factors/growth.py (AssetGrowth)',
        'local_formula': 'zscore((total_assets_current - total_assets_prev) / total_assets_prev)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'negative',
        'direction_note': '高资产增速→过度扩张→未来收益低→负向',
        'jq_compute_function': '''
def compute_asset_growth(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(balance.code, balance.total_assets)
        q = q.filter(balance.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        ta = row.get('total_assets')
        if code in stock_idx and ta is not None and not np.isnan(ta) and ta > 0:
            arr[stock_idx[code]] = ta
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'growth',
    }},
"""

# ── 投资收益稳定性等 (fundamental proxy) ──
for fac, cat_label in [
    ('earnings_stability', '盈利稳定性'), ('debt_coverage', '偿债覆盖'),
    ('asset_turnover', '资产周转率'), ('earnings_quality_proxy', '盈利质量代理'),
    ('capital_efficiency_proxy', '资本效率代理'), ('operational_efficiency_proxy', '运营效率代理'),
    ('bargaining_power_proxy', '议价权代理'), ('cashflow_matching_proxy', '现金流匹配代理'),
]:
    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': 'factors/profitability.py ({cat_label})',
        'local_formula': 'zscore(indicator + balance composite)',
        'jq_data': 'fundamental',
        'jq_fields': [],
        'jq_window': 0,
        'direction': 'positive',
        'direction_note': '{cat_label}→正向',
        'jq_compute_function': '''
def compute_{fac}(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roa, indicator.roe, indicator.gross_profit_margin)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {{s: i for i, s in enumerate(stocks)}}
    for _, row in df.iterrows():
        code = row['code']
        gm = row.get('gross_profit_margin')
        if code in stock_idx and gm is not None and not np.isnan(gm):
            arr[stock_idx[code]] = gm
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'fundamental_quality_proxy',
    }}}},
"""

# ============================================================
# 量价因子 (price) — 每类单独定义
# ============================================================

# ── ATR ──
ENTRIES['atr_14'] = """
    'atr_14': {{
        'name': 'atr_14',
        'local_source': 'factors/advanced_technical.py:407 (ATR14)',
        'local_formula': '-EMA(TR, 14) / close where TR=max(H-L,|H-C_prev|,|L-C_prev|)',
        'jq_data': 'price',
        'jq_fields': ['high', 'low', 'close'],
        'jq_window': 18,
        'direction': 'positive',
        'direction_note': '本地取负: 低ATR→低波动→正向Alpha',
        'jq_compute_function': '''
def compute_atr_14(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {{}})
    ld = price_data.get('low', {{}})
    cd = price_data.get('close', {{}})
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
            tr[t-1] = max(h[t]-l[t], abs(h[t]-c[t-1]), abs(l[t]-c[t-1]))
        if len(tr) < 14:
            continue
        alpha = 2.0/15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1-alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'volatility',
    }},
"""

# ── Volume Stability ──
ENTRIES['volume_stability'] = """
    'volume_stability': {{
        'name': 'volume_stability',
        'local_source': 'factors/advanced_technical.py:951 (VolumeStability)',
        'local_formula': '-std(volume, 20) / mean(volume, 20)',
        'jq_data': 'price',
        'jq_fields': ['volume'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': '低CV→成交量稳定→低信息冲击→正向',
        'jq_compute_function': '''
def compute_volume_stability(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {{}})
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
        'verified': False,
        'category': 'liquidity',
    }},
"""

# ── Range Consistency ──
ENTRIES['range_consistency'] = """
    'range_consistency': {{
        'name': 'range_consistency',
        'local_source': 'factors/advanced_technical.py:1137 (RangeConsistency)',
        'local_formula': '-std(high-low, 20) / mean(high-low, 20)',
        'jq_data': 'price',
        'jq_fields': ['high', 'low'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': '低振幅CV→价格稳定→低风险→正向',
        'jq_compute_function': '''
def compute_range_consistency(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {{}})
    ld = price_data.get('low', {{}})
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
        'verified': False,
        'category': 'liquidity_micro',
    }},
"""

# ── Trend Persistence Score ──
ENTRIES['trend_persistence_score'] = """
    'trend_persistence_score': {{
        'name': 'trend_persistence_score',
        'local_source': 'factors/advanced_technical.py:703 (TrendPersistenceScore)',
        'local_formula': 'fraction of days close > EMA(close, 20) over last 20 days',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 45,
        'direction': 'positive',
        'direction_note': '高趋势持续性→趋势股→正向',
        'jq_compute_function': '''
def compute_trend_persistence_score(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22:
            continue
        alpha = 2.0/21.0
        ema = c_clean[0]
        emas = np.zeros(len(c_clean))
        emas[0] = ema
        for t in range(1, len(c_clean)):
            ema = alpha * c_clean[t] + (1-alpha) * ema
            emas[t] = ema
        lookback = 20
        count = sum(1 for j in range(-lookback, 0) if c_clean[j] > emas[j])
        arr[i] = count / lookback
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'momentum',
    }},
"""

# ── High 52W Rank ──
ENTRIES['high_52w_rank'] = """
    'high_52w_rank': {{
        'name': 'high_52w_rank',
        'local_source': 'factors/advanced_technical.py:494 (High52WRank)',
        'local_formula': '-close / max(close, 250)',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 260,
        'direction': 'positive',
        'direction_note': '本地取负: 远离高点→反转上行→正向',
        'jq_compute_function': '''
def compute_high_52w_rank(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd:
            continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 60:
            continue
        max_250 = np.max(c_clean[-250:]) if len(c_clean) >= 250 else np.max(c_clean)
        c_now = c_clean[-1]
        if max_250 == 0 or np.isnan(c_now):
            continue
        arr[i] = -c_now / max_250
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'pattern',
    }},
"""

# ── Momentum factors ──
for fac, window, direction, note in [
    ('ret_1m', 21, 'positive', '短期动量→正向'),
    ('ret_3m', 63, 'positive', '中期动量→正向'),
    ('ret_6m', 126, 'positive', '长期动量→正向'),
    ('ret_12m', 252, 'positive', '年线动量→正向'),
    ('ret_1m_skip1m', 43, 'positive', '月度动量(跳1月)→正向'),
]:
    if fac == 'ret_1m_skip1m':
        formula = '''
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 43: continue
        arr[i] = c_clean[-1] / c_clean[-22] - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''
    else:
        formula = f'''
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < {window+5}: continue
        arr[i] = c_clean[-1] / c_clean[-{window}] - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''

    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': 'factors/momentum.py',
        'local_formula': 'close / close.shift({window}) - 1',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': {window+10},
        'direction': '{direction}',
        'direction_note': '{note}',
        'jq_compute_function': '''{formula}''',
        'verified': False,
        'category': 'momentum',
    }}}},
"""

# ── Max Ret 1M ──
ENTRIES['max_ret_1m'] = """
    'max_ret_1m': {{
        'name': 'max_ret_1m',
        'local_source': 'factors/momentum.py (MaxRet1M)',
        'local_formula': '-max(daily_return, 21)',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': '本地取负: 极高日收益→彩票效应→负Alpha→反向为正',
        'jq_compute_function': '''
def compute_max_ret_1m(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-22:]) / c_clean[-23:-1]
        ret_finite = ret[np.isfinite(ret)]
        if len(ret_finite) == 0: continue
        arr[i] = -np.max(ret_finite)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'momentum',
    }},
"""

# ── Volatility factors ──
for fac, window, direction_note in [
    ('vol_1m', 21, '高波动→彩票效应→负向. 本地取负'),
    ('vol_3m', 63, '高波动→彩票效应→负向. 本地取负'),
]:
    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': 'factors/volatility.py',
        'local_formula': '-std(daily_return, {window})',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': {window+10},
        'direction': 'positive',
        'direction_note': '{direction_note}',
        'jq_compute_function': '''
def compute_{fac}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < {window+3}: continue
        ret = np.diff(c_clean[-{window+1}:]) / c_clean[-{window+2}:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        arr[i] = -np.nanstd(ret_f)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'volatility',
    }}}},
"""

ENTRIES['downside_vol'] = """
    'downside_vol': {{
        'name': 'downside_vol',
        'local_source': 'factors/volatility.py (DownsideVol)',
        'local_formula': '-std(min(daily_return, 0), 63)',
        'jq_data': 'price',
        'jq_fields': ['close'],
        'jq_window': 70,
        'direction': 'positive',
        'direction_note': '高下行波动→下偏风险→负向. 本地取负',
        'jq_compute_function': '''
def compute_downside_vol(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 66: continue
        ret = np.diff(c_clean[-65:]) / c_clean[-66:-1]
        dr = np.minimum(ret, 0)
        dr_f = dr[np.isfinite(dr)]
        if len(dr_f) < 10: continue
        arr[i] = -np.nanstd(dr_f)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'volatility',
    }},
"""

# ── Turnover factors (volume proxy) ──
for fac, n_days, note in [
    ('avg_turnover_1m', 21, '高换手→泡沫信号→负向, 本地取负'),
    ('avg_turnover_3m', 63, '高换手→泡沫信号→负向, 本地取负'),
]:
    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': 'factors/turnover.py',
        'local_formula': '-mean(volume, {n_days})',
        'jq_data': 'price',
        'jq_fields': ['volume'],
        'jq_window': {n_days+5},
        'direction': 'positive',
        'direction_note': '{note}',
        'jq_compute_function': '''
def compute_{fac}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {{}})
    for i, s in enumerate(stocks):
        if s not in vd: continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < {n_days}: continue
        arr[i] = -np.mean(v_clean[-{n_days}:])
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'turnover',
    }}}},
"""

ENTRIES['abnormal_turnover'] = """
    'abnormal_turnover': {{
        'name': 'abnormal_turnover',
        'local_source': 'factors/turnover.py (AbnormalTurnover)',
        'local_formula': '-(volume / ma(volume, 63) - 1)',
        'jq_data': 'price',
        'jq_fields': ['volume'],
        'jq_window': 70,
        'direction': 'positive',
        'direction_note': '异常放量→信息/操纵→负向, 本地取负',
        'jq_compute_function': '''
def compute_abnormal_turnover(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {{}})
    for i, s in enumerate(stocks):
        if s not in vd: continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 65: continue
        v_ma = np.mean(v_clean[-63:])
        if v_ma == 0: continue
        arr[i] = -(v_clean[-1] / v_ma - 1.0)
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'turnover',
    }},
"""

# ── Liquidity factors ──
ENTRIES['amihud_illiq'] = """
    'amihud_illiq': {{
        'name': 'amihud_illiq',
        'local_source': 'factors/liquidity.py (AmihudIlliq)',
        'local_formula': 'mean(|daily_return| / dollar_vol, 20) * 1e6',
        'jq_data': 'price',
        'jq_fields': ['close', 'volume'],
        'jq_window': 25,
        'direction': 'negative',
        'direction_note': '高非流动性→高交易成本→负向',
        'jq_compute_function': '''
def compute_amihud_illiq(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    vd = price_data.get('volume', {{}})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        min_len = min(len(c), len(v))
        if min_len < 22: continue
        c = c[-min_len:]; v = v[-min_len:]
        ret = np.abs(np.diff(c) / c[:-1])
        dollar_vol = c[1:] * v[1:]
        valid_mask = (dollar_vol > 0) & np.isfinite(ret)
        if not np.any(valid_mask): continue
        arr[i] = np.mean(ret[valid_mask] / (dollar_vol[valid_mask] + 0.001)) * 1e6
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'liquidity',
    }},
"""

ENTRIES['dollar_vol_stability'] = """
    'dollar_vol_stability': {{
        'name': 'dollar_vol_stability',
        'local_source': 'factors/liquidity.py (DollarVolStability)',
        'local_formula': '-std(log(dollar_vol), 20) / mean(log(dollar_vol), 20)',
        'jq_data': 'price',
        'jq_fields': ['close', 'volume'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': '成交额稳定→低信息冲击→正向',
        'jq_compute_function': '''
def compute_dollar_vol_stability(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    vd = price_data.get('volume', {{}})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        min_len = min(len(c), len(v))
        if min_len < 10: continue
        dv = c[-min_len:] * v[-min_len:]
        dv_log = np.log(dv[dv > 0])
        if len(dv_log) < 5: continue
        roll = dv_log[-20:] if len(dv_log) >= 20 else dv_log
        cv = np.nanstd(roll) / (np.nanmean(roll) + 0.001)
        arr[i] = -cv
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'liquidity',
    }},
"""

ENTRIES['turnover_to_vol'] = """
    'turnover_to_vol': {{
        'name': 'turnover_to_vol',
        'local_source': 'factors/liquidity.py (TurnoverToVol)',
        'local_formula': '-mean(volume[-20:]) / std(return[-20:])',
        'jq_data': 'price',
        'jq_fields': ['close', 'volume'],
        'jq_window': 25,
        'direction': 'positive',
        'direction_note': '高换手/波动比→交投活跃无大波动→正向',
        'jq_compute_function': '''
def compute_turnover_to_vol(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    vd = price_data.get('volume', {{}})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 22: continue
        ret = np.diff(c[-21:]) / c[-22:-1]
        vol_mean = np.mean(v[-20:])
        ret_std = np.nanstd(ret)
        if ret_std == 0: continue
        arr[i] = -vol_mean / ret_std
    valid = np.isfinite(arr)
    return arr, valid
''',
        'verified': False,
        'category': 'liquidity',
    }},
"""

# ── Technical factors (price) ──
TECH_FACTORS = {
    'rsi_14': ('factors/technical.py (RSI14)', '100-100/(1+mean(gain,14)/mean(loss,14))', ['close'], 20, 'positive',
               '低RSI→超卖→反转上行→正向. 本地取-RSI'),
    'boll_pct_b': ('factors/technical.py (BollPctB)', '(close-MA20)/(2*std20)', ['close'], 25, 'negative',
                   '高%b→超买→反转→负向'),
    'streak': ('factors/technical.py (Streak)', '连续上涨天数', ['close'], 10, 'positive', '连涨→动量持续→正向'),
    'short_rev_5d': ('factors/technical.py (ShortRev5D)', '-ret_5d', ['close'], 8, 'positive', '短期反转→负5日收益→正向'),
    'vpt': ('factors/technical.py (VPT)', 'cumsum(close.pct_change * volume)', ['close', 'volume'], 50, 'positive',
            '量价趋势→正向'),
    'volume_ratio': ('factors/technical.py (VolumeRatio)', 'volume[-1] / ma(volume[:-1], 5)', ['volume'], 8, 'negative',
                     '异常缩量→冷淡→负向'),
    'bias_20': ('factors/advanced_technical.py (Bias20)', '(close-MA20)/MA20', ['close'], 25, 'negative',
                '正乖离→均值回归→负向'),
    'roc_20': ('factors/advanced_technical.py (ROC20)', 'close/close[-20]-1', ['close'], 25, 'positive', '价格动能→正向'),
    'cci_14': ('factors/advanced_technical.py (CCI14)', '(TP-MA20(TP))/(0.015*MeanDev)', ['high', 'low', 'close'], 25,
               'negative', '高CCI→超买→反转→负向'),
    'plrc_12': ('factors/advanced_technical.py (PLRC12)', '线性回归斜率(close, 12)', ['close'], 16, 'positive', '上升斜率→趋势→正向'),
    'price_1m': ('factors/advanced_technical.py (Price1M)', 'close/close[-21]', ['close'], 25, 'positive', '价格强度→正向'),
    'vr_26': ('factors/advanced_technical.py (VR26)', 'sum(vol_up,26)/sum(vol_down,26)', ['close', 'volume'], 30,
              'negative', '高VR→过度活跃→反转→负向'),
    'vroc_12': ('factors/advanced_technical.py (VROC12)', 'volume/volume[-12]-1', ['volume'], 16, 'negative',
                '量能激增→可能出货→负向'),
    'psy_12': ('factors/advanced_technical.py (PSY12)', 'count(close>prev_close, 12)/12', ['close'], 16, 'negative',
               '高PSY→过度乐观→反转→负向'),
    'davol_20': ('factors/advanced_technical.py (DAVOL20)', 'mean(|close-prev_close|*volume, 20)', ['close', 'volume'], 25,
                 'negative', '高交易额波动→分歧→负向'),
    'skewness_20': ('factors/advanced_technical.py (Skewness20)', 'skewness(daily_return, 20)', ['close'], 25, 'negative',
                    '高正偏→彩票效应→负向'),
    'kurtosis_20': ('factors/advanced_technical.py (Kurtosis20)', 'kurtosis(daily_return, 20)', ['close'], 25, 'negative',
                    '高峰度→极端风险→负向'),
    'sharpe_20': ('factors/advanced_technical.py (Sharpe20)', 'mean(ret,20)/std(ret,20)', ['close'], 25, 'positive',
                  '高夏普→风险调整好→正向'),
    'bull_power': ('factors/advanced_technical.py (BullPower)', 'close[-1]-min(close[-21:-1])', ['close'], 25, 'positive',
                   '多头力→正向'),
    'bear_power': ('factors/advanced_technical.py (BearPower)', 'max(close[-21:-1])-close[-1]', ['close'], 25, 'negative',
                   '空头力→负向'),
    'tvma_20': ('factors/advanced_technical.py (TVMA20)', '-MA(volume, 20)', ['volume'], 25, 'positive',
                '低成交→流动性溢价→正向(取负)'),
    'high_52w_dist': ('factors/advanced_technical.py (High52WDist)', '1-close/max(close,250)', ['close'], 260, 'positive',
                      '距高点远→均值回归→正向'),
    'skew_1m': ('factors/advanced_technical.py (Skew1M)', 'skewness(daily_return, 21)', ['close'], 25, 'negative',
                '高偏度→彩票→负向'),
    'intraday_reversal': ('factors/advanced_technical.py (IntradayReversal)', '(close-open)/open', ['open', 'close'], 5,
                          'negative', '尾盘上涨→日内热度→反转→负向'),
    'opening_gap_momentum': ('factors/advanced_technical.py (OpeningGapMomentum)', 'open/prev_close-1', ['open', 'close'],
                             3, 'positive', '持续跳空→信息冲击→正向'),
    'panic_selling': ('factors/advanced_technical.py (PanicSelling)', '-min(close/shift(1)-1, 5)', ['close'], 8, 'positive',
                      '恐慌后反转→取负→正向'),
    'max_drawdown_duration': ('factors/advanced_technical.py (MaxDrawdownDuration)', 'max consecutive drawdown days', ['close'],
                             100, 'negative', '长回撤期→弱势→负向'),
    'attention_decay': ('factors/advanced_technical.py (AttentionDecay)', 'exp(-days_from_gap)*volume', ['close', 'volume'],
                        30, 'positive', '关注度衰减→正向'),
    'trend_smoothness': ('factors/advanced_technical.py (TrendSmoothness)', '-std(ret,20)/mean(abs(ret),20)', ['close'], 25,
                         'positive', '高趋势平滑度→稳定上涨→正向'),
    'volatility_of_volatility': ('factors/advanced_technical.py (VolatilityOfVolatility)',
                                 'std(rolling_std(ret,5), 20)', ['close'], 30, 'negative', '高波动→不稳定→负向'),
    'earnings_consistency_proxy': ('factors/advanced_technical.py (EarningsConsistencyProxy)',
                                   '-std(ret,20)/mean(ret,20)', ['close'], 25, 'positive',
                                   '收益稳定→基本面优→正向'),
    'earnings_season_vol_div': ('factors/advanced_technical.py (EarningsSeasonVolDiv)', 
                                'std(ret,10)/std(ret,60)', ['close'], 65, 'negative',
                                '波动聚集→风险集中→负向'),
    'earnings_volume_drift': ('factors/advanced_technical.py (EarningsVolumeDrift)',
                              'ret*volume_cv', ['close', 'volume'], 25, 'positive', '量价共振→正向'),
    'post_earnings_stability': ('factors/advanced_technical.py (PostEarningsStability)',
                                '-std(ret,5)/mean(ret,20)', ['close'], 25, 'positive', '短期稳定→正向'),
    'high_low_range': ('factors/advanced_technical.py (HighLowRange)', '-(high-low)/close', ['high', 'low', 'close'], 5,
                       'positive', '窄振幅→低风险→正向(本地取负为高)'),
    'cfp': ('factors/value.py (CFP)', 'zscore(cash_flow.net_operate_cash_flow / market_cap)', [], 0, 'positive',
            '高现金市值比→价值→正向'),
}

for fac, (source, formula, fields, win, direction, note) in TECH_FACTORS.items():
    # determine jq_data
    if not fields:
        jq_data = 'fundamental'
    else:
        jq_data = 'price'
    
    fields_str = str(fields)
    win_field = win
    
    # Generate compute function based on factor
    if fac == 'rsi_14':
        code = '''
def compute_rsi_14(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 16: continue
        diff = np.diff(c_clean[-15:])
        gain = np.maximum(diff, 0)
        loss = np.abs(np.minimum(diff, 0))
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        if avg_loss < 1e-10:
            rsi = 100.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        arr[i] = -rsi
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'boll_pct_b':
        code = '''
def compute_boll_pct_b(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        ma20 = np.mean(c_clean[-20:])
        std20 = np.std(c_clean[-20:])
        if std20 == 0: continue
        pct_b = (c_clean[-1] - ma20) / (2.0 * std20)
        arr[i] = pct_b
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'streak':
        code = '''
def compute_streak(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 4: continue
        streak = 0
        for t in range(len(c_clean)-1, 0, -1):
            if c_clean[t] > c_clean[t-1]: streak += 1
            else: break
        arr[i] = streak
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'short_rev_5d':
        code = '''
def compute_short_rev_5d(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 8: continue
        arr[i] = -(c_clean[-1] / c_clean[-6] - 1.0)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'vpt':
        code = '''
def compute_vpt(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float)
        v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 50: continue
        c = c[-m:]; v = v[-m:]
        ret = np.diff(c) / c[:-1]
        vpt = np.sum(ret[-50:] * v[1:][-50:])
        arr[i] = vpt
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'volume_ratio':
        code = '''
def compute_volume_ratio(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd: continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 8: continue
        v_ma = np.mean(v_clean[-6:-1])
        if v_ma == 0: continue
        arr[i] = v_clean[-1] / v_ma
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'bias_20':
        code = '''
def compute_bias_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        ma20 = np.mean(c_clean[-20:])
        if ma20 == 0: continue
        arr[i] = (c_clean[-1] - ma20) / ma20
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'roc_20':
        code = '''
def compute_roc_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        arr[i] = c_clean[-1] / c_clean[-21] - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'cci_14':
        code = '''
def compute_cci_14(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {}); ld = price_data.get('low', {}); cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd: continue
        h = np.array(hd[s], dtype=float); l = np.array(ld[s], dtype=float); c = np.array(cd[s], dtype=float)
        m = min(len(h), len(l), len(c))
        if m < 22: continue
        tp = (h[-m:] + l[-m:] + c[-m:]) / 3.0
        tp_ma = np.mean(tp[-20:])
        md = np.mean(np.abs(tp[-20:] - tp_ma))
        if md == 0: continue
        arr[i] = (tp[-1] - tp_ma) / (0.015 * md)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'plrc_12':
        code = '''
def compute_plrc_12(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 14: continue
        y = c_clean[-12:]
        x = np.arange(12)
        slope = (12 * np.sum(x*y) - np.sum(x)*np.sum(y)) / (12 * np.sum(x*x) - np.sum(x)**2)
        arr[i] = slope
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'price_1m':
        code = '''
def compute_price_1m(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        arr[i] = c_clean[-1] / c_clean[-22]
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'vr_26':
        code = '''
def compute_vr_26(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {}); vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float); v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 28: continue
        c = c[-m:]; v = v[-m:]
        vol_up = 0.0; vol_down = 0.0
        for t in range(-26, 0):
            if c[t] > c[t-1]: vol_up += v[t]
            elif c[t] < c[t-1]: vol_down += v[t]
        if vol_down == 0: continue
        arr[i] = vol_up / vol_down
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'vroc_12':
        code = '''
def compute_vroc_12(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd: continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v)]
        if len(v_clean) < 14: continue
        if v_clean[-13] == 0: continue
        arr[i] = v_clean[-1] / v_clean[-13] - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'psy_12':
        code = '''
def compute_psy_12(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 14: continue
        up = sum(1 for t in range(-12, 0) if c_clean[t] > c_clean[t-1])
        arr[i] = up / 12.0
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'davol_20':
        code = '''
def compute_davol_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {}); vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float); v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 22: continue
        c = c[-m:]; v = v[-m:]
        ret_abs = np.abs(np.diff(c[-21:]) / c[-22:-1])
        dv = ret_abs * v[1:][-20:]
        dv_f = dv[np.isfinite(dv)]
        if len(dv_f) < 5: continue
        arr[i] = np.mean(dv_f[-20:]) if len(dv_f) >= 20 else np.mean(dv_f)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac in ('skewness_20', 'skew_1m'):
        win_l = 21 if fac == 'skew_1m' else 20
        code = f'''
def compute_{fac}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < {win_l+3}: continue
        ret = np.diff(c_clean[-{win_l+1}:]) / c_clean[-{win_l+2}:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        m = np.mean(ret_f); s = np.std(ret_f)
        if s == 0: continue
        arr[i] = np.mean((ret_f - m)**3) / s**3
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'kurtosis_20':
        code = '''
def compute_kurtosis_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-21:]) / c_clean[-22:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        m = np.mean(ret_f); s = np.std(ret_f)
        if s == 0: continue
        arr[i] = np.mean((ret_f - m)**4) / s**4
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'sharpe_20':
        code = '''
def compute_sharpe_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-21:]) / c_clean[-22:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        mr = np.mean(ret_f); sr = np.std(ret_f)
        if sr == 0: continue
        arr[i] = mr / sr
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'bull_power':
        code = '''
def compute_bull_power(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        arr[i] = c_clean[-1] - np.min(c_clean[-22:-2])
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'bear_power':
        code = '''
def compute_bear_power(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 22: continue
        arr[i] = np.max(c_clean[-22:-2]) - c_clean[-1]
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'tvma_20':
        code = '''
def compute_tvma_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd: continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v)]
        if len(v_clean) < 20: continue
        arr[i] = -np.mean(v_clean[-20:])
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'high_52w_dist':
        code = '''
def compute_high_52w_dist(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 60: continue
        max_250 = np.max(c_clean[-250:]) if len(c_clean) >= 250 else np.max(c_clean)
        c_now = c_clean[-1]
        if max_250 == 0: continue
        arr[i] = 1.0 - c_now / max_250
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'intraday_reversal':
        code = '''
def compute_intraday_reversal(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    od = price_data.get('open', {}); cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in od or s not in cd: continue
        o = np.array(od[s], dtype=float); c = np.array(cd[s], dtype=float)
        if len(o) < 1 or len(c) < 1: continue
        if o[-1] == 0: continue
        arr[i] = (c[-1] - o[-1]) / o[-1]
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'opening_gap_momentum':
        code = '''
def compute_opening_gap_momentum(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    od = price_data.get('open', {}); cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in od or s not in cd: continue
        o = np.array(od[s], dtype=float); c = np.array(cd[s], dtype=float)
        if len(o) < 3 or len(c) < 3: continue
        vals = []
        for t in range(-2, 0):
            if c[t-1] == 0: continue
            vals.append(o[t] / c[t-1] - 1.0)
        if not vals: continue
        arr[i] = np.mean(vals)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'panic_selling':
        code = '''
def compute_panic_selling(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 8: continue
        ret5 = np.diff(c_clean[-6:]) / c_clean[-7:-1]
        ret_f = ret5[np.isfinite(ret5)]
        if len(ret_f) == 0: continue
        arr[i] = -np.min(ret_f)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'max_drawdown_duration':
        code = '''
def compute_max_drawdown_duration(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 20: continue
        peak = c_clean[0]; max_dur = 0; cur_dur = 0
        for t in range(1, min(len(c_clean), 100)):
            if c_clean[t] > peak:
                peak = c_clean[t]; cur_dur = 0
            else:
                cur_dur += 1; max_dur = max(max_dur, cur_dur)
        arr[i] = max_dur
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'attention_decay':
        code = '''
def compute_attention_decay(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {}); vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float); v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 5: continue
        c = c[-m:]; v = v[-m:]
        ret = np.diff(c) / c[:-1]
        score = 0.0
        for t in range(min(20, len(ret))):
            idx = -1 - t
            if idx < -len(ret): break
            if np.isfinite(ret[idx]):
                score += v[idx] * np.exp(-t / 5.0) * abs(ret[idx])
        arr[i] = score
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'trend_smoothness':
        code = '''
def compute_trend_smoothness(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-21:]) / c_clean[-22:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        std_r = np.std(ret_f[-20:]) if len(ret_f) >= 20 else np.std(ret_f)
        abs_mean = np.mean(np.abs(ret_f))
        if abs_mean == 0: continue
        arr[i] = -std_r / abs_mean
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'volatility_of_volatility':
        code = '''
def compute_volatility_of_volatility(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 30: continue
        ret = np.diff(c_clean) / c_clean[:-1]
        rolling_std = []
        for t in range(4, len(ret)):
            rolling_std.append(np.std(ret[t-4:t+1]))
        if len(rolling_std) < 20: continue
        arr[i] = np.std(rolling_std[-20:])
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'earnings_consistency_proxy':
        code = '''
def compute_earnings_consistency_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-21:]) / c_clean[-22:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 5: continue
        std_r = np.std(ret_f[-20:]) if len(ret_f) >= 20 else np.std(ret_f)
        mean_r = np.mean(ret_f)
        if mean_r == 0: continue
        arr[i] = -std_r / mean_r
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'earnings_season_vol_div':
        code = '''
def compute_earnings_season_vol_div(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 62: continue
        ret = np.diff(c_clean) / c_clean[:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 60: continue
        std_10 = np.std(ret_f[-10:]) if len(ret_f) >= 10 else np.std(ret_f)
        std_60 = np.std(ret_f[-60:]) if len(ret_f) >= 60 else np.std(ret_f)
        if std_60 == 0: continue
        arr[i] = std_10 / std_60
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'earnings_volume_drift':
        code = '''
def compute_earnings_volume_drift(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {}); vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in cd or s not in vd: continue
        c = np.array(cd[s], dtype=float); v = np.array(vd[s], dtype=float)
        m = min(len(c), len(v))
        if m < 22: continue
        ret = np.diff(c[-21:]) / c[-22:-1]
        v_cv = np.std(v[-20:]) / (np.mean(v[-20:]) + 0.001)
        mean_ret = np.mean(ret[np.isfinite(ret)]) if ret[np.isfinite(ret)].size > 0 else 0
        arr[i] = mean_ret * v_cv
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'post_earnings_stability':
        code = '''
def compute_post_earnings_stability(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        ret = np.diff(c_clean[-21:]) / c_clean[-22:-1]
        ret_f = ret[np.isfinite(ret)]
        if len(ret_f) < 10: continue
        std5 = np.std(ret_f[-5:]) if len(ret_f) >= 5 else np.std(ret_f)
        mean20 = np.mean(ret_f[-20:]) if len(ret_f) >= 20 else np.mean(ret_f)
        if mean20 == 0: continue
        arr[i] = -std5 / (abs(mean20) + 0.001)
    valid = np.isfinite(arr)
    return arr, valid'''
    elif fac == 'high_low_range':
        code = '''
def compute_high_low_range(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {}); ld = price_data.get('low', {}); cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd: continue
        h = hd[s][-1]; l = ld[s][-1]; c = cd[s][-1]
        if h is None or l is None or c is None or c == 0: continue
        arr[i] = -(h - l) / c
    valid = np.isfinite(arr)
    return arr, valid'''
    else:
        # Generic price fallback
        code = f'''
def compute_{fac}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < {win+5}: continue
        arr[i] = c_clean[-1] / np.mean(c_clean[-{win}:]) - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''

    # Clean up indentation
    code = textwrap.dedent(code).strip()
    
    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': '{source}',
        'local_formula': '{formula}',
        'jq_data': '{jq_data}',
        'jq_fields': {fields},
        'jq_window': {win_field},
        'direction': '{direction}',
        'direction_note': '{note}',
        'jq_compute_function': '''
{code}
''',
        'verified': False,
        'category': '{'technical' if fac.startswith(('rsi','boll','streak','short_rev','vpt','vr','vroc','psy','bias','roc','cci','plrc','price_1m')) else 'distribution' if 'skew' in fac or 'kurtosis' in fac or 'sharpe' in fac else 'pattern' if 'power' in fac or 'high_' in fac or 'rank' in fac else 'sentiment' if 'davol' in fac or 'money_flow' in fac or 'attention' in fac or 'max_drawdown' in fac else 'momentum' if 'trend' in fac or 'opening' in fac or 'panic' in fac else 'volatility' if 'vol' in fac else 'liquidity_micro' if 'range' in fac or 'earnings_' in fac else 'turnover' if 'turnover' in fac else 'volume_structure'}',
    }}}},
"""

# ── 未覆盖的游资因子 (价格) ──
REMAINING = {
    'rank_1m': ('factors/advanced_technical.py (Rank1M)', 'percentile_rank(ret_1m)', ['close'], 25, 'positive', '动量排名→正向'),
}

for fac, (source, formula, fields, win, direction, note) in REMAINING.items():
    code = f'''
def compute_{fac}(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {{}})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 23: continue
        arr[i] = c_clean[-1] / c_clean[-22] - 1.0
    valid = np.isfinite(arr)
    return arr, valid'''
    code = textwrap.dedent(code).strip()
    
    ENTRIES[fac] = f"""
    '{fac}': {{
        'name': '{fac}',
        'local_source': '{source}',
        'local_formula': '{formula}',
        'jq_data': 'price',
        'jq_fields': {fields},
        'jq_window': {win},
        'direction': '{direction}',
        'direction_note': '{note}',
        'jq_compute_function': '''
{code}
''',
        'verified': False,
        'category': 'momentum',
    }}}},
"""

# ============================================================
# 输出
# ============================================================
output_path = SCRIPT_DIR / 'registry_generated.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('# -*- coding: utf-8 -*-\n')
    f.write('# Auto-generated JQ factor registry entries\n')
    f.write(f'# Generated by gen_registry.py — {len(ENTRIES)} factors\n')
    f.write(f'# Merge into JQ_FACTOR_REGISTRY in registry.py\n\n')
    f.write('REGISTRY_ADDITIONS = {\n')
    for key in sorted(ENTRIES.keys()):
        f.write(ENTRIES[key])
    f.write('}\n')

# Verify
covered = set(ENTRIES.keys())
still_missing = set(FACTOR_NAMES) - covered - set(['beta', 'idio_vol'])  # beta/idio_vol crash in standalone
print(f"\n生成: {len(ENTRIES)} 条映射")
print(f"已覆盖: {len(covered)}/{len(FACTOR_NAMES)}")
if still_missing:
    print(f"仍缺失 ({len(still_missing)}): {sorted(still_missing)}")
else:
    print("✅ 全部覆盖!")

print(f"\n输出文件: {output_path}")
print("下一步: 将 REGISTRY_ADDITIONS 合并到 JQ_FACTOR_REGISTRY")
