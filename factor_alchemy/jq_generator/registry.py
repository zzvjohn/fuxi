# -*- coding: utf-8 -*-
"""
JQ Factor Registry — 因子 → JQ 等价实现映射

每个条目:
  {
    'name': 因子名称 (对应 factors/*.py 中的 name 属性),
    'local_source': 本地源码文件 + 行号,
    'local_formula': 本地公式 (简洁描述),
    'jq_data': 'price' | 'fundamental' | 'fundamental_multi',
    'jq_fields': JQ get_price 需要的字段列表 (如 ['close', 'volume', 'high', 'low', 'open']),
    'jq_cols': JQ get_fundamentals 需要的 query 列 (如 'indicator.roe'),
    'jq_window': 最小回溯天数,
    'direction': 'positive' | 'negative' (正向=原始值越大越好, 负向=原始值越大越差),
    'direction_note': 方向说明 (为什么正向/负向),
    'jq_compute_function': JQ 平台上的 numpy 计算逻辑 (伪代码),
    'verified': bool — 是否已通过双源交叉验证,
    'category': 因子类别,
  }

JQ 平台数据源分类:
  - price: 纯量价因子, 只需 get_price(['open','high','low','close','volume'])
  - fundamental: 单次 get_fundamentals(query(...)) 即可
  - fundamental_multi: 需要多次 query (如 f_score 需 indicator/balance/cash_flow)
"""

JQ_FACTOR_REGISTRY = {
    # ============================================================
    # 量价因子 (price) — 最可靠, 无基本面前视偏差风险
    # ============================================================

    'relative_spread_proxy': {
    'name': 'relative_spread_proxy',
    'local_source': 'factors/advanced_technical.py:1191 (RelativeSpreadProxy)',
    'local_formula': '-(high - low) / (close + 0.001)',
    'jq_data': 'price',
    'jq_fields': ['high', 'low', 'close'],
    'jq_window': 5,
    'direction': 'positive',
    'direction_note': '小价差→高流动性低交易成本→正向Alpha',
        'jq_compute_function': '''
def compute_relative_spread_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    high_d = price_data.get('high', {})
    low_d = price_data.get('low', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in high_d or s not in low_d or s not in close_d:
            continue
        h = np.array(high_d[s], dtype=float)
        l = np.array(low_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(h) < 1 or len(l) < 1 or len(c) < 1:
            continue
        hv = h[-1]; lv = l[-1]; cv = c[-1]
        if np.isnan(hv) or np.isnan(lv) or np.isnan(cv) or cv == 0:
            continue
        arr[i] = -(hv - lv) / (cv + 0.001)
    valid = np.isfinite(arr)
    return arr, valid
        ''',
    'verified': True,  # v7.6 JQ验证通过
    'category': 'liquidity_micro',
    },

    'gap_up': {
    'name': 'gap_up',
    'local_source': 'factors/advanced_technical.py:651 (GapUp)',
    'local_formula': 'open / close.shift(1) - 1',
    'jq_data': 'price',
    'jq_fields': ['open', 'close'],
    'jq_window': 5,
    'direction': 'positive',
    'direction_note': '向上跳空→信息冲击→短期动量延续',
    'jq_compute_function': '''
def compute_gap_up(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    open_d = price_data.get('open', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in open_d or s not in close_d:
            continue
        o = np.array(open_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(o) < 2 or len(c) < 2:
            continue
        o_today = o[-1]
        c_prev = c[-2]
        if np.isnan(o_today) or np.isnan(c_prev) or c_prev == 0:
            continue
        arr[i] = o_today / c_prev - 1
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': True,  # v7.6 JQ验证通过
    'category': 'momentum',
    },

    'turnover_change': {
    'name': 'turnover_change',
    'local_source': 'factors/turnover.py:64 (TurnoverChange) + WORKAROUND',
    'local_formula': '-(MA5(volume) / MA20(volume) - 1) [volume代理, 数学等价]',
    'jq_data': 'price',
    'jq_fields': ['volume'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '换手骤升→信息/操纵→负Alpha→取负为正向. volume MA比值代理(流通股本在MA5/MA20中约掉)',
    'jq_compute_function': '''
def compute_turnover_change(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < 10:
            continue
        avg5 = np.mean(v[-5:])
        avg20 = np.mean(v[-20:])
        if avg20 == 0 or np.isnan(avg20):
            continue
        change = avg5 / avg20 - 1
        arr[i] = -change
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': True,  # v7.6 JQ验证通过
    'category': 'turnover',
    },

    'vm_diff': {
    'name': 'vm_diff',
    'local_source': 'factors/advanced_technical.py:545 (VMDiff)',
    'local_formula': '-clip(EMA12(vol) / EMA26(vol) - 1, -5, 5)',
    'jq_data': 'price',
    'jq_fields': ['volume'],
    'jq_window': 30,
    'direction': 'positive',
    'direction_note': '量MACD金叉放量→可能拉升出货→负向, 取负为正. 本地已在compute内取负',
    'jq_compute_function': '''
def compute_vm_diff(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v)]
        if len(v_arr) < 26:
            continue
        span12 = 12; span26 = 26
        alpha12 = 2.0 / (span12 + 1); alpha26 = 2.0 / (span26 + 1)
        ema12 = v_arr[0]; ema26 = v_arr[0]
        for t in range(1, len(v_arr)):
            ema12 = alpha12 * v_arr[t] + (1 - alpha12) * ema12
            ema26 = alpha26 * v_arr[t] + (1 - alpha26) * ema26
        if ema26 == 0:
            continue
        diff_norm = ema12 / ema26 - 1
        arr[i] = -np.clip(diff_norm, -5, 5)
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,  # 待 JQ 交叉验证
    'category': 'volume_structure',
    },

    'ret_open_2d_proxy': {
    'name': 'ret_open_2d_proxy',
    'local_source': 'factors/advanced_technical.py:1280 (RetOpen2DProxy)',
    'local_formula': 'mean(open / close.shift(1) - 1, 2)',
    'jq_data': 'price',
    'jq_fields': ['open', 'close'],
    'jq_window': 5,
    'direction': 'positive',
    'direction_note': '连续2日跳空向上→隔夜信息累积→动量延续',
    'jq_compute_function': '''
def compute_ret_open_2d_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    open_d = price_data.get('open', {})
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in open_d or s not in close_d:
            continue
        o = np.array(open_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        if len(o) < 3 or len(c) < 3:
            continue
        vals = []
        for t in range(-2, 0):
            ot = o[t]; cp = c[t - 1]
            if np.isnan(ot) or np.isnan(cp) or cp == 0:
                continue
            vals.append(ot / cp - 1)
        if len(vals) == 0:
            continue
        arr[i] = np.mean(vals)
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,  # 待 JQ 交叉验证
    'category': 'momentum',
    },

    'accrual_quality_proxy': {
    'name': 'accrual_quality_proxy',
    'local_source': 'factors/advanced_technical.py:1077 (AccrualQualityProxy)',
    'local_formula': '-std(ret, 20) / (mean(|ret|, 20) + 0.001)',
    'jq_data': 'price',
    'jq_fields': ['close'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '低CV→收益稳定→高应计质量→正向Alpha. 本地已在compute内取负',
    'jq_compute_function': '''
def compute_accrual_quality_proxy(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in close_d:
            continue
        c = np.array(close_d[s], dtype=float)
        c_arr = c[np.isfinite(c)]
        if len(c_arr) < 8:
            continue
        ret = np.diff(c_arr) / c_arr[:-1]
        if len(ret) < 5:
            continue
        ret_std = np.nanstd(ret[-20:]) if len(ret) >= 20 else np.nanstd(ret)
        ret_abs_mean = np.nanmean(np.abs(ret[-20:])) if len(ret) >= 20 else np.nanmean(np.abs(ret))
        if ret_abs_mean is None or np.isnan(ret_abs_mean):
            continue
        quality = -(ret_std / (ret_abs_mean + 0.001))
        arr[i] = quality
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,  # 待 JQ 交叉验证
    'category': 'fundamental_quality_proxy',
    },

    'volume_climax_reversal': {
    'name': 'volume_climax_reversal',
    'local_source': 'factors/advanced_technical.py:982 (VolumeClimaxReversal)',
    'local_formula': '-clip(vol.shift(1) / ma(vol,20).shift(1) - 1, lower=0)',
    'jq_data': 'price',
    'jq_fields': ['volume'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '天量后缩量→买方枯竭→反转上行',
    'jq_compute_function': '''
def compute_volume_climax_reversal(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v)]
        if len(v_arr) < 21:
            continue
        vol_ma = np.mean(v_arr[-21:-1])
        if vol_ma == 0:
            continue
        vol_ratio = v_arr[-2] / vol_ma - 1.0
        if vol_ratio > 0:
            arr[i] = -vol_ratio
        else:
            arr[i] = 0.0
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'volume_structure',
    },

    'money_flow_20': {
    'name': 'money_flow_20',
    'local_source': 'factors/advanced_technical.py:270 (MoneyFlow20)',
    'local_formula': '-((h+l+c)/3 * vol / ma((h+l+c)/3 * vol, 20) - 1)',
    'jq_data': 'price',
    'jq_fields': ['high', 'low', 'close', 'volume'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '资金持续流入→可能过热→负向, 取负为正',
    'jq_compute_function': '''
def compute_money_flow_20(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    high_d = price_data.get('high', {})
    low_d = price_data.get('low', {})
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in high_d or s not in low_d or s not in close_d or s not in vol_d:
            continue
        h = np.array(high_d[s], dtype=float)
        l = np.array(low_d[s], dtype=float)
        c = np.array(close_d[s], dtype=float)
        v = np.array(vol_d[s], dtype=float)
        min_len = min(len(h), len(l), len(c), len(v))
        if min_len < 21:
            continue
        h = h[-min_len:]; l = l[-min_len:]; c = c[-min_len:]; v = v[-min_len:]
        tp = (h + l + c) / 3.0
        mf = tp * v
        mf_ma = np.mean(mf[-20:])
        if mf_ma == 0:
            continue
        arr[i] = -(mf[-1] / mf_ma - 1.0)
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'sentiment',
    },

    'dollar_vol_20d': {
    'name': 'dollar_vol_20d',
    'local_source': 'factors/liquidity.py:43 (DollarVol20D)',
    'local_formula': '-log(mean(close * volume, 20))',
    'jq_data': 'price',
    'jq_fields': ['close', 'volume'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '低成交额→流动性溢价→正向(本地取负后zscore, 低成交额zscore高)',
    'jq_compute_function': '''
def compute_dollar_vol_20d(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.array(close_d[s], dtype=float)
        v = np.array(vol_d[s], dtype=float)
        min_len = min(len(c), len(v))
        if min_len < 10:
            continue
        dollar_vol = c[-min_len:] * v[-min_len:]
        avg_dv = np.nanmean(
            dollar_vol[-20:]) if len(dollar_vol) >= 20 else np.nanmean(dollar_vol)
        if np.isnan(avg_dv) or avg_dv <= 0:
            continue
        arr[i] = -np.log(avg_dv)
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'liquidity',
    },

    'turnover_cv_20d': {
    'name': 'turnover_cv_20d',
    'local_source': 'factors/liquidity.py:97 (TurnoverCv20D)',
    'local_formula': '-std(turnover, 20) / mean(turnover, 20) [本地用turnover数据; JQ用volume代理CV]',
    'jq_data': 'price',
    'jq_fields': ['volume'],
    'jq_window': 25,
    'direction': 'positive',
    'direction_note': '低CV→换手率稳定→流动性稳定→正向',
    'jq_compute_function': '''
def compute_turnover_cv_20d(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vol_d = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vol_d:
            continue
        v = np.array(vol_d[s], dtype=float)
        v_arr = v[np.isfinite(v) & (v > 0)]
        if len(v_arr) < 10:
            continue
        roll = v_arr[-20:] if len(v_arr) >= 20 else v_arr
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        cv = std_v / mean_v
        arr[i] = -cv
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'liquidity',
    },

    'turnover_std': {
    'name': 'turnover_std',
    'local_source': 'factors/turnover.py:46 (TurnoverStd)',
    'local_formula': '-std(turnover, 20) [本地daily_basic.turnover; JQ用valuation.turnover_rate真值]',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': ('本地=-20日换手率标准差(越稳越好). 改用JQ valuation.turnover_rate真值(与daily_basic.turnover同义=换手率%), '
    '在g.turnover_rate_hist维护每周快照, 取近20周std取负. '
    '消除volume-CV代理的截面漂移(与本地-std(turnover) rank相关仅0.56). '
    'get_fundamentals(q, date=context.previous_date)为gross_margin同款已验证可工作模式, '
    '规避"换手率取数全家桶三坑"(逐日loop静默空返回).'),
    'jq_compute_function': '''
def compute_turnover_std(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    # 1. 取当日 turnover_rate 快照 (与 gross_margin 同款已验证 get_fundamentals 模式)
    try:
        q = query(valuation.code, valuation.turnover_rate)
        q = q.filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[tstd] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    snap = {}
    for _, row in df.iterrows():
        code = row['code']
        tr = row.get('turnover_rate')
        if tr is not None and not (isinstance(tr, float) and np.isnan(tr)):
            snap[code] = float(tr)
    # 2. 维护滚动历史 (每周一个快照)
    if not hasattr(g, 'turnover_rate_hist') or g.turnover_rate_hist is None:
        g.turnover_rate_hist = {}
    for s in stocks:
        hist = g.turnover_rate_hist.setdefault(s, [])
        if s in snap:
            hist.append(snap[s])
        if len(hist) > 20:
            hist[:] = hist[-20:]
    # 3. 计算近20周 std 取负
    stock_idx = {s: i for i, s in enumerate(stocks)}
    valid = np.zeros(n, dtype=bool)
    for s, v in snap.items():
        if s in stock_idx:
            hist = g.turnover_rate_hist.get(s, [])
            hv = [x for x in hist if x is not None and not np.isnan(x)]
            if len(hv) >= 10:
                arr[stock_idx[s]] = -np.std(hv, ddof=1)
                valid[stock_idx[s]] = True
    return arr, valid
    ''',
    'verified': False,
    'category': 'turnover',
    },

    # ============================================================
    # 基本面因子 (fundamental) — 单次 get_fundamentals
    # ============================================================

    'gross_margin': {
    'name': 'gross_margin',
    'local_source': 'factors/profitability.py:192 (GrossMargin)',
    'local_formula': 'cross_sectional_zscore(indicator.gross_profit_margin)',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': '本地返回+zscore(gm), 偏好高毛利率. 实证ICIR依批次不同可能为正或负, 以本地run_fa.py产出日志为准',
    'jq_compute_function': '''
def compute_gross_margin(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.gross_profit_margin)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[gm] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        gm = row.get('gross_profit_margin')
        if code in stock_idx and gm is not None and not np.isnan(gm):
            arr[stock_idx[code]] = gm
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': True,  # v7.6 JQ验证通过
    'category': 'profitability',
    },

    'earnings_growth_yoy': {
    'name': 'earnings_growth_yoy',
    'local_source': 'factors/growth.py:27 (EarningsGrowthYoY)',
    'local_formula': 'cross_sectional_zscore(indicator.inc_net_profit_year_on_year)',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': '盈利增长→基本面改善→正向. 聚宽字段: indicator.inc_net_profit_year_on_year',
    'jq_compute_function': '''
def compute_earnings_growth_yoy(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.inc_net_profit_year_on_year)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[eg] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('inc_net_profit_year_on_year')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'growth',
    },

    'roe': {
    'name': 'roe',
    'local_source': 'factors/profitability.py:133 (ROE)',
    'local_formula': 'cross_sectional_zscore(indicator.roe)',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': '高ROE→盈利能力强→正向',
    'jq_compute_function': '''
def compute_roe(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roe)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        log.error('[roe] FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('roe')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'profitability',
    },

    'roa': {
    'name': 'roa',
    'local_source': 'factors/profitability.py:150 (ROA)',
    'local_formula': 'cross_sectional_zscore(indicator.roa)',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': '高ROA→资产利用效率高→正向',
    'jq_compute_function': '''
def compute_roa(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.roa)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('roa')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'profitability',
    },

    'net_margin': {
    'name': 'net_margin',
    'local_source': 'factors/profitability.py:216 (NetMargin)',
    'local_formula': 'cross_sectional_zscore(indicator.net_profit_margin)',
    'jq_data': 'fundamental',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': '高净利率→盈利质量好→正向',
    'jq_compute_function': '''
def compute_net_margin(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(indicator.code, indicator.net_profit_margin)
        q = q.filter(indicator.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
    except:
        return arr, np.zeros(n, dtype=bool)
    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        val = row.get('net_profit_margin')
        if code in stock_idx and val is not None and not np.isnan(val):
            arr[stock_idx[code]] = val
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,
    'category': 'profitability',
    },

    # ============================================================
    # 复合基本面因子 (fundamental_multi) — 多次 query
    # ============================================================

    'f_score': {
    'name': 'f_score',
    'local_source': 'factors/profitability.py:16 (PiotroskiFScore) — 9/9项全量',
    'local_formula': 'Piotroski F-Score 0-9: ROA>0+CFO>0+DeltaROA>0+Accrual<0+DeltaLever<0+DeltaCurRatio>0+NoEqOffer+DeltaMargin>0+DeltaTurn>0',
    'jq_data': 'fundamental_multi',
    'jq_fields': [],
    'jq_window': 0,
    'direction': 'positive',
    'direction_note': 'F-Score高→财务健康→正向. JQ仅6/9项可用(缺ocfps/assets_turn/roe_ttm子集), 但方向一致',
    'jq_compute_function': '''
def compute_f_score(stocks, context):
    n = len(stocks)
    arr = np.full(n, np.nan)
    try:
        q = query(
            indicator.code,
            indicator.roa, indicator.roe, indicator.gross_profit_margin,
            indicator.inc_net_profit_year_on_year
        ).filter(indicator.code.in_(stocks))
        df1 = get_fundamentals(q, date=context.previous_date)
    except:
        df1 = None
    try:
        q = query(
            balance.code, balance.current_ratio,
            balance.total_assets, balance.total_liability
        ).filter(balance.code.in_(stocks))
        df2 = get_fundamentals(q, date=context.previous_date)
    except:
        df2 = None
    try:
        q = query(cash_flow.code, cash_flow.net_operate_cash_flow).filter(cash_flow.code.in_(stocks))
        df3 = get_fundamentals(q, date=context.previous_date)
    except:
        df3 = None
    if df1 is None and df2 is None:
        return arr, np.zeros(n, dtype=bool)
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for s in stocks:
        idx = stock_idx[s]
        score = 0
        roa_v = None; gm_v = None; roe_v = None
        cr_v = None; lev_v = None; cfo_v = None
        if df1 is not None:
            row1 = df1[df1['code'] == s]
            if len(row1) > 0:
                roa_v = row1.iloc[0].get('roa')
                gm_v = row1.iloc[0].get('gross_profit_margin')
                roe_v = row1.iloc[0].get('roe')
        if df2 is not None:
            row2 = df2[df2['code'] == s]
            if len(row2) > 0:
                cr_v = row2.iloc[0].get('current_ratio')
                ta = row2.iloc[0].get('total_assets') or 0
                tl = row2.iloc[0].get('total_liability') or 0
                if ta > 0:
                    lev_v = tl / ta
        if df3 is not None:
            row3 = df3[df3['code'] == s]
            if len(row3) > 0:
                cfo_v = row3.iloc[0].get('net_operate_cash_flow')
        if roa_v is not None and roa_v > 0:
            score += 1
        if cfo_v is not None and cfo_v > 0:
            score += 1
        if roe_v is not None and roe_v > 0:
            score += 1
        if gm_v is not None and gm_v > 0:
            score += 1
        if cr_v is not None and cr_v > 1:
            score += 1
        if lev_v is not None and lev_v < 0.7:
            score += 1
        arr[idx] = score
    valid = np.isfinite(arr)
    return arr, valid
    ''',
    'verified': False,  # JQ仅6/9项可用
    'category': 'profitability',
    },
    }

# ============================================================
# 合并自动生成的 76 条缺失因子映射
# (gen_registry.py → registry_generated.py)
# ============================================================
from .registry_generated import REGISTRY_ADDITIONS
JQ_FACTOR_REGISTRY.update(REGISTRY_ADDITIONS)


def get_factor_jq_meta(factor_name):

    """获取单个因子的 JQ 元信息"""
    return JQ_FACTOR_REGISTRY.get(factor_name)


def list_available_factors(category=None, verified_only=False):
    """列出所有可用因子"""
    factors = []
    for name, meta in JQ_FACTOR_REGISTRY.items():
        if category and meta.get('category') != category:
            continue
        if verified_only and not meta.get('verified'):
            continue
        factors.append(name)
    return sorted(factors)


def get_required_price_fields(factor_names):
    """获取一组因子需要的所有 price 字段的并集"""
    fields = set()
    for name in factor_names:
        meta = JQ_FACTOR_REGISTRY.get(name)
        if meta and meta['jq_data'] == 'price':
            fields.update(meta['jq_fields'])
    return sorted(fields)


def get_max_window(factor_names):
    """获取一组因子需要的最大回溯窗口"""
    max_w = 0
    for name in factor_names:
        meta = JQ_FACTOR_REGISTRY.get(name)
        if meta:
            max_w = max(max_w, meta.get('jq_window', 0))
    return max_w
