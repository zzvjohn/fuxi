# -*- coding: utf-8 -*-
"""Fix indentation of all broken jq_compute_function blocks in registry.py.
Each block lost its nested indentation; we rewrite with correct 4-space levels.
"""
import re

PATH = 'registry.py'
src = open(PATH, encoding='utf-8').read()

FIXED = {}

FIXED['compute_gap_up'] = '''def compute_gap_up(stocks, price_data):
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
'''

FIXED['compute_turnover_change'] = '''def compute_turnover_change(stocks, price_data):
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
'''

FIXED['compute_vm_diff'] = '''def compute_vm_diff(stocks, price_data):
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
'''

FIXED['compute_ret_open_2d_proxy'] = '''def compute_ret_open_2d_proxy(stocks, price_data):
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
'''

FIXED['compute_accrual_quality_proxy'] = '''def compute_accrual_quality_proxy(stocks, price_data):
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
'''

FIXED['compute_volume_climax_reversal'] = '''def compute_volume_climax_reversal(stocks, price_data):
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
'''

FIXED['compute_money_flow_20'] = '''def compute_money_flow_20(stocks, price_data):
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
'''

FIXED['compute_turnover_cv_20d'] = '''def compute_turnover_cv_20d(stocks, price_data):
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
'''

FIXED['compute_turnover_std'] = '''def compute_turnover_std(stocks, context):
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
'''

FIXED['compute_gross_margin'] = '''def compute_gross_margin(stocks, context):
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
'''

FIXED['compute_earnings_growth_yoy'] = '''def compute_earnings_growth_yoy(stocks, context):
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
'''

FIXED['compute_roe'] = '''def compute_roe(stocks, context):
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
'''

FIXED['compute_roa'] = '''def compute_roa(stocks, context):
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
'''

FIXED['compute_net_margin'] = '''def compute_net_margin(stocks, context):
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
'''

FIXED['compute_f_score'] = '''def compute_f_score(stocks, context):
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
'''

count = 0
for name, fixed in FIXED.items():
    pat = re.compile(r"def " + name + r"\([^)]*\):.*?\n'''", re.DOTALL)
    new_src, n = pat.subn(fixed + "'''", src)
    if n == 0:
        print('WARN: no match for', name)
    elif n > 1:
        print('WARN: multiple matches for', name, n)
    src = new_src
    count += n

open(PATH, 'w', encoding='utf-8').write(src)
print('replaced blocks:', count)

# validate
blocks = re.findall(r"'jq_compute_function':\s*'''(.*?)'''", src, re.DOTALL)
bad = 0
for b in blocks:
    fn = re.search(r'def (\w+)\(', b)
    nm = fn.group(1) if fn else '?'
    try:
        ast_par = __import__('ast')
        ast_par.parse(b)
    except SyntaxError as e:
        bad += 1
        print('  STILL BROKEN:', nm, e.lineno, e.msg)
print('total blocks:', len(blocks), '| broken:', bad)
