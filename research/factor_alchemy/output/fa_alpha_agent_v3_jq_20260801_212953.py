# -*- coding: utf-8 -*-
# FA AlphaAgent v3 -- LLM因子生成 + 三重约束 + V3集成
# 6 个复合, 等权 rank-product Ensemble
# Generated: 2026-08-01 21:29:53
#
# AlphaAgent 三重约束:
#   1. Originality: AST相似度 < 0.70 vs V3因子池
#   2. Hypothesis-Factor Alignment: 经济逻辑一致性 >= 3/5
#   3. Complexity: AST节点数 <= 28
#
# 入选复合:
#   comp1: comp1_overnight_tvma -- 隔夜累计x量价趋势: V3最强复合
#   comp2: comp2_dollar_turnover -- 成交额x换手稳定性: V3次强
#   comp3: comp3_moneyflow_ret3m -- 资金流x动量反转: V3原复合
#   comp4: comp5_alpha_gen1_000_tvma_20 -- 流动性质量 x 趋势: 日内振幅/成交量均值排名取负。高振幅低量=流动性差=流动性溢价。截面排名消除量纲
#   comp5: comp6_alpha_gen1_030_turnover_std_cv -- 动量 x 换手稳定性: 截面取负的(20日均-60日均)。趋势减弱=反转前兆。A股短周期特征。
#   comp6: comp7_alpha_gen1_051_money_flow_20 -- 波动率 x 资金流: 截面取负(5日价差)。价格窄幅波动=筹码锁定=即将突破。盘整信号。

import datetime
import numpy as np
import pandas as pd
import gc

BENCHMARK = '000300.XSHG'
TOP_N = 30
REBALANCE_DAY = 5
PRICE_WINDOW = 130
PRICE_FIELDS = ['close', 'open', 'high', 'low', 'volume']
CHUNK = 500

# ============================================================
# V3 底因子实现
# ============================================================

def compute_dollar_vol_20d(stocks, price_data):
    """成交额对数: -log(mean(close*volume, 20))"""
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 20 or len(v) < 20:
            continue
        c_arr = np.array(c[-20:], dtype=float)
        v_arr = np.array(v[-20:], dtype=float)
        dv = np.mean(v_arr * c_arr)
        arr[i] = -np.log(max(dv, 1e-8)); valid[i] = True
    return arr, valid

def compute_money_flow_20(stocks, price_data):
    """资金流: -((h+l+c)/3*vol / MA20((h+l+c)/3*vol) - 1)"""
    high = price_data.get("high", {})
    low = price_data.get("low", {})
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        h = high.get(s); l = low.get(s); c = close.get(s); v = volume.get(s)
        if h is None or l is None or c is None or v is None:
            continue
        if len(c) < 21 or len(v) < 21 or len(h) < 21 or len(l) < 21:
            continue
        h_arr = np.array(h[-21:], dtype=float)
        l_arr = np.array(l[-21:], dtype=float)
        c_arr = np.array(c[-21:], dtype=float)
        v_arr = np.array(v[-21:], dtype=float)
        mf = (h_arr + l_arr + c_arr) / 3.0 * v_arr
        ma20 = np.mean(mf[-20:])
        if ma20 < 1: continue
        arr[i] = -(mf[-1] / ma20 - 1.0); valid[i] = True
    return arr, valid

def compute_overnight_5d(stocks, price_data):
    """隔夜累积收益: rolling_sum(open/close.shift(1)-1, 5)"""
    close = price_data.get("close", {})
    open_p = price_data.get("open", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 6 or len(o) < 6:
            continue
        c_arr = np.array(c[-6:], dtype=float)
        o_arr = np.array(o[-6:], dtype=float)
        gaps = o_arr[1:] / c_arr[:-1] - 1.0
        arr[i] = np.sum(gaps); valid[i] = True
    return arr, valid

def compute_ret_3m(stocks, price_data):
    """3月动量反转: -(close/close.shift(60)-1)"""
    close = price_data.get("close", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        c = close.get(s)
        if c is None or len(c) < 61:
            continue
        c_arr = np.array(c[-61:], dtype=float)
        arr[i] = -(c_arr[-1] / max(c_arr[0], 1e-8) - 1.0); valid[i] = True
    return arr, valid

def compute_turnover_std_cv(stocks, price_data):
    """成交量变异系数(取负): -std(volume,20)/mean(volume,20)"""
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        v = volume.get(s)
        if v is None or len(v) < 20:
            continue
        v_arr = np.array(v[-20:], dtype=float)
        m = np.mean(v_arr)
        if m < 1: continue
        arr[i] = -np.std(v_arr) / m; valid[i] = True
    return arr, valid

def compute_tvma_20(stocks, price_data):
    """20日成交额均线偏离: -clip(close*volume/MA20(close*volume), 0, 10)"""
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 21 or len(v) < 21:
            continue
        c_arr = np.array(c[-21:], dtype=float)
        v_arr = np.array(v[-21:], dtype=float)
        amount = v_arr * c_arr
        ma20 = np.mean(amount[-20:])
        if ma20 < 1: continue
        ratio = np.clip(amount[-1] / ma20, 0, 10)
        arr[i] = -ratio; valid[i] = True
    return arr, valid


# ============================================================
# AlphaAgent 新因子实现 (LLM生成 + 三重约束)
# ============================================================

def compute_v1(stocks, price_data):
    """日内振幅/成交量均值排名取负。高振幅低量=流动性差=流动性溢价。"""
    high = price_data.get("high", {})
    low = price_data.get("low", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    raw = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 25
    for i, s in enumerate(stocks):
        if (len(high.get(s, [])) < W or len(low.get(s, [])) < W
            or len(volume.get(s, [])) < W):
            continue
        h_arr = np.array(high.get(s)[-W:], dtype=float)
        l_arr = np.array(low.get(s)[-W:], dtype=float)
        v_arr = np.array(volume.get(s)[-W:], dtype=float)
        amp = h_arr[-1] - l_arr[-1]
        vmean = np.mean(v_arr[-20:])
        if vmean < 1: continue
        raw[i] = amp / vmean; valid[i] = True
    # Cross-sectional rank percentile (0~1), then negate
    v = raw[valid]
    if len(v) < 3: return raw, valid
    order = np.argsort(np.argsort(v))
    raw[valid] = order / (len(v) - 1)
    arr = -raw
    return arr, valid

def compute_f_(stocks, price_data):
    """截面取负的(20日均-60日均)。趋势减弱=反转前兆。A股短周期特征。"""
    close = price_data.get("close", {})
    n = len(stocks)
    raw = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if close.get(s) is None or len(close.get(s, [])) < W:
            continue
        c_arr = np.array(close.get(s)[-W:], dtype=float)
        ma20 = np.mean(c_arr[-20:])
        ma60 = np.mean(c_arr[-60:])
        if ma60 < 0.01: continue
        raw[i] = ma20 - ma60; valid[i] = True
    # Cross-sectional rank percentile, then negate
    v = raw[valid]
    if len(v) < 3: return raw, valid
    order = np.argsort(np.argsort(v))
    raw[valid] = order / (len(v) - 1)
    arr = -raw
    return arr, valid

def compute_f_5(stocks, price_data):
    """截面取负(5日价差)。价格窄幅波动=筹码锁定=即将突破。盘整信号。"""
    close = price_data.get("close", {})
    n = len(stocks)
    raw = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 10
    for i, s in enumerate(stocks):
        if close.get(s) is None or len(close.get(s, [])) < W:
            continue
        c_arr = np.array(close.get(s)[-W:], dtype=float)
        rng = np.max(c_arr[-5:]) - np.min(c_arr[-5:])
        raw[i] = rng; valid[i] = True
    # Cross-sectional rank percentile, then negate
    v = raw[valid]
    if len(v) < 3: return raw, valid
    order = np.argsort(np.argsort(v))
    raw[valid] = order / (len(v) - 1)
    arr = -raw
    return arr, valid


# ============================================================
# 配对复合
# ============================================================

def compute_score_comp1_overnight_tvma(stocks, price_data):
    """隔夜累计x量价趋势: V3最强复合"""
    a, va = compute_overnight_5d(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_comp2_dollar_turnover(stocks, price_data):
    """成交额x换手稳定性: V3次强"""
    a, va = compute_dollar_vol_20d(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_comp3_moneyflow_ret3m(stocks, price_data):
    """资金流x动量反转: V3原复合"""
    a, va = compute_money_flow_20(stocks, price_data)
    b, vb = compute_ret_3m(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_comp5_alpha_gen1_000_tvma_20(stocks, price_data):
    """流动性质量 x 趋势: 日内振幅/成交量均值排名取负。高振幅低量=流动性差=流动性溢价。截面排名消除量纲"""
    a, va = compute_v1(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_comp6_alpha_gen1_030_turnover_std_cv(stocks, price_data):
    """动量 x 换手稳定性: 截面取负的(20日均-60日均)。趋势减弱=反转前兆。A股短周期特征。"""
    a, va = compute_f_(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_comp7_alpha_gen1_051_money_flow_20(stocks, price_data):
    """波动率 x 资金流: 截面取负(5日价差)。价格窄幅波动=筹码锁定=即将突破。盘整信号。"""
    a, va = compute_f_5(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order = np.argsort(np.argsort(b[vb]))
        rb[vb] = order / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid


# ============================================================
# Ensemble: 等权 rank-product
# ============================================================

def compute_ensemble(stocks, price_data):
    n_stocks = len(stocks)
    composite = np.ones(n_stocks)
    score_funcs = [
        (compute_score_comp1_overnight_tvma,),
        (compute_score_comp2_dollar_turnover,),
        (compute_score_comp3_moneyflow_ret3m,),
        (compute_score_comp5_alpha_gen1_000_tvma_20,),
        (compute_score_comp6_alpha_gen1_030_turnover_std_cv,),
        (compute_score_comp7_alpha_gen1_051_money_flow_20,),
    ]
    for (fn,) in score_funcs:
        arr, valid = fn(stocks, price_data)
        if not valid.any():
            continue
        arr = np.where(valid, arr, np.nan)
        arr = np.where(np.isnan(arr), 0.5, arr)
        composite *= np.power(arr + 1e-8, 0.16666666666666666)
    return composite

# ============================================================
# 数据加载 (V2模式)
# ============================================================

def load_price_data(stocks, end_dt, fields, window):
    result = {f: {} for f in fields}
    PanelClass = getattr(pd, 'Panel', None)
    for i in range(0, len(stocks), CHUNK):
        batch = stocks[i:i+CHUNK]
        for field in fields:
            try:
                px = get_price(batch, end_date=end_dt, count=window,
                               fields=field, frequency='daily', fq='pre',
                               skip_paused=False)
                if PanelClass is not None and isinstance(px, PanelClass):
                    df = px[field] if field in px.items else px.minor_xs(field)
                else:
                    df = px
                if isinstance(df, pd.DataFrame):
                    result[field].update({c: df[c].values for c in df.columns if c in batch})
            except Exception:
                continue
        del batch; gc.collect()
    return result


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('order', 'info')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0), type='stock')
    set_slippage(FixedSlippage(0.003))
    g.cycle = 0
    log.info('[Init] FA AlphaAgent v3 -- LLM + Triple Constraints + V3 Ensemble')
    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')


def rebalance(context):
    g.cycle += 1
    prev_date = context.previous_date

    all_stocks = list(get_all_securities(['stock'], prev_date).index)
    log.info(f'[W{g.cycle}] Step1: {len(all_stocks)} stocks')

    cd = get_current_data()
    universe = []
    npaused = nnew = 0
    for s in all_stocks:
        try:
            if cd[s].paused:
                npaused += 1; continue
            if 'ST' in get_security_info(s).display_name:
                continue
            si = get_security_info(s)
            if si and si.start_date:
                if (prev_date - si.start_date).days < 252:
                    nnew += 1; continue
            universe.append(s)
        except Exception:
            pass
    log.info(f'[W{g.cycle}] Step2: {len(universe)} kept (p:{npaused} n:{nnew})')

    if len(universe) < TOP_N:
        log.warn(f'[W{g.cycle}] < TOP_N SKIP'); return

    price_data = load_price_data(universe, prev_date, PRICE_FIELDS, PRICE_WINDOW)
    nclose = len(price_data.get('close', {}))
    log.info(f'[W{g.cycle}] Step3: close={nclose}')
    if nclose == 0:
        log.warn(f'[W{g.cycle}] ZERO data SKIP'); return

    composite = compute_ensemble(universe, price_data)
    valid = np.isfinite(composite)
    nvalid = int(np.sum(valid))
    if nvalid < TOP_N:
        log.warn(f'[W{g.cycle}] valid={nvalid} < TOP_N SKIP'); return

    scores = pd.Series(composite, index=universe)
    scores = scores[valid].dropna()
    top = scores.nlargest(TOP_N).index.tolist()

    cur = list(context.portfolio.positions.keys())
    wgt = 1.0 / TOP_N
    for s in cur:
        if s not in top:
            try: order_target_value(s, 0)
            except Exception: pass
    for s in top:
        try:
            if s in cd and cd[s].high_limit <= cd[s].last_price:
                continue
            order_target_value(s, context.portfolio.total_value * wgt)
        except Exception:
            pass

    if g.cycle <= 4 or g.cycle % 13 == 0:
        log.info(f'[W{g.cycle}] Hold {len(top)} | Val {context.portfolio.total_value:.0f}')