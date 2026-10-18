# -*- coding: utf-8 -*-
# FA LLM Pairs -- LLM 提案 + K-fold OOS 筛选
# 3 个配对复合, 等权 rank-product Ensemble
# Generated: 2026-08-01 13:07:38
#
# 入选配对 (K-fold avg_ICIR, cv, grade):
#   volume_climax_reversal                    +0.4458  cv=0.3457  B  (llm_pair_025)
#   earnings_volume_drift                     +0.4377  cv=0.2641  A  (llm_pair_024)
#   opening_gap_momentum__tvma_20             +0.3153  cv=0.3961  B  (llm_pair_008)

# 因子: earnings_volume_drift, opening_gap_momentum, tvma_20, volume_climax_reversal

import datetime
import numpy as np
import pandas as pd
import gc

BENCHMARK = '000300.XSHG'
TOP_N = 30
REBALANCE_DAY = 5
PRICE_WINDOW = 130
PRICE_FIELDS = ['close', 'open', 'volume']
CHUNK = 500

# ── 因子实现 ──

def compute_earnings_volume_drift(stocks, price_data):
    """缩量上涨: clip(ret_5d, [-0.3,0.3]) * clip(-vol_chg, [-1,2]), ret_5d>0 only"""
    close = price_data.get("close", {})
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s); v = volume.get(s)
        if c is None or v is None or len(c) < 12 or len(v) < 12:
            continue
        c_arr = np.array(c[-12:], dtype=float)
        v_arr = np.array(v[-12:], dtype=float)
        ret_5d = c_arr[-1] / max(c_arr[-6], 1e-8) - 1.0
        if ret_5d <= 0:
            arr[i] = 0; valid[i] = True; continue
        m5 = np.mean(v_arr[-5:])
        m5p = np.mean(v_arr[-11:-6]) if len(v_arr) >= 11 else np.mean(v_arr[-10:-5])
        if m5p < 1:
            continue
        vol_chg = m5 / m5p - 1.0
        signal = np.clip(ret_5d, -0.3, 0.3) * np.clip(-vol_chg, -1.0, 2.0)
        arr[i] = signal; valid[i] = True
    return arr, valid

def compute_opening_gap_momentum(stocks, price_data):
    """开盘跳空动量: (open/close.shift(1)-1).rolling(3).mean()"""
    close = price_data.get("close", {})
    open_p = price_data.get("open", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s); o = open_p.get(s)
        if c is None or o is None or len(c) < 5 or len(o) < 5:
            continue
        c_arr = np.array(c[-5:], dtype=float)
        o_arr = np.array(o[-5:], dtype=float)
        gaps = o_arr[1:] / c_arr[:-1] - 1.0  # 最近4个跳空
        if len(gaps) < 3:
            continue
        arr[i] = np.mean(gaps[-3:]); valid[i] = True
    return arr, valid

def compute_tvma_20(stocks, price_data):
    """20日成交额均线: MA20(volume*close)"""
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
        amount = v_arr * c_arr
        arr[i] = np.mean(amount); valid[i] = True
    return arr, valid

def compute_volume_climax_reversal(stocks, price_data):
    """天量反转: -(vol_lag1/vol_ma20_lag1 - 1).clip(lower=0)"""
    volume = price_data.get("volume", {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = volume.get(s)
        if v is None or len(v) < 22:
            continue
        v_arr = np.array(v[-22:], dtype=float)
        v_ma = np.mean(v_arr)
        if v_ma < 1:
            continue
        v_lag = v_arr[-2]  # vol_shift(1)
        ratio = v_lag / v_ma - 1.0
        arr[i] = -max(ratio, 0)
        valid[i] = True
    return arr, valid

# ── 配对复合 & 单因子 ──

def compute_score_volume_climax_reversal(stocks, price_data):
    """volume_climax_reversal -- standalone rank-pct"""
    a, va = compute_volume_climax_reversal(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    arr = ra
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_earnings_volume_drift(stocks, price_data):
    """earnings_volume_drift -- standalone rank-pct"""
    a, va = compute_earnings_volume_drift(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5)
    if va.any():
        order = np.argsort(np.argsort(a[va]))
        ra[va] = order / (va.sum() - 1) if va.sum() > 1 else ra[va]
    arr = ra
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

def compute_score_opening_gap_momentum__tvma_20(stocks, price_data):
    """opening_gap_momentum x tvma_20 -> rank-product"""
    a, va = compute_opening_gap_momentum(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    n = len(stocks)
    ra = np.full(n, 0.5); rb = np.full(n, 0.5)
    if va.any():
        order_a = np.argsort(np.argsort(a[va]))
        ra[va] = order_a / (va.sum() - 1) if va.sum() > 1 else ra[va]
    if vb.any():
        order_b = np.argsort(np.argsort(b[vb]))
        rb[vb] = order_b / (vb.sum() - 1) if vb.sum() > 1 else rb[vb]
    arr = ra * rb
    valid = np.isfinite(arr) & (arr > 0)
    return arr, valid

# ── Ensemble ──

def compute_ensemble(stocks, price_data):
    n_stocks = len(stocks)
    composite = np.ones(n_stocks)
    score_funcs = [
        (compute_score_volume_climax_reversal,),
        (compute_score_earnings_volume_drift,),
        (compute_score_opening_gap_momentum__tvma_20,),
    ]
    for (fn,) in score_funcs:
        arr, valid = fn(stocks, price_data)
        if not valid.any():
            continue
        arr = np.where(valid, arr, np.nan)
        arr = np.where(np.isnan(arr), 0.5, arr)
        composite *= np.power(arr + 1e-8, 0.3333)
    return composite


# ── 数据加载 (V2模式) ──

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


# ── 初始化和调仓 ──

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
    log.info("[Init] FA LLM Pairs -- 3 pairs, K-fold OOS screened")
    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')


def rebalance(context):
    g.cycle += 1
    prev_date = context.previous_date

    all_stocks = list(get_all_securities(['stock'], prev_date).index)
    log.info(f"[W{g.cycle}] Step1: {len(all_stocks)} stocks")

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
    log.info(f"[W{g.cycle}] Step2: {len(universe)} kept (p:{npaused} n:{nnew})")

    if len(universe) < TOP_N:
        log.warn(f"[W{g.cycle}] < TOP_N SKIP"); return

    price_data = load_price_data(universe, prev_date, PRICE_FIELDS, PRICE_WINDOW)
    nclose = len(price_data.get('close', {})); nvol = len(price_data.get('volume', {}))
    log.info(f"[W{g.cycle}] Step3: close={nclose} vol={nvol}")
    if nclose == 0:
        log.warn(f"[W{g.cycle}] ZERO data SKIP"); return

    composite = compute_ensemble(universe, price_data)
    valid = np.isfinite(composite)
    nvalid = int(np.sum(valid))
    if nvalid < TOP_N:
        log.warn(f"[W{g.cycle}] valid={nvalid} < TOP_N SKIP"); return

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
        log.info(f"[W{g.cycle}] Hold {len(top)} | Val {context.portfolio.total_value:.0f}")
