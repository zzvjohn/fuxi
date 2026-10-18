# -*- coding: utf-8 -*-
"""
V0.3 自动化因子组合策略 — Ralph Loop v0.3 生成
时间: 2026-08-07 18:29:50
因子: 4 个 (numpy向量化 + run_weekly)
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.lookback = 130
    g.trade_days = 0
    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("V0.3 Ralph Loop 策略: %d 因子" % 4)

# ═══ 辅助 ═══
def rank_pct(arr, valid):
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5); result[~valid] = 0.5; return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5); result[valid] = pct; result[~valid] = 0.5
    return result

# ═══ 因子 (动态生成 - per-stock pandas Series 计算) ═══
def factor_gp_breed_001(stocks, price_data):
    """((open_p / close_p.shift(1)) - (close_p.pct_change(5) - close_p.pct_change(20)))"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    opend = price_data.get("open", {})

    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30: continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        close = close_p  # alias        open_arr = opend.get(s)
        if open_arr is None or len(open_arr) < 30: continue
        open_p = pd.Series(np.array(open_arr, dtype=float))
        open = open_p  # alias

        try:
            result = ((open_p / close_p.shift(1)) - (close_p.pct_change(5) - close_p.pct_change(20))).rolling(5).mean()
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid

def factor_gp_breed_004(stocks, price_data):
    """(-(close_p.pct_change(20) * ((close_p.rolling(10).mean() / volume_p.rolling(30)."""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    volumed = price_data.get("volume", {})

    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30: continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        close = close_p  # alias        volume_arr = volumed.get(s)
        if volume_arr is None or len(volume_arr) < 30: continue
        volume_p = pd.Series(np.array(volume_arr, dtype=float))
        volume = volume_p  # alias

        try:
            result = (-(close_p.pct_change(20) * ((close_p.rolling(10).mean() / volume_p.rolling(30).mean()) - 1)))
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid

def factor_gp_breed_007(stocks, price_data):
    """(((volume_p / volume_p.rolling(60).mean()) - 1) * hl_ratio.ewm(span=50).mean().p"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    highd = price_data.get("high", {})
    lowd = price_data.get("low", {})
    volumed = price_data.get("volume", {})

    for i, s in enumerate(stocks):
        high_arr = highd.get(s)
        if high_arr is None or len(high_arr) < 30: continue
        high_p = pd.Series(np.array(high_arr, dtype=float))
        high = high_p  # alias        low_arr = lowd.get(s)
        if low_arr is None or len(low_arr) < 30: continue
        low_p = pd.Series(np.array(low_arr, dtype=float))
        low = low_p  # alias        volume_arr = volumed.get(s)
        if volume_arr is None or len(volume_arr) < 30: continue
        volume_p = pd.Series(np.array(volume_arr, dtype=float))
        volume = volume_p  # alias
        hl_ratio = high_p / low_p
        try:
            result = (((volume_p / volume_p.rolling(60).mean()) - 1) * hl_ratio.ewm(span=50).mean().pct_change(30).ewm(span=15).mean().pct_change(5))
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid

def factor_gp_breed_008(stocks, price_data):
    """((-(volume_p * low_p).rolling(20).std()) / (volume_p * close_p).rolling(20).mean"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    lowd = price_data.get("low", {})
    volumed = price_data.get("volume", {})

    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30: continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        close = close_p  # alias        low_arr = lowd.get(s)
        if low_arr is None or len(low_arr) < 30: continue
        low_p = pd.Series(np.array(low_arr, dtype=float))
        low = low_p  # alias        volume_arr = volumed.get(s)
        if volume_arr is None or len(volume_arr) < 30: continue
        volume_p = pd.Series(np.array(volume_arr, dtype=float))
        volume = volume_p  # alias

        try:
            result = ((-(volume_p * low_p).rolling(20).std()) / (volume_p * close_p).rolling(20).mean())
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══ 因子名→函数映射 ═══
FACTOR_FUNC_MAP = {

        "gp_breed_001": factor_gp_breed_001,
        "gp_breed_004": factor_gp_breed_004,
        "gp_breed_007": factor_gp_breed_007,
        "gp_breed_008": factor_gp_breed_008
}

# ═══ 复合适分 ═══
def compute_ensemble(stocks, price_data):
    n = len(stocks); composite = np.ones(n) * 0.5
    valid_any = np.zeros(n, dtype=bool)
    count = 0
    for name, func in FACTOR_FUNC_MAP.items():
        arr, v = func(stocks, price_data)
        if int(np.sum(v)) < 10: continue
        rp = rank_pct(arr, v)
        composite[v] *= rp[v]
        count += 1; valid_any = valid_any | v
    return composite, valid_any

# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5: return

    prev_date = context.previous_date

    # Universe
    universe = get_all_securities(["stock"], prev_date).index.tolist()
    CHUNK = 500; filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k+CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused or cd[s].is_st: continue
                if cd[s].high_limit <= cd[s].last_price: continue
                filtered.append(s)
        except Exception: filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:2000]
    if len(universe) < g.stock_num: return

    # Price data
    lb = g.lookback
    px_c = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="close", skip_paused=False, fq="pre")
    px_o = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="open", skip_paused=False, fq="pre")
    px_h = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="high", skip_paused=False, fq="pre")
    px_l = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="low", skip_paused=False, fq="pre")
    px_v = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="volume", skip_paused=False, fq="pre")

    # Panel兼容
    _P = getattr(pd, "Panel", None)
    if _P is not None:
        if isinstance(px_c, _P): px_c = px_c["close"] if "close" in getattr(px_c,"items",[]) else px_c.minor_xs("close")
        if isinstance(px_o, _P): px_o = px_o["open"] if "open" in getattr(px_o,"items",[]) else px_o.minor_xs("open")
        if isinstance(px_h, _P): px_h = px_h["high"] if "high" in getattr(px_h,"items",[]) else px_h.minor_xs("high")
        if isinstance(px_l, _P): px_l = px_l["low"] if "low" in getattr(px_l,"items",[]) else px_l.minor_xs("low")
        if isinstance(px_v, _P): px_v = px_v["volume"] if "volume" in getattr(px_v,"items",[]) else px_v.minor_xs("volume")

    price_data = {
        "close": {s: px_c[s].values for s in universe if s in px_c.columns},
        "open": {s: px_o[s].values for s in universe if s in px_o.columns},
        "high": {s: px_h[s].values for s in universe if s in px_h.columns},
        "low": {s: px_l[s].values for s in universe if s in px_l.columns},
        "volume": {s: px_v[s].values for s in universe if s in px_v.columns},
    }

    valid_u = [s for s in universe if len(price_data["close"].get(s,[])) >= 120]
    if len(valid_u) < g.stock_num: return

    composite, valid = compute_ensemble(valid_u, price_data)
    comp = np.where(valid, composite, np.inf)

    topn = g.stock_num
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100: continue
                order_target_value(s, 0)
            except: pass

    if len(top_stocks) == 0: return

    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            lp = get_current_data()[s].last_price
            if lp <= 0 or w/lp < 100: continue
            order_target_value(s, w); ordered += 1
        except: continue
    log.info("ordered %d/%d" % (ordered, len(top_stocks)))
    gc.collect()
