# -*- coding: utf-8 -*-
"""
V0.3 自动化因子组合策略 — Ralph Loop v0.3 生成
时间: 2026-08-07 17:31:27 (修复版: numpy向量化 + run_weekly)
因子: 6 个 (S5通过 + 真实因子库)
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.lookback = 130  # 覆盖 max(factor windows)
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
    log.info("V0.3 Ralph Loop 策略: 6 因子, Top-80 等权, 周频")

# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def rank_pct(arr, valid):
    """截面百分位排名, NaN被忽略"""
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5)
        result[~valid] = 0.5
        return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    result[~valid] = 0.5
    return result

def rolling_std(arr, window):
    """numpy rolling std"""
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanstd(arr[i - window + 1:i + 1])
    return result

def rolling_mean(arr, window):
    """numpy rolling mean"""
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmean(arr[i - window + 1:i + 1])
    return result

# ═══════════════════════════════════════════════════════════
# 因子计算 (纯 numpy, 每只股独立)
# ═══════════════════════════════════════════════════════════

def factor_volume_stability(stocks, price_data):
    """-(volume 20日 std / volume 20日均值) — 量能稳定性"""
    n = len(stocks)
    volume_dict = price_data.get("volume", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = volume_dict.get(s)
        if v is None or len(v) < 25:
            continue
        v = np.array(v, dtype=float)[-20:]
        std_v = np.nanstd(v)
        mean_v = np.nanmean(v)
        if mean_v > 0:
            arr[i] = -(std_v / mean_v)
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def factor_dollar_vol_stability(stocks, price_data):
    """-(成交额 20日 std / 成交额 20日均值) — 成交额稳定性"""
    n = len(stocks)
    close_dict = price_data.get("close", {})
    volume_dict = price_data.get("volume", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        v = volume_dict.get(s)
        if c is None or v is None or len(c) < 25 or len(v) < 25:
            continue
        c = np.array(c, dtype=float)[-20:]
        v = np.array(v, dtype=float)[-20:]
        dv = c * v
        std_dv = np.nanstd(dv)
        mean_dv = np.nanmean(dv)
        if mean_dv > 0:
            arr[i] = -(std_dv / mean_dv)
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def factor_trend_smoothness(stocks, price_data):
    """close 20日 R² 趋势拟合 — 趋势平滑度"""
    n = len(stocks)
    close_dict = price_data.get("close", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    x = np.arange(20)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 25:
            continue
        c = np.array(c, dtype=float)[-20:]
        if np.nanstd(c) < 1e-10:
            arr[i] = 0.0
            valid[i] = True
            continue
        valid_mask = ~np.isnan(c)
        if valid_mask.sum() < 10:
            continue
        corr = np.corrcoef(x[valid_mask], c[valid_mask])[0, 1]
        arr[i] = corr ** 2
        valid[i] = True
    return arr, valid

def factor_relative_value_zscore(stocks, price_data):
    """-(close - ma120) / std120 — 相对价值zscore(取负=做多低估)"""
    n = len(stocks)
    close_dict = price_data.get("close", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 125:
            continue
        c = np.array(c, dtype=float)[-125:]
        ma = np.nanmean(c[-120:])
        std = np.nanstd(c[-120:])
        if std > 0:
            arr[i] = -(c[-1] - ma) / std
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def factor_range_consistency(stocks, price_data):
    """-(high/low - 1) 20日 std — 振幅一致性"""
    n = len(stocks)
    high_dict = price_data.get("high", {})
    low_dict = price_data.get("low", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        h = high_dict.get(s)
        l = low_dict.get(s)
        if h is None or l is None or len(h) < 25 or len(l) < 25:
            continue
        h = np.array(h, dtype=float)[-20:]
        l = np.array(l, dtype=float)[-20:]
        ratio = h / l - 1
        std_r = np.nanstd(ratio)
        if not np.isnan(std_r):
            arr[i] = -std_r
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def factor_vol_of_vol(stocks, price_data):
    """-ret_5d_std 的 20日 std — 波动率的波动率(取负)"""
    n = len(stocks)
    close_dict = price_data.get("close", {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 50:
            continue
        c = np.array(c, dtype=float)[-50:]
        ret = c[1:] / c[:-1] - 1
        # rolling 5d std
        vol5 = np.full(len(ret), np.nan)
        for j in range(4, len(ret)):
            vol5[j] = np.nanstd(ret[j-4:j+1])
        vol5 = vol5[~np.isnan(vol5)]
        if len(vol5) < 25:
            continue
        vol_of_vol = np.nanstd(vol5[-20:])
        if not np.isnan(vol_of_vol):
            arr[i] = -vol_of_vol
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

# ═══════════════════════════════════════════════════════════
# 复合评分
# ═══════════════════════════════════════════════════════════

def compute_ensemble(stocks, price_data):
    """Rank-pct 乘积复合: 6 因子等权 rank 乘积"""
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid_any = np.zeros(n, dtype=bool)

    factor_funcs = [
        factor_volume_stability,
        factor_dollar_vol_stability,
        factor_trend_smoothness,
        factor_relative_value_zscore,
        factor_range_consistency,
        factor_vol_of_vol,
    ]

    count = 0
    for func in factor_funcs:
        arr, v = func(stocks, price_data)
        n_valid = int(np.sum(v))
        if n_valid < 10:
            continue
        rp = rank_pct(arr, v)
        composite[v] *= rp[v]
        count += 1
        valid_any = valid_any | v

    return composite, valid_any

# ═══════════════════════════════════════════════════════════
# 周度调仓
# ═══════════════════════════════════════════════════════════

def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:  # warmup
        return

    prev_date = context.previous_date

    # ── Universe ─────────────────────────────────────────
    universe = get_all_securities(["stock"], prev_date).index.tolist()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k + CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused:
                    continue
                if cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:2000]
    if len(universe) < g.stock_num:
        return

    # ── Load price data ──────────────────────────────────
    lookback = g.lookback
    px_close = get_price(universe, count=lookback, end_date=prev_date,
                         frequency="daily", fields="close",
                         skip_paused=False, fq="pre")
    px_open = get_price(universe, count=lookback, end_date=prev_date,
                        frequency="daily", fields="open",
                        skip_paused=False, fq="pre")
    px_high = get_price(universe, count=lookback, end_date=prev_date,
                        frequency="daily", fields="high",
                        skip_paused=False, fq="pre")
    px_low = get_price(universe, count=lookback, end_date=prev_date,
                       frequency="daily", fields="low",
                       skip_paused=False, fq="pre")
    px_volume = get_price(universe, count=lookback, end_date=prev_date,
                          frequency="daily", fields="volume",
                          skip_paused=False, fq="pre")

    # Panel 兼容
    _Panel = getattr(pd, "Panel", None)
    if _Panel is not None:
        if isinstance(px_close, _Panel):
            px_close = px_close["close"] if "close" in getattr(px_close, "items", []) else px_close.minor_xs("close")
        if isinstance(px_open, _Panel):
            px_open = px_open["open"] if "open" in getattr(px_open, "items", []) else px_open.minor_xs("open")
        if isinstance(px_high, _Panel):
            px_high = px_high["high"] if "high" in getattr(px_high, "items", []) else px_high.minor_xs("high")
        if isinstance(px_low, _Panel):
            px_low = px_low["low"] if "low" in getattr(px_low, "items", []) else px_low.minor_xs("low")
        if isinstance(px_volume, _Panel):
            px_volume = px_volume["volume"] if "volume" in getattr(px_volume, "items", []) else px_volume.minor_xs("volume")

    price_data = {
        "close": {s: px_close[s].values for s in universe if s in px_close.columns},
        "open": {s: px_open[s].values for s in universe if s in px_open.columns},
        "high": {s: px_high[s].values for s in universe if s in px_high.columns},
        "low": {s: px_low[s].values for s in universe if s in px_low.columns},
        "volume": {s: px_volume[s].values for s in universe if s in px_volume.columns},
    }

    valid_universe = [s for s in universe
                      if len(price_data["close"].get(s, [])) >= 120]
    if len(valid_universe) < g.stock_num:
        return

    # ── Compute composite ────────────────────────────────
    composite, valid = compute_ensemble(valid_universe, price_data)
    composite_arr = np.where(valid, composite, np.inf)

    # ── Select top-N ─────────────────────────────────────
    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[:topn]
    top_stocks = [valid_universe[i] for i in top_idx
                  if composite_arr[i] < np.inf]

    # ── Sell ─────────────────────────────────────────────
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target_value(s, 0)
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    # ── Buy ──────────────────────────────────────────────
    weight = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            last_px = get_current_data()[s].last_price
            if last_px <= 0 or weight / last_px < 100:
                continue
            order_target_value(s, weight)
            ordered += 1
        except Exception:
            continue
    log.info("ordered %d/%d stocks" % (ordered, len(top_stocks)))
    gc.collect()
