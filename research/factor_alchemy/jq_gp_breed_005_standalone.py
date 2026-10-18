# -*- coding: utf-8 -*-
"""
gp_breed_005 单因子独立回测
因子公式: hl_ratio.rolling(5).mean()
生成时间: 2026-08-10 | 范式: 波动 | Local ICIR: 0.170 — 滞后IC + 内联报告)
用途: 独立验证 gp_breed_005 的 Rank IC/ICIR + 单因子选股收益
P-001 合规: 输出 per-factor IC/ICIR
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.lookback = 130
    g.trade_days = 0

    # --- IC 收集 (滞后法: 本周算上周因子→本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.ic_reported = False

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
    log.info("gp_breed_005 单因子 v2 (滞后IC) 启动")


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


# ═══ 因子: gp_breed_005 ═══
def factor_gp_breed_005(stocks, price_data):
    """hl_ratio.rolling(5).mean() -- 5日高低比均值"""
    n = len(stocks)
    arr = np.full(n, np.nan); valid = np.zeros(n, dtype=bool)
    highd = price_data.get('high', {}); lowd = price_data.get('low', {})
    for i, s in enumerate(stocks):
        h_arr = highd.get(s); l_arr = lowd.get(s)
        if h_arr is None or l_arr is None or len(h_arr) < 15: continue
        h = pd.Series(np.array(h_arr, dtype=float)); l = pd.Series(np.array(l_arr, dtype=float))
        try:
            hl = h / l
            result = hl.rolling(5).mean()
            v = result.dropna()
            arr[i] = v.iloc[-1] if len(v) > 0 else np.nan
            valid[i] = not pd.isna(arr[i])
        except Exception: continue
    return arr, valid
# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"gp_breed_005": factor_gp_breed_005}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] gp_breed_005: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  gp_breed_005  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 0: 滞后 IC 计算 (上周因子 → 本周收益)
    # ═══════════════════════════════════════════
    # 先加载当前周的 close（用于计算上周→本周的收益）
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0","3","6"))][:2000]

    # 快速获取当前周的 close 快照（只需要最新一条）
    px_c_snap = get_price(universe_all, count=2, end_date=prev_date,
                          frequency="daily", fields="close",
                          skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c_snap, _P):
        px_c_snap = px_c_snap["close"] if "close" in getattr(px_c_snap, "items", []) else px_c_snap.minor_xs("close")

    snap_closes = {}
    for s in universe_all:
        if s in px_c_snap.columns and len(px_c_snap[s].values) > 0:
            snap_closes[s] = float(px_c_snap[s].values[-1])

    # 用上周的因子值 + 本周收盘价 → 计算 Rank IC
    if len(g.last_factor_stocks) > 0 and g.last_factor_values is not None:
        fwd_rets = []
        factor_vals = []
        for i, s in enumerate(g.last_factor_stocks):
            prev_close = g.last_closes.get(s)
            curr_close = snap_closes.get(s)
            fv = g.last_factor_values[i]
            if (prev_close and prev_close > 0
                and curr_close and curr_close > 0
                and not np.isnan(fv)):
                ret = (curr_close / prev_close) - 1.0
                fwd_rets.append(ret)
                factor_vals.append(fv)

        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                g.ic_records.append(float(ic))
            except:
                pass

    # ═══════════════════════════════════════════
    # Step 1: 完整数据加载 & 因子计算
    # ═══════════════════════════════════════════
    filtered = []
    CHUNK = 500
    for k in range(0, len(universe_all), CHUNK):
        chunk = universe_all[k:k+CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused or cd[s].is_st: continue
                if cd[s].high_limit <= cd[s].last_price: continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:2000]
    if len(universe) < g.stock_num:
        return

    lb = g.lookback
    px_c = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="close", skip_paused=False, fq="pre")
    px_o = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="open", skip_paused=False, fq="pre")
    px_h = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="high", skip_paused=False, fq="pre")
    px_l = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="low", skip_paused=False, fq="pre")
    px_v = get_price(universe, count=lb, end_date=prev_date, frequency="daily", fields="volume", skip_paused=False, fq="pre")

    if _P is not None:
        if isinstance(px_c, _P): px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
        if isinstance(px_o, _P): px_o = px_o["open"] if "open" in getattr(px_o, "items", []) else px_o.minor_xs("open")
        if isinstance(px_h, _P): px_h = px_h["high"] if "high" in getattr(px_h, "items", []) else px_h.minor_xs("high")
        if isinstance(px_l, _P): px_l = px_l["low"] if "low" in getattr(px_l, "items", []) else px_l.minor_xs("low")
        if isinstance(px_v, _P): px_v = px_v["volume"] if "volume" in getattr(px_v, "items", []) else px_v.minor_xs("volume")

    price_data = {
        "close":  {s: px_c[s].values for s in universe if s in px_c.columns},
        "open":   {s: px_o[s].values for s in universe if s in px_o.columns},
        "high":   {s: px_h[s].values for s in universe if s in px_h.columns},
        "low":    {s: px_l[s].values for s in universe if s in px_l.columns},
        "volume": {s: px_v[s].values for s in universe if s in px_v.columns},
    }

    valid_u = [s for s in universe if len(price_data["close"].get(s, [])) >= 120]
    if len(valid_u) < g.stock_num:
        return

    arr, v = factor_gp_breed_005(valid_u, price_data)
    if int(np.sum(v)) < 10:
        return

    rp = rank_pct(arr, v)

    # ═══════════════════════════════════════════
    # Step 2: 保存快照供下周 IC 计算
    # ═══════════════════════════════════════════
    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in price_data["close"] and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓
    # ═══════════════════════════════════════════
    comp = np.where(v, rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100: continue
                order_target_value(s, 0)
            except:
                pass

    if len(top_stocks) == 0:
        return

    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            lp = get_current_data()[s].last_price
            if lp <= 0 or w / lp < 100: continue
            order_target_value(s, w)
            ordered += 1
        except:
            continue

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("gp_breed_005 | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    # 每 26 周 (约半年) 输出一次 IC 报告；最后也会自然打印
    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        g.ic_reported = True

    gc.collect()
