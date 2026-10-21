# -*- coding: utf-8 -*-
"""
E 道幸存候选 #0 单因子独立回测 (GFlowNet 产出)
================================================================
候选表达式 : add(ln_mcap, atr_14)
可读含义   : 对数市值(小市值) + 14日ATR(低波动) 的等权相加, 做多「小市值+低波动」

因子公式 (逐位对齐 factor_alchemy/factors/{size,advanced_technical}.py):
  ln_mcap   = cross_sectional_zscore( -log(market_cap) )      # 中位数去中心, std 标准化, 负向(小市值→高)
  atr_14    = - ATR14 / close                                  # TR=EMA(TR,14), 负向(低波动→高), 不做截面标准化
  composite = ln_mcap + atr_14                                  # 注意: 量纲不对称(zscore vs 原始比), 忠实复刻缓存口径
  signal    = cross_sectional_rank(composite)                   # 截面百分位, HIGH=做多

方向: HIGH = 做多 (E1 实测 signed ic_mean = +0.0525)
E 道表现: E1 ic_mean=0.0454 覆盖=89.1% | 收缩ICIR=0.229 | E1-E4 全过

P-001 合规: 回测结束输出单因子 Rank IC / ICIR (均值)
E5 预注册说明: 本候选为纯截面(点状时点)因子, 无历史时序标准化 → 无 lookahead,
             JQ 回测口径与 E1-E4 完全一致, 是三者中最忠实的 E5 前向复验。
Local: 手写 (2026-10-21, E 道移植 Path B)
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 50
    g.lookback = 30            # ATR14 需要 ~15 交易日, 30 足够
    g.trade_days = 0

    # --- IC 收集 (滞后法: 本周算上周因子 → 本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}

    set_benchmark("000905.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    log.set_level("system", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("E道候选#0 add(ln_mcap, atr_14) 单因子回测启动")


# ═══ 辅助 ═══
def rank_pct(arr, valid):
    """截面百分位 rank → [0,1], 与本地 data_loader.cross_sectional_rank 一致。"""
    n = len(arr)
    out = np.full(n, 0.5)
    v = arr[valid]
    if len(v) < 3:
        return out
    from scipy.stats import rankdata
    ranks = rankdata(v)
    out[valid] = (ranks - 1) / (len(v) - 1)
    return out


def cs_zscore_median(arr, valid):
    """截面中位数去中心 + std 标准化 (复刻 factor_alchemy cross_sectional_zscore, min 10)。"""
    n = len(arr)
    out = np.full(n, np.nan)
    v = arr[valid]
    if len(v) < 10:
        return out
    center = np.median(v)
    scale = v.std()
    if scale > 0:
        out[valid] = (arr[valid] - center) / scale
    return out


def _get_mcap_map(universe, date):
    """get_fundamentals 取 market_cap(亿元) → {code: float}。valuation 为 JQ 引擎全局名。"""
    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(universe))
    df = get_fundamentals(q, date=date)
    if df is None or len(df) == 0:
        return {}
    if "code" not in df.columns or "market_cap" not in df.columns:
        return {}
    m = {}
    for _, row in df.iterrows():
        c = row["code"]
        v = row["market_cap"]
        if c and (v == v) and v > 0:
            m[c] = float(v)
    return m


# ═══ 因子: add(ln_mcap, atr_14) ═══
def factor_add_ln_mcap_atr14(stocks, close_df, high_df, low_df, mcap_map):
    """返回 (arr, valid)。close/high/low_df: DataFrame (days × stocks)。"""
    close = close_df.reindex(columns=stocks)
    high = high_df.reindex(columns=stocks)
    low = low_df.reindex(columns=stocks)

    # --- atr_14 = -ATR14/close ---
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1.values, tr2.values), tr3.values)
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    atr = tr.ewm(span=14, min_periods=14, axis=0).mean()
    atr_norm = (atr / close).iloc[-1]          # 最后一行 = prev_date
    atr_factor = -atr_norm                     # 负向

    # --- ln_mcap = cs_zscore(-log(mcap)) ---
    mcap_arr = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        if s in mcap_map:
            mcap_arr[i] = mcap_map[s]
    mcap_valid = np.isfinite(mcap_arr) & (mcap_arr > 0)
    ln_mcap_raw = np.full(len(stocks), np.nan)
    ln_mcap_raw[mcap_valid] = -np.log(mcap_arr[mcap_valid])
    ln_mcap = cs_zscore_median(ln_mcap_raw, mcap_valid)

    # --- composite = ln_mcap + atr_14 ---
    comp = np.full(len(stocks), np.nan)
    for i, s in enumerate(stocks):
        a = atr_factor.get(s, np.nan)
        b = ln_mcap[i]
        if np.isfinite(a) and np.isfinite(b):
            comp[i] = a + b
    valid = np.isfinite(comp)
    return comp, valid


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] add(ln_mcap, atr_14): NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  add(ln_mcap, atr_14)  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    g.trade_days += 1
    if g.trade_days < 5:
        return
    prev_date = context.previous_date

    # Step 0: 股票池过滤
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]
    filtered = []
    try:
        cd = get_current_data()
        for s in universe_all:
            try:
                if cd[s].paused or cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
            except Exception:
                continue
    except Exception:
        filtered = list(universe_all)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:450]
    if len(universe) < g.stock_num:
        return

    # Step 1: 取价 (close/high/low 分三次独立查询)
    px_c = get_price(universe, count=g.lookback, end_date=prev_date, frequency="daily",
                     fields="close", skip_paused=False, fq="pre")
    px_h = get_price(universe, count=g.lookback, end_date=prev_date, frequency="daily",
                     fields="high", skip_paused=False, fq="pre")
    px_l = get_price(universe, count=g.lookback, end_date=prev_date, frequency="daily",
                     fields="low", skip_paused=False, fq="pre")
    # JQ 引擎 get_price 默认返回 Panel(3D)，即使单字段也要拆成 2D DataFrame (dates × stocks)
    _P = getattr(pd, "Panel", None)
    if _P is not None:
        if isinstance(px_c, _P):
            px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
        if isinstance(px_h, _P):
            px_h = px_h["high"] if "high" in getattr(px_h, "items", []) else px_h.minor_xs("high")
        if isinstance(px_l, _P):
            px_l = px_l["low"] if "low" in getattr(px_l, "items", []) else px_l.minor_xs("low")
    if px_c is None or px_c.shape[0] < 15 or px_c.shape[1] == 0:
        return
    valid_u = [s for s in universe
               if s in px_c.columns and int((~px_c[s].isna()).sum()) >= 15]
    if len(valid_u) < g.stock_num:
        return

    mcap_map = _get_mcap_map(valid_u, prev_date)
    try:
        arr, v = factor_add_ln_mcap_atr14(valid_u, px_c, px_h, px_l, mcap_map)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] 因子计算异常: %r at %s" % (_fe, prev_date))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        return

    rp = rank_pct(arr, v)

    # Step 1.5: 滞后 IC
    snap_closes = {}
    for s in px_c.columns:
        lv = px_c[s].iloc[-1]
        if np.isfinite(lv):
            snap_closes[s] = float(lv)
    if len(g.last_factor_stocks) > 0 and g.last_factor_values is not None:
        fwd_rets, factor_vals = [], []
        for i, s in enumerate(g.last_factor_stocks):
            prev_close = g.last_closes.get(s)
            curr_close = snap_closes.get(s)
            fv = g.last_factor_values[i]
            if (prev_close and prev_close > 0 and curr_close and curr_close > 0
                    and not np.isnan(fv)):
                fwd_rets.append((curr_close / prev_close) - 1.0)
                factor_vals.append(fv)
        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_vals, fwd_rets)
                if ic == ic:
                    g.ic_records.append(float(ic))
            except Exception:
                pass

    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # Step 2: 选股 (HIGH=做多: 取最高 rp)
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    cd_order = get_current_data()
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                if cd_order[s].paused:
                    continue
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target(s, 0)
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    w = context.portfolio.total_value / len(top_stocks)
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break
            d = cd_order[s]
            if d.paused or d.is_st or d.high_limit <= d.last_price:
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or w / ref_px < 100:
                continue
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, w)
        except Exception:
            continue

    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("add(ln_mcap, atr_14) | week=%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, int(np.sum(v)), latest_ic, len(g.ic_records)))
    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
