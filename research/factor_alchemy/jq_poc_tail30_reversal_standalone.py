# -*- coding: utf-8 -*-
"""
尾盘30分钟「反转」单因子独立回测 — 分钟因子 JQ 验证通道 (P-004 一期 · 反向验证)
================================================================================
背景: tail30_momentum 做多版 2021-2026 JQ 实测 -59.37%/Sharpe -0.866/MDD 62%,
      IC 5.5年稳定为负 (半年ICIR -0.13~-0.53, IC>0率41-43%) → 方向证伪。
      IC 稳定为负意味着反向 (做多尾盘走弱股 = 尾盘跳水次日反弹) 可能含正 alpha。
本文件: 在已修复执行层 (手数取整) 的骨架上, 将选股方向翻转为
      tail30_reversal = -(tail30_momentum), 即买入上交易日尾盘30分钟涨幅最低的股票。

数据源:   get_price(unit='5m', count=12, end_date=prev_date 15:00) — 单次调用全池向量化
          池 800 只 × 12 根 = 9600 行, 远低于 5.5万行红线 (JQ 铁律)
防未来:   用 context.previous_date 的分钟数据, 下周开盘调仓, 无前视
通道状态: 已由做多版确认 ([DIAG] minute OK rows=9600), 无需再验证通道

JQ 平台设置: 回测区间 2021-01-01 ~ 2026-08-13, 周频, 沪深300基准
"""
import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 60
    g.trade_days = 0

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}

    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type="stock")
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time="open", reference_security="000300.XSHG")
    log.info("tail30_reversal POC (尾盘30分钟反转方向验证) 启动")


# ═══ 辅助 ═══
def rank_pct(arr, valid):
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


# ═══ 因子: tail30_momentum (分钟宽表向量化) ═══
def factor_tail30_momentum(stocks, prev_date):
    """prev_date 最后30分钟涨跌幅 = close(15:00)/close(14:30)-1。

    实现: 单次 get_price(unit='5m', count=12) 拉全池上一交易日最后 12 根 5m bar
          (14:00~15:00), 宽表 pivot 后向量化: bar[-1]/bar[-7]-1。
    铁律 #14: 禁止逐只 eval, 必须宽表向量运算。
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid

    end_dt = pd.Timestamp(prev_date) + pd.Timedelta(hours=15)

    def _pull(fq):
        kw = dict(security=stocks, count=12, end_date=end_dt,
                  frequency="5m", fields="close",
                  skip_paused=False, panel=False)
        if fq is not None:
            kw["fq"] = fq
        return get_price(**kw)

    df = None
    try:
        df = _pull("pre")
    except Exception as e1:
        if not getattr(g, "diag_min_fq", False):
            log.info("[DIAG] minute FAIL (fq=pre): %r — 尝试不复权" % (e1,))
            g.diag_min_fq = True
        try:
            df = _pull(None)
        except Exception as e2:
            if not getattr(g, "diag_min_fail", False):
                log.info("[DIAG] minute FAIL (fq=None): %r" % (e2,))
                g.diag_min_fail = True
            return arr, valid

    if df is None or len(df) == 0:
        if not getattr(g, "diag_min_empty", False):
            log.info("[DIAG] minute EMPTY at %s" % prev_date)
            g.diag_min_empty = True
        return arr, valid
    if not getattr(g, "diag_min_ok", False):
        log.info("[DIAG] minute OK rows=%d cols=%s dtypes=%s" % (
            len(df), list(df.columns), {c: str(df[c].dtype) for c in df.columns}))
        g.diag_min_ok = True

    # 长表 (time, code, close) -> 宽表 (index=time, columns=code)
    try:
        wide = df.pivot_table(index="time", columns="code", values="close")
    except Exception as e:
        if not getattr(g, "diag_min_pivot", False):
            log.info("[DIAG] minute pivot FAIL: %r" % (e,))
            g.diag_min_pivot = True
        return arr, valid

    if len(wide) < 7:
        if not getattr(g, "diag_min_short", False):
            log.info("[DIAG] minute bars=%d (<7, 不足覆盖14:30~15:00) at %s" % (len(wide), prev_date))
            g.diag_min_short = True
        return arr, valid

    close_end = wide.iloc[-1]
    close_1430 = wide.iloc[-7]
    ret = close_end / close_1430 - 1.0   # Series: code -> tail30 return

    for i, s in enumerate(stocks):
        try:
            v = float(ret.get(s, np.nan))
        except Exception:
            v = np.nan
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and close_1430.get(s, 0) > 0:
            arr[i] = v
            valid[i] = True
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"tail30_momentum": factor_tail30_momentum}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] tail30_reversal: NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  tail30_reversal  Per-Factor IC/ICIR Report (POC)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 周度调仓 (lhb_freq_20d v2 修复版骨架) ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 0: 滞后 IC 计算 (上周因子 -> 本周收益)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]

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
            except Exception:
                pass

    # ═══════════════════════════════════════════
    # Step 1: 股票池过滤 (逐只 try, 修复版)
    # ═══════════════════════════════════════════
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
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:800]
    if len(universe) < g.stock_num:
        return

    arr, v = FACTOR_FUNC_MAP["tail30_momentum"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] tail30 valid=%d (<10) at %s — 不交易 (分钟数据源异常?)" % (int(np.sum(v)), prev_date))
            g.diag_low_valid = True
        return

    # 反转方向: 因子值 = -tail30_momentum (低尾盘动量 = 高分)
    rp = rank_pct(-arr, v)

    # Step 2: 保存快照供下周 IC 计算 (存组合实际使用的反转分数, IC报告直接反映反转方向)
    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # Step 3: 选股 & 调仓 (REVERSAL = 取最低 tail30, 即最高 rp)
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

    cd_order = get_current_data()

    # ── 平仓: 不在新持仓名单里的股票清仓 ──
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

    # ── 开仓: 等权买入 (铁律#17: 手数取整, 跳过金额不足100股) ──
    # 2026-08-23 修复: 显式整手下单, 数量恒为100整数倍, 杜绝 <100股委托失败。
    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break
            d = cd_order[s]
            if d.paused or d.is_st:
                continue
            if d.high_limit <= d.last_price:
                continue
            px = d.last_price if (d.last_price and d.last_price > 0) else d.day_open
            if px <= 0:
                continue
            lots = int(w / (px * 100))
            if lots < 1:
                continue
            order(s, lots * 100)
            ordered += 1
        except Exception:
            continue

    # Step 4: 日志 & 定期 IC 报告
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("tail30_reversal POC | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
