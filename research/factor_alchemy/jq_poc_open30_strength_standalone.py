# -*- coding: utf-8 -*-
"""
开盘30分钟强度 单因子独立回测 — 分钟因子 JQ 方向探测通道 (P-20260823-004)
========================================================================
背景: 高频微观结构范式 0 覆盖; JQ 分钟通道已由 tail30 POC 验证可用
      (单次全池 800只×12根=9600行, 性能达标)。
      tail30_momentum 教训: A股尾盘30分钟动量是稳定反向指标 (做多 -59.37% 证伪)。
      因此本文件不预设方向: 组合按【原始因子做多】运行, 用 IC 符号判定方向。

因子定义:
  open30_ret = close(10:00) / close(9:35) - 1
  (开盘 9:35→10:00 的 25 分钟累计收益; 9:30 集合竞价段不含, 与 tail30
   close(15:00)/close(14:30) 口径对称)

判读规则 (回测后按此定方向):
  - IC mean > 0 且 IC>0 率 ≥ 55%  → 开盘强度 = 动量 (做多高开强度股)
  - IC mean < 0 且 IC<0 率 ≥ 55%  → 开盘强度 = 反转 (做多低开强度股, 同 overnight_reversal 族)
  - 两者皆不满足 → 无信号, reserve
  - 半年 ICIR (最近26周) 是主要判据, 全窗 ICIR 为参考

数据源: get_price(unit='5m', count=6, end_date=prev_date 10:00) — 单次调用全池向量化
        池 800 只 × 6 根 = 4800 行, 远低于 5.5万行红线 (JQ 铁律)
防未来: 用 context.previous_date 的分钟数据, 下周开盘调仓, 无前视
骨架:   lhb_freq_20d v2 修复版执行层 (逐只 try / order_target 清仓 /
        day_open 口径 / 现金 break / DIAG 日志 / 手数取整)
        —— 禁止使用旧版 chunk 骨架 (曾导致回测零组合收益)

JQ 平台设置: 回测区间 2021-01-01 ~ 2026-08-13, 周频, 沪深300基准
"""
import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 60
    g.trade_days = 0

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关) ---
    g.ic_records = []
    g.ic_dates = []
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
    log.info("open30_strength POC (开盘30分钟强度 方向探测) 启动")


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


# ═══ 因子: open30_ret (开盘30分钟强度, 分钟宽表向量化) ═══
def factor_open30_ret(stocks, prev_date):
    """prev_date 开盘 9:35→10:00 涨跌幅 = close(10:00)/close(9:35)-1。

    实现: 单次 get_price(unit='5m', count=6) 拉全池上一交易日最早 6 根 5m bar
          (9:35~10:00), 宽表 pivot 后向量化: bar[-1]/bar[0]-1。
    铁律 #14: 禁止逐只 eval, 必须宽表向量运算。
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid

    end_dt = pd.Timestamp(prev_date) + pd.Timedelta(hours=10)

    def _pull(fq):
        kw = dict(security=stocks, count=6, end_date=end_dt,
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

    if len(wide) < 6:
        if not getattr(g, "diag_min_short", False):
            log.info("[DIAG] minute bars=%d (<6, 不足覆盖9:35~10:00) at %s" % (len(wide), prev_date))
            g.diag_min_short = True
        return arr, valid

    close_1000 = wide.iloc[-1]
    close_0935 = wide.iloc[0]
    ret = close_1000 / close_0935 - 1.0   # Series: code -> open30 return

    for i, s in enumerate(stocks):
        try:
            v = float(ret.get(s, np.nan))
        except Exception:
            v = np.nan
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and close_0935.get(s, 0) > 0:
            arr[i] = v
            valid[i] = True
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"open30_strength": factor_open30_ret}


# ═══ IC/ICIR 报告 (含半年窗口方向探测) ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] open30_strength: NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))

    # 半年窗口 (最近 26 周)
    h1 = ic_arr[-26:]
    h1_mean = float(np.mean(h1))
    h1_std = float(np.std(h1))
    h1_icir = h1_mean / h1_std if h1_std > 0 else 0.0
    h1_pos = float(np.mean(h1 > 0))

    # 方向判定
    if h1_mean > 0 and h1_pos >= 0.55:
        verdict = "MOMENTUM (开盘强度=动量, 做多高开强度)"
    elif h1_mean < 0 and (1 - h1_pos) >= 0.55:
        verdict = "REVERSAL (开盘强度=反转, 做多低开强度)"
    else:
        verdict = "NO_SIGNAL (方向不显著, reserve)"

    log.info("=" * 60)
    log.info("  open30_strength  Per-Factor IC/ICIR Report (POC 方向探测)")
    log.info("=" * 60)
    log.info("  [全窗]  Rank IC mean:  %+.4f" % ic_mean)
    log.info("  [全窗]  ICIR:          %+.4f" % icir)
    log.info("  [全窗]  IC > 0 ratio:  %.1f%%" % (ic_pos * 100))
    log.info("  [半年]  Rank IC mean:  %+.4f" % h1_mean)
    log.info("  [半年]  ICIR:          %+.4f" % h1_icir)
    log.info("  [半年]  IC > 0 ratio:  %.1f%%" % (h1_pos * 100))
    log.info("  Total IC samples:      %d" % len(ic_arr))
    log.info("  >>> 方向判定: %s" % verdict)
    log.info("=" * 60)


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
                g.ic_dates.append(str(prev_date))
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

    arr, v = FACTOR_FUNC_MAP["open30_strength"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] open30 valid=%d (<10) at %s — 不交易 (分钟数据源异常?)" % (int(np.sum(v)), prev_date))
            g.diag_low_valid = True
        return

    # 方向探测: 组合按原始因子做多 (高开强度=高分), IC 符号另行判定方向
    rp = rank_pct(arr, v)

    # Step 2: 保存快照供下周 IC 计算
    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # Step 3: 选股 & 调仓 (做多 = 开盘强度最高)
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
    log.info("open30_strength POC | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 26 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
