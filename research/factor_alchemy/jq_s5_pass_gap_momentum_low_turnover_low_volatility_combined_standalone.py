# -*- coding: utf-8 -*-
"""
gap_momentum_low_turnover_low_volatility_combined 单因子独立回测
因子含义: 跳空动量 × 低换手 × 低波动 三重条件复合 (截面版, 手工实现)
  signal = (close/open - 1 截面rank) * (1 - (volume/volume.20d均值) 截面rank) * (1 - (20日收益std) 截面rank)
  三重条件: ①信息驱动跳空动量(缺口收益) ②低换手(过滤情绪日) ③低波动(过滤高波动) → 仅保留最干净的信息驱动跳空
Local (round-3 S5 PASS): ic=+0.0600 / icir=+0.3383 / +IC% 61.0% | direction=long (HIGH=做多)
生成时间: 2026-09-01 | 方向: HIGH=做多
P-001 合规: 输出 per-factor Rank IC/ICIR
"""

import numpy as np
import pandas as pd

FACTOR_NAME = "gap_momentum_low_turnover_low_volatility_combined"


def initialize(context):
    g.stock_num = 30
    g.lookback = 65
    g.trade_days = 0

    # --- IC 收集 (滞后法: 本周算上周因子→本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.ic_reported = False
    # --- 暴露监控 (P-20260814-004) ---
    g.exposure_records = []   # [(week, held_std, universe_std), ...]

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
    log.info("%s 单因子 (round-3 PASS, 截面, 滞后IC) 启动" % FACTOR_NAME)


# ═══ 辅助: 最终横截面 rank (选股用) ═══
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


# ═══ 因子实现 (截面版, 手工) ═══
def factor_gap_momentum_low_turnover_low_volatility_combined(stocks, close_df, extra_dfs=None):
    """跳空动量×低换手×低波动:
    signal = gap截面rank * (1 - turnover截面rank) * (1 - vol20截面rank)
    HIGH = 做多"""
    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    ex = extra_dfs or {}
    open_df = ex.get("open")
    volume = ex.get("volume")
    if open_df is None:
        open_df = nan_df
    if volume is None:
        volume = nan_df
    open_df = open_df.reindex(index=close_df.index, columns=close_df.columns)
    volume = volume.reindex(index=close_df.index, columns=close_df.columns)

    gap = close_df / open_df - 1.0
    turnover = volume / (volume.rolling(20).mean() + 1e-8)
    vol20 = close_df.pct_change().rolling(20).std()

    # 每日横截面 rank (axis=1 = 每天一截面)
    gap_rank = gap.rank(pct=True, axis=1)
    low_turn = 1.0 - turnover.rank(pct=True, axis=1)
    low_vol = 1.0 - vol20.rank(pct=True, axis=1)

    signal = gap_rank * low_turn * low_vol
    last = signal.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


FACTOR_FUNC_MAP = {FACTOR_NAME: factor_gap_momentum_low_turnover_low_volatility_combined}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] %s: NO IC RECORDS" % FACTOR_NAME)
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  %s  Per-Factor IC/ICIR Report (P-001)" % FACTOR_NAME)
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 暴露监控报告 (P-20260814-004) ═══
def _print_exposure_report():
    if len(g.exposure_records) < 10:
        log.info("[EXPOSURE] %s: 样本不足 (%d < 10), 跳过暴露报告" % (FACTOR_NAME, len(g.exposure_records)))
        return
    held = np.array([e[1] for e in g.exposure_records], dtype=float)
    univ = np.array([e[2] for e in g.exposure_records], dtype=float)
    held = held[~np.isnan(held)]; univ = univ[~np.isnan(univ)]
    if len(held) < 10:
        return
    base_n = min(20, max(4, len(held) // 2))
    base = held[:base_n]; recent = held[-4:]
    base_mean, recent_mean = float(np.mean(base)), float(np.mean(recent))
    ratio = recent_mean / base_mean if base_mean > 0 else float("nan")
    log.info("=" * 55)
    log.info("  %s  Factor Exposure Report (P-004)" % FACTOR_NAME)
    log.info("=" * 55)
    log.info("  held std baseline(前%d期): %.4f" % (base_n, base_mean))
    log.info("  held std recent (近4期):   %.4f" % recent_mean)
    log.info("  universe std mean:         %.4f" % float(np.mean(univ)))
    log.info("  exposure vol ratio:        %.2fx" % ratio)
    if ratio >= 2.0:
        log.info("  [WARN] 因子暴露波动率放大 %.1fx! 拥挤/踩踏风险, 建议降权或暂停" % ratio)
    elif ratio >= 1.5:
        log.info("  [WATCH] 因子暴露波动率上升 %.1fx, 进入观察名单" % ratio)
    else:
        log.info("  [OK] 暴露波动率稳定")
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # Step 0: 股票池过滤 (lhb v2 修复版: 逐只 try, 单只异常剔除不污染池子)
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
                continue          # 单只数据异常 → 剔除, 不污染池子
    except Exception:
        filtered = list(universe_all)   # 整体异常才回退 (保持可用性)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:500]
    if len(universe) < g.stock_num:
        return

    # Step 1: 单次 get_price(close) + 条件字段加载 (v3 性能骨架)
    lb = g.lookback
    px_c = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="close", skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c, _P):
        px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
    if px_c is None or px_c.shape[0] < 22 or px_c.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s — 数据加载问题?" % (
                prev_date, getattr(px_c, "shape", None)))
            g.diag_no_data = True
        return

    close_df = px_c.reindex(columns=universe)   # 缺失列 → NaN 列 → valid=False 自动排除
    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= 30]
    if len(valid_u) < g.stock_num:
        return

    extra_dfs = {}

    # 开盘价 (open)
    px_o = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="open", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_o, _P):
        px_o = px_o["open"] if "open" in getattr(px_o, "items", []) else px_o.minor_xs("open")
    if px_o is not None and px_o.shape[0] > 0 and px_o.shape[1] > 0:
        extra_dfs["open"] = px_o.reindex(columns=universe)

    # 成交量 (volume)
    px_v = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="volume", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_v, _P):
        px_v = px_v["volume"] if "volume" in getattr(px_v, "items", []) else px_v.minor_xs("volume")
    if px_v is not None and px_v.shape[0] > 0 and px_v.shape[1] > 0:
        extra_dfs["volume"] = px_v.reindex(columns=universe)

    # 2026-08-17: 因子计算异常必须打日志
    try:
        arr, v = FACTOR_FUNC_MAP[FACTOR_NAME](valid_u, close_df, extra_dfs)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] %s 因子计算异常: %r at %s (data shape=%s, univ=%d) — 公式与 JQ 环境不兼容?" % (
                FACTOR_NAME, _fe, prev_date, getattr(px_c, "shape", None), len(valid_u)))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] %s valid=%d (<10) at %s — 不交易 (因子计算失败?)" % (FACTOR_NAME, nv, prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    # Step 1.5: 快照 + 滞后 IC 计算 (上周因子 → 本周收益)
    snap_closes = {}
    for s in px_c.columns:
        last_v = px_c[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

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
                if ic == ic:               # 非 NaN
                    g.ic_records.append(float(ic))
            except Exception:
                pass

    # Step 2: 保存快照供下周 IC 计算
    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in close_df.columns and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # Step 3: 选股 & 调仓 (HIGH = 做多: 取最高 rp)
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    # P-20260814-004: 记录本期暴露
    try:
        held_vals = rp[top_idx][:len(top_stocks)]
        g.exposure_records.append((g.trade_days,
                                   float(np.std(held_vals)) if len(held_vals) > 1 else 0.0,
                                   float(np.std(rp[v])) if int(np.sum(v)) > 1 else 0.0))
        if len(g.exposure_records) > 500:
            g.exposure_records = g.exposure_records[-400:]
    except Exception:
        pass

    cd_order = get_current_data()

    # ── 平仓: 不在新持仓名单里的股票清仓 ──
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                if cd_order[s].paused:          # 停牌卖不出, 下周再试
                    continue
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target(s, 0)              # 清仓语义: 整单卖出含零股
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    # ── 开仓: 等权买入 ──
    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break                           # 现金不足, 停止开仓
            d = cd_order[s]
            if d.paused or d.is_st:             # 停牌/ST 不买
                continue
            if d.high_limit <= d.last_price:    # 已涨停买不进
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0 or w / ref_px < 100: # 开盘价口径判断能否买100股
                continue
            # P-20260901-008: 差额微调保护 — order_target_value 对已持仓股票按
            # |target_value - current_value| 差额下单, 差额<100股时 JQ 报
            # "开仓/平仓数量不能小于 100" (2026-09-01 9因子回测累计3.5万次失败)。
            # 微调差额不足一手时保持现状, 消除刷屏级委托失败。
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, w)
            ordered += 1
        except Exception:
            continue

    # Step 4: 日志 & 定期 IC 报告
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("%s | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        FACTOR_NAME, g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        _print_exposure_report()
        g.ic_reported = True

    gc.collect()
