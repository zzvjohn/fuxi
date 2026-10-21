# -*- coding: utf-8 -*-
"""
E 道幸存候选 #2 单因子独立回测 (GFlowNet 产出)
================================================================
候选表达式 : ts_zscore(sp)
可读含义   : S/P (=1/市销率TTM, 价值因子) 的「逐股时序标准化」, 做多相对自身历史更便宜的股票

因子公式 (逐位对齐 factor_alchemy/factors/value.py + gflownet/grammar.py ts_zscore):
  base       = cross_sectional_zscore( 1 / ps_ttm )        # ps_ttm>0; 中位数去中心, std 标准化
  composite  = ts_zscore(base)                              # 逐股(逐列)时序 z-score
  signal     = cross_sectional_rank(composite)              # 截面百分位, HIGH=做多

方向: HIGH = 做多 (E1 实测 signed ic_mean = +0.0907)
E 道表现: E1 ic_mean=0.0913 覆盖=88.5% | 收缩ICIR=0.639 | E1-E4 全过

P-001 合规: 回测结束输出单因子 Rank IC / ICIR (均值)

E5 预注册 / 保真性说明 (重要):
  E1-E4 的 ts_zscore 用的是【全样本 2021-2026 逐股 z-score】, 含未来信息 (in-sample)。
  本 JQ 回测为 E5 前向复验, 时序标准化改为【52 周滚动窗口, 仅用 prev_date 及以前数据】,
  无 lookahead。故 JQ 的 Rank IC 预期会低于 E1 的 0.0907 —— 这是 in-sample→OOS 的正常衰减,
  不是口径错误。若需严格逐位对齐 in-sample, 请改用全样本 z-score (但会引入前视偏差)。
  性能: 每期 ~54 次 get_fundamentals(周频估值) 查询, 回测较慢属预期。
Local: 手写 (2026-10-21, E 道移植 Path B)
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 50
    g.hist_days = 270        # ~54 个周频点(1年) 用于逐股时序 z-score
    g.trade_days = 0

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
    log.info("E道候选#2 ts_zscore(sp) 单因子回测启动")


# ═══ 辅助 ═══
def rank_pct(arr, valid):
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


def ts_zscore_last(hist):
    """hist: DataFrame (周 × 股, 最旧→最新)。返回最后一行的逐股时序 z-score (1D)。"""
    out = np.full(hist.shape[1], np.nan)
    for j in range(hist.shape[1]):
        col = hist.iloc[:, j].astype(float).values
        valid = np.isfinite(col)
        if valid.sum() < 20:
            continue
        mu = col[valid].mean()
        sd = col[valid].std()
        if sd < 1e-12:
            continue
        last = col[-1]
        if np.isfinite(last):
            out[j] = (last - mu) / sd
    return out


def _fetch_ps_hist(universe, week_ends):
    """周频 ps_ratio → DataFrame (周 × 股) 的 1/ps。valuation 为 JQ 引擎全局名。"""
    hist = pd.DataFrame(np.nan, index=week_ends, columns=universe)
    for d in week_ends:
        q = query(valuation.code, valuation.ps_ratio).filter(valuation.code.in_(universe))
        df = get_fundamentals(q, date=d)
        if df is None or len(df) == 0:
            continue
        if "code" not in df.columns or "ps_ratio" not in df.columns:
            continue
        for _, row in df.iterrows():
            c = row["code"]
            v = row["ps_ratio"]
            if c in hist.columns and (v == v) and v > 0:
                hist.at[d, c] = 1.0 / float(v)
    return hist


# ═══ 因子: ts_zscore(sp) ═══
def factor_tszscore_sp(stocks, hist_sp):
    """hist_sp: DataFrame (周 × 股), 值 = 1/ps_ratio。返回 (arr, valid)。"""
    # base = 逐周截面 z-score
    base = pd.DataFrame(np.nan, index=hist_sp.index, columns=hist_sp.columns)
    for w in range(hist_sp.shape[0]):
        row = hist_sp.iloc[w].values
        valid = np.isfinite(row)
        if valid.sum() < 10:
            continue
        base.iloc[w] = cs_zscore_median(row, valid)
    # ts_zscore 最后一行
    arr = ts_zscore_last(base)
    valid = np.isfinite(arr)
    return arr, valid


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] ts_zscore(sp): NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  ts_zscore(sp)  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


def _trade_days(end_date, count):
    """返回截至 end_date 的近 count 个交易日列表（date 对象）。

    部分 JQ 引擎（jqboson）全局命名空间不含 get_trade_days（报 NameError），
    这里改用指数价格历史推导交易日，等价且稳妥。
    """
    px = get_price("000300.XSHG", count=count, end_date=end_date,
                   frequency="daily", fields="close", fq="pre")
    if px is None or len(px) == 0:
        return []
    out = []
    for t in px.index:
        try:
            out.append(t.date())
        except Exception:
            out.append(t)
    return out


# ═══ 周度调仓 ═══
def rebalance(context):
    g.trade_days += 1
    if g.trade_days < 5:
        return
    prev_date = context.previous_date

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

    # 周频历史窗口
    days = _trade_days(prev_date, g.hist_days)
    week_ends = days[::5]
    if not week_ends or week_ends[-1] != prev_date:
        week_ends = list(week_ends) + [prev_date]

    try:
        hist_sp = _fetch_ps_hist(universe, week_ends)
        arr, v = factor_tszscore_sp(universe, hist_sp)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] 因子计算异常: %r at %s" % (_fe, prev_date))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        return

    rp = rank_pct(arr, v)

    # 滞后 IC (用 close 最后一行)
    px_c = get_price(universe, count=2, end_date=prev_date, frequency="daily",
                     fields="close", skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c, _P):
        px_c = px_c["close"] if "close" in getattr(px_c, "items", []) else px_c.minor_xs("close")
    snap_closes = {}
    if px_c is not None and px_c.shape[0] > 0:
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

    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # 选股 (HIGH=做多)
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

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
    log.info("ts_zscore(sp) | week=%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, int(np.sum(v)), latest_ic, len(g.ic_records)))
    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
