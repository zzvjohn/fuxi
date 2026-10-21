# -*- coding: utf-8 -*-
"""
E 道幸存候选 #1 单因子独立回测 (GFlowNet 产出)
================================================================
候选表达式 : ts_zscore(abnormal_turnover)
可读含义   : 异常换手(自身20日换手 相对 行业均值 的负偏离) 的「逐股时序标准化」,
             做多相对行业换手更冷清(低换手)的股票

因子公式 (逐位对齐 factor_alchemy/factors/turnover.py + gflownet/grammar.py ts_zscore):
  avg_20d    = turnover.rolling(20, min_periods=10).mean()          # 自身20日均换手
  de_mean    = avg_20d - 行业均值(avg_20d)                          # 按 (date, sw_l1) 分组
  base       = cross_sectional_zscore( -de_mean )                   # = 行业均值 - 自身, 中位数标准化
  composite  = ts_zscore(base)                                     # 逐股时序 z-score
  signal     = cross_sectional_rank(composite)                     # 截面百分位, HIGH=做多

方向: HIGH = 做多 (E1 实测 signed ic_mean = +0.0817)
E 道表现: E1 ic_mean=0.0802 覆盖=89.5% | 收缩ICIR=0.474 | E1-E4 全过

P-001 合规: 回测结束输出单因子 Rank IC / ICIR (均值)

E5 预注册 / 保真性说明 (重要):
  ① E1-E4 的 ts_zscore 用【全样本逐股 z-score】(in-sample 含未来信息); 本回测改为
     【52 周滚动窗口, 仅用 prev_date 及以前数据】, 无 lookahead ⇒ JQ Rank IC 预期低于 E1, 属正常衰减。
  ② 原始 20 日均换手在 JQ 周频复刻中近似为【4 周滚动均值】(≈20 交易日); 行业分类取
     prev_date 时点(get_industry 一次), 忽略 1 年窗口内行业迁移。
  ③ 性能: 每期 ~54 次 get_fundamentals(周频换手) + 1 次 get_industry, 回测较慢属预期。
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
    log.info("E道候选#1 ts_zscore(abnormal_turnover) 单因子回测启动")


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


def _fetch_turnover_hist(universe, week_ends):
    """周频 turnover_ratio(%) → DataFrame (周 × 股)。valuation 为 JQ 引擎全局名。"""
    hist = pd.DataFrame(np.nan, index=week_ends, columns=universe)
    for d in week_ends:
        q = query(valuation.code, valuation.turnover_ratio).filter(valuation.code.in_(universe))
        df = get_fundamentals(q, date=d)
        if df is None or len(df) == 0:
            continue
        if "code" not in df.columns or "turnover_ratio" not in df.columns:
            continue
        for _, row in df.iterrows():
            c = row["code"]
            v = row["turnover_ratio"]
            if c in hist.columns and (v == v) and v >= 0:
                hist.at[d, c] = float(v)
    return hist


def _norm_industry(ind_map):
    """get_industry 返回 {code: {sw_l1:{industry_name,...},...}}，归一化为 {code: 行业名 str}。

    直接拿内层 dict 当 key 会报 ``unhashable type: 'dict'``；这里抽出 sw_l1 行业名。
    """
    out = {}
    for s, info in (ind_map or {}).items():
        if isinstance(info, dict):
            l1 = info.get('sw_l1')
            if isinstance(l1, dict):
                name = l1.get('industry_name')
                if name:
                    out[s] = name
                    continue
        elif isinstance(info, str) and info:
            out[s] = info
    return out


def _abnormal_turnover_base(hist_to, ind_map):
    """hist_to: 周×股 turnover_ratio。返回 base = zscore_cs(行业均值 - 自身4周均换手) (周×股)。"""
    own_avg = hist_to.rolling(4, min_periods=2, axis=0).mean()
    base = pd.DataFrame(np.nan, index=own_avg.index, columns=own_avg.columns)
    for w in range(own_avg.shape[0]):
        row = own_avg.iloc[w]
        ind_vals = {}
        for s in row.index:
            ind = ind_map.get(s)
            v = row[s]
            if ind is not None and np.isfinite(v):
                ind_vals.setdefault(ind, []).append(v)
        ind_mean = {k: float(np.mean(vs)) for k, vs in ind_vals.items()}
        # abnormal = 行业均值 - 自身
        abnormal = np.full(row.shape[0], np.nan)
        for j, s in enumerate(row.index):
            ind = ind_map.get(s)
            v = row[s]
            if ind is not None and ind in ind_mean and np.isfinite(v):
                abnormal[j] = ind_mean[ind] - v
        valid = np.isfinite(abnormal)
        if valid.sum() < 10:
            continue
        base.iloc[w] = cs_zscore_median(abnormal, valid)
    return base


# ═══ 因子: ts_zscore(abnormal_turnover) ═══
def factor_tszscore_abn_turnover(stocks, hist_to, ind_map):
    base = _abnormal_turnover_base(hist_to, ind_map)
    arr = ts_zscore_last(base)
    valid = np.isfinite(arr)
    return arr, valid


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] ts_zscore(abnormal_turnover): NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  ts_zscore(abnormal_turnover)  Per-Factor IC/ICIR Report (P-001)")
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

    days = _trade_days(prev_date, g.hist_days)
    week_ends = days[::5]
    if not week_ends or week_ends[-1] != prev_date:
        week_ends = list(week_ends) + [prev_date]

    try:
        ind_map = _norm_industry(get_industry(universe, date=prev_date) or {})
        hist_to = _fetch_turnover_hist(universe, week_ends)
        arr, v = factor_tszscore_abn_turnover(universe, hist_to, ind_map)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] 因子计算异常: %r at %s" % (_fe, prev_date))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            _cov = float(np.isfinite(hist_to.values).sum()) / max(1, hist_to.size)
            log.info("[DIAG] 有效信号不足 nv=%d at %s | 换手历史覆盖率=%.3f" % (nv, prev_date, _cov))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

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
    log.info("ts_zscore(abnormal_turnover) | week=%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, int(np.sum(v)), latest_ic, len(g.ic_records)))
    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
