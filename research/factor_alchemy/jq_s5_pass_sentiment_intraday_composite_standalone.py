# -*- coding: utf-8 -*-
"""
sentiment_intraday_5f_composite 复合因子独立回测 (P0.5 复合探测产出, 周频候选)
================================================================================
来源: P0.5 复合实验 (p0_5_composite_probe.py, 2026-08-22) — 日频路线证伪后的意外产出
复合 = 5 因子等权 rank 和 (打散 v1 形态):
  overnight_raw = open/close.shift(1) - 1.0                     [隔夜动量, 2d峰值]
  overnight5    = (open/close.shift(1)-1.0).rolling(5,min3).sum [5日隔夜动量]
  intraday_rev  = -(close/open - 1.0)                           [日内反转]
  lhb_freq_20d  = -lhb_flag.rolling(20).sum()                   [龙虎榜事件反转]
  overreaction  = -close.pct_change(1).rolling(20).max()        [过度反应反转]
方向: 全部 HIGH=做多 (与本地 P0.5 公式方向一致)

本地证据 (270天全市场口径, 严格 open-to-open):
  IC(1d)=+0.040  IC(2d)=+0.051(t6.6)  IC(3d)=+0.048  IC(5d)=+0.042(t5.6)
  IC(10d)=+0.058  IC(20d)=+0.071  半段稳健 (+0.047/+0.055)
  日频被证伪的原因 = 换手 (top50 单边日换手 76% → 年成本66% >> 毛alpha 5.6%)
  但周频持有期 (~5d) 换手大幅下降, 成本可控 → 作为周频单因子候选提交 JQ 裁决

执行层 = lhb_freq_20d v2 修复版 + v3_overreaction 向量化 + 铁律17手数取整:
  ① 池过滤逐只 try (停牌/ST/涨停剔除)  ② 平仓 order_target(s,0) 清仓语义
  ③ 开仓 day_open 口径 + 手数取整 order_target(s, shares整百) + 金额不足100股跳过
  ④ 现金不足提前 break  ⑤ 向量化宽表因子 (两次 get_price 各1字段, ≤5.5万行红线)
  ⑥ 每因子滞后 Rank IC/ICIR 输出 (P-001 合规, 6条: 复合+5子因子)

JQ 性能: 池800 × 30d × 2字段 = 4.8万行/周期 (≤5.5万), get_price 2次/周期 (≤2),
  get_billboard_list 1次/周期 (lhb v2 实测可跑完全窗)
回测区间建议: 2021-01-01 ~ 2026-07-30 (与 v1 打散/King 对齐)
裁决: 周频单因子 Sharpe≥0.5 量级 → 走 trigger_d_plus 入库, 成为池内第一个"情绪×日内"族成员
"""
import numpy as np
import pandas as pd

TOP_N = 30
UNIVERSE_SIZE = 800
LOOKBACK = 30


def initialize(context):
    g.stock_num = TOP_N
    g.universe_size = UNIVERSE_SIZE
    g.lookback = LOOKBACK
    g.trade_days = 0
    g.ic_reported = False

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关), 6 条: 复合 + 5 子因子 ---
    g.factor_names = ["composite", "overnight_raw", "overnight5",
                      "intraday_rev", "lhb_freq_20d", "overreaction"]
    g.ic_records = {}
    g.last_stocks = {}
    g.last_values = {}
    for name in g.factor_names:
        g.ic_records[name] = []
        g.last_stocks[name] = []
        g.last_values[name] = None
    g.last_closes = {}

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
    log.info("sentiment_intraday_5f_composite (P0.5复合, 周频) 启动")


# ═══ 辅助 ═══
def _rankdata_impl(a):
    """rankdata 降级实现: scipy 不可用时自实现"""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        order = np.argsort(np.argsort(a))
        return order.astype(float) + 1.0


def rank_pct(arr, valid):
    """截面 rank pct: 有效处 (ranks-1)/(n-1), 无效处 NaN (供 nanmean 复合, 对齐 P0.5)"""
    n = len(arr)
    result = np.full(n, np.nan)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        return result
    ranks = _rankdata_impl(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result[valid] = pct
    return result


# ═══ 因子: lhb_freq_20d (数据源 get_billboard_list, 参考 lhb v2 已验证实现) ═══
def _norm_code(c):
    c = str(c)
    if "." in c:
        return c
    if c.startswith(("6", "9", "5")):
        return c + ".XSHG"
    return c + ".XSHE"


def factor_lhb_freq_20d(stocks, prev_date):
    """-lhb_flag.rolling(20).sum() 的 JQ 等价: 近20交易日龙虎榜上榜次数取负; HIGH=做多
    API 失败/为空时 valid 全 False → 该因子 NaN, 复合 nanmean 自动忽略 (P0.5 语义)"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if len(stocks) == 0:
        return arr, valid
    start = pd.Timestamp(prev_date) - pd.Timedelta(days=40)
    try:
        df = get_billboard_list(stock_list=None, start_date=start,
                                end_date=prev_date, count=None)
    except Exception as e:
        if not getattr(g, "diag_err", False):
            log.info("[DIAG] get_billboard_list FAIL: %r" % (e,))
            g.diag_err = True
        return arr, valid
    if df is None or len(df) == 0:
        if not getattr(g, "diag_empty", False):
            log.info("[DIAG] get_billboard_list EMPTY for %s ~ %s" % (start, prev_date))
            g.diag_empty = True
        return arr, valid
    try:
        d = df[df["direction"].astype(str).str.upper() == "ALL"]
        if len(d) == 0:
            d = df
    except Exception:
        d = df
    try:
        d = d.copy()
        d["_c"] = d["code"].map(_norm_code)
        c = d.drop_duplicates(["_c", "day"]).groupby("_c").size()
    except Exception:
        try:
            c = d.groupby("code").size()
        except Exception:
            return arr, valid
    for i, s in enumerate(stocks):
        arr[i] = -float(c.get(s, 0))
        valid[i] = True
    return arr, valid


# ═══ IC/ICIR 报告 (P-001 合规: 每因子单独输出) ═══
def _print_ic_report():
    log.info("=" * 60)
    log.info("  sentiment_intraday_5f_composite  Per-Factor IC/ICIR (P-001)")
    log.info("=" * 60)
    n_any = 0
    for name in g.factor_names:
        rec = g.ic_records.get(name, [])
        n_any = max(n_any, len(rec))
        if len(rec) == 0:
            log.info("  %-14s NO IC RECORDS" % name)
            continue
        ic_arr = np.array(rec)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos = float(np.mean(ic_arr > 0))
        log.info("  %-14s IC=%+.4f std=%.4f ICIR=%+.4f IC>0=%.1f%% N=%d" % (
            name, ic_mean, ic_std, icir, ic_pos * 100, len(ic_arr)))
    log.info("=" * 60)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 1: 股票池过滤 (lhb v2 修复版: 逐只 try, 单只异常剔除不污染池子)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:g.universe_size]

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
    universe = filtered[:g.universe_size]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 2: 两次 get_price (open/close 各1字段, 池800×30d×2=4.8万行 ≤5.5万红线)
    #   JQ 铁律: skip_paused=False, fq="pre", fields 字符串
    # ═══════════════════════════════════════════
    lb = g.lookback
    open_df = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                        fields="open", skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(open_df, _P):
        open_df = open_df["open"] if "open" in getattr(open_df, "items", []) else open_df.minor_xs("open")
    close_df = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                         fields="close", skip_paused=False, fq="pre")
    if _P is not None and isinstance(close_df, _P):
        close_df = close_df["close"] if "close" in getattr(close_df, "items", []) else close_df.minor_xs("close")

    if (open_df is None or close_df is None
            or open_df.shape[0] < 22 or close_df.shape[0] < 22
            or open_df.shape[1] == 0 or close_df.shape[1] == 0):
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s open=%s close=%s — 数据加载问题?" % (
                prev_date, getattr(open_df, "shape", None), getattr(close_df, "shape", None)))
            g.diag_no_data = True
        return
    open_df = open_df.reindex(columns=universe)
    close_df = close_df.reindex(columns=universe)

    # ═══════════════════════════════════════════
    # Step 3: 向量化宽表因子 (与 P0.5 本地公式 1:1, 铁律14)
    #   每因子取最后一行 = prev_date 当期因子值 → 截面 rank pct → nanmean 复合
    # ═══════════════════════════════════════════
    overnight_raw = open_df / close_df.shift(1) - 1.0
    overnight5 = overnight_raw.rolling(5, min_periods=3).sum()
    intraday_rev = -(close_df / open_df - 1.0)
    # ffill+pct_change(None) ≡ 旧版默认 pad 填充语义 (停牌日前收价填充, 与 JQ fill_paused=True 一致), 且跨 pandas 版本无弃用警告
    overreaction = -close_df.ffill().pct_change(fill_method=None).rolling(20).max()

    fw = {}
    fw["overnight_raw"] = overnight_raw.iloc[-1].reindex(universe).values.astype(float)
    fw["overnight5"] = overnight5.iloc[-1].reindex(universe).values.astype(float)
    fw["intraday_rev"] = intraday_rev.iloc[-1].reindex(universe).values.astype(float)
    fw["overreaction"] = overreaction.iloc[-1].reindex(universe).values.astype(float)
    lhb_arr, lhb_valid = factor_lhb_freq_20d(universe, prev_date)
    fw["lhb_freq_20d"] = lhb_arr

    rp = {}
    valid_cnt = {}
    for name in ["overnight_raw", "overnight5", "intraday_rev", "lhb_freq_20d", "overreaction"]:
        v = np.isfinite(fw[name])
        rp[name] = rank_pct(fw[name], v)
        valid_cnt[name] = int(np.sum(v))

    # 复合 = 等权 rank 和 (nanmean, 跳过无效因子 — P0.5 逐因子独立 rank 后 mean 的语义)
    stack = np.vstack([rp[n] for n in ["overnight_raw", "overnight5",
                                       "intraday_rev", "lhb_freq_20d", "overreaction"]])
    with np.errstate(invalid="ignore"):
        comp = np.nanmean(stack, axis=0)
    comp = np.where(np.isfinite(comp), comp, np.nan)
    valid_c = np.isfinite(comp)
    nv = int(np.sum(valid_c))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] composite valid=%d (<10) at %s 子因子valid=%s — 不交易" % (
                nv, prev_date, valid_cnt))
            g.diag_low_valid = True
        return

    # ═══════════════════════════════════════════
    # Step 4: 滞后 IC 计算 (上周因子 → 本周收益), 6 条各自独立
    #   快照复用 close_df 最后一行 (prev_date 收盘)
    # ═══════════════════════════════════════════
    snap_closes = {}
    for s in close_df.columns:
        last_v = close_df[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

    for name in g.factor_names:
        if len(g.last_stocks[name]) > 0 and g.last_values[name] is not None:
            fwd_rets = []
            factor_vals = []
            for i, s in enumerate(g.last_stocks[name]):
                prev_close = g.last_closes.get(s)
                curr_close = snap_closes.get(s)
                fv = g.last_values[name][i]
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
                        g.ic_records[name].append(float(ic))
                except Exception:
                    pass

    g.last_stocks["composite"] = list(universe)
    g.last_values["composite"] = comp.copy()
    for name in ["overnight_raw", "overnight5", "intraday_rev", "lhb_freq_20d", "overreaction"]:
        g.last_stocks[name] = list(universe)
        g.last_values[name] = rp[name].copy()

    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 5: 选股 & 调仓 (复合分 HIGH = 做多)
    #   铁律17: 等权金额下单手数取整 order_target(s, shares整百),
    #           金额不足100股的标的跳过 (order_target_value 高价股<100股委托失败)
    # ═══════════════════════════════════════════
    comp_order = np.where(valid_c, -comp, np.inf)
    topn = min(g.stock_num, nv)
    top_idx = np.argsort(comp_order)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp_order[i] < np.inf]

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

    # ── 开仓: 等权买入 (手数取整) ──
    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            if context.portfolio.available_cash < w * 0.5:
                break                           # 现金不足(前面平仓失败连锁), 停止开仓
            d = cd_order[s]
            if d.paused or d.is_st:             # 停牌/ST 不买
                continue
            if d.high_limit <= d.last_price:    # 已涨停买不进
                continue
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0:
                continue
            target_shares = int(w / ref_px) // 100 * 100   # 手数取整 (整百股)
            if target_shares < 100:             # 金额不足100股 → 跳过 (铁律17)
                continue
            order_target(s, target_shares)
            ordered += 1
        except Exception:
            continue

    # ═══════════════════════════════════════════
    # Step 6: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records["composite"][-1] if g.ic_records["composite"] else 0.0
    if g.trade_days <= 8:
        log.info("senti5f | week=%d | valid=%d | top=%d | ordered=%d/%d | 首只=%s(comp=%.3f) | IC=%.4f" % (
            g.trade_days, nv, len(top_stocks), ordered, len(top_stocks),
            top_stocks[0] if top_stocks else "-",
            float(comp[top_idx[0]]) if len(top_idx) > 0 else 0.0,
            latest_ic))
    else:
        log.info("senti5f | week=%d | ordered=%d/%d | valid=%d | IC=%.4f | total_ic=%d" % (
            g.trade_days, ordered, len(top_stocks), nv,
            latest_ic, len(g.ic_records["composite"])))

    if len(g.ic_records["composite"]) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        g.ic_reported = True

    gc.collect()
