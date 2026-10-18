# -*- coding: utf-8 -*-
"""
变体A 6因子复合回测 (MMR 全池搜索冠军, 2026-08-21, v2)
  组成: CAND-gpevo006 + KING-overnight5 + KING-tvma20 + KING-dollar20 + KING-tstd_cv + KING-nearhigh
  合成: 各因子横截面 rank(pct) 等权平均 → top60 周频等权调仓
  本地模拟: +737%/Sharpe 1.48/MaxDD -27.1% (周频, 2021-2026, 仅方向参考)
  铁律: ① P-001 每因子输出单独 Rank IC/ICIR ② 执行层骨架 = lhb v2 修复版
       ③ get_price ≤2 次调用 ④ 停牌/涨停过滤 ⑤ 等权下单手数取整+现金 break
  v2 修复 (2026-08-21 首轮回测日志审计):
  - 执行层: order_target_value → order_target(手数取整); 首轮 3478 笔 "数量不能小于100"
    失败 (差额<100股调整被 JQ 拒单, ~20% 失败率); v2 差额<100 跳过、零股先清零头
  - IC 报告: 去掉 ic_reported 一次性标志, 每 26 周输出滚动全周期 per-factor IC/ICIR
  v1 实测 (2021-01-01~2026-07-30): +173.32% / Sharpe 0.685 / MaxDD 32.91%
因子定义 (与 _combo_search.py POOL 一致):
  gpevo006   = -open                                (低开盘价)
  overnight5 = (open/close.shift(1)-1).rolling(5).sum()  (隔夜跳空5日累计)
  tvma20     = -(amt/amt.rolling(20).mean()).clip(0,10)  (缩量, amt=close*volume)
  dollar20   = -log(amt.rolling(20).mean())              (低成交额)
  tstd_cv    = -(vol.rolling(20).std()/vol.rolling(20).mean())  (低波动)
  nearhigh   = -(close.rolling(10).mean()/close.rolling(60).max())  (远离60日高点)
方向: 全部 HIGH=做多 (公式自带负号, rank 大者买入)
"""
import numpy as np
import pandas as pd

FACTOR_NAMES = ["gpevo006", "overnight5", "tvma20", "dollar20", "tstd_cv", "nearhigh"]


def initialize(context):
    g.stock_num = 60
    g.lookback = 120          # nearhigh 需 60 日, 留余量
    g.trade_days = 0

    # --- IC 收集 (P-001: 每因子滞后法, 上周因子→本周收益秩相关) ---
    g.ic_records = {n: [] for n in FACTOR_NAMES}
    g.last_snap = {n: None for n in FACTOR_NAMES}   # (stocks, rank_values, closes)
    g.diag_no_data = False
    g.diag_factor_err = False

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
    log.info("变体A 6因子复合 (MMR冠军, 滞后IC) 启动")


# ═══ 辅助 ═══
def rank_pct(arr, valid):
    """scipy rankdata 口径: pct = (rank-1)/(n_valid-1); invalid -> 0.0 (组合最低分)"""
    n = len(arr)
    result = np.zeros(n)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result[valid] = pct
    return result


# ═══ 因子计算 (宽表向量化, 与 S5 引擎语义一致) ═══
def compute_factors(close_df, open_df, volume_df):
    """返回 {name: (arr, valid_mask)} 全池最后一行横截面"""
    stocks = list(close_df.columns)
    amt = close_df * volume_df
    out = {}

    # gpevo006: -open
    arr = (-open_df.iloc[-1]).reindex(stocks).values.astype(float)
    out["gpevo006"] = (arr, np.isfinite(arr))

    # overnight5: (open/close.shift(1)-1).rolling(5).sum()
    arr = ((open_df / close_df.shift(1) - 1.0).rolling(5, min_periods=3).sum().iloc[-1]
           ).reindex(stocks).values.astype(float)
    out["overnight5"] = (arr, np.isfinite(arr))

    # tvma20: -(amt/amt.rolling(20).mean()).clip(0,10)
    arr = (-(amt / amt.rolling(20, min_periods=10).mean()).clip(0, 10).iloc[-1]
           ).reindex(stocks).values.astype(float)
    out["tvma20"] = (arr, np.isfinite(arr))

    # dollar20: -log(amt.rolling(20).mean())
    arr = (-np.log(amt.rolling(20, min_periods=10).mean()).iloc[-1]
           ).reindex(stocks).values.astype(float)
    out["dollar20"] = (arr, np.isfinite(arr))

    # tstd_cv: -(vol.std20/vol.mean20)
    arr = (-(volume_df.rolling(20, min_periods=10).std()
             / volume_df.rolling(20, min_periods=10).mean()).iloc[-1]
           ).reindex(stocks).values.astype(float)
    out["tstd_cv"] = (arr, np.isfinite(arr))

    # nearhigh: -(close.ma10/close.max60)
    arr = (-(close_df.rolling(10, min_periods=5).mean()
             / close_df.rolling(60, min_periods=30).max()).iloc[-1]
           ).reindex(stocks).values.astype(float)
    out["nearhigh"] = (arr, np.isfinite(arr))

    return out


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    log.info("=" * 60)
    log.info("  变体A 6因子复合 Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 60)
    any_rec = False
    for n in FACTOR_NAMES:
        recs = g.ic_records[n]
        if len(recs) == 0:
            log.info("  %-12s NO IC RECORDS" % n)
            continue
        any_rec = True
        ic_arr = np.array(recs)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos = float(np.mean(ic_arr > 0))
        log.info("  %-12s RankIC=%+.4f  ICIR=%+.4f  +IC%%=%.1f%%  n=%d" % (
            n, ic_mean, icir, ic_pos * 100, len(ic_arr)))
    if any_rec:
        # 组合信号 IC (每周组合分 vs 下周收益)
        comb = g.ic_records.get("_combo", [])
        if len(comb) > 0:
            c_arr = np.array(comb)
            cm, cs = float(np.mean(c_arr)), float(np.std(c_arr))
            log.info("  %-12s RankIC=%+.4f  ICIR=%+.4f  n=%d  (组合分)" % (
                "COMBO", cm, cm / cs if cs > 0 else 0.0, len(c_arr)))
    log.info("=" * 60)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 0: 股票池过滤 (lhb v2 修复版: 逐只 try)
    # ═══════════════════════════════════════════
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
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:400]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 1: 单次 get_price 拉 close/open/volume (≤5.5万行, 1 次调用)
    # ═══════════════════════════════════════════
    lb = g.lookback
    px = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                   fields=["close", "open", "volume"], skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        close_df = px["close"] if "close" in getattr(px, "items", []) else px.minor_xs("close")
        open_df = px["open"] if "open" in getattr(px, "items", []) else px.minor_xs("open")
        volume_df = px["volume"] if "volume" in getattr(px, "items", []) else px.minor_xs("volume")
    elif isinstance(px.columns, pd.MultiIndex):
        close_df = px["close"]
        open_df = px["open"]
        volume_df = px["volume"]
    else:
        if not g.diag_no_data:
            log.info("[DIAG] get_price 返回结构异常 type=%s at %s" % (type(px).__name__, prev_date))
            g.diag_no_data = True
        return

    if close_df is None or close_df.shape[0] < 22 or close_df.shape[1] == 0:
        if not g.diag_no_data:
            log.info("[DIAG] get_price 空/不足 at %s shape=%s" % (
                prev_date, getattr(close_df, "shape", None)))
            g.diag_no_data = True
        return

    close_df = close_df.reindex(columns=universe)
    open_df = open_df.reindex(columns=universe) if open_df is not None else pd.DataFrame(
        np.nan, index=close_df.index, columns=universe)
    volume_df = volume_df.reindex(columns=universe) if volume_df is not None else pd.DataFrame(
        np.nan, index=close_df.index, columns=universe)

    valid_u = [s for s in universe
               if s in close_df.columns and int((~close_df[s].isna()).sum()) >= 30]
    if len(valid_u) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 2: 计算 6 因子 → 各自 rank → 等权组合
    # ═══════════════════════════════════════════
    try:
        factors = compute_factors(close_df, open_df, volume_df)
    except Exception as _fe:
        if not g.diag_factor_err:
            log.info("[DIAG] 因子计算异常: %r at %s (shape=%s) — 公式与 JQ 环境不兼容?" % (
                _fe, prev_date, getattr(close_df, "shape", None)))
            g.diag_factor_err = True
        return

    ranks = {}
    n_valid = np.zeros(len(valid_u), dtype=int)
    for n in FACTOR_NAMES:
        arr, v = factors[n]
        ranks[n] = rank_pct(arr, v)
        n_valid += v.astype(int)

    # 组合分 = 6 因子 rank 等权平均; 至少 3 个因子 valid 才可入选
    combo = np.zeros(len(valid_u))
    for n in FACTOR_NAMES:
        combo += ranks[n]
    combo /= len(FACTOR_NAMES)
    eligible = n_valid >= 3
    if int(eligible.sum()) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 2.5: 滞后 IC 快照 (上周因子/组合 → 本周收益)
    # ═══════════════════════════════════════════
    snap_closes = {}
    for s in close_df.columns:
        last_v = close_df[s].iloc[-1]
        if np.isfinite(last_v):
            snap_closes[s] = float(last_v)

    for n in FACTOR_NAMES + ["_combo"]:
        snap = g.last_snap.get(n)
        if snap is None:
            continue
        stks, fvals, prev_closes = snap
        fwd_rets, f_vals = [], []
        for s, fv in zip(stks, fvals):
            pc, cc = prev_closes.get(s), snap_closes.get(s)
            if (pc and pc > 0 and cc and cc > 0 and not np.isnan(fv)):
                fwd_rets.append((cc / pc) - 1.0)
                f_vals.append(fv)
        if len(fwd_rets) >= 30:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(f_vals, fwd_rets)
                if ic == ic:   # 非 NaN
                    g.ic_records[n].append(float(ic))
            except Exception:
                pass

    # 更新快照
    for n in FACTOR_NAMES:
        g.last_snap[n] = (valid_u, list(ranks[n]), dict(snap_closes))
    g.last_snap["_combo"] = (valid_u, list(combo), dict(snap_closes))

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓 (lhb v2 修复版执行层)
    # ═══════════════════════════════════════════
    comp = np.where(eligible, -combo, np.inf)
    topn = min(g.stock_num, int(eligible.sum()))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    cd_order = get_current_data()

    # ── 平仓: 不在新名单 → order_target(s,0) 清仓 ──
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

    # ── 开仓/调整: 手数取整 order_target (铁律17完整版 v2, 2026-08-21) ──
    # 根因修复: order_target_value 调整式下单, 目标与持仓差额 <100 股时 JQ 拒单
    # (实测 3478 笔 "开/平仓数量不能小于 100" 失败, ~20% 下单失败率)
    # v2: 全部按股数 order_target, 100 股整数倍; 差额<100 不调整; 零股持仓先清零头
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
            ref_px = d.day_open if (d.day_open and d.day_open > 0) else d.last_price
            if ref_px <= 0:
                continue
            target_amount = int(w / ref_px / 100.0) * 100
            if target_amount < 100:
                continue                      # 金额不足 100 股: 跳过 (铁律17)
            pos = context.portfolio.positions.get(s)
            cur_amount = int(pos.total_amount) if pos is not None else 0
            if cur_amount % 100 != 0:
                # 送股/分红产生零股: 一次性卖出零头 (卖出零股合法), 本周目标让位
                order_target(s, cur_amount - cur_amount % 100)
                ordered += 1
                continue
            if abs(target_amount - cur_amount) < 100:
                continue                      # 差额 <100 股: 不调整, 避免 JQ 拒单
            order_target(s, target_amount)
            ordered += 1
        except Exception:
            continue

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    if g.trade_days % 13 == 0:
        ics = {n: (g.ic_records[n][-1] if g.ic_records[n] else 0.0) for n in FACTOR_NAMES}
        log.info("variantA | week=%d | ordered=%d/%d | pool=%d | IC=%s" % (
            g.trade_days, ordered, len(top_stocks), int(eligible.sum()),
            {k: round(v, 4) for k, v in ics.items()}))

    # P-001 v2: 每 26 周(约半年) 输出滚动全周期 per-factor IC/ICIR
    # (原 ic_reported 一次性标志导致只有 2021-12 n=47 一次报告, 2026-08-21 修复)
    if g.trade_days >= 30 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
