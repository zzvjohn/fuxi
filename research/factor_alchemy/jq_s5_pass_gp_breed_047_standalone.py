# -*- coding: utf-8 -*-
"""
gp_breed_047 单因子独立回测 (S5 通过因子)
因子含义: GP mutate from template pool
Local: 泄漏恢复 2026-09-05: 原 未知标识符: amplitude, turnover (派生字段已按本地口径展开)
生成时间: 2026-09-05 00:12 | 方向: HIGH=做多 (与 S5JointFilter.backtest_factor ascending=False 一致)
P-001 合规: 输出 per-factor Rank IC/ICIR
"""

import numpy as np
import pandas as pd

def initialize(context):
    g.stock_num = 80
    g.lookback = 75
    g.trade_days = 0

    # --- IC 收集 (滞后法: 本周算上周因子→本周收益的秩相关) ---
    g.ic_records = []
    g.last_factor_stocks = []
    g.last_factor_values = None
    g.last_closes = {}
    g.ic_reported = False
    # --- 暴露监控 (P-20260814-004): 每期因子值横截面 std 时序 ---
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
    log.info("gp_breed_047 单因子 (S5通过, 滞后IC) 启动")


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


# ═══ 因子: gp_breed_047 ═══
def _rank_pct(obj):
    """pandas .rank(pct=True, na_option='keep') 的 JQ 兼容实现 (axis=0 逐列时序排名)。

    语义与本地 S5 引擎 (pandas 3.x): pct = rank / n_non_na, 单值列 -> 1.0, NaN 保持 NaN。
    """
    if isinstance(obj, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=obj.index, columns=obj.columns)
        for c in obj.columns:
            out[c] = _rank_pct(obj[c])
        return out
    v = obj.values.astype(float)
    mask = np.isfinite(v)
    out = np.full(len(v), np.nan)
    k = int(mask.sum())
    if k >= 1:
        from scipy.stats import rankdata
        r = rankdata(v[mask])
        out[mask] = r / k
    return pd.Series(out, index=obj.index)
_EXPR = "(-np.maximum(np.minimum(((high_p - low_p) / close_p.shift(1)), (volume_p / close_p.clip(lower=0.01))), close_p.ewm(span=10).mean()).pipe(_rank_pct))"
_GLOBALS = {
    "np": np, "pd": pd,
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,
    "exp": np.exp, "sign": np.sign,
    "maximum": np.maximum, "minimum": np.minimum,
    "where": np.where, "clip": np.clip,
    "range": range, "len": len, "int": int, "float": float,
    "list": list, "dict": dict, "tuple": tuple,
    "_rank_pct": _rank_pct,
}
def factor_gp_breed_047(stocks, close_df, extra_dfs=None):
    """GP mutate from template pool
    公式 (S5 通过口径): (-np.maximum(np.minimum(((high_p - low_p) / close_p.shift(1)), (volume_p / close_p.clip(lower=0.01))), close_p.ewm(span=10).mean()).pipe(_rank_pct))
    向量化实现 (与 S5 引擎宽表语义一致): 全池一次 DataFrame 运算, 取最后一行横截面
    HIGH = 做多 (与 S5JointFilter.backtest_factor ascending=False 一致)"""
    nan_df = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    ex = extra_dfs or {}
    def _pick(key):
        df = ex.get(key)
        if df is None:
            return nan_df
        return df.reindex(index=close_df.index, columns=close_df.columns)
    _LOCALS = {
        "close": close_df, "open": _pick("open"), "high": _pick("high"),
        "low": _pick("low"), "volume": _pick("volume"), "amount": _pick("money"),
        "close_p": close_df, "open_p": _pick("open"), "high_p": _pick("high"),
        "low_p": _pick("low"), "volume_p": _pick("volume"), "amount_p": _pick("money"),
        "returns": close_df.pct_change(),
    }
    val = eval(_EXPR, _GLOBALS, _LOCALS)
    if isinstance(val, pd.DataFrame):
        last = val.iloc[-1]
    elif isinstance(val, pd.Series):
        last = val
    else:
        last = pd.Series(np.full(len(stocks), float(val)), index=stocks)
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid
# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {"gp_breed_047": factor_gp_breed_047}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    """P-001 合规: 输出单因子 Rank IC 和 ICIR"""
    if len(g.ic_records) == 0:
        log.info("[IC] gp_breed_047: NO IC RECORDS")
        return

    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    icir    = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos  = float(np.mean(ic_arr > 0))

    log.info("=" * 55)
    log.info("  gp_breed_047  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 暴露监控报告 (P-20260814-004) ═══
def _print_exposure_report():
    """因子暴露波动率监控: 持仓股因子值横截面 std 的时序统计。

    背景: 2026-07 量化集体回撤 (幻方单月-22.15%), 先行指标 = 动量因子波动率翻数倍。
    用途: (1) 提前预警因子波动放大; (2) 供 D+ 蒸馏拥挤维度的归因输入。
    """
    if len(g.exposure_records) < 10:
        log.info("[EXPOSURE] gp_breed_047: 样本不足 (%d < 10), 跳过暴露报告" % len(g.exposure_records))
        return
    held = np.array([e[1] for e in g.exposure_records], dtype=float)
    univ = np.array([e[2] for e in g.exposure_records], dtype=float)
    held = held[~np.isnan(held)]; univ = univ[~np.isnan(univ)]
    if len(held) < 10:
        return
    # 基线 = 前 20 期 (或前一半); 最新 = 最近 4 期均值
    base_n = min(20, max(4, len(held) // 2))
    base = held[:base_n]; recent = held[-4:]
    base_mean, recent_mean = float(np.mean(base)), float(np.mean(recent))
    ratio = recent_mean / base_mean if base_mean > 0 else float("nan")
    log.info("=" * 55)
    log.info("  gp_breed_047  Factor Exposure Report (P-004)")
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

    # ═══════════════════════════════════════════
    # Step 0: 股票池过滤 (lhb v2 修复版: 逐只 try, 单只异常剔除不污染池子)
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0","3","6"))][:2000]

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
    universe = [s for s in filtered if s.startswith(("0","3","6"))][:400]
    if len(universe) < g.stock_num:
        return

    # ═══════════════════════════════════════════
    # Step 1: 单次 get_price(close) + 条件字段加载 (v3 性能骨架)
    #   原版 3 次全市场查询 (snap count=2 / close count=lb / volume count=lb)
    #   → 1 次 close + 按需 extra_fields; IC 快照复用本数据最后一行
    # ═══════════════════════════════════════════
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

    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="high", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["high"] if "high" in getattr(px_x, "items", []) else px_x.minor_xs("high")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["high"] = px_x.reindex(columns=universe)
    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="low", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["low"] if "low" in getattr(px_x, "items", []) else px_x.minor_xs("low")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["low"] = px_x.reindex(columns=universe)
    px_x = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                     fields="volume", skip_paused=False, fq="pre")
    if _P is not None and isinstance(px_x, _P):
        px_x = px_x["volume"] if "volume" in getattr(px_x, "items", []) else px_x.minor_xs("volume")
    if px_x is not None and px_x.shape[0] > 0 and px_x.shape[1] > 0:
        extra_dfs["volume"] = px_x.reindex(columns=universe)
    # 2026-08-17: 因子计算异常必须打日志 (旧版 silent valid=0 掩盖了 per-stock eval 事故)
    try:
        arr, v = FACTOR_FUNC_MAP["gp_breed_047"](valid_u, close_df, extra_dfs)
    except Exception as _fe:
        if not getattr(g, "diag_factor_err", False):
            log.info("[DIAG] gp_breed_047 因子计算异常: %r at %s (data shape=%s, univ=%d) — 公式与 JQ 环境不兼容?" % (
                _fe, prev_date, getattr(px_c, "shape", None), len(valid_u)))
            g.diag_factor_err = True
        return
    nv = int(np.sum(v))
    if nv < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] gp_breed_047 valid=%d (<10) at %s — 不交易 (因子计算失败?)" % (nv, prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    # ═══════════════════════════════════════════
    # Step 1.5: 快照 + 滞后 IC 计算 (上周因子 → 本周收益)
    #   v3: 快照复用 px_c 最后一行 (prev_date 收盘), 不再单独全市场查询
    # ═══════════════════════════════════════════
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

    # ═══════════════════════════════════════════
    # Step 2: 保存快照供下周 IC 计算
    # ═══════════════════════════════════════════
    g.last_factor_stocks = valid_u
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in valid_u:
        if s in close_df.columns and s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 3: 选股 & 调仓 (HIGH = 做多: 取最高 rp)
    #   lhb v2 修复版: ①停牌股下单前再拦一道 ②平仓 order_target(s,0) 清仓语义
    #   ③开仓开盘价口径 ④现金不足提前 break
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    # P-20260814-004: 记录本期暴露 (持仓股因子值 std + 全池因子值 std)
    try:
        held_vals = rp[top_idx][:len(top_stocks)]
        g.exposure_records.append((g.trade_days,
                                   float(np.std(held_vals)) if len(held_vals) > 1 else 0.0,
                                   float(np.std(rp[v])) if int(np.sum(v)) > 1 else 0.0))
        if len(g.exposure_records) > 500:  # 内存防御 (5.5年周频 ~290 期)
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
                break                           # 现金不足(前面平仓失败连锁), 停止开仓
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

    # ═══════════════════════════════════════════
    # Step 4: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("gp_breed_047 | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()
        _print_exposure_report()
        g.ic_reported = True

    gc.collect()
