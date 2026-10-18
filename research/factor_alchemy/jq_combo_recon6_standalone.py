# -*- coding: utf-8 -*-
"""
【组合】受限重构6: breed002+overreac+mf20+nearhigh+overnight5+dollar20
必选2候选(breed002+overreac)+MMR补齐4王者, 内部|corr|均值0.118/最大0.359。overreac为池内最正交且SR合格候选(0.396)。本地模拟+150.3%/SR0.732(仅方向参考)。
【参数】池 500 只 / top 30 / lookback 70 / 周频 / 等权 rank-product 合成
【回测建议区间】2021-01-01 ~ 2026-07-30 (与单因子 JQ 对齐)
【超时自救】若撞 1800s 限制: g.universe_size 降至 300, 或缩短回测区间
【P-001 合规】每因子输出滞后法 Rank IC/ICIR (半年报)
"""
import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 30
    g.lookback = 70
    g.universe_size = 500
    g.trade_days = 0
    g.ic_records = {"breed002": [], "overreac": [], "mf20": [], "nearhigh": [], "overnight5": [], "dollar20": []}
    g.last_values = {"breed002": [], "overreac": [], "mf20": [], "nearhigh": [], "overnight5": [], "dollar20": []}
    g.last_stocks = None
    g.last_closes = {}
    g.last_report_week = 0

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
    log.info("受限重构6: breed002+overreac+mf20+nearhigh+overnight5+dollar20 | 6 因子复合 | 池500 top30 启动")


# ═══ 辅助 ═══
def _rankdata_impl(a):
    """rankdata 降级实现: scipy 不可用时自实现"""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        return np.argsort(np.argsort(a)).astype(float) + 1.0


def rank_pct(arr, valid):
    """铁律15: rank(pct=True) 兼容实现, pct = rank/n_valid (与 pandas 3.x 语义一致)"""
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        return np.full(n, 0.5)
    ranks = _rankdata_impl(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    return result


def _unwrap_panel(px, field):
    """get_price 返回 Panel/DataFrame(MultiIndex)/dict 三层兼容拆字段"""
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        items = getattr(px, "items", [])
        if field in items:
            return px[field]
        try:
            return px.minor_xs(field)
        except Exception:
            return None
    if isinstance(px, dict):
        return px.get(field)
    if hasattr(px, "columns"):
        cols = px.columns
        if isinstance(cols, pd.MultiIndex):
            names = [c for c in cols.names if c]
            if field in cols.get_level_values(0):
                return px[field]
            if len(cols.levels) > 1 and field in cols.get_level_values(1):
                return px.xs(field, axis=1, level=1)
        return px
    return px


# ═══ 因子 (向量化宽表, 每因子一次 DataFrame 运算) ═══

def factor_breed002(stocks, data):
    """JQ冠军 +143.5%/SR0.51 (高价×20日前动量回看, 本质价格水平/微盘暴露)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = high_df - close_df.pct_change(20).shift(20)
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def factor_overreac(stocks, data):
    """过度反应反转 JQ +75.5%/SR0.396 (最正交候选, max|corr|0.366)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = -close_df.pct_change(1).rolling(20).max()
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def factor_mf20(stocks, data):
    """HLC3*量 相对20日均值 (王者底仓, 与tvma20共线0.873)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = -(((high_df + low_df + close_df) / 3.0 * volume_df) / ((high_df + low_df + close_df) / 3.0 * volume_df).rolling(20).mean() - 1.0)
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def factor_nearhigh(stocks, data):
    """距60日高点距离 (王者底仓)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = -(close_df.rolling(10).mean() / close_df.rolling(60).max())
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def factor_overnight5(stocks, data):
    """隔夜收益5日和 (王者底仓)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = (open_df / close_df.shift(1) - 1.0).rolling(5).sum()
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


def factor_dollar20(stocks, data):
    """成交额20日均值对数 (王者底仓)"""
    high_df = data["high"]
    close_df = data["close"]
    low_df = data["low"]
    volume_df = data["volume"]
    open_df = data["open"]
    f = -np.log((close_df * volume_df).rolling(20).mean())
    last = f.iloc[-1]
    arr = last.reindex(stocks).values.astype(float)
    valid = np.isfinite(arr)
    return arr, valid


# ═══ 因子名 → 函数 ═══
FACTOR_FUNC_MAP = {
    "breed002": factor_breed002,
    "overreac": factor_overreac,
    "mf20": factor_mf20,
    "nearhigh": factor_nearhigh,
    "overnight5": factor_overnight5,
    "dollar20": factor_dollar20,
}
FACTOR_NAMES = ["breed002", "overreac", "mf20", "nearhigh", "overnight5", "dollar20"]


# ═══ IC/ICIR 报告 (P-001) ═══
def _print_ic_report():
    for name in FACTOR_NAMES:
        recs = g.ic_records.get(name, [])
        if len(recs) == 0:
            log.info("[IC] %-12s NO IC RECORDS" % name)
            continue
        ic_arr = np.array(recs)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos = float(np.mean(ic_arr > 0))
        log.info("[IC] %-12s n=%4d  ICmean=%+.4f  ICIR=%+.4f  IC>0=%.1f%%" % (
            name, len(ic_arr), ic_mean, icir, ic_pos * 100))


# ═══ 周度调仓 ═══
def rebalance(context):
    g.trade_days += 1
    if g.trade_days < 5:
        return
    prev_date = context.previous_date

    # ── Step 1: 池过滤 (停牌/ST/涨停, 逐只 try) ──
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe_all), CHUNK):
        chunk = universe_all[k:k + CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                try:
                    if cd[s].paused or cd[s].is_st:
                        continue
                    if cd[s].high_limit <= cd[s].last_price:
                        continue
                    filtered.append(s)
                except Exception:
                    continue
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:g.universe_size]
    if len(universe) < g.stock_num:
        return

    # ── Step 2: get_price (单次多字段, Panel 兼容拆分) ──
    lb = g.lookback
    px = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                   fields=["high", "close", "low", "volume", "open"], skip_paused=False, fq="pre")
    close_df = _unwrap_panel(px, "close")
    if close_df is None or close_df.shape[0] < 22 or close_df.shape[1] == 0:
        if not getattr(g, "diag_no_data", False):
            log.info("[DIAG] get_price 空/不足 at %s shape=%s" % (prev_date, getattr(close_df, "shape", None)))
            g.diag_no_data = True
        return
    data = {"close": close_df}
    for f in ["high", "low", "volume", "open"]:
        df = _unwrap_panel(px, f)
        if df is None:
            if not getattr(g, "diag_missing_field", False):
                log.info("[DIAG] 字段 %s 缺失 at %s — 检查 fields 返回结构" % (f, prev_date))
                g.diag_missing_field = True
            return
        data[f] = df

    # ── Step 3: 逐因子计算 + rank_pct ──
    rps = {}
    for name in FACTOR_NAMES:
        arr, v = FACTOR_FUNC_MAP[name](universe, data)
        nv = int(np.sum(v))
        if nv < 10:
            if not getattr(g, "diag_low_valid_%s" % name, False):
                log.info("[DIAG] %s valid=%d (<10) at %s — 该因子本周不参与" % (name, nv, prev_date))
                setattr(g, "diag_low_valid_%s" % name, True)
            continue
        rps[name] = (rank_pct(arr, v), v)

    if len(rps) == 0:
        return

    # ── Step 4: 滞后 IC (上周各因子 rank → 本周收益 spearmanr) ──
    snap_closes = {}
    for s in close_df.columns:
        lv = close_df[s].iloc[-1]
        if np.isfinite(lv):
            snap_closes[s] = float(lv)

    if g.last_stocks is not None:
        for name in FACTOR_NAMES:
            lv_arr = g.last_values.get(name)
            if lv_arr is None:
                continue
            factor_vals = []
            rets = []
            for i, s in enumerate(g.last_stocks):
                pc = g.last_closes.get(s)
                cc = snap_closes.get(s)
                fv = lv_arr[i] if i < len(lv_arr) else np.nan
                if pc and cc and pc > 0 and cc > 0 and not np.isnan(fv):
                    factor_vals.append(fv)
                    rets.append((cc / pc) - 1.0)
            if len(factor_vals) >= 30:
                try:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(factor_vals, rets)
                    if ic == ic:
                        g.ic_records.setdefault(name, []).append(float(ic))
                except Exception:
                    pass

    # ── Step 5: 保存快照 ──
    g.last_stocks = list(universe)
    g.last_values = {name: rps[name][0].copy() for name in rps}
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ── Step 6: 等权 rank-product 合成选股 ──
    first_rp = None
    for name in rps:
        rp = rps[name][0]
        first_rp = rp.copy() if first_rp is None else first_rp + rp
    if first_rp is None:
        return
    composite = first_rp / len(rps)
    comp = np.where(np.isfinite(composite), -composite, np.inf)
    topn = min(g.stock_num, len(composite))
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [universe[i] for i in top_idx if comp[i] < np.inf]

    cd_order = get_current_data()

    # ── 平仓 ──
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

    # ── 开仓: 手数取整下单 (v2 修复, 防高价股委托失败) ──
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
            lots = int(w / ref_px / 100) * 100
            if lots < 100:
                continue
            order_target(s, lots)
            ordered += 1
        except Exception:
            continue

    # ── Step 7: 日志 & 定期 IC 报告 ──
    if g.trade_days <= 8:
        log.info("[DIAG] recon6 | week=%d | valid因子=%d/%d | top=%d | ordered=%d | cash=%.0f" % (
            g.trade_days, len(rps), len(FACTOR_NAMES), len(top_stocks), ordered, context.portfolio.available_cash))
    if g.trade_days - g.last_report_week >= 26:
        g.last_report_week = g.trade_days
        log.info("=" * 55)
        log.info("  受限重构6: breed002+overreac+mf20+nearhigh+overnight5+dollar20  每因子 IC/ICIR 半年报 (P-001)")
        log.info("=" * 55)
        _print_ic_report()
        log.info("=" * 55)
