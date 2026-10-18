# -*- coding: utf-8 -*-
"""
debt_dev_250 单因子独立回测 (本地周频裁决 PASS 因子, 待 JQ 定级)
因子含义: 资产负债率相对过去250交易日(≈50周)均值的偏离 — 加杠杆扩张溢价 (正IC方向, HIGH=做多)
Local (2021-01~2026-09, 5750股全市场, 公告日次日生效口径):
  周频 Rank IC=+0.0096 / 周频ICIR=+0.2334 (tau_w=0.15 ✅ WEEKLY PASS); 2周ICIR=+0.3906
  行业中性后不衰反升 ICIR=+0.2618; 防前视(次日生效) 0.2310 几乎零衰减
  fwd20日频: IC=+0.0219/ICIR=+0.4628/+IC%=73.0%
  年度: 2021 0.41/2022 0.53/2023 0.12/2024 0.08/2025 0.25/2026 0.15 (2023-24疲软, 周期性)
数据源: get_fundamentals(balance, date=prev_date) — 公告日次日生效, 无前视
P-001 合规: 输出 per-factor Rank IC/ICIR (周频滞后法)
请在 JQ 平台设置回测区间: 2019-01-01 ~ 2026-08-31 (2018 为预热期, 50周窗口需60周预热)
修订 2026-09-03: ①修复 balance.debt_to_total_asset 字段不存在 (JQ 无预计算负债率字段,
  改 total_liability/total_assets*100 现算); ②财务查询分块 300 只/次防 query 长度超限;
  ③get_price 传 panel=False 消除每周 Panel deprecation 警告 (2026-09-03 首跑日志观察)。
  ④🔴 修复 IC 采集故障: panel=False 多股单字段返回【长表】(time/code/close), 原按宽表解析
    → snap_closes 恒空 → 无 IC 记录 (qoq_change 首跑 281 周 IC 全 0 已实证)。
    本文件盘面版本若重跑会同样全 0, 现已兼容 长表pivot/Panel/MultiIndex 三种形态 + DIAG。
"""

import numpy as np
import pandas as pd
from collections import deque

def initialize(context):
    g.stock_num = 80
    g.trade_days = 0

    # --- 滚动窗口: code -> deque(maxlen=50) 存最近50个"周五可得"负债率 (≈250交易日) ---
    g.debt_win = {}
    g.warm_done = False
    g.warm_ok = 0

    # --- IC 收集 (滞后法: 上周因子 -> 本周收益的秩相关) ---
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
    log.info("debt_dev_250 单因子 (资产负债率250日偏离, 周频滞后IC) 启动")


# ═══ 辅助 ═══
def _norm_code(c):
    c = str(c)
    if "." in c:
        return c
    if c.startswith(("6", "9", "5")):
        return c + ".XSHG"
    return c + ".XSHE"


def _win_mean(q):
    """deque 均值 (铁律: 禁裸 sum, 显式循环)"""
    if q is None or len(q) == 0:
        return np.nan
    s = 0.0
    for x in q:
        s += float(x)
    return s / len(q)


def rank_pct(arr, valid):
    n = len(arr)
    valid_arr = arr[valid]
    if len(valid_arr) < 3:
        result = np.full(n, 0.5)
        return result
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / (len(valid_arr) - 1)
    result = np.full(n, 0.5)
    result[valid] = pct
    return result


# ═══ 数据源: 公告日次日生效的负债率快照 ═══
def _fetch_debt_snapshot(stocks, d):
    """get_fundamentals(date=d): d日已披露的最新一期财报 → 下一个交易日可用 (无前视)
    返回 {code: debt_to_total_asset(%)}
    JQ balance 表无预计算负债率字段 (debt_to_total_asset 为 Tushare 字段名),
    统一用 total_liability/total_assets*100 现算。
    分块 300 只/query: JQ query 字符串长度限制 ~1万字符, 2000 只全塞会超限报错。"""
    stocks = list(stocks)
    out = {}
    for i in range(0, len(stocks), 300):
        chunk = stocks[i:i + 300]
        q = query(
            balance.code,
            balance.total_liability,
            balance.total_assets,
        ).filter(balance.code.in_(chunk))
        try:
            df = get_fundamentals(q, date=d)
        except Exception as e:
            if not getattr(g, "diag_fund_err", False):
                log.info("[DIAG] get_fundamentals FAIL at %s: %r" % (d, e))
                g.diag_fund_err = True
            continue
        if df is None or len(df) == 0:
            if not getattr(g, "diag_fund_empty", False):
                log.info("[DIAG] get_fundamentals EMPTY at %s" % d)
                g.diag_fund_empty = True
            continue
        if not getattr(g, "diag_fund_ok", False):
            log.info("[DIAG] get_fundamentals OK at %s rows=%d cols=%s code_sample=%s" % (
                d, len(df), list(df.columns), df["code"].head(3).tolist() if "code" in df.columns else "NO_code_col"))
            g.diag_fund_ok = True

        df = df.copy()
        df["_c"] = df["code"].map(_norm_code)
        df = df.drop_duplicates("_c", keep="first")  # 最新披露一期优先
        for _, row in df.iterrows():
            try:
                ta = row.get("total_assets")
                tl = row.get("total_liability")
                if ta is None or tl is None:
                    continue
                ta, tl = float(ta), float(tl)
                if ta <= 0:
                    continue
                v = tl / ta * 100.0
                if v < 0 or v > 150:
                    continue
                out[row["_c"]] = v
            except Exception:
                continue
    return out


# ═══ 因子: debt_dev_250 ═══
def factor_debt_dev_250(stocks, prev_date):
    """debt_ff - rolling(250交易日).mean() 的 JQ 等价 (周频窗口≈50周; min_periods=12周)"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        q = g.debt_win.get(s)
        if q is not None and len(q) >= 12:
            m = _win_mean(q)
            arr[i] = q[-1] - m
            valid[i] = True
    return arr, valid


FACTOR_FUNC_MAP = {"debt_dev_250": factor_debt_dev_250}


# ═══ IC/ICIR 报告 ═══
def _print_ic_report():
    if len(g.ic_records) == 0:
        log.info("[IC] debt_dev_250: NO IC RECORDS")
        return
    ic_arr = np.array(g.ic_records)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))
    log.info("=" * 55)
    log.info("  debt_dev_250  Per-Factor IC/ICIR Report (P-001)")
    log.info("=" * 55)
    log.info("  Rank IC mean:       %+.4f" % ic_mean)
    log.info("  Rank IC std:         %.4f" % ic_std)
    log.info("  ICIR (mean/std):    %+.4f" % icir)
    log.info("  IC > 0 ratio:       %.1f%%" % (ic_pos * 100))
    log.info("  Total IC samples:   %d" % len(ic_arr))
    log.info("=" * 55)


# ═══ 周度调仓 ═══
def rebalance(context):
    import gc
    g.trade_days += 1

    prev_date = context.previous_date

    # ═══════════════════════════════════════════
    # Step 0: 股票池
    # ═══════════════════════════════════════════
    universe_all = get_all_securities(["stock"], prev_date).index.tolist()
    universe_all = [s for s in universe_all if s.startswith(("0", "3", "6"))][:2000]
    if len(universe_all) < 200:
        return

    # ═══════════════════════════════════════════
    # Step 1: 窗口预热 (一次性, 60个历史周四快照 → 50周窗口满)
    #   date=周四 = 周四收盘已披露 → 周五盘中可用 (与本地"周五取样"对齐)
    # ═══════════════════════════════════════════
    if not g.warm_done:
        _dates = []
        _d = pd.Timestamp(prev_date)
        _d -= pd.Timedelta(days=5)
        while len(_dates) < 60:
            if _d.weekday() == 3:  # 周四
                _dates.append(_d)
            _d -= pd.Timedelta(days=1)  # 逐日回溯 (7日步长会锁死 weekday)
        for _w in _dates:
            try:
                snap = _fetch_debt_snapshot(universe_all, _w)
                for _s, _v in snap.items():
                    g.debt_win.setdefault(_s, deque(maxlen=50)).append(_v)
                g.warm_ok += 1
            except Exception:
                continue
        g.warm_done = True
        _nwin = len([q for q in g.debt_win.values() if len(q) >= 12])
        log.info("[DIAG] warm done: snapshots=%d/%d, windows_ready=%d" % (
            g.warm_ok, len(_dates), _nwin))

    # 每周期 push 最新快照 (公告日次日生效: 用 prev_date 已披露数据)
    snap = _fetch_debt_snapshot(universe_all, prev_date)
    for _s, _v in snap.items():
        g.debt_win.setdefault(_s, deque(maxlen=50)).append(_v)

    if g.trade_days < 5:
        return

    # ═══════════════════════════════════════════
    # Step 2: 滞后 IC 计算 (上周因子 -> 本周收益)
    # ═══════════════════════════════════════════
    px_c_snap = get_price(universe_all, count=2, end_date=prev_date,
                          frequency="daily", fields="close",
                          skip_paused=False, fq="pre", panel=False)
    # 统一为 rows=日期 / columns=股票 的 close 宽表。
    # JQ panel=False 多股单字段实际返回【长表】(列名 time/code/close), 不是宽表;
    # 老环境可能返回 Panel 或 MultiIndex columns。三种形态全兼容:
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px_c_snap, _P):
        px_c_snap = px_c_snap["close"] if "close" in getattr(px_c_snap, "items", []) else px_c_snap.minor_xs("close")
    elif isinstance(px_c_snap, pd.DataFrame) and isinstance(px_c_snap.columns, pd.MultiIndex) \
            and "close" in px_c_snap.columns.get_level_values(0):
        px_c_snap = px_c_snap.xs("close", axis=1, level=0)
    elif isinstance(px_c_snap, pd.DataFrame) and "code" in px_c_snap.columns:
        # 长表 [time, code, close] → pivot 宽表
        idx_col = "time" if "time" in px_c_snap.columns else px_c_snap.columns[0]
        px_c_snap = px_c_snap.pivot(index=idx_col, columns="code", values="close")

    snap_closes = {}
    for s in universe_all:
        if s in px_c_snap.columns and len(px_c_snap[s].values) > 0:
            snap_closes[s] = float(px_c_snap[s].values[-1])
    if not getattr(g, "diag_close_ok", False):
        log.info("[DIAG] price_parse shape=%s snap_closes=%d/%d at %s" % (
            str(getattr(px_c_snap, "shape", "?")), len(snap_closes), len(universe_all), prev_date))
        g.diag_close_ok = True

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
            except:
                pass

    # ═══════════════════════════════════════════
    # Step 3: 池过滤 (逐只 try, 与 lhb v2 同口径)
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
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:1500]
    if len(universe) < g.stock_num:
        return

    arr, v = FACTOR_FUNC_MAP["debt_dev_250"](universe, prev_date)
    if int(np.sum(v)) < 10:
        if not getattr(g, "diag_low_valid", False):
            log.info("[DIAG] debt_dev_250 valid=%d (<10) at %s — 不交易 (窗口未满或数据源异常?)" % (int(np.sum(v)), prev_date))
            g.diag_low_valid = True
        return

    rp = rank_pct(arr, v)

    # ── 保存快照供下周 IC 计算 ──
    g.last_factor_stocks = universe
    g.last_factor_values = rp.copy()
    g.last_closes = {}
    for s in universe:
        if s in snap_closes:
            g.last_closes[s] = snap_closes[s]

    # ═══════════════════════════════════════════
    # Step 4: 选股 & 调仓 (HIGH = 做多: 取最高 rp)
    # ═══════════════════════════════════════════
    comp = np.where(v, -rp, np.inf)
    topn = min(g.stock_num, int(np.sum(v)))
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
            except:
                pass

    if len(top_stocks) == 0:
        return

    # ── 开仓: 等权 ──
    # 注: bar 内 available_cash 不随下单实时更新, 下方 break 实际很少触发;
    # 真实资金约束由 JQ 撮合层按可用现金自动截断, 持仓实际接近满仓等权
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
            if ref_px <= 0 or w / ref_px < 100:
                continue
            _pos = context.portfolio.positions.get(s)
            if _pos is not None and getattr(_pos, "total_amount", 0) > 0:
                _delta = int(w / ref_px) - int(_pos.total_amount)
                if abs(_delta) < 100:
                    continue
            order_target_value(s, w)
            ordered += 1
        except:
            continue

    # ═══════════════════════════════════════════
    # Step 5: 日志 & 定期 IC 报告
    # ═══════════════════════════════════════════
    latest_ic = g.ic_records[-1] if g.ic_records else 0.0
    log.info("debt_dev_250 | week=%d | ordered=%d/%d | stocks=%d | IC=%.4f | total_ic=%d" % (
        g.trade_days, ordered, len(top_stocks), int(np.sum(v)),
        latest_ic, len(g.ic_records)))

    if len(g.ic_records) >= 20 and g.trade_days % 26 == 0:
        _print_ic_report()

    gc.collect()
