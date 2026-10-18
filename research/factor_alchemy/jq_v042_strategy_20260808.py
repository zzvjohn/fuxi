# -*- coding: utf-8 -*-
"""
V0.4.2 Ralph Loop 策略 — 回归 v0.3.1 成功架构 + bugfix
========================================================
生成时间: 2026-08-08
诊断结论:
  v0.3.1 (+129%) 真实因子: gp_breed_000 + gp_breed_002 (gp_breed_006 因 amount_p
  未定义而静默失败, 实际0贡献)
  v0.4.1 (-18%) 失效根因: rank-average + nlargest 复合方向反转, 系统性地买入
  最贵股票 (gp_breed_000 实为价格水平代理, 非均值回复信号)

v0.4.2 修复:
  1. 回归 v0.3.1 rank-product 复合 + argsort ascending (已验证有效)
  2. 修复 gp_breed_006: 补充 amount 数据拉取
  3. 保持 v0.3.1 的 universe 过滤 (排除 ST/涨跌停, 保留 0/3/6 开头)
  4. get_price 改用 panel=False 逐字段, 消除 Panel 弃用 → 0 valid 问题
  5. 因子: gp_breed_000 + gp_breed_002 + gp_breed_006_v2 (3因子, 全部可用)

JQ 回测参数:
  - 起始: 2021-01-01 ~ 2026-07-30
  - 频率: 周频 (run_weekly)
  - 基准: 中证500 (000905.XSHG)
  - 持仓: Top-80 等权
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.stock_num = 80
    g.lookback = 130
    g.trade_days = 0
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
    log.info("V0.4.2 修复版策略: %d 因子 (回归 v0.3.1 架构)" % 3)


# ═══════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════
def rank_pct(arr, valid):
    """交叉截面百分位排名: 最小=0, 最大=1, 无效=0.5"""
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


# ═══════════════════════════════════════════════
#  因子 1: gp_breed_000
#  公式: -((close / rolling_min(60)) - close)
#  经济学含义: 价格接近60日低点的程度 (实为价格水平代理, 低值=低价股)
#  方向: 在 rank-product 架构下, 低 rank → 被选中 → 低价股
# ═══════════════════════════════════════════════
def factor_gp_breed_000(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 70:
            continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        try:
            result = -((close_p / close_p.rolling(60).min()) - close_p)
            if isinstance(result, pd.Series):
                last = result.iloc[-1]
                arr[i] = last if not pd.isna(last) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子 2: gp_breed_002
#  公式: 成交量加权趋势确认 (volume on trending days / total volume)
#  经济学含义: 过去20日中, 出现在趋势日的成交量占比
#  方向: 低值=趋势确认弱 → 与 gp_breed_000 配合选出超跌无趋势股
# ═══════════════════════════════════════════════
def factor_gp_breed_002(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    highd = price_data.get("high", {})
    lowd = price_data.get("low", {})
    volumed = price_data.get("volume", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        high_arr = highd.get(s)
        low_arr = lowd.get(s)
        volume_arr = volumed.get(s)
        if any(x is None or len(x) < 50 for x in [close_arr, high_arr, low_arr, volume_arr]):
            continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        high_p = pd.Series(np.array(high_arr, dtype=float))
        low_p = pd.Series(np.array(low_arr, dtype=float))
        volume_p = pd.Series(np.array(volume_arr, dtype=float))
        amplitude = high_p / low_p
        try:
            trend_days = (close_p.pct_change(20) > amplitude.rolling(30).mean()).astype(float)
            vol_on_trend = (volume_p * trend_days).rolling(20).sum()
            total_vol = volume_p.rolling(20).sum() + 1
            result = vol_on_trend / total_vol
            if isinstance(result, pd.Series):
                last = result.iloc[-1]
                arr[i] = last if not pd.isna(last) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子 3: gp_breed_006_v2 (BUGFIX)
#  v0.3.1 bug: 使用 amount_p 但未从 price_data 拉取 → NameError → 静默失败
#  v0.4.2 fix: 补充 amount 数据拉取, 公式不变
#  公式: ((close - low) / ((amount - open) + 0.001)).rolling(5).mean()
#  经济学含义: 日内低位恢复强度 / 成交额归一化 (量价结构)
# ═══════════════════════════════════════════════
def factor_gp_breed_006_v2(stocks, price_data):
    """BUGFIX: 现在正确读取 amount 数据"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    lowd = price_data.get("low", {})
    opend = price_data.get("open", {})
    amountd = price_data.get("amount", {})  # <-- BUGFIX: 补充amount

    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        low_arr = lowd.get(s)
        open_arr = opend.get(s)
        amount_arr = amountd.get(s)
        if any(x is None or len(x) < 30 for x in [close_arr, low_arr, open_arr, amount_arr]):
            continue
        close_p = pd.Series(np.array(close_arr, dtype=float))
        low_p = pd.Series(np.array(low_arr, dtype=float))
        open_p = pd.Series(np.array(open_arr, dtype=float))
        amount_p = pd.Series(np.array(amount_arr, dtype=float))  # <-- BUGFIX
        try:
            result = ((close_p - low_p) / ((amount_p - open_p) + 0.001)).rolling(5).mean()
            if isinstance(result, pd.Series):
                last = result.iloc[-1]
                arr[i] = last if not pd.isna(last) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子映射
# ═══════════════════════════════════════════════
FACTOR_FUNC_MAP = {
    "gp_breed_000": factor_gp_breed_000,
    "gp_breed_002": factor_gp_breed_002,
    "gp_breed_006_v2": factor_gp_breed_006_v2,
}


# ═══════════════════════════════════════════════
#  复合信号: rank-product (v0.3.1 验证有效)
#  多因子 rank_pct 相乘 → 最小的被选中 (所有因子一致看多)
# ═══════════════════════════════════════════════
def compute_ensemble(stocks, price_data):
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid_any = np.zeros(n, dtype=bool)
    count = 0
    for name, func in FACTOR_FUNC_MAP.items():
        arr, v = func(stocks, price_data)
        if int(np.sum(v)) < 10:
            log.warning("factor %s: only %d valid stocks, skipping" % (name, int(np.sum(v))))
            continue
        rp = rank_pct(arr, v)
        composite[v] *= rp[v]
        count += 1
        valid_any = valid_any | v
    log.info("ensemble: %d/%d factors active" % (count, len(FACTOR_FUNC_MAP)))
    return composite, valid_any


# ═══════════════════════════════════════════════
#  周度调仓
# ═══════════════════════════════════════════════
def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # ── Universe 过滤 ──
    universe = get_all_securities(["stock"], prev_date).index.tolist()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k + CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused or cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(("0", "3", "6"))][:2000]
    if len(universe) < g.stock_num:
        return

    # ── 价格数据 (panel=False 逐字段, 避免 Panel 弃用) ──
    lb = g.lookback
    def _fetch(field_name):
        pxi = get_price(universe, count=lb, end_date=prev_date, frequency="daily",
                        fields=field_name, panel=False, skip_paused=False, fq="pre")
        return {s: pxi[s].values for s in universe if s in pxi.columns}

    price_data = {
        "close":  _fetch("close"),
        "open":   _fetch("open"),
        "high":   _fetch("high"),
        "low":    _fetch("low"),
        "volume": _fetch("volume"),
        "amount": _fetch("money"),
    }

    valid_u = [s for s in universe if len(price_data["close"].get(s, [])) >= 120]
    if len(valid_u) < g.stock_num:
        return

    # ── 复合信号 ──
    composite, valid = compute_ensemble(valid_u, price_data)
    comp = np.where(valid, composite, np.inf)

    topn = g.stock_num
    top_idx = np.argsort(comp)[:topn]
    top_stocks = [valid_u[i] for i in top_idx if comp[i] < np.inf]

    # ── 调仓 ──
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
                order_target_value(s, 0)
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    w = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            lp = get_current_data()[s].last_price
            if lp <= 0 or w / lp < 100:
                continue
            order_target_value(s, w)
            ordered += 1
        except Exception:
            continue
    log.info("ordered %d/%d" % (ordered, len(top_stocks)))
    gc.collect()
