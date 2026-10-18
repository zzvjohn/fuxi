# -*- coding: utf-8 -*-
"""
V0.4 Ralph Loop 策略 — Directed GP 育种 (蒸馏版)
================================================
生成时间: 2026-08-07 23:03:25 (修复版: literal \\n bug fixed)
因子来源: GP 交叉育种 + S5 联合过滤 + JQ 反馈蒸馏
策略架构: run_weekly + 纯 numpy per-stock 计算 + rank_pct 复合

JQ 回测参数:
- 起始: 2021-01-01
- 结束: 2026-07-30 (或 latest)
- 频率: 周频 (run_weekly)
- 基准: 中证500 (000905.XSHG)
- Warmup: 125天 (约 6 个月)

因子列表 (5 个):
  1. momentum_acceleration: close_p.pct_change(5) - close_p.pct_change(20)
  2. gp_breed_006: (-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())
  3. intraday_reversal: -(close_p / open_p - 1).rolling(5).mean()
  4. gp_breed_008: (np.abs(close_p) - 1).rolling(5).mean()
  5. gp_breed_011: (high_p - close_p.pct_change(20).shift(20))
"""

from jqdata import *
import pandas as pd
import numpy as np

# P-20260901-012 JQ沙箱: 裸 sum() 实为 np.sum, np.sum(dict.values()) 原样返回
# dict_values / np.sum(generator) 抛 TypeError -> 一律 _sum_vals() 手动累加
def _sum_vals(seq):
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t

def initialize(context):
    """初始化"""
    g.universe = get_index_stocks("000905.XSHG")
    g.N = 5
    g.top_n = min(80, len(g.universe) // 4)
    g.warmup_days = 125
    g.min_positions = 5
    g.weekly_rebalance = True
    g.price_cache = {}
    
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_benchmark('000905.XSHG')
    
    run_weekly(rebalance, weekday=1, time='open')
    log.info(f"V0.4 Directed GP 策略: {g.N} 因子")


def get_price_series(universe, field, count, end_date):
    """获取价格数据, 返回 {stock: numpy_array} 字典"""
    field_name_map = {"close": "close", "open": "open", "high": "high",
                       "low": "low", "volume": "volume", "money": "amount"}
    jq_field = field_name_map.get(field, field)
    px = get_price(universe, count=count, end_date=end_date,
                   frequency="daily", fields=jq_field,
                   skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        px = px[jq_field] if jq_field in getattr(px, "items", []) else px.minor_xs(jq_field)
    return {s: px[s].values for s in universe if s in px.columns}


def factor_momentum_acceleration(stocks, price_data):
    """close_p.pct_change(5) - close_p.pct_change(20)"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            result = close_p.pct_change(5) - close_p.pct_change(20)
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


def factor_gp_breed_006(stocks, price_data):
    """(-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    opend = price_data.get("open", {})
    highd = price_data.get("high", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30:
            continue
        open_arr = opend.get(s)
        if open_arr is None or len(open_arr) < 30:
            continue
        high_arr = highd.get(s)
        if high_arr is None or len(high_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            open_p = pd.Series(np.array(open_arr, dtype=float))
            high_p = pd.Series(np.array(high_arr, dtype=float))
            hl_ratio = (high_p - close_p.shift(1)) / close_p.shift(1).replace(0, np.nan)
            result = (-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


def factor_intraday_reversal(stocks, price_data):
    """-(close_p / open_p - 1).rolling(5).mean()"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    opend = price_data.get("open", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30:
            continue
        open_arr = opend.get(s)
        if open_arr is None or len(open_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            open_p = pd.Series(np.array(open_arr, dtype=float))
            result = -(close_p / open_p - 1).rolling(5).mean()
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


def factor_gp_breed_008(stocks, price_data):
    """(np.abs(close_p) - 1).rolling(5).mean()"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            result = (np.abs(close_p) - 1).rolling(5).mean()
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


def factor_gp_breed_011(stocks, price_data):
    """(high_p - close_p.pct_change(20).shift(20))"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    highd = price_data.get("high", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        if close_arr is None or len(close_arr) < 30:
            continue
        high_arr = highd.get(s)
        if high_arr is None or len(high_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            high_p = pd.Series(np.array(high_arr, dtype=float))
            result = (high_p - close_p.pct_change(20).shift(20))
            if isinstance(result, pd.Series):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else result.dropna().iloc[-1] if len(result.dropna()) > 0 else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══ 因子映射 ═══
FACTOR_FUNC_MAP = {
    "momentum_acceleration": factor_momentum_acceleration,
    "gp_breed_006": factor_gp_breed_006,
    "intraday_reversal": factor_intraday_reversal,
    "gp_breed_008": factor_gp_breed_008,
    "gp_breed_011": factor_gp_breed_011,
}


def get_required_price_fields():
    """收集所有因子需要的 JQ fields"""
    return {"close", "open", "high"}


def rebalance(context):
    """周频调仓"""
    universe = g.universe
    
    # ── 数据拉取 ──
    lb = 125
    prev_date = context.previous_date
    fields = get_required_price_fields()
    
    price_data = {}
    for field in fields:
        try:
            price_data[field] = get_price_series(universe, field, lb, prev_date)
        except Exception as e:
            log.warning(f"fetch {field} failed: {e}")
    
    stocks = [s for s in universe if s in price_data.get("close", {})]
    if len(stocks) < g.min_positions:
        return
    
    # ── 逐因子计算 ──
    factor_scores = []
    for factor_name, factor_fn in FACTOR_FUNC_MAP.items():
        try:
            raw, valid = factor_fn(stocks, price_data)
            if valid.sum() >= g.min_positions:
                scores = pd.Series(raw, index=stocks)
                scores[~valid] = np.nan
                pct = scores.rank(pct=True, na_option='bottom')
                factor_scores.append((factor_name, pct))
        except Exception as e:
            log.warning(f"factor {factor_name} failed: {e}")
    
    if not factor_scores:
        return
    
    # ── 等权复合 ──
    composite = _sum_vals(s for _, s in factor_scores) / len(factor_scores)
    composite = composite.dropna()
    
    if len(composite) < g.min_positions:
        return
    
    # ── Top-N 选股 ──
    top = composite.nlargest(g.top_n)
    top_stocks = list(top.index)
    
    # ── 执行调仓 ──
    weight = 1.0 / len(top_stocks)
    for s in context.portfolio.positions:
        if s not in top_stocks:
            order_target(s, 0)
    
    for s in top_stocks:
        order_target_value(s, context.portfolio.total_value * weight)
    
    log.info(f"rebalance: {len(top_stocks)} stocks weighted by {g.N} factors")
