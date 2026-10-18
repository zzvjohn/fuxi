# -*- coding: utf-8 -*-
"""
V0.4.1 Ralph Loop 策略 — 修复版 (v0.4 bugfix)
==============================================
生成时间: 2026-08-07 23:42
修复内容:
  1. gp_breed_006 hl_ratio 定义修正: S5=high/low, 之前JQ误写为(high-close_prev)/close_prev
  2. 移除 gp_breed_008: np.abs(close)-1 = 价格垃圾因子 (S5 false positive)
  3. 移除 gp_breed_011: high - pct_change = 单位混合垃圾 (S5 false positive)
  4. 新增 gp_breed_000: v0.3.1 验证有效因子 (-((close/rolling_min(60))-close))
  5. S5 新增 _check_formula_sanity() 防止垃圾因子漏网

因子来源: v0.3.1 验证因子 + v0.4 S5过滤因子 (高质量子集)
策略架构: run_weekly + 纯 numpy per-stock 计算 + rank_pct 复合

JQ 回测参数:
- 起始: 2021-01-01
- 结束: 2026-07-30
- 频率: 周频 (run_weekly)
- 基准: 中证500 (000905.XSHG)
- Warmup: 125天 (约 6 个月)

因子列表 (4 个):
  1. momentum_acceleration: close_p.pct_change(5) - close_p.pct_change(20)  [趋势动量域]
  2. intraday_reversal: -(close_p / open_p - 1).rolling(5).mean()  [价格形态域]
  3. gp_breed_006 (fixed): (-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())  [价格形态域, hl_ratio=high/low]
  4. gp_breed_000: -((close / rolling_min(60)) - close)  [均值回复域]
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
    g.N = 4
    g.top_n = min(80, len(g.universe) // 4)
    g.warmup_days = 125
    g.min_positions = 5
    g.weekly_rebalance = True
    g.price_cache = {}
    
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_benchmark('000905.XSHG')
    
    run_weekly(rebalance, weekday=1, time='open')
    log.info(f"V0.4.1 修复版策略: {g.N} 因子")


def get_price_series(universe, field, count, end_date):
    """获取价格数据, 返回 {stock: numpy_array} 字典"""
    field_name_map = {
        "close": "close", "open": "open", "high": "high",
        "low": "low", "volume": "volume", "money": "amount"
    }
    jq_field = field_name_map.get(field, field)
    px = get_price(universe, count=count, end_date=end_date,
                   frequency="daily", fields=jq_field,
                   skip_paused=False, fq="pre")
    _P = getattr(pd, "Panel", None)
    if _P is not None and isinstance(px, _P):
        px = px[jq_field] if jq_field in getattr(px, "items", []) else px.minor_xs(jq_field)
    return {s: px[s].values for s in universe if s in px.columns}


# ═══════════════════════════════════════════════
#  因子 1: momentum_acceleration
#  域: 趋势动量 | 公式: close_p.pct_change(5) - close_p.pct_change(20)
# ═══════════════════════════════════════════════
def factor_momentum_acceleration(stocks, price_data):
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
            if hasattr(result, 'iloc'):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子 2: intraday_reversal
#  域: 价格形态 | 公式: -(close_p / open_p - 1).rolling(5).mean()
# ═══════════════════════════════════════════════
def factor_intraday_reversal(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    opend = price_data.get("open", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        open_arr = opend.get(s)
        if close_arr is None or open_arr is None or len(close_arr) < 30 or len(open_arr) < 30:
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            open_p = pd.Series(np.array(open_arr, dtype=float))
            result = -(close_p / open_p - 1).rolling(5).mean()
            if hasattr(result, 'iloc'):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子 3: gp_breed_006 (fixed)
#  域: 价格形态 | hl_ratio = high / low (修复!)
#  公式: (-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())
# ═══════════════════════════════════════════════
def factor_gp_breed_006(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    closed = price_data.get("close", {})
    opend = price_data.get("open", {})
    highd = price_data.get("high", {})
    lowd = price_data.get("low", {})
    for i, s in enumerate(stocks):
        close_arr = closed.get(s)
        open_arr = opend.get(s)
        high_arr = highd.get(s)
        low_arr = lowd.get(s)
        if any(x is None or len(x) < 30 for x in [close_arr, open_arr, high_arr, low_arr]):
            continue
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            open_p = pd.Series(np.array(open_arr, dtype=float))
            high_p = pd.Series(np.array(high_arr, dtype=float))
            low_p = pd.Series(np.array(low_arr, dtype=float))
            # FIXED: hl_ratio = high / low (matches S5 definition)
            hl_ratio = high_p / low_p.replace(0, np.nan)
            result = (-((close_p / open_p) - (high_p * hl_ratio)).rolling(5).mean())
            if hasattr(result, 'iloc'):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else np.nan
            else:
                arr[i] = float(result)
            valid[i] = not pd.isna(arr[i])
        except Exception:
            continue
    return arr, valid


# ═══════════════════════════════════════════════
#  因子 4: gp_breed_000 (v0.3.1 验证有效)
#  域: 均值回复 | 公式: -((close / rolling_min(60)) - close)
#  含义: 价格偏离60日低点程度的反向信号
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
        try:
            close_p = pd.Series(np.array(close_arr, dtype=float))
            min60 = close_p.rolling(60).min()
            result = -((close_p / min60.replace(0, np.nan)) - close_p)
            if hasattr(result, 'iloc'):
                arr[i] = result.iloc[-1] if not pd.isna(result.iloc[-1]) else np.nan
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
    "momentum_acceleration": factor_momentum_acceleration,
    "intraday_reversal": factor_intraday_reversal,
    "gp_breed_006": factor_gp_breed_006,
    "gp_breed_000": factor_gp_breed_000,
}


def get_required_price_fields():
    """收集所有因子需要的 JQ fields"""
    return {"close", "open", "high", "low"}


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
