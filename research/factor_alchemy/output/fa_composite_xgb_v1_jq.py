# -*- coding: utf-8 -*-
"""
FA Composite XGB v1r — XGBoost 交互对验证 (反转版) | 5复合 JQ 真回测
======================================================================
生成: 2026-07-30 15:30 | 方向: 动量→反转 (ret_1m/roc_20/bias_20/max_ret_1m 取负)

v1 原始版: 纯动量 → JQ -90%+ (A股短期反转)
v1r 反转版: 全部翻转为反转信号

5 rank-pct 复合 (全部含反转 ret_1m 侧):
  comp_ret_1m_x_max_ret_1m = 反转 × 反转 (避开暴涨的跌股)
  comp_bias_20_x_rank_1m = 反转 × 反转 (低于均线的跌股)
  comp_ret_1m_x_roc_20 = 反转² (强反转)
  comp_ret_1m_x_high_low_range = 反转 × 窄幅 (沉闷跌股)
  comp_turnover_change_x_rank_1m = 量能 × 反转 (放量下跌)

OHLCV 因子(6): bias_20, high_low_range, max_ret_1m, ret_1m, roc_20, turnover_change
基本面因子(0): 

参数: 持仓=30, 周频(周四14:50), 基准=沪深300
成本: 佣金万1 + 印花千1 + 滑点0.3%
================================================================
"""

from __future__ import division
import datetime
import gc
import numpy as np
import pandas as pd

# P-20260901-012 JQ沙箱: 裸 sum() 实为 np.sum, np.sum(dict.values()) 原样返回
# dict_values / np.sum(generator) 抛 TypeError -> 一律 _sum_vals() 手动累加
def _sum_vals(seq):
    _t = 0.0
    for _v in seq:
        _t = _t + _v
    return _t


# ==================== 策略参数 ====================
STOCK_NUM      = 30
MAX_TURNOVER   = 0.60
MCAP_MIN       = 5
EXCLUDE_PREFIX = ('8', '4')
CHUNK          = 500
LOOKBACK       = 65

# ICIR 权重
ICIR = {
    'comp_ret_1m_x_max_ret_1m': 0.2122,
    'comp_bias_20_x_rank_1m': 0.1945,
    'comp_ret_1m_x_roc_20': 0.2064,
    'comp_ret_1m_x_high_low_range': 0.1701,
    'comp_turnover_change_x_rank_1m': 0.2168,
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        if hasattr(x, '__iter__'):
            xs = list(x)
            return float(xs[0]) if xs else 0.0
        return 0.0

_WSUM = _sum_vals(_num(v) for v in ICIR.values())
COMPOSITE_WEIGHTS = {k: _num(v) / _WSUM for k, v in ICIR.items()}

DEBUG = True


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0,
    ), type='stock')
    set_slippage(FixedSlippage(0.003))

    run_weekly(main, weekday=4, time='14:50')

    g.stock_num     = STOCK_NUM
    g.max_turnover  = MAX_TURNOVER
    g.mcap_min      = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX
    g.debug         = DEBUG


# ==================== 股票池 ====================
def get_stock_pool(context):
    date = context.previous_date
    try:
        all_stocks = get_index_stocks('000985.XSHG', date=date)
    except Exception:
        all_stocks = list(get_all_securities(types=['stock'], date=date).index)

    stocks = [s for s in all_stocks if not s.startswith(g.exclude_prefix)]
    if len(stocks) == 0:
        return []

    q = query(valuation.code, valuation.market_cap).filter(
        valuation.code.in_(stocks), valuation.market_cap >= g.mcap_min)
    df = get_fundamentals(q, date=context.previous_date)
    stocks = list(df['code']) if not df.empty else []
    if len(stocks) == 0:
        return []

    cd = get_current_data()
    final = []
    for s in stocks:
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        final.append(s)
    return final


# ==================== 价格加载 ====================
def _get_price_field(stocks, end_dt, count, field):
    px = get_price(stocks, end_date=end_dt, count=count,
                   frequency='daily', fields=field,
                   skip_paused=False, fq='pre')
    _Panel = getattr(pd, 'Panel', None)
    if _Panel is not None and isinstance(px, _Panel):
        px = px[field] if field in getattr(px, 'items', []) else px.minor_xs(field)
    return px



# ==================== OHLCV 因子计算 (分块) ====================
def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df,
                                stocks, fundamental_data):
    """计算 7 个 OHLCV 因子 — 返回标量 Series (stock→value)."""
    result = {}

    # bias_20: -(close-MA20)/MA20 (反转: 买低于均线的)
    ma20 = close_df.rolling(20, min_periods=10).mean()
    result['bias_20'] = -((close_df - ma20) / (ma20 + 1e-9)).iloc[-1]

    # high_low_range: -(high - low) / close  (窄幅=高值, 保持)
    result['high_low_range'] = (-(high_df - low_df) / close_df).iloc[-1]

    # max_ret_1m: -20日最大单日涨幅 (反转: 避开暴涨股)
    daily_ret = close_df.pct_change()
    result['max_ret_1m'] = -daily_ret.rolling(20, min_periods=10).max().iloc[-1]

    # ret_1m: 20日反转 (取负 = 买跌不买涨, 与V3 ret_3m 一致)
    result['ret_1m'] = -(close_df / close_df.shift(20) - 1.0).iloc[-1]

    # roc_20: 同上反转
    result['roc_20'] = -(close_df / close_df.shift(20) - 1.0).iloc[-1]

    # turnover_change: volume MA5/MA20 比值 (量能变化)
    ma5 = volume_df.rolling(5, min_periods=3).mean()
    ma20_v = volume_df.rolling(20, min_periods=10).mean()
    result['turnover_change'] = (ma5 / (ma20_v + 1e-9)).iloc[-1]

    return result


# ==================== 截面 rank % ====================
def _rank_pct(series):
    s = series.dropna()
    if len(s) == 0:
        return series
    return s.rank(pct=True)


# ==================== 复合因子构建 ====================
def build_composites(base_df):
    """5 复合 (rank_pct 乘积)."""

    # Rank all base factors
    r_bias_20 = _rank_pct(base_df['bias_20'])
    r_high_low_range = _rank_pct(base_df['high_low_range'])
    r_max_ret_1m = _rank_pct(base_df['max_ret_1m'])
    r_rank_1m = _rank_pct(base_df['ret_1m'])
    r_ret_1m = _rank_pct(base_df['ret_1m'])
    r_roc_20 = _rank_pct(base_df['roc_20'])
    r_turnover_change = _rank_pct(base_df['turnover_change'])

    return {
        'comp_ret_1m_x_max_ret_1m': r_ret_1m * r_max_ret_1m,
        'comp_bias_20_x_rank_1m': r_bias_20 * r_rank_1m,
        'comp_ret_1m_x_roc_20': r_ret_1m * r_roc_20,
        'comp_ret_1m_x_high_low_range': r_ret_1m * r_high_low_range,
        'comp_turnover_change_x_rank_1m': r_turnover_change * r_rank_1m,
    }


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    if g.debug:
        log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    end_dt = context.previous_date

    base_factors = ['bias_20', 'high_low_range', 'max_ret_1m', 'ret_1m', 'roc_20', 'turnover_change']
    base_data = {f: {} for f in base_factors}


    # 分块加载 OHLCV
    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        res = compute_base_factors_chunk(open_df, high_df, low_df, close_df,
                                          volume_df, chunk, base_data)
        for f in base_factors:
            if f in res:
                base_data[f].update(res[f].to_dict())

        del open_df, high_df, low_df, close_df, volume_df, res
        gc.collect()

    base_df = pd.DataFrame(base_data)
    comp = build_composites(base_df)

    # ICIR 加权合成
    total = (
        COMPOSITE_WEIGHTS['comp_ret_1m_x_max_ret_1m'] * comp['comp_ret_1m_x_max_ret_1m'] +
        COMPOSITE_WEIGHTS['comp_bias_20_x_rank_1m'] * comp['comp_bias_20_x_rank_1m'] +
        COMPOSITE_WEIGHTS['comp_ret_1m_x_roc_20'] * comp['comp_ret_1m_x_roc_20'] +
        COMPOSITE_WEIGHTS['comp_ret_1m_x_high_low_range'] * comp['comp_ret_1m_x_high_low_range'] +
        COMPOSITE_WEIGHTS['comp_turnover_change_x_rank_1m'] * comp['comp_turnover_change_x_rank_1m']
    )
    total = total.dropna().sort_values(ascending=False)
    if len(total) == 0:
        return

    target = total.head(g.stock_num).index.tolist()
    if g.debug:
        log.info('[%s] top3 scores: %s' % (date_str,
                 ', '.join('%.4f' % v for v in total.head(3).values)))

    rebalance(context, target)


# ==================== 调仓 ====================
def rebalance(context, target):
    target_set = set(target)
    positions = context.portfolio.positions
    current = set(positions.keys())

    total_value = context.portfolio.total_value
    unit = total_value / g.stock_num
    turnover = 0.0
    for s in current - target_set:
        turnover += abs(positions[s].value)
    for s in target:
        cur = positions[s].value if s in positions else 0.0
        turnover += abs(unit - cur)
    turnover = turnover / total_value if total_value > 0 else 0.0

    if turnover > g.max_turnover and g.debug:
        log.info('[rebalance] turnover=%.2f > cap=%.2f' % (turnover, g.max_turnover))

    for s in current - target_set:
        order_target_value(s, 0)

    cd = get_current_data()
    for s in target:
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        order_target_value(s, unit)