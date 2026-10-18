# -*- coding: utf-8 -*-
"""
================================================================
FA Composite v3 — 5复合 JQ 真回测  [ICIR≥0.30 / 因果+基本面扩容 / FDR/ENB诊断]
================================================================
2026-07-28 生成 | 多窗口 WFA pooled: +44.5%/SR1.87/MaxDD-23.5% | 全样本: +24.0%/SR1.01

5 rank-percentile 乘积复合因子 (80 base 池 → 3160 配对 → 5 选中):
  comp1 = overnight_5d(+)         × tvma_20(+)                ICIR +0.772  权重 20.2%
  comp2 = dollar_vol_20d(+)       × turnover_std_cv(+)         ICIR +0.747  权重 20.2%
  comp3 = money_flow_20(+)        × value_momentum(+)          ICIR +0.695  权重 20.0%
  comp4 = ret_open_2d(+)          × skewness_20(+)             ICIR +0.684  权重 19.9%
  comp5 = gap_up(+)               × relative_spread(+)          ICIR +0.687  权重 19.8%

v2→v3 变更: earnings_volume_drift → gap_up×relative_spread (composer 因果+基本面扩容后仍选同一结构)
         FDR(BH): 2379→2075(87%) | ENB(选中集)=1.02(≈1独立信号) | 分年ICIR: 100%正年但2026衰减

★ value_momentum 本身是嵌套复合: rank_pct(1/PB) × rank_pct(-ret_3m)

参数: 持仓=30, 周频(周五14:50), 基准=沪深300
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

# ICIR 权重 (v3: 5复合, 来自 composite_v3 训练, ICIR≥0.30)
ICIR = {
    'comp1_overnight_tvma':       0.7719,
    'comp2_dollar_turnover':      0.7472,
    'comp3_moneyflow_value':      0.6946,
    'comp4_retopen_skew':         0.6843,
    'comp5_gapup_spread':         0.6870,
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


# ==================== PB 快照 ====================
def load_pb_map(context, stocks):
    q = query(valuation.code, valuation.pb_ratio).filter(
        valuation.code.in_(stocks))
    val = get_fundamentals(q, date=context.previous_date)
    return dict(zip(val['code'], val['pb_ratio'])) if not val.empty else {}


# ==================== 逐股票标量因子 (分块计算) ====================
def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df,
                                stocks, pb_map):
    """10 个逐股票标量因子 (5复合 = rank_pct 乘积配对)."""

    dollar_vol_raw = close_df * volume_df

    # 1. dollar_vol_20d = -log(mean(close*volume, 20))
    avg_dv = dollar_vol_raw.rolling(20, min_periods=10).mean()
    dollar_vol_20d = -np.log(avg_dv.iloc[-1].replace(0, np.nan))

    # 2. turnover_std_cv = -CV(volume, 20)  [volume CV 代理]
    vs = volume_df.rolling(20, min_periods=10).std()
    vm = volume_df.rolling(20, min_periods=10).mean()
    turnover_std_cv = -(vs / vm.replace(0, np.nan)).iloc[-1]

    # 3. overnight_5d = rolling_sum(open/close.shift(1)-1, 5)
    ovn = open_df / close_df.shift(1) - 1.0
    overnight_5d = ovn.rolling(5, min_periods=3).sum().iloc[-1]

    # 4. tvma_20 = -clip(money/ma(money,20), 0, 10)
    money_raw = close_df * volume_df
    ma20 = money_raw.rolling(20, min_periods=20).mean()
    tvma_20 = (-(money_raw / ma20.replace(0, np.nan))).clip(0, 10).iloc[-1]

    # 5. money_flow_20 = -((h+l+c)/3*vol/ma20-1)
    tp = (high_df + low_df + close_df) / 3.0
    mf = tp * volume_df
    mf_ma = mf.rolling(20, min_periods=20).mean()
    money_flow_20 = -(mf / mf_ma.replace(0, np.nan) - 1.0).iloc[-1]

    # 6. bp = 1/PB
    bp = pd.Series({
        s: (1.0 / pb_map[s] if (s in pb_map and pb_map[s] and pb_map[s] > 0)
            else np.nan)
        for s in stocks
    })

    # 7. ret_3m = -(close/close.shift(60)-1) [反转]
    ret_3m = -(close_df / close_df.shift(60) - 1.0).iloc[-1]

    # 8. ret_open_2d = mean(open/close.shift(1)-1, 2)
    ro = open_df / close_df.shift(1) - 1.0
    ret_open_2d = ro.iloc[-2:].mean()

    # 9. skewness_20 = -rolling(20).skew() of daily ret
    daily_ret = close_df.pct_change()
    sk = daily_ret.rolling(20, min_periods=10).skew()
    skewness_20 = -sk.iloc[-1]

    # 10. gap_up = open/close.shift(1)-1
    gap_up = (open_df / close_df.shift(1) - 1.0).iloc[-1]

    # 11. relative_spread = -(high-low)/(close+0.001)
    sp = -(high_df - low_df) / (close_df.replace(0, np.nan) + 0.001)
    relative_spread = sp.iloc[-1]

    return {
        'dollar_vol_20d':       dollar_vol_20d,
        'turnover_std_cv':      turnover_std_cv,
        'overnight_5d':         overnight_5d,
        'tvma_20':              tvma_20,
        'money_flow_20':        money_flow_20,
        'bp':                   bp,
        'ret_3m':               ret_3m,
        'ret_open_2d':          ret_open_2d,
        'skewness_20':          skewness_20,
        'gap_up':               gap_up,
        'relative_spread':      relative_spread,
    }


# ==================== 截面 rank % [0,1] ====================
def _rank_pct(series):
    s = series.dropna()
    if len(s) == 0:
        return series
    return s.rank(pct=True)


# ==================== 复合因子构建 ====================
def build_composites(base_df):
    """5 复合 (rank_pct 乘积), 无 base 因子."""

    r_ovn    = _rank_pct(base_df['overnight_5d'])
    r_tvma   = _rank_pct(base_df['tvma_20'])
    r_dv     = _rank_pct(base_df['dollar_vol_20d'])
    r_tstd   = _rank_pct(base_df['turnover_std_cv'])
    r_mf     = _rank_pct(base_df['money_flow_20'])
    r_bp     = _rank_pct(base_df['bp'])
    r_r3m    = _rank_pct(base_df['ret_3m'])
    r_ro2    = _rank_pct(base_df['ret_open_2d'])
    r_skew   = _rank_pct(base_df['skewness_20'])
    r_gap    = _rank_pct(base_df['gap_up'])
    r_spread = _rank_pct(base_df['relative_spread'])

    # value_momentum (嵌套复合: rank(rank(1/PB) × rank(-ret_3m)))
    r_vm = _rank_pct(r_bp * r_r3m)

    return {
        'comp1_overnight_tvma':      r_ovn    * r_tvma,
        'comp2_dollar_turnover':     r_dv     * r_tstd,
        'comp3_moneyflow_value':     r_mf     * r_vm,
        'comp4_retopen_skew':        r_ro2    * r_skew,
        'comp5_gapup_spread':        r_gap    * r_spread,
    }


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    if g.debug:
        log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    pb_map = load_pb_map(context, stocks)
    end_dt = context.previous_date

    factor_names = ['dollar_vol_20d', 'turnover_std_cv', 'overnight_5d',
                    'tvma_20', 'money_flow_20', 'bp', 'ret_3m',
                    'ret_open_2d', 'skewness_20', 'gap_up', 'relative_spread']
    base_data = {f: {} for f in factor_names}

    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        res = compute_base_factors_chunk(open_df, high_df, low_df, close_df,
                                          volume_df, chunk, pb_map)
        for f in factor_names:
            base_data[f].update(res[f].to_dict())

        del open_df, high_df, low_df, close_df, volume_df, res
        gc.collect()

    base_df = pd.DataFrame(base_data)
    comp = build_composites(base_df)

    # ICIR 加权合成 (5复合 × 归一化ICIR权重)
    total = (COMPOSITE_WEIGHTS['comp1_overnight_tvma']    * comp['comp1_overnight_tvma']
             + COMPOSITE_WEIGHTS['comp2_dollar_turnover']   * comp['comp2_dollar_turnover']
             + COMPOSITE_WEIGHTS['comp3_moneyflow_value']   * comp['comp3_moneyflow_value']
             + COMPOSITE_WEIGHTS['comp4_retopen_skew']      * comp['comp4_retopen_skew']
             + COMPOSITE_WEIGHTS['comp5_gapup_spread']      * comp['comp5_gapup_spread'])
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
