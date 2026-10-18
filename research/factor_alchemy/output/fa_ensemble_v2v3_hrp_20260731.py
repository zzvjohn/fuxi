# -*- coding: utf-8 -*-
"""
================================================================
FA Ensemble v2v3 — V2+V3 配对复合池 + IVP 持仓权重
================================================================
2026-07-31 | V2(+146%) + V3(+127%) 系统化 Ensemble | HRP→IVP简化(聚宽无scipy)

核心创新:
  1. V2+V3 全部 6 个配对复合取等权 Ensemble (论文共识:等权>优化)
  2. IVP (Inverse Volatility Portfolio) 替代等权持仓 (MaxDD治理)

6个复合 (V2+V3 并集):
  comp1 = overnight_5d       × tvma_20              流动性微观结构
  comp2 = dollar_vol_20d     × turnover_std_cv       成交额×波动
  comp3 = money_flow_20      × value_momentum        资金流×估值动量 (嵌套复合)
  comp4 = ret_open_2d        × skewness_20           开盘动量×尾部偏度
  comp5 = earnings_volume_drift                      缩量上涨=筹码锁定 (V2)
  comp6 = gap_up             × relative_spread       跳空×振幅 (V3)

★ value_momentum 嵌套: rank(rank(1/PB) × rank(-ret_3m))
★ Ensemble: 6个 composite rank → 平均 → 选 top-30
★ IVP 权重: w_i ∝ 1/std(52周收益), 替代等权

参数: 持仓=30, 周频(周五14:50), 基准=沪深300
成本: 佣金万1+印花千1+滑点0.3%
================================================================
"""

from __future__ import division
import datetime
import gc
import numpy as np
import pandas as pd


# ==================== 策略参数 ====================
STOCK_NUM      = 30
MAX_TURNOVER   = 0.60
MCAP_MIN       = 5
EXCLUDE_PREFIX = ('8', '4')
CHUNK          = 500
LOOKBACK       = 65
IVP_LOOKBACK   = 52   # IVP 所需周数
ENABLE_IVP     = True  # 开关: True=IVP逆波动率加权 / False=等权30持仓


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0001,
        close_commission=0.0001,
        min_commission=5,
    ), type='stock')
    set_slippage(FixedSlippage(0.003))

    run_weekly(main, weekday=5, time='14:50')

    g.stock_num     = STOCK_NUM
    g.max_turnover  = MAX_TURNOVER
    g.mcap_min      = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX

    # IVP 状态: 存储每周 close 用于计算波动率权重
    g.weekly_close  = {}   # stock -> list [last N weekly closes]
    g.last_week     = None


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


# ==================== 逐股票标量因子 ====================
def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df,
                               stocks, pb_map):
    """12 base 因子 → 6 composite 的原子输入."""

    dollar_vol_raw = close_df * volume_df

    # 1. dollar_vol_20d = -log(mean(close*volume, 20))
    avg_dv = dollar_vol_raw.rolling(20, min_periods=10).mean()
    dollar_vol_20d = -np.log(avg_dv.iloc[-1].replace(0, np.nan))

    # 2. turnover_std_cv = -CV(volume, 20)
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
        s: (1.0 / pb_map[s] if (s in pb_map and pb_map[s] and pb_map[s] > 0) else np.nan)
        for s in stocks
    })

    # 7. ret_3m = -(close/close.shift(60)-1)
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

    # ★ 12. earnings_volume_drift (V2)
    #     signal = clip(ret_5d,±0.3)×clip(-vol_chg,±1/2), ret_5d>0 only
    last_close = close_df.iloc[-1]
    close_6d_ago = close_df.iloc[-6] if len(close_df) >= 6 else close_df.iloc[0]
    ret_5d = last_close / close_6d_ago.replace(0, np.nan) - 1.0
    vol_recent = volume_df.iloc[-5:]
    vol_prior = volume_df.iloc[-11:-6] if len(volume_df) >= 11 else volume_df.iloc[-10:-5]
    m5  = vol_recent.mean() if len(vol_recent) > 0 else pd.Series(np.nan, index=volume_df.columns)
    m5p = vol_prior.mean() if len(vol_prior) > 0 else pd.Series(np.nan, index=volume_df.columns)
    vol_chg = (m5 / m5p.replace(0, np.nan) - 1.0).fillna(0)
    signal = ret_5d.clip(-0.3, 0.3) * (-vol_chg).clip(-1, 2)
    earnings_volume_drift = signal.where(ret_5d > 0, 0)

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
        'earnings_volume_drift': earnings_volume_drift,
    }


# ==================== 截面 rank % [0,1] ====================
def _rank_pct(series):
    s = series.dropna()
    if len(s) == 0:
        return series
    return s.rank(pct=True)


# ==================== 6 个配对复合 ====================
def build_composites(base_df):
    """
    6 个复合 (V2+V3 并集), 每个 = rank_pct(f1) × rank_pct(f2).
    earnings_volume_drift 本身已是 rank-pct 标量, 直接纳入 Ensemble.
    """

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
    r_evd    = _rank_pct(base_df['earnings_volume_drift'])

    # 嵌套复合: value_momentum = rank(rank(1/PB) × rank(-ret_3m))
    r_vm = _rank_pct(r_bp * r_r3m)

    return {
        'comp1_overnight_tvma':      r_ovn    * r_tvma,
        'comp2_dollar_turnover':     r_dv     * r_tstd,
        'comp3_moneyflow_value':     r_mf     * r_vm,
        'comp4_retopen_skew':        r_ro2    * r_skew,
        'comp5_earnings_vol_drift':  r_evd,                       # V2 base
        'comp6_gapup_spread':        r_gap    * r_spread,         # V3 only
    }


# ==================== Ensemble: 等权平均 rank ====================
def ensemble_score(composites):
    """
    6 个 composite → 各自 rank_pct → 平均 → 最终得分.
    等权 Ensemble — 论文共识: 等权 > 优化.
    """
    ranks = {}
    for name, score in composites.items():
        r = _rank_pct(score)
        if r.notna().sum() > 0:
            ranks[name] = r

    if len(ranks) == 0:
        return pd.Series(dtype=float)

    # 等权平均
    ensemble = pd.DataFrame(ranks).mean(axis=1)
    return ensemble.dropna().sort_values(ascending=False)


# ==================== IVP 权重 (JQ 内联实现, 不依赖 scipy) ====================
def compute_ivp_weights(returns_2d):
    """
    Inverse Volatility Portfolio: w_i ∝ 1/σ_i
    returns_2d: shape (T, N) 周收益率矩阵
    """
    n = returns_2d.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    stds = np.nanstd(returns_2d, axis=0)
    stds = np.maximum(stds, 1e-9)  # 防除零
    inv_vol = 1.0 / stds
    w = inv_vol / inv_vol.sum()
    return w


# ==================== 周 close 历史更新 ====================
def update_weekly_history(context, stocks, close_df):
    """
    记录每周的 close 价格, 保留最近 IVP_LOOKBACK 周.
    close_df: DataFrame(index=stock, value=close_price) — 最新日收盘价快照
    """
    current_week = context.current_dt.isocalendar()[1]
    current_year = context.current_dt.year
    week_key = (current_year, current_week)

    if g.last_week == week_key:
        return  # 同一周不重复记录

    g.last_week = week_key

    # 记录本周 close
    for s in stocks:
        if s not in close_df or pd.isna(close_df[s]):
            continue
        if s not in g.weekly_close:
            g.weekly_close[s] = []
        g.weekly_close[s].append(close_df[s])
        # 只保留最近 IVP_LOOKBACK 周
        if len(g.weekly_close[s]) > IVP_LOOKBACK:
            g.weekly_close[s] = g.weekly_close[s][-IVP_LOOKBACK:]


def get_ivp_weights_for_stocks(target_stocks):
    """
    对 target_stocks 从 g.weekly_close 提取周收益 → IVP 权重.
    数据不足则 fallback 等权.
    """
    n = len(target_stocks)
    if n <= 1:
        return np.ones(n) / max(n, 1)

    # 提取周 close → 周收益
    prices = {}
    for s in target_stocks:
        if s in g.weekly_close and len(g.weekly_close[s]) >= 10:
            prices[s] = np.array(g.weekly_close[s])

    if len(prices) < 2:
        return np.ones(n) / n

    # 对齐长度: 取共同最小长度
    min_len = min(len(p) for p in prices.values())
    if min_len < 5:
        return np.ones(n) / n

    # 构建收益矩阵 (T, N_stocks_with_data)
    aligned_stocks = list(prices.keys())
    rets = np.zeros((min_len - 1, len(aligned_stocks)))
    for j, s in enumerate(aligned_stocks):
        p = prices[s][-min_len:]
        rets[:, j] = p[1:] / np.maximum(p[:-1], 1e-9) - 1.0

    # IVP
    ivp_w = compute_ivp_weights(rets)

    # 映射回 target_stocks 的顺序
    weights = np.ones(n) / n
    stock_to_idx = {s: i for i, s in enumerate(target_stocks)}
    for j, s in enumerate(aligned_stocks):
        weights[stock_to_idx[s]] = ivp_w[j]

    return weights / weights.sum()


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    pb_map = load_pb_map(context, stocks)
    end_dt = context.previous_date

    # 加载价格数据 (分块, V2 模式)
    factor_names = [
        'dollar_vol_20d', 'turnover_std_cv', 'overnight_5d',
        'tvma_20', 'money_flow_20', 'bp', 'ret_3m',
        'ret_open_2d', 'skewness_20', 'gap_up',
        'relative_spread', 'earnings_volume_drift',
    ]
    base_data = {f: {} for f in factor_names}

    # 额外保存 close 快照用于 IVP 历史更新
    close_snapshot = {}

    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        # 记录本周 close 快照 (最后一行)
        if isinstance(close_df, pd.DataFrame) and not close_df.empty:
            last_close = close_df.iloc[-1]
            for s in chunk:
                if s in last_close.index:
                    close_snapshot[s] = float(last_close[s])

        res = compute_base_factors_chunk(open_df, high_df, low_df, close_df,
                                          volume_df, chunk, pb_map)
        for f in factor_names:
            base_data[f].update(res[f].to_dict())

        del open_df, high_df, low_df, close_df, volume_df, res
        gc.collect()

    # 更新 IVP 周度历史
    update_weekly_history(context, stocks, close_snapshot)

    # 构建 6 个配对复合
    base_df = pd.DataFrame(base_data)
    composites = build_composites(base_df)

    # Ensemble: 等权平均 rank
    total = ensemble_score(composites)
    if len(total) == 0:
        log.warn('[%s] Ensemble empty — SKIP' % date_str)
        return

    target = total.head(g.stock_num).index.tolist()
    log.info('[%s] top3: %s scores=%.3f/%.3f/%.3f' % (
        date_str,
        ', '.join(s[:8] for s in target[:3]),
        total.iloc[0], total.iloc[1], total.iloc[2]
    ))

    # IVP 权重 (可关闭)
    if ENABLE_IVP:
        w = get_ivp_weights_for_stocks(target)
    else:
        w = np.ones(g.stock_num) / g.stock_num
    rebalance(context, target, w)


# ==================== 调仓 (IVP 权重) ====================
def rebalance(context, target, weights):
    """
    target: top-30 股票
    weights: IVP 权重 (等长, Σ=1)
    """
    target_set = set(target)
    positions = context.portfolio.positions
    current = set(positions.keys())

    total_value = context.portfolio.total_value

    # 卖出不在 target 中的
    for s in current - target_set:
        order_target_value(s, 0)

    # 买入 / 调整 target
    cd = get_current_data()
    for i, s in enumerate(target):
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        order_target_value(s, total_value * weights[i])

    # 每 4 周日志
    if context.current_dt.isocalendar()[1] % 4 == 0:
        held = len([s for s in target if s in current])
        w_str = ', '.join('%.1f%%' % (w * 100) for w in weights[:5])
        log.info('Week %d | Hold %d | IVP top5: %s | Cash %.0f | Total %.0f' % (
            context.current_dt.isocalendar()[1], held, w_str,
            context.portfolio.cash, total_value))
