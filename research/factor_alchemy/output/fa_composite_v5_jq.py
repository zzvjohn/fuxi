# -*- coding: utf-8 -*-
"""
================================================================
FA Composite v5 — 跨周期 regime 感知 10复合 + 中性化 + HRP  [JQ 真回测]
================================================================
2026-07-29 生成 | 本地 regime_lasso.py --selector composite --icir 0.30 --tag v5 结果落地

★ V5 相对 V4 的核心差异:
  V4 = 固定 5复合(V2因子) 单-regime.
  V5 = regime 感知: 不同市场状态用不同因子组合 (small_bull vs large_dominant),
        中性化在预处理层 (in-pipeline), 选股层用 HRP.

★ regime 划分 (与本地 RegimeDetector 对齐的简化版):
  small_bull   (CSI500-CSI300 近21日收益差 > -2%)  -> 使用 stage_0 的 5 复合 (占 ~47% 周)
  large_dominant(否则)                              -> 使用 stage_2 的 5 复合 (占 ~42% 周)

★ stage_0 (5 复合, 权重已 ICIR 归一化):
  c1 money_flow_20        × opening_gap_momentum     w=0.206
  c2 overnight_5d         × volume_ratio            w=0.202
  c3 dollar_vol_20d      × tvma_20                w=0.200
  c4 gap_up              × relative_spread_proxy    w=0.198
  c5 money_flow_20        × volatility_of_volatility w=0.194

★ stage_2 (5 复合, 权重已 ICIR 归一化):
  c1 dollar_vol_20d      × relative_spread_proxy    w=0.205
  c2 price_1m            × vol_3m                 w=0.202
  c3 bull_power          × turnover_to_vol         w=0.200
  c4 relative_spread_proxy× ret_6m                 w=0.197
  c5 avg_turnover_1m    × volume_ratio            w=0.197

★ 复合结构 = rank_pct(A) × rank_pct(B) (与本地完全一致, 已证最简单有效)
★ 中性化 = 行业(SW一级)/市值 横截面 OLS 残差 (预处理层)
★ 持仓层 = HRP 层次风险平价 (替代等权, 降集中度)

⚠️ JQ 限制: 聚宽禁碰 turnover API, 故 2 个换手率因子用成交量代理:
   turnover_to_vol   -> mean(volume,20) / std(ret,20)  (原: 换手率/波动)
   avg_turnover_1m -> -mean(volume,20)            (原: -月均换手率)
   其余 13 个因子均为 OHLCV/收益, JQ 原生可算, 方向与本地一致.

参数: 持仓=30, 周频(周五14:50), 基准=沪深300
成本: 佣金万1 + 印花千1 + 滑点0.3%
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
LOOKBACK       = 130      # >= ret_6m 的 shift(120) + 缓冲
HRP_LOOKBACK   = 52       # HRP 协方差回看周数

# ==================== regime 复合定义 (ICIR 已归一化) ====================
STAGE0_COMP = {
    'c1': ('money_flow_20',         'opening_gap_momentum',     0.206),
    'c2': ('overnight_5d',          'volume_ratio',            0.202),
    'c3': ('dollar_vol_20d',        'tvma_20',                0.200),
    'c4': ('gap_up',                'relative_spread_proxy',    0.198),
    'c5': ('money_flow_20',         'volatility_of_volatility', 0.194),
}
STAGE2_COMP = {
    'c1': ('dollar_vol_20d',        'relative_spread_proxy',   0.205),
    'c2': ('price_1m',              'vol_3m',                 0.202),
    'c3': ('bull_power',            'turnover_to_vol',         0.200),
    'c4': ('relative_spread_proxy',  'ret_6m',                0.197),
    'c5': ('avg_turnover_1m',      'volume_ratio',            0.197),
}

# 15 个基础因子名 (供 DataFrame 列对齐)
BASE_FACTORS = [
    'money_flow_20', 'opening_gap_momentum', 'overnight_5d', 'volume_ratio',
    'dollar_vol_20d', 'tvma_20', 'gap_up', 'relative_spread_proxy',
    'volatility_of_volatility', 'price_1m', 'vol_3m', 'bull_power',
    'turnover_to_vol', 'ret_6m', 'avg_turnover_1m',
]

DEBUG = True


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


# ==================== 市值 + 行业加载 ====================
def load_mcap_map(stocks, ctx):
    q = query(valuation.code, valuation.market_cap).filter(
        valuation.code.in_(stocks))
    val = get_fundamentals(q, date=ctx.previous_date)
    if val.empty:
        return {}
    return dict(zip(val['code'], val['market_cap']))


def load_industry_map(stocks):
    ind_map = {}
    for s in stocks:
        try:
            ind = get_industry(s, 'sw_l1')
            if ind is not None:
                if hasattr(ind, 'index_name'):
                    ind_map[s] = ind.index_name or 'Unknown'
                elif hasattr(ind, 'name'):
                    ind_map[s] = ind.name or 'Unknown'
                else:
                    ind_map[s] = str(ind)
            else:
                ind_map[s] = 'Unknown'
        except Exception:
            ind_map[s] = 'Unknown'
    return ind_map


# ==================== 逐股票 15 基础因子 (分块计算) ====================
def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df, stocks):
    """返回 {factor: Series(stock->value)}; 每个因子取窗口末值 (LOOKBACK 天)."""
    # 1. money_flow_20 = -(TP*vol / MA(TP*vol,20) - 1)
    tp = (high_df + low_df + close_df) / 3.0
    mf = tp * volume_df
    mf_ma = mf.rolling(20, min_periods=10).mean()
    mfnorm = mf / mf_ma.replace(0, np.nan) - 1.0
    money_flow_20 = -mfnorm.clip(-0.9, 5).iloc[-1]

    # 2. opening_gap_momentum = mean(open/close.shift(1)-1, 3d)
    gap = open_df / close_df.shift(1) - 1.0
    opening_gap_momentum = gap.rolling(3, min_periods=3).mean().iloc[-1]

    # 3. overnight_5d = sum(open/close.shift(1)-1, 5d)
    overnight_5d = gap.rolling(5, min_periods=3).sum().iloc[-1]

    # 4. volume_ratio = -(volume / MA(volume,5))  [用 volume, 非换手率]
    vma5 = volume_df.rolling(5, min_periods=5).mean()
    volume_ratio = -(volume_df / vma5.replace(0, np.nan)).iloc[-1]

    # 5. dollar_vol_20d = -log(mean(close*volume, 20))
    dv = close_df * volume_df
    dv_ma = dv.rolling(20, min_periods=10).mean()
    dollar_vol_20d = -np.log(dv_ma.replace(0, np.nan)).iloc[-1]

    # 6. tvma_20 = -(money / MA(money,20))
    money = close_df * volume_df
    money_ma = money.rolling(20, min_periods=20).mean()
    tvma_20 = -(money / money_ma.replace(0, np.nan)).clip(0, 10).iloc[-1]

    # 7. gap_up = open/close.shift(1) - 1
    gap_up = (open_df / close_df.shift(1) - 1.0).iloc[-1]

    # 8. relative_spread_proxy = -(high-low)/(close+0.001)  [纯 OHLC, JQ 可算]
    relative_spread_proxy = -((high_df - low_df) / (close_df + 0.001)).iloc[-1]

    # 9. volatility_of_volatility = -std(ret,5).rolling(20).std()
    ret = close_df.pct_change()
    v5 = ret.rolling(5, min_periods=5).std()
    volatility_of_volatility = -v5.rolling(20, min_periods=20).std().iloc[-1]

    # 10. price_1m = -(close / MA(close,21) - 1)
    ma21 = close_df.rolling(21, min_periods=21).mean()
    price_1m = -(close_df / ma21.replace(0, np.nan) - 1.0).iloc[-1]

    # 11. vol_3m = -std(daily_ret, 60)
    vol_3m = -ret.rolling(60, min_periods=30).std().iloc[-1]

    # 12. bull_power = -((high - EWMA(close,13)) / close)
    ema13 = close_df.ewm(span=13, min_periods=13).mean()
    bull_power = -((high_df - ema13) / close_df).iloc[-1]

    # 13. turnover_to_vol (★JQ代理: 原换手率/波动 -> 成交量/波动)
    v_avg20 = volume_df.rolling(20, min_periods=10).mean()
    ret_std20 = ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    turnover_to_vol = -(v_avg20 / ret_std20).iloc[-1]

    # 14. ret_6m = -(close / close.shift(120) - 1)
    ret_6m = -(close_df / close_df.shift(120).replace(0, np.nan) - 1.0).iloc[-1]

    # 15. avg_turnover_1m (★JQ代理: 原 -月均换手 -> -月均成交量)
    avg_turnover_1m = -v_avg20.iloc[-1]

    return {
        'money_flow_20':         money_flow_20,
        'opening_gap_momentum':  opening_gap_momentum,
        'overnight_5d':          overnight_5d,
        'volume_ratio':           volume_ratio,
        'dollar_vol_20d':        dollar_vol_20d,
        'tvma_20':               tvma_20,
        'gap_up':                gap_up,
        'relative_spread_proxy':  relative_spread_proxy,
        'volatility_of_volatility': volatility_of_volatility,
        'price_1m':             price_1m,
        'vol_3m':               vol_3m,
        'bull_power':            bull_power,
        'turnover_to_vol':       turnover_to_vol,
        'ret_6m':               ret_6m,
        'avg_turnover_1m':       avg_turnover_1m,
    }


# ==================== ★ 行业/市值中性化 (复用 V4, 纯 numpy) ====================
def neutralize_factor_cs(factor_vals, valid_mask, industries, log_mcap):
    n = len(factor_vals)
    residuals = factor_vals.copy()
    valid = np.where(valid_mask)[0]
    if len(valid) < 30:
        return residuals
    y = factor_vals[valid]
    inds = [industries[i] for i in valid]
    uinds = sorted(set(inds))
    if len(uinds) > 1:
        X_ind = np.column_stack([(np.array(inds) == u).astype(float) for u in uinds[1:]])
    else:
        X_ind = np.empty((len(valid), 0))
    X = np.column_stack([np.ones((len(valid), 1)), X_ind, log_mcap[valid].reshape(-1, 1)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        residuals[valid] = y - y_hat
    except Exception:
        pass
    return residuals


def neutralize_all_factors(base_data, industry_map, mcap_map):
    stocks = list(base_data[BASE_FACTORS[0]].keys())
    n = len(stocks)
    industries = [industry_map.get(s, 'Unknown') for s in stocks]
    log_mcap = np.array([
        np.log(mcap_map[s]) if (s in mcap_map and mcap_map.get(s, 0) > 0) else np.nan
        for s in stocks
    ])
    neutralized = {}
    for fn in BASE_FACTORS:
        fdict = base_data[fn]
        f_vals = np.array([fdict.get(s, np.nan) for s in stocks])
        valid_mask = pd.notna(f_vals) & pd.notna(log_mcap) & (log_mcap > 0) & np.isfinite(f_vals)
        if valid_mask.sum() < 30:
            neutralized[fn] = fdict
            continue
        res = neutralize_factor_cs(f_vals, valid_mask, industries, log_mcap)
        neutralized[fn] = {s: (float(res[i]) if pd.notna(res[i]) else np.nan)
                           for i, s in enumerate(stocks)}
    return neutralized


# ==================== 截面 rank % [0,1] ====================
def _rank_pct(series):
    s = series.dropna()
    if len(s) == 0:
        return series
    return s.rank(pct=True)


# ==================== ★ HRP 权重 (纯 numpy, 无 scipy, 复用 V4) ====================
def hrp_weights_numpy(returns):
    n = returns.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    if n == 2:
        stds = np.std(returns, axis=0) + 1e-9
        w = 1.0 / (stds ** 2)
        return w / w.sum()
    cov = np.cov(returns, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    cluster_members = {i: {i} for i in range(n)}
    tree = []
    for merge_step in range(n - 1):
        active = list(cluster_members.keys())
        min_dist = np.inf
        min_pair = (active[0], active[1])
        for i_idx, ci in enumerate(active):
            for j_idx, cj in enumerate(active):
                if i_idx >= j_idx:
                    continue
                members_i = cluster_members[ci]
                members_j = cluster_members[cj]
                d = 0.0
                for a in members_i:
                    for b in members_j:
                        d += dist[a, b]
                d /= len(members_i) * len(members_j)
                if d < min_dist:
                    min_dist = d
                    min_pair = (ci, cj)
        ci, cj = min_pair
        new_id = n + merge_step
        cluster_members[new_id] = cluster_members[ci] | cluster_members[cj]
        del cluster_members[ci]
        del cluster_members[cj]
        tree.append((ci, cj, min_dist))

    def _get_order(node_id):
        if node_id < n:
            return [node_id]
        left, right, _ = tree[node_id - n]
        return _get_order(left) + _get_order(right)

    root_id = n + (n - 2)
    sorted_idx = _get_order(root_id)
    if len(sorted_idx) != n:
        sorted_idx = list(range(n))
    w = np.ones(n)

    def _bisect(indices):
        if len(indices) <= 1:
            return
        best_var = np.inf
        best_k = len(indices) // 2
        for k in range(1, len(indices)):
            left = indices[:k]
            right = indices[k:]
            cov_l = cov[np.ix_(left, left)]
            cov_r = cov[np.ix_(right, right)]
            wl = np.ones(len(left)) / max(len(left), 1)
            wr = np.ones(len(right)) / max(len(right), 1)
            var_l = max(wl @ cov_l @ wl, 1e-12)
            var_r = max(wr @ cov_r @ wr, 1e-12)
            total_var = var_l + var_r
            if total_var < best_var:
                best_var = total_var
                best_k = k
        left = indices[:best_k]
        right = indices[best_k:]
        cov_l = cov[np.ix_(left, left)]
        cov_r = cov[np.ix_(right, right)]
        wl = np.ones(len(left)) / max(len(left), 1)
        wr = np.ones(len(right)) / max(len(right), 1)
        var_l = max(wl @ cov_l @ wl, 1e-12)
        var_r = max(wr @ cov_r @ wr, 1e-12)
        alloc_l = (1.0 / var_l) / (1.0 / var_l + 1.0 / var_r)
        alloc_r = 1.0 - alloc_l
        w[left] *= alloc_l * len(indices) / max(len(left), 1)
        w[right] *= alloc_r * len(indices) / max(len(right), 1)
        _bisect(left)
        _bisect(right)

    _bisect(sorted_idx)
    w = w / w.sum()
    floor = 0.3 / n
    if floor > 0:
        w = np.maximum(w, floor)
        w = w / w.sum()
    return w


def compute_hrp_weights_from_prices(close_df, target_stocks):
    n = len(target_stocks)
    if n <= 1:
        return np.ones(n) / max(n, 1)
    available = [s for s in target_stocks if s in close_df.columns]
    if len(available) < 2:
        return np.ones(n) / n
    price_pool = close_df[available].tail(HRP_LOOKBACK).dropna(axis=1, how='all')
    if price_pool.shape[1] < 2:
        return np.ones(n) / n
    price_pool = price_pool.fillna(method='ffill').fillna(method='bfill')
    weekly = price_pool.resample('W').last().dropna(how='all')
    if len(weekly) < 8:
        return np.ones(n) / n
    rets = weekly.pct_change().dropna(how='all')
    if len(rets) < 4:
        return np.ones(n) / n
    rets = rets.dropna(axis=1, thresh=int(len(rets) * 0.5))
    if rets.shape[1] < 2:
        return np.ones(n) / n
    ret_arr = rets.values
    w_arr = hrp_weights_numpy(ret_arr)
    hrp_map = dict(zip(rets.columns, w_arr))
    w = np.array([hrp_map.get(s, 1.0 / n) for s in target_stocks])
    return w / w.sum()


# ==================== regime 检测 (简化版, 对齐 RegimeDetector) ====================
def detect_regime(context):
    """small_bull: CSI500 近21日收益 - CSI300 近21日收益 > -2% -> stage_0
       否则 large_dominant -> stage_2."""
    try:
        csi500 = get_price('000905.XSHG', end_date=context.previous_date,
                          count=30, frequency='daily', fields='close',
                          skip_paused=True)['close']
        csi300 = get_price('000300.XSHG', end_date=context.previous_date,
                          count=30, frequency='daily', fields='close',
                          skip_paused=True)['close']
        r500 = csi500[-1] / csi500[-21] - 1.0 if len(csi500) >= 21 else 0.0
        r300 = csi300[-1] / csi300[-21] - 1.0 if len(csi300) >= 21 else 0.0
        spread = r500 - r300
        return 'small_bull' if spread > -0.02 else 'large_dominant'
    except Exception:
        return 'small_bull'  # 默认 stage_0


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    if g.debug:
        log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    mcap_map = load_mcap_map(stocks, context)
    industry_map = load_industry_map(stocks)
    end_dt = context.previous_date

    base_data = {f: {} for f in BASE_FACTORS}
    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        res = compute_base_factors_chunk(open_df, high_df, low_df, close_df,
                                       volume_df, chunk)
        for f in BASE_FACTORS:
            base_data[f].update(res[f].to_dict())

        del open_df, high_df, low_df, close_df, volume_df, res
        gc.collect()

    # ★ 行业/市值中性化 (预处理层) ★
    base_data = neutralize_all_factors(base_data, industry_map, mcap_map)

    # 截面 rank % [0,1]
    base_df = pd.DataFrame(base_data)
    for f in BASE_FACTORS:
        if f in base_df.columns:
            base_df[f] = _rank_pct(base_df[f])

    # ★ regime 感知选择复合集 ★
    regime = detect_regime(context)
    comp_defs = STAGE0_COMP if regime == 'small_bull' else STAGE2_COMP

    score = pd.Series(0.0, index=base_df.index)
    for cname, (fa, fb, w) in comp_defs.items():
        if fa in base_df.columns and fb in base_df.columns:
            score = score + w * base_df[fa] * base_df[fb]

    score = score.dropna().sort_values(ascending=False)
    if len(score) == 0:
        return

    target = score.head(g.stock_num).index.tolist()
    if g.debug:
        log.info('[%s] regime=%s | top3: %s' % (
            date_str, regime,
            ', '.join('%.4f' % v for v in score.head(3).values)))

    # ★ HRP 权重 ★
    all_close = _get_price_field(stocks, end_dt, LOOKBACK + HRP_LOOKBACK, 'close')
    hrp_w = compute_hrp_weights_from_prices(all_close, target)

    rebalance_hrp(context, target, hrp_w)


# ==================== HRP 调仓 ====================
def rebalance_hrp(context, target, hrp_w):
    target_set = set(target)
    positions = context.portfolio.positions
    current = set(positions.keys())
    total_value = context.portfolio.total_value

    eq_unit = total_value / g.stock_num
    turnover = 0.0
    for s in current - target_set:
        turnover += abs(positions[s].value)
    for i, s in enumerate(target):
        cur = positions[s].value if s in positions else 0.0
        turnover += abs(eq_unit - cur)
    turnover = turnover / total_value if total_value > 0 else 0.0

    if turnover > g.max_turnover and g.debug:
        log.info('[rebalance] turnover=%.2f > cap=%.2f' % (turnover, g.max_turnover))

    for s in current - target_set:
        order_target_value(s, 0)

    cd = get_current_data()
    for i, s in enumerate(target):
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        target_value = total_value * hrp_w[i]
        order_target_value(s, target_value)
