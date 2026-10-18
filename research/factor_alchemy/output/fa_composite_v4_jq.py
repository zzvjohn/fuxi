# -*- coding: utf-8 -*-
"""
================================================================
FA Composite v4 — V2因子 + 行业/市值中性化 + HRP权重  [JQ 真回测]
================================================================
2026-07-28 生成 | 基于 V2(+146.13%/SR0.68/MaxDD-33.14%) → 加双层降回撤

4 rank-percentile 乘积复合因子 + 1 base 因子 (同 V2):
  comp1 = overnight_5d(+)        × tvma_20(+)                ICIR +0.772  权重 20.2%
  comp2 = dollar_vol_20d(+)      × turnover_std_cv(+)         ICIR +0.747  权重 20.2%
  comp3 = money_flow_20(+)       × value_momentum(+)          ICIR +0.695  权重 20.0%
  comp4 = ret_open_2d(+)         × skewness_20(+)             ICIR +0.684  权重 19.8%
  base  = earnings_volume_drift(+)                            ICIR +1.068  权重 19.8%

v4 新增:
  A. 预处理层: 行业(SW一级)/市值中性化 — 对10个base因子逐期横截面OLS取残差
  B. 持仓层: HRP(层次风险平价)权重 — Top-30选股后用纯numpy HRP替代等权

★ value_momentum 本身是嵌套复合: rank_pct(1/PB) × rank_pct(-ret_3m)
  earnings_volume_drift = clip(ret_5d,±0.3)×clip(-vol_chg,±2), ret_5d>0 only

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
HRP_LOOKBACK   = 52   # HRP 协方差回看周数

# ICIR 权重 (v2: 4复合+1base, 来自 composite_v2_th03 训练)
ICIR = {
    'comp1_overnight_tvma':       0.7719,
    'comp2_dollar_turnover':      0.7472,
    'comp3_moneyflow_value':      0.6946,
    'comp4_retopen_skew':         0.6843,
    'earnings_volume_drift':      1.0679,
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


# ==================== 基本面 + 行业加载 ====================
def load_fundamentals_batch(context, stocks):
    """
    一枪头查询 PB + 市值, 减少 JQ API 调用次数.
    Returns: (pb_map, mcap_map)
    """
    q = query(valuation.code, valuation.pb_ratio, valuation.market_cap).filter(
        valuation.code.in_(stocks))
    val = get_fundamentals(q, date=context.previous_date)
    if val.empty:
        return {}, {}
    pb_map = dict(zip(val['code'], val['pb_ratio']))
    mcap_map = dict(zip(val['code'], val['market_cap']))
    return pb_map, mcap_map


def load_industry_map(stocks):
    """
    获取申万一级行业分类.
    返回 dict: stock → industry_label
    """
    ind_map = {}
    for s in stocks:
        try:
            ind = get_industry(s, 'sw_l1')
            # get_industry 可能返回 None, str, 或 ShenwanIndustry 对象
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


# ==================== 逐股票标量因子 (分块计算, 同 V2) ====================
def compute_base_factors_chunk(open_df, high_df, low_df, close_df, volume_df,
                                stocks, pb_map):
    """11 base factor 计算 (与 V2 完全一致)."""

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
        s: (1.0 / pb_map[s] if (s in pb_map and pb_map[s] and pb_map[s] > 0)
            else np.nan)
        for s in stocks
    })

    # 7. ret_3m = -(close/close.shift(60)-1)
    ret_3m = -(close_df / close_df.shift(60) - 1.0).iloc[-1]

    # 8. ret_open_2d = mean(open/close.shift(1)-1, 2)
    ro = open_df / close_df.shift(1) - 1.0
    ret_open_2d = ro.iloc[-2:].mean()

    # 9. skewness_20 = -rolling(20).skew()
    daily_ret = close_df.pct_change()
    sk = daily_ret.rolling(20, min_periods=10).skew()
    skewness_20 = -sk.iloc[-1]

    # 10. earnings_volume_drift
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
        'earnings_volume_drift': earnings_volume_drift,
    }


# ==================== ★ NEW ★ 行业/市值中性化 ====================
def neutralize_factor_cs(factor_vals, valid_mask, industries, log_mcap):
    """
    单期横截面中性化: factor ~ industry_dummies + log(mcap) → residuals.
    纯 numpy 实现, JQ 兼容.

    Parameters
    ----------
    factor_vals : (N,) float
    valid_mask  : (N,) bool
    industries  : list of str, len=N
    log_mcap    : (N,) float (NaN where invalid)

    Returns
    -------
    residuals : (N,) float (NaN 保持 NaN)
    """
    n = len(factor_vals)
    residuals = factor_vals.copy()
    valid = np.where(valid_mask)[0]

    if len(valid) < 30:
        return residuals

    y = factor_vals[valid]

    # Industry dummies
    inds = [industries[i] for i in valid]
    uinds = sorted(set(inds))
    if len(uinds) > 1:
        X_ind = np.column_stack([
            (np.array(inds) == u).astype(float) for u in uinds[1:]
        ])
    else:
        X_ind = np.empty((len(valid), 0))

    # Design matrix: [const, industry_dummies, log_mcap]
    X = np.column_stack([
        np.ones((len(valid), 1)),
        X_ind,
        log_mcap[valid].reshape(-1, 1),
    ])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        residuals[valid] = y - y_hat
    except Exception:
        pass

    return residuals


def neutralize_all_factors(base_data, industry_map, mcap_map):
    """
    对所有 base factor 做行业/市值中性化.
    base_data: {factor_name: {stock: value}}
    返回: 中性化后的 base_data (same structure)
    """
    stocks = list(base_data[list(base_data.keys())[0]].keys())
    n = len(stocks)

    # 行业 + log(mcap) 数组
    industries = [industry_map.get(s, 'Unknown') for s in stocks]
    log_mcap = np.array([
        np.log(mcap_map[s]) if (s in mcap_map and mcap_map.get(s, 0) > 0) else np.nan
        for s in stocks
    ])

    neutralized = {}
    for fn, fdict in base_data.items():
        f_vals = np.array([fdict.get(s, np.nan) for s in stocks])
        valid_mask = pd.notna(f_vals) & pd.notna(log_mcap) & (log_mcap > 0) & np.isfinite(f_vals)
        if valid_mask.sum() < 30:
            neutralized[fn] = fdict  # 不够样本, 回退原始
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


# ==================== 复合因子 + base 构建 (同 V2) ====================
def build_composites(base_df):
    """4 复合 + 1 base (earnings_volume_drift 直接 rank_pct)."""

    r_ovn  = _rank_pct(base_df['overnight_5d'])
    r_tvma = _rank_pct(base_df['tvma_20'])
    r_dv   = _rank_pct(base_df['dollar_vol_20d'])
    r_tstd = _rank_pct(base_df['turnover_std_cv'])
    r_mf   = _rank_pct(base_df['money_flow_20'])
    r_bp   = _rank_pct(base_df['bp'])
    r_r3m  = _rank_pct(base_df['ret_3m'])
    r_ro2  = _rank_pct(base_df['ret_open_2d'])
    r_skew = _rank_pct(base_df['skewness_20'])

    # value_momentum (嵌套复合)
    r_vm = _rank_pct(r_bp * r_r3m)

    # earnings_volume_drift 直接 rank_pct (不是复合)
    r_evd = _rank_pct(base_df['earnings_volume_drift'])

    return {
        'comp1_overnight_tvma':      r_ovn  * r_tvma,
        'comp2_dollar_turnover':     r_dv   * r_tstd,
        'comp3_moneyflow_value':     r_mf   * r_vm,
        'comp4_retopen_skew':        r_ro2  * r_skew,
        'earnings_volume_drift':     r_evd,
    }


# ==================== ★ NEW ★ HRP 权重 (纯 numpy, 无 scipy) ====================
def hrp_weights_numpy(returns):
    """
    层次风险平价 (HRP) — 纯 numpy 实现, JQ 兼容 (无 scipy 依赖).

    Algorithm (López de Prado 2016):
      1. Covariance → Correlation → Distance
      2. Agglomerative clustering (average linkage)
      3. Quasi-diagonalization (leaf ordering)
      4. Recursive bisection risk allocation

    Parameters
    ----------
    returns : (T, N) array  资产收益率 (T=周数, N=资产数)

    Returns
    -------
    weights : (N,)  归一化正权重
    """
    n = returns.shape[1]
    if n <= 1:
        return np.ones(n) / max(n, 1)
    if n == 2:
        stds = np.std(returns, axis=0) + 1e-9
        w = 1.0 / (stds ** 2)
        return w / w.sum()

    # Step 1: Covariance → Correlation → Distance
    cov = np.cov(returns, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))

    # Step 2: Agglomerative clustering (average linkage, O(n³))
    cluster_members = {i: {i} for i in range(n)}
    tree = []  # [(left, right, distance)]

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

    # Step 3: Quasi-diagonalization (leaf ordering from tree)
    def _get_order(node_id):
        if node_id < n:
            return [node_id]
        left, right, _ = tree[node_id - n]
        return _get_order(left) + _get_order(right)

    root_id = n + (n - 2)
    sorted_idx = _get_order(root_id)

    if len(sorted_idx) != n:
        sorted_idx = list(range(n))

    # Step 4: Recursive bisection risk allocation
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

    # Floor constraint: min weight = 0.3 * equal_weight
    floor = 0.3 / n
    if floor > 0:
        w = np.maximum(w, floor)
        w = w / w.sum()

    return w


def compute_hrp_weights_from_prices(close_df, target_stocks):
    """
    从日线价格 DataFrame 计算目标股票的 HRP 权重.

    Parameters
    ----------
    close_df : DataFrame(index=date, columns=stock)
        日线收盘价 (含 LOOKBACK+HRP_LOOKBACK 天数据)
    target_stocks : list  当前持仓目标

    Returns
    -------
    hrp_w : (N,) array  归一化权重
    """
    n = len(target_stocks)
    if n <= 1:
        return np.ones(n) / max(n, 1)

    # 提取目标股票的价格
    available = [s for s in target_stocks if s in close_df.columns]
    if len(available) < 2:
        return np.ones(n) / n

    price_pool = close_df[available].tail(HRP_LOOKBACK).dropna(axis=1, how='all')
    if price_pool.shape[1] < 2:
        return np.ones(n) / n

    # Ffill minor gaps (JQ 停牌日价格为 NaN)
    price_pool = price_pool.fillna(method='ffill').fillna(method='bfill')

    # 周收益率 (取每周最后一个有效价格)
    weekly = price_pool.resample('W').last().dropna(how='all')
    if len(weekly) < 8:
        return np.ones(n) / n

    rets = weekly.pct_change().dropna(how='all')
    if len(rets) < 4:
        return np.ones(n) / n

    # 仅保留数据覆盖率 ≥ 50% 的股票
    rets = rets.dropna(axis=1, thresh=int(len(rets) * 0.5))
    if rets.shape[1] < 2:
        return np.ones(n) / n

    ret_arr = rets.values
    w_arr = hrp_weights_numpy(ret_arr)

    # Map back to original stock order
    hrp_map = dict(zip(rets.columns, w_arr))
    w = np.array([hrp_map.get(s, 1.0 / n) for s in target_stocks])
    return w / w.sum()


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    if g.debug:
        log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    # 一枪头加载 PB + 市值
    pb_map, mcap_map = load_fundamentals_batch(context, stocks)
    industry_map = load_industry_map(stocks)

    end_dt = context.previous_date

    factor_names = ['dollar_vol_20d', 'turnover_std_cv', 'overnight_5d',
                    'tvma_20', 'money_flow_20', 'bp', 'ret_3m',
                    'ret_open_2d', 'skewness_20', 'earnings_volume_drift']
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

    # ★ NEW: 行业/市值中性化 ★
    base_data = neutralize_all_factors(base_data, industry_map, mcap_map)

    # Build composites + ICIR-weighted scoring
    base_df = pd.DataFrame(base_data)
    comp = build_composites(base_df)

    total = (COMPOSITE_WEIGHTS['comp1_overnight_tvma']    * comp['comp1_overnight_tvma']
             + COMPOSITE_WEIGHTS['comp2_dollar_turnover']   * comp['comp2_dollar_turnover']
             + COMPOSITE_WEIGHTS['comp3_moneyflow_value']   * comp['comp3_moneyflow_value']
             + COMPOSITE_WEIGHTS['comp4_retopen_skew']      * comp['comp4_retopen_skew']
             + COMPOSITE_WEIGHTS['earnings_volume_drift']   * comp['earnings_volume_drift'])
    total = total.dropna().sort_values(ascending=False)
    if len(total) == 0:
        return

    target = total.head(g.stock_num).index.tolist()
    if g.debug:
        log.info('[%s] top3 scores: %s' % (date_str,
                 ', '.join('%.4f' % v for v in total.head(3).values)))

    # ★ NEW: HRP 权重 (需要全池 close 数据用于协方差估计) ★
    all_close = _get_price_field(stocks, end_dt, LOOKBACK + HRP_LOOKBACK, 'close')
    hrp_w = compute_hrp_weights_from_prices(all_close, target)

    rebalance_hrp(context, target, hrp_w)


# ==================== ★ NEW ★ HRP 调仓 ====================
def rebalance_hrp(context, target, hrp_w):
    """按 HRP 权重调仓 (替代等权)."""
    target_set = set(target)
    positions = context.portfolio.positions
    current = set(positions.keys())

    total_value = context.portfolio.total_value

    # Turnover check (用等权假设)
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

    # 平仓: 不在目标中的股票
    for s in current - target_set:
        order_target_value(s, 0)

    # 开仓/调仓: 按 HRP 权重
    cd = get_current_data()
    for i, s in enumerate(target):
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        # HRP 权重 × 总资产 = 目标持仓金额
        target_value = total_value * hrp_w[i]
        order_target_value(s, target_value)
