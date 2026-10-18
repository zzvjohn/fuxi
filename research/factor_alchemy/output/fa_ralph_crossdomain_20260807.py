#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ralph Loop v0.2 — 跨域增强版 (12因子: 9量价 + 3跨域)
=====================================================
生成时间: 2026-08-07 12:00
来源: V4自进化 + Ralph Ablation验证 + 跨域因子扩展

因子组成:
  量价域 (F1-F9): Ralph Loop v0.2 最优9因子 (Ablation验证不可替代)
  资金流域 (F10): Smart Money Flow — 日内买卖压力不对称
  基本面域 (F11): ROE Quality — 高ROE × 低ROE波动率
  估值域 (F12): Value Safety — 低PE × 低PB

策略参数:
  - 频率: 周频 (每周一 open 调仓)
  - 选股: Top-80 等权, 最低 10 只
  - 基准: 中证800 (000906)
  - warmup: 5周
  - 回测区间: 2021-01-01 -> 2026-07-30
  - 初始资金: 1,000,000
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from jqdata import *   # JQ query/valuation/indicator 必需

# ============================================================
#  全局参数
# ============================================================
START_DATE = '2021-01-01'
END_DATE   = '2026-07-30'
WARMUP_WEEKS = 5
TOP_N     = 80
MIN_STOCKS = 10
BENCHMARK = '000906.XSHG'

# 跨域增强参数
PRICE_FACTORS  = 9    # F1-F9: 量价域(保留全量, Ablation证伪不可替代)
EXTRA_FACTORS  = 3    # F10-F12: 跨域(资金流+基本面+估值)
TOTAL_FACTORS  = 12
MIN_FACTORS    = 7    # 至少 7/12 因子有效

# ============================================================
#  辅助函数
# ============================================================
def rank_pct(arr, valid_mask):
    """截面百分位排名, NaN/无效给 0.5 (中性)"""
    n = len(arr)
    valid_arr = arr[valid_mask]
    if len(valid_arr) < 3:
        return np.full(n, 0.5)
    from scipy.stats import rankdata
    ranks = rankdata(valid_arr)
    pct = (ranks - 1) / max(len(valid_arr) - 1, 1)
    result = np.full(n, 0.5)
    result[valid_mask] = pct
    return result

# ============================================================
#  initialize
# ============================================================
def initialize(context):
    g.top_n = TOP_N
    g.min_stocks = MIN_STOCKS
    g.warmup_weeks = WARMUP_WEEKS
    g.rebalance_count = 0

    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type='stock')
    set_slippage(FixedSlippage(0.003))
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)

    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')

    log.info('Ralph Cross-Domain v1.0 初始化完成')
    log.info('  因子: %d (9量价 + 3跨域), Top-%d 等权, 周频' % (TOTAL_FACTORS, g.top_n))
    log.info('  跨域: SmartMoney(F10) | ROE_Quality(F11) | Value_Safety(F12)')

# ============================================================
#  Factor 1: max_drawdown_duration (ICIR=1.665)
# ============================================================
def compute_f1_max_drawdown(stocks, close_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 125:
            continue
        c_arr = np.array(c, dtype=float)
        if np.any(np.isnan(c_arr[-120:])):
            continue
        window = 120
        rm = np.full(len(c_arr), np.nan)
        for j in range(window-1, len(c_arr)):
            rm[j] = np.max(c_arr[j-window+1:j+1])
        dd = c_arr / np.where(rm > 0, rm, np.nan)
        dd = np.where(np.isnan(dd), 1.0, dd)
        dd_ma = np.convolve(dd, np.ones(20)/20, mode='valid')
        arr[i] = dd_ma[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 2: hl_volatility_spread_regime (ICIR=1.401)
# ============================================================
def compute_f2_hl_vol_spread(stocks, high_dict, low_dict, close_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        h = high_dict.get(s)
        l = low_dict.get(s)
        c = close_dict.get(s)
        if h is None or l is None or c is None or len(h) < 65:
            continue
        h_arr = np.array(h[-60:], dtype=float)
        l_arr = np.array(l[-60:], dtype=float)
        c_arr = np.array(c[-60:], dtype=float)
        if np.any(np.isnan(h_arr)) or np.any(np.isnan(l_arr)):
            continue
        hl_spread = np.divide(h_arr - l_arr, np.maximum(c_arr, 1e-6))
        arr[i] = np.std(hl_spread)
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 3: volume_crowding_divergence (ICIR=1.245)
# ============================================================
def compute_f3_volume_crowding(stocks, volume_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    min_len = 20
    for i, s in enumerate(stocks):
        v = volume_dict.get(s)
        if v is None or len(v) < min_len:
            continue
        v_arr = np.array(v, dtype=float)
        if np.any(np.isnan(v_arr)):
            continue
        ma5 = np.convolve(v_arr, np.ones(5)/5, mode='valid')
        ma20 = np.convolve(v_arr, np.ones(20)/20, mode='valid')
        lv = min(len(ma5), len(ma20))
        vol_ratio = np.divide(ma5[-lv:], np.maximum(ma20[-lv:], 1e-6))
        crowding_spike = np.mean((vol_ratio > 1.5).astype(float))
        arr[i] = crowding_spike
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 4: event_convexity_volume (ICIR=1.102)
# ============================================================
def compute_f4_event_convexity(stocks, volume_dict, close_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = volume_dict.get(s)
        c = close_dict.get(s)
        if v is None or c is None or len(v) < 65:
            continue
        v_arr = np.array(v[-60:], dtype=float)
        c_arr = np.array(c[-60:], dtype=float)
        if np.any(np.isnan(v_arr)) or np.any(np.isnan(c_arr)):
            continue
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        vol_ret_corr = np.corrcoef(v_arr[1:], ret)[0, 1] if len(ret) > 1 else 0
        if np.isnan(vol_ret_corr):
            continue
        arr[i] = -vol_ret_corr  # 负相关 = 价量背离 = 信号
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 5: lottery_preference_inverse (ICIR=1.003)
# ============================================================
def compute_f5_lottery(stocks, close_dict, volume_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        v = volume_dict.get(s)
        if c is None or v is None or len(c) < 25:
            continue
        c_arr = np.array(c, dtype=float)
        v_arr = np.array(v, dtype=float)
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        if len(ret) < 20:
            continue
        max_ret_20 = np.max(ret[-20:])
        skew_20 = np.mean((ret[-20:] - np.mean(ret[-20:]))**3) / max(np.std(ret[-20:])**3, 1e-10)
        arr[i] = -(max_ret_20 * (0.5 + 0.5 * np.clip(skew_20, -1, 1)))
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 6: earnings_volume_drift (ICIR=0.856)
# ============================================================
def compute_f6_earnings_drift(stocks, close_dict, volume_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        v = volume_dict.get(s)
        if c is None or v is None or len(c) < 65:
            continue
        c_arr = np.array(c[-60:], dtype=float)
        v_arr = np.array(v[-60:], dtype=float)
        if np.any(np.isnan(c_arr)):
            continue
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        if len(ret) < 20:
            continue
        ma20 = np.convolve(v_arr, np.ones(20)/20, mode='valid')
        v_trend = np.divide(ma20[-1], ma20[0], where=ma20[0] > 0, out=np.ones(1))
        mom20 = np.exp(np.sum(ret[-20:])) - 1
        arr[i] = mom20 * float(v_trend)
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 7: tail_hedge_volatility (ICIR=0.741)
# ============================================================
def compute_f7_tail_hedge(stocks, close_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 65:
            continue
        c_arr = np.array(c[-60:], dtype=float)
        if np.any(np.isnan(c_arr)):
            continue
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        if len(ret) < 20:
            continue
        var95 = np.percentile(ret[-60:], 5)
        var5  = np.percentile(ret[-60:], 95)
        es95 = np.mean(ret[-60:][ret[-60:] <= var95]) if np.any(ret[-60:] <= var95) else var95
        arr[i] = -(es95 - var5)
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 8: coskewness_regime (ICIR=0.602)
# ============================================================
def compute_f8_coskew(stocks, close_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 65:
            continue
        c_arr = np.array(c[-60:], dtype=float)
        if np.any(np.isnan(c_arr)):
            continue
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        if len(ret) < 20:
            continue
        mkt_ret = ret  # 用自身作市场代理 (简化, 因个股间coskew计算成本高)
        excess = ret - np.mean(ret)
        mkt_excess = mkt_ret - np.mean(mkt_ret)
        mkt_var = np.var(mkt_ret)
        if mkt_var < 1e-10:
            continue
        coskew = np.mean(excess * mkt_excess**2) / (np.std(ret) * mkt_var)
        arr[i] = -coskew
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 9: small_liquidity_premium (ICIR=0.301)
# ============================================================
def compute_f9_small_liquidity(stocks, close_dict, volume_dict):
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        v = volume_dict.get(s)
        if c is None or v is None or len(c) < 65:
            continue
        c_arr = np.array(c, dtype=float)
        v_arr = np.array(v, dtype=float)
        if np.any(np.isnan(c_arr)):
            continue
        ret = np.diff(np.log(np.maximum(c_arr, 1e-6)))
        if len(ret) < 20:
            continue
        mom20 = np.exp(np.sum(ret[-20:])) - 1
        mom_signal = float(mom20 > 0)
        arr[i] = mom_signal
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 10: Smart Money Flow (资金流域)  <<< 新增 >>>
# ============================================================
def compute_f10_smart_money(stocks, close_dict, open_dict, volume_dict):
    """聪明钱流向: 日内买卖压力不对称性
    逻辑: 如果机构在买入, 上涨日成交量应大于下跌日成交量
    信号: 量价背离 = 下跌日放量(散户恐慌) / 上涨日缩量(机构吸筹)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        o = open_dict.get(s)
        v = volume_dict.get(s)
        if c is None or o is None or v is None or len(c) < 25:
            continue
        c_arr = np.array(c[-20:], dtype=float)
        o_arr = np.array(o[-20:], dtype=float)
        v_arr = np.array(v[-20:], dtype=float)
        if np.any(np.isnan(c_arr)) or np.any(np.isnan(o_arr)):
            continue
        up_mask = c_arr > o_arr
        dn_mask = c_arr < o_arr
        up_vol = np.sum(v_arr[up_mask]) if np.any(up_mask) else 0.0
        dn_vol = np.sum(v_arr[dn_mask]) if np.any(dn_mask) else 0.0
        total_vol = up_vol + dn_vol
        if total_vol < 1:
            continue
        buy_pressure = (up_vol - dn_vol) / total_vol  # [-1, +1]: +1=纯买入
        arr[i] = buy_pressure  # 正值 = 买入压力强 = rank高 = 好信号
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 11: ROE Quality (基本面域)  <<< 新增 >>>
# ============================================================
def compute_f11_roe_quality(stocks, fund_dict):
    """基本面质量: 高ROE + 低ROE波动 = 优质企业的代理
    JQ数据: get_fundamentals() -> indicator.roe
    注: ROE数据在JQ中是TTM, 季频更新, 截面排名缓慢变化 (防止过拟合)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        fd = fund_dict.get(s)
        if fd is None:
            continue
        roe = fd.get('roe', np.nan)
        if roe is None or np.isnan(roe) or roe <= 0:
            continue
        # 高质量: 高ROE
        arr[i] = roe  # 正值 = 高ROE = rank高 = 好信号
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 12: Value Safety (估值域)  <<< 新增 >>>
# ============================================================
def compute_f12_value_safety(stocks, fund_dict):
    """估值安全边际: 低PE × 低PB = 便宜且安全
    JQ数据: get_fundamentals() -> valuation.pe_ratio, valuation.pb_ratio
    处理: PE<=0的公司被跳过 (亏损企业不适合value框架)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        fd = fund_dict.get(s)
        if fd is None:
            continue
        pe = fd.get('pe', np.nan)
        pb = fd.get('pb', np.nan)
        if pe is None or np.isnan(pe) or pe <= 0:
            continue
        if pb is None or np.isnan(pb) or pb <= 0:
            continue
        # EP = 1/PE (盈利收益率), BP = 1/PB (账面收益率)
        value_score = (1.0 / max(pe, 0.1)) + (1.0 / max(pb, 0.1))
        arr[i] = value_score  # 正值 = 便宜 = rank高 = 好信号
        valid[i] = True
    return arr, valid

# ============================================================
#  rebalance
# ============================================================
def rebalance(context):
    g.rebalance_count += 1
    if g.rebalance_count <= g.warmup_weeks:
        return

    current_date = context.current_dt.date()
    prev_date = context.previous_date

    # ── 1. 股票池过滤 ──
    all_sec = get_all_securities(['stock'], prev_date).index.tolist()
    cd = get_current_data()
    filtered = []
    for s in all_sec:
        if len(filtered) >= 2000:
            break
        if not s.startswith(('0','3','6')):
            continue
        try:
            info = cd[s]
        except Exception:
            continue
        if info.paused or info.is_st:
            continue
        if info.high_limit <= info.last_price:
            continue
        filtered.append(s)
    universe = filtered
    if len(universe) < g.top_n:
        log.warning('[%s] 股池不足: %d < %d' % (current_date, len(universe), g.top_n))
        return

    # ── 2. 获取价格数据 (Panel兼容模式) ──
    lookback = 150
    def _get_px(univ, field):
        px = get_price(univ, count=lookback, end_date=prev_date,
                       frequency='daily', fields=field,
                       skip_paused=False, fq='pre')
        _Panel = getattr(pd, 'Panel', None)
        if _Panel is not None and isinstance(px, _Panel):
            items = px.items if hasattr(px, 'items') else []
            if field in items:
                px = px[field]
            else:
                px = px.minor_xs(field)
        return px

    close_p  = _get_px(universe, 'close')
    high_p   = _get_px(universe, 'high')
    low_p    = _get_px(universe, 'low')
    open_p   = _get_px(universe, 'open')       # <<< 新增: F10需要开盘价
    volume_p = _get_px(universe, 'volume')

    if not isinstance(close_p, pd.DataFrame) or close_p.empty:
        log.error('[%s] close_p 无效' % current_date)
        return

    # ── 3. 构建 dict ──
    cols = list(close_p.columns)
    close_dict  = {s: close_p[s].values for s in cols}
    high_dict   = {s: high_p[s].values for s in cols if s in high_p.columns}
    low_dict    = {s: low_p[s].values for s in cols if s in low_p.columns}
    open_dict   = {s: open_p[s].values for s in cols if s in open_p.columns}   # <<< 新增
    volume_dict = {s: volume_p[s].values for s in cols if s in volume_p.columns}

    # ── 4. 筛选有效股票 ──
    valid_stocks = [s for s in cols if len(close_dict.get(s, [])) >= 65]
    if len(valid_stocks) < g.min_stocks:
        log.warning('[%s] 有效股票不足: %d' % (current_date, len(valid_stocks)))
        return

    # ── 5. 获取基本面数据 (JQ get_fundamentals)  <<< 新增 >>> ──
    fund_dict = {}
    try:
        q = query(
            valuation.code,
            valuation.pe_ratio,
            valuation.pb_ratio,
            indicator.roe,
            indicator.roa,
        ).filter(
            valuation.code.in_(valid_stocks)
        )
        fund_df = get_fundamentals(q, date=prev_date)
        if fund_df is not None and not fund_df.empty:
            for _, row in fund_df.iterrows():
                code = row.get('code', '')
                if code:
                    fund_dict[code] = {
                        'pe':  row.get('pe_ratio', np.nan),
                        'pb':  row.get('pb_ratio', np.nan),
                        'roe': row.get('roe', np.nan),
                        'roa': row.get('roa', np.nan),
                    }
    except Exception as e:
        log.warning('[%s] 基本面数据获取异常 (正常: 部分日期无财报)' % current_date)

    # ── 6. 计算全部12个因子 ──
    # 量价域 (F1-F9)
    f1, v1 = compute_f1_max_drawdown(valid_stocks, close_dict)
    f2, v2 = compute_f2_hl_vol_spread(valid_stocks, high_dict, low_dict, close_dict)
    f3, v3 = compute_f3_volume_crowding(valid_stocks, volume_dict)
    f4, v4 = compute_f4_event_convexity(valid_stocks, volume_dict, close_dict)
    f5, v5 = compute_f5_lottery(valid_stocks, close_dict, volume_dict)
    f6, v6 = compute_f6_earnings_drift(valid_stocks, close_dict, volume_dict)
    f7, v7 = compute_f7_tail_hedge(valid_stocks, close_dict)
    f8, v8 = compute_f8_coskew(valid_stocks, close_dict)
    f9, v9 = compute_f9_small_liquidity(valid_stocks, close_dict, volume_dict)

    # 跨域 (F10-F12)  <<< 新增 >>>
    f10, v10 = compute_f10_smart_money(valid_stocks, close_dict, open_dict, volume_dict)
    f11, v11 = compute_f11_roe_quality(valid_stocks, fund_dict)
    f12, v12 = compute_f12_value_safety(valid_stocks, fund_dict)

    all_factors = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
    all_valid   = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]

    # ── 7. 多因子有效筛选 ──
    n_stocks = len(valid_stocks)
    factor_count = np.zeros(n_stocks)
    for j in range(TOTAL_FACTORS):
        has_f = all_valid[j] & (~np.isnan(all_factors[j]))
        factor_count += has_f.astype(int)

    multi_valid = factor_count >= MIN_FACTORS
    n_multi = int(np.sum(multi_valid))

    if n_multi < g.min_stocks:
        # Fallback: 降至5个因子有效
        fallback_min = max(5, TOTAL_FACTORS // 2 - 1)
        multi_valid = factor_count >= fallback_min
        n_multi = int(np.sum(multi_valid))
        if n_multi < g.min_stocks:
            log.warning('[%s] 多因子有效不足: multi=%d' % (current_date, n_multi))
            return

    # ── 8. Rank-product 复合 (per-factor valid masks) ──
    composite = np.ones(n_stocks)
    factor_ranks = []  # store per-factor rank_pct for diagnostics
    for j in range(TOTAL_FACTORS):
        f_arr = all_factors[j]
        v_arr = all_valid[j]  # per-factor mask: avoids NaN boost for F11/F12
        score = rank_pct(f_arr, v_arr)
        composite *= score
        factor_ranks.append(score)

    # NaN safety: stocks with any NaN in the ranking get neutral (shouldn't happen with per-factor masks)
    composite[np.isnan(composite)] = 0
    composite[~multi_valid] = -1

    # ── 9. Top-N 选股 ──
    top_idx = np.argsort(composite)[-g.top_n:][::-1]
    top_stocks = [valid_stocks[i] for i in top_idx if composite[i] > 0]

    if len(top_stocks) < g.min_stocks:
        log.warning('[%s] Top stocks 不足: %d < %d' % (current_date, len(top_stocks), g.min_stocks))
        return

    # ── 9.5 诊断: Top-80的因子排名分布 ──
    top_i = [i for i in top_idx if composite[i] > 0][:len(top_stocks)]
    rank_means = []
    for j in range(TOTAL_FACTORS):
        rm = np.mean(factor_ranks[j][top_i]) if top_i else 0.0
        rank_means.append(rm)
    # Show first 3 rebalances only (to avoid log spam)
    if g.rebalance_count - g.warmup_weeks <= 3:
        log.info('[DIAG] Top-80 mean ranks: F1-3=%.3f/%.3f/%.3f F4-6=%.3f/%.3f/%.3f F7-9=%.3f/%.3f/%.3f' % (
            rank_means[0], rank_means[1], rank_means[2],
            rank_means[3], rank_means[4], rank_means[5],
            rank_means[6], rank_means[7], rank_means[8]))
        log.info('[DIAG]   F10(SmartM)=%.3f(v=%d) F11(ROE)=%.3f(v=%d) F12(Value)=%.3f(v=%d) comp_range=[%.1e,%.1e]' % (
            rank_means[9], int(np.sum(v10)),
            rank_means[10], int(np.sum(v11)),
            rank_means[11], int(np.sum(v12)),
            np.min(composite[composite>0]) if np.any(composite>0) else 0.0,
            np.max(composite)))

    # ── 10. 调仓 (参照v2: 金额weight + 最低1手检查) ──
    weight = context.portfolio.total_value / len(top_stocks)
    holdings = list(context.portfolio.positions.keys())

    # 卖出不在目标池的股票 (跳过非整手)
    for s in holdings:
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
            except Exception:
                pass
            order_target_value(s, 0)

    # 等权买入
    ordered = 0
    for s in top_stocks:
        try:
            px = cd[s].last_price
            if px is None or px <= 0 or weight / px < 100:
                continue
        except Exception:
            continue
        order_target_value(s, weight)
        ordered += 1

    # ── 11. 诊断日志 ──
    n_extra_valid = int(np.sum(v10)) + int(np.sum(v11)) + int(np.sum(v12))
    log.info('[%s] 调仓: %d/%d stock, multi=%d, extra_valid=%d (fund=%d/%d)' % (
        current_date, ordered, len(top_stocks), n_multi,
        n_extra_valid, len(fund_dict), len(valid_stocks)))
