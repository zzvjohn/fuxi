#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ralph Loop v0.2 — Stage 3 JQ 验证策略 (全量 V4 诊断批次) [v2 修复版]
=================================================================
生成时间: 2026-08-06 00:32  |  修复时间: 2026-08-06 22:00
来源: V4 自进化架构全量扫描 (223因子 → 9个通过四阶段验证)
架构: 9因子等权 rank-product 复合, 周频, Top-80

修复内容 (v2):
  1. fields='close' (字符串) 替代 fields=['close'] (列表) — 根因修复
  2. Panel兼容处理 (参照王者策略)
  3. 因子计算移入 rebalance, 去掉 after_close (消除静默崩溃)
  4. order_target_value + 最低1手检查 (消除 order_amount=0)
  5. log.set_level('order','error') 抑制订单噪音

因子列表 (按ICIR降序):
  1. max_drawdown_duration              IC=+0.1246  ICIR=+1.665
  2. hl_volatility_spread_regime_stable  IC=+0.0457  ICIR=+0.541
  3. volume_crowding_divergence          IC=+0.0305  ICIR=+0.423
  4. event_driven_convexity_fade         IC=+0.0279  ICIR=+0.376
  5. lottery_demand_suppression_score    IC=+0.0722  ICIR=+0.369
  6. earnings_pre_drift_alignment        IC=+0.0355  ICIR=+0.350
  7. idiosyncratic_tail_hedge_premium    IC=+0.0314  ICIR=+0.343
  8. harvey_siddique_coskew              IC=+0.0396  ICIR=+0.317
  9. small_cap_liquidity_quality         IC=+0.0299  ICIR=+0.301

策略参数:
  - 频率: 周频 (每周一 open 调仓, 参照王者)
  - 选股: Top-80 等权, 最低 10 只
  - 基准: 中证800 (000906)
  - warmup: 5周
  - 回测区间: 2021-01-01 → 2026-07-30
  - 初始资金: 1,000,000
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
#  全局参数
# ============================================================
START_DATE = '2021-01-01'
END_DATE   = '2026-07-30'
WARMUP_WEEKS = 5
TOP_N = 70
MIN_STOCKS = 10
BENCHMARK = '000906.XSHG'

# ============================================================
#  辅助函数
# ============================================================
def rank_pct(arr, valid):
    """截面百分位排名, NaN被忽略, 无效给0.5"""
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

# safe_get_price 已废弃 — 改用 panel=False 单次多字段调用 (4x性能提升)

# ============================================================
#  initialize
# ============================================================
def initialize(context):
    g.top_n = TOP_N
    g.min_stocks = MIN_STOCKS
    g.warmup_weeks = WARMUP_WEEKS
    g.rebalance_count = 0  # 调仓计数器 (参照王者 trade_days)

    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    # 佣金: 参照王者
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type='stock')
    set_slippage(FixedSlippage(0.003))

    # 抑制订单噪音日志
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)

    # 周一开盘调仓 (参照王者)
    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')

    log.info('Ralph Ablation C 初始化完成')
    log.info('  因子: 5(C因子子集), Top-%d 等权, 周频' % g.top_n)
    log.info('  优化: get_current_data单次 | 2000只break | lookback=150')

# ============================================================
#  Factor 1: max_drawdown_duration (ICIR=1.665)
# ============================================================
def compute_f1_max_drawdown(stocks, close_dict):
    """韧性信号: 从长期回撤中修复的程度"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 145:  # 120日回撤+20日平滑=需要140+数据点
            continue
        c_arr = np.array(c, dtype=float)
        window = 120
        rm = np.full(len(c_arr), np.nan)
        for j in range(window-1, len(c_arr)):
            rm[j] = np.max(c_arr[j-window+1:j+1])
        dd = c_arr / np.where(rm > 0, rm, np.nan)
        # 20日均值 (保持NaN语义, 与原版rolling一致)
        dd_ma = np.full(len(dd), np.nan)
        for j in range(19, len(dd)):
            win = dd[j-19:j+1]
            if np.sum(~np.isnan(win)) >= 15:
                dd_ma[j] = np.nanmean(win)
        if not np.isnan(dd_ma[-1]):
            arr[i] = dd_ma[-1]
            valid[i] = True
    return arr, valid

# ============================================================
#  Factor 2: hl_volatility_spread (ICIR=0.541)
# ============================================================
def compute_f2_hl_vol_spread(stocks, high_dict, low_dict, close_dict):
    """Parkinson波动率 - 收盘波动率价差"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        h = high_dict.get(s)
        l = low_dict.get(s)
        c = close_dict.get(s)
        if h is None or l is None or c is None:
            continue
        h_arr = np.array(h, dtype=float)
        l_arr = np.array(l, dtype=float)
        c_arr = np.array(c, dtype=float)
        if len(h_arr) < 25:
            continue
        # Parkinson vol
        hl_ratio = np.log(np.maximum(h_arr, 1e-6) / np.maximum(l_arr, 1e-6))
        pk = np.sqrt(1.0 / (4.0 * np.log(2)) * hl_ratio**2)
        pk_ma20 = np.convolve(pk, np.ones(20)/20, mode='valid')
        # Close vol
        rets = np.diff(c_arr) / c_arr[:-1]
        close_vol_20 = np.array([np.std(rets[max(0,j-19):j+1]) for j in range(len(rets))])
        # Align lengths
        min_len = min(len(pk_ma20), len(close_vol_20))
        if min_len < 5:
            continue
        pk_aligned = pk_ma20[-min_len:]
        cv_aligned = close_vol_20[-min_len:]
        spread = (pk_aligned - cv_aligned) / (pk_aligned + cv_aligned + 1e-6)
        # 5-day smooth
        if len(spread) >= 5:
            arr[i] = np.mean(spread[-5:])
        else:
            arr[i] = spread[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 3: volume_crowding_divergence (ICIR=0.423)
# ============================================================
def compute_f3_volume_crowding(stocks, volume_dict):
    """成交量拥挤度背离 — 做空拥挤暴露"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid_mask = np.zeros(n, dtype=bool)
    # 先收集所有股票的 vol_z
    vol_z_dict = {}
    for i, s in enumerate(stocks):
        v = volume_dict.get(s)
        if v is None or len(v) < 65:
            continue
        v_arr = np.array(v, dtype=float)
        ma20 = np.convolve(v_arr, np.ones(20)/20, mode='valid')
        ma60 = np.convolve(v_arr, np.ones(60)/60, mode='valid')
        # align
        l = min(len(ma20), len(ma60))
        ma20_a = ma20[-l:]
        ma60_a = ma60[-l:]
        vol_z = (ma20_a - ma60_a) / np.maximum(ma60_a, 1e-6)
        vol_z_dict[s] = vol_z
        valid_mask[i] = True

    valid_stocks = [stocks[i] for i in range(n) if valid_mask[i]]
    if len(valid_stocks) < 3:
        return arr, valid_mask

    # 截面均值 (cross-sectional mean of vol_z for each time step)
    min_zl = min(len(vol_z_dict[s]) for s in valid_stocks)
    cross_mean = np.zeros(min_zl)
    for s in valid_stocks:
        cross_mean += vol_z_dict[s][-min_zl:]
    cross_mean /= len(valid_stocks)

    # 计算背离
    for i, s in enumerate(stocks):
        if not valid_mask[i]:
            continue
        vz = vol_z_dict[s][-min_zl:]
        deviation = vz - cross_mean
        deviation = np.clip(deviation, -2, 2)
        v_arr = np.array(volume_dict[s], dtype=float)
        # vol_ratio: 5日均量 / 20日均量, 需对齐长度
        ma5  = np.convolve(v_arr, np.ones(5)/5,  mode='valid')
        ma20 = np.convolve(v_arr, np.ones(20)/20, mode='valid')
        lv = min(len(ma5), len(ma20))
        vol_ratio = ma5[-lv:] / np.maximum(ma20[-lv:], 1e-6)
        vol_ratio = vol_ratio[-min_zl:]
        crowding_spike = (vol_ratio > 1.5).astype(float)
        if len(crowding_spike) >= 5:
            crowding_spike = np.convolve(crowding_spike, np.ones(5)/5, mode='valid')
            deviation = deviation[-len(crowding_spike):]
        signal = -np.abs(deviation) * crowding_spike
        if len(signal) >= 5:
            arr[i] = np.mean(signal[-5:])
        elif len(signal) > 0:
            arr[i] = signal[-1]
    return arr, valid_mask

# ============================================================
#  Factor 4: event_driven_convexity_fade (ICIR=0.376)
# ============================================================
def compute_f4_event_convexity(stocks, volume_dict, close_dict):
    """极端成交后的拥挤瓦解"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = volume_dict.get(s)
        c = close_dict.get(s)
        if v is None or c is None or len(v) < 25:
            continue
        v_arr = np.array(v, dtype=float)
        c_arr = np.array(c, dtype=float)
        # vol_ma20, vol_std20
        window = 20
        ma20 = np.convolve(v_arr, np.ones(window)/window, mode='valid')
        if len(ma20) < 5:
            continue
        # rolling std
        vol_std20 = np.array([np.std(v_arr[max(0,j-window+1):j+1]) for j in range(window-1, len(v_arr))])
        # Align
        l = min(len(ma20), len(vol_std20))
        ma20_a = ma20[-l:]
        std20_a = vol_std20[-l:]
        extreme = (v_arr[-l:] > ma20_a + 2.0 * std20_a).astype(float)
        # extreme recent (max of last 3)
        extreme_recent = np.zeros(l)
        for j in range(l):
            extreme_recent[j] = np.max(extreme[max(0,j-2):j+1])
        # mom_5
        rets = np.diff(c_arr) / c_arr[:-1]
        mom5 = np.zeros(l)
        for j in range(l):
            idx = len(c_arr) - l + j
            if idx >= 5:
                mom5[j] = (c_arr[idx] / c_arr[idx-5] - 1)
            else:
                mom5[j] = 0
        mom5 = np.clip(mom5, -0.2, 0.2)
        crowd_chase = (mom5 > 0.03).astype(float)
        bubble = extreme_recent * crowd_chase
        if len(bubble) >= 5:
            bubble_ma = np.convolve(bubble, np.ones(5)/5, mode='valid')
            arr[i] = -np.mean(bubble_ma[-3:]) if len(bubble_ma) >= 3 else -bubble_ma[-1]
        elif len(bubble) > 0:
            arr[i] = -bubble[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 5: lottery_suppression (ICIR=0.369)
# ============================================================
def compute_f5_lottery(stocks, close_dict, volume_dict):
    """MAX效应: 做空彩票股"""
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
        rets = np.diff(c_arr) / c_arr[:-1]
        if len(rets) < 60:
            continue
        # max_ret_1m (20 day rolling max)
        max_ret_20 = np.array([np.max(rets[max(0,j-19):j+1]) for j in range(len(rets))])
        # lottery = 1 - max_ret (higher max_ret = lottery = lower score)
        lottery = 1.0 - max_ret_20
        lottery = np.clip(lottery, 0, 2)
        # vol_regime
        vol_20 = np.array([np.std(rets[max(0,j-19):j+1]) for j in range(len(rets))])
        vol_60 = np.array([np.std(rets[max(0,j-59):j+1]) for j in range(len(rets))])
        vol_regime = np.divide(vol_20, vol_60, where=vol_60>0, out=np.ones_like(vol_20))
        vol_regime = np.clip(vol_regime, 0.5, 2.0)
        vol_gate = 1.0 - np.abs(vol_regime - 1.0)
        vol_gate = np.clip(vol_gate, 0, 1)
        # vol_trend
        ma20_v = np.convolve(v_arr, np.ones(20)/20, mode='valid')
        ma60_v = np.convolve(v_arr, np.ones(60)/60, mode='valid')
        lv = min(len(ma20_v), len(ma60_v))
        vol_trend = np.divide(ma20_v[-lv:], ma60_v[-lv:], where=ma60_v[-lv:]>0, out=np.ones(lv))
        vol_trend = np.clip(vol_trend, 0.7, 1.5)
        # Align all signals
        min_l = min(len(lottery), len(vol_gate), lv)
        signal = lottery[-min_l:] * vol_gate[-min_l:] * vol_trend[-min_l:]
        # 10-day smooth
        if len(signal) >= 10:
            arr[i] = np.mean(signal[-10:])
        elif len(signal) > 0:
            arr[i] = signal[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 6: earnings_pre_drift (ICIR=0.350)
# ============================================================
def compute_f6_earnings_drift(stocks, close_dict, volume_dict):
    """超额动量×量能一致性"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    # 先收集5日收益
    ret5_dict = {}
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 10:
            continue
        c_arr = np.array(c, dtype=float)
        ret5 = c_arr[5:] / c_arr[:-5] - 1.0
        ret5_dict[s] = ret5
        valid[i] = True

    valid_stocks = [stocks[i] for i in range(n) if valid[i]]
    if len(valid_stocks) < 3:
        return arr, valid

    # 截面均值作为市场收益
    min_ret_l = min(len(ret5_dict[s]) for s in valid_stocks)
    mkt_ret = np.zeros(min_ret_l)
    for s in valid_stocks:
        mkt_ret += ret5_dict[s][-min_ret_l:]
    mkt_ret /= len(valid_stocks)

    for i, s in enumerate(stocks):
        if not valid[i]:
            continue
        r5 = ret5_dict[s][-min_ret_l:]
        excess = r5 - mkt_ret
        excess = np.clip(excess, -0.2, 0.2)
        # vol_compression
        v = volume_dict.get(s)
        if v is not None:
            v_arr = np.array(v, dtype=float)
            ma5 = np.convolve(v_arr, np.ones(5)/5, mode='valid')
            ma20 = np.convolve(v_arr, np.ones(20)/20, mode='valid')
            lv = min(len(ma5), len(ma20))
            vr = np.divide(ma5[-lv:], ma20[-lv:], where=ma20[-lv:]>0, out=np.ones(lv))
            vr = np.clip(vr, 0.5, 1.5)
            vc = 1.0 - vr
            vc = np.clip(vc, -0.5, 0.5)
            # Align
            la = min(len(excess), len(vc))
            alignment = excess[-la:] * vc[-la:]
            if len(alignment) >= 3:
                arr[i] = np.mean(alignment[-3:])
            elif len(alignment) > 0:
                arr[i] = alignment[-1]
        else:
            arr[i] = excess[-1]
    return arr, valid

# ============================================================
#  Factor 7: tail_hedge_premium (ICIR=0.343)
# ============================================================
def compute_f7_tail_hedge(stocks, close_dict):
    """特质低波动+残差负偏度"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    # 收集所有收益序列
    rets_dict = {}
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 65:
            continue
        c_arr = np.array(c, dtype=float)
        rets = np.diff(c_arr) / c_arr[:-1]
        rets_dict[s] = rets
        valid[i] = True

    valid_stocks = [stocks[i] for i in range(n) if valid[i]]
    if len(valid_stocks) < 5:
        return arr, valid

    # 市场收益 (等权)
    min_l = min(len(rets_dict[s]) for s in valid_stocks)
    mkt = np.zeros(min_l)
    for s in valid_stocks:
        mkt += rets_dict[s][-min_l:]
    mkt /= len(valid_stocks)

    lookback = 60
    for i, s in enumerate(stocks):
        if not valid[i]:
            continue
        r = rets_dict[s][-min_l:]
        if len(r) < lookback:
            continue
        # Rolling beta
        idio_vols = np.zeros(min_l)
        idio_skews = np.zeros(min_l)
        for j in range(lookback-1, min_l):
            ri = r[j-lookback+1:j+1]
            rm = mkt[j-lookback+1:j+1]
            # beta via OLS
            cov_rm = np.cov(ri, rm)[0, 1] if np.var(rm) > 1e-10 else 0
            var_rm = np.var(rm) if np.var(rm) > 1e-10 else 1e-6
            beta = cov_rm / var_rm
            residual = ri - beta * rm
            idio_vols[j] = np.std(residual)
            # skew (simplified: third moment)
            r_std = np.std(residual)
            if r_std > 1e-10:
                idio_skews[j] = np.mean((residual - np.mean(residual))**3) / (r_std**3)

        # low vol + neg skew → hedge premium
        vol_rank = np.zeros(min_l)
        skew_neg = (idio_skews < 0).astype(float)
        # Simplified vol rank: just compare to median
        med_vol = np.median(idio_vols[-20:]) if len(idio_vols) >= 20 else np.median(idio_vols)
        vol_rank = (idio_vols < med_vol).astype(float)
        hedge = vol_rank * skew_neg
        if len(hedge) >= 10:
            arr[i] = np.mean(hedge[-10:])
        elif len(hedge) > 0:
            arr[i] = np.mean(hedge[-5:]) if len(hedge) >= 5 else hedge[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 8: harvey_siddique_coskew (ICIR=0.317)
# ============================================================
def compute_f8_coskew(stocks, close_dict):
    """Harvey-Siddique协偏度"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    # 收集收益
    rets_dict = {}
    for i, s in enumerate(stocks):
        c = close_dict.get(s)
        if c is None or len(c) < 65:
            continue
        c_arr = np.array(c, dtype=float)
        rets = np.diff(c_arr) / c_arr[:-1]
        rets_dict[s] = rets
        valid[i] = True

    valid_stocks = [stocks[i] for i in range(n) if valid[i]]
    if len(valid_stocks) < 5:
        return arr, valid

    min_l = min(len(rets_dict[s]) for s in valid_stocks)
    mkt = np.zeros(min_l)
    for s in valid_stocks:
        mkt += rets_dict[s][-min_l:]
    mkt /= len(valid_stocks)
    mkt_sq = mkt ** 2

    lookback = 60
    for i, s in enumerate(stocks):
        if not valid[i]:
            continue
        r = rets_dict[s][-min_l:]
        if len(r) < lookback:
            continue
        # Rolling coskew
        coskews = np.zeros(min_l)
        for j in range(lookback-1, min_l):
            ri = r[j-lookback+1:j+1]
            rm_sq = mkt_sq[j-lookback+1:j+1]
            num = np.mean(ri * rm_sq)
            denom = np.std(ri) * np.mean(rm_sq) if np.mean(rm_sq) > 1e-10 else 1e-6
            coskews[j] = -num / max(denom, 1e-10)

        if len(coskews) >= 5:
            arr[i] = np.mean(coskews[-5:])
        elif len(coskews) > 0:
            arr[i] = coskews[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  Factor 9: small_cap_liquidity (ICIR=0.301)
# ============================================================
def compute_f9_small_liquidity(stocks, close_dict, volume_dict):
    """小盘Amihud流动性改善×动量"""
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
        dv = c_arr * v_arr
        rets = np.diff(c_arr) / c_arr[:-1]
        # Amihud
        amihud = np.abs(rets) / np.maximum(dv[1:], 1e6)
        window = 20
        amihud_20 = np.convolve(amihud, np.ones(window)/window, mode='valid')
        amihud_60 = np.convolve(amihud, np.ones(60)/60, mode='valid')
        la = min(len(amihud_20), len(amihud_60))
        improvement = -(amihud_20[-la:] / np.maximum(amihud_60[-la:], 1e-6) - 1)
        improvement = np.clip(improvement, -0.8, 0.5)
        # momentum
        mom20 = c_arr[20:] / c_arr[:-20] - 1
        mom20 = np.clip(mom20, -0.3, 0.3)
        mom_signal = (mom20 > 0).astype(float)
        # Align
        la2 = min(len(improvement), len(mom_signal))
        signal = improvement[-la2:] * mom_signal[-la2:]
        if len(signal) >= 5:
            arr[i] = np.mean(signal[-5:])
        elif len(signal) > 0:
            arr[i] = signal[-1]
        valid[i] = True
    return arr, valid

# ============================================================
#  rebalance: 周频调仓 (周一 open)
# ============================================================
def rebalance(context):
    import gc
    current_date = context.current_dt.date()

    # warmup: 前N周跳过调仓 (参照王者: 计数器模式)
    g.rebalance_count += 1
    if g.rebalance_count <= g.warmup_weeks:
        return

    prev_date = context.previous_date

    # ── 1. 股票池过滤 (优化版: 单次get_current_data + 提前break) ──
    all_sec = get_all_securities(['stock'], prev_date).index.tolist()
    cd = get_current_data()  # 只调1次, 非每500只
    filtered = []
    for s in all_sec:
        if len(filtered) >= 2000:  # 够2000就停
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

    # ── 2. 获取价格数据 (Panel模式=JQ兼容性最好, v2验证通过) ──
    lookback = 150  # F1需140日(120+20), 留10日余量
    def _get_px(univ, field):
        px = get_price(univ, count=lookback, end_date=prev_date,
                       frequency='daily', fields=field,
                       skip_paused=False, fq='pre')
        # Panel兼容提取 (v2验证模式)
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
    volume_p = _get_px(universe, 'volume')

    if not isinstance(close_p, pd.DataFrame) or close_p.empty:
        log.error('[%s] close_p 无效: type=%s' % (current_date, type(close_p).__name__))
        return

    # ── 3. 构建 dict ──
    cols = list(close_p.columns)
    close_dict  = {s: close_p[s].values for s in cols}
    high_dict   = {s: high_p[s].values for s in cols if s in high_p.columns}
    low_dict    = {s: low_p[s].values for s in cols if s in low_p.columns}
    volume_dict = {s: volume_p[s].values for s in cols if s in volume_p.columns}

    # ── 4. 筛选有效股票 (至少65天数据) ──
    valid_stocks = [s for s in cols if len(close_dict.get(s, [])) >= 65]
    if len(valid_stocks) < g.min_stocks:
        log.warning('[%s] 有效股票不足: %d' % (current_date, len(valid_stocks)))
        return

    # ── 5. 计算9个因子 ──
    f1, v1 = compute_f1_max_drawdown(valid_stocks, close_dict)
    f2, v2 = compute_f2_hl_vol_spread(valid_stocks, high_dict, low_dict, close_dict)
    f3, v3 = compute_f3_volume_crowding(valid_stocks, volume_dict)
    f4, v4 = compute_f4_event_convexity(valid_stocks, volume_dict, close_dict)
    f5, v5 = compute_f5_lottery(valid_stocks, close_dict, volume_dict)
    f6, v6 = compute_f6_earnings_drift(valid_stocks, close_dict, volume_dict)
    f7, v7 = compute_f7_tail_hedge(valid_stocks, close_dict)
    f8, v8 = compute_f8_coskew(valid_stocks, close_dict)
    f9, v9 = compute_f9_small_liquidity(valid_stocks, close_dict, volume_dict)

    all_factors = [f1, f2, f3, f5, f9]
    all_valid   = [v1, v2, v3, v5, v9]

    # ── 6. 筛选: 至少3个因子有效的股票 ──
    n_stocks = len(valid_stocks)
    combined_valid = np.ones(n_stocks, dtype=bool)
    factor_count = np.zeros(n_stocks)
    for j in range(5):
        has_f = all_valid[j] & (~np.isnan(all_factors[j]))
        factor_count += has_f.astype(int)
        combined_valid = combined_valid & has_f

    # Ablation C: 至少3个因子有效 (5因子子集)
    min_factors = 3
    multi_valid = factor_count >= min_factors

    n_pass = int(np.sum(combined_valid))
    n_multi = int(np.sum(multi_valid))

    if n_multi < g.min_stocks:
        log.warning('[%s] 多因子有效股票不足: strict=%d multi=%d' % (current_date, n_pass, n_multi))
        # Fallback: 降为至少3个因子有效
        min_factors = 3
        multi_valid = factor_count >= min_factors
        n_multi = int(np.sum(multi_valid))
        if n_multi < g.min_stocks:
            return

    # ── 7. Rank-product 复合 ──
    composite = np.ones(n_stocks)
    count = 0
    for j in range(5):
        f_arr = all_factors[j]
        score = rank_pct(f_arr, multi_valid)  # 对multi_valid的股票排名
        composite *= score
        count += 1

    composite[~multi_valid] = -1

    # ── 8. Top-N 选股 ──
    top_idx = np.argsort(composite)[-g.top_n:][::-1]
    top_stocks = [valid_stocks[i] for i in top_idx if composite[i] > 0]

    if len(top_stocks) < g.min_stocks:
        log.warning('[%s] Top选股不足: %d' % (current_date, len(top_stocks)))
        return

    # ── 9. 卖出不在目标池的股票 ──
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue
            except Exception:
                pass
            order_target_value(s, 0)

    if len(top_stocks) == 0:
        return

    # ── 10. 等权买入 (参照王者: order_target_value + 最低1手检查) ──
    weight = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    for s in top_stocks:
        try:
            last_px = get_current_data()[s].last_price
            if last_px <= 0 or weight / last_px < 100:
                continue
        except Exception:
            continue
        order_target_value(s, weight)
        ordered += 1

    log.info('[%s] 调仓: %d/%d stock, valid=%d multi=%d' %
             (current_date, ordered, len(top_stocks), n_pass, n_multi))
    gc.collect()
