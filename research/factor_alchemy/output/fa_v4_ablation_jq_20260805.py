# -*- coding: utf-8 -*-
"""
V4 Paradigm Ablation: 隔离测试每个新复合的独立贡献
=====================================================
生成于: 2026-08-05
目的: 确定 V4 7复合 vs 王者6复合 -16pp 的归因

运行方式: 修改下方 ABLATION_MODE 选择要测试的变体:
  "core4"   — 仅4底仓 (K1-K4, 对标王者去掉comp3/comp6)
  "V4A"     — 仅情绪×日内 (intraday_amplitude × overnight_reversal)
  "V4B"     — 仅截面交互 (volume_concentration × money_flow)
  "V4C"     — 仅行业轮动 (industry_relative_strength × ret_3m)
  "full7"   — 完整7复合 (= V4 v3)

王者基准: +182.57% / Sharpe 0.69 / MaxDD -35.41%
V4 v3:    +166.20% / Sharpe 0.64 / MaxDD -35.09%
"""

ABLATION_MODE = "core4"  # ← 修改此处切换变体

import datetime
import pandas as pd
from jqlib.technical_analysis import *


def initialize(context):
    g.stock_num = 80
    g.trade_days = 0
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)
    log.set_level('strategy', 'info')
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0001, close_commission=0.0001),
                   type='stock')
    set_slippage(FixedSlippage(0.003))
    run_weekly(rebalance, 1, time='open', reference_security='000300.XSHG')


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def rank_pct(arr, valid):
    import numpy as np
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


# ═══════════════════════════════════════════════════════════
# 基础因子 (与 V4 v3 完全一致)
# ═══════════════════════════════════════════════════════════

def compute_dollar_vol_20d(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or volume.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(c) < 25 or len(v) < 25:
            continue
        dv = np.mean(c[-20:] * v[-20:])
        arr[i] = -np.log(dv + 1) if dv > 0 else 0
        valid[i] = True
    return arr, valid


def compute_overnight_5d(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or open_.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        if len(c) < 7 or len(o) < 7:
            continue
        gap = o[-5:] / np.roll(c, 1)[-5:] - 1
        arr[i] = np.sum(gap)
        valid[i] = True
    return arr, valid


def compute_tvma_20(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or volume.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(c) < 25 or len(v) < 25:
            continue
        cv = c[-20:] * v[-20:]
        ma_cv = np.mean(cv)
        curr_cv = c[-1] * v[-1]
        arr[i] = -(curr_cv / ma_cv) if ma_cv > 0 else 0
        valid[i] = True
    return arr, valid


def compute_turnover_std_cv(stocks, price_data):
    import numpy as np
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if volume.get(s) is None:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        if len(v) < 25:
            continue
        std_v = np.nanstd(v[-20:])
        mean_v = np.nanmean(v[-20:])
        arr[i] = -(std_v / mean_v) if mean_v > 0 else 0
        valid[i] = True
    return arr, valid


def compute_nf_ff7913(stocks, price_data):
    import numpy as np
    n = len(stocks)
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if volume.get(s) is None or len(volume.get(s, [])) < W:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        ma5 = np.nanmean(v[-5:])
        ma60 = np.nanmean(v[-60:])
        if ma60 > 0:
            arr[i] = -(ma5 / ma60)
        else:
            arr[i] = -1.0
        valid[i] = True
    return arr, valid


def compute_f_(stocks, price_data):
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 67:
            continue
        rets = c[1:] / c[:-1] - 1
        recent = rets[-20:]
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid


def compute_money_flow_20(stocks, price_data):
    import numpy as np
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if any(high.get(s) is None or len(high.get(s, [])) < 25 for _ in [1]):
            continue
        h = np.array(high.get(s, []), dtype=float)
        l_ = np.array(low.get(s, []), dtype=float)
        c = np.array(close.get(s, []), dtype=float)
        v = np.array(volume.get(s, []), dtype=float)
        if len(h) < 25 or len(l_) < 25:
            continue
        typical = (h[-20:] + l_[-20:] + c[-20:]) * v[-20:] / 3
        curr_tp = (h[-1] + l_[-1] + c[-1]) * v[-1] / 3
        ma_tp = np.mean(typical)
        arr[i] = -(curr_tp - ma_tp)
        valid[i] = True
    return arr, valid


# ═══════════════════════════════════════════════════════════
# V4 范式因子
# ═══════════════════════════════════════════════════════════

def compute_intraday_amplitude(stocks, price_data):
    import numpy as np
    high = price_data.get('high', {})
    low = price_data.get('low', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if any(x.get(s) is None or len(x.get(s, [])) < 7 for x in [high, low, open_]):
            continue
        h = np.array(high.get(s, []), dtype=float)
        l_ = np.array(low.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        daily_range = (h[-5:] - l_[-5:]) / np.where(o[-5:] > 0, o[-5:], 1)
        avg_range = np.nanmean(daily_range)
        arr[i] = -avg_range
        valid[i] = True
    return arr, valid


def compute_overnight_reversal(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    open_ = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None or open_.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        o = np.array(open_.get(s, []), dtype=float)
        if len(c) < 7 or len(o) < 7:
            continue
        prev_c = np.roll(c, 1)[-5:]
        ov_gap = o[-5:] / np.where(prev_c > 0, prev_c, 1) - 1
        intra = c[-5:] / np.where(o[-5:] > 0, o[-5:], 1) - 1
        reversal_signal = []
        for g, it in zip(ov_gap, intra):
            if g > 0 and it < 0:
                reversal_signal.append(-abs(it))
            elif g < 0 and it > 0:
                reversal_signal.append(abs(it))
            else:
                reversal_signal.append(0)
        arr[i] = np.mean(reversal_signal) if reversal_signal else 0
        valid[i] = True
    return arr, valid


def compute_volume_concentration(stocks, price_data):
    import numpy as np
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if volume.get(s) is None:
            continue
        v = np.array(volume.get(s, []), dtype=float)
        if len(v) < 25:
            continue
        short_ma = np.nanmean(v[-5:])
        long_ma = np.nanmean(v[-20:])
        if long_ma > 0:
            arr[i] = -(short_ma / long_ma)
        else:
            arr[i] = -1.0
        valid[i] = True
    return arr, valid


def compute_ret_3m(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 65:
            continue
        arr[i] = -(c[-1] / c[-61] - 1)
        valid[i] = True
    return arr, valid


def compute_industry_relative_strength(stocks, price_data):
    import numpy as np
    close = price_data.get('close', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    stock_mom = {}
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 25:
            continue
        stock_mom[s] = c[-1] / c[-21] - 1
    if len(stock_mom) < 30:
        return arr, valid
    try:
        industry_map = get_industry([s for s in stocks if s in stock_mom])
    except Exception:
        industry_map = {}
    ind_mom = {}
    for s, mom in stock_mom.items():
        ind = industry_map.get(s, {}).get('sw_l1', {}).get('industry_name', '未知')
        if ind not in ind_mom:
            ind_mom[ind] = []
        ind_mom[ind].append(mom)
    ind_avg = {}
    for ind, moms in ind_mom.items():
        if len(moms) >= 3:
            ind_avg[ind] = np.nanmean(moms)
    for i, s in enumerate(stocks):
        if s not in stock_mom:
            continue
        ind = industry_map.get(s, {}).get('sw_l1', {}).get('industry_name', '未知')
        avg_mom = ind_avg.get(ind)
        if avg_mom is not None:
            arr[i] = stock_mom[s] - avg_mom
            valid[i] = True
    return arr, valid


# ═══════════════════════════════════════════════════════════
# 复合得分 (按 ABLATION_MODE 选择)
# ═══════════════════════════════════════════════════════════

def _make_pair(func_a, func_b):
    """生成配对复合函数"""
    def composite(stocks, price_data):
        import numpy as np
        a, va = func_a(stocks, price_data)
        b, vb = func_b(stocks, price_data)
        v = va & vb
        ra = rank_pct(a, v)
        rb = rank_pct(b, v)
        return np.where(v, ra * rb, 0.5), v
    return composite


# 底仓4复合
K1 = _make_pair(compute_overnight_5d, compute_tvma_20)
K2 = _make_pair(compute_dollar_vol_20d, compute_turnover_std_cv)
K3 = _make_pair(compute_nf_ff7913, compute_tvma_20)
K4 = _make_pair(compute_f_, compute_dollar_vol_20d)

# V4 范式3复合
V4A = _make_pair(compute_intraday_amplitude, compute_overnight_reversal)
V4B = _make_pair(compute_volume_concentration, compute_money_flow_20)
V4C = _make_pair(compute_industry_relative_strength, compute_ret_3m)


def compute_ensemble(stocks, price_data):
    """根据 ABLATION_MODE 返回不同复合集"""
    import numpy as np

    mode_map = {
        "core4": [K1, K2, K3, K4],
        "V4A":   [V4A],
        "V4B":   [V4B],
        "V4C":   [V4C],
        "full7": [K1, K2, K3, K4, V4A, V4B, V4C],
    }
    funcs = mode_map.get(ABLATION_MODE, mode_map["core4"])

    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)

    for func in funcs:
        score, v = func(stocks, price_data)
        if int(np.sum(v)) > 0:
            composite[v] *= score[v]
            valid = valid & v

    return composite, valid


# ═══════════════════════════════════════════════════════════
# 主调仓 (v3 fix: order_target 份额制 + delta-aware)
# ═══════════════════════════════════════════════════════════

def rebalance(context):
    import numpy as np
    import gc

    g.trade_days += 1
    if g.trade_days < 5:
        return

    prev_date = context.previous_date

    # Universe
    universe = get_all_securities(['stock'], prev_date).index.tolist()
    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k+CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                if cd[s].paused:
                    continue
                if cd[s].is_st:
                    continue
                if cd[s].high_limit <= cd[s].last_price:
                    continue
                filtered.append(s)
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(('0', '3', '6'))][:2000]
    if len(universe) < g.stock_num:
        return

    # 数据加载
    lookback = 70
    px_close = get_price(universe, count=lookback, end_date=prev_date,
                         frequency='daily', fields='close',
                         skip_paused=False, fq='pre')
    px_open = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields='open',
                        skip_paused=False, fq='pre')
    px_high = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields='high',
                        skip_paused=False, fq='pre')
    px_low = get_price(universe, count=lookback, end_date=prev_date,
                       frequency='daily', fields='low',
                       skip_paused=False, fq='pre')
    px_volume = get_price(universe, count=lookback, end_date=prev_date,
                          frequency='daily', fields='volume',
                          skip_paused=False, fq='pre')

    _Panel = getattr(pd, 'Panel', None)
    if _Panel is not None:
        for px, fname in [(px_close, 'close'), (px_open, 'open'),
                           (px_high, 'high'), (px_low, 'low'),
                           (px_volume, 'volume')]:
            if isinstance(px, _Panel):
                if fname in getattr(px, 'items', []):
                    px = px[fname]
                else:
                    px = px.minor_xs(fname)
            if fname == 'close':
                px_close = px
            elif fname == 'open':
                px_open = px
            elif fname == 'high':
                px_high = px
            elif fname == 'low':
                px_low = px
            elif fname == 'volume':
                px_volume = px

    price_data = {
        'close': {s: px_close[s].values for s in universe if s in px_close.columns},
        'open': {s: px_open[s].values for s in universe if s in px_open.columns},
        'high': {s: px_high[s].values for s in universe if s in px_high.columns},
        'low': {s: px_low[s].values for s in universe if s in px_low.columns},
        'volume': {s: px_volume[s].values for s in universe if s in px_volume.columns},
    }

    valid_universe = [s for s in universe
                      if len(price_data['close'].get(s, [])) >= 65
                      and len(price_data['volume'].get(s, [])) >= 65]
    if len(valid_universe) < g.stock_num:
        return

    # Ensemble
    composite, valid = compute_ensemble(valid_universe, price_data)
    composite_arr = np.where(valid, composite, -1)

    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[-topn:][::-1]
    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]

    # 卖出
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                amt = int(pos.total_amount)
                if amt < 100:
                    continue
                order_target(s, 0)
            except Exception:
                pass

    if len(top_stocks) == 0:
        return

    # 买入 (order_target 份额制 + delta-aware)
    weight_per_stock = context.portfolio.total_value / len(top_stocks)
    ordered = 0
    skipped_price = 0
    skipped_delta = 0
    for s in top_stocks:
        try:
            last_px = get_current_data()[s].last_price
            if last_px <= 0:
                continue
            target_shares = max(0, int(weight_per_stock / last_px / 100) * 100)
            current_shares = 0
            if s in context.portfolio.positions:
                current_shares = int(context.portfolio.positions[s].total_amount)
            if target_shares == 0 and current_shares == 0:
                skipped_price += 1
                continue
            delta = abs(target_shares - current_shares)
            if 0 < delta < 100:
                skipped_delta += 1
                continue
            if delta == 0:
                continue
        except Exception:
            continue
        order_target(s, target_shares)
        ordered += 1

    if skipped_price > 0 or skipped_delta > 0:
        log.info('[%s] ordered %d/%d (skip: %d price, %d delta<1lot)'
                 % (ABLATION_MODE, ordered, len(top_stocks),
                    skipped_price, skipped_delta))
    else:
        log.info('[%s] ordered %d/%d stocks'
                 % (ABLATION_MODE, ordered, len(top_stocks)))

    gc.collect()
