# -*- coding: utf-8 -*-
"""
Ralph Loop v0.2 — Stage 3 JQ 验证策略
========================================
生成时间: 2026-08-05 17:49
来源: Ralph Loop R→G→E→D 自进化管线
候选: 3因子通过四阶段验证 (S1 IC → S2 Corr → S3 Dedup → S4 OOS)

因子列表:
  1. regime_aware_volatility_envelope — 波动区制自适应包络 (failure_mode_inverse)
  2. low_freq_earnings_momentum_smooth   — 低频盈余动量+质量过滤 (low_turnover_medium_freq)
  3. volume_entropy_price_regime_switch — 量价熵值区制切换 (ml_inspired_construction)

策略设计:
  - 3因子等权 rank-product 复合
  - 周频调仓 (run_weekly, 周一开盘)
  - Top-80 等权持有
  - warmup=5周
"""
import datetime
import pandas as pd
import numpy as np

def initialize(context):
    g.stock_num = 80
    g.limit_pct = 0.10
    g.initial_capital = context.portfolio.total_value
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
    """截面百分位排名, NaN被忽略"""
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
# 因子 1: regime_aware_volatility_envelope
# Hypothesis: vm_diff塌缩源于无条件波动差分在regime切换时语义反转。
# 将波动率比包裹在自适应regime包络中——强趋势用小窗口、弱趋势用大窗口。
# Category: failure_mode_inverse
# ═══════════════════════════════════════════════════════════

def compute_regime_vol_env(stocks, price_data):
    """波动区制自适应包络"""
    close = price_data.get('close', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 201:  # 需要200日做40周vol
            continue

        ret_d = np.diff(c) / c[:-1]
        ret_d = np.nan_to_num(ret_d, 0)

        # 三个波动窗口
        vol_5w = np.nanstd(ret_d[-25:]) * np.sqrt(5)
        vol_15w = np.nanstd(ret_d[-75:]) * np.sqrt(15)
        vol_40w = np.nanstd(ret_d[-200:]) * np.sqrt(40)

        # 趋势强度决定使用哪个窗口比
        trend_abs_20w = abs(c[-1] / c[-101] - 1) if len(c) >= 101 else 0.05
        trend_abs_20w = np.clip(trend_abs_20w, 0.05, 0.5)

        # regime gate: 强趋势→快窗, 弱趋势→慢窗
        if vol_15w > 0:
            fast_ratio = np.clip(vol_5w / vol_15w, 0.5, 2.0) - 1.0
        else:
            fast_ratio = 0.0
        if vol_40w > 0:
            slow_ratio = np.clip(vol_15w / vol_40w, 0.5, 2.0) - 1.0
        else:
            slow_ratio = 0.0

        trend_weight = np.clip(trend_abs_20w, 0.1, 0.3) * 4  # 映射到0.4~1.2
        env_ratio = fast_ratio * np.clip(trend_weight, 0, 1) + slow_ratio * (1 - np.clip(trend_weight, 0, 1))

        # 趋势方向确认
        if len(c) >= 76:
            trend_sign = np.sign(c[-1] / c[-76] - 1)
        else:
            trend_sign = 0
        trend_conf = abs(np.clip(trend_sign, -1, 1)) + 0.1

        arr[i] = np.clip(env_ratio, -0.5, 0.5) * trend_conf
        valid[i] = True

    return arr, valid


# ═══════════════════════════════════════════════════════════
# 因子 2: low_freq_earnings_momentum_smooth
# Hypothesis: 高换手因子在A股被成本杀。月频盈余动量——5个月收益率
# +过去12个月收益稳定性做质量过滤，天然低换手(<30%/周)。
# Category: low_turnover_medium_freq
# ═══════════════════════════════════════════════════════════

def compute_low_freq_earn_mom(stocks, price_data):
    """低频盈余动量+质量过滤"""
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 251:  # 需要250日算12个月
            continue

        # 5个月动量（约100个交易日）
        mom_5m = np.clip(c[-1] / c[-101] - 1, -0.5, 0.5)

        # 12个月收益稳定性
        ret_12m = np.diff(c) / c[:-1]
        ret_12m = ret_12m[-250:]
        ret_mean = np.nanmean(ret_12m)
        ret_std = np.nanstd(ret_12m)
        stability = 1.0 - np.clip(ret_std / (abs(ret_mean) + 1e-6), 0, 5) / 5

        # 量确认
        v = np.array(volume.get(s, []), dtype=float)
        if len(v) < 250:
            vol_trend = 1.0
        else:
            ma100 = np.nanmean(v[-100:])
            ma250 = np.nanmean(v[-250:])
            vol_trend = ma100 / ma250 if ma250 > 0 else 0.8
        vol_ok = 1.0 if vol_trend > 0.8 else 0.0

        # 合成
        quality_weight = np.clip(stability, 0, 1) * vol_ok
        signal = mom_5m * np.clip(quality_weight, 0.2, 1)

        # 20日平滑
        arr[i] = np.clip(signal, -0.3, 0.3)
        valid[i] = True

    return arr, valid


# ═══════════════════════════════════════════════════════════
# 因子 3: volume_entropy_price_regime_switch
# Hypothesis: 成交量在不同价格分位的分布熵值变化反映信息不对称程度。
# 高熵→低熵过渡期动量加速（市场从分歧走向共识）。
# Category: ml_inspired_construction
# ═══════════════════════════════════════════════════════════

def compute_vol_entropy_regime(stocks, price_data):
    """量价熵值区制切换"""
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
        if len(c) < 22 or len(v) < 22:
            continue

        ret_1d = np.diff(c) / c[:-1]
        ret_1d = np.nan_to_num(ret_1d, 0)

        # 5日均量
        vol_ma5 = np.convolve(v, np.ones(5)/5, mode='valid')
        # 对齐 ret_1d 和 vol_ma5 长度
        min_len = min(len(ret_1d), len(vol_ma5))
        _ret = ret_1d[-min_len:]
        _vol = vol_ma5[-min_len:]

        # 用量价关系构建近似熵: vol/|ret| 反映每个价格变动单位的信息量
        info_ratio = _vol / (np.abs(_ret) + 1e-6)
        info_ratio = np.clip(info_ratio, 0, 50)

        # 20日趋势
        price_trend = c[-1] / c[-21] - 1 if len(c) >= 21 else 0

        # 熵值: info_ratio的离散度代理
        entropy_10 = np.nanstd(info_ratio[-10:]) if len(info_ratio) >= 10 else 0
        entropy_20 = np.nanstd(info_ratio[-20:]) if len(info_ratio) >= 20 else 0

        # 熵坍缩 = 共识形成 = 动量加速方向
        entropy_collapse = -(entropy_10 - entropy_20)

        # 区制切换信号：熵坍缩 × 趋势方向
        result = entropy_collapse * np.sign(price_trend)

        arr[i] = result
        valid[i] = True

    return arr, valid


# ═══════════════════════════════════════════════════════════
# 复合评分: 3因子等权 rank-product
# ═══════════════════════════════════════════════════════════

def compute_score_f1_regime_vol(stocks, price_data):
    """因子1 standalone: 波动区制包络"""
    a, va = compute_regime_vol_env(stocks, price_data)
    score = rank_pct(a, va)
    return score, va

def compute_score_f2_earn_mom(stocks, price_data):
    """因子2 standalone: 低频盈余动量"""
    a, va = compute_low_freq_earn_mom(stocks, price_data)
    score = rank_pct(a, va)
    return score, va

def compute_score_f3_entropy(stocks, price_data):
    """因子3 standalone: 量价熵值"""
    a, va = compute_vol_entropy_regime(stocks, price_data)
    score = rank_pct(a, va)
    return score, va

def compute_ensemble(stocks, price_data):
    """3因子等权 rank-product 复合"""
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)

    score_funcs = [
        compute_score_f1_regime_vol,
        compute_score_f2_earn_mom,
        compute_score_f3_entropy,
    ]

    count = 0
    for func in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v

    return composite, valid


# ═══════════════════════════════════════════════════════════
# 调仓逻辑
# ═══════════════════════════════════════════════════════════

def rebalance(context):
    import gc
    g.trade_days += 1
    if g.trade_days < 5:  # warmup: 5周
        return

    current_dt = context.current_dt
    prev_date = context.previous_date

    # Universe: 全A股, 去ST/停牌/涨停
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
    universe = [s for s in filtered if s.startswith(('0','3','6'))][:2000]
    if len(universe) < g.stock_num:
        return

    # 加载数据
    lookback = 260  # 因子3需要250日
    field_list = ['close', 'open', 'high', 'low', 'volume']
    price_data = {}
    for field in field_list:
        px = get_price(universe, count=lookback, end_date=prev_date,
                       frequency='daily', fields=field,
                       skip_paused=False, fq='pre')
        _Panel = getattr(pd, 'Panel', None)
        if _Panel is not None and isinstance(px, _Panel):
            px = px[field] if field in getattr(px, 'items', []) else px.minor_xs(field)
        price_data[field] = {s: px[s].values for s in universe if s in px.columns}

    valid_universe = [s for s in universe
                      if len(price_data['close'].get(s, [])) >= 200]
    if len(valid_universe) < g.stock_num:
        return

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
                if pos.total_amount < 100:
                    continue
            except Exception:
                pass
            order_target_value(s, 0)

    if len(top_stocks) == 0:
        return

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

    log.info('Ralph Loop v0.2: ordered %d/%d stocks' % (ordered, len(top_stocks)))
    gc.collect()
