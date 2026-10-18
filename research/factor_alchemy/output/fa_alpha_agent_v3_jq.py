import datetime
import pandas as pd
from jqlib.technical_analysis import *

# ══════════════ A4 指数刹车 — 终版默认开启 (2026-08-25 固化) ══════════════
# 中证1000(000852) 15交易日收益 <= -12% → 本次调仓清仓并跳过, 每周复检, 指标修复后自然恢复
# JQ A/B 双验证 (21.1.1~26.7.30):
#   打散v1b:   +105.14%/0.494/31.8% vs 基线 +95.22%/0.420/32.1%  → +9.92pp/+0.074
#   AlphaAgent v3 (本策略): +198.24%/0.78/27.59% vs 基线 +182.57%/0.689/35.41% → +15.67pp/+0.091/-7.82pp
# 代价: 急跌后 V 反被空仓错过后半段 (2026 约 -1.6pp, 保险成本)
# 基线版备份: output/fa_alpha_agent_v3_jq_nogate_20260825.py (A/B 对照用)
GATE_INDEX = True
IDX_BRAKE_CODE = "000852.XSHG"   # 中证1000
IDX_BRAKE_WIN = 15               # 15 交易日窗口 (含当日, 与打散v1b一致)
IDX_BRAKE_TH = -0.12             # 跌幅阈值 -12%

def _check_index_brake(context):
    """返回 True = 触发刹车 (已清仓, 本次调仓跳过)"""
    g.gate_stats["weeks_total"] += 1
    prev_date = context.previous_date
    try:
        idx_px = get_price(IDX_BRAKE_CODE, end_date=prev_date,
                           count=IDX_BRAKE_WIN + 1, fields=["close"], fq="pre")
        if len(idx_px) < IDX_BRAKE_WIN + 1:
            return False
        ret15 = float(idx_px["close"].iloc[-1] / idx_px["close"].iloc[0] - 1.0)
    except Exception as e:
        log.warn("gate index fetch fail: %s" % str(e)[:60])
        return False
    if ret15 > IDX_BRAKE_TH:
        return False
    g.gate_stats["index"] += 1
    g.gate_stats["weeks_gated"] += 1
    cd = get_current_data()
    sold = 0
    for x in list(context.portfolio.positions.keys()):
        try:
            if cd[x].paused:
                continue
            pos = context.portfolio.positions[x]
            if pos.total_amount < 100:
                continue
            order_target(x, 0)
            sold += 1
        except Exception:
            continue
    log.info("GATE index_brake_%.1f%% | %s | clear=%d | gated_weeks=%d" % (
        ret15 * 100, prev_date, sold, g.gate_stats["weeks_gated"]))
    return True



def initialize(context):
    g.stock_num = 80
    g.limit_pct = 0.10
    g.initial_capital = context.portfolio.total_value
    g.trade_days = 0
    g.gate_stats = {"index": 0, "weeks_gated": 0, "weeks_total": 0}
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
    log.info("AlphaAgent v3 allweather: 6复合 + A4指数刹车 (GATE_INDEX=%s, %s 15日<=%.0f%%)" % (
        GATE_INDEX, IDX_BRAKE_CODE, IDX_BRAKE_TH * 100))

def rank_pct(arr, valid):
    '''截面百分位排名, NaN被忽略'''
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

def zscore(arr, valid):
    import numpy as np
    v = arr[valid]
    if len(v) < 3:
        return np.zeros(len(arr))
    m, s = np.mean(v), np.std(v)
    if s < 1e-10:
        return np.zeros(len(arr))
    result = np.zeros(len(arr))
    result[valid] = (v - m) / s
    return result

def compute_dollar_vol_20d(stocks, price_data):
    """成交额对数(取负)"""
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

def compute_money_flow_20(stocks, price_data):
    """资金流偏移(取负)"""
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

def compute_overnight_5d(stocks, price_data):
    """隔夜累计5日"""
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

def compute_ret_3m(stocks, price_data):
    """3月收益(取负=反转)"""
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

def compute_turnover_std_cv(stocks, price_data):
    """换手变异系数(取负)"""
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

def compute_tvma_20(stocks, price_data):
    """量价趋势偏移(取负)"""
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

def compute_nf_ff7913(stocks, price_data):
    """5日/60日均量截面取负。长期缩量=筹码锁定/惜售, A股底部特征。"""
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
    """最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。"""
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
        # 下行风险指标
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_nf_02a304(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高比率=被过度惩罚。"""
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
        # 下行风险指标
        min_ret = np.nanmin(recent)
        std_ret = np.nanstd(recent)
        if std_ret > 0:
            arr[i] = -min_ret / std_ret
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_score_comp1_overnight_tvma(stocks, price_data):
    """隔夜累计x量价趋势: V3最强复合"""
    a, va = compute_overnight_5d(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp2_dollar_turnover(stocks, price_data):
    """成交额x换手稳定性: V3次强"""
    a, va = compute_dollar_vol_20d(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp3_moneyflow_ret3m(stocks, price_data):
    """资金流x动量反转: V3原value_momentum简化版"""
    a, va = compute_money_flow_20(stocks, price_data)
    b, vb = compute_ret_3m(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp4_tvma_20(stocks, price_data):
    """5日/60日均量截面取负。长期缩量=筹码锁定/惜售, A股底部特征。v1/v2缺 x tvma_20: 互补评分=10."""
    a, va = compute_nf_ff7913(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp5_dollar_vol_20d(stocks, price_data):
    """最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。 x dollar_vol_20d: 互补评分=1"""
    a, va = compute_f_(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp6_money_flow_20(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高 x money_flow_20: 互补"""
    a, va = compute_nf_02a304(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_ensemble(stocks, price_data):
    import numpy as np
    n = len(stocks)
    composite = np.ones(n) * 0.5
    valid = np.ones(n, dtype=bool)
    score_funcs = [
        (compute_score_comp1_overnight_tvma,),
        (compute_score_comp2_dollar_turnover,),
        (compute_score_comp3_moneyflow_ret3m,),
        (compute_score_comp4_tvma_20,),
        (compute_score_comp5_dollar_vol_20d,),
        (compute_score_comp6_money_flow_20,),
    ]
    count = 0
    for (func,) in score_funcs:
        score, v = func(stocks, price_data)
        n_v = int(np.sum(v))
        if n_v > 0:
            composite[v] *= score[v]
            count += 1
            valid = valid & v
    return composite, valid

def rebalance(context):
    import numpy as np
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return
    current_dt = context.current_dt
    prev_date = context.previous_date

    # ── Step 0 (A4 allweather): 指数刹车 ──
    if GATE_INDEX and _check_index_brake(context):
        return

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
    universe = [s for s in filtered if s.startswith(('0','3','6'))][:2000]
    if len(universe) < g.stock_num:
        return

    # Load price data (参照V3策略: 不用panel=False + Panel兼容)
    lookback = 70
    field = 'close'
    px_close = get_price(universe, count=lookback, end_date=prev_date,
                         frequency='daily', fields=field,
                         skip_paused=False, fq='pre')
    field = 'open'
    px_open = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields=field,
                        skip_paused=False, fq='pre')
    field = 'high'
    px_high = get_price(universe, count=lookback, end_date=prev_date,
                        frequency='daily', fields=field,
                        skip_paused=False, fq='pre')
    field = 'low'
    px_low = get_price(universe, count=lookback, end_date=prev_date,
                       frequency='daily', fields=field,
                       skip_paused=False, fq='pre')
    field = 'volume'
    px_volume = get_price(universe, count=lookback, end_date=prev_date,
                          frequency='daily', fields=field,
                          skip_paused=False, fq='pre')

    # Panel兼容处理
    _Panel = getattr(pd, 'Panel', None)
    if _Panel is not None:
        if isinstance(px_close, _Panel):
            px_close = px_close['close'] if 'close' in getattr(px_close, 'items', []) else px_close.minor_xs('close')
        if isinstance(px_open, _Panel):
            px_open = px_open['open'] if 'open' in getattr(px_open, 'items', []) else px_open.minor_xs('open')
        if isinstance(px_high, _Panel):
            px_high = px_high['high'] if 'high' in getattr(px_high, 'items', []) else px_high.minor_xs('high')
        if isinstance(px_low, _Panel):
            px_low = px_low['low'] if 'low' in getattr(px_low, 'items', []) else px_low.minor_xs('low')
        if isinstance(px_volume, _Panel):
            px_volume = px_volume['volume'] if 'volume' in getattr(px_volume, 'items', []) else px_volume.minor_xs('volume')

    price_data = {
        'close': {s: px_close[s].values for s in universe if s in px_close.columns},
        'open': {s: px_open[s].values for s in universe if s in px_open.columns},
        'high': {s: px_high[s].values for s in universe if s in px_high.columns},
        'low': {s: px_low[s].values for s in universe if s in px_low.columns},
        'volume': {s: px_volume[s].values for s in universe if s in px_volume.columns},
    }

    valid_universe = [s for s in universe if len(price_data['close'].get(s, [])) >= 65]
    if len(valid_universe) < g.stock_num:
        return

    composite, valid = compute_ensemble(valid_universe, price_data)
    composite_arr = np.where(valid, composite, -1)

    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[-topn:][::-1]
    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]

    # Sell: skip positions too small to close
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
    log.info('ordered %d/%d stocks' % (ordered, len(top_stocks)))
    gc.collect()
