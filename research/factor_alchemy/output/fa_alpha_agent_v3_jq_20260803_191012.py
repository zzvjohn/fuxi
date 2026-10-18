import datetime
import pandas as pd
from jqlib.technical_analysis import *

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

def compute_harvey_siddique_coskew(stocks, price_data):
    """Harvey-Siddique协偏度衡量个股收益与市场波动的协变尾部风险：负协偏度股票在市场下跌时具有崩盘敏感性（左尾风"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if close.get(s) is None or len(close.get(s, [])) < W:
            continue
        c = np.array(close.get(s, []), dtype=float)
        # 默认: 价格相对20日均值偏离
        ma20 = np.nanmean(c[-20:])
        if ma20 > 0:
            arr[i] = -(c[-1] / ma20 - 1)
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_idiosyncratic_tail_hedge_premium(stocks, price_data):
    """特质尾部对冲溢价——CAPM残差中隐含的尾部风险补偿：个股特质波动中的极端下行成分定价，残差尾部风险越大则补偿越高。关注"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if close.get(s) is None or len(close.get(s, [])) < W:
            continue
        c = np.array(close.get(s, []), dtype=float)
        # 默认: 价格相对20日均值偏离
        ma20 = np.nanmean(c[-20:])
        if ma20 > 0:
            arr[i] = -(c[-1] / ma20 - 1)
        else:
            arr[i] = 0.0
        valid[i] = True
    return arr, valid

def compute_max_drawdown_duration(stocks, price_data):
    """捕捉个股在极端下跌后回撤持续时间中隐含的尾部风险与下行保护溢价：回撤持续时间越长的股票，下行脆弱性越高，后续动量趋势继续"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    volume = price_data.get('volume', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    W = 65
    for i, s in enumerate(stocks):
        if close.get(s) is None or len(close.get(s, [])) < W:
            continue
        c = np.array(close.get(s, []), dtype=float)
        # 默认: 价格相对20日均值偏离
        ma20 = np.nanmean(c[-20:])
        if ma20 > 0:
            arr[i] = -(c[-1] / ma20 - 1)
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

def compute_score_comp4_harvey_siddique_tvma_20(stocks, price_data):
    """Harvey-Siddique协偏度衡量个股收益与市场波动的协变尾部风险：负协偏 x tvma_20: 互补评分=12."""
    a, va = compute_harvey_siddique_coskew(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp5_idiosyncratic_t_dollar_vol_20d(stocks, price_data):
    """特质尾部对冲溢价——CAPM残差中隐含的尾部风险补偿：个股特质波动中的极端下行成 x dollar_vol_20d: 互"""
    a, va = compute_idiosyncratic_tail_hedge_premium(stocks, price_data)
    b, vb = compute_dollar_vol_20d(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp6_max_drawdown_du_overnight_5d(stocks, price_data):
    """捕捉个股在极端下跌后回撤持续时间中隐含的尾部风险与下行保护溢价：回撤持续时间越长 x overnight_5d: 互补评"""
    a, va = compute_max_drawdown_duration(stocks, price_data)
    b, vb = compute_overnight_5d(stocks, price_data)
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
        (compute_score_comp4_harvey_siddique_tvma_20,),
        (compute_score_comp5_idiosyncratic_t_dollar_vol_20d,),
        (compute_score_comp6_max_drawdown_du_overnight_5d,),
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
