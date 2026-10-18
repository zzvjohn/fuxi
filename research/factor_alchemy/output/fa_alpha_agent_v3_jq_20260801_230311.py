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
        # 检查所有4个字段
        if (high.get(s) is None or len(high.get(s, [])) < 25 or
            low.get(s) is None or len(low.get(s, [])) < 25 or
            close.get(s) is None or len(close.get(s, [])) < 25 or
            volume.get(s) is None or len(volume.get(s, [])) < 25):
            continue
        h = np.array(high.get(s), dtype=float)
        l_ = np.array(low.get(s), dtype=float)
        c = np.array(close.get(s), dtype=float)
        v = np.array(volume.get(s), dtype=float)
        typical = (h[-20:] + l_[-20:] + c[-20:]) * v[-20:] / 3
        curr_tp = (h[-1] + l_[-1] + c[-1]) * v[-1] / 3
        ma_tp = np.mean(typical)
        arr[i] = -(curr_tp - ma_tp)
        valid[i] = True
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        log.warn('money_flow_20: 0/%d valid (high_keys=%d low_keys=%d close_keys=%d vol_keys=%d)' %
                 (n, len(high), len(low), len(close), len(volume)))
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

def compute_nf_61d217(stocks, price_data):
    """截面rank(量增速x价增速)。量价同向=趋势健康, 量价背离=反转信号。v1/v2缺失。"""
    import numpy as np
    n = len(stocks)
    close = price_data.get('close', {})
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        if close.get(s) is None:
            continue
        c = np.array(close.get(s, []), dtype=float)
        if len(c) < 25:
            continue
        arr[i] = c[-1] / c[-21] - 1
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
    """最小日收益(取负)/波动率。极端下跌相对波动水平, 捕捉尾部风险溢价。 x tvma_20: 互补评分=10.0"""
    a, va = compute_f_(stocks, price_data)
    b, vb = compute_tvma_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp5_money_flow_20(stocks, price_data):
    """截面取负((取负(20日最小收益))/波动率)。极端下行风险相对总风险的定价。高 x money_flow_20: 互补"""
    a, va = compute_nf_02a304(stocks, price_data)
    b, vb = compute_money_flow_20(stocks, price_data)
    valid = va & vb
    ra = rank_pct(a, valid)
    rb = rank_pct(b, valid)
    score = np.where(valid, ra * rb, 0.5)
    return score, valid

def compute_score_comp6_turnover_std_cv(stocks, price_data):
    """截面rank(量增速x价增速)。量价同向=趋势健康, 量价背离=反转信号。v1/ x turnover_std_cv: """
    a, va = compute_nf_61d217(stocks, price_data)
    b, vb = compute_turnover_std_cv(stocks, price_data)
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
        (compute_score_comp1_overnight_tvma, 'comp1_overnight_tvma'),
        (compute_score_comp2_dollar_turnover, 'comp2_dollar_turnover'),
        (compute_score_comp3_moneyflow_ret3m, 'comp3_moneyflow_ret3m'),
        (compute_score_comp4_tvma_20, 'comp4_tail_tvma'),
        (compute_score_comp5_money_flow_20, 'comp5_downside_moneyflow'),
        (compute_score_comp6_turnover_std_cv, 'comp6_volprice_tstd'),
    ]
    count = 0
    for (func, label) in score_funcs:
        try:
            score, v = func(stocks, price_data)
            n_valid = int(np.sum(v))
            if n_valid > 0:
                composite[v] *= score[v]
                count += 1
                valid = valid & v
            else:
                log.warn('Ensemble %s: 0 valid stocks, skipping' % label)
        except Exception as e:
            log.error('Ensemble %s failed: %s' % (label, str(e)))
    log.info('Ensemble: %d/%d composites active, final_valid=%d' %
             (count, len(score_funcs), int(np.sum(valid))))
    return composite, valid

def rebalance(context):
    import numpy as np
    import gc
    g.trade_days += 1
    if g.trade_days < 5:
        return
    current_dt = context.current_dt
    prev_date = context.previous_date
    log.info('[rebalance] day=%s trade_days=%d' % (str(current_dt)[:10], g.trade_days))

    # Universe
    try:
        all_sec = get_all_securities(['stock'], prev_date)
        universe = all_sec.index.tolist()
    except Exception as e:
        log.error('[rebalance] get_all_securities failed: %s' % str(e))
        return
    log.info('[rebalance] all_sec count=%d' % len(universe))

    CHUNK = 500
    filtered = []
    for k in range(0, len(universe), CHUNK):
        chunk = universe[k:k+CHUNK]
        try:
            cd = get_current_data()
            for s in chunk:
                try:
                    if cd[s].paused:
                        continue
                    if cd[s].is_st:
                        continue
                    if cd[s].high_limit <= cd[s].last_price:
                        continue
                    filtered.append(s)
                except Exception:
                    continue
        except Exception:
            filtered.extend(chunk)
    universe = [s for s in filtered if s.startswith(('0','3','6'))][:2000]
    log.info('[rebalance] post-filter universe=%d' % len(universe))
    if len(universe) < g.stock_num:
        log.warn('[rebalance] universe too small: %d < %d' % (len(universe), g.stock_num))
        return

    # Load price data
    lookback = 70
    try:
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
    except Exception as e:
        log.error('[rebalance] get_price failed: %s' % str(e))
        return

    # Panel兼容处理（参照V3策略）
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

    n_close = len(px_close) if hasattr(px_close, '__len__') else 0
    n_cols = len(px_close.columns) if hasattr(px_close, 'columns') else 0
    log.info('[rebalance] px_close rows=%d cols=%d' % (n_close, n_cols))

    price_data = {
        'close': {},
        'open': {},
        'high': {},
        'low': {},
        'volume': {},
    }
    for s in universe:
        if s in px_close.columns:
            price_data['close'][s] = px_close[s].values
        if s in px_open.columns:
            price_data['open'][s] = px_open[s].values
        if s in px_high.columns:
            price_data['high'][s] = px_high[s].values
        if s in px_low.columns:
            price_data['low'][s] = px_low[s].values
        if s in px_volume.columns:
            price_data['volume'][s] = px_volume[s].values

    valid_universe = [s for s in universe if len(price_data['close'].get(s, [])) >= 70]
    log.info('[rebalance] price_data sizes: close=%d open=%d high=%d low=%d volume=%d' %
             (len(price_data['close']), len(price_data['open']),
              len(price_data['high']), len(price_data['low']),
              len(price_data['volume'])))
    log.info('[rebalance] valid_universe=%d (need >=%d)' % (len(valid_universe), g.stock_num))
    if len(valid_universe) < g.stock_num:
        log.warn('[rebalance] valid_universe too small: %d < %d' % (len(valid_universe), g.stock_num))
        return

    try:
        composite, valid = compute_ensemble(valid_universe, price_data)
    except Exception as e:
        log.error('[rebalance] compute_ensemble failed: %s' % str(e))
        return

    composite_arr = np.where(valid, composite, -1)
    n_valid = int(np.sum(valid))
    if n_valid > 0:
        log.info('[rebalance] composite range=[%.4f, %.4f] valid=%d/%d' %
                 (float(np.min(composite_arr[valid])), float(np.max(composite_arr[valid])),
                  n_valid, len(valid_universe)))
    else:
        log.warn('[rebalance] all composites returned 0 valid stocks')
        return

    topn = g.stock_num
    top_idx = np.argsort(composite_arr)[-topn:][::-1]
    top_stocks = [valid_universe[i] for i in top_idx if composite_arr[i] > -1]
    log.info('[rebalance] top_stocks=%d' % len(top_stocks))

    # Sell: skip positions too small to close
    for s in list(context.portfolio.positions.keys()):
        if s not in top_stocks:
            try:
                pos = context.portfolio.positions[s]
                if pos.total_amount < 100:
                    continue  # too small, position was already fractional
            except Exception:
                pass
            order_target_value(s, 0)

    if len(top_stocks) == 0:
        log.warn('[rebalance] no stocks selected')
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
    log.info('[rebalance] ordered %d/%d stocks weight=%.2f' % (ordered, len(top_stocks), weight))
    gc.collect()
