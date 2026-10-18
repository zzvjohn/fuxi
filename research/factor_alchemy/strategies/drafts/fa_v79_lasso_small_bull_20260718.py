# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 JQ 自动生成 — small_bull
================================================================
生成时间: 2026-07-19 00:02
生成器: jq_generator (P0.2)

因子组合:
  volume_stability  (w=0.251)   低CV→成交量稳定→低信息冲击→正向
  range_consistency  (w=0.212)   低振幅CV→价格稳定→低风险→正向
  atr_14  (w=0.198)   本地取负: 低ATR→低波动→正向Alpha
  high_low_range  (w=0.196)   窄振幅→低风险→正向(本地取负为高)
  ret_6m  (w=0.143)   长期动量→正向

参数:
  持仓数=30, 市值下限=5亿, 最大换手=60%
  调仓=周5 14:50, 基准=000300.XSHG
  保护限价=否 (直接市价单)

成本: QMT实盘 = 佣金万1 + 印花千1 + 滑点0.3%
      JQ FixedSlippage(0.003) = 0.003元/股, 低估真实成本
================================================================
"""

from __future__ import division
import datetime
import numpy as np
import pandas as pd


# ==================== 策略参数 ====================
STOCK_NUM      = 30
MAX_TURNOVER   = 0.60
MCAP_MIN       = 5
EXCLUDE_PREFIX = ('8', '4')

# 因子权重 (由 JQGenerator 自动生成)
WEIGHTS = {
    'volume_stability': 0.2510,
    'range_consistency': 0.2120,
    'atr_14': 0.1980,
    'high_low_range': 0.1960,
    'ret_6m': 0.1430
}
FACTOR_NAMES = ['volume_stability', 'range_consistency', 'atr_14', 'high_low_range', 'ret_6m']

# 数据加载参数
PRICE_FIELDS = ['close', 'high', 'low', 'volume']
PRICE_WINDOW = 141   # 最大回溯 + 5天余量

DEBUG = True
USE_PROTECTION_LIMIT = False


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

    g.stock_num      = STOCK_NUM
    g.max_turnover   = MAX_TURNOVER
    g.mcap_min       = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX
    g.weights        = WEIGHTS
    g.factor_names   = FACTOR_NAMES
    g.price_fields   = PRICE_FIELDS
    g.price_window   = PRICE_WINDOW
    g.last_holdings  = set()
    g.debug          = DEBUG
    g.turnover_rate_hist = {}   # turnover_std 滚动历史缓存


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')

    stocks = get_stock_pool(context)
    n_stocks = len(stocks)
    if g.debug:
        log.info('[%s] STEP1 pool=%d stocks' % (date_str, n_stocks))
    if n_stocks < g.stock_num:
        return

    price_data = load_price_data(stocks, context)

    factor_arrays = {}
    factor_valid = {}

    for fname in g.factor_names:
        try:
            arr, valid = compute_factor(fname, stocks, context, price_data)
            factor_arrays[fname] = arr
            factor_valid[fname] = valid
            if g.debug:
                log.info('[%s] STEP2 %s: valid=%d/%d (%.1f%%)' % (
                    date_str, fname,
                    np.sum(valid), n_stocks,
                    100 * np.sum(valid) / max(n_stocks, 1)))
        except Exception as e:
            log.error('[%s] STEP2 %s FAILED: %s' % (date_str, fname, str(e)))
            factor_arrays[fname] = np.full(n_stocks, np.nan)
            factor_valid[fname] = np.zeros(n_stocks, dtype=bool)

    # 截面 z-score (median/std + clip[-3,3])
    z_arrays = {}
    for fname in g.weights.keys():
        arr = factor_arrays[fname]
        valid = factor_valid[fname]
        z_raw = zscore_numpy(arr, valid)
        z_arrays[fname] = np.where(z_raw > -900, z_raw, np.nan)

    # 加权合成 (NaN-safe, 逐股权重归一化)
    composite = np.zeros(n_stocks)
    wsum = np.zeros(n_stocks)
    for fname, w in g.weights.items():
        z = z_arrays[fname]
        mask = np.isfinite(z)
        composite[mask] += w * z[mask]
        wsum[mask] += abs(w)

    mask_valid = wsum > 0
    composite[mask_valid] = composite[mask_valid] / wsum[mask_valid]
    composite[~mask_valid] = -999.0
    n_composite = np.sum(mask_valid)

    if g.debug:
        order = np.argsort(composite)[::-1]
        top_idx = order[:min(5, n_composite)]
        log.info('[%s] STEP4 composite: total=%d valid=%d top5_val=%s' % (
            date_str, n_stocks, n_composite,
            str(np.round(composite[top_idx][:5], 4))))

    if n_composite < g.stock_num:
        return

    sorted_idx = np.argsort(composite)[::-1]
    cand_n = min(len(sorted_idx), g.stock_num * 3)
    cand_codes = [stocks[i] for i in sorted_idx[:cand_n] if composite[i] > -900]
    try:
        cur = get_current_data()
    except Exception:
        cur = None
    target = []
    n_skip_pause = 0
    n_skip_limit = 0
    for s in cand_codes:
        if len(target) >= g.stock_num:
            break
        if cur is not None:
            try:
                cd = cur[s]
                if cd.paused:
                    n_skip_pause += 1
                    continue
                lp = getattr(cd, 'last_price', None)
                hl = getattr(cd, 'high_limit', None)
                if lp is not None and hl is not None and hl > 0 and lp >= hl - 1e-6:
                    n_skip_limit += 1
                    continue
            except Exception:
                pass
        target.append(s)

    if g.debug:
        log.info('[%s] STEP5 target=%d, skip_pause=%d skip_limit=%d, top3: %s' %
                 (date_str, len(target), n_skip_pause, n_skip_limit, str(target[:3])))

    rebalance(context, target)


# ==================== 股票池 ====================
def get_stock_pool(context):
    prev = context.previous_date
    date_str_pool = prev.strftime('%Y-%m-%d')

    try:
        all_sec = get_all_securities(['stock'], date=date_str_pool)
        all_stocks = list(all_sec.index)
        if len(all_stocks) == 0:
            raise ValueError('empty')
    except:
        all_sec = get_all_securities(['stock'])
        all_stocks = list(all_sec.index)

    stocks = [s for s in all_stocks if not s.startswith(EXCLUDE_PREFIX)]

    try:
        current_data = get_current_data()
    except:
        current_data = {}

    filtered = []
    for s in stocks:
        try:
            if s in current_data:
                cd = current_data[s]
                if cd.is_st or (cd.name and 'ST' in str(cd.name)):
                    continue
                if cd.paused:
                    continue
        except:
            pass
        try:
            start_date = all_sec.loc[s, 'start_date']
            if start_date is not None:
                if not isinstance(start_date, pd.Timestamp):
                    start_date = pd.Timestamp(start_date)
                if (context.current_dt - start_date).days < 180:
                    continue
        except:
            pass
        filtered.append(s)

    if len(filtered) == 0:
        return []

    date_str = prev.strftime('%Y-%m-%d')
    mcap_ok = []
    for i in range(0, len(filtered), 2000):
        batch = filtered[i:i + 2000]
        try:
            q = query(
                valuation.code,
                valuation.circulating_market_cap
            ).filter(valuation.code.in_(batch))
            df = get_fundamentals(q, date=date_str)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    code = row['code']
                    mcap_val = row.get('circulating_market_cap')
                    if mcap_val is not None and not np.isnan(mcap_val) and mcap_val >= MCAP_MIN:
                        mcap_ok.append(code)
        except:
            pass

    if len(mcap_ok) > 1500:
        mcap_ok = mcap_ok[:1500]
    return mcap_ok


# ==================== 数据加载 ====================
def load_price_data(stocks, context):
    n = len(stocks)
    end_dt = context.previous_date
    fields = ['close', 'high', 'low', 'volume']
    result = {f: {} for f in fields}

    for field in fields:
        is_vol = field in ('volume',)
        for b in range(0, n, 500):
            batch = stocks[b:b + 500]
            try:
                px = get_price(batch, end_date=end_dt, count=141,
                               fields=[field], skip_paused=False)
                if px is None or len(px) == 0:
                    continue
                if isinstance(px, dict):
                    for s in batch:
                        if s in px and px[s] is not None and len(px[s]) > 0:
                            dfc = px[s]
                            col = field if field in dfc.columns else dfc.columns[0]
                            ser = dfc[col].ffill() if not is_vol else dfc[col].fillna(0)
                            result[field][s] = np.array(ser.values, dtype=float)
                else:
                    df = px[field] if hasattr(px, 'minor_axis') else px
                    df = df.ffill() if not is_vol else df.fillna(0)
                    for s in batch:
                        if s in df.columns:
                            result[field][s] = np.array(df[s].values, dtype=float)
            except Exception as e:
                if g.debug:
                    log.error('[load] %s FAILED: %s' % (field, str(e)))
    return result


# ==================== 因子调度 ====================
def compute_factor(fname, stocks, context, price_data):
    n = len(stocks)
    if fname == 'volume_stability':
        return compute_volume_stability(stocks, price_data)
    elif fname == 'range_consistency':
        return compute_range_consistency(stocks, price_data)
    elif fname == 'atr_14':
        return compute_atr_14(stocks, price_data)
    elif fname == 'high_low_range':
        return compute_high_low_range(stocks, price_data)
    elif fname == 'ret_6m':
        return compute_ret_6m(stocks, price_data)
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

def compute_volume_stability(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    vd = price_data.get('volume', {})
    for i, s in enumerate(stocks):
        if s not in vd:
            continue
        v = np.array(vd[s], dtype=float)
        v_clean = v[np.isfinite(v) & (v > 0)]
        if len(v_clean) < 10:
            continue
        roll = v_clean[-20:] if len(v_clean) >= 20 else v_clean
        std_v = np.nanstd(roll)
        mean_v = np.nanmean(roll)
        if mean_v == 0:
            continue
        arr[i] = -std_v / mean_v
    valid = np.isfinite(arr)
    return arr, valid

def compute_range_consistency(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        min_len = min(len(h), len(l))
        if min_len < 10:
            continue
        rng = np.abs(h[-min_len:] - l[-min_len:])
        roll = rng[-20:] if len(rng) >= 20 else rng
        std_r = np.nanstd(roll)
        mean_r = np.nanmean(roll)
        if mean_r == 0:
            continue
        arr[i] = -std_r / mean_r
    valid = np.isfinite(arr)
    return arr, valid

def compute_atr_14(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {})
    ld = price_data.get('low', {})
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd:
            continue
        h = np.array(hd[s], dtype=float)
        l = np.array(ld[s], dtype=float)
        c = np.array(cd[s], dtype=float)
        if len(h) < 16 or len(l) < 16 or len(c) < 16:
            continue
        tr = np.zeros(len(c) - 1)
        for t in range(1, len(c)):
            tr[t-1] = max(h[t]-l[t], abs(h[t]-c[t-1]), abs(l[t]-c[t-1]))
        if len(tr) < 14:
            continue
        alpha = 2.0/15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1-alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid

def compute_high_low_range(stocks, price_data):
    n = len(stocks)
    arr = np.full(n, np.nan)
    hd = price_data.get('high', {}); ld = price_data.get('low', {}); cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in hd or s not in ld or s not in cd: continue
        h = hd[s][-1]; l = ld[s][-1]; c = cd[s][-1]
        if h is None or l is None or c is None or c == 0: continue
        arr[i] = -(h - l) / c
    valid = np.isfinite(arr)
    return arr, valid

n = len(stocks)
    arr = np.full(n, np.nan)
    cd = price_data.get('close', {})
    for i, s in enumerate(stocks):
        if s not in cd: continue
        c = np.array(cd[s], dtype=float)
        c_clean = c[np.isfinite(c)]
        if len(c_clean) < 131: continue
        arr[i] = c_clean[-1] / c_clean[-126] - 1.0
    valid = np.isfinite(arr)
    return arr, valid

# ==================== 工具函数 ====================
def zscore_numpy(arr, valid_mask):
    n = len(arr)
    result = np.full(n, -999.0)
    if np.sum(valid_mask) < 10:
        return result
    vals = arr[valid_mask]
    m = np.nanmedian(vals)
    s = np.nanstd(vals)
    if s > 0 and not np.isnan(s):
        z_vals = (vals - m) / s
        z_vals = np.clip(z_vals, -3, 3)
    else:
        z_vals = np.zeros(len(vals))
    result[valid_mask] = z_vals
    return result


def _isnan(x):
    try:
        return np.isnan(x)
    except:
        return False


# ==================== #12 补丁: 科创板/创业板保护限价 ====================
def board_limit_factors(code):
    if code.startswith(('688', '689')):
        return 1.2, 0.8
    if code.startswith(('300', '301')):
        return 1.2, 0.8
    if code.startswith(('8', '4')):
        return 1.3, 0.7
    return 1.1, 0.9


def order_target_with_limit(context, code, target_value, is_buy, prev_close=None):
    if target_value == 0:
        return order_target_value(code, 0)
    if not USE_PROTECTION_LIMIT:
        return order_target_value(code, target_value)
    limit_px = None
    if prev_close is not None and not (isinstance(prev_close, float) and np.isnan(prev_close))             and prev_close > 0:
        up, down = board_limit_factors(code)
        limit_px = prev_close * (up if is_buy else down)
    if limit_px is not None:
        try:
            return order_target_value(code, target_value, LimitOrderStyle(limit_px))
        except Exception:
            pass
    return order_target_value(code, target_value)


# ==================== 调仓 ====================
def rebalance(context, target):
    current_holdings = set(context.portfolio.positions.keys())
    target_set = set(target)
    to_sell = current_holdings - target_set

    all_codes = list(current_holdings | target_set)
    prev_close = {}
    if all_codes:
        try:
            pc = history(1, '1d', 'close', all_codes, skip_paused=False, df=False)
            for c in all_codes:
                if c in pc and len(pc[c]) > 0:
                    prev_close[c] = pc[c][-1]
        except Exception:
            prev_close = {}

    for s in to_sell:
        order_target_with_limit(context, s, 0, False, prev_close.get(s))

    n = len(target)
    if n == 0:
        return

    total_value = context.portfolio.total_value
    per_value = total_value / max(n, 1)

    turnover_rate = len(to_sell) / max(len(current_holdings), 1)
    if turnover_rate > g.max_turnover:
        max_new = int(n * g.max_turnover)
        target = target[:max_new]
        per_value = total_value / max(len(target), 1)

    for s in target:
        order_target_with_limit(context, s, per_value, True, prev_close.get(s))

    n_pos = len(context.portfolio.positions)
    invested = 0.0
    for p in context.portfolio.positions.values():
        try:
            invested += p.value
        except Exception:
            pass
    if g.debug:
        log.info('[FILLS] %s positions=%d invested=%.0f (%.1f%% of %.0f)' % (
            context.current_dt.strftime('%Y-%m-%d'), n_pos, invested,
            100.0 * invested / max(total_value, 1), total_value))

    g.last_holdings = target_set
