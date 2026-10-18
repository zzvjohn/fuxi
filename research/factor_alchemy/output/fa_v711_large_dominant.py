# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 JQ 自动生成 — fa_v711_large_dominant
================================================================
生成时间: 2026-07-20 19:29
生成器: jq_generator (P0.2)

因子组合:
  earnings_stability  (w=-0.513)   盈利同比增速波动小→盈利稳定→正向. JQ维护inc_net_profit_year_on_year近40周历史, 取s
  atr_14  (w=0.425)   auto-generated placeholder
  earnings_season_vol_div  (w=0.061)   中报窗口量价背离→机构吸筹→正向(价涨量缩的背离度)

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
    'earnings_stability': -0.5134,
    'atr_14': 0.4253,
    'earnings_season_vol_div': 0.0613
}
FACTOR_NAMES = ['earnings_stability', 'atr_14', 'earnings_season_vol_div']

# 数据加载参数
PRICE_FIELDS = ['close', 'high', 'low', 'volume']
PRICE_WINDOW = 35   # 最大回溯 + 5天余量

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
            n_valid = int(np.sum(valid))
            if n_valid == 0:
                # 全 NaN: 占位符/数据缺失/逻辑错误的强提醒, 不再静默
                log.error('[%s] STEP2 %s: valid=0/%d (0.0%%) <<< 因子完全塌缩! 检查 jq_compute_function 是否为占位符或价格数据缺失' % (
                    date_str, fname, n_stocks))
            elif g.debug:
                log.info('[%s] STEP2 %s: valid=%d/%d (%.1f%%)' % (
                    date_str, fname,
                    n_valid, n_stocks,
                    100 * n_valid / max(n_stocks, 1)))
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
                px = get_price(batch, end_date=end_dt, count=35,
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
    if fname == 'earnings_stability':
        return compute_earnings_stability(stocks, context)
    elif fname == 'atr_14':
        return compute_atr_14(stocks, price_data)
    elif fname == 'earnings_season_vol_div':
        return compute_earnings_season_vol_div(stocks, price_data)
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

def compute_earnings_stability(stocks, context):
    """盈利稳定性 = -std(inc_net_profit_year_on_year 历史). 波动小=稳定=正向.
    对齐本地 EarningsStability(-rolling(8q)std(netprofit_yoy)).
    预热: 首次调用用过去~10个季度末快照填充历史, 保证第1个交易日即有效."""
    import calendar
    n = len(stocks)
    arr = np.full(n, np.nan)
    if not hasattr(g, 'earnings_yoy_hist') or g.earnings_yoy_hist is None:
        g.earnings_yoy_hist = {}

    def _snap(dt):
        snap = {}
        try:
            end = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else dt
            q = query(indicator.code, indicator.inc_net_profit_year_on_year).filter(
                indicator.code.in_(stocks))
            df = get_fundamentals(q, date=end)
        except Exception as e:
            if g.debug:
                log.error('[earnings_stability] get_fundamentals FAILED: %s' % str(e))
            return snap
        if df is None or len(df) == 0:
            return snap
        for _, row in df.iterrows():
            code = row['code']
            val = row.get('inc_net_profit_year_on_year')
            if val is not None and np.isfinite(val):
                snap[code] = float(val)
        return snap

    # 首次预热: 过去~10个季度末快照
    if not getattr(g, 'earnings_yoy_warmed', False):
        end = context.previous_date
        y, m = end.year, end.month
        q = (m - 1) // 3
        qe_month = q * 3 + 3
        qe_y = y
        if qe_month > 12:
            qe_month -= 12; qe_y += 1
        ends = [datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1])]
        for _ in range(9):
            qe_month -= 3
            if qe_month < 1:
                qe_month += 12; qe_y -= 1
            ends.append(datetime.date(qe_y, qe_month, calendar.monthrange(qe_y, qe_month)[1]))
        for d in ends:
            s0 = _snap(d)
            for s in stocks:
                if s in s0:
                    h = g.earnings_yoy_hist.setdefault(s, [])
                    h.append(s0[s])
                    if len(h) > 40:
                        h[:] = h[-40:]
        g.earnings_yoy_warmed = True

    # 当前快照
    snap = _snap(context.previous_date)
    for s in stocks:
        if s in snap:
            h = g.earnings_yoy_hist.setdefault(s, [])
            h.append(snap[s])
            if len(h) > 40:
                h[:] = h[-40:]
    stock_idx = {s: i for i, s in enumerate(stocks)}
    valid = np.zeros(n, dtype=bool)
    for s in snap:
        if s in stock_idx:
            h = g.earnings_yoy_hist.get(s, [])
            hv = [x for x in h if x is not None and np.isfinite(x)]
            if len(hv) >= 4:
                arr[stock_idx[s]] = -np.std(hv, ddof=1)
                valid[stock_idx[s]] = True
    return arr, valid

def compute_atr_14(stocks, price_data):
    """ATR(14) normalized by close, negated. Low ATR = positive signal."""
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
            tr[t - 1] = max(
                h[t] - l[t],
                abs(h[t] - c[t - 1]),
                abs(l[t] - c[t - 1]))
        if len(tr) < 14:
            continue
        alpha = 2.0 / 15.0
        ema = tr[0]
        for t in range(1, len(tr)):
            ema = alpha * tr[t] + (1 - alpha) * ema
        c_now = c[-1]
        if c_now == 0 or np.isnan(c_now) or np.isnan(ema):
            continue
        arr[i] = -ema / c_now
    valid = np.isfinite(arr)
    return arr, valid

def compute_earnings_season_vol_div(stocks, price_data):
    """中报窗口量价背离度: 价变与量变秩相关取负(背离→正向), 5日平滑.

    对齐本地 EarningsSeasonVolDiv:
      pct = close.pct_change(); vol = volume.pct_change()
      5日窗口内 (pct, vol) 的 Spearman秩相关取负 → 价涨量缩(背离)为正
      取最近5个窗口的均值作为截面信号
    """
    n = len(stocks)
    arr = np.full(n, np.nan)
    close_d = price_data.get('close', {})
    vol_d = price_data.get('volume', {})
    win = 5
    for i, s in enumerate(stocks):
        if s not in close_d or s not in vol_d:
            continue
        c = np.asarray(close_d[s], dtype=float)
        v = np.asarray(vol_d[s], dtype=float)
        if len(c) < 15 or len(v) < 15:
            continue
        c = c[-20:]; v = v[-20:]
        pct = np.diff(c) / c[:-1]
        denom = np.where(v[:-1] != 0, v[:-1], np.nan)
        vc = np.diff(v) / denom
        pct = np.nan_to_num(pct)
        vc = np.nan_to_num(vc)
        L = len(pct)
        rho = np.full(L, np.nan)
        for t in range(win - 1, L):
            x = pct[t - win + 1:t + 1]
            y = vc[t - win + 1:t + 1]
            rx = x.argsort().argsort().astype(float)
            ry = y.argsort().argsort().astype(float)
            mr = rx.mean(); my = ry.mean()
            num = ((rx - mr) * (ry - my)).sum()
            den = np.sqrt(((rx - mr) ** 2).sum() * ((ry - my) ** 2).sum())
            rho[t] = (-num / den) if (den > 0 and np.isfinite(den)) else 0.0
        last5 = rho[-win:]
        val = np.nanmean(last5)
        if np.isfinite(val):
            arr[i] = val
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
