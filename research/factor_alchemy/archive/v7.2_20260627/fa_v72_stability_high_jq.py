# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 v7.2 — 高稳定 S↑ 策略 (聚宽版) [v3 numpy重写]
================================================================
来源: NSGA-II 三目标帕累托前沿 #3 (Q=0.202, S=1.000)
OOS (2025-2026.06): 年化 -26.8%, Sharpe -0.59, MaxDD -44.8%

因子组合 (等权加权):
  plrc_12             (w=0.341)  价格回归斜率:    趋势向上 → 动量延续 → 正向
  earnings_growth_yoy (w=0.266)  盈利同比增长率:  高增长 → 质量溢价 → 正向
  turnover_cv_20d     (w=0.200)  换手率变异系数:  低CV → 稳定 → 正向
  psy_12              (w=0.192)  心理线(反转):    低乐观 → 情绪修复 → 正向

★ v4 修复: NaN-safe 权重归一化 (参考 composite_factor_jq_v7_weekly_fast.py)
★ v4 修复: 缺失因子不参与加权, 按可用因子归一化 (不再用-999拖累composite)
★ v4 修复: 详细每步日志, 精确追踪 composite=0 根因

策略参数:
  - Top30 等权, 周频调仓 (每周五)
  - 全A股票池 (排除688/689/8/4, 市值>=10亿)
  - QMT实盘成本: 佣金万1 + 印花税千1 + 滑点0.3%
================================================================
"""
from __future__ import division
import numpy as np

# ==================== 策略参数 ====================
STOCK_NUM      = 30          # 持仓数量
MAX_TURNOVER   = 0.60        # 最大换手率限制
MCAP_MIN       = 10          # 最低市值(亿)
EXCLUDE_PREFIX = ('688', '689', '8', '4')  # 排除板块

# S↑ 因子权重 (帕累托前沿 #3)
WEIGHTS = {
    'plrc':           0.341,
    'earn_growth':    0.266,
    'turnover_cv':    0.200,
    'psy':            0.192,
}

# DEBUG开关
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

    g.stock_num      = STOCK_NUM
    g.max_turnover   = MAX_TURNOVER
    g.mcap_min       = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX
    g.weights        = WEIGHTS
    g.last_holdings  = set()
    g.debug          = DEBUG


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')

    # 1. 获取股票池
    stocks = get_stock_pool(context)
    n_stocks = len(stocks)
    if g.debug:
        log.info('[%s] STEP1 pool=%d stocks' % (date_str, n_stocks))

    if n_stocks < g.stock_num:
        log.warn('[%s] stock pool too small: %d < %d' % (date_str, n_stocks, g.stock_num))
        return

    # 2. 计算各因子 (numpy array)
    factor_arrays = {}
    factor_valid = {}

    for fname in ['plrc', 'earn_growth', 'turnover_cv', 'psy']:
        try:
            arr, valid = compute_factor(fname, stocks, context)
            factor_arrays[fname] = arr
            factor_valid[fname] = valid
            if g.debug:
                log.info('[%s] STEP2 %s: raw=%d nan=%d valid=%d (%.1f%%)' % (
                    date_str, fname, len(arr), np.sum(np.isnan(arr)),
                    np.sum(valid), 100 * np.sum(valid) / max(n_stocks, 1)))
        except Exception as e:
            log.error('[%s] STEP2 %s FAILED: %s' % (date_str, fname, str(e)))
            factor_arrays[fname] = np.full(n_stocks, np.nan)
            factor_valid[fname] = np.zeros(n_stocks, dtype=bool)

    # 3. 截面 z-score (先用-999标记, 然后转NaN)
    z_arrays = {}
    for fname in ['plrc', 'earn_growth', 'turnover_cv', 'psy']:
        arr = factor_arrays[fname]
        valid = factor_valid[fname]
        z_raw = zscore_numpy(arr, valid)
        n_before = np.sum(z_raw > -900)
        # ★ v4: 将 -999 转为 NaN, 缺失因子不参与加权
        z_arrays[fname] = np.where(z_raw > -900, z_raw, np.nan)
        if g.debug:
            log.info('[%s] STEP3 %s zscore: valid=%d/%d (nan=%d)' % (
                date_str, fname, n_before, n_stocks,
                np.sum(np.isnan(z_arrays[fname]))))

    # 4. 加权合成 (NaN-safe, 按可用因子归一化)
    composite = np.zeros(n_stocks)
    wsum = np.zeros(n_stocks)
    for fname, w in g.weights.items():
        z = z_arrays[fname]
        mask = np.isfinite(z)
        composite[mask] += w * z[mask]
        wsum[mask] += abs(w)

    mask_valid = wsum > 0
    composite[mask_valid] = composite[mask_valid] / wsum[mask_valid]
    # 所有因子都缺失 → 极低分
    composite[~mask_valid] = -999.0
    n_composite = np.sum(mask_valid)

    if g.debug:
        # 打印前5的composite
        order = np.argsort(composite)[::-1]
        top_n = min(5, n_composite)
        top_idx = order[:top_n]
        log.info('[%s] STEP4 composite: total=%d valid=%d top5_val=%s top5=%s' % (
            date_str, n_stocks, n_composite,
            str(np.round(composite[top_idx][:top_n], 4)),
            str([stocks[i] for i in top_idx[:min(3, top_n)]])))

    if n_composite < g.stock_num:
        for fname in g.weights:
            n_valid_f = np.sum(np.isfinite(z_arrays[fname]))
            log.warn('[%s] DIAG %s: %d/%d have valid zscore' % (
                date_str, fname, n_valid_f, n_stocks))
        log.warn('[%s] composite too few: %d < %d (all factors missing for %d stocks)' % (
            date_str, n_composite, g.stock_num, n_stocks - n_composite))
        return

    # 5. 选Top30
    sorted_idx = np.argsort(composite)[::-1]
    target = []
    for idx in sorted_idx:
        if composite[idx] > -900 and len(target) < g.stock_num:
            target.append(stocks[idx])

    if g.debug:
        log.info('[%s] STEP5 target=%d stocks, top3: %s' % (date_str, len(target), str(target[:3])))

    if len(target) < g.stock_num:
        log.warn('[%s] target too few after filter: %d' % (date_str, len(target)))
        return

    # 6. 调仓
    rebalance(context, target)


# ==================== 股票池 ====================
def get_stock_pool(context):
    prev = context.previous_date
    current_date = context.current_dt

    try:
        all_sec = get_all_securities(['stock'], date=prev)
        all_stocks = list(all_sec.index)
    except:
        all_sec = get_all_securities(['stock'])
        all_stocks = list(all_sec.index)

    if g.debug:
        log.info('[Pool] step1 all_sec=%d' % len(all_stocks))

    stocks = [s for s in all_stocks if not s.startswith(g.exclude_prefix)]
    if g.debug:
        log.info('[Pool] step2 prefix=%d' % len(stocks))

    try:
        current_data = get_current_data()
    except:
        current_data = {}

    excluded_st = 0; excluded_paused = 0; excluded_ipo = 0
    filtered = []
    for s in stocks:
        try:
            if s in current_data:
                cd = current_data[s]
                if cd.is_st or (cd.name and 'ST' in str(cd.name)):
                    excluded_st += 1; continue
                if cd.paused:
                    excluded_paused += 1; continue
        except:
            pass
        try:
            start_date = all_sec.loc[s, 'start_date']
            if start_date is not None:
                if not isinstance(start_date, pd.Timestamp):
                    start_date = pd.Timestamp(start_date)
                if (current_date - start_date).days < 180:
                    excluded_ipo += 1; continue
        except:
            pass
        filtered.append(s)

    if g.debug:
        log.info('[Pool] step3 ST=%d paused=%d IPO=%d → %d' % (
            excluded_st, excluded_paused, excluded_ipo, len(filtered)))

    if len(filtered) == 0:
        return []

    date_str = prev.strftime('%Y-%m-%d')
    mcap_ok = []; mcap_nan = 0; mcap_small = 0

    for i in range(0, len(filtered), 2000):
        batch = filtered[i:i + 2000]
        try:
            q = query(valuation.code, valuation.circulating_market_cap
                     ).filter(valuation.code.in_(batch))
            df = get_fundamentals(q, date=date_str)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    code = row['code']
                    mcap_val = row.get('circulating_market_cap')
                    if mcap_val is None or np.isnan(mcap_val):
                        mcap_nan += 1
                    elif mcap_val >= g.mcap_min:
                        mcap_ok.append(code)
                    else:
                        mcap_small += 1
            else:
                mcap_nan += len(batch)
        except Exception as e:
            log.warn('[Pool] fund batch %d FAILED: %s' % (i, str(e)))
            mcap_nan += len(batch)

    if g.debug:
        log.info('[Pool] step4 mcap>=%dE: %d (nan=%d small=%d)' % (
            MCAP_MIN, len(mcap_ok), mcap_nan, mcap_small))

    if len(mcap_ok) > 1500:
        mcap_ok = mcap_ok[:1500]

    return mcap_ok


# ==================== 因子计算 ====================

def compute_factor(fname, stocks, context):
    n = len(stocks)
    if fname == 'plrc':
        return compute_plrc_12(stocks, context)
    elif fname == 'earn_growth':
        return compute_earnings_growth(stocks, context)
    elif fname == 'turnover_cv':
        return compute_turnover_cv(stocks, context)
    elif fname == 'psy':
        return compute_psy_12(stocks, context)
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)


def compute_plrc_12(stocks, context):
    """
    plrc_12: 60日价格线性回归斜率 (约12周)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel = history(65, unit='1d', field='close',
                        security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[plrc] history() FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    x = np.arange(1, 61, dtype=float)
    x_mean = np.mean(x)
    x_demean = x - x_mean
    sum_sq_x = np.sum(x_demean ** 2)

    for i, s in enumerate(stocks):
        if s not in panel or len(panel[s]) < 50:
            continue
        closes = np.array(panel[s])[-60:]
        closes = closes[~np.isnan(closes)]
        if len(closes) < 50:
            continue
        y_mean = np.mean(closes)
        if y_mean == 0:
            continue
        y = closes / y_mean
        y_demean = y - np.mean(y)
        arr[i] = np.sum(x_demean * y_demean) / sum_sq_x

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_earnings_growth(stocks, context):
    """
    归属母公司净利润同比增长率 (季度更新)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        q = query(indicator.code, indicator.inc_net_profit_year_on_year)
        df = get_fundamentals(q, date=context.previous_date)
    except Exception as e:
        log.error('[eg] get_fundamentals FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    if df is None or len(df) == 0:
        return arr, np.zeros(n, dtype=bool)

    # Build code→index mapping for fast lookup
    stock_idx = {s: i for i, s in enumerate(stocks)}
    for _, row in df.iterrows():
        code = row['code']
        if code not in stock_idx:
            continue
        growth = row['inc_net_profit_year_on_year']
        if growth is not None and not np.isnan(growth):
            arr[stock_idx[code]] = np.clip(growth, -200, 500)

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_turnover_cv(stocks, context):
    """
    换手率变异系数: -std(turnover_20d) / mean(turnover_20d)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel = history(25, unit='1d', field='money',
                        security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[tcv] history(money) FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    try:
        q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(stocks))
        df_funda = get_fundamentals(q, date=context.previous_date)
        mcap_map = {}
        if df_funda is not None and len(df_funda) > 0:
            for _, row in df_funda.iterrows():
                code = row['code']
                mcap = row['market_cap']
                if mcap and mcap > 0:
                    mcap_map[code] = mcap
    except Exception as e:
        log.error('[tcv] get_fundamentals FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if s not in panel or s not in mcap_map:
            continue
        col = pd.Series(panel[s]).dropna()
        if len(col) < 10:
            continue
        mcap_yuan = mcap_map[s] * 1e8
        to_vals = col.values[-20:] / mcap_yuan
        mean_val = np.mean(to_vals)
        if mean_val == 0:
            continue
        arr[i] = -np.std(to_vals) / mean_val

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_psy_12(stocks, context):
    """
    心理线: -count(close>prev_close, 12d) / 12
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel = history(15, unit='1d', field='close',
                        security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[psy] history() FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if s not in panel or len(panel[s]) < 13:
            continue
        closes = np.array(panel[s])[-13:]
        closes = closes[~np.isnan(closes)]
        if len(closes) < 13:
            continue
        up_count = np.sum(closes[1:] > closes[:-1])
        arr[i] = -(up_count / 12.0)

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


# ==================== 工具函数 ====================

def zscore_numpy(arr, valid_mask):
    """
    截面 z-score + 缩尾[-3,3] + 缺失值= -999
    参考 daily_ga_composite_jq.py 已验证模式
    """
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


def rebalance(context, target):
    """等权调仓"""
    current_holdings = set(context.portfolio.positions.keys())
    target_set = set(target)

    to_sell = current_holdings - target_set
    for s in to_sell:
        order_target_value(s, 0)

    n = len(target)
    if n == 0:
        return
    total_value = context.portfolio.total_value
    per_value = total_value / max(n, 1)

    turnover = len(to_sell) / max(len(current_holdings), 1)
    if turnover > g.max_turnover:
        max_new = int(n * g.max_turnover)
        target = target[:max_new]
        per_value = total_value / max(len(target), 1)

    for s in target:
        if s in context.portfolio.positions:
            order_target_value(s, per_value)
        else:
            order_value(s, per_value)

    g.last_holdings = target_set
