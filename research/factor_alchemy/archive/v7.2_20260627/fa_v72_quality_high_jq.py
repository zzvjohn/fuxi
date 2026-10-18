# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 v7.2 — 高质量 Q↑ 策略 (聚宽版) [v3 numpy重写]
================================================================
来源: NSGA-II 三目标帕累托前沿 #5 (Q=0.331, S=0.181)
OOS (2025-2026.06): 年化 +11.0%, Sharpe 0.57, MaxDD -32.7%

因子组合 (等权加权):
  turnover_cv_20d     (w=0.335)  换手率变异系数:  低CV → 流动性稳定 → 正向
  dollar_vol_20d      (w=0.327)  日均成交额(对数):  低成交额 → 流动性溢价 → 正向
  davol_20            (w=0.170)  量比异动(20/120):  低量比 → 调整到位 → 正向
  abnormal_turnover   (w=0.168)  异常换手:          低异常 → 无操纵 → 正向

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

# Q↑ 因子权重 (帕累托前沿 #5)
WEIGHTS = {
    'turnover_cv':    0.335,
    'dollar_vol':     0.327,
    'davol':          0.170,
    'abnormal_to':    0.168,
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

    for fname in ['turnover_cv', 'dollar_vol', 'davol', 'abnormal_to']:
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
    for fname in ['turnover_cv', 'dollar_vol', 'davol', 'abnormal_to']:
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

    # Step 1: 全市场
    try:
        all_sec = get_all_securities(['stock'], date=prev)
        all_stocks = list(all_sec.index)
    except:
        all_sec = get_all_securities(['stock'])
        all_stocks = list(all_sec.index)

    if g.debug:
        log.info('[Pool] step1 all_sec=%d' % len(all_stocks))

    # Step 2: 排除板块
    stocks = [s for s in all_stocks if not s.startswith(g.exclude_prefix)]
    if g.debug:
        log.info('[Pool] step2 prefix=%d' % len(stocks))

    # Step 3: 排除ST/停牌/新股
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

    # Step 4: 市值过滤
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

    # Step 5: 限制数量
    if len(mcap_ok) > 1500:
        mcap_ok = mcap_ok[:1500]

    return mcap_ok


# ==================== 因子计算 ====================

def compute_factor(fname, stocks, context):
    """统一因子计算入口, 返回 (array, valid_mask)"""
    n = len(stocks)
    if fname == 'turnover_cv':
        return compute_turnover_cv(stocks, context)
    elif fname == 'dollar_vol':
        return compute_dollar_vol(stocks, context)
    elif fname == 'davol':
        return compute_davol(stocks, context)
    elif fname == 'abnormal_to':
        return compute_abnormal_turnover(stocks, context)
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)


def compute_turnover_cv(stocks, context):
    """
    换手率变异系数: -std(turnover_20d) / mean(turnover_20d)
    用 money/market_cap 近似换手率
    ★ v3: 返回 (np.array, valid_mask)
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
        log.error('[tcv] get_fundamentals(market_cap) FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if s not in panel or s not in mcap_map:
            continue
        col = pd.Series(panel[s]).dropna()
        if len(col) < 9:
            continue
        mcap_yuan = mcap_map[s] * 1e8
        to_vals = col.values[-20:] / mcap_yuan
        mean_val = np.mean(to_vals)
        if mean_val == 0:
            continue
        arr[i] = -np.std(to_vals) / mean_val

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_dollar_vol(stocks, context):
    """
    日均成交额(对数取负): -log(avg(close*volume, 20d))
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel_close = history(25, unit='1d', field='close',
                              security_list=stocks, skip_paused=True, df=False)
        panel_vol = history(25, unit='1d', field='volume',
                            security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[dv] history FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if s not in panel_close or s not in panel_vol:
            continue
        if len(panel_close[s]) < 15 or len(panel_vol[s]) < 15:
            continue
        close_vals = np.array(panel_close[s])[-20:]
        vol_vals = np.array(panel_vol[s])[-20:]
        dv = close_vals * vol_vals
        dv_mean = np.mean(dv[~np.isnan(dv)])
        if dv_mean <= 0 or np.isnan(dv_mean):
            continue
        arr[i] = -np.log(dv_mean)

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_davol(stocks, context):
    """
    量比异动: -(20日/120日成交额均值比)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel_close = history(130, unit='1d', field='close',
                              security_list=stocks, skip_paused=True, df=False)
        panel_vol = history(130, unit='1d', field='volume',
                            security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[davol] history FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    for i, s in enumerate(stocks):
        if s not in panel_close or s not in panel_vol:
            continue
        if len(panel_close[s]) < 110 or len(panel_vol[s]) < 110:
            continue
        close_vals = np.array(panel_close[s])
        vol_vals = np.array(panel_vol[s])
        money = close_vals * vol_vals
        short_ma = np.mean(money[-20:][~np.isnan(money[-20:])])
        long_ma = np.mean(money[-120:][~np.isnan(money[-120:])])
        if long_ma == 0 or np.isnan(short_ma) or np.isnan(long_ma):
            continue
        ratio = short_ma / long_ma
        arr[i] = -np.clip(ratio, 0, 10)

    valid = np.isfinite(arr) & ~np.isnan(arr)
    return arr, valid


def compute_abnormal_turnover(stocks, context):
    """
    异常换手率 (截面去均值版)
    """
    n = len(stocks)
    arr = np.full(n, np.nan)

    try:
        panel = history(25, unit='1d', field='money',
                        security_list=stocks, skip_paused=True, df=False)
    except Exception as e:
        log.error('[ato] history(money) FAILED: %s' % str(e))
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
        log.error('[ato] get_fundamentals FAILED: %s' % str(e))
        return arr, np.zeros(n, dtype=bool)

    raw_to = np.full(n, np.nan)
    for i, s in enumerate(stocks):
        if s not in panel or s not in mcap_map:
            continue
        col = pd.Series(panel[s]).dropna()
        if len(col) < 10:
            continue
        mcap_yuan = mcap_map[s] * 1e8
        to_vals = col.values[-20:] / mcap_yuan
        raw_to[i] = np.mean(to_vals)

    valid_idx = np.where(np.isfinite(raw_to))[0]
    if len(valid_idx) < 30:
        return arr, np.zeros(n, dtype=bool)

    mean_to = np.mean(raw_to[valid_idx])
    std_to = np.std(raw_to[valid_idx])
    if std_to == 0 or np.isnan(std_to):
        return arr, np.zeros(n, dtype=bool)

    arr = np.full(n, np.nan)
    arr[valid_idx] = -((raw_to[valid_idx] - mean_to) / std_to)

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
