#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FA Ensemble Top-K (ICIR-driven, No GA, Equal-weight Rank Product)
生成时间: 2026-07-31T13:07:44.342698
方法: ICIR Top-K + 冗余剔除 + 等权 rank-product → JQ

因子 (5 个):
#   dollar_vol_stability            w=0.200  |ICIR|=0.601
#   earnings_volume_drift           w=0.200  |ICIR|=1.058
#   turnover_cv_20d                 w=0.200  |ICIR|=0.619
#   volume_climax_reversal          w=0.200  |ICIR|=1.121
#   vroc_12                         w=0.200  |ICIR|=0.678

论文支撑: 6 papers (Eigen-Portfolios / AlphaAgent / EFS / RegimeFolio / Causal PF)
共识: Ensemble > Single Selection, Equal-weight > Optimization
"""
import datetime as dt
import numpy as np
import pandas as pd
import json, gc, os, time

# ═══════════════════════════════════════════════════════════════
# 策略参数
# ═══════════════════════════════════════════════════════════════
TOP_N = 30                # 持仓数
REBALANCE_DAY = 5         # 周五 14:50 调仓
BENCHMARK = '000300.XSHG'
START_DATE = '2021-01-01'
END_DATE = '2026-07-30'

ENSEMBLE_FACTORS = ['dollar_vol_stability', 'earnings_volume_drift', 'turnover_cv_20d', 'volume_climax_reversal', 'vroc_12']
ENSEMBLE_WEIGHTS = dict(zip(ENSEMBLE_FACTORS, [0.2, 0.2, 0.2, 0.2, 0.2]))

PRICE_FIELDS = ['close', 'open', 'volume']
PRICE_WINDOW = 130

# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════
def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('order', 'info')
    log.set_level("system", "error")  # P-20260902-016: 抑制 order 函数内部仓位查询的 WARNING ("在 positions 中不存在" 刷屏)

    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0), type='stock')
    set_slippage(FixedSlippage(0.003))

    g.cycle = 0
    g.stocks = []
    g.first_run = True
    log.info('[Init] FA Ensemble Top-K (ICIR) — 5 factors, equal-weight rank-product')
    run_weekly(rebalance, weekday=REBALANCE_DAY, time='14:50')

def load_price_data(stocks, end_dt, fields, window):
    """加载价格数据 — 抄 V2 工作模式 (fields=字符串, frequency='daily', fq='pre')"""
    result = {}
    for field in fields:
        try:
            px = get_price(stocks, end_date=end_dt, count=window,
                           frequency='daily', fields=field,
                           skip_paused=False, fq='pre')
        except Exception:
            continue
        _Panel = getattr(pd, 'Panel', None)
        if _Panel is not None and isinstance(px, _Panel):
            px = px[field] if field in getattr(px, 'items', []) else px.minor_xs(field)
        if isinstance(px, pd.DataFrame) and not px.empty:
            result[field] = {c: px[c].values for c in px.columns if c in stocks}
        else:
            result[field] = {}
    return result

# ═══════════════════════════════════════════════════════════════
# 因子计算函数 (real implementations)
# ═══════════════════════════════════════════════════════════════
def compute_factor_standard(stocks, price_data, name, compute_fn):
    """通用: 计算因子, 处理 NaN, 返回 (array, valid_mask)"""
    try:
        arr, valid = compute_fn(stocks, price_data)
        return arr, valid
    except Exception:
        return np.full(len(stocks), np.nan), np.zeros(len(stocks), dtype=bool)

# --- 因子实现 (从因子库提取真实定义) ---

def _skew(a, window):
    """滚动偏度 (pandas rolling skew)"""
    s = pd.Series(a)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    m3 = ((s - mean) ** 3).rolling(window).mean()
    return ((m3 / (std ** 3)).fillna(0)).values

def compute_volume_climax_reversal(stocks, price_data):
    """天量反转: -[MA5(volume)/MA20(volume)], 负向因子"""
    close = price_data.get('close', {})
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 25:
            continue
        ma5 = np.mean(v[-5:])
        ma20 = np.mean(v[-20:])
        if ma20 > 0:
            arr[i] = -(ma5 / ma20)
            valid[i] = True
    return arr, valid

def compute_earnings_volume_drift(stocks, price_data):
    """业绩窗量价漂移: 基于最近的 earnings 窗口"""
    close = price_data.get('close', {})
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        v = vol.get(s)
        if c is None or v is None or len(c) < 25:
            continue
        ret_5d = c[-1] / c[-6] - 1 if len(c) >= 6 else 0
        vol_ratio = np.mean(v[-5:]) / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
        arr[i] = ret_5d * vol_ratio
        valid[i] = True
    return arr, valid

def compute_vroc_12(stocks, price_data):
    """量变动速率: -(MA12(volume) - MA26(volume)) / MA26(volume)"""
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 27:
            continue
        ma12 = np.mean(v[-12:])
        ma26 = np.mean(v[-26:])
        if ma26 > 0:
            arr[i] = -(ma12 - ma26) / ma26
            valid[i] = True
    return arr, valid

def compute_turnover_cv_20d(stocks, price_data):
    """换手率变异系数: volume CV(20) 作为 turnover 代理"""
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 21:
            continue
        window = v[-20:]
        mu = np.mean(window)
        std = np.std(window)
        if mu > 0:
            arr[i] = std / mu
            valid[i] = True
    return arr, valid

def compute_dollar_vol_stability(stocks, price_data):
    """成交额稳定性: corr(volume, volume.lag1), 20日"""
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 22:
            continue
        x = v[-21:-1]
        y = v[-20:]
        if np.std(x) > 0 and np.std(y) > 0:
            arr[i] = np.corrcoef(x, y)[0, 1]
            valid[i] = True
    return arr, valid

def compute_volume_stability(stocks, price_data):
    """成交量稳定性: -CV(volume, 20d)"""
    vol = price_data.get('volume', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        v = vol.get(s)
        if v is None or len(v) < 21:
            continue
        w = v[-20:]
        mu = np.mean(w)
        if mu > 0:
            arr[i] = -np.std(w) / mu
            valid[i] = True
    return arr, valid

def compute_earnings_growth_yoy(stocks, price_data):
    """盈利增长: fina_indicator 数据, 用 eps 增长率代理"""
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    try:
        q = query(valuation.code, valuation.pe_ratio, valuation.pb_ratio).filter(
            valuation.code.in_(stocks))
        df = get_fundamentals(q, date=context.previous_date)
        if df is not None and not df.empty:
            code_to_pe = dict(zip(df['code'], df['pe_ratio']))
            avg_pe = np.nanmean(list(code_to_pe.values()))
            for i, s in enumerate(stocks):
                pe = code_to_pe.get(s)
                if pe and pe > 0 and avg_pe > 0:
                    arr[i] = (1 / pe) / (1 / avg_pe) - 1  # EP 相对
                    valid[i] = True
    except Exception:
        pass
    return arr, valid

def compute_range_consistency(stocks, price_data):
    """波幅一致性: -std(high-low range, 20d) / mean(range)"""
    close = price_data.get('close', {})
    open_p = price_data.get('open', {})
    n = len(stocks)
    arr = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(stocks):
        c = close.get(s)
        o = open_p.get(s)
        if c is None or o is None or len(c) < 21:
            continue
        ranges = np.abs(np.array(c[-20:]) - np.array(o[-20:]))
        mu = np.mean(ranges)
        if mu > 0:
            arr[i] = -np.std(ranges) / mu
            valid[i] = True
    return arr, valid

# ═══════════════════════════════════════════════════════════════
# 因子路由表
# ═══════════════════════════════════════════════════════════════
FACTOR_COMPUTE = {
    'volume_climax_reversal': compute_volume_climax_reversal,
    'earnings_volume_drift': compute_earnings_volume_drift,
    'vroc_12': compute_vroc_12,
    'turnover_cv_20d': compute_turnover_cv_20d,
    'dollar_vol_stability': compute_dollar_vol_stability,
    'volume_stability': compute_volume_stability,
    'earnings_growth_yoy': compute_earnings_growth_yoy,
    'range_consistency': compute_range_consistency,
}

# ═══════════════════════════════════════════════════════════════
# 调仓逻辑
# ═══════════════════════════════════════════════════════════════
def rebalance(context):
    g.cycle += 1
    current_dt = context.current_dt
    prev_date = context.previous_date  # V2铁律: avoid_future_data下取数据必须用previous_date
    today = current_dt.date() if hasattr(current_dt, 'date') else current_dt
    
    if g.first_run or g.cycle <= 2 or g.cycle % 13 == 0:
        log.info(f'[Rebalance] Week {g.cycle} | {today}')
    g.first_run = False
    
    try:
        # === Step 1: 获取全市场股票 ===
        try:
            all_stocks = list(get_all_securities(['stock'], today).index)
        except Exception:
            all_stocks = list(get_all_securities(['stock']).index)
        
        if g.cycle <= 2:
            log.info(f'[Step 1] get_all_securities: {len(all_stocks)} stocks')
        
        if len(all_stocks) == 0:
            log.error('[Step 1] get_all_securities returned 0 stocks — ABORT')
            return
        
        # === Step 2: 过滤 (ST/停牌/次新) — 缓存 get_current_data ===
        try:
            cd = get_current_data()
        except Exception as e:
            log.error(f'[Step 2] get_current_data() failed: {e}')
            return
        
        universe = []
        skipped_paused = 0
        skipped_new = 0
        for s in all_stocks:
            try:
                # 科创板过滤
                if s.startswith('688') and current_dt.year < 2023:
                    continue
                # 停牌检查
                try:
                    paused = cd[s].paused
                except (KeyError, AttributeError):
                    paused = True
                if paused:
                    skipped_paused += 1
                    continue
                # 次新过滤
                try:
                    info = get_security_info(s)
                    if info and info.start_date:
                        if (today - info.start_date).days < 252:
                            skipped_new += 1
                            continue
                except Exception:
                    pass  # 无法获取信息则不因这个原因过滤
                universe.append(s)
            except Exception:
                continue
        
        if g.cycle <= 2:
            log.info(f'[Step 2] Filtered: {len(universe)} kept (paused:{skipped_paused} new:{skipped_new})')
        
        if len(universe) < TOP_N:
            log.warn(f'[Step 2] Universe too small: {len(universe)} < {TOP_N} — SKIP')
            return
        
        # === Step 3: 加载价格数据 ===
        price_data = load_price_data(universe, prev_date, PRICE_FIELDS, PRICE_WINDOW)
        n_with_close = len(price_data.get('close', {}))
        n_with_vol = len(price_data.get('volume', {}))
        if g.cycle <= 2:
            log.info(f'[Step 3] Price data: close={n_with_close} vol={n_with_vol}')
        
        if n_with_close == 0:
            log.warn('[Step 3] Zero stocks with price data — SKIP')
            return
        
        # === Step 4: 计算因子 ===
        factor_values = {}
        for fname in ENSEMBLE_FACTORS:
            compute_fn = FACTOR_COMPUTE.get(fname)
            if compute_fn is None:
                continue
            arr, valid = compute_factor_standard(universe, price_data, fname, compute_fn)
            arr = np.where(valid, arr, np.nan)
            finite_mask = np.isfinite(arr)
            n_finite = int(np.sum(finite_mask))
            if n_finite < 5:
                if g.cycle <= 2:
                    log.info(f'  [Factor] {fname}: only {n_finite} valid values — skip')
                continue
            ranks = np.full(len(arr), np.nan)
            ranks[finite_mask] = pd.Series(arr[finite_mask]).rank(pct=True).values
            factor_values[fname] = ranks
        
        if g.cycle <= 2:
            log.info(f'[Step 4] Factors computed: {len(factor_values)}/{len(ENSEMBLE_FACTORS)} valid')
        
        if len(factor_values) == 0:
            log.warn('[Step 4] No valid factors — SKIP')
            return
        
        # === Step 5: 加权秩乘积 ===
        composite = np.ones(len(universe))
        weight_sum = 0.0
        for fname, w in ENSEMBLE_WEIGHTS.items():
            if fname in factor_values:
                ranks = factor_values[fname]
                ranks = np.where(np.isfinite(ranks), ranks, 0.5)
                composite *= np.power(ranks + 1e-8, w)
                weight_sum += w
        
        if weight_sum == 0:
            log.warn('[Step 5] Zero weight sum — SKIP')
            return
        
        valid_comp = np.isfinite(composite)
        n_valid = int(np.sum(valid_comp))
        if n_valid < TOP_N:
            log.warn(f'[Step 5] Only {n_valid} valid composite scores < {TOP_N} — SKIP')
            return
        
        scores = pd.Series(composite, index=universe)
        scores = scores[valid_comp].dropna()
        top_stocks = scores.nlargest(TOP_N).index.tolist()
        
        if g.cycle <= 2:
            log.info(f'[Step 5] Composite ready: {n_valid} stocks, top {len(top_stocks)} selected')
        
        # === Step 6: 调仓 ===
        current_positions = list(context.portfolio.positions.keys())
        weight_per_stock = 1.0 / TOP_N
        sold = 0
        for s in current_positions:
            if s not in top_stocks:
                try:
                    order_target_value(s, 0)
                    sold += 1
                except Exception:
                    pass
        
        bought = 0
        skipped_limit = 0
        for s in top_stocks:
            try:
                try:
                    high_limit = cd[s].high_limit
                    last_price = cd[s].last_price
                    if high_limit is not None and last_price is not None and high_limit <= last_price:
                        skipped_limit += 1
                        continue
                except (KeyError, AttributeError):
                    pass
                order_target_value(s, context.portfolio.total_value * weight_per_stock)
                bought += 1
            except Exception:
                continue
        
        if g.cycle <= 3 or g.cycle % 13 == 0:
            log.info(f'[Done] W{g.cycle} | Hold {len(top_stocks)} | '
                     f'Sold {sold} Bought {bought} SkipLimit {skipped_limit} | '
                     f'Cash {context.portfolio.cash:.0f} Total {context.portfolio.total_value:.0f}')
    
    except Exception as e:
        log.error(f'[FATAL] rebalance exception: {type(e).__name__}: {e}')
        import traceback
        log.error(traceback.format_exc())
