"""
Build the regime-adaptive merged JQ strategy file.

Combines large_dominant and small_bull regimes into one adaptive strategy
that detects market regime at each rebalance and switches factor weights.
"""
import sys, os, re, textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jq_generator.registry import JQ_FACTOR_REGISTRY

# ── Weights ───────────────────────────────────────────
LARGE_DOMINANT_WEIGHTS = {'atr_14': 0.359, 'dollar_vol_20d': 0.286, 'bp': 0.217, 'ep': 0.138}
SMALL_BULL_WEIGHTS = {'volume_stability': 0.251, 'range_consistency': 0.212, 'atr_14': 0.198,
                      'high_low_range': 0.196, 'ret_6m': 0.143}

# All unique factors across both regimes
ALL_FACTORS = sorted(set(LARGE_DOMINANT_WEIGHTS.keys()) | set(SMALL_BULL_WEIGHTS.keys()))

# Which factors need context vs price_data
FUNDAMENTAL_FACTORS = {f for f in ALL_FACTORS
                       if JQ_FACTOR_REGISTRY[f]['jq_data'] in ('fundamental', 'fundamental_multi')}

# Max window and required fields
MAX_WINDOW = max(JQ_FACTOR_REGISTRY[f]['jq_window'] for f in ALL_FACTORS)
PRICE_FIELDS = sorted(set(
    field for f in ALL_FACTORS
    for field in JQ_FACTOR_REGISTRY[f]['jq_fields']
))

# ── Generate factor dispatch ──────────────────────────
dispatch_cases = []
for idx, fname in enumerate(sorted(ALL_FACTORS)):
    kw = "if" if idx == 0 else "elif"
    if fname in FUNDAMENTAL_FACTORS:
        call = f"compute_{fname}(stocks, context)"
    else:
        call = f"compute_{fname}(stocks, price_data)"
    dispatch_cases.append(f"    {kw} fname == '{fname}':\n        return {call}")
DISPATCH_BODY = "\n".join(dispatch_cases)

# ── Decode factor functions ───────────────────────────
def dedent_code(raw_code):
    """Normalize factor function indentation to standard 4-space."""
    lines = raw_code.strip().split('\n')
    # Remove common leading whitespace
    if not lines:
        return raw_code
    # Find minimum indent (excluding empty lines and def line)
    min_indent = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip())
        # def line defines the base
        if line.lstrip().startswith('def '):
            min_indent = leading
            break
    if min_indent is None:
        return raw_code
    # Remove min_indent from all non-empty lines
    result = []
    for line in lines:
        if line.strip():
            if len(line) >= min_indent:
                result.append(line[min_indent:])
            else:
                result.append(line.lstrip())
        else:
            result.append('')
    return '\n'.join(result)

factor_funcs = {}
for fname in ALL_FACTORS:
    raw = JQ_FACTOR_REGISTRY[fname]['jq_compute_function']
    factor_funcs[fname] = dedent_code(raw)

# ── Build the strategy file ───────────────────────────
def build_strategy():
    w_large_str = ",\n".join(f"        '{f}': {w:.4f}" for f, w in LARGE_DOMINANT_WEIGHTS.items())
    w_small_str = ",\n".join(f"        '{f}': {w:.4f}" for f, w in SMALL_BULL_WEIGHTS.items())
    all_factors_py = "[" + ", ".join(f"'{f}'" for f in sorted(ALL_FACTORS)) + "]"
    price_fields_py = str(PRICE_FIELDS)

    factor_funcs_block = "\n\n".join(factor_funcs[f] for f in sorted(ALL_FACTORS))

    code = f'''# -*- coding: utf-8 -*-
"""
================================================================
因子炼金术 — Regime 自适应复合策略
================================================================
根据市场状态 (large_dominant vs small_bull) 动态切换因子权重组合。

Regime 判定逻辑:
  - small_bull:  CSI500 4周收益 - CSI300 4周收益 > 2% AND 全A 4周收益 > -1%
  - large_dominant:  其他所有情况
  - 回退默认: large_dominant (保守取向)

large_dominant 因子组合 (价值/质量/低波):
  atr_14(0.359) + dollar_vol_20d(0.286) + bp(0.217) + ep(0.138)

small_bull 因子组合 (流动性/量价):
  volume_stability(0.251) + range_consistency(0.212) + atr_14(0.198)
  + high_low_range(0.196) + ret_6m(0.143)

两个组合回测 (2021+):
  large_dominant: +56.21%, Sharpe 0.27, MaxDD -25.12%
  small_bull:     +19.53%, Sharpe -0.04, MaxDD -24.69%, 2026单独-10%

参数:
  持仓数=30, 市值下限=5亿, 最大换手=60%
  调仓=周五 14:50, 基准=000300.XSHG
  保护限价=否 (直接市价单)

成本: QMT实盘 = 佣金万1 + 印花千1 + 滑点0.3%
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

# 两个 regime 的因子权重
LARGE_DOMINANT_WEIGHTS = {{
{w_large_str}
}}

SMALL_BULL_WEIGHTS = {{
{w_small_str}
}}

# 所有因子名 (含两个 regime 的并集)
ALL_FACTOR_NAMES = {all_factors_py}

# 数据加载参数
PRICE_FIELDS = {price_fields_py}
PRICE_WINDOW = {MAX_WINDOW + 5}   # 最大回溯 + 5天余量

# Regime 检测参数
REGIME_LOOKBACK_DAYS = 35   # ~5周交易日

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
    g.large_weights  = LARGE_DOMINANT_WEIGHTS
    g.small_weights  = SMALL_BULL_WEIGHTS
    g.all_factors    = ALL_FACTOR_NAMES
    g.price_fields   = PRICE_FIELDS
    g.price_window   = PRICE_WINDOW
    g.last_holdings  = set()
    g.regime_history = {{}}   # 记录历史regime用于调试
    g.debug          = DEBUG


# ==================== Regime 检测 ====================
def detect_regime(context):
    """
    实时检测当前市场 regime。
    
    方法: 取 CSI300 / CSI500 / CSI_ALL 近5周日线收盘,
         计算20日滚动收益 (≈4周), 比较风格价差和趋势。
    
    Returns:
        'small_bull'  — 小盘强势+上涨 (用 small_bull 权重)
        'large_dominant' — 大盘强势或其他 (用 large_dominant 权重, 默认)
    """
    indices = {{
        'CSI300':  '000300.XSHG',
        'CSI500':  '000905.XSHG',
        'CSI_ALL': '000985.XSHG',
    }}
    
    rets = {{}}
    for name, code in indices.items():
        try:
            px = get_price(
                code,
                end_date=context.previous_date,
                count=REGIME_LOOKBACK_DAYS,
                fields=['close'],
                skip_paused=False,
                frequency='daily'
            )
            if px is None:
                rets[name] = 0.0
                continue
            
            # 提取收盘价序列
            if isinstance(px, pd.DataFrame):
                closes = px['close'].values
            elif isinstance(px, dict):
                df = px[code] if code in px else None
                if df is not None and len(df) > 0:
                    col = 'close' if 'close' in df.columns else df.columns[0]
                    closes = df[col].values
                else:
                    rets[name] = 0.0
                    continue
            else:
                rets[name] = 0.0
                continue
            
            if len(closes) >= 21:
                rets[name] = closes[-1] / closes[-21] - 1.0
            else:
                rets[name] = 0.0
        except Exception:
            rets[name] = 0.0
    
    style_spread = rets['CSI500'] - rets['CSI300']
    trend = rets['CSI_ALL']
    
    # 分类: small_bull = 小盘跑赢且市场上涨
    if style_spread > 0.02 and trend > -0.01:
        return 'small_bull'
    else:
        return 'large_dominant'


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')

    # ── Step 0: 检测 Regime 并选择权重 ──
    regime = detect_regime(context)
    if regime == 'small_bull':
        active_weights = g.small_weights
    else:
        active_weights = g.large_weights
    
    active_factors = list(active_weights.keys())
    
    if g.debug:
        log.info('[%s] REGIME=%s factors=%s' % (date_str, regime, str(active_factors)))
    
    # ── Step 1: 股票池 ──
    stocks = get_stock_pool(context)
    n_stocks = len(stocks)
    if g.debug:
        log.info('[%s] STEP1 pool=%d stocks' % (date_str, n_stocks))
    if n_stocks < g.stock_num:
        return

    # ── Step 2: 加载数据 ──
    price_data = load_price_data(stocks, context)

    # ── Step 3: 计算所有因子 (计算全部因子以节省时间，只对active加权) ──
    factor_arrays = {{}}
    factor_valid = {{}}

    for fname in g.all_factors:
        try:
            arr, valid = compute_factor(fname, stocks, context, price_data)
            factor_arrays[fname] = arr
            factor_valid[fname] = valid
            if g.debug and fname in active_factors:
                log.info('[%s] STEP2 %s: valid=%d/%d (%.1f%%)' % (
                    date_str, fname,
                    np.sum(valid), n_stocks,
                    100 * np.sum(valid) / max(n_stocks, 1)))
        except Exception as e:
            log.error('[%s] STEP2 %s FAILED: %s' % (date_str, fname, str(e)))
            factor_arrays[fname] = np.full(n_stocks, np.nan)
            factor_valid[fname] = np.zeros(n_stocks, dtype=bool)

    # ── Step 4: 截面 z-score (仅 active factors) ──
    z_arrays = {{}}
    for fname in active_factors:
        arr = factor_arrays[fname]
        valid = factor_valid[fname]
        z_raw = rank_standardize(arr, valid)
        z_arrays[fname] = np.where(z_raw > -900, z_raw, np.nan)

    # ── Step 5: 加权合成 (NaN-safe, 逐股权重归一化) ──
    composite = np.zeros(n_stocks)
    wsum = np.zeros(n_stocks)
    for fname, w in active_weights.items():
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
        top_codes = [stocks[i] for i in top_idx if composite[i] > -900]
        log.info('[%s] STEP4 regime=%s composite: total=%d valid=%d top5=%s' % (
            date_str, regime, n_stocks, n_composite, str(top_codes)))

    if n_composite < g.stock_num:
        return

    # ── Step 6: 选股 (停牌/涨停过滤) ──
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
        log.info('[%s] STEP5 target=%d skip_pause=%d skip_limit=%d top3=%s' %
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
        current_data = {{}}

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
    fields = {price_fields_py}
    result = {{f: {{}} for f in fields}}

    for field in fields:
        is_vol = field in ('volume',)
        for b in range(0, n, 500):
            batch = stocks[b:b + 500]
            try:
                px = get_price(batch, end_date=end_dt, count={MAX_WINDOW + 5},
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
{DISPATCH_BODY}
    else:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)


# ==================== 因子实现 ====================
{factor_funcs_block}


# ==================== 工具函数 ====================
def rank_standardize(arr, valid_mask):
    """截面 rank-based: percentile → [-3, 3] 线性映射 (v6)"""
    n = len(arr)
    result = np.full(n, -999.0)
    V = int(np.sum(valid_mask))
    if V < 10:
        return result
    vals = arr[valid_mask]
    order = np.argsort(vals)
    ranks = np.empty(V)
    ranks[order] = np.arange(1, V + 1)
    pct = ranks / (V + 1)
    z = (pct - 0.5) * 6
    result[valid_mask] = z
    return result


def _isnan(x):
    try:
        return np.isnan(x)
    except:
        return False


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
    if prev_close is not None and not (isinstance(prev_close, float) and np.isnan(prev_close)) \\
            and prev_close > 0:
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
    prev_close = {{}}
    if all_codes:
        try:
            pc = history(1, '1d', 'close', all_codes, skip_paused=False, df=False)
            for c in all_codes:
                if c in pc and len(pc[c]) > 0:
                    prev_close[c] = pc[c][-1]
        except Exception:
            prev_close = {{}}

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
'''
    return code


# ── Write file ────────────────────────────────────────
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies', 'drafts',
                           'fa_v79_regime_adaptive.py')
output_path = os.path.normpath(output_path)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

code = build_strategy()

# Convert CRLF → LF, strip trailing whitespace
lines = code.split('\n')
cleaned = '\n'.join(line.rstrip() for line in lines)
if not cleaned.endswith('\n'):
    cleaned += '\n'

with open(output_path, 'w', encoding='utf-8', newline='') as f:
    f.write(cleaned)

print(f"Written: {output_path}")
print(f"Lines: {cleaned.count(chr(10))}")
print(f"Factors: {sorted(ALL_FACTORS)}")
