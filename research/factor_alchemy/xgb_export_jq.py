"""Retrain XGBoost on full 2021-2024 data, export model + generate JQ code."""
import pickle, numpy as np, pandas as pd, json, time
from pathlib import Path
import xgboost as xgb

OUT = Path(__file__).parent / 'output'

# 1. Load data
print("[1] Loading data...", flush=True)
with open(OUT / 'xgb_feature_matrix.pkl', 'rb') as f:
    feature_df = pickle.load(f)
with open(OUT / 'factor_cache.pkl', 'rb') as f:
    cache_ret1m = pickle.load(f)['ret_1m']
weekly_dates = cache_ret1m.index

# Load target (recompute from daily close)
DATA_CSV = Path(__file__).parent.parent.parent / 'data' / 'backtest_universe'
csv_files = list(DATA_CSV.glob('*.csv'))
csv_codes = {f.stem: f for f in csv_files}
matched = [s for s in cache_ret1m.columns if s in csv_codes]
daily_close = {}
for code in matched:
    df = pd.read_csv(csv_codes[code], parse_dates=['trade_date'], usecols=['trade_date','close'])
    daily_close[code] = df.sort_values('trade_date').set_index('trade_date')['close']

print("[2] Computing targets...", flush=True)
targets = {}
t0 = time.time()
for ci, code in enumerate(matched):
    s = daily_close[code]
    for wd in weekly_dates:
        wd_pd = pd.Timestamp(wd)
        before = s[s.index <= wd_pd]
        if len(before) == 0: continue
        after = s[(s.index > wd_pd) & (s.index <= wd_pd + pd.Timedelta(days=7))]
        if len(after) == 0: continue
        targets[(wd_pd, code)] = (after.iloc[-1] / before.iloc[-1]) - 1.0
    if (ci+1) % 200 == 0:
        print(f'  {ci+1}/{len(matched)} ({time.time()-t0:.0f}s)', flush=True)

target_s = pd.Series(targets)
target_s.index.names = ['date','code']

# Align
common = feature_df.index.intersection(target_s.index)
X = feature_df.loc[common].fillna(0).values.astype(np.float32)
y_raw = target_s.loc[common]
y = y_raw.clip(y_raw.quantile(0.01), y_raw.quantile(0.99)).values.astype(np.float32)

# Split: train on 2021-2024 for JQ export
idx_dates = feature_df.loc[common].index.get_level_values('date')
train_mask = idx_dates <= pd.Timestamp('2024-12-31')
X_train, y_train = X[train_mask], y[train_mask]
print(f"  Train: {X_train.shape[0]} rows", flush=True)

# 3. Train with simpler params for smaller model
print("[3] Training XGBoost (full 2021-2024)...", flush=True)
model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.08, max_depth=3,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=15, n_estimators=100, random_state=42, n_jobs=4, tree_method='hist',
)
model.fit(X_train, y_train, verbose=False)
print(f"  Done. Trees: {model.n_estimators}", flush=True)

# 4. Export model as JSON
print("[4] Exporting model...", flush=True)
model_json = model.get_booster().get_dump(dump_format='json')
# Parse trees: XGBoost JSON dump format → flat list of nodes
compact_trees = []
for tree_str in model_json:
    root = json.loads(tree_str)  # nested dict with 'children'

    nodes = []  # flat list
    def flatten(node_dict):
        idx = len(nodes)
        if 'leaf' in node_dict:
            nodes.append({'leaf': float(node_dict['leaf'])})
        else:
            nodes.append({
                'feat': int(node_dict['split'][1:]),  # "f10" → 10
                'thresh': float(node_dict['split_condition']),
                'left': -1,  # fill after children are flattened
                'right': -1,
            })
            left_idx = flatten(node_dict['children'][0])
            right_idx = flatten(node_dict['children'][1])
            nodes[idx]['left'] = left_idx
            nodes[idx]['right'] = right_idx
        return idx

    flatten(root)
    compact_trees.append(nodes)

print(f"  Total nodes: {sum(len(t) for t in compact_trees)}", flush=True)

# 5. Generate JQ code
print("[5] Generating JQ code...", flush=True)

feature_names = list(feature_df.columns)
compact_json = json.dumps(compact_trees, separators=(',', ':'))

# Estimate embedded JSON size
json_size = len(compact_json)
print(f"  Model JSON size: {json_size} bytes ({json_size/1024:.1f} KB)", flush=True)

jq_code = f'''# -*- coding: utf-8 -*-
"""
FA Composite XGB Alpha — XGBoost 残差信号 × turnover_change | JQ 真回测
==========================================================================
xgb_alpha: 19个原始基元特征 → XGBoost(100树×深度3) → 纯 ML 信号
复合: rank(xgb_alpha) × rank(turnover_change)  本地 ICIR=3.43 (2025-2026)

原始基元 (全部从 OHLCV 计算, 无依赖任何已有因子):
  收益: r_1d, r_5d, r_10d, r_20d, r_40d, r_60d
  波动率: vol_5d, vol_10d, vol_20d
  最大回撤: maxdd_5d, maxdd_10d, maxdd_20d
  量比: vratio_5d, vratio_20d
  振幅: amp_5d, amp_20d
  涨跌比: up_ratio
  连涨连跌: streak
  跳空: gap_avg

参数: 持仓=30, 周频(周四14:50), 基准=沪深300
成本: 佣金万1 + 印花千1 + 滑点0.3%
========================================================================
"""

from __future__ import division
import datetime
import gc
import numpy as np
import pandas as pd
import json

# ==================== 策略参数 ====================
STOCK_NUM      = 30
MAX_TURNOVER   = 0.60
MCAP_MIN       = 5
EXCLUDE_PREFIX = ('8', '4')
CHUNK          = 500
LOOKBACK       = 65
MIN_BARS       = 20

# ==================== XGBoost 模型 (嵌入式) ====================
_XGB_TREES = json.loads(\'\'\'{compact_json}\'\'\')

def _xgb_predict(feats):
    """单样本 XGBoost 推理. feats: 19维 numpy array."""
    score = 0.0
    for tree_nodes in _XGB_TREES:
        node_idx = 0
        while True:
            node = tree_nodes[node_idx]
            if 'leaf' in node:
                score += node['leaf']
                break
            if feats[node['feat']] < node['thresh']:
                node_idx = node['left']
            else:
                node_idx = node['right']
    return score

def _xgb_predict_batch(feats_matrix):
    """批量推理. feats_matrix: (n_stocks, 19)."""
    return np.array([_xgb_predict(feats_matrix[i]) for i in range(len(feats_matrix))])


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        if hasattr(x, '__iter__'):
            xs = list(x)
            return float(xs[0]) if xs else 0.0
        return 0.0


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0001, close_commission=0.0001,
        close_today_commission=0, min_commission=0,
    ), type='stock')
    set_slippage(FixedSlippage(0.003))

    run_weekly(main, weekday=4, time='14:50')

    g.stock_num     = STOCK_NUM
    g.max_turnover  = MAX_TURNOVER
    g.mcap_min      = MCAP_MIN
    g.exclude_prefix = EXCLUDE_PREFIX
    g.debug         = True


# ==================== 股票池 ====================
def get_stock_pool(context):
    date = context.previous_date
    try:
        all_stocks = get_index_stocks('000985.XSHG', date=date)
    except Exception:
        all_stocks = list(get_all_securities(types=['stock'], date=date).index)

    stocks = [s for s in all_stocks if not s.startswith(g.exclude_prefix)]
    if len(stocks) == 0:
        return []

    q = query(valuation.code, valuation.market_cap).filter(
        valuation.code.in_(stocks), valuation.market_cap >= g.mcap_min)
    df = get_fundamentals(q, date=context.previous_date)
    stocks = list(df['code']) if not df.empty else []
    if len(stocks) == 0:
        return []

    cd = get_current_data()
    final = []
    for s in stocks:
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        final.append(s)
    return final


# ==================== 价格加载 ====================
def _get_price_field(stocks, end_dt, count, field):
    px = get_price(stocks, end_date=end_dt, count=count,
                   frequency='daily', fields=field,
                   skip_paused=False, fq='pre')
    _Panel = getattr(pd, 'Panel', None)
    if _Panel is not None and isinstance(px, _Panel):
        px = px[field] if field in getattr(px, 'items', []) else px.minor_xs(field)
    return px


# ==================== 原始基元 + XGBoost 推理 ====================
def compute_xgb_alpha(open_df, high_df, low_df, close_df, volume_df):
    """计算 19 个原始基元特征 → XGBoost 推理 → xgb_alpha 标量."""
    n_stocks = close_df.shape[1]
    if n_stocks == 0:
        return pd.Series(dtype=float)

    # Ensure enough data
    if close_df.shape[0] < MIN_BARS:
        return pd.Series(index=close_df.columns, dtype=float)

    close_arr = close_df.values
    open_arr = open_df.values
    high_arr = high_df.values
    low_arr = low_df.values
    volume_arr = volume_df.values

    n_rows, n_cols = close_arr.shape
    feats = np.zeros((n_cols, 19), dtype=np.float32)
    feats.fill(np.nan)

    # r_1d ~ r_60d (收益基元)
    for i, h in enumerate([1, 5, 10, 20, 40, 60]):
        if n_rows > h:
            feats[:, i] = close_arr[-1] / close_arr[-(h+1)] - 1.0
        else:
            feats[:, i] = np.nan

    # Daily returns
    daily_ret = np.diff(close_arr, axis=0) / (close_arr[:-1] + 1e-12)
    n_dr = daily_ret.shape[0]

    # vol_5d, vol_10d, vol_20d
    for j, h in enumerate([5, 10, 20]):
        if n_dr >= h:
            feats[:, 6+j] = np.nanstd(daily_ret[-h:], axis=0)
        else:
            feats[:, 6+j] = np.nan

    # maxdd_5d, maxdd_10d, maxdd_20d
    for j, h in enumerate([5, 10, 20]):
        if n_rows >= h:
            window = close_arr[-h:]
            peak = np.maximum.accumulate(window, axis=0)
            dd = (window[-1] - peak[-1]) / (peak[-1] + 1e-12)
            feats[:, 9+j] = dd
        else:
            feats[:, 9+j] = np.nan

    # vratio_5d, vratio_20d
    for j, h in enumerate([5, 20]):
        if n_rows >= h:
            feats[:, 12+j] = volume_arr[-1] / (np.nanmean(volume_arr[-h:], axis=0) + 1e-12)
        else:
            feats[:, 12+j] = np.nan

    # amp_5d, amp_20d
    for j, h in enumerate([5, 20]):
        if n_rows >= h:
            hl = high_arr[-h:] - low_arr[-h:]
            feats[:, 14+j] = np.nanmean(hl / (close_arr[-h:] + 1e-12), axis=0)
        else:
            feats[:, 14+j] = np.nan

    # up_ratio
    if n_dr >= 20:
        feats[:, 16] = np.nanmean(daily_ret[-20:] > 0, axis=0)
    else:
        feats[:, 16] = np.nan

    # streak (连续涨跌天数)
    if n_dr >= 10:
        signs = np.sign(daily_ret[-10:])
        feats[:, 17] = 0.0
        for s_idx in range(n_cols):
            col_signs = signs[:, s_idx]
            last = col_signs[-1]
            if last == 0:
                feats[s_idx, 17] = 0.0
            else:
                cnt = 0
                for k in range(len(col_signs)-1, -1, -1):
                    if col_signs[k] == last:
                        cnt += 1
                    else:
                        break
                feats[s_idx, 17] = cnt if last > 0 else -cnt
    else:
        feats[:, 17] = np.nan

    # gap_avg (5日开盘跳空均值)
    if n_rows >= 6:
        gaps = open_arr[-5:] / close_arr[-6:-1] - 1.0
        feats[:, 18] = np.nanmean(gaps, axis=0)
    else:
        feats[:, 18] = np.nan

    # Fill NaN with 0 (matching training behavior)
    feats = np.nan_to_num(feats)

    # XGBoost inference
    scores = _xgb_predict_batch(feats)

    return pd.Series(scores, index=close_df.columns)


# ==================== 截面 rank % ====================
def _rank_pct(series):
    s = series.dropna()
    if len(s) == 0:
        return series
    return s.rank(pct=True)


# ==================== 主逻辑 ====================
def main(context):
    date_str = context.current_dt.strftime('%Y-%m-%d')
    stocks = get_stock_pool(context)
    if g.debug:
        log.info('[%s] pool=%d' % (date_str, len(stocks)))
    if len(stocks) < g.stock_num:
        return

    end_dt = context.previous_date

    # 分块计算 xgb_alpha 和 turnover_change
    xgb_scores = {{}}
    turnover_scores = {{}}

    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        # xgb_alpha
        xgb_res = compute_xgb_alpha(open_df, high_df, low_df, close_df, volume_df)
        xgb_scores.update(xgb_res.to_dict())

        # turnover_change: volume MA5 / MA20
        ma5 = volume_df.rolling(5, min_periods=3).mean()
        ma20 = volume_df.rolling(20, min_periods=10).mean()
        tc = (ma5 / (ma20.replace(0, np.nan) + 1e-9)).iloc[-1]
        turnover_scores.update(tc.to_dict())

        del open_df, high_df, low_df, close_df, volume_df, xgb_res, tc, ma5, ma20
        gc.collect()

    xgb_series = pd.Series(xgb_scores)
    tc_series = pd.Series(turnover_scores)

    # 复合: rank(xgb_alpha) × rank(turnover_change)
    r_xgb = _rank_pct(xgb_series)
    r_tc = _rank_pct(tc_series)
    total = r_xgb * r_tc
    total = total.dropna().sort_values(ascending=False)

    if len(total) == 0:
        return

    target = total.head(g.stock_num).index.tolist()
    if g.debug:
        log.info('[%s] top3 scores: %s' % (date_str,
                 ', '.join('%.4f' % v for v in total.head(3).values)))

    rebalance(context, target)


# ==================== 调仓 ====================
def rebalance(context, target):
    target_set = set(target)
    positions = context.portfolio.positions
    current = set(positions.keys())

    total_value = context.portfolio.total_value
    unit = total_value / g.stock_num
    turnover = 0.0
    for s in current - target_set:
        turnover += abs(positions[s].value)
    for s in target:
        cur = positions[s].value if s in positions else 0.0
        turnover += abs(unit - cur)
    turnover = turnover / total_value if total_value > 0 else 0.0

    if turnover > g.max_turnover and g.debug:
        log.info('[rebalance] turnover=%.2f > cap=%.2f' % (turnover, g.max_turnover))

    for s in current - target_set:
        order_target_value(s, 0)

    cd = get_current_data()
    for s in target:
        try:
            if cd[s].paused:
                continue
            if cd[s].last_price >= cd[s].high_limit * 0.995:
                continue
        except Exception:
            pass
        order_target_value(s, unit)
'''

# Save JQ code
jq_path = OUT / 'fa_composite_xgb_alpha_jq.py'
with open(jq_path, 'w', encoding='utf-8-sig') as f:
    f.write(jq_code)

json_kb = json_size / 1024
print(f"  Saved: {jq_path} ({json_kb:.1f} KB model, {len(jq_code)/1024:.1f} KB total)", flush=True)

# Verify code compiles
import ast
try:
    compile(jq_code, str(jq_path), 'exec')
    print("  SYNTAX OK", flush=True)
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}", flush=True)

print("Done!", flush=True)
