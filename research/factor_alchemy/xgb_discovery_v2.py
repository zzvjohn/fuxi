"""
XGBoost Factor Discovery v2 — 修复验证管道
===========================================
修复项 vs v1:
  1. 数据源: Tushare daily_prices (5745只, 非 backtest_universe 1081只幸存者偏误)
  2. 前向收益: 真实周收益 close[W+1]/close[W]-1 (非 ret_1m 代理)
  3. 训练: Expanding window OOS (非静态全量)
  4. ICIR: 纯 OOS 预测的截面 Spearman

Pipeline:
  [1] 加载 Tushare daily_prices.csv → pivot date×stock
  [2] 对齐 factor_cache 周频日期
  [3] 计算 19 个原始基元特征 + 真实前向周收益 target
  [4] Expanding window XGBoost:
      - Fold 1: train 2021-2022 → OOS 2023
      - Fold 2: train 2021-2023 → OOS 2024
      - Fold 3: train 2021-2024 → OOS 2025-2026
  [5] 池化 OOS 预测 → 计算 ICIR
  [6] 若通过 → 全量训练 + JQ 代码导出
"""
import pickle, numpy as np, pandas as pd, json, time, gc, os
from pathlib import Path
from scipy.stats import spearmanr
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SRC = Path(__file__).parent
OUT = SRC / 'output'
CACHE_PATH = OUT / 'factor_cache.pkl'
TUSHARE_CSV = SRC.parent.parent / 'data' / 'raw' / 'daily_prices.csv'
CHECKPOINT_DIR = OUT / 'xgb_v2_checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ==================== 配置 ====================
FORWARD_DAYS = 7           # 前向7个自然日到下周
MIN_WEEKLY_STOCKS = 100    # 每周最少股票数
MIN_TRAIN_WEEKS = 100      # 最少训练周数

# ==================== 步骤 1: 加载 Tushare 日线 ====================
def load_tushare_daily():
    ckpt = CHECKPOINT_DIR / 'daily_panels.pkl'
    if ckpt.exists():
        print("[1] Loading Tushare daily panels from checkpoint...", flush=True)
        with open(ckpt, 'rb') as f:
            return pickle.load(f)

    print("[1] Loading Tushare daily_prices.csv...", flush=True)
    t0 = time.time()
    df = pd.read_csv(TUSHARE_CSV, dtype={'trade_date': str})
    print(f"    Loaded {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    # Mixed formats: 13.3M rows "%Y%m%d" + 77K rows "%Y-%m-%d" (2026-07+)
    mask_dash = df['trade_date'].str.contains('-')
    n_dash = mask_dash.sum()
    print(f"    Dates: {len(df)-n_dash:,} Ymd + {n_dash:,} ISO ({time.time()-t0:.0f}s)", flush=True)
    # Convert via concat (avoids StringDtype TypeError)
    dt_ymd = pd.to_datetime(df.loc[~mask_dash, 'trade_date'], format='%Y%m%d')
    dt_iso = pd.to_datetime(df.loc[mask_dash, 'trade_date'])
    df['trade_date'] = pd.concat([dt_ymd, dt_iso]).sort_index()
    print(f"    Parsed dates ({time.time()-t0:.0f}s)", flush=True)
    df = df.sort_values(['ts_code', 'trade_date'])

    print(f"    Raw: {len(df):,} rows, {df.ts_code.nunique()} stocks", flush=True)

    # Pivot: trade_date × ts_code for each field
    panels = {}
    for field in ['open', 'high', 'low', 'close', 'vol']:
        p = df.pivot_table(index='trade_date', columns='ts_code', values=field, aggfunc='first')
        panels[field] = p.sort_index()
        print(f"    {field}: {p.shape[0]} days × {p.shape[1]} stocks", flush=True)

    # Save checkpoint
    with open(ckpt, 'wb') as f:
        pickle.dump(panels, f)
    return panels


# ==================== 步骤 2: 对齐周频日期 ====================
def align_weekly_dates(panels):
    with open(CACHE_PATH, 'rb') as f:
        cache_ret1m = pickle.load(f)['ret_1m']
    weekly_dates = sorted(cache_ret1m.index)
    print(f"\n[2] Factor cache weekly dates: {len(weekly_dates)} weeks", flush=True)
    print(f"    Range: {weekly_dates[0].strftime('%Y-%m-%d')} ~ {weekly_dates[-1].strftime('%Y-%m-%d')}", flush=True)

    # Find closest trading day ≤ each weekly date
    close_dates = panels['close'].index
    t0 = time.time()

    # For each weekly date, find the most recent trading day
    weekly_map = {}
    for wd in weekly_dates:
        wd_pd = pd.Timestamp(wd)
        avail = close_dates[close_dates <= wd_pd]
        if len(avail) > 0:
            weekly_map[wd_pd] = avail[-1]  # last trading day ≤ weekly date

    print(f"    Mapped {len(weekly_map)} weekly dates to trading days ({time.time()-t0:.1f}s)", flush=True)
    return weekly_dates, weekly_map


# ==================== 特征函数 (与 JQ compute_xgb_alpha 完全一致) ====================
FEATURE_NAMES = [
    'r_1d', 'r_5d', 'r_10d', 'r_20d', 'r_40d', 'r_60d',
    'vol_5d', 'vol_10d', 'vol_20d',
    'maxdd_5d', 'maxdd_10d', 'maxdd_20d',
    'vratio_5d', 'vratio_20d',
    'amp_5d', 'amp_20d',
    'up_ratio',
    'streak',
    'gap_avg',
]

def compute_raw_features(close_arr, open_arr, high_arr, low_arr, vol_arr):
    """
    19 个原始基元。输入 numpy arrays: (n_days, n_stocks).
    返回 (n_stocks, 19) float32.
    """
    n_rows, n_cols = close_arr.shape
    feats = np.full((n_cols, 19), np.nan, dtype=np.float32)

    # r_Xd: 收益率基元
    for i, horizon in enumerate([1, 5, 10, 20, 40, 60]):
        if n_rows > horizon:
            feats[:, i] = close_arr[-1] / close_arr[-(horizon+1)] - 1.0

    # daily_ret
    daily_ret = np.diff(close_arr, axis=0) / (close_arr[:-1] + 1e-12)
    n_dr = daily_ret.shape[0]

    # vol_Xd
    for j, horizon in enumerate([5, 10, 20]):
        if n_dr >= horizon:
            feats[:, 6+j] = np.nanstd(daily_ret[-horizon:], axis=0)

    # maxdd_Xd
    for j, horizon in enumerate([5, 10, 20]):
        if n_rows >= horizon:
            window = close_arr[-horizon:]
            peak = np.maximum.accumulate(window, axis=0)
            feats[:, 9+j] = (window[-1] - peak[-1]) / (peak[-1] + 1e-12)

    # vratio_Xd
    for j, horizon in enumerate([5, 20]):
        if n_rows >= horizon:
            feats[:, 12+j] = vol_arr[-1] / (np.nanmean(vol_arr[-horizon:], axis=0) + 1e-12)

    # amp_Xd
    for j, horizon in enumerate([5, 20]):
        if n_rows >= horizon:
            hl = high_arr[-horizon:] - low_arr[-horizon:]
            feats[:, 14+j] = np.nanmean(hl / (close_arr[-horizon:] + 1e-12), axis=0)

    # up_ratio
    if n_dr >= 20:
        feats[:, 16] = np.nanmean(daily_ret[-20:] > 0, axis=0)

    # streak
    if n_dr >= 10:
        signs = np.sign(daily_ret[-10:])
        for si in range(n_cols):
            cs = signs[:, si]
            last = cs[-1]
            if last == 0:
                feats[si, 17] = 0.0
            else:
                cnt = 0
                for k in range(len(cs)-1, -1, -1):
                    if cs[k] == last: cnt += 1
                    else: break
                feats[si, 17] = cnt if last > 0 else -cnt

    # gap_avg
    if n_rows >= 6:
        gaps = open_arr[-5:] / close_arr[-6:-1] - 1.0
        feats[:, 18] = np.nanmean(gaps, axis=0)

    return feats


# ==================== 步骤 3: 计算特征矩阵 + 前向收益 target ====================
def build_weekly_features(panels, weekly_dates, weekly_map):
    # Always recompute — checkpoint bugs waste more time than recomputation
    print("[3] Computing weekly features + forward returns...", flush=True)
    close_p = panels['close']
    open_p = panels['open']
    high_p = panels['high']
    low_p = panels['low']
    vol_p = panels['vol']

    rows = []
    t0 = time.time()
    n_processed = 0

    for wi, wd in enumerate(weekly_dates):
        wd_pd = pd.Timestamp(wd)
        if wd_pd not in weekly_map:
            continue

        td = weekly_map[wd_pd]
        td_pos = close_p.index.get_loc(td)

        # Need at least 61 days of history
        if td_pos < 60:
            continue

        # Slice data window: td_pos - 60 to td_pos (61 days)
        start_pos = td_pos - 60
        window_cl = close_p.iloc[start_pos:td_pos+1]
        window_op = open_p.iloc[start_pos:td_pos+1]
        window_hi = high_p.iloc[start_pos:td_pos+1]
        window_lo = low_p.iloc[start_pos:td_pos+1]
        window_vo = vol_p.iloc[start_pos:td_pos+1]

        if len(window_cl) < 20:
            continue

        # Find stocks with enough data in this window
        valid_mask = window_cl.notna().sum() >= 20
        valid_stocks = valid_mask[valid_mask].index.tolist()
        if len(valid_stocks) < MIN_WEEKLY_STOCKS:
            continue

        # Extract arrays (keep date×stock shape, compatible with compute_raw_features)
        cl_arr = window_cl[valid_stocks].values.astype(np.float32)
        op_arr = window_op[valid_stocks].values.astype(np.float32)
        hi_arr = window_hi[valid_stocks].values.astype(np.float32)
        lo_arr = window_lo[valid_stocks].values.astype(np.float32)
        vo_arr = window_vo[valid_stocks].values.astype(np.float32)

        feats = compute_raw_features(cl_arr, op_arr, hi_arr, lo_arr, vo_arr)
        feats = np.nan_to_num(feats, nan=0.0)

        # Forward return: find next week's close
        next_wd = wd_pd + pd.Timedelta(days=FORWARD_DAYS)
        next_avail = close_p.index[close_p.index > wd_pd]
        if len(next_avail) == 0:
            continue
        next_td = None
        for nt in next_avail:
            if nt <= next_wd:
                next_td = nt
        if next_td is None:
            # Next week's data not available yet — near end of dataset
            continue

        next_close = close_p.loc[next_td, valid_stocks]
        cur_close = close_p.loc[td, valid_stocks]
        fwd_ret = (next_close.values / cur_close.values - 1.0)

        # Guard: ensure dimensions match (some stocks may be missing at certain dates)
        n_stocks = len(valid_stocks)
        if feats.shape[0] != n_stocks or len(fwd_ret) != n_stocks:
            continue

        # Build records
        for si, stock in enumerate(valid_stocks):
            if not np.isnan(fwd_ret[si]):
                row = [wd_pd, stock, *feats[si].tolist(), fwd_ret[si]]
                rows.append(row)

        n_processed += 1
        if n_processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {n_processed}/{len(weekly_dates)} weeks, {len(rows):,} rows ({elapsed:.0f}s)", flush=True)
        gc.collect()  # lighter collection per week

    col_names = ['date', 'stock'] + FEATURE_NAMES + ['fwd_ret']
    result = pd.DataFrame(rows, columns=col_names)
    result = result.set_index(['date', 'stock'])

    # Create feature matrix
    feature_df = result[FEATURE_NAMES]
    target_s = result['fwd_ret']

    # Winsorize target
    lo, hi = target_s.quantile(0.01), target_s.quantile(0.99)
    target_s = target_s.clip(lo, hi)

    print(f"    Final: {len(feature_df):,} rows, {feature_df.shape[1]} features", flush=True)
    print(f"    Target stats: mean={target_s.mean():.4f}, std={target_s.std():.4f}, "
          f"range=({target_s.min():.4f}, {target_s.max():.4f})", flush=True)

    return {'features': feature_df, 'target': target_s}


# ==================== 步骤 4: Expanding Window OOS XGBoost ====================
def expanding_window_train(data):
    print("[4] Expanding window XGBoost training...", flush=True)
    feature_df = data['features']
    target_s = data['target']
    dates = feature_df.index.get_level_values('date')

    # Define folds
    folds = [
        ('2021-2022→2023',  pd.Timestamp('2010-01-01'), pd.Timestamp('2022-12-31'),
         pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31')),
        ('2021-2023→2024',  pd.Timestamp('2010-01-01'), pd.Timestamp('2023-12-31'),
         pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31')),
        ('2021-2024→2025+', pd.Timestamp('2010-01-01'), pd.Timestamp('2024-12-31'),
         pd.Timestamp('2025-01-01'), pd.Timestamp('2026-12-31')),
    ]

    xgb_params = {
        'objective': 'reg:squarederror', 'learning_rate': 0.05, 'max_depth': 4,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_weight': 15, 'n_estimators': 150, 'random_state': 42,
        'n_jobs': 4, 'tree_method': 'hist',
    }

    all_oos = []
    fold_results = []

    for fold_name, train_start, train_end, test_start, test_end in folds:
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        X_tr = feature_df[train_mask].fillna(0).values.astype(np.float32)
        y_tr = target_s[train_mask].values.astype(np.float32)
        X_te = feature_df[test_mask].fillna(0).values.astype(np.float32)
        y_te = target_s[test_mask].values.astype(np.float32)

        if len(X_tr) < 1000 or len(X_te) < 100:
            print(f"  {fold_name}: SKIP (train={len(X_tr)}, test={len(X_te)})", flush=True)
            continue

        print(f"  {fold_name}: train={len(X_tr):,}, test={len(X_te):,}", flush=True)
        model_fold = xgb.XGBRegressor(**xgb_params)
        model_fold.fit(X_tr, y_tr, verbose=False)

        y_pred = model_fold.predict(X_te)
        sp = spearmanr(y_te, y_pred)[0] if len(y_te) > 1 else 0

        # Pool predictions
        pred_s = pd.Series(y_pred, index=feature_df[test_mask].index, name='xgb_alpha')
        all_oos.append(pred_s)

        fold_results.append({
            'fold': fold_name,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': len(X_tr),
            'n_test': len(X_te),
            'test_spearman': float(sp),
        })
        print(f"    Test Spearman: {sp:.4f}", flush=True)

    all_oos_preds = pd.concat(all_oos)
    print(f"  Total OOS predictions: {len(all_oos_preds):,}", flush=True)

    result = {'predictions': all_oos_preds, 'fold_results': fold_results}
    return result


# ==================== 步骤 5: ICIR 验证 ====================
def compute_oos_icir(oos_result, data):
    print("[5] Computing OOS ICIR...", flush=True)
    preds = oos_result['predictions']
    target_s = data['target']

    # Align predictions with targets
    common = preds.index.intersection(target_s.index)
    pred_aligned = preds.loc[common]
    target_aligned = target_s.loc[common]

    # Weekly cross-sectional IC
    dates = pred_aligned.index.get_level_values('date').unique()
    ic_list = []
    for d in dates:
        p = pred_aligned.loc[d].dropna()
        t = target_aligned.loc[d].dropna()
        cs = p.index.intersection(t.index)
        if len(cs) < 30:
            continue
        ic = spearmanr(p[cs].values, t[cs].values)[0]
        if not np.isnan(ic):
            ic_list.append(ic)

    ic_arr = np.array(ic_list)
    icir = ic_arr.mean() / (ic_arr.std() + 1e-9)

    print(f"    ICIR: {icir:.4f}", flush=True)
    print(f"    IC mean: {ic_arr.mean():.4f}, IC std: {ic_arr.std():.4f}", flush=True)
    print(f"    +IC%: {np.mean(ic_arr > 0):.1%}, n_weeks: {len(ic_arr)}", flush=True)

    # Per-fold IC summary
    for fr in oos_result['fold_results']:
        t_start = fr['test_start']
        t_end = fr['test_end']
        fold_preds = pred_aligned.loc[(pred_aligned.index.get_level_values('date') >= t_start) &
                                       (pred_aligned.index.get_level_values('date') <= t_end)]
        fold_ic_list = []
        for d in fold_preds.index.get_level_values('date').unique():
            p = fold_preds.loc[d].dropna()
            t = target_aligned.loc[d].dropna()
            cs = p.index.intersection(t.index)
            if len(cs) < 30:
                continue
            ic = spearmanr(p[cs].values, t[cs].values)[0]
            if not np.isnan(ic):
                fold_ic_list.append(ic)
        if fold_ic_list:
            fa = np.array(fold_ic_list)
            print(f"    {fr['fold']}: ICIR={fa.mean()/fa.std():.4f}, Spearman={fr['test_spearman']:.4f}, "
                  f"+IC%={np.mean(fa>0):.1%}",
                  flush=True)

    # Pivot for factor viewing
    xgb_matrix = pred_aligned.unstack(level='stock')
    print(f"    Prediction matrix: {xgb_matrix.shape[0]} weeks × {xgb_matrix.shape[1]} stocks", flush=True)
    print(f"    Date range: {xgb_matrix.index[0].strftime('%Y-%m-%d')} ~ {xgb_matrix.index[-1].strftime('%Y-%m-%d')}", flush=True)

    return icir, ic_arr


# ==================== 步骤 6: 全量训练 + JQ 导出 ====================
def export_jq_code_if_pass(data, icir, threshold=0.30):
    if icir < threshold:
        print(f"\n[6] ICIR {icir:.4f} < {threshold} → SKIP JQ export", flush=True)
        return

    print(f"\n[6] ICIR {icir:.4f} >= {threshold} → Training full model + exporting JQ...", flush=True)

    feature_df = data['features']
    target_s = data['target']

    # Train on 2021-2024 (same as JQ backtest start)
    dates = feature_df.index.get_level_values('date')
    train_mask = dates <= pd.Timestamp('2024-12-31')
    X_train = feature_df[train_mask].fillna(0).values.astype(np.float32)
    y_train = target_s[train_mask].values.astype(np.float32)

    print(f"    Full training: {X_train.shape[0]:,} rows", flush=True)

    model_full = xgb.XGBRegressor(
        objective='reg:squarederror', learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=15, n_estimators=150, random_state=42,
        n_jobs=4, tree_method='hist',
    )
    model_full.fit(X_train, y_train, verbose=False)

    # Export trees
    model_json = model_full.get_booster().get_dump(dump_format='json')
    compact_trees = []
    for tree_str in model_json:
        root = json.loads(tree_str)
        nodes = []
        def flatten(node_dict):
            idx = len(nodes)
            if 'leaf' in node_dict:
                nodes.append({'leaf': float(node_dict['leaf'])})
            else:
                nodes.append({
                    'feat': int(node_dict['split'][1:]),
                    'thresh': float(node_dict['split_condition']),
                    'left': -1, 'right': -1,
                })
                left_idx = flatten(node_dict['children'][0])
                right_idx = flatten(node_dict['children'][1])
                nodes[idx]['left'] = left_idx
                nodes[idx]['right'] = right_idx
            return idx
        flatten(root)
        compact_trees.append(nodes)

    compact_json = json.dumps(compact_trees, separators=(',', ':'))
    print(f"    Model: {len(compact_trees)} trees, {sum(len(t) for t in compact_trees)} nodes, "
          f"{len(compact_json)/1024:.1f} KB", flush=True)

    # Generate JQ code
    jq_code = f'''# -*- coding: utf-8 -*-
"""
FA Composite XGB Alpha v2 — Expanding OOS XGBoost 残差信号 | JQ 真回测
=========================================================================
v2 vs v1:
  - v1: backtest_universe(1081幸存者) + ret_1m代理target + 静态全量训练 → 样本内过拟合
  - v2: Tushare daily_prices(5745只/含退市) + 真实周前向收益 + Expanding OOS 3折

xgb_alpha: 19个原始基元 → XGBoost(150树×深度4) → OOS ICIR={icir:.4f}
复合: rank(xgb_alpha) × rank(turnover_change)

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

    # r_1d ~ r_60d
    for i, h in enumerate([1, 5, 10, 20, 40, 60]):
        if n_rows > h:
            feats[:, i] = close_arr[-1] / close_arr[-(h+1)] - 1.0
        else:
            feats[:, i] = np.nan

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

    # streak
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

    # gap_avg
    if n_rows >= 6:
        gaps = open_arr[-5:] / close_arr[-6:-1] - 1.0
        feats[:, 18] = np.nanmean(gaps, axis=0)
    else:
        feats[:, 18] = np.nan

    feats = np.nan_to_num(feats)
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

    xgb_scores = {{}}
    turnover_scores = {{}}

    for i in range(0, len(stocks), CHUNK):
        chunk = stocks[i:i + CHUNK]
        open_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'open')
        high_df   = _get_price_field(chunk, end_dt, LOOKBACK, 'high')
        low_df    = _get_price_field(chunk, end_dt, LOOKBACK, 'low')
        close_df  = _get_price_field(chunk, end_dt, LOOKBACK, 'close')
        volume_df = _get_price_field(chunk, end_dt, LOOKBACK, 'volume')

        xgb_res = compute_xgb_alpha(open_df, high_df, low_df, close_df, volume_df)
        xgb_scores.update(xgb_res.to_dict())

        ma5 = volume_df.rolling(5, min_periods=3).mean()
        ma20 = volume_df.rolling(20, min_periods=10).mean()
        tc = (ma5 / (ma20.replace(0, np.nan) + 1e-9)).iloc[-1]
        turnover_scores.update(tc.to_dict())

        del open_df, high_df, low_df, close_df, volume_df, xgb_res, tc, ma5, ma20
        gc.collect()

    xgb_series = pd.Series(xgb_scores)
    tc_series = pd.Series(turnover_scores)

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
    jq_path = OUT / 'fa_composite_xgb_alpha_v2_jq.py'
    with open(jq_path, 'w', encoding='utf-8-sig') as f:
        f.write(jq_code)

    # Verify syntax
    import ast
    try:
        compile(jq_code, str(jq_path), 'exec')
        syntax_ok = True
    except SyntaxError as e:
        syntax_ok = False
        print(f"    SYNTAX ERROR: {e}", flush=True)

    print(f"    Saved: {jq_path} ({len(jq_code)/1024:.1f} KB, syntax={'OK' if syntax_ok else 'FAIL'})", flush=True)

    return jq_path


# ==================== Main ====================
if __name__ == '__main__':
    t_start = time.time()

    panels = load_tushare_daily()
    weekly_dates, weekly_map = align_weekly_dates(panels)
    data = build_weekly_features(panels, weekly_dates, weekly_map)
    oos_result = expanding_window_train(data)
    icir, ic_arr = compute_oos_icir(oos_result, data)

    # Save report (convert Timestamps to strings)
    clean_folds = []
    for fr in oos_result['fold_results']:
        cf = {}
        for k, v in fr.items():
            if isinstance(v, pd.Timestamp):
                cf[k] = v.strftime('%Y-%m-%d')
            else:
                cf[k] = v
        clean_folds.append(cf)

    report = {
        'oos_icir': float(icir),
        'oos_ic_mean': float(ic_arr.mean()),
        'oos_ic_std': float(ic_arr.std()),
        'oos_ic_positive_pct': float(np.mean(ic_arr > 0)),
        'oos_n_weeks': len(ic_arr),
        'folds': clean_folds,
        'n_predictions': len(oos_result['predictions']),
        'feature_names': FEATURE_NAMES,
    }
    with open(OUT / 'xgb_alpha_v2_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: ICIR={icir:.4f}, +IC%={np.mean(ic_arr>0):.1%}, n={len(ic_arr)}", flush=True)
    print(f"Folds: {oos_result['fold_results']}", flush=True)

    export_jq_code_if_pass(data, icir, threshold=0.20)

    print(f"Total time: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("Done!", flush=True)
