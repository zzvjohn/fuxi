"""
xgb_factor_discovery.py — XGBoost 从原始 OHLCV 学习残差信号, 作为新因子注入 V3
=====================================================================
策略: 从 daily OHLCV 计算"原始基元特征", expanding window 训练 XGBoost,
      OOS 预测值 = 新因子 "xgb_alpha", 注入 V3 80 因子池 → ICIR 筛选

与已证伪路径的关键区别:
  - 目标: 纯前向收益 (close[t+5]/close[t]-1), 非循环 target
  - 输出: 单一新因子 (非组合权重/交互对)
  - 特征: 原始量价基元 (非已有精炼因子), 最小重叠
  - 决策权: V3 ICIR 筛选 + rank-product (XGBoost 不参与组合)
"""

import os, sys, pickle, json, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
DATA_CSV = ROOT.parent.parent / 'data' / 'backtest_universe'
CACHE_PATH = ROOT / 'output' / 'factor_cache.pkl'
OUTPUT_DIR = ROOT / 'output'

# ==================== 配置 ====================
MIN_DAILY_BARS = 20            # 最少需要的历史日线

# ==================== 步骤 1: 加载因子缓存 ====================
print("[1/6] Loading factor cache...", flush=True)
t0 = time.time()
# 只加载 ret_1m 用于日期/股票对齐, 不加载全部 95 因子
with open(CACHE_PATH, 'rb') as f:
    cache_ret1m = pickle.load(f)['ret_1m']

weekly_dates = cache_ret1m.index
stock_universe = cache_ret1m.columns.tolist()
print(f"  Weekly dates: {len(weekly_dates)}, stock universe: {len(stock_universe)}", flush=True)

# ==================== 步骤 2: 加载每日 OHLCV ====================
print("\n[2/6] Loading daily OHLCV CSVs...", flush=True)
csv_files = list(DATA_CSV.glob('*.csv'))
csv_codes = {f.stem: f for f in csv_files}
print(f"  CSV files: {len(csv_files)}", flush=True)
matched = [s for s in stock_universe if s in csv_codes]
print(f"  Matched: {len(matched)}", flush=True)

daily_data = {}
for i, code in enumerate(matched):
    df = pd.read_csv(csv_codes[code], parse_dates=['trade_date'], usecols=['trade_date','open','high','low','close','vol'])
    df = df.sort_values('trade_date').set_index('trade_date')
    daily_data[code] = df
print(f"  Loaded {len(daily_data)} stocks, {time.time()-t0:.0f}s", flush=True)

# ==================== 步骤 3: 计算原始基元特征 ====================
print("\n[3/6] Computing raw primitives...", flush=True)

def compute_primitives(daily_df, target_date):
    close = daily_df['close']
    open_p = daily_df['open']
    high = daily_df['high']
    low = daily_df['low']
    volume = daily_df['vol']

    mask = daily_df.index <= target_date
    if not mask.any() or mask.sum() < MIN_DAILY_BARS:
        return None
    close = close[mask]
    open_p = open_p[mask]
    high = high[mask]
    low = low[mask]
    volume = volume[mask]

    f = {}
    # 收益
    for h in [1, 5, 10, 20, 40, 60]:
        f[f'r_{h}d'] = close.iloc[-1] / close.iloc[-(h+1)] - 1.0 if len(close) > h else np.nan
    # 波动率
    dr = close.pct_change()
    for h in [5, 10, 20]:
        f[f'vol_{h}d'] = dr.iloc[-h:].std() if len(dr) >= h else np.nan
    # 最大回撤
    for h in [5, 10, 20]:
        w = close.iloc[-h:]
        peak = w.cummax()
        dd = (w - peak) / peak
        f[f'maxdd_{h}d'] = dd.min()
    # 量比
    for h in [5, 20]:
        f[f'vratio_{h}d'] = volume.iloc[-1] / (volume.iloc[-h:].mean() + 1e-9) if len(volume) >= h else np.nan
    # 振幅
    for h in [5, 20]:
        f[f'amp_{h}d'] = ((high.iloc[-h:] - low.iloc[-h:]) / close.iloc[-h:]).mean()
    # 涨跌比
    f['up_ratio'] = (dr.iloc[-20:] > 0).mean() if len(dr) >= 20 else np.nan
    # 连涨连跌
    if len(dr) >= 10:
        signs = np.sign(dr.iloc[-10:].values)
        ls = signs[-1]
        streak = 0
        for s in reversed(signs):
            if s == ls and s != 0: streak += 1
            else: break
        f['streak'] = streak if ls > 0 else -streak
    else:
        f['streak'] = np.nan
    # 开盘跳空
    if len(open_p) >= 5:
        gaps = open_p.iloc[-5:] / close.shift(1).iloc[-5:] - 1.0
        f['gap_avg'] = gaps.mean()
    else:
        f['gap_avg'] = np.nan

    return f

all_rows = []
all_codes = []
all_dates = []
t3 = time.time()
for wi, wd in enumerate(weekly_dates):
    wd_pd = pd.Timestamp(wd)
    for code in matched:
        prim = compute_primitives(daily_data[code], wd_pd)
        if prim is None: continue
        all_rows.append(prim)
        all_codes.append(code)
        all_dates.append(wd_pd)
    if (wi+1) % 50 == 0:
        print(f"  Week {wi+1}/{len(weekly_dates)}, rows={len(all_rows)} ({time.time()-t3:.0f}s)", flush=True)

print(f"  Done: {len(all_rows)} rows, {time.time()-t3:.0f}s", flush=True)

feature_df = pd.DataFrame(all_rows, index=pd.MultiIndex.from_arrays([all_dates, all_codes], names=['date','code']))
print(f"  Shape: {feature_df.shape}, NaN: {feature_df.isna().mean().mean():.3f}", flush=True)

# Checkpoint save
ckpt_path = OUTPUT_DIR / 'xgb_feature_matrix.pkl'
with open(ckpt_path, 'wb') as f:
    pickle.dump(feature_df, f)
print(f"  Checkpoint: {ckpt_path}", flush=True)

# ==================== 步骤 4: 前向收益 target ====================
print("\n[4/6] Computing forward returns...", flush=True)
targets = {}
for code in matched:
    df = daily_data[code]
    for wd in weekly_dates:
        wd_pd = pd.Timestamp(wd)
        before = df[df.index <= wd_pd]
        if len(before) == 0: continue
        cur_close = before['close'].iloc[-1]
        next_end = wd_pd + pd.Timedelta(days=7)
        after = df[(df.index > wd_pd) & (df.index <= next_end)]
        if len(after) == 0: continue
        nxt_close = after['close'].iloc[-1]
        targets[(wd_pd, code)] = (nxt_close / cur_close) - 1.0

target_series = pd.Series(targets)
target_series.index.names = ['date','code']
print(f"  Targets: {len(target_series)}", flush=True)

# 对齐
common = feature_df.index.intersection(target_series.index)
X = feature_df.loc[common].fillna(0).values.astype(np.float32)
y = target_series.loc[common].clip(target_series.quantile(0.01), target_series.quantile(0.99)).values.astype(np.float32)
print(f"  Aligned: {len(common)} rows, {X.shape[1]} features", flush=True)

# ==================== 步骤 5: Expanding Window XGBoost ====================
print("\n[5/6] Training XGBoost...", flush=True)
import xgboost as xgb

# Time split: train ≤2023, val=2024, test=2025+
train_idx = feature_df.loc[common].index.get_level_values('date') <= pd.Timestamp('2023-12-31')
val_idx = (feature_df.loc[common].index.get_level_values('date') > pd.Timestamp('2023-12-31')) & \
          (feature_df.loc[common].index.get_level_values('date') <= pd.Timestamp('2024-12-31'))
test_idx = feature_df.loc[common].index.get_level_values('date') > pd.Timestamp('2024-12-31')

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_date_idx = feature_df.loc[common].index[test_idx]

print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}", flush=True)

model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=10, n_estimators=300, random_state=42, n_jobs=4, tree_method='hist',
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = model.predict(X_test)

from scipy.stats import spearmanr
test_spearman = spearmanr(y_test, y_pred)[0]
print(f"  Test Spearman: {test_spearman:.4f}", flush=True)

# Feature importance
imp = model.feature_importances_
for name, v in sorted(zip(feature_df.columns, imp), key=lambda x: -x[1])[:10]:
    print(f"    {name}: {v:.4f}", flush=True)

# ==================== 步骤 6: 保存 & ICIR ====================
print("\n[6/6] Saving xgb_alpha factor...", flush=True)
xgb_pred = pd.Series(y_pred, index=test_date_idx, name='xgb_alpha')
xgb_matrix = xgb_pred.unstack(level='code')
print(f"  Test matrix: {xgb_matrix.shape[0]} weeks x {xgb_matrix.shape[1]} stocks", flush=True)

out_pkl = OUTPUT_DIR / 'xgb_alpha_factor.pkl'
with open(out_pkl, 'wb') as f:
    pickle.dump(xgb_matrix, f)
print(f"  Saved: {out_pkl}", flush=True)

# ICIR (test period, 与 V3 方法一致)
ic_list = []
for d in xgb_matrix.index:
    alpha = xgb_matrix.loc[d].dropna()
    if len(alpha) < 30: continue
    # Target: ret_1m shift(-1)/4 代理
    cd = pd.Timestamp(d)
    if cd in cache_ret1m.index:
        idx = cache_ret1m.index.get_loc(cd)
        if idx + 1 < len(cache_ret1m):
            fwd = cache_ret1m.iloc[idx + 1]
            common_stocks = alpha.index.intersection(fwd.index)
            if len(common_stocks) >= 30:
                ic = spearmanr(alpha[common_stocks].values, fwd[common_stocks].values)[0]
                if not np.isnan(ic):
                    ic_list.append(ic)

if len(ic_list) >= 20:
    ic_arr = np.array(ic_list)
    icir = ic_arr.mean() / (ic_arr.std() + 1e-9)
    print(f"  Test ICIR: {icir:.4f}  (mean={ic_arr.mean():.4f}, std={ic_arr.std():.4f}, +IC%={np.mean(ic_arr>0):.1%})", flush=True)

# Save report
report = {
    'test_spearman': float(test_spearman),
    'test_icir': float(icir) if len(ic_list) >= 20 else None,
    'test_ic_mean': float(ic_arr.mean()) if len(ic_list) >= 20 else None,
    'test_ic_std': float(ic_arr.std()) if len(ic_list) >= 20 else None,
    'test_rows': int(X_test.shape[0]),
    'test_weeks': len(xgb_matrix),
    'test_stocks': int(xgb_matrix.shape[1]),
    'feature_importance': {name: float(v) for name, v in zip(feature_df.columns, imp)},
}
with open(OUTPUT_DIR / 'xgb_alpha_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nDone! xgb_alpha_factor.pkl + report saved.", flush=True)
