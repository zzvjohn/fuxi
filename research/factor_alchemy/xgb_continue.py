"""从 checkpoint 继续 XGBoost 因子发现 (步骤 4-6)."""
import pickle, numpy as np, pandas as pd, time, json
from pathlib import Path
from scipy.stats import spearmanr
import xgboost as xgb

OUTPUT_DIR = Path(__file__).parent / 'output'
CACHE_PATH = OUTPUT_DIR / 'factor_cache.pkl'

# Load checkpoint
print('[4/6] Loading checkpoint...', flush=True)
with open(OUTPUT_DIR / 'xgb_feature_matrix.pkl', 'rb') as f:
    feature_df = pickle.load(f)
print(f'  Shape: {feature_df.shape}', flush=True)

# Load factor cache
with open(CACHE_PATH, 'rb') as f:
    cache_ret1m = pickle.load(f)['ret_1m']
weekly_dates = cache_ret1m.index

# Load daily close only
print('[5/6] Loading daily close...', flush=True)
DATA_CSV = Path(__file__).parent.parent.parent / 'data' / 'backtest_universe'
csv_files = list(DATA_CSV.glob('*.csv'))
csv_codes = {f.stem: f for f in csv_files}
stock_universe = cache_ret1m.columns.tolist()
matched = [s for s in stock_universe if s in csv_codes]
daily_close = {}
for code in matched:
    df = pd.read_csv(csv_codes[code], parse_dates=['trade_date'], usecols=['trade_date','close'])
    daily_close[code] = df.sort_values('trade_date').set_index('trade_date')['close']
print(f'  Loaded {len(daily_close)} stocks', flush=True)

# Compute targets (vectorized per stock)
print('[6/6] Computing targets + training XGBoost...', flush=True)
targets = {}
t0 = time.time()
for ci, code in enumerate(matched):
    s = daily_close[code]
    for wd in weekly_dates:
        wd_pd = pd.Timestamp(wd)
        before = s[s.index <= wd_pd]
        if len(before) == 0: continue
        cur_c = before.iloc[-1]
        after = s[(s.index > wd_pd) & (s.index <= wd_pd + pd.Timedelta(days=7))]
        if len(after) == 0: continue
        targets[(wd_pd, code)] = (after.iloc[-1] / cur_c) - 1.0
    if (ci+1) % 200 == 0:
        print(f'  Target {ci+1}/{len(matched)} ({time.time()-t0:.0f}s)', flush=True)

target_s = pd.Series(targets)
target_s.index.names = ['date','code']
print(f'  Targets: {len(target_s)}', flush=True)

# Align
common = feature_df.index.intersection(target_s.index)
X = feature_df.loc[common].fillna(0).values.astype(np.float32)
y_raw = target_s.loc[common]
y = y_raw.clip(y_raw.quantile(0.01), y_raw.quantile(0.99)).values.astype(np.float32)
print(f'  Aligned: {len(common)} rows', flush=True)

# Split
idx_dates = feature_df.loc[common].index.get_level_values('date')
t_mask = idx_dates <= pd.Timestamp('2023-12-31')
v_mask = (idx_dates > pd.Timestamp('2023-12-31')) & (idx_dates <= pd.Timestamp('2024-12-31'))
e_mask = idx_dates > pd.Timestamp('2024-12-31')
print(f'  Train/Val/Test: {t_mask.sum()}/{v_mask.sum()}/{e_mask.sum()}', flush=True)

# Train
model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=10, n_estimators=200, random_state=42, n_jobs=4, tree_method='hist',
)
model.fit(X[t_mask], y[t_mask], eval_set=[(X[v_mask], y[v_mask])], verbose=False)
y_pred = model.predict(X[e_mask])
test_sp = spearmanr(y[e_mask], y_pred)[0]
print(f'  Test Spearman: {test_sp:.4f}', flush=True)

# Importance
imp = model.feature_importances_
for name, v in sorted(zip(feature_df.columns, imp), key=lambda x: -x[1])[:10]:
    print(f'    {name}: {v:.4f}', flush=True)

# Save factor
test_idx = feature_df.loc[common].index[e_mask]
xgb_pred = pd.Series(y_pred, index=test_idx, name='xgb_alpha')
xgb_matrix = xgb_pred.unstack(level='code')
print(f'  Matrix: {xgb_matrix.shape}', flush=True)

with open(OUTPUT_DIR / 'xgb_alpha_factor.pkl', 'wb') as f:
    pickle.dump(xgb_matrix, f)

# ICIR
ic_list = []
for d in xgb_matrix.index:
    alpha = xgb_matrix.loc[d].dropna()
    if len(alpha) < 30: continue
    cd = pd.Timestamp(d)
    if cd in cache_ret1m.index:
        idx = cache_ret1m.index.get_loc(cd)
        if idx + 1 < len(cache_ret1m):
            fwd = cache_ret1m.iloc[idx + 1]
            cs = alpha.index.intersection(fwd.index)
            if len(cs) >= 30:
                ic = spearmanr(alpha[cs].values, fwd[cs].values)[0]
                if not np.isnan(ic): ic_list.append(ic)

ic_arr = np.array(ic_list)
icir = ic_arr.mean() / (ic_arr.std() + 1e-9)
print(f'  ICIR: {icir:.4f} (mean={ic_arr.mean():.4f} std={ic_arr.std():.4f} +IC%={np.mean(ic_arr>0):.1%} n={len(ic_list)})', flush=True)

report = {
    'test_spearman': float(test_sp), 'test_icir': float(icir),
    'test_ic_mean': float(ic_arr.mean()), 'test_ic_std': float(ic_arr.std()),
    'test_rows': int(e_mask.sum()), 'test_weeks': len(xgb_matrix),
    'test_stocks': int(xgb_matrix.shape[1]),
    'feature_importance': {n: float(v) for n, v in zip(feature_df.columns, imp)},
}
with open(OUTPUT_DIR / 'xgb_alpha_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print('Done!', flush=True)
