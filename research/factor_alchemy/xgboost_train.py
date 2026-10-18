# -*- coding: utf-8 -*-
"""
XGBoost 非线性因子组合 — 完整训练脚本
=====================================
步骤:
 1. 加载 FA 输出的 5 因子 CSV (本地已有)
 2. 从 Tushare 获取周频 close 价格 (1 次调用，分批 400 只)
 3. 计算 forward 1-week return
 4. 训练 XGBoost
 5. 导出 243 行查表 + 性能报告
"""
import pandas as pd
import numpy as np
import os, sys, time, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from credentials import get_tushare_api
pro = get_tushare_api()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
RESEARCH_DIR = os.path.join(os.path.dirname(__file__), '..')

# ============================================
# [1] 加载 5 因子数据
# ============================================
print("=" * 60)
print("[1] 加载 5 因子 CSV")
print("=" * 60)

factor_dir = OUTPUT_DIR
fnames = ['max_ret_1m', 'vol_3m', 'cfp', 'sp', 'davol_20']
factors = {}
all_stocks = set()
date_sets = []

for fn in fnames:
    fpath = os.path.join(factor_dir, 'factor_%s.csv' % fn)
    df = pd.read_csv(fpath, index_col=0)
    factors[fn] = df
    all_stocks.update(df.columns.tolist())
    date_sets.append(set(df.index))
    print("  %s: %d weeks × %d stocks" % (fn, df.shape[0], df.shape[1]))

common_dates = sorted(list(set.intersection(*date_sets)))
all_stocks = sorted(all_stocks)
print(f"\n  共 {len(common_dates)} 周, {len(all_stocks)} 只股票, 日期 {common_dates[0]} ~ {common_dates[-1]}")

# ============================================
# [2] 获取周频 close 价格
# ============================================
print("\n" + "=" * 60)
print("[2] 获取周频 close 价格 (Tushare weekly)")
print("=" * 60)

# 取需要的日期范围 + 1 周
start_date = common_dates[0]
end_date = common_dates[-1]

# 使用 pro.weekly 获取全市场周频数据 (分批避免超时)
all_close = []
BATCH = 100
for i in range(0, len(all_stocks), BATCH):
    batch_codes = ','.join(all_stocks[i:i+BATCH])
    try:
        df = pro.weekly(
            ts_code=batch_codes,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            fields='ts_code,trade_date,close'
        )
        if df is not None and len(df) > 0:
            all_close.append(df)
            print(f"  Batch {i//BATCH}: {len(df)} rows")
        time.sleep(0.15)  # 避免频率限制
    except Exception as e:
        print(f"  Batch {i//BATCH} ERROR: {e}")
        time.sleep(1)

if not all_close:
    print("[ERROR] 无法获取价格数据, 退出")
    sys.exit(1)

price_df = pd.concat(all_close, ignore_index=True)
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
price_df = price_df.pivot_table(values='close', index='trade_date', columns='ts_code', aggfunc='last')
print(f"  价格矩阵: {price_df.shape}")

# 对齐到 common_dates (最近交易日)
price_aligned = price_df.reindex(pd.DatetimeIndex(common_dates), method='ffill')
print(f"  对齐后: {price_aligned.shape}")

# ============================================
# [3] 构建训练集: 5因子(t) → forward_return(t→t+1)
# ============================================
print("\n" + "=" * 60)
print("[3] 构建训练数据集")
print("=" * 60)

records = []
for i in range(len(common_dates) - 1):
    t = common_dates[i]
    t1 = common_dates[i + 1]
    if i % 30 == 0:
        print(f"  Processing {t} ...")
    
    # Get 5 factor values for this week
    factor_dict = {}
    for fn in fnames:
        if t in factors[fn].index:
            factor_dict[fn] = factors[fn].loc[t]
    
    if len(factor_dict) < 5:
        continue
    
    # Find common stocks
    common_cols = set(factor_dict[list(factor_dict.keys())[0]].dropna().index)
    for fn in fnames:
        common_cols &= set(factor_dict[fn].dropna().index)
    
    if t in price_aligned.index and t1 in price_aligned.index:
        price_t = price_aligned.loc[t]
        price_t1 = price_aligned.loc[t1]
        common_cols &= set(price_t.dropna().index) & set(price_t1.dropna().index)
    
    common_cols = sorted(common_cols)
    if len(common_cols) < 50:
        continue
    
    # Build record
    for col in common_cols:
        try:
            fwd_ret = price_t1[col] / price_t[col] - 1
            if abs(fwd_ret) > 0.5:  # skip extreme returns
                continue
            rec = {
                'date': t,
                'stock': col,
                'max_ret_1m': factor_dict['max_ret_1m'][col],
                'vol_3m': factor_dict['vol_3m'][col],
                'cfp': factor_dict['cfp'][col],
                'sp': factor_dict['sp'][col],
                'davol_20': factor_dict['davol_20'][col],
                'fwd_ret_1w': fwd_ret,
            }
            records.append(rec)
        except:
            continue

train_df = pd.DataFrame(records)
print(f"\n  训练集: {len(train_df)} 行")
print(f"  fwd_ret stats: mean={train_df['fwd_ret_1w'].mean():.4f} std={train_df['fwd_ret_1w'].std():.4f}")

# ============================================
# [4] 训练 XGBoost
# ============================================
print("\n" + "=" * 60)
print("[4] XGBoost 训练")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

feature_cols = fnames
X = train_df[feature_cols].values
y = train_df['fwd_ret_1w'].values

# 80/20 时间分割 (最后20%为验证)
n = len(train_df)
split_idx = int(n * 0.8)
# Sort by date to ensure temporal split
train_df_sorted = train_df.sort_values('date').reset_index(drop=True)
X = train_df_sorted[feature_cols].values
y = train_df_sorted['fwd_ret_1w'].values
n = len(X)
split_idx = int(n * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

xgb = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=1,
)
xgb.fit(X_train, y_train)

y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)
ic_train = np.corrcoef(y_train, y_pred_train)[0, 1]
ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]

print(f"\n  R2 train: {r2_score(y_train, y_pred_train):.4f}")
print(f"  R2 test:  {r2_score(y_test, y_pred_test):.4f}")
print(f"  IC train: {ic_train:.4f}")
print(f"  IC test:  {ic_test:.4f}")

# Feature importance
importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  Feature importance:")
for k, v in importance.items():
    print(f"    {k}: {v:.4f}")

# ============================================
# [5] 导出查表 (3^5 = 243 行)
# ============================================
print("\n" + "=" * 60)
print("[5] 导出 243 行查表")
print("=" * 60)

# 计算分位点
quartiles = {}
for fn in fnames:
    vals = train_df[fn].dropna()
    q33 = np.percentile(vals, 33.33)
    q67 = np.percentile(vals, 66.67)
    quartiles[fn] = (q33, q67)
    print(f"  {fn}: 33%={q33:.3f}, 67%={q67:.3f}")

# 生成 3^5 组合
from itertools import product

def bucketize(val, q33, q67):
    if val <= q33:
        return 0  # low
    elif val <= q67:
        return 1  # mid
    else:
        return 2  # high

lookup = {}
for combo in product([0, 1, 2], repeat=5):
    # 取每档的代表值 (中位数)
    rep = []
    for j, fn in enumerate(fnames):
        vals = train_df[fn].dropna()
        q33, q67 = quartiles[fn]
        if combo[j] == 0:
            rep.append(np.median(vals[vals <= q33])) if (vals <= q33).any() else rep.append(q33)
        elif combo[j] == 1:
            mask = (vals > q33) & (vals <= q67)
            rep.append(np.median(vals[mask])) if mask.any() else rep.append((q33+q67)/2)
        else:
            rep.append(np.median(vals[vals > q67])) if (vals > q67).any() else rep.append(q67)
    
    pred = float(xgb.predict(np.array([rep]))[0])
    key = ''.join(str(c) for c in combo)
    lookup[key] = {
        'combo': combo,
        'rep_values': rep,
        'pred_return': pred,
    }

# Save lookup table
lookup_file = os.path.join(RESEARCH_DIR, 'xgboost_5f_lookup.pkl')
with open(lookup_file, 'wb') as f:
    pickle.dump({
        'lookup': lookup,
        'quartiles': quartiles,
        'feature_importance': importance.to_dict(),
        'ic_train': ic_train,
        'ic_test': ic_test,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
    }, f)
print(f"\n  查表保存到: {lookup_file}")
print(f"  共 {len(lookup)} 行")

# 也保存为 CSV (方便查看)
csv_rows = []
for key, info in sorted(lookup.items()):
    csv_rows.append({
        'combo': key,
        'c0': info['combo'][0], 'c1': info['combo'][1],
        'c2': info['combo'][2], 'c3': info['combo'][3], 'c4': info['combo'][4],
        'pred_ret': info['pred_return'],
    })
pd.DataFrame(csv_rows).to_csv(os.path.join(RESEARCH_DIR, 'xgboost_5f_lookup.csv'), index=False)

# ============================================
# [6] 对比: 线性加权 vs XGBoost
# ============================================
print("\n" + "=" * 60)
print("[6] 对比: 线性加权 vs XGBoost")
print("=" * 60)

w = [0.229, 0.211, 0.199, 0.182, 0.178]
linear_score = X_test[:, 0] * w[0] + X_test[:, 1] * w[1] + X_test[:, 2] * w[2] + X_test[:, 3] * w[3] + X_test[:, 4] * w[4]
ic_linear = np.corrcoef(y_test, linear_score)[0, 1]
xgboost_score = xgb.predict(X_test)
ic_xgb = np.corrcoef(y_test, xgboost_score)[0, 1]

print(f"  线性权重 IC: {ic_linear:.4f}")
print(f"  XGBoost IC:   {ic_xgb:.4f}")
print(f"  提升:         {ic_xgb - ic_linear:+.4f}")

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)
