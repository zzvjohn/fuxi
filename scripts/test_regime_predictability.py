"""Test whether HMM regimes are predictable from lagged market features."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
from collections import Counter

def load_index_weekly_ret(fpath):
    """Load index daily CSV -> weekly return series indexed by iso_week."""
    df = pd.read_csv(fpath, encoding='utf-8-sig')
    # Find date column
    date_col = None
    for c in ['trade_date', 'date', 'dt']:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column in {fpath}: {df.columns.tolist()}")
    # Parse dates
    s = df[date_col]
    # Numeric columns: treat as YYYYMMDD (mixed format misinterprets epoch-based dates)
    if s.dtype.kind in ('i', 'f'):
        df['dt'] = pd.to_datetime(s.astype(int).astype(str), format='%Y%m%d')
    else:
        df['dt'] = pd.to_datetime(s, format='mixed')
    # Resample to weekly
    w_close = df.set_index('dt')['close'].astype(float).resample('W').last().dropna()
    w_close = w_close.to_frame('close')
    w_close['iso_w'] = w_close.index.strftime('%G-W%V')
    w_ret = w_close.set_index('iso_w')['close'].pct_change().dropna()
    return w_ret

# -- Load HMM --
hmm = pd.read_csv('data/regime_labels_hmm.csv')
hmm['dt'] = pd.to_datetime(hmm['dt'])
hmm = hmm.set_index('dt')
hmm['iso_w'] = hmm.index.strftime('%G-W%V')
hmm = hmm.set_index('iso_w')

# -- Load index data --
w300 = load_index_weekly_ret('data/raw/index_000300_daily.csv')
w500 = load_index_weekly_ret('data/raw/index_000905_daily.csv')
print(f"CSI300: {len(w300)} weeks, CSI500: {len(w500)} weeks")

# -- Merge --
common = hmm.index.intersection(w300.index).intersection(w500.index)
print(f"Common weeks: {len(common)}")

data = pd.DataFrame(index=common)
data['state'] = hmm.loc[common, 'hmm_state']
data['r300'] = w300.loc[common]
data['r500'] = w500.loc[common]
data['spread'] = data['r500'] - data['r300']
print(f"r300 NaN: {data['r300'].isna().sum()}, r500 NaN: {data['r500'].isna().sum()}")

# -- Lagged features --
for w in [2, 4, 8, 12]:
    data[f'r300_{w}w'] = data['r300'].rolling(w).sum()
    data[f'r500_{w}w'] = data['r500'].rolling(w).sum()
    data[f'spread_{w}w'] = data['spread'].rolling(w).sum()
    data[f'vol_{w}w'] = data['r300'].rolling(w).std()

# Target: t+1 regime
data['target'] = data['state'].shift(-1)
data['target_label'] = data['target'].map({0: 0, 2: 1, 3: 2, 4: 2})
target_names = {0: 'bear', 1: 'oscillation', 2: 'bull'}

# Drop rows with NaN features or target
before = len(data)
data = data.dropna(subset=['target_label'] + [c for c in data.columns if c.endswith('w')])
print(f"After dropna: {len(data)} (dropped {before - len(data)} header rows)")
print(f"Target dist: {dict(sorted(Counter(data['target_label']).items()))}")

feature_cols = [c for c in data.columns if c.endswith('w')]

# -- Train/Test split --
data['year'] = data.index.map(lambda x: int(x[:4]))
train = data[data['year'] <= 2023]
test = data[data['year'] > 2023]

X_train = train[feature_cols].values
y_train = train['target_label'].values
X_test = test[feature_cols].values
y_test = test['target_label'].values
print(f"\nTrain: {len(train)} (2021-2023), class={dict(sorted(Counter(y_train).items()))}")
print(f"Test:  {len(test)} (2024-2026), class={dict(sorted(Counter(y_test).items()))}")

# -- Baseline --
mc = Counter(y_train).most_common(1)[0]
baseline_acc = (y_test == mc[0]).mean()
print(f"\nBaseline (always '{target_names[mc[0]]}'): acc={baseline_acc:.1%}")

# -- Classifiers --
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}
for name, clf in [
    ('Logistic', LogisticRegression(max_iter=5000, C=0.1, random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('GBDT', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=42)),
]:
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    results[name] = {'pred': pred, 'proba': proba, 'acc': acc, 'cm': cm}
    
    print(f"\n{'='*55}")
    print(f"{name}: OOS acc={acc:.1%} (vs baseline {(acc-baseline_acc):+.1%})")
    print(f"        pred=bear  pred=osc  pred=bull")
    for i, label in enumerate(['bear', 'oscillation', 'bull']):
        print(f"  {label:>11} | {' '.join(f'{cm[i,j]:>9}' for j in range(3))}")
    for i, label in enumerate(['bear', 'oscillation', 'bull']):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"  {label}: prec={prec:.1%}, rec={rec:.1%}")

# -- Feature importance --
rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train_s, y_train)
importances = rf.feature_importances_
print(f"\n{'='*55}")
print("Top features (RF):")
for idx in np.argsort(importances)[::-1][:6]:
    print(f"  {feature_cols[idx]}: {importances[idx]:.4f}")

# -- Ensemble simulation --
print(f"\n{'='*55}")
print("SIMULATED ENSEMBLE PERFORMANCE")
print("=" * 55)

# Per-state JQ backtest total returns
perf = {0: 0.3345, 1: 0.6465, 2: 0.6908}
perf_names = {0: 'state_0(33%)', 1: 'state_2(65%)', 2: 'state_4(69%)'}

# Best strategy per true state from historical results
best_for = {0: 1, 1: 1, 2: 2}

valid_states = [s for s in [0,1,2] if (y_test == s).sum() > 0]

perfect_ret = sum(perf[best_for[s]] * (y_test == s).sum() / len(y_test) for s in valid_states)
equal_ret = sum(perf.values()) / 3

# GBDT results
gbdt_pred = results['GBDT']['pred']
gbdt_proba = results['GBDT']['proba']

hard_ret = sum(perf[s] * (gbdt_pred == s).sum() / len(y_test) for s in valid_states)
soft_ret = sum(perf[s] * gbdt_proba[:, s].sum() / len(y_test) for s in valid_states)

# Worst case (always-wrong)
wrong = {0: 2, 1: 2, 2: 1}
worst_ret = sum(perf[wrong[s]] * (y_test == s).sum() / len(y_test) for s in valid_states)

print(f"Equal weight                  : ret= {equal_ret:+.0%}")
print(f"Perfect foresight (theo max)  : ret= {perfect_ret:+.0%}")
print(f"Hard-switch (GBDT)            : ret= {hard_ret:+.0%}  ({hard_ret/perfect_ret:.0%} of max)")
print(f"Soft-blend (GBDT proba)       : ret= {soft_ret:+.0%}  ({soft_ret/perfect_ret:.0%} of max)")
print(f"Worst-case (always-wrong)     : ret= {worst_ret:+.0%}")

# Risk
cm = results['GBDT']['cm']
dangerous = cm[0, 2] + cm[1, 2]
total = len(y_test)
print(f"\nDangerous misclass (bear/osc→bull): {dangerous}/{total} ({dangerous/total:.1%})")

print(f"\n{'='*55}")
print("VERDICT")
print("=" * 55)
uplift = (soft_ret - equal_ret) / equal_ret
if soft_ret > equal_ret * 1.02:
    print(f">>> Soft-blend regime-adaptive IS worthwhile")
    print(f"    Uplift: {uplift:+.0%} vs equal weight")
    print(f"    Achieves {soft_ret/perfect_ret:.0%} of theoretical max")
    print(f"    >>> RECOMMEND: Build regime-adaptive ensemble with GBDT probability blending")
else:
    print(f">>> Benefit is marginal: {uplift:+.1%} vs equal weight")
    print(f"    >>> Consider simpler equal-weight blend instead")
