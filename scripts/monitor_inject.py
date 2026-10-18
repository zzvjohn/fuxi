"""One-shot FactorMiner injection for automation monitoring."""
import pickle, pandas as pd, numpy as np, json, sys, os
from datetime import datetime
from collections import Counter

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'research/factor_alchemy')

# ── Load data ──
daily = pd.read_csv('data/raw/daily_prices.csv', dtype={'ts_code': str, 'trade_date': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
daily = daily[(daily['trade_date'] >= pd.Timestamp('2025-11-03')) & (daily['trade_date'] <= pd.Timestamp('2026-08-03'))]
daily = daily.drop_duplicates(subset=['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])

close_df = daily.pivot_table(index='trade_date', columns='ts_code', values='close', aggfunc='last')
open_df = daily.pivot_table(index='trade_date', columns='ts_code', values='open', aggfunc='last')
high_df = daily.pivot_table(index='trade_date', columns='ts_code', values='high', aggfunc='last')
low_df = daily.pivot_table(index='trade_date', columns='ts_code', values='low', aggfunc='last')
vol_df = daily.pivot_table(index='trade_date', columns='ts_code', values='vol', aggfunc='last')

min_days = 40
valid = close_df.columns[close_df.count() >= min_days]
close_df = close_df[valid]; open_df = open_df[valid]
high_df = high_df[valid]; low_df = low_df[valid]; vol_df = vol_df[valid]

# ── Align with cache ──
cache = pickle.load(open('data/cache/existing_factor_pool.pkl', 'rb'))
shape_counts = Counter()
for k, v in cache.items():
    if hasattr(v, 'shape') and not k.startswith('_'):
        shape_counts[v.shape] += 1
majority_shape = shape_counts.most_common(1)[0][0]
sample_factor = [v for k, v in cache.items() if hasattr(v, 'shape') and v.shape == majority_shape and not k.startswith('_')][0]
common_cols = sample_factor.columns.intersection(close_df.columns)
close_df = close_df[common_cols]; open_df = open_df[common_cols]
high_df = high_df[common_cols]; low_df = low_df[common_cols]; vol_df = vol_df[common_cols]
print(f'Data: {close_df.shape} (aligned)')

# ── Import tools ──
from psi_orthogonalizer import psi_orthogonalize
from fri import compute_fri

# ── Find pending candidates ──
pool_df = pd.read_csv('data/passed_factor_pool.csv', encoding='utf-8-sig')
candidates = pool_df[pool_df['status'] == 'candidate']
injected = {
    'max_drawdown_duration', 'harvey_siddique_coskew',
    'earnings_pre_drift_alignment', 'capital_efficiency_proxy'
}
pending = candidates[~candidates['name'].isin(injected)]

print(f'Pending: {len(pending)} factors')
print('=' * 60)

existing_pool = {
    k: v for k, v in cache.items()
    if not k.startswith('_') and hasattr(v, 'shape') and v.shape[1] == len(common_cols)
}
print(f'Pool (aligned): {len(existing_pool)} factors with {len(common_cols)} stocks')
fwd = close_df.shift(-5) / close_df - 1
results = []

for idx, (_, row) in enumerate(pending.iterrows()):
    name = row['name']
    formula = str(row['formula'])
    direction = row.get('direction', 'long')

    print(f'[{idx+1}/{len(pending)}] {name} ...', end=' ', flush=True)

    try:
        local_ns = {
            'close_p': close_df, 'open_p': open_df, 'high_p': high_df,
            'low_p': low_df, 'volume_p': vol_df,
            'close': close_df, 'open_': open_df, 'high': high_df,
            'low': low_df, 'volume': vol_df,
            'np': np, 'pd': pd,
        }

        try:
            fv = eval(compile(formula, '<f>', 'eval'), local_ns)
        except SyntaxError:
            exec(compile(formula, '<f>', 'exec'), local_ns)
            fv = local_ns.get('factor')
            if fv is None:
                fv = local_ns.get('result')

        if fv is None:
            print('FAIL: no factor/result variable')
            results.append({'name': name, 'status': 'FAIL', 'error': 'no output'})
            continue

        if not isinstance(fv, pd.DataFrame):
            fv = pd.DataFrame(fv, index=close_df.index, columns=close_df.columns)

        fv = fv.reindex(index=close_df.index, columns=close_df.columns)
        if direction == 'short':
            fv = -fv

        # Winsorize
        for col in fv.columns:
            cd = fv[col].dropna()
            if len(cd) > 50:
                lo, hi = cd.quantile(0.01), cd.quantile(0.99)
                fv[col] = fv[col].clip(lo, hi)

        # Z-score
        fv_z = fv.subtract(fv.mean(axis=1), axis=0).div(
            fv.std(axis=1).replace(0, np.nan), axis=0
        )

        # Psi orthogonalization
        psi = psi_orthogonalize(name, fv_z, existing_pool, verbose=False)

        if psi.rejection_recommended:
            print(f'PSI-REJECT R2={psi.r_squared:.3f}')
            results.append({'name': name, 'status': 'PSI_REJECT', 'psi_r2': psi.r_squared})
            continue

        # FRI
        fr = compute_fri(name, psi.orthogonalized_values, fwd, None, existing_pool, verbose=False)

        psi_flag = 'PSI(clean)' if psi.r_squared < 0.1 else f'PSI(R2={psi.r_squared:.2f})'
        print(
            f'FRI={fr.fri:.3f}({fr.grade}) '
            f'P={fr.precision:.3f} S={fr.persistence:.3f} '
            f'C={fr.consistency:.3f} N={fr.novelty:.3f} '
            f'{psi_flag}'
        )

        results.append({
            'name': name, 'status': 'PASS',
            'fri': float(fr.fri), 'grade': fr.grade,
            'precision': float(fr.precision),
            'persistence': float(fr.persistence),
            'consistency': float(fr.consistency),
            'novelty': float(fr.novelty),
            'psi_r2': float(psi.r_squared),
            'max_corr': float(getattr(fr, 'max_corr_with_existing', 0)),
        })

    except Exception as e:
        msg = str(e)[:150]
        print(f'FAIL: {msg}')
        results.append({'name': name, 'status': 'FAIL', 'error': msg})

print()
print(f'=== Results: {sum(1 for r in results if r["status"]=="PASS")}/{len(results)} PASS ===')
for r in results:
    if r['status'] == 'PASS':
        print(f'  OK  {r["name"]}: FRI={r["fri"]:.3f}/{r["grade"]} PsiR2={r["psi_r2"]:.3f} N={r["novelty"]:.3f} S={r["persistence"]:.3f}')
    elif r['status'] == 'PSI_REJECT':
        print(f'  REJ {r["name"]}: PSI_REJECT R2={r["psi_r2"]:.3f}')
    else:
        print(f'  FAIL {r["name"]}: {r.get("error","?")[:80]}')

# Save
with open('data/injection_round_20260803.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f'\nSaved: data/injection_round_20260803.json')
