"""xgb_alpha 复合 vs V3 参考复合 — 测试期 (2025-2026) ICIR 对比."""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

OUT = Path(__file__).parent / 'output'
with open(OUT/'xgb_alpha_factor.pkl','rb') as f: xgb_alpha = pickle.load(f)
with open(OUT/'factor_cache.pkl','rb') as f: cache = pickle.load(f)

ret1m = cache['ret_1m']
common_dates = sorted(set(xgb_alpha.index) & set(ret1m.index))

def rp(s): return s.dropna().rank(pct=True)

def calc_icir(scores_df):
    icl = []
    for d in common_dates:
        sc = scores_df.loc[d].dropna()
        if len(sc) < 30: continue
        idx = ret1m.index.get_loc(d)
        if idx+1 >= len(ret1m): continue
        fwd = ret1m.iloc[idx+1].dropna()
        cs = sc.index.intersection(fwd.index)
        if len(cs) < 30: continue
        ic = spearmanr(sc[cs].values, fwd[cs].values)[0]
        if not np.isnan(ic): icl.append(ic)
    if len(icl) < 10: return None, None, len(icl)
    a = np.array(icl)
    return a.mean()/(a.std()+1e-9), a.mean(), len(icl)

def make_pair(a_df, b_df, label):
    comp = pd.DataFrame(index=common_dates, columns=a_df.columns.union(b_df.columns), dtype=float)
    for d in common_dates:
        sa = a_df.loc[d].dropna()
        sb = b_df.loc[d].dropna()
        cs = sa.index.intersection(sb.index)
        if len(cs) < 30: continue
        comp.loc[d, cs] = rp(sa[cs]) * rp(sb[cs])
    return label, *calc_icir(comp)

def make_solo(a_df, label):
    comp = pd.DataFrame(index=common_dates, columns=a_df.columns, dtype=float)
    for d in common_dates:
        sa = a_df.loc[d].dropna()
        if len(sa) >= 30:
            comp.loc[d, sa.index] = rp(sa)
    return label, *calc_icir(comp)

results = []

# xgb_alpha × clean factors
clean = ['high_low_range','turnover_std','dollar_vol_20d','money_flow_20',
         'overnight_5d','tvma_20','ret_3m','ret_6m','turnover_change','vroc_12']
for fn in clean:
    if fn in cache:
        r = make_pair(xgb_alpha, cache[fn], f'xgb_alpha × {fn}')
        if r[1] is not None: results.append(r)

# xgb solo
r = make_solo(xgb_alpha, 'xgb_alpha (solo)')
if r[1] is not None: results.append(r)

# V3 ref composites
if 'overnight_5d' in cache and 'tvma_20' in cache:
    r = make_pair(cache['overnight_5d'], cache['tvma_20'], 'V3: overnight × tvma')
    if r[1] is not None: results.append(r)

if 'dollar_vol_20d' in cache and 'turnover_std' in cache:
    r = make_pair(cache['dollar_vol_20d'], cache['turnover_std'], 'V3: dollar_vol × turnover_std')
    if r[1] is not None: results.append(r)

if 'money_flow_20' in cache and 'bp' in cache and 'ret_3m' in cache:
    comp = pd.DataFrame(index=common_dates, columns=cache['money_flow_20'].columns, dtype=float)
    for d in common_dates:
        mf = cache['money_flow_20'].loc[d].dropna()
        bp_s = cache['bp'].loc[d].dropna()
        r3m = cache['ret_3m'].loc[d].dropna()
        cs = mf.index.intersection(bp_s.index).intersection(r3m.index)
        if len(cs) < 30: continue
        vm = rp(rp(bp_s[cs]) * rp(-r3m[cs]))
        comp.loc[d, cs] = rp(mf[cs]) * vm
    icir, mean_ic, n = calc_icir(comp)
    if icir is not None:
        results.append(('V3: money_flow × value_momentum', icir, mean_ic, n))

# Sort
results.sort(key=lambda x: -x[1])
print(f'{"Composite":50s} {"ICIR":>8s} {"MeanIC":>8s}  n')
print('-'*75)
for name, icir, mean_ic, n in results:
    print(f'{name:50s} {icir:+8.4f} {mean_ic:+8.4f} {n:3d}')
