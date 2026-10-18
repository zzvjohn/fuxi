# -*- coding: utf-8 -*-
"""新因子正交检验 + ICIR (简化版)"""
import pandas as pd, numpy as np, os
RAW_DIR = 'e:/quant/data/raw'

print("加载...")
daily = pd.read_csv(os.path.join(RAW_DIR, 'daily_prices.csv'), dtype={'trade_date': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic = pd.read_csv(os.path.join(RAW_DIR, 'daily_basic.csv'), dtype={'trade_date': str})
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

close_wide = daily.pivot_table('close', 'trade_date', 'ts_code', aggfunc='last')
open_wide  = daily.pivot_table('open', 'trade_date', 'ts_code', aggfunc='last')
high_wide  = daily.pivot_table('high', 'trade_date', 'ts_code', aggfunc='last')
low_wide   = daily.pivot_table('low', 'trade_date', 'ts_code', aggfunc='last')
vol_wide   = daily.pivot_table('vol', 'trade_date', 'ts_code', aggfunc='last')
mcap_wide  = basic.pivot_table('total_mv', 'trade_date', 'ts_code', aggfunc='last')
pb_wide    = basic.pivot_table('pb', 'trade_date', 'ts_code', aggfunc='last')
ps_wide    = basic.pivot_table('ps', 'trade_date', 'ts_code', aggfunc='last')

common = sorted(set(close_wide.columns) & set(open_wide.columns) & set(high_wide.columns) &
                set(low_wide.columns) & set(vol_wide.columns) &
                set(mcap_wide.columns) & set(pb_wide.columns) & set(ps_wide.columns))
print(f"共同股票: {len(common)}")

# 截面: 2024-2025, 每月15号左右取一个
all_dates = close_wide.index
# 取2024-01-01 之后的日期
mask = all_dates >= pd.Timestamp('2024-01-01')
valid_dates = all_dates[mask]
# 每30天取一个截面
step = max(1, len(valid_dates) // 24)
snapshot_dates = valid_dates[::step][:24].tolist()
print(f"截面: {len(snapshot_dates)} 个, {snapshot_dates[0].date()} ~ {snapshot_dates[-1].date()}")

def zscore(x):
    m=np.nanmedian(x); s=np.nanstd(x)
    return np.where(s>0,(x-m)/s,0)

all_ics = []; all_corrs = []; all_combo_ics = []
factor_names = ['size','ps','pb','str','gap_rev','rpv','shadow']

LOOKBACK = 65; FWD = 20

for md in snapshot_dates:
    # 找准确的索引位置
    date_positions = np.where(all_dates == md)[0]
    if len(date_positions) == 0:
        pos = all_dates.get_indexer([md], method='pad')[0]
    else:
        pos = date_positions[0]

    if pos < LOOKBACK or pos + FWD >= len(all_dates):
        continue

    # 提取数据
    cl = close_wide[common].values[pos-LOOKBACK:pos+1].astype(np.float64)
    op = open_wide[common].values[pos-LOOKBACK:pos+1].astype(np.float64)
    hi = high_wide[common].values[pos-LOOKBACK:pos+1].astype(np.float64)
    lo = low_wide[common].values[pos-LOOKBACK:pos+1].astype(np.float64)
    vo = vol_wide[common].values[pos-LOOKBACK:pos+1].astype(np.float64)

    # 基本面
    fi = mcap_wide.index.get_indexer([md], method='pad')[0]
    mc  = mcap_wide[common].iloc[fi].values.astype(np.float64)
    pbb = pb_wide[common].iloc[fi].values.astype(np.float64)
    pss = ps_wide[common].iloc[fi].values.astype(np.float64)

    ok = (~np.isnan(cl[-1]) & (mc>0) & (pbb>0) & (pss>0) &
          ~np.isnan(op[-1]) & ~np.isnan(hi[-1]) & ~np.isnan(lo[-1]))
    if ok.sum() < 100: continue

    cl=cl[:,ok]; op=op[:,ok]; hi=hi[:,ok]; lo=lo[:,ok]; vo=vo[:,ok]
    mc=mc[ok]; pbb=pbb[ok]; pss=pss[ok]; n=cl.shape[1]

    # v5b 因子
    f_size = zscore(-np.log(mc))
    f_ps   = zscore(-np.log(pss))
    f_pb   = zscore(-np.log(pbb))
    to20 = vo[-21:-1]
    f_str = zscore(-np.nanstd(to20,axis=0)/np.maximum(np.nanmean(to20,axis=0),1e-8))

    # 隔夜反转
    gap = op[-21:] / cl[-22:-1] - 1
    f_gap = zscore(-np.nanmean(gap, axis=0))

    # RPV
    dr = cl[-21:] / cl[-22:-1] - 1; dv = vo[-21:]
    rpv = np.zeros(n)
    for j in range(n):
        rj=dr[:,j]; vj=dv[:,j]; okj=(~np.isnan(rj))&(~np.isnan(vj))&(vj>0)
        if okj.sum()>=10:
            rpv[j]=np.corrcoef(np.argsort(np.argsort(rj[okj])),np.argsort(np.argsort(vj[okj])))[0,1]
    f_rpv = zscore(-rpv)

    # 影线
    upper = hi - np.maximum(cl, op)
    lower = np.minimum(cl, op) - lo
    body = np.maximum(np.abs(cl - op), 1e-8)
    shadow = np.nanmean((upper[-21:]+lower[-21:])/body[-21:], axis=0)
    f_shadow = zscore(shadow)

    # Forward 收益
    fwd_close = close_wide[common].values[pos+FWD, ok].astype(np.float64)
    fwd_ret = fwd_close / cl[-1] - 1

    f_arr = np.column_stack([f_size,f_ps,f_pb,f_str,f_gap,f_rpv,f_shadow])
    okr = ~np.any(np.isnan(f_arr),axis=1) & ~np.isnan(fwd_ret)
    if okr.sum()<100: continue
    f_arr=f_arr[okr]; r=fwd_ret[okr]

    # 相关系数
    corr = np.corrcoef(f_arr.T)
    all_corrs.append(pd.DataFrame(corr, columns=factor_names, index=factor_names))

    # 单因子 IC
    ics={}
    for i,fn in enumerate(factor_names): ics[fn]=np.corrcoef(f_arr[:,i],r)[0,1]
    all_ics.append(ics)

    # 组合
    v5b = 0.35*f_arr[:,0]+0.25*f_arr[:,1]+0.22*f_arr[:,2]+0.18*f_arr[:,3]
    combos = {
        'v5b': ('v5b',v5b),
        'v5b+gap': ('+隔夜', 0.85*v5b+0.15*f_arr[:,4]),
        'v5b+rpv': ('+RPV', 0.85*v5b+0.15*f_arr[:,5]),
        'v5b+shadow': ('+影线', 0.85*v5b+0.15*f_arr[:,6]),
        'v5b+gap+rpv': ('+隔夜+RPV', 0.80*v5b+0.10*f_arr[:,4]+0.10*f_arr[:,5]),
        'v5b+gap+shadow': ('+隔夜+影线', 0.80*v5b+0.10*f_arr[:,4]+0.10*f_arr[:,6]),
        'v5b+rpv+shadow': ('+RPV+影线', 0.80*v5b+0.10*f_arr[:,5]+0.10*f_arr[:,6]),
        'v5b+all3': ('+全部3', 0.75*v5b+0.08*f_arr[:,4]+0.08*f_arr[:,5]+0.09*f_arr[:,6]),
    }
    combo_ics={}
    for k,(nm,sc) in combos.items(): combo_ics[k]=(nm,np.corrcoef(sc,r)[0,1])
    all_combo_ics.append(combo_ics)

n_snap = len(all_ics)
print(f"\n有效截面: {n_snap}")

# ==== 相关系数 ====
avg_corr=sum(all_corrs)/n_snap
print(f"\n{'='*60}")
print("因子相关系数矩阵:")
print(f"{'='*60}")
print(avg_corr.round(3).to_string())

print(f"\n新因子 vs v5b 正交性:")
v5b=['size','ps','pb','str']
for nc in ['gap_rev','rpv','shadow']:
    cors=[abs(avg_corr.loc[nc,vc]) for vc in v5b]
    a=np.mean(cors); f="OK_indep" if a<0.15 else ("OK_accept" if a<0.25 else "WARN")
    print(f"  {nc:10s} avg|r|={a:.3f} {f}")

# ==== 单因子ICIR ====
print(f"\n{'='*60}")
print("单因子 ICIR:")
print(f"{'因子':>10s}  {'IC均值':>7s}  {'IC std':>7s}  {'ICIR':>7s}")
print("-"*40)
for fn in factor_names:
    v=[a[fn] for a in all_ics]
    m=np.mean(v); s=np.std(v); icir=m/s if s>0 else 0
    print(f"{fn:>10s}  {m:+.4f}   {s:.4f}   {icir:+.3f}")

# ==== 组合ICIR ====
print(f"\n{'='*60}")
print("组合 ICIR:")
print(f"{'组合':>20s}  {'ICIR':>7s}")
print("-"*35)
best_name, best_icir = '', -999
for key in ['v5b','v5b+gap','v5b+rpv','v5b+shadow','v5b+gap+rpv','v5b+gap+shadow','v5b+rpv+shadow','v5b+all3']:
    icv=[]
    for c in all_combo_ics:
        if key in c: icv.append(c[key][1])
    if len(icv)<3: continue
    m=np.mean(icv); s=np.std(icv); icir=m/s if s>0 else 0
    nm=combos[key][0]
    print(f"{nm:>20s}  {icir:+.3f}")
    if icir>best_icir: best_name,best_icir=nm,icir

print(f"\n最优: {best_name}  (ICIR={best_icir:.3f})")
