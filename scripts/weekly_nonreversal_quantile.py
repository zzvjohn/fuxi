# -*- coding: utf-8 -*-
"""
Phase 1b — 非反转因子分组回测验证
============================================================
对候选因子做周频分层回测，确认Q1-Q5单调性和多空收益
非反转候选: 低PB 低PS 低PE 低波动 异常量 振幅 影线 均线偏离 换手稳定
============================================================
"""
import pandas as pd, numpy as np, os, sys, io
import warnings; warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW  = 'e:/quant/data/raw'
OUT  = 'e:/quant/reports'

daily = pd.read_csv(os.path.join(RAW, 'daily_prices.csv'))
basic = pd.read_csv(os.path.join(RAW, 'daily_basic.csv'))
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
daily = daily[~daily['ts_code'].str.startswith(('688','689'))]
basic = basic[~basic['ts_code'].str.startswith(('688','689'))]

all_dates = sorted(daily['trade_date'].unique())
weekly_dates = [d for i,d in enumerate(all_dates) if i>0 and d.weekday()==0]

def last_day(target, dates):
    for d in reversed(dates):
        if d < target: return d
    return dates[0]

def ndays(n, target, dates):
    idx = dates.index(target) if target in dates else len(dates)-1
    return dates[max(0, idx-n)]

# ============================================================
# 构建面板
# ============================================================
print("构建非反转因子面板...")
rows = []

for wi, wd in enumerate(weekly_dates):
    if wi < 10: continue
    if wi % 50 == 0: print(f"  {wd.strftime('%Y-%m')} ({wi}/{len(weekly_dates)})")
    
    prev = last_day(wd, all_dates)
    d5 = ndays(5, prev, all_dates)
    d20 = ndays(20, prev, all_dates)
    d60 = ndays(60, prev, all_dates)

    # 量价数据
    p = daily[daily['trade_date']==prev].copy()
    if len(p) < 100: continue
    for d, col in [(d5,'c5'),(d20,'c20'),(d60,'c60')]:
        tmp = daily[daily['trade_date']==d][['ts_code','close']].rename(columns={'close':col})
        p = p.merge(tmp, on='ts_code', how='left')
    p = p[(p['close']>0)&(p['c5'].notna())&(p['c60'].notna())]

    # 基本面
    b = basic[basic['trade_date']==prev][['ts_code','pb','ps','pe_ttm','total_mv','turnover_rate']]
    p = p.merge(b, on='ts_code', how='left')
    p = p[p['pb'] > 0]

    if len(p) < 100: continue
    c,o,h,l,v,pc = p['close'],p['open'],p['high'],p['low'],p['vol'],p['pre_close']

    # === 非反转因子 ===
    p['date'] = wd
    p['ret_5d'] = c / p['c5'] - 1

    # Q组: 估值
    p['Q1_pb']   = -np.log(p['pb'])
    p['Q2_ps']   = -np.log(p['ps'])
    p['Q3_pe']   = -np.log(p['pe_ttm'])
    p['Q4_size'] = -np.log(p['total_mv'])

    # V组: 量价信息
    p['amp'] = (h - l) / c
    # 5日成交量数据
    d5_data = daily[(daily['trade_date']>=d5)&(daily['trade_date']<=prev)]
    to5 = d5_data.groupby('ts_code')['vol'].mean()
    to5s = d5_data.groupby('ts_code')['vol'].std()
    to20 = daily[(daily['trade_date']>=d20)&(daily['trade_date']<=prev)]
    to20m = to20.groupby('ts_code')['vol'].mean()
    vol5 = d5_data.groupby('ts_code')['pct_chg'].std()

    p = p.join(to5.rename('to5'), on='ts_code')
    p = p.join(to5s.rename('to5s'), on='ts_code')
    p = p.join(to20m.rename('to20'), on='ts_code')
    p = p.join(vol5.rename('vol5'), on='ts_code')

    vr = (p['to5'] / p['to20']).clip(0.3, 5)
    p['V1_abn_vol'] = -abs(vr - 1)
    p['V2_amp_brk'] = p['amp'] * vr

    # T组: 技术
    body = abs((c - o).values)
    p['us'] = (h - p[['close','open']].max(axis=1)) / (body + 1e-8)
    p['ls'] = (p[['close','open']].min(axis=1) - l) / (body + 1e-8)
    p['T1_shadow'] = p['ls'] - p['us']
    p['T2_ma_dev'] = -(c - p['c5']) / (p['c5'] + 1e-8)

    # P组: 保护
    p['P1_lowvol'] = -p['vol5']
    p['P2_str']    = -p['to5s'] / (p['to5'] + 1e-8)

    keep_cols = ['date','ts_code','ret_5d','close',
        'Q1_pb','Q2_ps','Q3_pe','Q4_size',
        'V1_abn_vol','V2_amp_brk',
        'T1_shadow','T2_ma_dev',
        'P1_lowvol','P2_str']
    rows.append(p[keep_cols].dropna(subset=['Q1_pb']))

panel = pd.concat(rows, ignore_index=True)
print(f"面板: {len(panel)} rows")

# ============================================================
# 未来1周收益 + 5分组回测
# ============================================================
fwd_rets = {}
for wi in range(len(weekly_dates)-1):
    cw = weekly_dates[wi]
    nw = weekly_dates[wi+1]
    np_prev = last_day(nw, all_dates)
    pc = daily[daily['trade_date']==cw][['ts_code','close']].rename(columns={'close':'c_c'})
    pn = daily[daily['trade_date']==np_prev][['ts_code','close']].rename(columns={'close':'c_n'})
    f = pc.merge(pn, on='ts_code', how='inner')
    f['fwd'] = f['c_n']/f['c_c']-1
    for _,r in f.iterrows():
        fwd_rets[(cw, r['ts_code'])] = r['fwd']

panel['fwd_ret'] = panel.apply(lambda r: fwd_rets.get((r['date'],r['ts_code']),np.nan), axis=1)
panel = panel.dropna(subset=['fwd_ret'])

# 分组回测
FACTORS = ['Q1_pb','Q2_ps','Q4_size',
           'V1_abn_vol','V2_amp_brk',
           'T1_shadow','T2_ma_dev',
           'P1_lowvol','P2_str']
N_Q = 5

print("\n" + "="*70)
print("非反转因子分组回测（Q1最高→Q5最低）")
print("="*70)

results = []
for fac in FACTORS:
    # 每期分组
    panel['q'] = np.nan
    for d, grp in panel.groupby('date'):
        if len(grp) < 50: continue
        grp_sorted = grp.sort_values(fac, ascending=False)
        n = len(grp_sorted)
        labels = np.repeat(range(1, N_Q+1), n//N_Q+1)[:n]
        panel.loc[grp_sorted.index, 'q'] = labels

    valid_q = panel[panel['q'].notna()]
    if len(valid_q) < 500: continue

    # 每期每分组平均收益
    q_ret = valid_q.groupby(['date','q'])['fwd_ret'].mean().unstack()
    q_avg = q_ret.mean()
    q_cum = (1 + q_ret).cumprod() - 1

    # IC
    ic_by_wk = valid_q.groupby('date').apply(lambda g: g[fac].corr(g['fwd_ret']) if len(g)>=30 else np.nan).dropna()
    icir = ic_by_wk.mean()/ic_by_wk.std() if ic_by_wk.std()>0 else 0

    results.append({
        '因子': fac, 'ICIR': round(icir,3),
        'Q1': f"{q_avg.get(1,0)*100:.2f}%",
        'Q5': f"{q_avg.get(5,0)*100:.2f}%",
        'Q1-Q5': f"{(q_avg.get(1,0)-q_avg.get(5,0))*100:.2f}%",
        '单向性': '✓' if all(q_avg.get(i,0)>=q_avg.get(i+1,0) for i in range(1,N_Q)) else '✗',
        'n_weeks': len(q_ret.dropna()),
    })

# 输出
rdf = pd.DataFrame(results).sort_values('ICIR', ascending=False)
for _, r in rdf.iterrows():
    mono = '单调' if r['单向性']=='✓' else '非单调'
    print(f"  {r['因子']:12s} ICIR={r['ICIR']:+.3f}  Q1={r['Q1']}  Q5={r['Q5']}  spread={r['Q1-Q5']}  {mono}")

rdf.to_csv(os.path.join(OUT, 'nonreversal_quantile_20260528.csv'), encoding='utf-8-sig')
print(f"\n结果: {OUT}/nonreversal_quantile_20260528.csv")
