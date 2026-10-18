"""诊断 Tushare vs JQData 价格差异 — 精简版"""
import sys, numpy as np, pandas as pd, jqdatasdk as jq
sys.path.insert(0, 'e:/quant')
from credentials import get_jqdata_auth
from scipy.stats import spearmanr
get_jqdata_auth()

TS_CODE = '300750.SZ'  # 宁德时代 — 数据最干净
JQ_CODE = '300750.XSHE'
START, END = '2025-04-01', '2026-04-17'

# ── TS ──
ts_raw = pd.read_csv('e:/quant/data/raw/daily_prices.csv',
    usecols=['ts_code','trade_date','open','high','low','close'],
    dtype={'ts_code':str,'trade_date':str}, low_memory=False)
ts_raw['trade_date'] = pd.to_datetime(ts_raw['trade_date'].astype(str).str[:10], format='mixed')
ts = ts_raw[ts_raw['ts_code'] == TS_CODE].set_index('trade_date')[['open','close','high','low']]
ts = ts[~ts.index.duplicated(keep='first')]  # 去重
ts = ts[(ts.index >= START) & (ts.index <= END)].sort_index()
print(f"TS: {len(ts)} days, {ts.index[0].date()} ~ {ts.index[-1].date()}")

# ── JQ post ──
jq_df = jq.get_price(JQ_CODE, start_date=START, end_date=END,
    frequency='daily', fields=['open','close','high','low'], fq='post', panel=False)
print(f"JQ: {len(jq_df)} days, {jq_df.index[0].date()} ~ {jq_df.index[-1].date()}")

# ── 对齐 ──
common = ts.index.intersection(jq_df.index)
ts = ts.loc[common]; jq_df = jq_df.loc[common]
print(f"Common: {len(common)} days")

# ── OHLC 对比 ──
print(f"\n{'Field':>6s} {'Pearson':>9s} {'Spearman':>9s} {'MAD%':>7s}  判定")
for f in ['close','open','high','low']:
    t = ts[f].values; j = jq_df[f].values
    v = ~(np.isnan(t) | np.isnan(j))
    r = np.corrcoef(t[v], j[v])[0,1]
    rho, _ = spearmanr(t[v], j[v])
    mad = np.median(np.abs(t[v]-j[v]) / np.abs(j[v]) * 100)
    sym = "O" if rho > 0.95 else ("?" if abs(rho) > 0.7 else "X")
    ps = f"+{r:.4f}" if r >= 0 else f"{r:.4f}"
    ss = f"+{rho:.4f}" if rho >= 0 else f"{rho:.4f}"
    print(f"{f:>6s} {ps:>9s} {ss:>9s} {mad:7.2f}%  {sym}")

# ── 前复权 vs 后复权 ──
print(f"\n--- 宁德时代: 三种复权对比 ---")
jq_pre = jq.get_price(JQ_CODE, start_date=START, end_date=END,
    frequency='daily', fields=['open','close','high','low'], fq='pre', panel=False)
common2 = ts.index.intersection(jq_pre.index)
ts2 = ts.loc[common2]; jq_p = jq_df.loc[common2]; jq_r = jq_pre.loc[common2]

for f in ['close','high']:
    t=ts2[f].values; jp=jq_p[f].values; jr=jq_r[f].values
    v=~(np.isnan(t)|np.isnan(jp)|np.isnan(jr))
    r_ts_post,_=spearmanr(t[v],jp[v])
    r_ts_pre,_=spearmanr(t[v],jr[v])
    r_pre_post,_=spearmanr(jp[v],jr[v])
    print(f"  {f}: TS-vs-JQ(post)={r_ts_post:+.4f}  TS-vs-JQ(pre)={r_ts_pre:+.4f}  JQ(pre-vs-post)={r_pre_post:+.4f}")

# ── 同代码 ATR(14) ──
print(f"\n--- ATR(14) 同代码对比 ---")
def atr14(h,l,c):
    tr=np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    return pd.Series(tr).rolling(14,min_periods=1).mean().values

a_ts=atr14(ts2['high'].values,ts2['low'].values,ts2['close'].values)
a_jq=atr14(jq_p['high'].values,jq_p['low'].values,jq_p['close'].values)
v=~(np.isnan(a_ts[14:])|np.isnan(a_jq[14:]))
pearson=np.corrcoef(a_ts[14:][v],a_jq[14:][v])[0,1]
rho,_=spearmanr(a_ts[14:][v],a_jq[14:][v])
ps = f"+{pearson:.4f}" if pearson >= 0 else f"{pearson:.4f}"
ss = f"+{rho:.4f}" if rho >= 0 else f"{rho:.4f}"
print(f"  ATR(14): Pearson={ps} Spearman={ss}")

# ── 单日价格样本 ──
print(f"\n--- 宁德时代 单日样本 ---")
print(f"  {'Date':>12s} {'TS_close':>10s} {'JQ_close':>10s} {'d%':>7s} | {'TS_high':>10s} {'JQ_high':>10s} {'d%':>7s}")
for i in range(5):
    d=common2[i].strftime('%Y-%m-%d')
    cd=abs(ts2['close'].iloc[i]-jq_p['close'].iloc[i])/jq_p['close'].iloc[i]*100
    hd=abs(ts2['high'].iloc[i]-jq_p['high'].iloc[i])/jq_p['high'].iloc[i]*100
    print(f"  {d} {ts2['close'].iloc[i]:10.2f} {jq_p['close'].iloc[i]:10.2f} {cd:6.3f}% | {ts2['high'].iloc[i]:10.2f} {jq_p['high'].iloc[i]:10.2f} {hd:6.3f}%")

print("\n完成!")
