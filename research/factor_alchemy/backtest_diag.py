# -*- coding: utf-8 -*-
"""诊断: no_regime_allf 本地回测方法论校验
- 复刻 regime_lasso WFA OOS: 固定全样本权重 + GROSS(无成本) + 仅OOS窗口(2024-09+)
- 对比 pipeline WFA OOS = +18.8% / Sharpe 1.28
若接近 => 方法论OK, 全样本负值=早期年份拖累(regime依赖)
"""
import sys, time, pickle, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, 'research/factor_alchemy')
from data.loader import load_all_data
from evaluation.ic_analysis import compute_ic_icir
warnings.filterwarnings('ignore')
t0=time.time()
data = load_all_data('2021-01-01','2026-06-25')
close = data['price_data'].get('close')
cw = close.resample('W').last(); cw.index=pd.to_datetime(cw.index)
fr = cw.shift(-1)/cw - 1
with open('research/factor_alchemy/output/factor_cache.pkl','rb') as f: cache=pickle.load(f)
def zc(s):
    mu, sd = s.mean(), s.std()
    return s*0 if (sd==0 or np.isnan(sd)) else (s-mu)/sd
nr={'relative_spread_proxy':0.3062,'bargaining_power_proxy':0.2457,'min_ret_1m':0.1786,'avg_turnover_3m':0.1650,'atr_14':0.1044}
fac=list(nr.keys())
# ICIR signs
signs={}
for f in fac:
    ff=cache[f].loc[fr.index,[c for c in cache[f].columns if c in fr.columns]]
    r=compute_ic_icir(ff,fr[[c for c in ff.columns]],min_samples=10)
    ic=r.get('icir',0) if isinstance(r,dict) else 0
    signs[f]=1.0 if (ic is None or np.isnan(ic) or ic>=0) else -1.0
def backtest(weeks, cost_on):
    pr=[]; h=set()
    for wk in weeks:
        comp=pd.Series(0.0,index=fr.columns)
        for f in fac:
            s=signs[f]*cache[f].loc[wk].reindex(comp.index)
            comp+=nr[f]*zc(s)
        comp=comp.dropna()
        if len(comp)<30: h=set(); continue
        tgt=set(comp.nlargest(30).index)
        c=0.0
        if h:
            to=(len(h-tgt)+len(tgt-h))/(2*30); c=to*0.0072 if cost_on else 0.0
        fr_lw = fr.loc[wk]
        v=[s for s in tgt if s in fr_lw.index and pd.notna(fr_lw[s])]
        if len(v)<15: h=set(); continue
        pr.append(fr.loc[wk,v].mean()-c); h=tgt
    pr=pd.Series(pr); cum=(1+pr).cumprod()-1
    ann=pr.mean()*52; sd=pr.std()*np.sqrt(52); sh=ann/sd if sd>0 else 0
    mdd=(cum-cum.cummax()).min()
    return dict(annual=round(ann,4),sharpe=round(sh,3),maxdd=round(mdd,4),n=len(pr),cum=round(cum.iloc[-1],3))
all_w=list(fr.index)
oos_w=[w for w in fr.index if w>=pd.Timestamp('2024-09-01')]
print('GROSS full-sample :', backtest(all_w,False), flush=True)
print('GROSS OOS(24-09+) :', backtest(oos_w,False), flush=True)
print('NET   OOS(24-09+) :', backtest(oos_w,True), flush=True)
print('DONE %.0fs'%(time.time()-t0), flush=True)
