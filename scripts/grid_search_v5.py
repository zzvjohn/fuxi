# -*- coding: utf-8 -*-
"""v5 参数网格搜索 — 本地回测筛选 → Top3上聚宽"""
import pandas as pd,numpy as np,os,sys
import warnings; warnings.filterwarnings('ignore')

RAW='e:/quant/data/raw'
daily=pd.read_csv(os.path.join(RAW,'daily_prices.csv'))
basic=pd.read_csv(os.path.join(RAW,'daily_basic.csv'))
daily['trade_date']=pd.to_datetime(daily['trade_date'],format='mixed')
basic['trade_date']=pd.to_datetime(basic['trade_date'],format='mixed')
daily=daily[~daily['ts_code'].str.startswith(('688','689'))]
basic=basic[~basic['ts_code'].str.startswith(('688','689'))]

all_dates=sorted(daily['trade_date'].unique())
weekly_dates=[d for i,d in enumerate(all_dates) if i>0 and d.weekday()==0]

def last_day(t,dates):
    for d in reversed(dates):
        if d<t:return d
    return dates[0]

def ndays(n,t,dates):
    idx=dates.index(t)if t in dates else len(dates)-1
    return dates[max(0,idx-n)]

# 预计算所有周期因子分（共享数据，各参数只变权重和过滤）
print("预计算因子分...")
factor_scores={}  # date -> {code: {size,ps,pb,str,vol}}
for wi,wd in enumerate(weekly_dates):
    if wi<10:continue
    if wi%80==0:print(f"  {wd.strftime('%Y-%m')}")
    prev=last_day(wd,all_dates)
    d5=ndays(5,prev,all_dates);d20=ndays(20,prev,all_dates)
    
    p=daily[daily['trade_date']==prev].copy()
    if len(p)<100:continue
    for d,col in [(d5,'c5'),(d20,'c20')]:
        tmp=daily[daily['trade_date']==d][['ts_code','close']].rename(columns={'close':col})
        p=p.merge(tmp,on='ts_code',how='left')
    p=p[p['close']>0]
    
    b=basic[basic['trade_date']==prev][['ts_code','pb','ps','total_mv','pe_ttm']]
    p=p.merge(b,on='ts_code',how='left')
    p=p[(p['pb']>0)&(p['ps']>0)&(p['pe_ttm']>0)&(p['total_mv']>=15)]
    if len(p)<60:continue
    
    d5_d=daily[(daily['trade_date']>=d5)&(daily['trade_date']<=prev)]
    to20=d5_d.groupby('ts_code')['vol'].agg(['mean','std'])
    vol5=d5_d.groupby('ts_code')['close'].apply(lambda x:np.log(x/x.shift(1)).std())
    
    scores={}
    for _,r in p.iterrows():
        s=r['ts_code']
        sc={}
        sc['size']=-np.log(r['total_mv'])
        sc['ps']=-np.log(r['ps'])
        sc['pb']=-np.log(r['pb'])
        if s in to20.index:
            sc['str']=-to20.loc[s,'std']/max(to20.loc[s,'mean'],1e-8)
        else:sc['str']=0
        sc['vol']=vol5.get(s,0.05)
        scores[s]=sc
    factor_scores[wd]=scores

# 未来收益
fwd_rets={}
for wi in range(len(weekly_dates)-1):
    cw=weekly_dates[wi];nw=weekly_dates[wi+1]
    np_prev=last_day(nw,all_dates)
    pc=daily[daily['trade_date']==cw][['ts_code','close']].rename(columns={'close':'c_c'})
    pn=daily[daily['trade_date']==np_prev][['ts_code','close']].rename(columns={'close':'c_n'})
    f=pc.merge(pn,on='ts_code',how='inner');f['fwd']=f['c_n']/f['c_c']-1
    for _,r in f.iterrows():fwd_rets[(cw,r['ts_code'])]=r['fwd']

# 网格
SIZE_WTS=[0.25,0.30,0.35,0.40]
STOCK_NS=[20,25,30,35]
STOPS=[-0.10,-0.12,-0.15,-0.18]
VOL_PCTS=[70,75,80,None]  # None=不过滤

def zscore(arr):
    vals=[v for v in arr.values() if isinstance(v,(int,float,np.floating)) and np.isfinite(v)]
    if len(vals)<10:return {k:0 for k in arr}
    m=np.nanmedian(vals);s=np.nanstd(vals)
    return {k:((v-m)/s if s>0 and isinstance(v,(int,float,np.floating)) and np.isfinite(v) else 0)for k,v in arr.items()}

best=[]
total=len(SIZE_WTS)*len(STOCK_NS)*len(STOPS)*len(VOL_PCTS)
n=0
for sw in SIZE_WTS:
    for ns in STOCK_NS:
        for st in STOPS:
            for vp in VOL_PCTS:
                n+=1
                if n%20==0:print(f"  {n}/{total}")
                ps_w=(1-sw)*0.35;pb_w=(1-sw)*0.35;str_w=(1-sw)*0.30
                
                vals=[];dates=[]
                for wd in sorted(factor_scores.keys()):
                    fs=factor_scores[wd]
                    if not fs:continue
                    
                    sz=zscore({k:v['size']for k,v in fs.items()})
                    ps=zscore({k:v['ps']for k,v in fs.items()})
                    pb=zscore({k:v['pb']for k,v in fs.items()})
                    st_=zscore({k:v['str']for k,v in fs.items()})
                    
                    # 低波过滤
                    if vp is not None:
                        vols={k:v['vol']for k,v in fs.items()}
                        cutoff=np.percentile(list(vols.values()),vp)
                    
                    sc={}
                    for k in fs:
                        s=sw*sz[k]+ps_w*ps[k]+pb_w*pb[k]+str_w*st_[k]
                        if vp is not None and vols[k]>cutoff:s-=10
                        sc[k]=s
                    
                    # 选Top N
                    top=sorted(sc,key=sc.get,reverse=True)[:ns*2]
                    # 简化的换手率控制:随机选N只
                    selected=top[:ns]
                    
                    ret=sum(fwd_rets.get((wd,s),0)for s in selected)/ns
                    vals.append(ret)
                    dates.append(wd)
                
                if len(vals)<30:continue
                rarr=np.array(vals)
                ann_ret=np.mean(rarr)*52
                ann_vol=np.std(rarr)*np.sqrt(52)
                sharpe=ann_ret/ann_vol if ann_vol>0 else 0
                
                # 简化MaxDD
                cum=(1+rarr).cumprod()
                peak=np.maximum.accumulate(cum)
                dd=(cum/peak-1).min()
                
                best.append({
                    'size_w':sw,'pos':ns,'stop':st,'vol_pct':'none'if vp is None else vp,
                    'ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),
                    'maxdd':round(dd*100,1)
                })

rdf=pd.DataFrame(best).sort_values('sharpe',ascending=False)
print(f"\n{'='*70}")
print("Top 10 参数组合")
print(f"{'='*70}")
print(f"{'size%':>6s} {'pos':>4s} {'stop':>6s} {'vol%':>5s} {'ret%':>8s} {'sharpe':>7s} {'maxdd%':>7s}")
for _,r in rdf.head(10).iterrows():
    print(f"  {r['size_w']:.0%}   {r['pos']:.0f}    {r['stop']:+.2f}   {str(r['vol_pct']):>5s}   {r['ret']:+7.1f}%  {r['sharpe']:+7.3f}  {r['maxdd']:+7.1f}%")

rdf.to_csv('e:/quant/reports/v5_gridsearch_20260528.csv',encoding='utf-8-sig')
