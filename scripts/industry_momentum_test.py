# -*- coding: utf-8 -*-
"""
v9 — 行业动量 vs 行业反转 对比测试
============================================================
盲区补充: 不测个股反转, 只测行业级别信号
测试项:
  A: 行业20日动量 (越最近的赢家行业越好)
  B: 行业5日反转 (最近跌的行业买, 涨的卖)
  C: UMR式行业动量反转混合 (高波动行业反转, 低波动行业动量)
============================================================
"""
import pandas as pd, numpy as np, os, sys, io
import warnings; warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW = 'e:/quant/data/raw'
daily = pd.read_csv(os.path.join(RAW, 'daily_prices.csv'))
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
daily = daily[~daily['ts_code'].str.startswith(('688','689'))]

all_dates = sorted(daily['trade_date'].unique())
weekly_dates = [d for i,d in enumerate(all_dates) if i>0 and d.weekday()==0]

def last_day(t, dates):
    for d in reversed(dates): 
        if d < t: return d
    return dates[0]
def ndays(n, t, dates):
    idx = dates.index(t) if t in dates else len(dates)-1
    return dates[max(0, idx-n)]

print("读取SW行业分类...")
sl = pd.read_csv(os.path.join(RAW, 'stock_list.csv'))
sw_map = dict(zip(sl['ts_code'], sl['sw_ind_code']))

# ============================================================
# 行业级别信号测试
# ============================================================
print("\n行业动量/反转 信号检验")
print("="*60)

results = {'A_mom': [], 'B_rev': [], 'C_umr': []}

for wi, wd in enumerate(weekly_dates):
    if wi < 20: continue
    prev = last_day(wd, all_dates)
    d5 = ndays(5, prev, all_dates)
    d20 = ndays(20, prev, all_dates)
    
    # 当前行业池
    curr_s = daily[daily['trade_date'] == prev][['ts_code', 'close']].copy()
    curr_s['sw'] = curr_s['ts_code'].map(sw_map)
    curr_s = curr_s[curr_s['sw'].notna()]
    
    # 行业20日表现
    for d, col in [(d5,'c5'),(d20,'c20')]:
        tmp = daily[daily['trade_date']==d][['ts_code','close']].rename(columns={'close':col})
        curr_s = curr_s.merge(tmp, on='ts_code', how='left')
    curr_s = curr_s.dropna()
    
    # 行业级别收益
    curr_s['ret_5d'] = curr_s['close'] / curr_s['c5'] - 1
    curr_s['ret_20d'] = curr_s['close'] / curr_s['c20'] - 1
    ind_ret = curr_s.groupby('sw').agg(ret5=('ret_5d','mean'), ret20=('ret_20d','mean')).reset_index()
    
    # A: 行业20日动量 (追强势行业)
    ind_ret['A_score'] = ind_ret['ret20']
    
    # B: 行业5日反转 (买近期跌的行业)
    ind_ret['B_score'] = -ind_ret['ret5']
    
    # C: UMR混合 (高波动行业反转, 低波动行业动量) — 简化为用ret20std作波动代理
    ind_std = curr_s.groupby('sw')['ret_20d'].std().reset_index()
    ind_std.columns = ['sw','vol20']
    ind_ret = ind_ret.merge(ind_std, on='sw', how='left')
    med_vol = ind_ret['vol20'].median()
    ind_ret['C_score'] = np.where(ind_ret['vol20'] > med_vol, 
                                   -ind_ret['ret5'],   # 高波动: 反转
                                   ind_ret['ret20'])    # 低波动: 动量
    
    # 下周期行业表现
    nw_prev = last_day(weekly_dates[wi+1], all_dates) if wi+1 < len(weekly_dates) else prev
    nw_s = daily[daily['trade_date'] == nw_prev][['ts_code', 'close']].copy()
    nw_s['sw'] = nw_s['ts_code'].map(sw_map)
    nw_ret = nw_s.groupby('sw')['close'].mean().reset_index()
    nw_ret.columns = ['sw','next_close']
    curr_close = curr_s.groupby('sw')['close'].mean().reset_index()
    curr_close.columns = ['sw','curr_close']
    fwd = curr_close.merge(nw_ret, on='sw', how='inner')
    fwd['fwd_ret'] = fwd['next_close'] / fwd['curr_close'] - 1
    
    merged = ind_ret.merge(fwd[['sw','fwd_ret']], on='sw', how='inner')
    if len(merged) < 15: continue
    
    for label, col in [('A_mom','A_score'), ('B_rev','B_score'), ('C_umr','C_score')]:
        ic = merged[col].corr(merged['fwd_ret'], method='spearman')
        results[label].append(ic)

print(f"\n{'信号':<10s} {'IC均值':>8s} {'ICIR':>8s} {'胜率':>6s} {'样本':>5s}")
for label in ['A_mom','B_rev','C_umr']:
    arr = np.array([x for x in results[label] if not np.isnan(x)])
    if len(arr) < 20: continue
    mic = np.mean(arr); sic = np.std(arr)
    icir = mic/sic if sic>0 else 0
    print(f"{label:<10s} {mic:+.4f}  {icir:+.3f}  {np.mean(arr>0):.1%}  {len(arr)}")
