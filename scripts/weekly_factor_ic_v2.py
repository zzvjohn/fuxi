# -*- coding: utf-8 -*-
"""
Phase 1 — 周频因子系统IC检验 v2 (完整版)
============================================================
因子来源: WQ Brain 28篇A股文章 + 现有18因子体系
检验方法: 周频Spearman Rank IC + ICIR
目标: 筛选 |ICIR| > 0.4 的因子组策略
============================================================
"""
import pandas as pd, numpy as np, os, sys, io
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW  = 'e:/quant/data/raw'
OUT  = 'e:/quant/reports'

# ============================================================
# 1. 加载与预处理
# ============================================================
print("="*70)
print("Phase 1: 周频因子 IC 检验 v2")
print("="*70)

daily = pd.read_csv(os.path.join(RAW, 'daily_prices.csv'))
basic = pd.read_csv(os.path.join(RAW, 'daily_basic.csv'))
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

# 去科创板
daily = daily[~daily['ts_code'].str.startswith(('688','689'))]
basic = basic[~basic['ts_code'].str.startswith(('688','689'))]

# 周度调仓日
all_dates = sorted(daily['trade_date'].unique())
weekly_dates = [d for i,d in enumerate(all_dates) if i>0 and d.weekday()==0]
print(f"交易日: {len(all_dates)}, 周频日: {len(weekly_dates)}")

def last_day(target, dates):
    for d in reversed(dates):
        if d < target: return d
    return dates[0]

def n_days_ago(n, target, dates):
    idx = dates.index(target) if target in dates else len(dates)-1
    return dates[max(0, idx-n)]

# ============================================================
# 2. 构建周频因子面板
# ============================================================
print("\n构建因子面板...")
all_rows = []
skipped = 0

for wi, wd in enumerate(weekly_dates):
    if wi < 10: continue  # 足够回溯
    if wi % 40 == 0: print(f"  {wd.strftime('%Y-%m')} ({wi}/{len(weekly_dates)})")
    
    prev = last_day(wd, all_dates)
    d5, d10, d20, d60 = [n_days_ago(n, prev, all_dates) for n in [5,10,20,60]]
    
    p_now = daily[daily['trade_date']==prev].copy()
    if len(p_now) < 100: continue
    
    for d, col in [(d5,'c5'),(d10,'c10'),(d20,'c20'),(d60,'c60')]:
        tmp = daily[daily['trade_date']==d][['ts_code','close']].rename(columns={'close':col})
        p_now = p_now.merge(tmp, on='ts_code', how='left')
    
    b_tmp = basic[basic['trade_date']==prev][['ts_code','pb','ps','pe_ttm','total_mv']]
    p_now = p_now.merge(b_tmp, on='ts_code', how='left')
    
    p_now = p_now[(p_now['close']>0)&(p_now['c5'].notna())&(p_now['c60'].notna())].copy()
    if len(p_now) < 100: continue
    
    c,o,h,l,v,pc = p_now['close'],p_now['open'],p_now['high'],p_now['low'],p_now['vol'],p_now['pre_close']
    
    # 基础计算
    p_now['r5' ] = c/p_now['c5' ]-1
    p_now['r10'] = c/p_now['c10']-1
    p_now['r20'] = c/p_now['c20']-1
    p_now['r60'] = c/p_now['c60']-1
    p_now['on' ] = o/pc-1   # 隔夜
    p_now['id' ] = c/o-1    # 日内
    p_now['amp'] = (h-l)/c
    p_now['gap'] = -p_now['on']
    
    # 5日和20日统计
    d5_daily = daily[(daily['trade_date']>=d5)&(daily['trade_date']<=prev)]
    d20_daily = daily[(daily['trade_date']>=d20)&(daily['trade_date']<=prev)]
    
    to5  = d5_daily.groupby('ts_code')['vol'].mean()
    to5s = d5_daily.groupby('ts_code')['vol'].std()
    to20 = d20_daily.groupby('ts_code')['vol'].mean()
    vol5 = d5_daily.groupby('ts_code')['pct_chg'].std()
    vol20= d20_daily.groupby('ts_code')['pct_chg'].std()
    ret5_min = d5_daily.groupby('ts_code')['pct_chg'].min()
    ret5_max = d5_daily.groupby('ts_code')['pct_chg'].max()
    ret5_skew = d5_daily.groupby('ts_code')['pct_chg'].apply(lambda x: x.skew() if len(x)>=3 else 0)
    
    p_now = p_now.join(to5.rename('to5'),     on='ts_code')
    p_now = p_now.join(to5s.rename('to5_std'),on='ts_code')
    p_now = p_now.join(to20.rename('to20'),    on='ts_code')
    p_now = p_now.join(vol5.rename('vol5'),    on='ts_code')
    p_now = p_now.join(vol20.rename('vol20'),  on='ts_code')
    p_now = p_now.join(ret5_min.rename('ret5_min'),on='ts_code')
    p_now = p_now.join(ret5_max.rename('ret5_max'),on='ts_code')
    p_now = p_now.join(ret5_skew.rename('ret5_skew'),on='ts_code')
    
    # 处理上/下影线
    body = abs(c - o)
    p_now['us'] = (h - p_now[['close','open']].max(axis=1)) / (body + 1e-8)
    p_now['ls'] = (p_now[['close','open']].min(axis=1) - l) / (body + 1e-8)
    
    # 5日均线
    ma5  = d5_daily.groupby('ts_code')['close'].mean()
    p_now = p_now.join(ma5.rename('ma5'), on='ts_code')
    
    # ====== A组: 换手率质量 (华泰/STR/UTR/东吴/tps) ======
    p_now['A1_to5']      = -p_now['to5'].fillna(0)                    # 5日均换手(反转)
    p_now['A2_str']      = -p_now['to5_std']/(p_now['to5']+1e-8)     # STR量稳
    p_now['A3_to_dev']   = -(p_now['to5']-p_now['to20'])/(p_now['to20']+1e-8) # 乖离
    p_now['A3_to_dev']   = -abs(p_now['A3_to_dev'])                    # 乖离绝对值
    
    # ====== B组: 增强反转 (UMR/球队硬币/隔夜日内/跳跃) ======
    p_now['B1_rev_5d']   = -p_now['r5']                               # 5日反转
    p_now['B1_rev_10d']  = -p_now['r10']                              # 10日反转
    p_now['B1_rev_20d']  = -p_now['r20']                              # 20日反转
    p_now['B2_umr']      = -p_now['r5'] * (1 + p_now['amp'].fillna(0.02) * 3) # 风险加权
    # 分层: 高波动→强反转
    med_vol = p_now['vol5'].median() if p_now['vol5'].notna().any() else 0.02
    p_now['B3_hi_vol_rev'] = np.where(p_now['vol5'].fillna(med_vol) > med_vol,
                                      -p_now['r5']*2, -p_now['r5'])
    p_now['B4_on_rev']   = -p_now['on']                               # 隔夜反转
    p_now['B5_id_rev']   = -p_now['id']                               # 日内反转
    p_now['B6_jump_rev'] = -np.where(abs(p_now['r5'].fillna(0)) > p_now['r5'].std()*2,
                                     p_now['r5'], 0)                  # 跳跃日反转
    # 球队硬币: 高确定性→反转, 低确定性→动量
    p_now['B7_coin']     = np.where(p_now['vol5'].fillna(med_vol) < med_vol,
                                   -p_now['r5'],  # 低波=高确定性→反转
                                   p_now['r5'])   # 高波=低确定性→动量
    
    # ====== C组: 价量互�� (方正/东吴RPV/异常量/高低位放量) ======
    p_now['vol_ratio']   = (p_now['to5']/(p_now['to20']+1e-8)).clip(0.3, 5)
    p_now['C1_pv']       = -p_now['r5'] * p_now['vol_ratio']          # 放量反转
    p_now['C2_amp_vol']  = -p_now['amp'].fillna(0.02) * p_now['vol_ratio']  # 振幅×量
    p_now['C3_abn_vol']  = -(p_now['vol_ratio'] - 1).abs()            # 异常量(反转)
    # 高位放量=危险, 低位放量=机会
    p_now['C4_pos_vol']  = np.where(p_now['r20'] > 0,
                                   -p_now['r5'] * p_now['vol_ratio'],  # 高位放量→反转
                                   p_now['r5'] * p_now['vol_ratio'])   # 低位放量→动量
    
    # ====== D组: 微观/技术 (振幅/均线/影线/信息分布) ======
    p_now['D1_swing']    = -p_now['amp'].fillna(0.02) * p_now['vol_ratio']  # 振幅×量
    p_now['D2_ma_dev']   = -(p_now['close'] - p_now['ma5'].fillna(p_now['close']))/(p_now['ma5']+1e-8) # 均线回归
    p_now['D3_shadow']   = p_now['ls'] - p_now['us']                  # 下影-上影(看涨)
    p_now['D4_uid']      = -(p_now['vol5'].fillna(med_vol)/(p_now['to5']+1e-8))  # 信息分布度
    
    # ====== E组: 经典周频 (PB/PS/规模/换手20) ======
    p_now['E1_pb']       = -p_now['pb'].fillna(p_now['pb'].median() if p_now['pb'].notna().any() else 2)
    p_now['E2_ps']       = -p_now['ps'].fillna(p_now['ps'].median() if p_now['ps'].notna().any() else 1)
    p_now['E3_size']     = -np.log(p_now['total_mv'].fillna(p_now['total_mv'].median())+1)
    p_now['E4_to20']     = -p_now['to20']
    p_now['E5_pe']       = -p_now['pe_ttm'].fillna(p_now['pe_ttm'].median() if p_now['pe_ttm'].notna().any() else 15)
    p_now['E6_lowvol']   = -p_now['vol20']
    
    # ====== F组: 行为/学术 ======
    p_now['F1_salience'] = -abs(p_now['r5'])                          # 凸显理论
    p_now['F2_skew']     = -p_now['ret5_skew']                        # 偏度反转
    p_now['F3_range']    = p_now['ret5_max'] - p_now['ret5_min']     # 极致收益差
    
    p_now['date'] = wd
    p_now['next_w'] = weekly_dates[wi+1] if wi+1 < len(weekly_dates) else None
    
    FACTORS = [c for c in p_now.columns if c not in 
               ['ts_code','trade_date','open','close','high','low','vol','amount',
                'pre_close','pct_chg','c5','c10','c20','c60','pb','ps','pe_ttm',
                'total_mv','r5','r10','r20','r60','on','id','amp','gap',
                'to5','to5_std','to20','vol5','vol20','ret5_min','ret5_max',
                'ret5_skew','ma5','us','ls','vol_ratio','date','next_w',
                'body']]
    
    keep_cols = ['date','ts_code','next_w'] + FACTORS
    all_rows.append(p_now[keep_cols].copy())

panel = pd.concat(all_rows, ignore_index=True)
print(f"面板: {len(panel)} rows, 因子数: {len(FACTORS)}")

# ============================================================
# 3. 未来收益
# ============================================================
print("\n计算未来1周收益...")
fwd_rets = {}
for wi in range(len(weekly_dates)-1):
    cw = weekly_dates[wi]; nw = weekly_dates[wi+1]
    np_prev = last_day(nw, all_dates)
    pc = daily[daily['trade_date']==cw][['ts_code','close']].rename(columns={'close':'c_c'})
    pn = daily[daily['trade_date']==np_prev][['ts_code','close']].rename(columns={'close':'c_n'})
    f = pc.merge(pn, on='ts_code', how='inner')
    f['fwd'] = f['c_n']/f['c_c']-1
    for _,r in f.iterrows():
        fwd_rets[(cw, r['ts_code'])] = r['fwd']

panel['fwd_ret'] = panel.apply(lambda r: fwd_rets.get((r['date'],r['ts_code']),np.nan), axis=1)

# ============================================================
# 4. IC分析
# ============================================================
print("\n计算周频IC...")
ic_results = {}
for fac in FACTORS:
    sub = panel[['date',fac,'fwd_ret']].dropna()
    if len(sub) < 500: continue
    
    ic_weekly = sub.groupby('date').apply(
        lambda g: g[fac].corr(g['fwd_ret'], method='spearman') if len(g)>=30 else np.nan
    ).dropna()
    
    if len(ic_weekly) < 20: continue
    
    mic = ic_weekly.mean(); sic = ic_weekly.std()
    icir = mic/sic if sic>0 else 0
    
    ic_results[fac] = {
        'IC_mean': round(mic,4), 'IC_std': round(sic,4),
        'ICIR': round(icir,3), 'IR_abs': round(abs(icir),3),
        'n': len(ic_weekly), 'pos': round((ic_weekly>0).mean(),3)
    }

# ============================================================
# 5. 结果
# ============================================================
results_df = pd.DataFrame(ic_results).T.sort_values('IR_abs', ascending=False)

# 因子名映射
name_map = {
    'A1_to5':'A1_换手5日','A2_str':'A2_STR量稳','A3_to_dev':'A3_换手乖离',
    'B1_rev_5d':'B1_反转5日','B1_rev_10d':'B1_反转10日','B1_rev_20d':'B1_反转20日',
    'B2_umr':'B2_UMR风险加权','B3_hi_vol_rev':'B3_高波反转',
    'B4_on_rev':'B4_隔夜反转','B5_id_rev':'B5_日内反转',
    'B6_jump_rev':'B6_跳跃反转','B7_coin':'B7_球队硬币',
    'C1_pv':'C1_价量反转','C2_amp_vol':'C2_振幅量比','C3_abn_vol':'C3_异常量',
    'C4_pos_vol':'C4_高低位放量',
    'D1_swing':'D1_振幅','D2_ma_dev':'D2_均线偏离','D3_shadow':'D3_影线',
    'D4_uid':'D4_信息分布度',
    'E1_pb':'E1_低PB','E2_ps':'E2_低PS','E3_size':'E3_小市值',
    'E4_to20':'E4_低换手20','E5_pe':'E5_低PE','E6_lowvol':'E6_低波动20',
    'F1_salience':'F1_凸显理论','F2_skew':'F2_偏度','F3_range':'F3_极致收益差',
}
results_df.index = results_df.index.map(lambda x: name_map.get(x,x))

print("\n" + "="*70)
print("周频因子IC检验结果 (|ICIR| > 0.3)")
print("="*70)
cols = ['IC_mean','IC_std','ICIR','pos','n']
strong = results_df[results_df['IR_abs'] > 0.3]
print(strong[cols].to_string())

# 与月频对比
monthly_baseline = {
    'B1_反转20日': ('月1M反转',-0.55),
    'B1_反转20日': ('月3M残差动量',-0.75),
    'E6_低波动20': ('月低波动',0.55),
    'E1_低PB':     ('月低PB',0.52),
    'E4_低换手20': ('月低换手',-0.51),
}
print("\n--- 月频 vs 周频 ICIR 对比 ---")
for fn,(mn,micir) in [('B1_反转20日','1M反转',-0.55),('E6_低波动20','低波动',0.55),
                        ('E1_低PB','低PB',0.52),('E4_低换手20','低换手',-0.51)]:
    wicir = ic_results.get(fn,{}).get('ICIR',0)
    print(f"  {mn:10s} 月频{micir:+.2f}  周频{wicir:+.2f}")

# Top因子（策略构建用）
top_factors = results_df[results_df['IR_abs'] > 0.4]
print(f"\n--- Top因子 (|ICIR|>0.4): {len(top_factors)}个 ---")
print(top_factors[['ICIR','pos']].to_string())

results_df.to_csv(os.path.join(OUT, 'weekly_factor_ic_v2_20260528.csv'), encoding='utf-8-sig')
print(f"\n结果: {OUT}/weekly_factor_ic_v2_20260528.csv")
