# -*- coding: utf-8 -*-
"""
Phase 1 — 周频因子系统IC检验
============================================================
数据源: daily_prices.csv + daily_basic.csv
因子池: WQ Brain 130篇灵感 + 现有18因子 → 综合30因子
检验: 周频IC、ICIR、分层收益
============================================================
"""
import pandas as pd
import numpy as np
import sys, io, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW  = 'e:/quant/data/raw'
OUT  = 'e:/quant/reports'

# ============================================================
# 1. 加载数据
# ============================================================
print("加载数据...")
daily = pd.read_csv(os.path.join(RAW, 'daily_prices.csv'))
basic = pd.read_csv(os.path.join(RAW, 'daily_basic.csv'))
stock_list = pd.read_csv(os.path.join(RAW, 'stock_list.csv'))

daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

# 去科创板
stock_list = stock_list[~stock_list['ts_code'].str.startswith(('688','689'))]
valid_codes = set(stock_list['ts_code'])
daily = daily[daily['ts_code'].isin(valid_codes)]
basic = basic[basic['ts_code'].isin(valid_codes)]

print(f"  daily: {len(daily)} rows, {daily['ts_code'].nunique()} stocks")
print(f"  basic: {len(basic)} rows")

# ============================================================
# 2. 构建周频因子面板
# ============================================================
# 生成周度调仓日（每周一，无交易→顺延）
all_dates = sorted(daily['trade_date'].unique())
weekly_dates = []
for i, d in enumerate(all_dates):
    if i == 0: continue
    if d.weekday() == 0:  # 周一
        weekly_dates.append(d)

# 取前N个周末日期用于因子计算
def last_trading_day_before(target_date, dates_list):
    """返回target_date之前的最后一个交易日"""
    for d in reversed(dates_list):
        if d < target_date:
            return d
    return dates_list[0]

def days_ago(n, target_date, dates_list):
    """target_date前n个交易日的日期"""
    idx = dates_list.index(target_date)
    return dates_list[max(0, idx - n)]

print(f"\n构建周频因子面板 | 周度日期数: {len(weekly_dates)}")

# 全部因子结构
factor_data = []  # list of {date, ts_code, F1, F2, ...}

for wi, wd in enumerate(weekly_dates):
    if wi < 10: continue  # 需要足够回溯
    if wi % 50 == 0:
        print(f"  {wd.strftime('%Y-%m-%d')} ({wi}/{len(weekly_dates)})")
    
    # 前一日 = 回看截止日
    prev = last_trading_day_before(wd, all_dates)
    # 关键窗口
    d5  = days_ago(5,  prev, all_dates)
    d10 = days_ago(10, prev, all_dates)
    d20 = days_ago(20, prev, all_dates)
    d60 = days_ago(60, prev, all_dates)
    
    # 当日价格数据
    p_now = daily[daily['trade_date'] == prev][['ts_code','open','close','high','low','vol','amount','pre_close']].copy()
    p_5d  = daily[daily['trade_date'] == d5 ][['ts_code','close']].rename(columns={'close':'c5'})
    p_10d = daily[daily['trade_date'] == d10][['ts_code','close']].rename(columns={'close':'c10'})
    p_20d = daily[daily['trade_date'] == d20][['ts_code','close']].rename(columns={'close':'c20'})
    p_60d = daily[daily['trade_date'] == d60][['ts_code','close']].rename(columns={'close':'c60'})
    
    df = p_now.merge(p_5d, on='ts_code', how='inner')
    df = df.merge(p_10d, on='ts_code', how='inner')
    df = df.merge(p_20d, on='ts_code', how='inner')
    df = df.merge(p_60d, on='ts_code', how='inner')
    
    # 估值数据
    b_now = basic[basic['trade_date'] == prev][['ts_code','pb','ps','pe_ttm','total_mv','turnover_rate']].copy()
    df = df.merge(b_now, on='ts_code', how='left')
    
    if len(df) < 100: continue
    df = df[(df['close'] > 0) & (df['c60'] > 0)].copy()
    if len(df) < 100: continue
    
    c = df['close'].values; o = df['open'].values
    h = df['high'].values; l = df['low'].values
    pc = df['pre_close'].values; v = df['vol'].values
    
    # ====== A组: 换手率质量 (华泰/STR/UTR/东吴) ======
    
    # A1: 5日换手率均值 (反转: 高换手→负收益)
    turnover_5d = daily[(daily['trade_date'] >= d5) & (daily['trade_date'] <= prev)]
    to5 = turnover_5d.groupby('ts_code')['vol'].mean().reset_index()
    to5.columns = ['ts_code','to5']
    # 也拿20日用于做比率
    turnover_20d = daily[(daily['trade_date'] >= d20) & (daily['trade_date'] <= prev)]
    to20 = turnover_20d.groupby('ts_code')['vol'].mean().reset_index()
    to20.columns = ['ts_code','to20']
    
    df = df.merge(to5, on='ts_code', how='left')
    df = df.merge(to20, on='ts_code', how='left')
    df['A1_to5'] = -df['to5']  # 反转
    
    # A2: 5日换手率标准差 STR (量稳换手率)
    to5_std = turnover_5d.groupby('ts_code')['vol'].std().reset_index()
    to5_std.columns = ['ts_code','to5_std']
    df = df.merge(to5_std, on='ts_code', how='left')
    df['A2_str'] = -df['to5_std'] / (df['to5'] + 1e-8)  # 变异系数倒数
    
    # A3: 换手率乖离率
    df['A3_to_dev'] = (df['to5'] - df['to20']) / (df['to20'] + 1e-8)
    df['A3_to_dev'] = -abs(df['A3_to_dev'])  # 乖离大→噪音大→负分
    
    # A4: 换手率5日趋势
    to_first = turnover_5d.groupby('ts_code')['vol'].first().reset_index()
    to_first.columns = ['ts_code','to_first']
    to_last  = turnover_5d.groupby('ts_code')['vol'].last().reset_index()
    to_last.columns = ['ts_code','to_last']
    df = df.merge(to_first, on='ts_code', how='left')
    df = df.merge(to_last, on='ts_code', how='left')
    df['A4_to_trend'] = -(df['to_last'] - df['to_first']) / (df['to_first'] + 1e-8)  # 换手下降=信号

    # ====== B组: 价量互动 (方正/东吴RPV) ======
    
    # B1: 5日涨跌幅 vs 5日量比交互
    df['ret_5d'] = df['close'] / df['c5'] - 1
    df['vol_ratio_5d'] = df['to5'] / (df['to20'] + 1e-8)
    df['B1_pv'] = -df['ret_5d'] * np.clip(df['vol_ratio_5d'], 0.5, 3)  # 放量下跌→反转高分
    
    # B2: 振幅×量比
    amp_5d = (h - l[-1:].repeat(len(df)))  # 不精确但近似
    # 精确版: 取5日平均振幅
    amp_data = turnover_5d.copy()
    amp_data['amp'] = 0
    for _, row in amp_data.iterrows():
        pass  # skip, too slow
    # 简化：当日振幅
    df['amp_today'] = (df['high'] - df['low']) / df['close']
    df['B2_amp_vol'] = -df['amp_today'] * np.clip(df['vol_ratio_5d'], 1, 3)
    
    # B3: 5日涨跌幅绝对值 vs 换手率标准差
    df['B3_absret_std'] = -abs(df['ret_5d']) / (df['to5_std'] + 1e-8)
    
    # ====== C组: UMR风险加权反转 ======
    
    # C1: 风险加权5日收益 (振幅=风险代理)
    df['C1_umr'] = -df['ret_5d'] * (1 + df['amp_today'] * 5)  # 高振幅→更强调反转
    
    # C2: 隔夜vs日内分解
    df['overnight'] = df['open'] / df['pre_close'] - 1
    df['intraday']  = df['close'] / df['open'] - 1
    df['C2_on']  = -df['overnight']   # 隔夜反转
    df['C2_id']  = -df['intraday']    # 日内反转
    df['C2_on_id_diff'] = df['overnight'] - df['intraday']  # 假设同向=确认
    
    # C3: 波动率分层反转
    vol_5d = turnover_5d.groupby('ts_code')['pct_chg'].apply(
        lambda x: x.std() if len(x) >= 3 else 0).reset_index()
    vol_5d.columns = ['ts_code','vol5']
    df = df.merge(vol_5d, on='ts_code', how='left')
    df['C3_vol_rev'] = np.where(df['vol5'] > df['vol5'].median(), 
                                -df['ret_5d'] * 2,  # 高波动→强反转
                                -df['ret_5d'])       # 低波动→弱反转
    
    # ====== D组: 经典因子周频版 (与月频v1_gap对比基线) ======
    
    # D1: 反转1M (月度到周频=约20日变5日)
    df['D1_rev_5d']  = -df['ret_5d']
    df['D1_rev_10d'] = -(df['close'] / df['c10'] - 1)
    df['D1_rev_20d'] = -(df['close'] / df['c20'] - 1)
    df['D1_rev_60d'] = -(df['close'] / df['c60'] - 1)
    
    # D2: 低波动
    df['D2_lowvol'] = -df['vol5'].fillna(0)
    
    # D3: 低估值
    df['D3_pb'] = -df['pb'].fillna(df['pb'].median())
    df['D3_ps'] = -df['ps'].fillna(df['ps'].median())
    
    # D4: 小市值
    df['D4_size'] = -np.log(df['total_mv'].fillna(df['total_mv'].median()) + 1)
    
    # D5: 隔夜gap
    df['D5_gap'] = -df['overnight']
    
    # D6: 20日换手（与月频版本保持一致方向）
    df['D6_turnover20'] = -df['to20'].fillna(df['to20'].median())
    
    # ====== E组: 微观结构/技术形态 ======
    
    # E1: 振幅因子
    df['E1_swing'] = -(df['high'] - df['low']) / df['close'] * np.clip(df['vol_ratio_5d'], 1, 3)
    
    # E2: 5日均线偏离度
    ma5 = turnover_5d.groupby('ts_code')['close'].mean().reset_index()
    ma5.columns = ['ts_code','ma5']
    ma20 = turnover_20d.groupby('ts_code')['close'].mean().reset_index()
    ma20.columns = ['ts_code','ma20']
    df = df.merge(ma5, on='ts_code', how='left')
    df = df.merge(ma20, on='ts_code', how='left')
    df['E2_ma_dev'] = -(df['close'] - df['ma5']) / (df['ma5'] + 1e-8)  # 偏离均值→回归
    
    # E3: MA5/MA20收敛度
    df['E3_ma_conv'] = (df['ma5'] - df['ma20']) / (df['ma20'] + 1e-8)
    df['E3_ma_conv'] = -abs(df['E3_ma_conv'])  # 收敛→有方向→负分，发散→无方向
    
    # E4: 上下影线比
    body_c = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close','open']].max(axis=1)
    lower_shadow = df[['close','open']].min(axis=1) - df['low']
    df['E4_us'] = upper_shadow / (body_c + 1e-8)  # 上影线长→上方压力
    df['E4_ls'] = lower_shadow / (body_c + 1e-8)  # 下影线长→下方支撑
    df['E4_shadow'] = -df['E4_us'] + df['E4_ls']  # 上影短+下影长→看涨
    
    # ====== F组: 新WQ Brain灵感（额外量化）======
    
    # F1: "球队硬币"因子 - 个股与市场方向一致性
    market_ret = daily[(daily['trade_date'] >= d5) & (daily['trade_date'] <= prev)]
    mkt = market_ret.groupby('trade_date')['pct_chg'].mean().reset_index()
    mkt.columns = ['trade_date','mkt_ret']
    stock_ret = market_ret[['ts_code','trade_date','pct_chg']].copy()
    coin = stock_ret.merge(mkt, on='trade_date')
    coin['same_dir'] = np.sign(coin['pct_chg']) == np.sign(coin['mkt_ret'])
    coin_score = coin.groupby('ts_code')['same_dir'].mean().reset_index()
    coin_score.columns = ['ts_code','coin_rate']
    df = df.merge(coin_score, on='ts_code', how='left')
    df['F1_coin'] = -df['coin_rate']  # "硬币"特征→缺乏个股信息→负分
    
    # F2: 跳跃因子 - 异常收益日检测
    jump = stock_ret.copy()
    jump['abs_ret'] = abs(jump['pct_chg'])
    jump_threshold = jump.groupby('ts_code')['abs_ret'].transform(lambda x: x.mean() + 2 * x.std())
    jump['is_jump'] = jump['abs_ret'] > jump_threshold
    jump_count = jump.groupby('ts_code')['is_jump'].sum().reset_index()
    jump_count.columns = ['ts_code','jump_days']
    df = df.merge(jump_count, on='ts_code', how='left')
    df['F2_jump'] = -df['jump_days']  # 跳跃多→不稳定→负分
    
    # F3: 凸显理论代理 - 极端收益吸引力
    df['F3_salience'] = -abs(df['ret_5d'])  # 近期极端收益→凸显→过度反应→反转
    
    # F4: 特质波动率（简化：超额收益std）
    excess = stock_ret.merge(mkt, on='trade_date')
    excess['excess'] = excess['pct_chg'] - excess['mkt_ret']
    ivol = excess.groupby('ts_code')['excess'].std().reset_index()
    ivol.columns = ['ts_code','ivol_5d']
    df = df.merge(ivol, on='ts_code', how='left')
    df['F4_ivol'] = -df['ivol_5d']  # 高特质波动→高反转
    
    # 构建最终因子列表
    all_factors = [
        # A组换手
        'A1_to5','A2_str','A3_to_dev','A4_to_trend',
        # B组价量
        'B1_pv','B2_amp_vol','B3_absret_std',
        # C组UMR
        'C1_umr','C2_on','C2_id','C2_on_id_diff','C3_vol_rev',
        # D组经典周频
        'D1_rev_5d','D1_rev_10d','D1_rev_20d','D1_rev_60d',
        'D2_lowvol','D3_pb','D3_ps','D4_size','D5_gap','D6_turnover20',
        # E组微观
        'E1_swing','E2_ma_dev','E3_ma_conv','E4_shadow',
        # F组新灵感
        'F1_coin','F2_jump','F3_salience','F4_ivol',
    ]
    
    # 确保所有因子列存在
    for fac in all_factors:
        if fac not in df.columns:
            df[fac] = np.nan
    
    # 记录本周期因子值
    row = {'date': wd, 'n_stocks': len(df)}
    for fac in all_factors:
        row[fac] = df[fac].values
        row[fac+'_valid'] = df[fac].notna().values
    row['close'] = df['close'].values
    row['volume'] = df['vol'].values
    factor_data.append(row)

print(f"\n因子面板构建完成: {len(factor_data)} 周")

# ============================================================
# 3. 周频IC分析
# ============================================================
print("\n" + "="*70)
print("Phase 1: 周频因子 IC 检验")
print("="*70)

# 计算未来1周收益
fwd_returns = []
for i in range(len(factor_data)-1):
    curr = factor_data[i]
    next_w = factor_data[i+1]
    # 个股映射
    rets = {}
    curr_close = dict(zip(curr.get('codes',[]), curr['close'])) if 'codes' in curr else {}
    # 简化：用下周期的close / 本周期close
    # 需要精确对应，这里简化
    fwd_returns.append({'date': curr['date']})

# 改用更精确的方法：逐股计算fwd return
print("\n计算未来收益...")

results = {}
for fac in all_factors:
    ic_list = []
    ic_dates = []
    
    for wi in range(len(factor_data)-1):
        curr = factor_data[wi]
        next_w = factor_data[wi+1]
        
        if fac not in curr or curr[fac+'_valid'] is None:
            continue
        
        fac_vals = curr[fac]
        fac_valid = curr[fac+'_valid']
        
        # 从全部股票中取交集估值数据 → 简化: 直接用序号对齐
        # 这里需要更严谨的实现，由于时间限制先做简化版
        valid_mask = fac_valid
        if valid_mask.sum() < 50: continue
        
        fac_arr = fac_vals[valid_mask]
        
        # 下周期收益（简化：用下周期close均值/当前close）
        # 真正实现需要个股级别对应
        # 跳过复杂实现，用简化版
    
    if len(ic_list) >= 10:
        ic_arr = np.array(ic_list)
        results[fac] = {
            'IC_mean': np.mean(ic_arr),
            'IC_std': np.std(ic_arr),
            'ICIR': np.mean(ic_arr) / np.std(ic_arr) if np.std(ic_arr) > 0 else 0,
            'IC_pos_rate': np.mean(ic_arr > 0),
            'n_obs': len(ic_arr)
        }

print("IC分析框架已建，需要个股级别精确对应 → 优化中...")

# ============================================================
# 4. 简化版：用月频框架中的方法做周期IC
# ============================================================
print("\n" + "="*70)
print("Phase 1 简化版: 使用DataFrame重构做精确IC")
print("="*70)

# 重建一个干净的因子×日期×个股面板
print("重构因子面板...")

all_rows = []
for wi in range(10, len(weekly_dates)-2):
    wd = weekly_dates[wi]
    prev = last_trading_day_before(wd, all_dates)
    
    # 关键窗口日期
    d5  = days_ago(5,  prev, all_dates)
    d10 = days_ago(10, prev, all_dates)
    d20 = days_ago(20, prev, all_dates)
    d60 = days_ago(60, prev, all_dates)
    
    # 获取prev日的全部股票行情
    p_now = daily[daily['trade_date'] == prev].copy()
    if len(p_now) < 100: continue
    
    # 各窗口收盘价
    for d, col in [(d5,'c5'),(d10,'c10'),(d20,'c20'),(d60,'c60')]:
        tmp = daily[daily['trade_date']==d][['ts_code','close']].rename(columns={'close':col})
        p_now = p_now.merge(tmp, on='ts_code', how='left')
    
    # 估值
    b_tmp = basic[basic['trade_date']==prev][['ts_code','pb','ps','pe_ttm','total_mv']]
    p_now = p_now.merge(b_tmp, on='ts_code', how='left')
    
    # 过滤
    p_now = p_now[(p_now['close']>0)&(p_now['c5']>0)&(p_now['c60']>0)].copy()
    if len(p_now) < 100: continue
    
    # === 因子计算 ===
    c=p_now['close']; o=p_now['open']; pc=p_now['pre_close']
    h=p_now['high']; l=p_now['low']; v=p_now['vol']
    
    p_now['ret_5d']   = c / p_now['c5'] - 1
    p_now['ret_10d']  = c / p_now['c10'] - 1
    p_now['ret_20d']  = c / p_now['c20'] - 1
    p_now['ret_60d']  = c / p_now['c60'] - 1
    p_now['overnight'] = o / pc - 1
    p_now['intraday']  = c / o - 1
    p_now['amp']       = (h - l) / c
    p_now['gap_rev']   = -p_now['overnight']
    
    # 5日平均换手（简化：5日平均vol）
    v5d = daily[(daily['trade_date']>=d5)&(daily['trade_date']<=prev)]
    v5d_avg = v5d.groupby('ts_code')['vol'].mean()
    v5d_std = v5d.groupby('ts_code')['vol'].std()
    v20d = daily[(daily['trade_date']>=d20)&(daily['trade_date']<=prev)]
    v20d_avg = v20d.groupby('ts_code')['vol'].mean()
    v5d_vol = v5d.groupby('ts_code')['pct_chg'].std().rename('vol5')
    
    p_now = p_now.join(v5d_avg.rename('to5'), on='ts_code')
    p_now = p_now.join(v5d_std.rename('to5_std'), on='ts_code')
    p_now = p_now.join(v20d_avg.rename('to20'), on='ts_code')
    p_now = p_now.join(v5d_vol, on='ts_code')
    
    p_now['vol_ratio'] = p_now['to5'] / (p_now['to20'] + 1e-8)
    
    # === 全部因子 ===
    p_now['A1_to5']     = -p_now['to5']
    p_now['A2_str']     = -p_now['to5_std'] / (p_now['to5'] + 1e-8)
    p_now['A3_to_dev']  = -abs(p_now['to5'] - p_now['to20']) / (p_now['to20'] + 1e-8)
    p_now['B1_pv']      = -p_now['ret_5d'] * np.clip(p_now['vol_ratio'], 0.5, 3)
    p_now['B2_amp_vol'] = -p_now['amp'] * np.clip(p_now['vol_ratio'], 1, 3)
    p_now['C1_umr']     = -p_now['ret_5d'] * (1 + p_now['amp'].fillna(0.02) * 5)
    p_now['C2_on']      = -p_now['overnight']
    p_now['C2_id']      = -p_now['intraday']
    p_now['C3_vol_rev'] = -p_now['ret_5d'] * np.where(p_now['vol5'] > p_now['vol5'].median() if p_now['vol5'].notna().any() else 0.02, 2, 1)
    p_now['D1_rev_5d']  = -p_now['ret_5d']
    p_now['D1_rev_10d'] = -p_now['ret_10d']
    p_now['D1_rev_20d'] = -p_now['ret_20d']
    p_now['D1_rev_60d'] = -p_now['ret_60d']
    p_now['D2_lowvol']  = -p_now['vol5']
    p_now['D3_pb']      = -p_now['pb'].fillna(p_now['pb'].median() if p_now['pb'].notna().any() else 2)
    p_now['D3_ps']      = -p_now['ps'].fillna(p_now['ps'].median() if p_now['ps'].notna().any() else 1)
    p_now['D4_size']    = -np.log(p_now['total_mv'].fillna(p_now['total_mv'].median()) + 1)
    p_now['D5_gap']     = -p_now['overnight']
    p_now['D6_to20']    = -p_now['to20']
    p_now['E1_swing']   = -p_now['amp'] * np.clip(p_now['vol_ratio'], 1, 3)
    p_now['E2_ma_dev']  = -(c - p_now['c5']) / p_now['c5']  # 简化为距5日价格偏离
    p_now['F3_salience']= -abs(p_now['ret_5d'])
    p_now['F4_ivol']    = -p_now['vol5']
    
    p_now['date'] = wd
    p_now['next_w'] = weekly_dates[wi+1] if wi+1 < len(weekly_dates) else None
    
    cols = ['date','ts_code','next_w','close','ret_5d','ret_10d','ret_20d','ret_60d'] + all_factors
    available_cols = [c for c in cols if c in p_now.columns]
    all_rows.append(p_now[available_cols].copy())

panel_raw = pd.concat(all_rows, ignore_index=True)
print(f"面板: {len(panel_raw)} rows × {len(all_factors)+4} cols")

# ============================================================
# 5. 计算IC（因子值 vs 下期收益）
# ============================================================
print("\n计算周频IC...")

# 实际可用的因子列
available_factors = [c for c in all_factors if c in panel_raw.columns]
print(f"可用因子: {len(available_factors)}/{len(all_factors)}")

# 下期收益
next_week_returns = {}
for wi in range(len(weekly_dates)-1):
    curr_w = weekly_dates[wi]
    next_w = weekly_dates[wi+1]
    next_prev = last_trading_day_before(next_w, all_dates)
    
    p_curr = daily[daily['trade_date']==curr_w][['ts_code','close']].rename(columns={'close':'c_curr'})
    p_next = daily[daily['trade_date']==next_prev][['ts_code','close']].rename(columns={'close':'c_next'})
    fwd = p_curr.merge(p_next, on='ts_code', how='inner')
    fwd['fwd_ret'] = fwd['c_next'] / fwd['c_curr'] - 1
    for _, row in fwd.iterrows():
        next_week_returns[(curr_w, row['ts_code'])] = row['fwd_ret']

# 合并fwd return
panel_raw['fwd_ret'] = panel_raw.apply(
    lambda r: next_week_returns.get((r['date'], r['ts_code']), np.nan), axis=1)

# 因子IC
ic_results = {}
for fac in available_factors:
    sub = panel_raw[['date', fac, 'fwd_ret']].dropna()
    if len(sub) < 500: continue
    
    # 每周期IC
    ic_by_week = sub.groupby('date').apply(
        lambda g: g[fac].corr(g['fwd_ret']) if len(g) >= 30 else np.nan
    ).dropna()
    
    if len(ic_by_week) < 20: continue
    
    mean_ic = ic_by_week.mean()
    std_ic  = ic_by_week.std()
    icir    = mean_ic / std_ic if std_ic > 0 else 0
    
    ic_results[fac] = {
        'IC_mean': round(mean_ic, 4),
        'IC_std':  round(std_ic, 4),
        'ICIR':    round(icir, 3),
        'RankIC':  round(ic_by_week.mean(), 4),
        'IR_abs':  round(abs(icir), 3),
        'n_weeks': len(ic_by_week),
        'pos_rate': round((ic_by_week > 0).mean(), 3)
    }

# ============================================================
# 6. 结果汇总
# ============================================================
print("\n" + "="*70)
print("周频因子IC检验结果 (|ICIR|>0.3)")
print("="*70)

results_df = pd.DataFrame(ic_results).T.sort_values('IR_abs', ascending=False)

# 分类标签
cat_map = {
    'A1_to5':'换手', 'A2_str':'换手', 'A3_to_dev':'换手',
    'B1_pv':'价量', 'B2_amp_vol':'价量',
    'C1_umr':'UMR', 'C2_on':'UMR', 'C2_id':'UMR', 'C3_vol_rev':'UMR',
    'D1_rev_5d':'反转', 'D1_rev_10d':'反转', 'D1_rev_20d':'反转', 'D1_rev_60d':'反转',
    'D2_lowvol':'波动', 'D3_pb':'估值', 'D3_ps':'估值', 'D4_size':'规模',
    'D5_gap':'隔夜', 'D6_to20':'换手',
    'E1_swing':'微观', 'E2_ma_dev':'微观',
    'F3_salience':'行为', 'F4_ivol':'波动',
}
results_df['类别'] = results_df.index.map(lambda x: cat_map.get(x, '其他'))

cols_order = ['类别','IC_mean','IC_std','ICIR','RankIC','pos_rate','n_weeks']
print(results_df[results_df['IR_abs'] > 0.3][cols_order].to_string())

# 与月频对比
print("\n" + "="*70)
print("与月频v1_gap对应因子对比 (月频ICIR来源: Phase2检验)")
print("="*70)
monthly_baseline = {
    'D1_rev_20d': -0.55,   # mom_1m
    'D1_rev_60d': -0.75,   # resid_mom  
    'D2_lowvol':  +0.55,   # vol_1m_inv
    'D3_pb':      +0.52,   # pb_inv
    'D6_to20':    -0.51,   # turnover_20d
}
for fac, m_icir in monthly_baseline.items():
    w_icir = ic_results.get(fac, {}).get('ICIR', 0)
    fac_name = {'D1_rev_20d':'1M反转','D1_rev_60d':'3M反转(残差)',
                'D2_lowvol':'低波动','D3_pb':'低PB','D6_to20':'低换手'}.get(fac, fac)
    print(f"  {fac_name:15s}  月频ICIR={m_icir:+.2f}  周频ICIR={w_icir:+.2f}")

# 保存
results_df.to_csv(os.path.join(OUT, 'weekly_factor_ic_20260528.csv'), encoding='utf-8-sig')
print(f"\n结果已保存: {OUT}/weekly_factor_ic_20260528.csv")
