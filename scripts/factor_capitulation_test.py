#!/usr/bin/env python3
"""
#16 帖子恐慌反转因子 — 本地 IC 检验
=====================================
提取三信号：振幅(high-low)/close + 放量min(vol/adv20,5) + 反转-ret_5d
跑 Spearman Rank IC vs 5/10/20日前向收益
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'reports')

# ========== 加载数据 ==========
print("[1/4] 加载日线...")
df = pd.read_csv(os.path.join(DATA_DIR, 'daily_prices.csv'))
df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
print(f"  日线: {len(df):,} 行, {df['ts_code'].nunique()} 只股票")
print(f"  日期范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

# 过滤近3年加速
df = df[df['trade_date'] >= '2022-01-01'].copy()
print(f"  过滤后: {len(df):,} 行")

# ========== 计算因子 ==========
print("\n[2/4] 计算帖子三信号因子...")

# 先算日收益率
df = df.sort_values(['ts_code', 'trade_date'])
df['ret_1d'] = df.groupby('ts_code')['close'].pct_change()

# 1. 日内振幅: (high - low) / close
df['amplitude'] = (df['high'] - df['low']) / df['close']

# 2. 20日均量
df['adv20'] = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20, min_periods=10).mean())
df['vol_ratio'] = df['vol'] / df['adv20']
# min(vol_ratio, 5) 缩尾
df['vol_ratio_capped'] = np.minimum(df['vol_ratio'], 5.0)

# 3. 5日收益反转
df['ret_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(5))
df['reversal_5d'] = -df['ret_5d']

# 4. 复合因子: rank(振幅) + rank(放量) - rank(收益) = rank(振幅) + rank(放量) + rank(反转)
#    注意: 帖子原文是 -rank(returns)，我们的 reversal_5d = -ret_5d，所以是 +rank(reversal_5d)
# 简化：暂不做时序衰减（ts_decay_linear），先看原始信号

# 前向收益（用于IC计算）
df['fwd_5d']  = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-5)  / x - 1)
df['fwd_10d'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-10) / x - 1)
df['fwd_20d'] = df.groupby('ts_code')['close'].transform(lambda x: x.shift(-20) / x - 1)

# 清理：至少需要 fwd_20d 有数据
df = df.dropna(subset=['amplitude', 'vol_ratio_capped', 'reversal_5d']).copy()
print(f"  有效观测: {len(df):,} 行, {df['ts_code'].nunique()} 只")

# ========== 加载行业分类 ==========
print("\n  加载行业分类...")
try:
    ind_df = pd.read_csv(os.path.join(DATA_DIR, '..', 'processed', 'stock_industry_sw.csv'))
    if 'industry_code' not in ind_df.columns:
        # Try stock_list.csv with industry info
        stock_list = pd.read_csv(os.path.join(DATA_DIR, 'stock_list.csv'))
        if 'industry' in stock_list.columns:
            ind_map = dict(zip(stock_list['ts_code'], stock_list['industry']))
        else:
            ind_map = {}
    else:
        ind_map = dict(zip(ind_df['ts_code'], ind_df['industry_code']))
    df['industry'] = df['ts_code'].map(ind_map)
    has_industry = df['industry'].notna().sum()
    print(f"  行业覆盖: {has_industry:,} / {len(df):,} ({has_industry/len(df)*100:.1f}%)")
except Exception as e:
    print(f"  行业加载失败: {e}, 跳过行业中性化")
    df['industry'] = 'unknown'
    has_industry = False

# ========== IC 计算 ==========
print("\n[3/4] 计算 Spearman Rank IC...")

def calc_ic(data, factor_col, fwd_col):
    """逐日 Spearman Rank IC"""
    ics = []
    for date, grp in data.groupby('trade_date'):
        valid = grp[[factor_col, fwd_col]].dropna()
        if len(valid) < 50:
            continue
        ic = valid[factor_col].rank().corr(valid[fwd_col].rank(), method='spearman')
        ics.append({'trade_date': date, 'ic': ic})
    return pd.DataFrame(ics)

# 因子列表
factors = {
    'amplitude':        '日内振幅 (high-low)/close',
    'vol_ratio_capped': '放量 min(vol/adv20,5)',
    'reversal_5d':      '5日反转 -ret_5d',
}

# 复合因子：逐日 rank 再等权加
print("  计算逐日 rank 复合因子...")
df['rank_amplitude']    = df.groupby('trade_date')['amplitude'].rank(pct=True)
df['rank_vol']          = df.groupby('trade_date')['vol_ratio_capped'].rank(pct=True)
df['rank_reversal']     = df.groupby('trade_date')['reversal_5d'].rank(pct=True)
df['composite_raw']     = df['rank_amplitude'] + df['rank_vol'] + df['rank_reversal']

# 行业中性化版本（如果有行业数据）
if has_industry:
    df['composite_ind_neut'] = df.groupby(['trade_date', 'industry'])['composite_raw'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    factors['composite_ind_neut'] = '复合因子(行业中性化)'

factors['composite_raw'] = '复合因子(原始)'

# 跑 IC
for horizon in [5, 10, 20]:
    print(f"\n--- 前向 {horizon}日 收益 ---")
    fwd_col = f'fwd_{horizon}d'
    results = []
    for fcol, fname in factors.items():
        ic_df = calc_ic(df, fcol, fwd_col)
        if len(ic_df) == 0:
            results.append([fname, 0, 0, 0, 0])
            continue
        ic_mean = ic_df['ic'].mean()
        ic_std  = ic_df['ic'].std()
        icir    = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos  = (ic_df['ic'] > 0).mean()
        results.append([fname, ic_mean, ic_std, icir, ic_pos])
    
    print(f"  {'因子':<25s} {'IC均值':>8s} {'IC_std':>8s} {'ICIR':>8s} {'IC>0%':>8s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        name, im, is_, ir, ip = r
        flag = "⭐" if abs(ir) > 0.3 else ("✓" if abs(ir) > 0.15 else " ")
        print(f"  {name:<25s} {im:>8.4f} {is_:>8.4f} {ir:>8.4f} {ip:>7.1%}  {flag}")

# ========== 分组收益 ==========
print("\n[4/4] 分组回测...")

def calc_group_returns(data, factor_col, fwd_col, n_groups=5):
    """逐日分5组等权，看Q5-Q1月差"""
    g_returns = []
    for date, grp in data.groupby('trade_date'):
        valid = grp[[factor_col, fwd_col]].dropna()
        if len(valid) < 5 * n_groups:
            continue
        valid['group'] = pd.qcut(valid[factor_col], n_groups, labels=False, duplicates='drop')
        if valid['group'].nunique() < n_groups:
            continue
        g_ret = valid.groupby('group')[fwd_col].mean()
        g_ret.name = date
        g_returns.append(g_ret)
    
    gr = pd.DataFrame(g_returns)
    monthly = (gr + 1).resample('M').prod() - 1 if isinstance(gr.index, pd.DatetimeIndex) else gr
    stats = []
    for q in range(n_groups):
        q_mean = gr[q].mean() * 100  # 转为%
        stats.append(q_mean)
    return stats

# 用 fwd_5d（前向5日收益）看分组效果
for fcol, fname in [('amplitude', '振幅'), ('composite_raw', '复合(原始)')]:
    if has_industry:
        continue  # 只跑两个关键因子节省时间
    print(f"\n  {fname}因子分组 (前向5日收益):")
    # 快速采样: 每月取第一个交易日
    monthly_sample = df[df['trade_date'].dt.day <= 5].copy()
    gstats = calc_group_returns(monthly_sample, fcol, 'fwd_5d')
    for q, gs in enumerate(gstats):
        print(f"    Q{q+1}: {gs:+.4f}%")

# ========== 时序衰减版复合因子 ==========
print("\n\n=== 时序衰减版复合因子 ===")
# ts_decay_linear(..., 3): 最近3天加权平均, 权重 = [3, 2, 1] / 6
# 先算每日 raw score，再做3日加权滑动
df2 = df.sort_values(['ts_code', 'trade_date']).copy()
df2['daily_raw'] = df2['rank_amplitude'] + df2['rank_vol'] + df2['rank_reversal']
df2['decay_weighted'] = df2.groupby('ts_code')['daily_raw'].transform(
    lambda x: x.rolling(3, min_periods=1).apply(
        lambda w: np.dot(w[::-1], np.arange(1, len(w)+1)) / sum(range(1, len(w)+1))
        if len(w) > 0 else w[-1]
    )
)

# IC
for horizon in [5, 10, 20]:
    fwd_col = f'fwd_{horizon}d'
    ic_df = calc_ic(df2, 'decay_weighted', fwd_col)
    if len(ic_df) > 0:
        im = ic_df['ic'].mean(); is_ = ic_df['ic'].std(); ir = im/is_
        ip = (ic_df['ic'] > 0).mean()
        flag = "⭐" if abs(ir) > 0.3 else ("✓" if abs(ir) > 0.15 else " ")
        print(f"  前向{horizon:2d}日: IC={im:+.4f}  ICIR={ir:+.4f}  IC>0={ip:.1%}  {flag}")

# ========== v5b 风格收益（振幅+放量 vs 小市值+价值） ==========
print("\n\n=== 与 v5b 因子对比 ===")
# v5b 因子: mcap, ps, pb
try:
    fund = pd.read_csv(os.path.join(DATA_DIR, '..', 'processed', 'fundamentals_quarterly.csv'))
    last_q = fund['end_date'].max()
    fund_latest = fund[fund['end_date'] == last_q].set_index('ts_code')
    df['mcap'] = df['ts_code'].map(fund_latest['market_cap']).fillna(0)
    df['pb']   = df['ts_code'].map(fund_latest['pb_ratio']).fillna(0)
    df['ps']   = df['ts_code'].map(fund_latest['ps_ratio']).fillna(0)
    valid_mask = (df['mcap'] > 0) & (df['pb'] > 0) & (df['ps'] > 0)
    df_v = df[valid_mask].copy()
    df_v['v5b_score'] = (0.35 * (-np.log(df_v['mcap'])).rank(pct=True) +
                         0.25 * (-np.log(df_v['ps'])).rank(pct=True) +
                         0.22 * (-np.log(df_v['pb'])).rank(pct=True))
    df_v['alpha_score'] = df_v['composite_raw'].rank(pct=True)
    # 相关性
    corr = df_v[['v5b_score', 'alpha_score', 'amplitude', 'vol_ratio_capped', 'reversal_5d']].corr()
    print("  v5b得分 vs Alpha得分 相关性:")
    print(corr[['v5b_score', 'alpha_score']].round(3).to_string())
    # IC
    for name, col in [('v5b_score', 'v5b得分'), ('alpha_score', 'Alpha得分')]:
        ic_df = calc_ic(df_v, name, 'fwd_10d')
        if len(ic_df) > 0:
            im = ic_df['ic'].mean(); is_ = ic_df['ic'].std(); ir = im/is_
            ip = (ic_df['ic'] > 0).mean()
            print(f"  {col:12s} → fwd_10d: IC={im:+.4f}  ICIR={ir:+.4f}  IC>0={ip:.1%}")
except Exception as e:
    print(f"  跳过v5b对比: {e}")

print("\n=== 完成 ===")
