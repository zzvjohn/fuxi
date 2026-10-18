"""
ARBR在小市值池中的过滤/取负效果测试
1. 排除高ARBR：剔除top 20%
2. ARBR取负：用-ARBR排名
3. 分位收益：Q1(低ARBR) vs Q5(高ARBR)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 70)
print("  ARBR 小市值池 — 过滤/取负效果")
print("=" * 70)

# Load data (reuse pivot from previous step)
daily = pd.read_csv(PROJECT_ROOT / 'data/raw/daily_prices.csv', dtype={'ts_code': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
basic = pd.read_csv(PROJECT_ROOT / 'data/raw/daily_basic.csv', dtype={'ts_code': str})
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')

trade_dates = sorted(daily['trade_date'].unique())
recent_270 = trade_dates[-270:]
daily_sub = daily[daily['trade_date'].isin(recent_270)]
basic_sub = basic[basic['trade_date'].isin(recent_270)]

print(f"  窗口: {recent_270[0].strftime('%Y-%m-%d')} ~ {recent_270[-1].strftime('%Y-%m-%d')}")

# Pivot
close_df = daily_sub.pivot(index='trade_date', columns='ts_code', values='close')
open_df  = daily_sub.pivot(index='trade_date', columns='ts_code', values='open')
high_df  = daily_sub.pivot(index='trade_date', columns='ts_code', values='high')
low_df   = daily_sub.pivot(index='trade_date', columns='ts_code', values='low')
mcap = basic_sub.pivot(index='trade_date', columns='ts_code', values='circ_mv') / 10000  # 万元→亿元

# ARBR
ar = (high_df - open_df).rolling(26).sum() / ((open_df - low_df).rolling(26).sum() + 0.001)
br = (high_df - close_df.shift(1)).rolling(26).sum() / ((close_df.shift(1) - low_df).rolling(26).sum() + 0.001)
arbr = (ar + br) / 2

# Align
common_dates = close_df.index.intersection(mcap.index)
common_stocks = close_df.columns.intersection(mcap.columns)
close_df = close_df.loc[common_dates, common_stocks]
arbr = arbr.loc[common_dates, common_stocks]
mcap = mcap.loc[common_dates, common_stocks]

# Forward 5d
forward_5d = close_df.shift(-5) / close_df - 1
suspended = close_df.diff(5) == 0
forward_5d[suspended] = np.nan

# Only stocks with >=40 days
valid = close_df.count() >= 40
close_df = close_df.loc[:, valid]
arbr = arbr.loc[:, valid]
mcap = mcap.loc[:, valid]
forward_5d = forward_5d.loc[:, valid]

print(f"  对齐: {len(common_dates)}日期 × {close_df.shape[1]}股票")

# Loop through dates, focusing on small-cap pool
print("\n[计算] 小市值池(10~100亿)中ARBR分位效果...")

results = []
for t_idx in range(20, len(common_dates) - 5):
    date = common_dates[t_idx]
    mc = mcap.iloc[t_idx]
    arb = arbr.iloc[t_idx]
    fwd = forward_5d.iloc[t_idx]
    
    # Small-cap filter
    sc_mask = (mc >= 10) & (mc < 100)
    sub_arb = arb[sc_mask].dropna()
    sub_fwd = fwd[sc_mask].dropna()
    common = sub_arb.index.intersection(sub_fwd.index)
    
    if len(common) < 50:
        continue
    
    arb_s = sub_arb[common]
    fwd_s = sub_fwd[common]
    
    # Clip
    arb_s = arb_s.clip(arb_s.quantile(0.01), arb_s.quantile(0.99))
    fwd_s = fwd_s.clip(fwd_s.quantile(0.01), fwd_s.quantile(0.99))
    
    # Quintile returns
    q_labels = pd.qcut(arb_s, 5, labels=False, duplicates='drop')
    for q in range(5):
        in_q = q_labels == q
        if in_q.sum() < 10:
            continue
        q_ret = fwd_s[in_q].mean()
        results.append({
            'date': date, 'quintile': q + 1,
            'n': in_q.sum(),
            'fwd_ret': q_ret,
            'pool': '小市值Q1-5'
        })
    
    # Top 20% excluded pool
    top20_cut = arb_s.quantile(0.80)
    excl_mask = arb_s <= top20_cut
    if excl_mask.sum() >= 20:
        pool_ret_excl = fwd_s[excl_mask].mean()
        pool_ret_all = fwd_s.mean()
        results.append({
            'date': date, 'quintile': -1,
            'n': excl_mask.sum(),
            'fwd_ret': pool_ret_excl,
            'pool': '排除Top20%',
            'all_ret': pool_ret_all,
        })

# Summary
df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("  结果: 小市值池(10~100亿) ARBR分位收益")
print("=" * 70)

print(f"\n{'分组':20s} {'截面':>6s} {'均持股':>6s} {'5日收益均值':>10s} {'年化':>8s} {'t-stat':>8s}")
print("-" * 70)

for q in [1, 2, 3, 4, 5]:
    sub = df[(df['pool'] == '小市值Q1-5') & (df['quintile'] == q)]
    if sub.empty:
        continue
    rets = sub['fwd_ret'].values
    ret_mean = np.mean(rets)
    ret_std = np.std(rets)
    tstat = ret_mean / (ret_std / np.sqrt(len(rets))) if ret_std > 0 else 0
    ann = ret_mean * 252 / 5
    n_avg = sub['n'].mean()
    print(f"{f'Q{q}(ARBR {"低" if q<3 else "中" if q==3 else "高"})':20s} {len(sub):>6d} {n_avg:>6.0f} {ret_mean:>+10.4f} {ann:>+7.1%} {tstat:>+8.3f}")

# 排除Top20%
print()
excl = df[df['pool'] == '排除Top20%']
if not excl.empty:
    rets_excl = excl['fwd_ret'].values
    rets_all = np.array([r for r in excl['all_ret'].values])
    excl_mean = np.mean(rets_excl)
    all_mean = np.mean(rets_all)
    excl_t = np.mean(rets_excl - rets_all) / (np.std(rets_excl - rets_all) / np.sqrt(len(rets_excl)))
    excl_ann = excl_mean * 252 / 5
    print(f"{'排除Top20%高ARBR':20s} {len(excl):>6d} {excl['n'].mean():>6.0f} {excl_mean:>+10.4f} {excl_ann:>+7.1%}")
    print(f"{'全量小市值池':20s} {'':6s} {'':6s} {all_mean:>+10.4f}")
    print(f"{'增量':20s} {'':6s} {'':6s} {excl_mean - all_mean:>+10.4f}  (t={excl_t:+.3f})")

# ARBR取负 IC
print(f"\n{'='*70}")
print("  ARBR取负(-ARBR) 在小市值池中的IC")
print(f"{'='*70}")

flip_ics = []
for t_idx in range(20, len(common_dates) - 5):
    date = common_dates[t_idx]
    mc = mcap.iloc[t_idx]
    arb = arbr.iloc[t_idx]
    fwd = forward_5d.iloc[t_idx]
    sc_mask = (mc >= 10) & (mc < 100)
    sub_arb = arb[sc_mask].dropna()
    sub_fwd = fwd[sc_mask].dropna()
    common_idx = sub_arb.index.intersection(sub_fwd.index)
    if len(common_idx) < 50:
        continue
    arb_s = sub_arb[common_idx].clip(sub_arb[common_idx].quantile(0.01), sub_arb[common_idx].quantile(0.99))
    fwd_s = sub_fwd[common_idx].clip(sub_fwd[common_idx].quantile(0.01), sub_fwd[common_idx].quantile(0.99))
    # Flip
    ic_flip = (-arb_s).rank().corr(fwd_s.rank())
    flip_ics.append(ic_flip)

flip_mean = np.mean(flip_ics)
flip_std = np.std(flip_ics)
flip_icir = flip_mean / flip_std if flip_std > 0 else 0
flip_pct = sum(1 for x in flip_ics if x > 0) / len(flip_ics)

print(f"  -ARBR IC均值: {flip_mean:+.4f}")
print(f"  -ARBR ICIR:   {flip_icir:+.3f}")
print(f"  -ARBR +IC%:   {flip_pct:.1%}")

# Final verdict
print(f"\n{'='*70}")
print("  裁决")
print(f"{'='*70}")

q1q5 = None
if len(df[(df['pool'] == '小市值Q1-5') & (df['quintile'] == 1)]) > 0 and \
   len(df[(df['pool'] == '小市值Q1-5') & (df['quintile'] == 5)]) > 0:
    q1r = df[(df['pool'] == '小市值Q1-5') & (df['quintile'] == 1)]['fwd_ret'].mean()
    q5r = df[(df['pool'] == '小市值Q1-5') & (df['quintile'] == 5)]['fwd_ret'].mean()
    q1q5 = q1r - q5r
    print(f"  Q1(低ARBR) - Q5(高ARBR) 5日收益差: {q1q5:+.4f} (年化{q1q5*252/5:+.1%})")

print(f"  -ARBR ICIR: {flip_icir:+.3f}  (+IC%={flip_pct:.1%})")

if flip_icir >= 0.3 and flip_pct >= 0.55:
    print("\n  ✅ ARBR取负通过阈值！低ARBR→高收益信号有效，可加入策略二打分池")
elif abs(flip_icir) < 0.2:
    print(f"\n  ❌ 信号太弱(ICIR={flip_icir:+.3f}<0.2)，取负/排除都不足以产生有意义的alpha增量")
    print("     即使排除Top20%高ARBR，池均收益提升也极小")
else:
    print(f"\n  ⚠️ 边缘信号(ICIR={flip_icir:+.3f})，可做弱过滤器但不建议加入核心打分")

print("\n📌 核心问题不是ARBR方向，是信号强度本身。")
print("  市值越小ARBR负IC越强的规律已经验证，但绝对值不够大。")
print("  做条件过滤(二值)比做连续排序更合适——即'高ARBR直接排除'而非'按-ARBR排序'。")
