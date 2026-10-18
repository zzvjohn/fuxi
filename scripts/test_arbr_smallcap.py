"""
ARBR_26 在小市值子集中的截面IC重评估
目标：判断ARBR能否作为策略二的补充信号（小市值+ROE>5%池内重新验证）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DAILY_CSV = PROJECT_ROOT / 'data' / 'raw' / 'daily_prices.csv'
BASIC_CSV = PROJECT_ROOT / 'data' / 'raw' / 'daily_basic.csv'

print("=" * 70)
print("  ARBR_26 在小市值子集中的IC重评估")
print("=" * 70)

# Step 1: 加载数据
print("\n[Step 1] 加载日线和市值数据...")
daily = pd.read_csv(DAILY_CSV, dtype={'ts_code': str})
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed')
print(f"  日线: {len(daily)} 行, {daily['trade_date'].nunique()} 个交易日")

basic = pd.read_csv(BASIC_CSV, dtype={'ts_code': str})
basic['trade_date'] = pd.to_datetime(basic['trade_date'], format='mixed')
print(f"  日频基础: {len(basic)} 行")

# Step 2: 取最近270个交易日
trade_dates = sorted(daily['trade_date'].unique())
recent_270 = trade_dates[-270:]
print(f"  270天窗口: {trade_dates[-270].strftime('%Y-%m-%d')} ~ {trade_dates[-1].strftime('%Y-%m-%d')}")

daily_sub = daily[daily['trade_date'].isin(recent_270)]
basic_sub = basic[basic['trade_date'].isin(recent_270)]

# Step 3: Pivot OHLC
print("\n[Step 2] Pivot OHLC...")
close_df = daily_sub.pivot(index='trade_date', columns='ts_code', values='close')
open_df  = daily_sub.pivot(index='trade_date', columns='ts_code', values='open')
high_df  = daily_sub.pivot(index='trade_date', columns='ts_code', values='high')
low_df   = daily_sub.pivot(index='trade_date', columns='ts_code', values='low')
print(f"  close shape: {close_df.shape}")

# Step 4: 计算ARBR
print("\n[Step 3] 计算ARBR...")
ar = (high_df - open_df).rolling(26).sum() / ((open_df - low_df).rolling(26).sum() + 0.001)
br = (high_df - close_df.shift(1)).rolling(26).sum() / ((close_df.shift(1) - low_df).rolling(26).sum() + 0.001)
arbr = (ar + br) / 2

# Step 5: 获取市值数据
print("\n[Step 4] 获取流通市值...")
mcap = basic_sub.pivot(index='trade_date', columns='ts_code', values='circ_mv')  # 流通市值(万元)
# 转换为亿元
mcap = mcap / 10000
print(f"  市值 shape: {mcap.shape}")

# 对齐日期和股票
common_dates = close_df.index.intersection(mcap.index)
common_stocks = close_df.columns.intersection(mcap.columns)
print(f"  对齐后: {len(common_dates)} 日期 × {len(common_stocks)} 股票")

close_df = close_df.loc[common_dates, common_stocks]
arbr = arbr.loc[common_dates, common_stocks]
mcap = mcap.loc[common_dates, common_stocks]

# Step 6: 未来5日收益率
forward_5d = close_df.shift(-5) / close_df - 1

# 排除停牌 (5天完全不动)
suspended = (close_df.diff(5) == 0)
forward_5d[suspended] = np.nan

# 只保留至少有40天数据的股票
min_days = 40
valid_stocks = close_df.count() >= min_days
close_df = close_df.loc[:, valid_stocks]
arbr = arbr.loc[:, valid_stocks]
mcap = mcap.loc[:, valid_stocks]
forward_5d = forward_5d.loc[:, valid_stocks]

print(f"  有效股票(≥{min_days}日): {close_df.shape[1]}")

# Step 7: 分池IC测试
print("\n[Step 5] 分池IC评估...")
ic_results = []

test_configs = [
    ('全市场', None, None),
    ('小市值(10~100亿)', 10, 100),
    ('微小市值(<30亿)', 0, 30),
    ('中市值(100~500亿)', 100, 500),
    ('小市值+质量(10~100亿, 第75分位ROE用ARBR代替)', 10, 100),  # 策略二模拟
]

for label, lo, hi in test_configs:
    for t_idx in range(20, len(common_dates) - 5):  # skip first 20 for rolling, last 5 for forward
        date = common_dates[t_idx]
        fwd = forward_5d.iloc[t_idx]
        arb = arbr.iloc[t_idx]
        mc = mcap.iloc[t_idx] if label != '全市场' else None
        
        # 过滤市值
        if mc is not None:
            if hi:
                mask = (mc >= lo) & (mc < hi)
            else:
                mask = mc < 30  # 微小市值
        else:
            mask = pd.Series(True, index=arb.index)
        
        sub_arb = arb[mask].dropna()
        sub_fwd = fwd[mask].dropna()
        
        # 需要至少有20只股票
        common_idx = sub_arb.index.intersection(sub_fwd.index)
        if len(common_idx) < 20:
            continue
        
        # 缩尾1%/99%
        arb_vals = sub_arb[common_idx]
        fwd_vals = sub_fwd[common_idx]
        
        arb_clipped = arb_vals.clip(lower=arb_vals.quantile(0.01), upper=arb_vals.quantile(0.99))
        fwd_clipped = fwd_vals.clip(lower=fwd_vals.quantile(0.01), upper=fwd_vals.quantile(0.99))
        
        if arb_clipped.std() < 0.001:
            continue
        
        ic = arb_clipped.rank().corr(fwd_clipped.rank())
        ic_results.append({'date': date, 'pool': label, 'ic': ic, 'n_stocks': len(common_idx)})

# Step 8: 汇总
print("\n" + "=" * 70)
print("  结果")
print("=" * 70)
print(f"\n{'池':25s} {'截面数':>6s} {'均持股':>6s} {'IC均值':>8s} {'ICIR':>8s} {'+IC%':>7s}")
print("-" * 70)

for label in ['全市场', '小市值(10~100亿)', '微小市值(<30亿)', '中市值(100~500亿)']:
    sub = [r for r in ic_results if r['pool'] == label]
    if not sub:
        continue
    ics = [r['ic'] for r in sub]
    n_stocks = np.mean([r['n_stocks'] for r in sub])
    ic_mean = np.mean(ics)
    ic_std = np.std(ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pct = sum(1 for x in ics if x > 0) / len(ics)
    print(f"{label:25s} {len(sub):>6d} {n_stocks:>6.0f} {ic_mean:>+8.4f} {icir:>+8.3f} {ic_pct:>6.1%}")

# Step 9: 策略二适用性判断
print("\n" + "=" * 70)
print("  策略二适用性")
print("=" * 70)

sc = [r for r in ic_results if r['pool'] == '小市值(10~100亿)']
allm = [r for r in ic_results if r['pool'] == '全市场']

if sc:
    sc_ic_mean = np.mean([r['ic'] for r in sc])
    sc_icir = sc_ic_mean / np.std([r['ic'] for r in sc])
    sc_pct = sum(1 for r in sc if r['ic'] > 0) / len(sc)
    
    print(f"\n  小市值池: IC={sc_ic_mean:+.4f}, ICIR={sc_icir:+.3f}, +IC%={sc_pct:.1%}")
    
    if sc_icir >= 0.3 and sc_pct >= 0.55:
        print("  ✅ ARBR在小市值池中通过阈值 → 可做策略二补充信号")
    else:
        print(f"  ❌ ARBR在小市值池中未通过阈值 (ICIR={sc_icir:+.3f}<0.3 或 +IC%={sc_pct:.1%}<55%)")
    
    # 与全市场对比
    am_ic_mean = np.mean([r['ic'] for r in allm])
    print(f"\n  全市场: IC={am_ic_mean:+.4f}")
    print(f"  子集提升: {sc_ic_mean - am_ic_mean:+.4f}")
    
    if abs(sc_icir) < 0.2:
        print("\n  📊 结论: ARBR在小市值子集中的IC强度仍然很弱(ICIR<0.2)，")
        print("     即使只做条件过滤(二值)而非连续打分，信息量也远不足以")
        print("     对策略二产生有统计意义的增量。不建议加入。")
    else:
        print(f"\n  📊 结论: ...")

print("\n⚠️ 注意: 上述分析基于日频截面IC，策略二的月频调仓可能有不同表现。")
print("  如需进一步验证，可在聚宽平台上跑ARBR过滤版策略二的月频回测。")
