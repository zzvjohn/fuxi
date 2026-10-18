"""
验证脚本 (2026-09-16): 2 个 Stage2 PASS 因子的 FA 注册与 IC 复现
=============================================================
口径 A (复现原实验): 与 scripts/daily_factor_hypothesis.py 一致
  - 不复权 close/amount (data/raw/daily_prices.csv)
  - 近 270 天日频面板, forward_5d = close.shift(-5)/close - 1
  - 截面 Spearman Rank IC; ICIR = |mean(ic)/std(ic)|
目标: 复现 market_state_risk_adjusted_momentum ICIR 0.620 / +IC% 76.3%
      复现 crowded_reversal_5d_large_order_flow   ICIR 0.335 / +IC% 60.0%
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factors import ALL_FACTORS            # noqa: E402
from config import FACTOR_DEFS             # noqa: E402

TARGETS = ['market_state_risk_adjusted_momentum', 'crowded_reversal_5d_large_order_flow']

print('=' * 70)
print('[1] 注册表检查')
print('=' * 70)
for t in TARGETS:
    print(f"  {t}: ALL_FACTORS={'✓' if t in ALL_FACTORS else '✗'}  "
          f"FACTOR_DEFS={'✓' if t in FACTOR_DEFS else '✗'}  "
          f"category={FACTOR_DEFS.get(t, {}).get('category')}")
print(f"  ALL_FACTORS 总数: {len(ALL_FACTORS)} | FACTOR_DEFS 总数: {len(FACTOR_DEFS)}")

# ---------------------------------------------------------------- 数据
print()
print('=' * 70)
print('[2] 加载数据 (原实验口径: 不复权日频, 近 270 天)')
print('=' * 70)
ROOT = Path('e:/quant')
local_path = ROOT / 'data' / 'raw' / 'daily_prices.csv'
today = datetime.now()
end_dt = pd.Timestamp(today.strftime('%Y-%m-%d'))
start_dt = pd.Timestamp((today - timedelta(days=270)).strftime('%Y-%m-%d'))

daily = pd.read_csv(
    local_path, dtype={'ts_code': str, 'trade_date': str},
    usecols=['ts_code', 'trade_date', 'close', 'vol', 'amount'],
)
daily['trade_date'] = pd.to_datetime(daily['trade_date'], format='mixed', errors='coerce')
daily = daily[(daily['trade_date'] >= start_dt) & (daily['trade_date'] <= end_dt)]
daily = daily.drop_duplicates(subset=['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date'])
close_df = daily.pivot(index='trade_date', columns='ts_code', values='close')
amount_df = daily.pivot(index='trade_date', columns='ts_code', values='amount')
print(f"  日线 {len(daily)} 条 | 面板 {close_df.shape} | {start_dt.date()} ~ {end_dt.date()}")

# 订单流 (moneyflow_daily.csv, trade_date 为 int YYYYMMDD)
mf = pd.read_csv(ROOT / 'data' / 'raw' / 'moneyflow_daily.csv',
                 usecols=['ts_code', 'trade_date', 'buy_lg_vol', 'sell_lg_vol'],
                 dtype={'ts_code': str})
mf['trade_date'] = pd.to_datetime(mf['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
mf = mf[mf['trade_date'] >= start_dt]
buy_lg = mf.pivot_table(index='trade_date', columns='ts_code', values='buy_lg_vol', aggfunc='sum')
sell_lg = mf.pivot_table(index='trade_date', columns='ts_code', values='sell_lg_vol', aggfunc='sum')
print(f"  订单流 {len(mf)} 行 | 覆盖 {mf['trade_date'].min().date()} ~ {mf['trade_date'].max().date()}")

price_data = {'close': close_df, 'amount': amount_df,
              'buy_lg_vol': buy_lg, 'sell_lg_vol': sell_lg}

# ---------------------------------------------------------------- IC
def compute_ic(factor_values, forward_returns):
    """与原实验完全一致的 Rank IC / ICIR 口径"""
    ics = []
    for dt in sorted(factor_values.index.intersection(forward_returns.index)):
        fv = factor_values.loc[dt].dropna()
        fr = forward_returns.loc[dt].dropna()
        common = fv.index.intersection(fr.index)
        if len(common) < 30:
            continue
        ic = pd.Series(fv[common]).rank().corr(pd.Series(fr[common]).rank())
        if not np.isnan(ic):
            ics.append((dt, ic))
    if not ics:
        return dict(ic_mean=np.nan, ic_std=np.nan, icir=np.nan, pos_pct=np.nan, n=0)
    s = pd.Series([x[1] for x in ics])
    return dict(ic_mean=s.mean(), ic_std=s.std(),
                icir=abs(s.mean() / s.std()) if s.std() > 0 else np.nan,
                pos_pct=(s > 0).mean(), n=len(s))

forward_5d = close_df.shift(-5) / close_df - 1

print()
print('=' * 70)
print('[3] IC 复现 (目标: 0.620 / 0.335)')
print('=' * 70)
EXPECT = {'market_state_risk_adjusted_momentum': 0.620,
          'crowded_reversal_5d_large_order_flow': 0.335}
rows = []
for name in TARGETS:
    obj = ALL_FACTORS[name]()
    fv = obj.compute(price_data=price_data, financial_data=None, valuation_data=None)
    if fv is None or fv.empty:
        print(f"  ✗ {name}: compute 返回空")
        rows.append((name, np.nan, np.nan, 0, EXPECT[name]))
        continue
    r = compute_ic(fv, forward_5d)
    exp = EXPECT[name]
    ok = '✓' if (not np.isnan(r['icir']) and abs(r['icir'] - exp) < 0.06) else '⚠'
    print(f"  {ok} {name}")
    print(f"      ICIR={r['icir']:.3f} (目标 {exp:.3f}) | IC={r['ic_mean']:+.4f} | "
          f"+IC%={r['pos_pct']*100:.1f}% | n={r['n']} 截面")
    rows.append((name, r['icir'], r['ic_mean'], r['n'], exp))

print()
print('=' * 70)
print('[4] 结果汇总')
print('=' * 70)
print(f"  {'因子':<42}{'实测ICIR':>9}{'目标':>8}{'IC均值':>10}{'截面数':>7}")
for name, icir, icm, n, exp in rows:
    print(f"  {name:<42}{icir:>9.3f}{exp:>8.3f}{icm:>10.4f}{n:>7}")
print()
print('  注: 本脚本用不复权 close 复现原实验口径; FA 生产管线用后复权 close_hfq,')
print('      数值会有小幅差异, 但符号方向与量级应一致。')
