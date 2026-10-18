# -*- coding: utf-8 -*-
"""
本地投票择时网格搜索
5个技术信号在沪深300上测试, 网格搜索最优投票阈值
信号: MA20, MACD, RSI, MA60, VolRatio
"""
import pandas as pd
import numpy as np
import os
from itertools import product

DATA = r'e:\quant\data\raw'

# ============================================================
# 1. 加载沪深300 & 计算信号
# ============================================================
idx = pd.read_csv(os.path.join(DATA, 'index_000300_daily.csv'),
                  encoding='utf-8-sig')
idx['trade_date'] = pd.to_datetime(idx['trade_date'], format='mixed')
idx = idx.sort_values('trade_date').reset_index(drop=True)
idx['ret'] = idx['close'].pct_change()

# ---- 信号1: MA20 ----
idx['ma20'] = idx['close'].rolling(20).mean()
idx['sig_ma20'] = (idx['close'] > idx['ma20']).astype(int)

# ---- 信号2: MACD ----
ema12 = idx['close'].ewm(span=12, adjust=False).mean()
ema26 = idx['close'].ewm(span=26, adjust=False).mean()
idx['macd'] = ema12 - ema26
idx['sig_macd'] = (idx['macd'] > 0).astype(int)

# ---- 信号3: RSI (14日, 非超买) ----
delta = idx['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
idx['rsi'] = 100 - 100 / (1 + rs)
idx['sig_rsi'] = (idx['rsi'] < 70).astype(int)  # 非超买=可持有

# ---- 信号4: MA60 ----
idx['ma60'] = idx['close'].rolling(60).mean()
idx['sig_ma60'] = (idx['close'] > idx['ma60']).astype(int)

# ---- 信号5: VolRatio (20日波动 < 60日波动) ----
idx['vol20'] = idx['ret'].rolling(20).std()
idx['vol60'] = idx['ret'].rolling(60).std()
idx['sig_vol'] = (idx['vol20'] < idx['vol60']).astype(int)

idx = idx.dropna()
print(f"Data: {len(idx)} days, {idx['trade_date'].min().date()} ~ {idx['trade_date'].max().date()}")

# ============================================================
# 2. 网格搜索: 信号组合 + 投票阈值
# ============================================================
signals = ['sig_ma20', 'sig_macd', 'sig_rsi', 'sig_ma60', 'sig_vol']
signal_labels = ['MA20', 'MACD', 'RSI<70', 'MA60', 'VolLow']

# 每个信号可选: 使用(1) 或 不使用(0)
# 投票阈值: 至少N个信号看好才持仓

results = []
total_combos = 2**len(signals)

for mask in product([0, 1], repeat=len(signals)):
    active_sigs = [s for s, m in zip(signals, mask) if m == 1]
    if len(active_sigs) == 0:
        continue

    # 计算总信号数
    idx['total_sig'] = idx[active_sigs].sum(axis=1)
    max_sig = len(active_sigs)

    # 测试不同阈值: 从 1 到 max_sig
    for threshold in range(1, max_sig + 1):
        # 持仓信号
        idx['position'] = (idx['total_sig'] >= threshold).astype(int).shift(1)
        # 策略收益
        idx['strat_ret'] = idx['position'] * idx['ret']

        if idx['strat_ret'].dropna().sum() == 0:
            continue

        rets = idx['strat_ret'].dropna()
        if len(rets) < 252:
            continue

        cum = (1 + rets).prod() - 1
        n_days = len(rets)
        ann_ret = (1 + cum) ** (252 / n_days) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        cum_series = np.cumprod(1 + rets.values)
        max_dd = ((cum_series - np.maximum.accumulate(cum_series)) / np.maximum.accumulate(cum_series)).min()
        win_rate = (rets > 0).mean()
        in_market = idx['position'].mean()

        active_names = [signal_labels[i] for i, m in enumerate(mask) if m == 1]

        results.append({
            'signals': '+'.join(active_names),
            'n_signals': len(active_sigs),
            'threshold': threshold,
            'rule': f'{threshold}/{len(active_sigs)}',
            'cum_return': round(cum * 100, 2),
            'ann_return': round(ann_ret * 100, 2),
            'sharpe': round(sharpe, 2),
            'max_dd': round(max_dd * 100, 2),
            'win_rate': round(win_rate * 100, 1),
            'in_market': round(in_market * 100, 1),
        })

res_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

# ============================================================
# 3. 输出最优
# ============================================================
print(f"\n{'='*80}")
print("投票择时最优方案 (按Sharpe排序)")
print(f"{'='*80}")
print(f"{'信号组合':<35} {'规则':>6} {'累计%':>8} {'年化%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'胜率%':>7} {'持仓%':>7}")
print('-'*80)

for _, r in res_df.head(15).iterrows():
    print(f"{r['signals']:<35} {r['rule']:>6} {r['cum_return']:>8.1f} {r['ann_return']:>8.1f} "
          f"{r['sharpe']:>7.2f} {r['max_dd']:>8.1f} {r['win_rate']:>7.1f} {r['in_market']:>7.1f}")

# 基准: 买入持有
idx_bh = idx['ret'].dropna()
bh_cum = (1 + idx_bh).prod() - 1
bh_ann = (1 + bh_cum) ** (252 / len(idx_bh)) - 1
bh_vol = idx_bh.std() * np.sqrt(252)
bh_sharpe = (bh_ann - 0.02) / bh_vol
print('-'*80)
print(f"{'基准: 买入持有':<35} {'':>6} {bh_cum*100:>8.1f} {bh_ann*100:>8.1f} "
      f"{bh_sharpe:>7.2f} {'-':>8} {'-':>7} {'100.0':>7}")

# 最优方案详情
best = res_df.iloc[0]
print(f"\n最优方案: {best['signals']}, 投票阈值 {best['rule']}")
print(f"  累计收益: {best['cum_return']}%, 年化: {best['ann_return']}%, Sharpe: {best['sharpe']}")
print(f"  最大回撤: {best['max_dd']}%, 胜率: {best['win_rate']}%, 持仓时间: {best['in_market']}%")
