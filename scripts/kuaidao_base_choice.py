# -*- coding: utf-8 -*-
"""快刀 base 架构选择: 常数缩放暴露 vs 全暴露 —— Sharpe 谁大?

结论预告(数学): 在 rf>0 下, 常数缩放 k 的 Sharpe(k) = (k*m - rf_d)/(k*s)*sqrt(250)
= (m - rf_d/k)/s*sqrt(250) 对 k **单调递增** ⇒ 常数缩仓**永远降低** Sharpe。
⇒ 报告表 6.2 里 "同暴露 61% 买入持有 Sharpe 0.82" 的 0.82 来自
  scaledBH(p_t 路径) —— 一个需要预知变体 p_t 的**基准**, 不是可实盘策略。
本脚本用真实净值验证上述判断, 并给出常数缩放前沿。
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
import os
os.chdir('e:/quant')
from kuaidao_timing_backtest_v4 import perf  # noqa: E402

NAV = 'data/kuaidao_test/full_backtest/timing4/nav_variants4.csv'
df = pd.read_csv(NAV, index_col=0)

print('=== 1. 基线 B_q_bh + 各 saved scaledBH 的绩效 ===')
rows = []
for col in df.columns:
    if col == 'bench_hs300':
        continue
    p = perf(df[col])
    rows.append((col, p['total'], p['sharpe'], p['mdd']))
for name, tot, sh, mdd in rows:
    print(f'{name:34s} total={tot:+8.2%} sharpe={sh:5.2f} mdd={mdd:7.1%}')

print('\n=== 2. 常数缩放 k × r_bh 的前沿 (Sharpe 是否单调?) ===')
b = df['B_q_bh']
r_bh = b.pct_change().dropna()
# 先确认 B 的实际年化波动与均值
print(f'B_q_bh: days={len(b)} mean_daily={r_bh.mean():.6f} std_daily={r_bh.std():.6f} '
      f'ann_ret={r_bh.mean()*250:.2%} ann_vol={r_bh.std()*np.sqrt(250):.2%} '
      f'mu/sigma={r_bh.mean()/r_bh.std()*np.sqrt(250):.3f}')
print(f'{"k":>6s} {"total":>9s} {"sharpe":>7s} {"mdd":>8s}')
front = []
for k in [0.40, 0.45, 0.50, 0.55, 0.60, 0.61, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]:
    nav = (1 + k * r_bh).cumprod()
    nav = pd.Series(np.r_[1.0, nav.values], index=b.index[:len(nav) + 1])
    p = perf(nav)
    front.append((k, p['total'], p['sharpe'], p['mdd']))
    print(f'{k:6.2f} {p["total"]:+9.2%} {p["sharpe"]:7.2f} {p["mdd"]:8.1%}')

print('\n=== 3. 判定 ===')
sh = [f[2] for f in front]
print(f'Sharpe 单调递增: {all(sh[i] < sh[i+1] for i in range(len(sh)-1))}')
print(f'k=0.61 -> Sharpe {sh[front.index(next(f for f in front if abs(f[0]-0.61)<1e-9))]:.2f} '
      f'| k=1.00 -> Sharpe {sh[-1]:.2f}')
json.dump(dict(frontier=[dict(k=k, total=t, sharpe=s, mdd=m) for k, t, s, m in front],
               saved={n: dict(total=t, sharpe=s, mdd=m) for n, t, s, m in rows}),
          open('data/kuaidao_test/full_backtest/timing4/base_choice.json', 'w'),
          ensure_ascii=False, indent=1)
print('[saved] timing4/base_choice.json')
