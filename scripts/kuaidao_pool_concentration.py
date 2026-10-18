# -*- coding: utf-8 -*-
"""快刀季频池 收益集中度诊断 (2026-09-11)

回答: "为什么任何价格型出场都亏?" —— 因为池子的收益来自极少数必须整段持有的标的。
口径: 对每个池期, 计算每个成员在该期内的收益; 观察集中度 (top-N 占比 / 极值 / 分布)。
"""
import sys, os, json, bisect
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import importlib
m = importlib.import_module('kuaidao_timing_backtest_v4')

OUT = 'data/kuaidao_test/full_backtest'
CAL = m.load_calendar()
EFFS = sorted(min([d for d in CAL if d > rd]) for rd, _ in m.QUARTERLY)
EFFS_ALL = EFFS + [CAL[-1]]

rows = []
for i in range(len(EFFS)):
    rd, tag = m.QUARTERLY[i]
    eff = EFFS[i]
    nxt = EFFS_ALL[i + 1]
    pool = m.load_pool(tag)
    for c in pool:
        d = m.get_px(c)
        if d is None or eff not in d.index or nxt not in d.index:
            continue
        r = float(d.loc[nxt, 'close']) / float(d.loc[eff, 'close']) - 1
        rows.append(dict(tag=tag, eff=eff, code=c, ret=r))

df = pd.DataFrame(rows)
print(f'position-periods: n={len(df)}  periods={df["tag"].nunique()}')
print('\n--- 逐期分布 ---')
print(f'{"period":10s} {"n":>3s} {"mean":>8s} {"median":>8s} {"max":>9s} {"min":>9s} '
      f'{"top3sum":>9s} {"sum":>9s} {"conc":>6s}')
for tag, g in df.groupby('tag', sort=False):
    s = g['ret'].sort_values(ascending=False)
    tot = s.sum()
    top3 = s.head(3).sum()
    print(f'{tag:10s} {len(g):3d} {g["ret"].mean():+7.2%} {g["ret"].median():+7.2%} '
          f'{g["ret"].max():+8.1%} {g["ret"].min():+8.1%} {top3:+8.2f} {tot:+8.2f} '
          f'{(top3/tot if tot else float("nan")):6.1%}')

print('\n--- 全样本 ---')
print(f'  中位收益={df["ret"].median():+.2%}  均值={df["ret"].mean():+.2%}')
print(f'  正收益占比={float((df["ret"]>0).mean()):.1%}   >+50% 占比={float((df["ret"]>0.5).mean()):.1%}'
      f'   >+100% 占比={float((df["ret"]>1.0).mean()):.1%}')
print(f'  前 10% 分位={df["ret"].quantile(0.9):+.1%}  后 10% 分位={df["ret"].quantile(0.1):+.1%}')
print(f'  最大={df["ret"].max():+.1%} ({df.loc[df["ret"].idxmax(),"code"]} {df.loc[df["ret"].idxmax(),"tag"]})')
print(f'  最小={df["ret"].min():+.1%} ({df.loc[df["ret"].idxmin(),"code"]} {df.loc[df["ret"].idxmin(),"tag"]})')
pos = df['ret'].clip(lower=0).sum()
top = df['ret'].nlargest(max(1, int(len(df) * 0.1))).clip(lower=0).sum()
print(f'  前 10% 仓位贡献了全部正收益的 {top/pos:.1%}')
# 单期最赚钱标的占该期等权收益的比例
print('\n--- 单期"关键标的" (最佳标的对等权收益的贡献) ---')
for tag, g in df.groupby('tag', sort=False):
    s = g['ret'].sort_values(ascending=False)
    eq = float(s.mean())
    best = float(s.iloc[0]); best_code = str(g.set_index('code')['ret'].idxmax())
    share = (best / len(g)) / eq if eq else float('nan')
    print(f'{tag:10s} 等权={eq:+7.2%}  最佳={best:+8.1%} ({best_code})  '
          f'贡献={best/len(g):+7.2%}pp = 等权的 {share:6.0%}')
json.dump(df.to_dict('records'), open(f'{OUT}/pool_concentration.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=str)
print(f'\n[saved] {OUT}/pool_concentration.json')
