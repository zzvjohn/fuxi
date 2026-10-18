# -*- coding: utf-8 -*-
"""P-20260918-001 前置: 龙虎榜三路席位分类命中/污染量化 (只读)。"""
import pandas as pd
import os

p = r'e:\quant\data\raw\top_inst.csv'
ti = pd.read_csv(p, low_memory=False, dtype={'ts_code': str, 'trade_date': str})
print('rows', len(ti), '| cols', list(ti.columns))
print('date range', ti['trade_date'].min(), '->', ti['trade_date'].max())

inst = ti['exalter'].str.contains('机构专用', na=False)
north = ti['exalter'].str.contains('深股通专用|沪股通专用', na=False, regex=True)
yz = (~ti['exalter'].str.contains('机构专用|深股通专用|沪股通专用', na=False, regex=True)) \
     & ti['exalter'].str.contains('营业部', na=False)

print()
print('--- 现行 substring 分类 ---')
print('机构专用      :', int(inst.sum()))
print('深/沪股通专用 :', int(north.sum()))
print('营业部(游资)  :', int(yz.sum()))
print('三路合计      :', int(inst.sum() + north.sum() + yz.sum()), '/ 总', len(ti))
print('未归类        :', int(len(ti) - inst.sum() - north.sum() - yz.sum()))

print()
print('--- 含 "股通" 但未进 north 桶 (疑似漏捕获) ---')
mg = ti['exalter'].str.contains('股通', na=False) & ~north
print('rows =', int(mg.sum()), '| distinct =', ti.loc[mg, 'exalter'].nunique())
print(ti.loc[mg, 'exalter'].value_counts().head(20).to_string())

print()
print('--- 含 "机构" 但未进 inst 桶 ---')
mi = ti['exalter'].str.contains('机构', na=False) & ~inst
print('rows =', int(mi.sum()), '| distinct =', ti.loc[mi, 'exalter'].nunique())
print(ti.loc[mi, 'exalter'].value_counts().head(15).to_string())

print()
print('--- 未归类席位 top30 (看是否真有漏桶) ---')
un = ~(inst | north | yz)
print('rows =', int(un.sum()), '| distinct =', ti.loc[un, 'exalter'].nunique())
print(ti.loc[un, 'exalter'].value_counts().head(30).to_string())

print()
print('--- north 桶内席位名 (现行, top15) ---')
print(ti.loc[north, 'exalter'].value_counts().head(15).to_string())

print()
print('--- 三路桶 逐年 rows ---')
ti['yr'] = ti['trade_date'].str[:4]
g = pd.DataFrame({'inst': inst, 'north': north, 'yz': yz, 'un': un, 'yr': ti['yr']})
print(g.groupby('yr')[['inst', 'north', 'yz', 'un']].sum().to_string())
