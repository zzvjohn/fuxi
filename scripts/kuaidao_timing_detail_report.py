# -*- coding: utf-8 -*-
"""快刀 v2 季频池子 + 买卖时点明细报告生成器。
输出: output/kuaidao_timing_pools_trades_20260910.md
数据源: full_backtest/pool_P*.csv (8期) + timing/trades_C_quarter_timing.csv + timing/stats.json
"""
import json
import os
import pandas as pd

ROOT = 'e:/quant/data/kuaidao_test/full_backtest'
OUT_MD = 'e:/quant/output/kuaidao_timing_pools_trades_20260910.md'

PERIODS = [
    ('P2023',   '2024-04-30 (年报+一季报)'),
    ('P2024H1', '2024-08-31 (中报)'),
    ('P2024Q3', '2024-10-31 (三季报)'),
    ('P2024',   '2025-04-30 (年报+一季报)'),
    ('P2025H1', '2025-08-31 (中报)'),
    ('P2025Q3', '2025-10-31 (三季报)'),
    ('P2025',   '2026-04-30 (年报+一季报)'),
    ('P2026H1', '2026-08-31 (中报)'),
]

stats = json.load(open(f'{ROOT}/timing/stats.json', encoding='utf-8'))
eff_rows = stats['C_quarter_timing']['periods']  # 生效日与进出场数
EFF = {r['tag']: r for r in eff_rows}

trC = pd.read_csv(f'{ROOT}/timing/trades_C_quarter_timing.csv', dtype={'code': str})
trC['date'] = trC['date'].astype(str)

pools = {}
for tag, _ in PERIODS:
    df = pd.read_csv(f'{ROOT}/pool_{tag}.csv', dtype={'code': str})
    pools[tag] = df

NM = {}
for tag, _ in PERIODS:
    for _, r in pools[tag].iterrows():
        NM[r['code']] = r['name']

# 交易日历 (从B变体净值拿全市场交易日序)
navs = pd.read_csv(f'{ROOT}/timing/nav_variants.csv', dtype={'date': str})
cal = sorted(navs['date'].unique())

def fmt(d):
    return f'{d[:4]}-{d[4:6]}-{d[6:]}'

def prev_trading_day(d):
    i = cal.index(d)
    return cal[i - 1]

def next_trading_day(d):
    i = cal.index(d)
    return cal[i + 1] if i + 1 < len(cal) else None

L = []
def A(s):
    L.append(s)
A('# 快刀 v2 回测：季频选股池 × 买卖时点全明细')
A('')
A('> 变体 C（季频池 + 日频技术择时，用户钦定规格）。价格均为后复权。')
A('> 生成时间 2026-09-11 00:44。数据源: `pool_P*.csv` × 8 期 + `timing/trades_C_quarter_timing.csv`。')
A('')
A('## 1. 换仓机制与时间轴')
A('')
A('| 环节 | 时点定义 |')
A('|---|---|')
A('| 选股决策日 | 财报披露截止日：4-30（年报+一季报）/ 8-31（中报）/ 10-31（三季报），全部数据 ann_date≤该日 |')
A('| 池内旧成员卖出 | **生效日前一交易日收盘**（B/C 两变体相同，如 2024-09-02 生效 → 2024-08-30 收盘卖出离池股） |')
A('| 池内新成员买入·B | **生效日收盘价（MOC）等权建仓**，建仓日不计收益（买入持有变体） |')
A('| 池内新成员买入·C | **入池后空仓待信号**：不自动建仓，出现日频买点（支撑/一阳穿4线）才以信号日收盘买入——C 仓位暴露 72% 的直接来源；期初 P2023 的 15 只同样等信号 |')
A('| 日频买入（C） | ①一阳穿4线：阳线实体穿越 MA5/10/20/30，**信号日收盘价成交**；②支撑位：low≤支撑×1.015 且收回上方（支撑=60日最低或3日确认摆动低点，均不含当日），收盘成交 |')
A('| 日频卖出（C） | 前高/压力位：阻力=60日最高价（仅当阻力在昨收上方），**t-1 预挂限价单**，t 日 high≥阻力则成交于 min(阻力, 当日开盘) |')
A('| 已卖出当日 | 不回购（防当日反复） |')
A('')
A('### 八期生效日与换仓规模')
A('')
A('| 期 | 选股日(披露截止) | 生效日(买入日) | 旧成员卖出日 | 池子 | 新进 | 离池 |')
A('|---|---|---|---|---|---|---|')
for tag, rd in PERIODS:
    e = EFF[tag]
    eff = e['eff']
    sell_day = '—（期初建仓）' if tag == 'P2023' else fmt(prev_trading_day(eff))
    A(f'| {tag} | {rd} | {fmt(eff)} | {sell_day} | {e["n"]} | {e["n_in"]} | {e["n_out"]} |')
A('')
A('期与期之间的窗口：生效日收盘买入 → 下一生效日前一交易日收盘（离池股最后计价日）。')
A('')

A('## 2. 八期池子全明细（15 只/期，标注进出场）')
A('')
for i, (tag, rd) in enumerate(PERIODS):
    e = EFF[tag]
    df = pools[tag].copy()
    prev_tag = PERIODS[i - 1][0] if i else None
    prev_codes = set(pools[prev_tag]['code']) if prev_tag else set()
    cur_codes = set(df['code'])
    leavers = [c for c in pools[prev_tag]['code'] if c not in cur_codes] if prev_tag else []
    st_map = {c: ('新进' if c not in prev_codes else '留任') for c in cur_codes}
    A(f'### {tag}｜选股日 {rd}｜生效日 {fmt(e["eff"])}')
    A('')
    if prev_tag:
        A(f'离池 {len(leavers)} 只：' + ('、'.join(f'{NM[c]}({c})' for c in leavers) if leavers else '无'))
    else:
        A('期初建仓，无离池。')
    A('')
    A('| # | 代码 | 名称 | 行业 | 总分 | +项 | -项 | 进出场 | 十项打分 |')
    A('|---|---|---|---|---|---|---|---|---|')
    for j, (_, r) in enumerate(df.iterrows(), 1):
        sc = r['scores'].replace("'", '').replace('"', '')
        A(f'| {j} | {r["code"]} | {r["name"]} | {r["industry"]} | {r["total"]} | {r["pos"]} | {r["neg"]} | {st_map[r["code"]]} | `{sc}` |')
    A('')

A('## 3. 换仓进出场名单与成交时点')
A('')
A('离池卖出两变体相同（生效日前一交易日收盘）。新进买入两变体不同：**B = 生效日收盘自动建仓**（价格取自 `trades_B_quarter_bh.csv`）；**C = 入池后等首个技术信号**（首个信号日与价格见 §4 对应期明细）。')
A('')
trB = pd.read_csv(f'{ROOT}/timing/trades_B_quarter_bh.csv', dtype={'code': str})
trB['date'] = trB['date'].astype(str)
for i, (tag, _) in enumerate(PERIODS):
    e = EFF[tag]
    eff = e['eff']
    df = pools[tag]
    prev_tag = PERIODS[i - 1][0] if i else None
    A(f'**{tag}**（生效 {fmt(eff)}）')
    A('')
    if i == 0:
        bb = trB[(trB['date'] == eff) & (trB['side'] == 'BUY')].set_index('code')['px']
        A(f'- 期初建仓 {len(bb)} 只：B 于 {fmt(eff)} 收盘全仓买入；C 中 15 只全部空仓待信号（首笔信号买入见 §4）。')
        A('')
        continue
    prev = pools[prev_tag]
    cur = set(df['code'])
    outs = [c for c in prev['code'] if c not in cur]
    ins = [c for c in df['code'] if c not in set(prev['code'])]
    sell_day = prev_trading_day(eff)
    bb = trB[(trB['date'] == eff) & (trB['side'] == 'BUY') & (trB['reason'] == '建仓')].set_index('code')['px']
    # C 口径: 新进股的首笔技术买入日
    c_first = {}
    for c in ins:
        g = trC[(trC['code'] == c) & (trC['side'] == 'BUY') & (trC['date'] >= eff)]
        if len(g):
            r0 = g.iloc[0]
            c_first[c] = f'{fmt(r0.date)} {r0.reason}'
    A(f'- 卖出 {len(outs)} 只（{fmt(sell_day)} 收盘，B/C 相同）：' + '、'.join(f'{NM.get(c, c)}({c})' for c in outs))
    A(f'- 新进 {len(ins)} 只：B 于 {fmt(eff)} 收盘买入（' + '、'.join(f'{NM.get(c, c)} @ {bb.get(c, "—")}' for c in ins) + '）；')
    if c_first:
        A('  C 入池后等信号，首笔买入：' + '；'.join(f'{NM.get(c, c)} {v}' for c, v in c_first.items()) + '。')
    else:
        A('  C 本期窗口内无信号触发（空仓踏空）。')
    A('')

A('## 4. 日频技术交易明细（仅 C 变体，全部 ' + str(len(trC)) + ' 笔）')
A('')
A('按期分组，含换仓新进股的首笔信号买入（C 不在生效日自动建仓，全部买入都源于此处的技术信号）。买入=信号日收盘成交（MOC）；卖出=限价单成交于 min(60日最高, 当日开盘)。')
A('')
for tag, _ in PERIODS:
    if tag == 'P2026H1':
        w0, w1 = EFF[tag]['eff'], cal[-1]
    else:
        idx = [t for t, _ in PERIODS].index(tag)
        w0 = EFF[tag]['eff']
        w1 = prev_trading_day(EFF[PERIODS[idx + 1][0]]['eff'])
    seg = trC[(trC['date'] >= w0) & (trC['date'] <= w1)].copy()
    A(f'### {tag}（窗口 {fmt(w0)} ~ {fmt(w1)}，{len(seg)} 笔）')
    A('')
    if len(seg) == 0:
        A('（本期无技术交易）')
        A('')
        continue
    A('| 日期 | 代码 | 名称 | 方向 | 成交价 | 触发原因 |')
    A('|---|---|---|---|---|---|')
    for r in seg.itertuples():
        A(f'| {fmt(r.date)} | {r.code} | {NM.get(r.code, "?")} | {r.side} | {r.px} | {r.reason} |')
    A('')

A('## 5. 汇总统计')
A('')
nb = (trC['side'] == 'BUY').sum(); ns = (trC['side'] == 'SELL').sum()
A(f'- C 变体共 {len(trC)} 笔：买 {nb} / 卖 {ns}。')
A('- 买入原因分布：' + '，'.join(f'{k} {v} 笔' for k, v in trC[trC.side == "BUY"].reason.value_counts().items()))
A('- 卖出原因分布：' + '，'.join(f'{k} {v} 笔' for k, v in trC[trC.side == "SELL"].reason.value_counts().items()))
# 持有天数: 每只票 FIFO 配对最近一次买入
holds = []
for code, g in trC[trC.code.isin(NM)].groupby('code'):
    buys = []
    for r in g.itertuples():
        if r.side == 'BUY':
            buys.append(r.date)
        else:
            if buys:
                b = buys.pop(0)  # FIFO 配对
                holds.append(cal.index(r.date) - cal.index(b))
A(f'- 完成回合（同票买→卖配对）{len(holds)} 个，平均持有 {sum(holds)/len(holds):.0f} 个交易日，中位 {sorted(holds)[len(holds)//2]} 日，最长 {max(holds)} / 最短 {min(holds)} 日（配对可跨换仓期：留任股不强制平仓）。')
A('- 期初（2024-05-06）：B 全仓 15 只收盘买入；C 15 只全部空仓待信号。期末 P2026H1 池 15 只中，期末前触发前高卖出的已平仓，其余持有至回测结束（2026-09-10）。')
A('')
A('### 逐期技术交易节拍')
A('')
A('| 期 | 窗口 | 买入 | 卖出 | 备注 |')
A('|---|---|---|---|---|')
for tag, _ in PERIODS:
    idx = [t for t, _ in PERIODS].index(tag)
    w0 = EFF[tag]['eff']
    w1 = prev_trading_day(EFF[PERIODS[idx + 1][0]]['eff']) if idx + 1 < len(PERIODS) else cal[-1]
    seg = trC[(trC['date'] >= w0) & (trC['date'] <= w1)]
    nb_ = (seg['side'] == 'BUY').sum(); ns_ = (seg['side'] == 'SELL').sum()
    rm = seg['reason'].value_counts().to_dict()
    note = '，'.join(f'{k}×{v}' for k, v in rm.items())
    A(f'| {tag} | {fmt(w0)}~{fmt(w1)} | {nb_} | {ns_} | {note} |')
A('')

os.makedirs('e:/quant/output', exist_ok=True)
open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
print('written', OUT_MD, len(L), 'lines')
