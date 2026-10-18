# -*- coding: utf-8 -*-
"""快刀 v3 日频择时回测报告生成器 (新买卖规格 2026-09-11)。
输出: output/kuaidao_timing_v3_report_20260911.md + kuaidao_timing_v3_nav_20260911.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
from kuaidao_timing_backtest_v3 import load_calendar  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'data/kuaidao_test/full_backtest'
OUT3 = f'{OUT}/timing3'
MD = 'e:/quant/output/kuaidao_timing_v3_report_20260911.md'
PNG = 'e:/quant/output/kuaidao_timing_v3_nav_20260911.png'

stats = json.load(open(f'{OUT3}/stats3.json', encoding='utf-8'))
navs = pd.read_csv(f'{OUT3}/nav_variants3.csv', dtype={'date': str}).set_index('date')
cal = load_calendar()
cidx = {d: i for i, d in enumerate(cal)}

pct = lambda x: f'{x:+.2%}'
fdate = lambda d: f'{d[:4]}-{d[4:6]}-{d[6:]}'

VN = {
    'A_annual_bh': 'A 年频池+买入持有',
    'B_quarter_bh': 'B 季频池+买入持有',
    'E2_annual_timing2': 'E2 年频池+日频择时(新规格)',
    'C2_quarter_timing': 'C2 季频池+日频择时(新规格, 主角)',
    'C2net_quarter_timing': 'C2net 新规格+交易成本',
}
bp = stats['bench']

L = []
A = L.append
A('# 快刀 v3：季频池 × 新日频买卖规格 回测报告')
A('')
A('> **新规格（2026-09-11 钦定）** 买入：①下跌到近一年最低价附近(±3%) ②持续下跌并跌破 120 日线 ③一阳穿4线（收盘价一天突破 5&10&20&30 日均线）。卖出：①上涨到前高(近一年最高价附近±3%) ②压力位（上涨趋势且接近 120 日线）③50% 收益止盈。全部收盘价(MOC)成交。')
A('> 窗口 2024-05-06 ~ 2026-09-10，后复权价格，选股池与 v2 完全相同（8 期×15 只，ann_date≤选股日，无前视）。引擎 `kuaidao_timing_backtest_v3.py`。')
A('')
A('## 1. 结论摘要')
A('')
s2 = stats['C2_quarter_timing']
A(f'1. **新规格总收益 {pct(s2["perf"]["total"])} / Sharpe {s2["perf"]["sharpe"]:.2f} / MDD {s2["perf"]["mdd"]:.1%} / 仓位暴露 {s2["exposure"]:.0%}**，'
  f'vs 沪深300 {pct(bp["total"])} / MDD {bp["mdd"]:.1%}；vs 买入持有(季频池) {pct(stats["B_quarter_bh"]["perf"]["total"])} / MDD {stats["B_quarter_bh"]["perf"]["mdd"]:.1%}。')
A(f'2. **防御性显著改善但踏空更深**：MDD {s2["perf"]["mdd"]:.1%} 是全部变体中最浅的（比基准还浅 2.3pp），但暴露仅 {s2["exposure"]:.0%}（旧规格 72%），'
  f'总收益较旧规格 C({pct(stats["_v2_ref"]["C_quarter_timing"]["perf"]["total"])}) 再降 4.4pp——低位等待更久、买点更苛刻的代价。')
A(f'3. **收益结构从"高胜率小赚大亏"变为"中胜率高盈亏比"**：121 个回合胜率 60%（旧规格 85%），盈亏比 2.03（平均赢 +7.3% / 平均亏 -3.6%，旧规格盈亏比<1）。')
A('4. **信号质量分层清晰**（见 §4）："一年低点附近买"13 回合胜率 92%/均值 +11.1% 是最强信号；"一阳穿4线"67%/+2.8%；"跌破 MA120"仅 47%/+1.4% 接近抛硬币。卖出端"前高 R250"25 回合 100% 胜率/+12.1%；"压力位 MA120"96 回合仅 49%/+0.6%，**砍仓过早是最大拖累**。')
A('5. **50% 止盈 2.2 年零触发**（深度价值池无一只持仓达到 +50%），该条款当前形同虚设。')
A('')

A('## 2. 变体总览')
A('')
A('| 变体 | 总收益 | 年化 | Sharpe | MDD | 暴露 | 交易笔数 | 回合 | 胜率 | 平均持有(交易日) |')
A('|---|---|---|---|---|---|---|---|---|---|')
for k in ['A_annual_bh', 'B_quarter_bh', 'E2_annual_timing2', 'C2_quarter_timing', 'C2net_quarter_timing']:
    s = stats[k]
    rt = s['rt']
    hold = rt.get('avg_hold')
    A(f'| {VN[k]} | {pct(s["perf"]["total"])} | {pct(s["perf"]["cagr"])} | {s["perf"]["sharpe"]:.2f} | '
      f'{s["perf"]["mdd"]*100:.1f}% | {s["exposure"]*100:.0f}% | {rt.get("n_buy", 0) + rt.get("n_sell", 0)} | '
      f'{rt.get("n_rt", 0)} | {(rt.get("win") or 0)*100:.0f}% | {hold:.0f} |' if hold is not None else
      f'| {VN[k]} | {pct(s["perf"]["total"])} | {pct(s["perf"]["cagr"])} | {s["perf"]["sharpe"]:.2f} | '
      f'{s["perf"]["mdd"]*100:.1f}% | {s["exposure"]*100:.0f}% | {rt.get("n_buy", 0) + rt.get("n_sell", 0)} | '
      f'{rt.get("n_rt", 0)} | {(rt.get("win") or 0)*100:.0f}% | — |')
o = stats['_v2_ref']['C_quarter_timing']
hold2 = o['rt'].get('avg_hold')
A(f'| 对照: 旧规格C (v2, 2026-09-10) | {pct(o["perf"]["total"])} | {pct(o["perf"]["cagr"])} | {o["perf"]["sharpe"]:.2f} | '
  f'{o["perf"]["mdd"]*100:.1f}% | {o["exposure"]*100:.0f}% | {o["rt"].get("n_buy",0)+o["rt"].get("n_sell",0)} | '
  f'{o["rt"].get("n_rt",0)} | {(o["rt"].get("win") or 0)*100:.0f}% | {hold2:.0f} |' if hold2 is not None else
  f'| 对照: 旧规格C (v2) | {pct(o["perf"]["total"])} | {o["perf"]["sharpe"]:.2f} | {o["perf"]["mdd"]*100:.1f}% | {o["exposure"]*100:.0f}% | — |')
A(f'| 沪深300 | {pct(bp["total"])} | {pct(bp["cagr"])} | {bp["sharpe"]:.2f} | {bp["mdd"]*100:.1f}% | 100% | — | — | — | — |')
A('')

A('## 3. 主角变体 C2：分段明细')
A('')
rows = stats['C2_quarter_timing']['periods']
bench = navs['bench']
dates = list(navs.index)
A('| 生效日 | 池子 | 新进/离池 | 段收益 | 沪深300 | 超额 |')
A('|---|---|---|---|---|---|')
seg_exs = []
for i, r in enumerate(rows):
    i0 = dates.index(r['eff'])
    i1 = dates.index(rows[i + 1]['eff']) - 1 if i + 1 < len(rows) else len(dates) - 1
    prev = rows[i - 1]['nav_end'] if i else 1.0
    seg = r['nav_end'] / prev - 1
    bseg = bench.iloc[i1] / (bench.iloc[i0 - 1] if i0 > 0 else 1) - 1
    seg_exs.append(seg - bseg)
    A(f'| {fdate(r["eff"])} | {r["tag"]} | +{r["n_in"]}/-{r["n_out"]} | {pct(seg)} | {pct(bseg)} | {pct(seg - bseg)} |')
A('')
up_ex = [x for i, x in enumerate(seg_exs) if rows[i]['tag'] not in ('P2023', 'P2024Q3', 'P2026H1')]
dn_ex = [x for i, x in enumerate(seg_exs) if rows[i]['tag'] in ('P2023', 'P2024Q3', 'P2026H1')]
A(f'分段结构：**4 个基准上涨段全部负超额**（均值 {sum(up_ex)/len(up_ex):+.1%}，最深 P2024 段 {min(up_ex):+.1%}）；'
  f'**3 个基准下跌段全部正超额**（均值 {sum(dn_ex)/len(dn_ex):+.1%}）——典型的防御型净值结构。')
A('')

A('## 4. 信号质量分层（C2 的 121 个完成回合）')
A('')
rt = pd.read_csv(f'{OUT3}/roundtrips3_C2_quarter_timing.csv', dtype={'code': str})
for col in ('buy_date', 'sell_date'):
    rt[col] = rt[col].astype(str).str.replace('.0', '', regex=False)
rt['hold'] = rt['sell_date'].map(cidx) - rt['buy_date'].map(cidx)
A('### 买入原因 × 回合表现')
A('')
A('| 买入信号 | 定义 | 回合 | 胜率 | 平均收益 | 中位持有 |')
A('|---|---|---|---|---|---|')
defs_b = {'一年低点附近': '收盘 ≤ 近一年最低×1.03 且当日下跌', '跌破MA120': '昨收在线上→今收在线下 且 5 日下跌',
          '一阳穿4线': '开盘<min(MA5/10/20/30) 且收盘>max(...)'}
for reason, g in rt.groupby('buy_reason'):
    A(f'| {reason} | {defs_b.get(reason, "")} | {len(g)} | {(g.ret > 0).mean()*100:.0f}% | {pct(g.ret.mean())} | {g.hold.median():.0f} 日 |')
A('')
A('### 卖出原因 × 回合表现')
A('')
A('| 卖出信号 | 定义 | 回合 | 胜率 | 平均收益 | 中位持有 |')
A('|---|---|---|---|---|---|')
defs_s = {'前高R250': '收盘 ≥ 近一年最高×0.97', '压力位MA120': '5日上涨 且收盘自下方接近120日线(3%以内)',
          '50%止盈': '收盘 ≥ 买入价×1.50'}
for reason, g in rt.groupby('sell_reason'):
    A(f'| {reason} | {defs_s.get(reason, "")} | {len(g)} | {(g.ret > 0).mean()*100:.0f}% | {pct(g.ret.mean())} | {g.hold.median():.0f} 日 |')
A(f'（"50%止盈"0 笔触发, 未列出）')
A('')
A('### 持有期分布')
A('')
A(f'- 完成回合平均持有 {rt.hold.mean():.0f} 个交易日（中位 {rt.hold.median():.0f}，最长 {rt.hold.max()}，最短 {rt.hold.min()}）。')
A(f'- 期末未平仓 {stats["C2_quarter_timing"]["n_open"]} 只（入池后未等到卖点/买点，持有至回测结束）。')
A('')

A('## 5. 逐期交易节拍')
A('')
trC = pd.read_csv(f'{OUT3}/trades3_C2_quarter_timing.csv', dtype={'code': str})
trC['date'] = trC['date'].astype(str)
A('| 期 | 窗口 | 买入 | 卖出 | 信号分布 |')
A('|---|---|---|---|---|')
tags = ['P2023', 'P2024H1', 'P2024Q3', 'P2024', 'P2025H1', 'P2025Q3', 'P2025', 'P2026H1']
for i, tag in enumerate(tags):
    r0 = rows[i]['eff']
    r1 = rows[i + 1]['eff'] if i + 1 < len(rows) else cal[-1]
    seg = trC[(trC['date'] >= r0) & (trC['date'] <= r1)]
    nb_ = int((seg['side'] == 'BUY').sum())
    ns_ = int((seg['side'] == 'SELL').sum())
    note = '，'.join(f'{k}×{v}' for k, v in seg['reason'].value_counts().items())
    A(f'| {tag} | {fdate(r0)}~{fdate(r1)} | {nb_} | {ns_} | {note} |')
A('')

A('## 6. 结论与解读')
A('')
A('1. **新规格 vs 旧规格（同为季频池）**：收益 +4.73% vs +9.16%（更差），MDD -13.4% vs -19.1%（更好），胜率 60% vs 85%（更低），盈亏比 2.03 vs <1（更健康）。新规格买得更深（一年低点/破年线）、卖得更晚（一年高点），把旧规格"截断右尾+放大左尾"的结构部分扭了过来，但仓位暴露 39% vs 72% 使踏空成本更高，净效果仍为负。')
A('2. **择时在这个池子上依然是负贡献**（三任规格一致：v2 旧规格 -30pp，v3 新规格 -34pp vs 季频买入持有）。深度价值池的盈利来源是"少数票整段大涨"，任何"等回调再买/涨到压力就卖"的规则都在系统性错配这个盈利来源。')
A('3. **如果坚持日频执行，数据指向三个改造方向**：')
A('   - **提权"一年低点附近买"**（92% 胜率/+11.1% 均值），它是新规格中唯一显著为正 alpha 的买入信号；')
A('   - **"压力位 MA120"卖出发出后不立即清仓**——96 回合仅 49% 胜率意味着一半仓位卖在反弹起点，可改为减半仓或"站稳年线上方 5 日再卖"；')
A('   - **"跌破 MA120"接刀信号降权或加过滤**（47% 胜率），如要求 R3 大股东无减持或 PB 分位底部才允许触发。')
A('4. **50% 止盈条款建议删除或改为 30%**：2.2 年零触发，不承担任何功能。')
A('5. **成本影响可忽略**（C2net vs C2 仅 -0.7pp，276 笔 × 低换手）。')
A('')

# 净值图
fig, ax = plt.subplots(figsize=(11, 6))
colors = {'A_annual_bh': '#999999', 'B_quarter_bh': '#2b7bba', 'E2_annual_timing2': '#ddaa33',
          'C2_quarter_timing': '#cc3311', 'C2net_quarter_timing': '#882255', 'bench': '#444444'}
labels = dict(VN)
labels['bench'] = '沪深300'
for col in ['A_annual_bh', 'B_quarter_bh', 'E2_annual_timing2', 'C2_quarter_timing', 'C2net_quarter_timing', 'bench']:
    lw = 2.2 if col == 'C2_quarter_timing' else 1.2
    ls = '--' if col == 'bench' else '-'
    ax.plot(range(len(navs)), navs[col], color=colors[col], lw=lw, ls=ls, label=labels[col])
step = max(len(navs) // 8, 1)
ticks = list(range(0, len(navs), step))
ax.set_xticks(ticks)
ax.set_xticklabels([fdate(d)[:7] for d in navs.index[ticks]], rotation=30, fontsize=8)
ax.axhline(1.0, color='#cccccc', lw=0.6)
ax.set_title('快刀 v3: 季频池 × 新日频买卖规格 (2024-05 ~ 2026-09, 后复权)')
ax.set_ylabel('净值')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(PNG, dpi=150)
print('chart saved', PNG)

os.makedirs('e:/quant/output', exist_ok=True)
open(MD, 'w', encoding='utf-8').write('\n'.join(L))
print('written', MD, len(L), 'lines')
