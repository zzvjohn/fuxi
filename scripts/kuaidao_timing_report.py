# -*- coding: utf-8 -*-
"""日频择时回测报告生成 (2026-09-10): 季度刷新选股池 + 日频技术买卖信号
读取: timing/stats.json, timing/nav_variants.csv, timing/trades_C_quarter_timing.csv, pool_*.csv
输出: output/kuaidao_timing_report_20260910.md + output/kuaidao_timing_nav_20260910.png
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'data/kuaidao_test/full_backtest'
REP = 'output'
TAG = '20260910'
os.makedirs(REP, exist_ok=True)

stats = json.load(open(f'{OUT}/timing/stats.json', encoding='utf-8'))
navs = pd.read_csv(f'{OUT}/timing/nav_variants.csv', dtype={'date': str}).set_index('date')
trC = pd.read_csv(f'{OUT}/timing/trades_C_quarter_timing.csv', dtype={'code': str})

VN = {  # 变体名 -> 展示名
    'A_annual_bh': '年频池+买入持有(复算v1,后复权)',
    'B_quarter_bh': '季频池+买入持有',
    'E_annual_timing': '年频池+日频择时',
    'C_quarter_timing': '季频池+日频择时(用户规格)',
    'D_quarter_timing_stop': '季频池+择时+破位止损',
    'Cnet_quarter_timing': '季频池+择时(含交易成本)',
}
COL = {'A_annual_bh': '#444444', 'B_quarter_bh': '#1f77b4', 'E_annual_timing': '#ff7f0e',
       'C_quarter_timing': '#d62728', 'D_quarter_timing_stop': '#2ca02c', 'bench': '#999999'}


def pct(x, d=2):
    return f'{x*100:+.{d}f}%'


def name_map():
    nm = {}
    for t in ['P2023', 'P2024H1', 'P2024Q3', 'P2024', 'P2025H1', 'P2025Q3', 'P2025', 'P2026H1']:
        f = f'{OUT}/pool_{t}.csv'
        if os.path.exists(f):
            df = pd.read_csv(f, dtype={'code': str})
            nm[t] = dict(zip(df['code'], df['name']))
    return nm


NM = name_map()
D = '2024-05-06'


def fmt_date(s):
    return f'{s[:4]}-{s[4:6]}-{s[6:]}'


# ---- 净值曲线图 ----
fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=150)
x = pd.to_datetime([fmt_date(d) for d in navs.index])
for k in ['bench', 'A_annual_bh', 'B_quarter_bh', 'E_annual_timing', 'C_quarter_timing', 'D_quarter_timing_stop']:
    lw = 2.2 if k == 'C_quarter_timing' else 1.3
    ls = '--' if k == 'D_quarter_timing_stop' else '-'
    ax.plot(x, navs[k] * 100, color=COL[k], lw=lw, ls=ls,
            label=f'{VN.get(k, "沪深300")}  {pct(navs[k].iloc[-1] - 1)}')
ax.axhline(100, color='#cccccc', lw=0.6)
ax.set_title('快刀策略 v2: 季度刷新选股池 + 日频技术买卖信号 (后复权, 2024-05 ~ 2026-09)')
ax.set_ylabel('净值 (期初=100)')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f'{REP}/kuaidao_timing_nav_{TAG}.png')
plt.close(fig)

# ---- MD 报告 ----
L = []
A = L.append
bp = stats['bench']

A('# 快刀斩股 v2 回测报告: 季度刷新选股池 + 日频技术买卖信号')
A('')
A(f'> 生成: 2026-09-10 | 引擎: `scripts/kuaidao_timing_backtest_v2.py` | 价格: 后复权 OHLC | 组合口径: 池内成员等权(日度重整), 与 v1 一致')
A('')
A('> **⚠️ 基准说明**: 沪深300 为价格指数(不含股息), 策略为后复权(近似全收益), 超额含约 2-3pp/年的股息偏差。')
A('> **⚠️ 离池强制卖出**: 季度刷新时离池股在生效日前一交易日收盘被动卖出(其生效日当日行情不计), 属池子换仓的隐含卖出。')
A('')
A('## 1. 策略口径')
A('')
A('### 1.1 选股池: 按财报披露季度刷新 (8 个刷新时点)')
A('')
A('| 刷新日(披露截止) | 报告期 | 池子 |')
A('|---|---|---|')
for rd, tag in [('20240430', 'P2023'), ('20240831', 'P2024H1'), ('20241031', 'P2024Q3'), ('20250430', 'P2024'),
                ('20250831', 'P2025H1'), ('20251031', 'P2025Q3'), ('20260430', 'P2025'), ('20260831', 'P2026H1')]:
    n = len(NM.get(tag, {}))
    A(f'| {fmt_date(rd)} | {tag} | {n} 只 |')
A('')
A('选股流程与 v1 完全一致(排雷 R1/R2/R3 → 价格门控 → 十项打分 → 裁决取 TOP15), 全部数据 ann_date≤刷新日, 无前视。')
A('季频刷新时基本面(年报口径的 s1/s2/s5/s6/s7 与 R2 中位数)期内冻结, 实际刷新项为: 估值与价格门控(PR/PS分位/周期PE/金融PB)、')
A('s8 增减持回购(12月窗口滑动)、s9 股东数(中报更新)、s3/s4 资产负债(缓存含季度报表, 中报时点用 H1 表)、R3 减持窗口。')
A('')
A('### 1.2 买卖执行标准 (用户钦定, 全部日频)')
A('')
A('| 动作 | 信号定义 | 撮合方式 |')
A('|---|---|---|')
A('| 买入① 一阳穿4线 | 阳线(close>open)且 open < min(MA5,10,20,30) 且 close > max(MA5,10,20,30), 均线不含当日 | 信号日收盘市价单(MOC) |')
A('| 买入② 前低/支撑位 | 当日 low ≤ 支撑×1.015 且 close 收回支撑上方 且 close>昨收; 支撑=过去60交易日最低价 或 3日确认摆动低点(120日内最近一个) | 信号日收盘市价单(MOC) |')
A('| 卖出① 前高/压力位 | 阻力=过去60交易日最高价(不含当日), 仅当阻力在昨收上方; 当日 high≥阻力成交 | t-1收盘预挂限价单, 成交价=min(阻力, 当日open) |')
A('| 卖出② 离池换仓 | 季度刷新离池 | 生效日前一交易日收盘 |')
A('| 卖出③ 破位止损 | 仅敏感性变体D: 收盘 < 60日最低×0.97 | 信号日收盘 |')
A('')
A('**无前视清单**: 均线/前低/前高/摆动低点全部仅用 t-1 及以前数据; 卖单价(t-1收盘已知)开盘前挂单标准撮合; ')
A('买单价含 t 日收盘故 MOC 成交且建仓日不计收益; 停牌日不交易(冻结前收盘, 计 0 收益入分母, 与 v1 一致)。')
A('')
A('### 1.3 变体设计 (拆分"季度刷新"与"日频择时"两个效应)')
A('')
A('| 变体 | 池子 | 交易 | 用途 |')
A('|---|---|---|---|')
A('| A | 年频(4月末) | 买入持有 | 复算 v1 后复权基线(引擎校验: +36.99% vs 上轮 +36.90% ✓) |')
A('| B | 季频 | 每期买入持有 | 单独度量"季度刷新选股池"的贡献 |')
A('| E | 年频 | 日频择时 | 单独度量"技术择时"的贡献 |')
A('| **C** | **季频** | **日频择时** | **用户钦定完整规格** |')
A('| D | 季频 | 择时+破位止损 | 敏感性: 卖出端只有"前高"而无止损的补丁 |')
A('| Cnet | 季频 | 择时含成本 | 佣金0.03%双边+印花0.05%卖出 |')
A('')
A('## 2. 绩效总览 (2024-05-06 ~ 2026-09-10, 后复权, 零成本除 Cnet)')
A('')
A('| 变体 | 总收益 | 年化 | Sharpe | 最大回撤 | 仓位暴露 | 交易次数 |')
A('|---|---|---|---|---|---|---|')
for k in ['A_annual_bh', 'B_quarter_bh', 'E_annual_timing', 'C_quarter_timing', 'D_quarter_timing_stop', 'Cnet_quarter_timing']:
    s = stats[k]
    rt = s.get('rt') or {}
    A(f'| {VN[k]} | {pct(s["perf"]["total"])} | {pct(s["perf"]["cagr"])} | {s["perf"]["sharpe"]:.2f} | '
      f'{s["perf"]["mdd"]*100:.1f}% | {s["exposure"]*100:.0f}% | {rt.get("n_buy",0)+rt.get("n_sell",0)} |')
A(f'| 沪深300(价格指数) | {pct(bp["total"])} | {pct(bp["cagr"])} | {bp["sharpe"]:.2f} | {bp["mdd"]*100:.1f}% | 100% | — |')
A('')
A(f'![净值曲线](kuaidao_timing_nav_{TAG}.png)')
A('')
A('## 3. 主角变体 C: 分段明细')
A('')
rowsC = stats['C_quarter_timing']['periods']
rowsB = stats['B_quarter_bh']['periods']
bench_s = navs['bench']
dates_idx = {d: i for i, d in enumerate(navs.index)}
A('| 生效日 | 池子 | 段收益 | 季频买入持有B | 沪深300 | C超额 |')
A('|---|---|---|---|---|---|')
for i, r in enumerate(rowsC):
    prev_nav = rowsC[i - 1]['nav_end'] if i else 1.0
    seg = r['nav_end'] / prev_nav - 1
    segb = rowsB[i]['nav_end'] / (rowsB[i - 1]['nav_end'] if i else 1.0) - 1
    i0 = dates_idx[r['eff']]
    i1 = dates_idx[rowsC[i + 1]['eff']] - 1 if i + 1 < len(rowsC) else len(navs.index) - 1
    bseg = bench_s.iloc[i1] / bench_s.iloc[i0 - 1] - 1 if i0 > 0 else bench_s.iloc[i1] - 1
    A(f'| {fmt_date(r["eff"])} | {r["tag"]} | {pct(seg)} | {pct(segb)} | {pct(bseg)} | {pct(seg - bseg)} |')
A('')
A('## 4. 池子季度演化 (季频刷新改了什么)')
A('')
A('| 换仓 | 留任 | 新进 | 离池 | 新进股票 | 离池股票 |')
A('|---|---|---|---|---|---|')
sched = [('20240430', 'P2023'), ('20240831', 'P2024H1'), ('20241031', 'P2024Q3'), ('20250430', 'P2024'),
         ('20250831', 'P2025H1'), ('20251031', 'P2025Q3'), ('20260430', 'P2025'), ('20260831', 'P2026H1')]
for i in range(1, len(sched)):
    _, t0 = sched[i - 1]
    rd, t1 = sched[i]
    s0, s1 = set(NM.get(t0, {})), set(NM.get(t1, {}))
    stay, inn, out = s0 & s1, s1 - s0, s0 - s1
    A(f'| {fmt_date(rd)} {t1} | {len(stay)} | {len(inn)} | {len(out)} | {", ".join(NM.get(t1,{}).get(c,c) for c in sorted(inn)) or "—"} | '
      f'{", ".join(NM.get(t0,{}).get(c,c) for c in sorted(out)) or "—"} |')
A('')
A('## 5. 交易统计 (变体 C)')
A('')
rt = stats['C_quarter_timing']['rt']
reason_b = trC[trC['side'] == 'BUY']['reason'].value_counts().to_dict()
reason_s = trC[trC['side'] == 'SELL']['reason'].value_counts().to_dict()
A(f'- 买入 {rt.get("n_buy", 0)} 笔: ' + ', '.join(f'{k} {v}' for k, v in reason_b.items()))
A(f'- 卖出 {rt.get("n_sell", 0)} 笔: ' + ', '.join(f'{k} {v}' for k, v in reason_s.items()) + '；另有离池强制卖出(计入池子换仓)')
if rt.get('win') is not None:
    A(f'- 完成回合 {rt.get("n_rt")} 个，胜率 {rt["win"]*100:.0f}%，平均持有 {rt["avg_hold"]:.0f} 个交易日')
A(f'- 信号频率: 一阳穿4线仅 {stats["C_quarter_timing"]["sig_a"]} 次/2.2年(严格口径属正常偏低), 支撑回踩买是绝对主力')
A(f'- 期末未平仓 {stats["C_quarter_timing"]["n_open"]} 只: ' +
  ', '.join(NM['P2026H1'].get(c, c) for c in stats['C_quarter_timing']['open_pos']))
A('')
A('## 6. 结论与解读')
A('')
pa, pb_, pe, pc, pd_ = (stats[k]['perf'] for k in ['A_annual_bh', 'B_quarter_bh', 'E_annual_timing', 'C_quarter_timing', 'D_quarter_timing_stop'])
A(f'1. **季度刷新选股池: 小幅正贡献。** B({pct(pb_["total"])}, Sharpe {pb_["sharpe"]:.2f}, MDD {pb_["mdd"]*100:.1f}%) vs A({pct(pa["total"])}, {pa["sharpe"]:.2f}, {pa["mdd"]*100:.1f}%) — '
  f'季频调仓 +{abs(pb_["total"]-pa["total"])*100:.1f}pp 且回撤更浅。8 期池子每次换 4~11 只(见 §4), 季报更新估值门控/s8/s9 确实在滚动纠错。')
A(f'2. **日频技术择时: 大幅负贡献(-30pp)。** 同为年频池, E({pct(pe["total"])}) vs A({pct(pa["total"])}); 同为季频池, C({pct(pc["total"])}) vs B({pct(pb_["total"])})。')
A('   - 机制: 分段看, 择时在下跌段确实缓冲(P2023窗 -6.1% vs 买入持有 -11.5%; P2025窗 -5.1% vs -9.5%), 但在主升段系统性踏空(P2024窗 +7.9% vs +31.7%)。')
A('   - 卖出端"前高卖"在震荡上行里反复把趋势票卖飞(卖在旧高点, 等不到回踩支撑就再无买点); 买入端"支撑买"逆势接刀, 85% 胜率但盈利被 15% 的破位持有(无止损时拿到换仓日)侵蚀。')
A('   - 暴露仅 72%: 缺的 28% 恰好是主升段。高胜率(85%)与低收益并存 = 小赚大亏的典型形态。')
A(f'3. **用户完整规格 C 总评**: {pct(pc["total"])}, Sharpe {pc["sharpe"]:.2f}, MDD {pc["mdd"]*100:.1f}% — 回撤比买入持有浅 10pp, 但总收益落后季频买入持有 30pp、落后基准 15pp。')
A(f'4. **破位止损(D)无增量**: {pct(pd_["total"])} vs C {pct(pc["total"])} — 止损后反弹缺席, 反而多亏 2.2pp; 说明"支撑买"的亏损票止损点卖在最低区域附近。')
A(f'5. **交易成本影响小**: Cnet {pct(stats["Cnet_quarter_timing"]["perf"]["total"])} vs C, 成本拖累 {abs(stats["Cnet_quarter_timing"]["perf"]["total"]-pc["total"])*100:.1f}pp(253 笔/2.2年)。')
A('')
A(f'**总判断**: 该择时规则组(支撑买/前高卖)与快刀池的盈利来源错配——池子收益来自少数票的中期趋势整段持有, 而"前高卖"截断右尾、"支撑买"放大左尾。')
A(f'若坚持日频执行, 更兼容的方向是: 卖出端保留"前高卖"但允许"突破前高后回踩不破转支撑"继续持有(即压力位与支撑位互换), 买入端加"一阳穿4线"权重(它 11 次触发的质量优于支撑回踩)。')
A('')
A('## 7. 已知局限')
A('')
A('- 池内成员等权、日度重整(与 v1 相同口径), 未模拟资金约束下的部分建仓; 离池强制卖出的交易成本未计入 Cnet。')
A('- 卖出只有"前高/压力"与(敏感性)破位, 无移动止盈; 买入在支撑回踩时本质是逆势接刀, 依赖池子基本面过滤兜底。')
A('- 一阳穿4线为严格口径(实体完整穿越四线), 触发频率低属正常; 行业分类/周期位置标注为当前时点知识(继承 v1)。')
A('- 沪深300 为价格指数; 未计融资融券与分红再投资时点差异。')

with open(f'{REP}/kuaidao_timing_report_{TAG}.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('[saved]', f'{REP}/kuaidao_timing_report_{TAG}.md')
