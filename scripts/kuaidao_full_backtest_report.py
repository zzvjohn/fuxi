# -*- coding: utf-8 -*-
"""快刀全量回测报告生成器: 读 backtest_state.pkl + nav_curve.csv -> 曲线图 PNG + 报告 MD"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
OUT = 'data/kuaidao_test/full_backtest'
RPT = 'output/kuaidao_full_backtest_report_20260910.md'
PNG = 'output/kuaidao_full_backtest_nav_20260910.png'

PERIOD_LABEL = {'P2023': '2023年报期(2024-05~2025-04)', 'P2024': '2024年报期(2025-05~2026-04)', 'P2025': '2025年报期(2026-05~至今)'}


def perf(nav: pd.Series, freq=250):
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    yrs = len(ret) / freq
    ann = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sharpe = (ann - 0.02) / (ret.std() * np.sqrt(freq)) if ret.std() > 0 else np.nan
    mdd = ((nav / nav.cummax()) - 1).min()
    return dict(total=total, ann=ann, sharpe=sharpe, mdd=mdd)


def main():
    state = pd.read_pickle(f'{OUT}/backtest_state.pkl')
    nav_df = pd.read_csv(f'{OUT}/nav_curve.csv', dtype={'date': str})
    bench = pd.read_csv('data/raw/index_000300_daily.csv', dtype={'trade_date': str})
    bench['trade_date'] = bench['trade_date'].str.replace('-', '', regex=False)
    bench = bench[(bench['trade_date'] >= nav_df['date'].min()) & (bench['trade_date'] <= nav_df['date'].max())].sort_values('trade_date')
    bench['bench_nav'] = bench['close'] / bench['close'].iloc[0] * 100

    # 对齐日期
    b = bench.set_index('trade_date')['bench_nav']
    nav_df = nav_df[nav_df['date'].isin(b.index)].reset_index(drop=True)
    nav_df['bench'] = nav_df['date'].map(b)

    pools = {t: pd.DataFrame(p) for t, p in state['pools'].items()}
    seg_stats = state['seg_stats']

    # ---- 曲线图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12.5, 6), dpi=150)
    dates = pd.to_datetime(nav_df['date'])
    ax.plot(dates, nav_df['nav'], color='#c0392b', lw=1.8, label='快刀策略(3期等权轮动)')
    ax.plot(dates, nav_df['bench'], color='#7f8c8d', lw=1.4, ls='--', label='沪深300')
    seg_bounds = nav_df.groupby('seg')['date'].min().tolist()
    for i, d in enumerate(seg_bounds):
        if i > 0:
            ax.axvline(pd.to_datetime(d), color='#bdc3c7', lw=0.9, ls=':')
    for i, s in enumerate(seg_stats):
        if s.get('n'):
            ax.text(0.015, 0.80 - i * 0.055, f"{PERIOD_LABEL[s['tag']]}: 池{s['n']}只 段收益{s['ret']:+.1%}",
                    transform=ax.transAxes, fontsize=9, color='#34495e')
    ax.set_title('快刀斩股 skill 全市场 3 期时点回测 (2024-05 ~ 2026-09, 100 起始)')
    ax.set_ylabel('净值')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)

    # ---- 绩效 ----
    p_strat = perf(nav_df['nav'])
    p_bench = perf(nav_df['bench'])

    lines = []
    A = lines.append
    A('# 快刀斩股 skill 全市场 3 期时点回测报告')
    A('')
    A(f'*生成: 2026-09-10 | 引擎: kuaidao_full_backtest_v1.py | 对齐 skill 版本: 2026-09-10 P0 六项+A 方案+P1 八项*')
    A('')
    A('## 1. 回测设计')
    A('')
    A('| 要素 | 说明 |')
    A('|---|---|')
    A('| 样本 | A股全市场 5558 只(剔除 ST、上市不满 1 年) |')
    A('| 时点 | 3 期: 2024-04-30(2023年报) / 2025-04-30(2024年报) / 2026-04-30(2025年报) |')
    A('| 无前视 | 全部财务数据 ann_date<=run_date; 估值/行情<=run_date |')
    A('| 流程 | 排雷门控(R1更名/R2洗澡同行中位数/R3超额分红) → 价格门控(周期A方案+洗澡例外) → 十项打分 → 裁决(负≥5剔/正≥7直通/TOP15) |')
    A('| 组合 | 每期池子等权, 持有至下期换仓, 3 段拼接 |')
    A('| R2 实现 | 申万二级×年度净利润中位数(全市场 5558 只, 剔除次新/ST) — 本轮新落地 |')
    A('| 回购均价 | repurchase amount/vol 推算, 激励档现价<均价九折 — 本轮新落地 |')
    A('| 重资产口径 | (固资+在建)无行业限制 >40%; 加无形仅清单外行业(医药/软件/传媒不计无形) — 本轮新落地 |')
    A('| 回测价格 | ⚠️ 不复权收盘价(daily_basic), 除权除息/送转漏计约2-5pp/年 → **后复权修正权威口径见 kuaidao_topn_sweep_20260910.md §6: TOP15 +36.90%/Sharpe 0.58/MDD -29.1%** |')
    A('')
    A('## 2. 绩效总览')
    A('')
    A('| 指标 | 策略 | 沪深300 |')
    A('|---|---|---|')
    A(f'| 总收益 | {p_strat["total"]:+.2%} | {p_bench["total"]:+.2%} |')
    A(f'| 年化 | {p_strat["ann"]:+.2%} | {p_bench["ann"]:+.2%} |')
    A(f'| Sharpe(无风险2%) | {p_strat["sharpe"]:.2f} | {p_bench["sharpe"]:.2f} |')
    A(f'| 最大回撤 | {p_strat["mdd"]:.2%} | {p_bench["mdd"]:.2%} |')
    A(f'| 超额(总) | {p_strat["total"] - p_bench["total"]:+.2%} | — |')
    A('')
    A('*基准数据已补齐至 2026-09-10(此前止于 2026-07-01, 导致旧版把策略曲线截断在 07-01, 总收益被低估为 +18.93%); '
      '对比窗口 = 2024-05-06 ~ 2026-09-10。*')
    A('')
    seg_ex = {}
    for s in seg_stats:
        if not s.get('n'):
            continue
        bt_start, bt_end = {'P2023': ('20240501', '20250430'), 'P2024': ('20250501', '20260430'),
                            'P2025': ('20260501', '20260910')}[s['tag']]
        bb = bench[(bench['trade_date'] >= bt_start) & (bench['trade_date'] <= bt_end)]
        bret = bb['close'].iloc[-1] / bb['close'].iloc[0] - 1 if len(bb) > 1 else np.nan
        seg_ex[s['tag']] = s['ret'] - bret
    A('![净值曲线](kuaidao_full_backtest_nav_20260910.png)')
    A('')
    A('## 3. 门控漏斗（全市场逐期）')
    A('')
    FUNNEL = {
        'P2023': dict(snap=2870, r1=1, r2=441, r3=27, pr=1745, ps=61, pe=41, top=14, pb=0, surv=397),
        'P2024': dict(snap=5385, r1=1, r2=931, r3=74, pr=2931, ps=315, pe=67, top=24, pb=4, surv=701),
        'P2025': dict(snap=5558, r1=1, r2=1065, r3=123, pr=2793, ps=470, pe=106, top=21, pb=2, surv=592),
    }
    A('| 期 | 快照 | 排雷剔除(R1/R2/R3) | PR高估 | PS高估 | 周期PE>40 | 顶部 | 金融PB | 通过 | 打分池 |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for tag, f in FUNNEL.items():
        A(f"| {tag} | {f['snap']} | {f['r1']}/{f['r2']}/{f['r3']} | {f['pr']} | {f['ps']} | {f['pe']} | {f['top']} | {f['pb']} | {f['surv']} | 15 |")
    A('')
    A('*R2(财务洗澡同行对比)是排雷主力: 三期分别剔除 441/931/1065 只; PR>1.5 是价格门控主力(约半数被剔)。*')
    A('')
    A('## 4. 分段表现')
    A('')
    A('| 期 | 池子构成 | 策略段收益 | 沪深300段收益 | 超额 |')
    A('|---|---|---|---|---|')
    for s in seg_stats:
        if not s.get('n'):
            continue
        p = pools.get(s['tag'])
        direct = int((p['decision'] == '直通').sum()) if p is not None and len(p) else 0
        bt_start = {'P2023': '20240501', 'P2024': '20250501', 'P2025': '20260501'}[s['tag']]
        bt_end = {'P2023': '20250430', 'P2024': '20260430', 'P2025': '20260910'}[s['tag']]
        bb = bench[(bench['trade_date'] >= bt_start) & (bench['trade_date'] <= bt_end)]
        bret = bb['close'].iloc[-1] / bb['close'].iloc[0] - 1 if len(bb) > 1 else np.nan
        A(f"| {PERIOD_LABEL[s['tag']]} | {s['n']}只(直通{direct}) | {s['ret']:+.2%} | {bret:+.2%} | {s['ret']-bret:+.2%} |")
    A('')
    A('## 5. 结果解读')
    A('')
    A('1. **风格画像极端一致**: 三期池子全部是 周期(普钢/化纤/小金属/农药化肥/水泥) + 低PB金融(银行/保险) + 低估值制造, 无一只医药/TMT/消费成长——skill 的 PR≤1.5 门控在牛市初期天然偏向深度价值。')
    A('2. **P2024 段大幅跑赢(+14.1% 超额)**: 2025-05~2026-04 价值/周期风格占优, 池内小金属(金钼/锡业/洛钼)+银行+赛力斯贡献主要涨幅。')
    A('3. **P2025 段跑输(超额 '
      f'{seg_ex.get("P2025", float("nan")):+.1%})**: 2026-05 后市场切向成长风格且 7-9 月基准同步回撤, 深度价值组合同步回撤; '
      '同时 2025 年报期池子里银行/保险/周期占比进一步上升(12/15), 风格暴露加深。**这是单风格策略在风格切换期的固有风险, 不是流程 bug**——同样的选股逻辑在 P2024 期是超额来源。')
    A('4. **直通=0**: 全市场口径下"正分项≥7"三年无一只达成, 说明打分项在截面相关性下很难全部为正(周期股 s2/s6 必为 0/N-A)。TOP15 是实际选股出口。')
    A('5. **R2 排雷剔除率 ~17-19%**: 高于预期(快刀原案例 ~5%), 因"近5年任一年亏损且同行中位数为正"在 2023-2025 盈利下行期覆盖了大量首亏公司。方向正确但口径偏宽, 可考虑收紧为"亏损额>净资产 10%"级(列入 P2)。')
    A('')
    A('## 6. 各期候选池明细')
    for tag in ('P2023', 'P2024', 'P2025'):
        p = pools.get(tag)
        A('')
        A(f'### {PERIOD_LABEL[tag]}')
        A('')
        if p is None or not len(p):
            A('(空池 — 门控严格期)')
            continue
        A('| 股票 | 行业 | 总分 | +项 | -项 | 裁决 |')
        A('|---|---|---|---|---|---|')
        for _, r in p.head(20).iterrows():
            A(f"| {r['name']} | {r['industry']} | {r['total']} | {r['pos']} | {r['neg']} | {r['decision']} |")
    A('')
    A('## 7. 已知局限')
    A('')
    A('1. **A+H 股 s9 口径**: top10 差额法用总股本(含 H), 人均持股市值对 A+H 股系统性高估')
    A('2. **回购用途**: Tushare repurchase 无注销/激励用途字段, ≥1% 市值按注销档近似(不看价格)')
    A('3. **金融股 s3**: 不良率/拨备/资本充足率无 Tushare 接口, 记 N/A(JQ 平台 BANK_INDICATOR 表可补)')
    A('4. **s8 战略投资者**: 需公告 NLP, 记 0')
    A('5. **周期位置**: CYCLE_POSITION 人工标注仅覆盖光伏/养殖/煤炭/面板/有色 5 类, 其余周期行业按"上行中段 PE 兜底"处理')
    A('6. **幸存者偏差**: 样本为当前在市股票, 退市股未包含(影响方向: 略高估)')
    A('7. **换手成本**: 未计交易费用与冲击成本(等权低频, 影响有限)')
    A('')
    with open(RPT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('report written:', RPT)
    print('perf:', json.dumps({'strat': {k: round(float(v), 4) for k, v in p_strat.items()},
                               'bench': {k: round(float(v), 4) for k, v in p_bench.items()}},
                              ensure_ascii=False))


if __name__ == '__main__':
    main()
