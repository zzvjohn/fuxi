"""
analyze_full_results.py — 全量评分卡 corner case 分析报告生成器
读 scorecard_full_*.json -> 生成 full_market_analysis.md
"""
import sys, json, glob, os
from pathlib import Path
from collections import Counter

OUT_DIR = Path('e:/quant/data/kuaidao_test')

def latest_scorecard():
    files = sorted(glob.glob(str(OUT_DIR / 'scorecard_full_*.json')))
    return files[-1] if files else None

def main():
    sc_path = latest_scorecard()
    if not sc_path:
        print('未找到 scorecard_full_*.json')
        return
    with open(sc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rep = data['report']
    cards = data['scorecards']
    lines = []
    A = lines.append

    A(f'# 快刀斩股全量 A 股测试分析 ({rep["run_ts"]})')
    A('')
    A(f'测试范围: **{rep["n_stocks"]} 只** (沪/深/北全市场, 含 B 股/ST/次新股, 无预过滤)')
    A('')
    A('## 1. 总体决策分布')
    A('')
    A('| 决策 | 数量 | 占比 |')
    A('|---|---|---|')
    for k, v in sorted(rep.get('decision', {}).items(), key=lambda x: -x[1]):
        A(f'| {k} | {v} | {v/rep["n_stocks"]:.1%} |')
    A('')
    A('## 2. 门控漏斗')
    A('')
    g1 = rep.get('gate1_verdict', {})
    A('### 门控1 (行业)')
    A('')
    A(f'| 判定 | 数量 |')
    A('|---|---|')
    for k, v in sorted(g1.items(), key=lambda x: -x[1]):
        A(f'| {k} | {v} |')
    A('')
    g2p = rep.get('gate2_pass', {})
    A('### 门控2 (公司) — pass=True/False/None')
    A(f'`{g2p}`')
    A('')
    A('| TOP3 项数 | 数量 |')
    A('|---|---|')
    for k, v in sorted(rep.get('gate2_top3', {}).items(), key=lambda x: -x[1]):
        A(f'| {k} | {v} |')
    A('')
    g2na = rep.get('gate2_na_reasons', {})
    if g2na:
        A('门控2 N/A 原因 Top:')
        A('')
        A('| 原因 | 数量 |')
        A('|---|---|')
        for k, v in g2na.items():
            A(f'| {k} | {v} |')
        A('')
    A('### 门控3 (价格)')
    A('')
    A(f'| 估值方法 | 数量 |')
    A('|---|---|')
    for k, v in sorted(rep.get('gate3_method', {}).items(), key=lambda x: -x[1]):
        A(f'| {k} | {v} |')
    A('')
    A(f'| 判定 | 数量 |')
    A('|---|---|')
    for k, v in sorted(rep.get('gate3_verdict', {}).items(), key=lambda x: -x[1]):
        A(f'| {k} | {v} |')
    A('')
    g3na = rep.get('gate3_na_reasons', {})
    if g3na:
        A('门控3 N/A 原因 Top:')
        A('')
        A('| 原因 | 数量 |')
        A('|---|---|')
        for k, v in g3na.items():
            A(f'| {k} | {v} |')
        A('')

    A('## 3. 打分项分布 (score: +1 / 0 / -1 / None) — 仅统计通过门控的股票')
    A('')
    sdist = rep.get('score_dist', {})
    A('| 打分项 | +1 | 0 | -1 | None(N/A) | 覆盖率 |')
    A('|---|---|---|---|---|---|')
    for k, v in sdist.items():
        tot = sum(v.values())
        non_null = sum(x for s, x in v.items() if s not in ('None', 'null'))
        A(f'| {k} | {v.get("1", 0)} | {v.get("0", 0)} | {v.get("-1", 0)} | {v.get("None", v.get("null", 0))} | {non_null/tot:.0%} |')
    A('')

    A('## 4. 稳定盈利三态 (B9 口径: 近3年每年净利>0)')
    A('')
    A(f'`{rep.get("stable_profit", {})}`')
    A('')
    A(f'周期行业占比: `{rep.get("cyclical", {})}`')
    A('')

    A('## 5. 特殊板块')
    A('')
    A(f'- 北交所/B股: {rep.get("n_bse_or_b", 0)} 只')
    A(f'- 科创板(688): {rep.get("n_star", 0)} 只')
    A(f'- 创业板(300): {rep.get("n_gem", 0)} 只')
    A('')

    A('## 6. 直通/TOP 排名')
    A('')
    A(f'直通候选池 (pos>=7): {rep.get("direct_n", 0)} 只')
    A('')
    A('| 排名 | 代码 | 名称 | 申万一级 | 总分 | + | - | 决策 |')
    A('|---|---|---|---|---|---|---|---|')
    for i, s in enumerate(rep.get('top30', []), 1):
        A(f'| {i} | {s["code"]} | {s["name"]} | {s["sw_l1"]} | {s["total"]} | {s["pos"]} | {s["neg"]} | {s["decision"]} |')
    A('')

    # ---- corner case 挖掘 ----
    A('## 7. Corner Case 清单 (全量新发现)')
    A('')
    # 7.1 s9 流通股本异常
    s9_ab = rep.get('s9_abnormal', [])
    if s9_ab:
        A(f'### 7.1 s9 流通股本≤0 异常: {len(s9_ab)} 只')
        A(f'`{s9_ab[:30]}`')
        A('')
    # 7.2 门控3 PR 调整后市值<=0 (净现金>总市值)
    adj_neg = []
    for s in cards:
        g = s.get('gating', {}).get('price', {})
        if g.get('method') == 'PR' and g.get('pass') is None and '调整为负' in str(g.get('detail', '')):
            adj_neg.append(s['code'])
    if adj_neg:
        A(f'### 7.2 PR 净现金调整后市值为负 (净现金>总市值): {len(adj_neg)} 只')
        A(f'`{adj_neg[:30]}`')
        A('')
    # 7.3 上市不满 3 年 (income_annual<3)
    new_list = [s for s in cards if s.get('is_stable_profit') is None and len(s.get('gating', {}).get('price', {}).get('detail', '')) > 0
                and '数据不足' in str(s.get('gating', {}).get('price', {}).get('detail', ''))]
    if new_list:
        A(f'### 7.3 门控3 近3年净利数据不足 (次新股): {len(new_list)} 只')
        A('')
    # 7.4 无分红
    no_div = sum(1 for s in cards if s.get('scores', {}).get('s5_dividend', {}).get('score') == -1)
    A(f'### 7.4 s5 所有记录均无分红: {no_div} 只')
    A('')
    # 7.5 无 top10 股东数据
    no_top10 = sum(1 for s in cards if s.get('scores', {}).get('s9_per_capita', {}).get('score') is None
                   and 'stk_top10_holders' in str(s.get('scores', {}).get('s9_per_capita', {}).get('evidence', '')))
    A(f'### 7.5 s9 top10 股东数据缺失: {no_top10} 只')
    A('')

    out_path = OUT_DIR / 'full_market_analysis.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'✅ {out_path}')

if __name__ == '__main__':
    main()
