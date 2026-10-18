# -*- coding: utf-8 -*-
"""价格口径对照: 不复权 vs 后复权 (2026-09-10)
同一批池子(排序/选股完全不变), 仅把回测价格从 daily_basic 不复权 close 换成
close x adj_factor (后复权, 含分红送转的持有回报), 重跑 7 组变体。
输出: 控制台对照表 + 追加节到 output/kuaidao_topn_sweep_20260910.md
"""
import sys, os
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import kuaidao_topn_sweep as S
from kuaidao_full_backtest_v1 import backtest

OLD = {  # 不复权结果 (kuaidao_topn_sweep v1 已固化)
    'top3': (-0.1551, -0.3715, -0.3355), 'top5': (0.0113, -0.0685, -0.2934),
    'top8': (-0.0604, -0.2302, -0.3139), 'top10': (0.0365, -0.0198, -0.339),
    'top15': (0.2519, 0.3795, -0.307), 'ranks6_15': (0.3808, 0.5592, -0.3149),
    'top5_strict': (-0.263, -0.7147, -0.3501),
}
LABEL = {'top3': 'TOP3', 'top5': 'TOP5', 'top8': 'TOP8', 'top10': 'TOP10', 'top15': 'TOP15(现行)',
         'ranks6_15': '第6~15名', 'top5_strict': 'TOP5(同分加严)'}


def load_hfq_db():
    db = pd.read_csv('data/kuaidao_test/full_market/full_db_daily.csv',
                     dtype={'ts_code': str, 'trade_date': str}, usecols=['ts_code', 'trade_date', 'close'])
    db = db.drop_duplicates(subset=['ts_code', 'trade_date'])
    adj = pd.read_csv('data/kuaidao_test/full_market/full_adj_factor.csv', dtype={'ts_code': str, 'trade_date': str})
    print(f'[hfq] db={len(db)} adj={len(adj)}', flush=True)
    m = db.merge(adj, on=['ts_code', 'trade_date'], how='left').sort_values(['ts_code', 'trade_date'])
    m['adj_factor'] = m.groupby('ts_code')['adj_factor'].ffill()
    m['adj_factor'] = m.groupby('ts_code')['adj_factor'].bfill()
    n_missing = int(m['adj_factor'].isna().sum())
    m['adj_factor'] = m['adj_factor'].fillna(1.0)
    m['close'] = m['close'] * m['adj_factor']
    print(f'[hfq] adj missing filled as 1.0: {n_missing} rows', flush=True)
    # 复权后跳空复检
    ret = m.groupby('ts_code')['close'].pct_change()
    n25 = int((ret.abs() > 0.25).sum())
    print(f'[hfq] 复权后隔夜|ret|>25% 事件: {n25} (不复权为 2177)', flush=True)
    return m[['ts_code', 'trade_date', 'close']]


def main():
    scores = S.load_scores()
    spools = {t: S.sorted_pool(scores[t]) for t in S.TAGS}
    db = load_hfq_db()
    bench = pd.read_csv('data/raw/index_000300_daily.csv', dtype={'trade_date': str})
    bench['trade_date'] = bench['trade_date'].str.replace('-', '', regex=False)
    mt = dict(db=db, bench=bench)

    variants = {}
    for n in S.NS:
        variants[f'top{n}'] = {t: spools[t].head(n) for t in S.TAGS}
    variants['ranks6_15'] = {t: spools[t].iloc[5:15] for t in S.TAGS}
    variants['top5_strict'] = {t: S.strict_pool(scores[t]).head(5) for t in S.TAGS}

    rows = []
    for name, pools in variants.items():
        nav_df, _, segs = backtest(pools, mt)
        p = S.perf(nav_df.set_index('date')['nav'])
        o = OLD[name]
        rows.append((name, o, p, segs))
        segs_s = ' | '.join(f"{ss['tag']}:{ss['ret']:+.1%}" for ss in segs if ss.get('n') is not None)
        print(f'[hfq] {name}: total={p["total"]:+.2%} (不复权 {o[0]:+.2%}, Δ{p["total"]-o[0]:+.2%}) '
              f'sharpe={p["sharpe"]:.2f} mdd={p["mdd"]:.1%} | {segs_s}', flush=True)

    # 追加到扫描报告
    RPT = 'output/kuaidao_topn_sweep_20260910.md'
    L = ['', '## 6. 价格口径修正: 不复权 → 后复权 (2026-09-10 深夜补测)', '',
         '**发现**: 回测价格此前取自 Tushare daily_basic 的 close(不复权), 全市场 385 万行中存在 2177 次'
         '隔夜|涨跌|>25% 的除权除息/送转/退市整理跳空(6 月分红季密集出现 -50%~-96% 幻觉下跌)。'
         '持有期收益被系统性低估(分红送转是股东真实回报, 不该计为亏损)。', '',
         '**修正**: 补拉全市场 adj_factor, 后复权价 = close × adj_factor(≈含红利再投的持有回报); '
         '选股/排序/池子完全不变, 仅回测价格更换。复权后隔夜|涨跌|>25% 事件大幅收敛。', '',
         '| 组合 | 总收益(不复权) | 总收益(后复权) | 差异 | Sharpe(后复权) | MDD(后复权) |',
         '|---|---|---|---|---|---|']
    for name, o, p, _ in rows:
        label = LABEL.get(name, name)
        L.append(f'| {label} | {o[0]:+.2%} | {p["total"]:+.2%} | {p["total"]-o[0]:+.2%} | {p["sharpe"]:.2f} | {p["mdd"]:.1%} |')
    L += ['', '*口径说明: 后复权=真实持有回报(含分红送转), 应以本节为净值收益的权威口径; '
          '§2 的不复权数字保留作对照。选股端(s9 人均市值/s8 回购比例/60日涨幅)仍用不复权价, 属可接受近似。*', '']
    with open(RPT, 'a', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('[report] appended', RPT, flush=True)


if __name__ == '__main__':
    main()
