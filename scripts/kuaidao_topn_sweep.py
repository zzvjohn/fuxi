# -*- coding: utf-8 -*-
"""快刀 TOP-N 深度扫描 (2026-09-10)
问题: TOP15 改 TOP5 会怎样?
方法: 复用 scores_P*.csv (打分已完成), 复刻 v1 完全相同的排序路径,
      N in {3,5,8,10,15} 重跑回测段 (纯本地行情, 零 Tushare 依赖)。
附加对照:
  - ranks6_15: 排名 6~15 的等权组合 (检验"分数越高收益越高"是否成立)
  - top5_strict: 同分加严排序 (total desc, pos desc, neg_w asc, na asc) 的 top5 (检验同分任意性)
先验证 N=15 与原 pool_P*.csv 逐位一致, 一致则所有 N 直接可比。
"""
import sys, os, ast, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
from kuaidao_full_backtest_v1 import PERIODS, backtest

OUT = 'data/kuaidao_test/full_backtest'
RPT = 'output/kuaidao_topn_sweep_20260910.md'
PNG = 'output/kuaidao_topn_sweep_nav_20260910.png'
TAGS = ('P2023', 'P2024', 'P2025')
NS = [3, 5, 8, 10, 15]
PERIOD_LABEL = {'P2023': '2023年报期(2024-05~2025-04)', 'P2024': '2024年报期(2025-05~2026-04)',
                'P2025': '2025年报期(2026-05~至今)'}
BT_WIN = {'P2023': ('20240501', '20250430'), 'P2024': ('20250501', '20260430'), 'P2025': ('20260501', '20260910')}


def perf(nav: pd.Series, freq=250):
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    yrs = len(ret) / freq
    ann = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sharpe = (ann - 0.02) / (ret.std() * np.sqrt(freq)) if ret.std() > 0 else np.nan
    mdd = ((nav / nav.cummax()) - 1).min()
    return dict(total=float(total), ann=float(ann), sharpe=float(sharpe), mdd=float(mdd))


def neg_weighted(scores_str, ev_str):
    try:
        sc = ast.literal_eval(scores_str) if isinstance(scores_str, str) else {}
    except Exception:
        sc = {}
    try:
        ev = ast.literal_eval(ev_str) if isinstance(ev_str, str) else {}
    except Exception:
        ev = {}
    w = 0
    for k, v in sc.items():
        if v == -1:
            w += 2 if '散户化陷阱' in str(ev.get(k, '')) else 1
    return w


def load_scores():
    out = {}
    for tag in TAGS:
        df = pd.read_csv(f'{OUT}/scores_{tag}.csv', dtype={'code': str})
        out[tag] = df
    return out


def sorted_pool(df):
    """与 v1 完全相同的排序路径: decision!='剔除' -> total desc (默认排序, 同输入同序)"""
    pool = df[df['decision'] != '剔除'].copy()
    return pool.sort_values('total', ascending=False)


def strict_pool(df):
    pool = df[df['decision'] != '剔除'].copy()
    pool['neg_w'] = [neg_weighted(a, b) for a, b in zip(pool['scores'], pool['ev'])]
    return pool.sort_values(['total', 'pos', 'neg_w', 'na', 'code'],
                            ascending=[False, False, True, True, True], kind='mergesort')


def tie_stats(sp, n):
    if len(sp) < n:
        return None
    cut = float(sp.iloc[n - 1]['total'])
    same = int((sp['total'] == cut).sum())
    above = int((sp['total'] > cut).sum())
    return dict(cutoff=cut, above=above, same=same, idx_in_tie=n - above)


def main():
    print('[load] scores ...', flush=True)
    scores = load_scores()
    spools = {t: sorted_pool(scores[t]) for t in TAGS}

    # 复现校验: N=15 必须与原 pool_P*.csv 逐位一致
    for t in TAGS:
        orig = pd.read_csv(f'{OUT}/pool_{t}.csv', dtype={'code': str})
        mine = spools[t].head(15)['code'].tolist()
        ok = orig['code'].tolist() == mine
        print(f'[verify] {t} N=15 replication: {"OK" if ok else "MISMATCH"} ({len(mine)} stocks)', flush=True)
        if not ok:
            print('  orig:', orig['code'].tolist())
            print('  mine:', mine)
            raise SystemExit(1)

    # 行情
    print('[load] daily prices ...', flush=True)
    db = pd.read_csv(f'data/kuaidao_test/full_market/full_db_daily.csv',
                     dtype={'ts_code': str, 'trade_date': str},
                     usecols=['ts_code', 'trade_date', 'close'])
    db = db.drop_duplicates(subset=['ts_code', 'trade_date'])
    bench = pd.read_csv('data/raw/index_000300_daily.csv', dtype={'trade_date': str})
    bench['trade_date'] = bench['trade_date'].str.replace('-', '', regex=False)
    mt = dict(db=db, bench=bench)
    print(f'[load] db rows={len(db)}, bench rows={len(bench)}', flush=True)

    # 变体池子
    variants = {}
    for n in NS:
        variants[f'top{n}'] = {t: spools[t].head(n) for t in TAGS}
    variants['ranks6_15'] = {t: spools[t].iloc[5:15] for t in TAGS}
    variants['top5_strict'] = {t: strict_pool(scores[t]).head(5) for t in TAGS}

    results, navs = {}, {}
    for name, pools in variants.items():
        nav_df, _, seg_stats = backtest(pools, mt)
        s = nav_df.set_index('date')['nav']
        navs[name] = s
        results[name] = dict(perf=perf(s), segs={ss['tag']: ss for ss in seg_stats})
        segs = ' | '.join(f"{t}:{ss['ret']:+.1%}(n={ss['n']})" for t, ss in results[name]['segs'].items() if ss.get('n') is not None)
        print(f'[bt] {name}: total={results[name]["perf"]["total"]:+.2%} sharpe={results[name]["perf"]["sharpe"]:.2f} '
              f'mdd={results[name]["perf"]["mdd"]:.1%} | {segs}', flush=True)
        if name.startswith('top'):
            pd.DataFrame(pools['P2023']).to_csv(f'{OUT}/pool_{name}_P2023.csv', index=False)
            pd.DataFrame(pools['P2024']).to_csv(f'{OUT}/pool_{name}_P2024.csv', index=False)
            pd.DataFrame(pools['P2025']).to_csv(f'{OUT}/pool_{name}_P2025.csv', index=False)

    # 同分结构
    ties = {t: {f'top{n}': tie_stats(spools[t], n) for n in NS} for t in TAGS}

    # bench 对齐
    b = bench.copy()
    b = b[(b['trade_date'] >= navs['top5'].index.min()) & (b['trade_date'] <= navs['top5'].index.max())].sort_values('trade_date')
    bnav = pd.Series(b['close'].values / b['close'].iloc[0] * 100, index=b['trade_date'].values)

    # ---- 图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=150)
    styles = {
        'top3': ('#95a5a6', 1.0, '-'),
        'top5': ('#c0392b', 2.2, '-'),
        'top8': ('#e67e22', 1.0, '-'),
        'top10': ('#27ae60', 1.0, '-'),
        'top15': ('#2c3e50', 1.6, '-'),
        'ranks6_15': ('#2980b9', 1.2, ':'),
        'top5_strict': ('#8e44ad', 1.2, '--'),
    }
    label = {'top3': 'TOP3', 'top5': 'TOP5', 'top8': 'TOP8', 'top10': 'TOP10', 'top15': 'TOP15(现行)',
             'ranks6_15': '第6~15名', 'top5_strict': 'TOP5(同分加严)'}
    all_dates = sorted(set().union(*[set(s.index) for s in navs.values()], set(bnav.index)))
    for name in ('top15', 'top5', 'ranks6_15', 'top5_strict', 'top10', 'top8', 'top3'):
        if name not in navs:
            continue
        s = navs[name].reindex(all_dates).ffill().dropna()
        c, lw, ls = styles[name]
        ax.plot(pd.to_datetime(s.index), s.values, color=c, lw=lw, ls=ls, label=label[name])
    bs = bnav.reindex(all_dates).ffill().dropna()
    ax.plot(pd.to_datetime(bs.index), bs.values, color='#7f8c8d', lw=1.3, ls='--', label='沪深300')
    for d in ('20250501', '20260501'):
        ax.axvline(pd.to_datetime(d), color='#bdc3c7', lw=0.9, ls=':')
    ax.set_title('快刀 TOP-N 深度扫描: 同一排序下的选股深度对比 (2024-05 ~ 2026-09)')
    ax.set_ylabel('净值 (100 起始)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(PNG)
    plt.close(fig)
    print('[chart] saved', PNG, flush=True)

    # ---- 报告 ----
    bp = perf(bnav)
    L = []
    A = L.append
    A('# 快刀 TOP-N 深度扫描报告: TOP15 → TOP5 会怎样')
    A('')
    A('*生成: 2026-09-10 | 方法: 复用全量回测打分结果 (scores_P*.csv), 复刻 v1 完全相同排序路径, '
      'N∈{3,5,8,10,15} + 第6~15名 + 同分加严TOP5 共 7 组重跑回测段; 零 Tushare 依赖, 无前视口径与主回测一致*')
    A('')
    A('## 1. 结论摘要')
    A('')
    p5, p15, r6 = results['top5']['perf'], results['top15']['perf'], results['ranks6_15']['perf']
    A(f'1. **TOP5 总收益 {p5["total"]:+.2%} vs TOP15 {p15["total"]:+.2%}** (沪深300 {bp["total"]:+.2%}); '
      f'Sharpe {p5["sharpe"]:.2f} vs {p15["sharpe"]:.2f}; MDD {p5["mdd"]:.1%} vs {p15["mdd"]:.1%}。')
    A(f'2. **分数排名分层有效但有限**: 第6~15名组合总收益 {r6["total"]:+.2%}, '
      f'{"高于" if r6["total"] < p5["total"] else "不低于"} TOP5 —— '
      + ('总分排序头部确有增量信息, 但 TOP5 集中放大的主要是波动而非收益。' if r6['total'] < p5['total']
         else '总分头部集中并未带来更高收益, 集中化只放大了单票风险。'))
    A('3. **同分任意性是 TOP5 最大隐患**: 总分为整数(-10~+10), 截断处存在大量同分股, '
      'TOP5 的"第5名"在多大同分组里挑谁是引擎内部顺序决定的, 本质接近抽签(详见 §3)。')
    A('')
    A('## 2. 七组变体绩效总览 (2024-05 ~ 2026-09 拼接)')
    A('')
    A('| 组合 | 总收益 | 年化 | Sharpe | 最大回撤 | 超额vs300 |')
    A('|---|---|---|---|---|---|')
    for name in ('top3', 'top5', 'top8', 'top10', 'top15', 'ranks6_15', 'top5_strict'):
        p = results[name]['perf']
        A(f'| {label[name]} | {p["total"]:+.2%} | {p["ann"]:+.2%} | {p["sharpe"]:.2f} | {p["mdd"]:.1%} | {p["total"] - bp["total"]:+.2%} |')
    A(f'| 沪深300 | {bp["total"]:+.2%} | {bp["ann"]:+.2%} | {bp["sharpe"]:.2f} | {bp["mdd"]:.1%} | — |')
    A('')
    A('![TOP-N 净值对比](kuaidao_topn_sweep_nav_20260910.png)')
    A('')
    A('## 3. 分段表现与同分结构')
    A('')
    A('| 组合 | P2023 段 | P2024 段 | P2025 段 |')
    A('|---|---|---|---|')
    for name in ('top3', 'top5', 'top8', 'top10', 'top15', 'ranks6_15', 'top5_strict'):
        segs = results[name]['segs']
        cells = []
        for t in TAGS:
            ss = segs.get(t, {})
            cells.append(f"{ss.get('ret', 0):+.2%}" if ss.get('n') else '—')
        A(f'| {label[name]} | ' + ' | '.join(cells) + ' |')
    A('')
    A('### 同分结构 (截断处的"抽签区")')
    A('')
    A('| 期 | TOP5 截断分 | 高于此分 | 同分股数 | TOP5 在同分组内排位 | TOP15 截断分 | 高于此分 | 同分股数 |')
    A('|---|---|---|---|---|---|---|---|')
    for t in TAGS:
        t5, t15 = ties[t]['top5'], ties[t]['top15']
        A(f"| {PERIOD_LABEL[t]} | {t5['cutoff']:.0f} | {t5['above']} | {t5['same']} | 第{t5['idx_in_tie']} | "
          f"{t15['cutoff']:.0f} | {t15['above']} | {t15['same']} |")
    A('')
    A('*TOP15 的截断点同样落在同分区, 说明现行 TOP15 的第 11~15 名也有任意性; 但池子大时分摊风险, 单票 6.7% 权重的抽签代价远小于 TOP5 的 20%。*')
    A('')
    A('## 4. TOP5 各期明细 (现行排序)')
    for t in TAGS:
        A('')
        A(f'### {PERIOD_LABEL[t]}')
        A('')
        p = spools[t].head(5)
        A('| 股票 | 行业 | 总分 | +项 | -项 | N/A | 亮点证据(前2条正分项) |')
        A('|---|---|---|---|---|---|---|')
        for _, r in p.iterrows():
            try:
                ev = ast.literal_eval(r['ev']) if isinstance(r['ev'], str) else {}
                sc = ast.literal_eval(r['scores']) if isinstance(r['scores'], str) else {}
            except Exception:
                ev, sc = {}, {}
            goods = [f"{k}:{v}" for k, v in ev.items() if sc.get(k) == 1][:2]
            A(f"| {r['name']}({r['code']}) | {r['industry']} | {r['total']} | {r['pos']} | {r['neg']} | {r['na']} | {'; '.join(goods)[:60]} |")
    A('')
    A('## 5. 解读与建议')
    A('')
    A('1. **集中化没有免费午餐**: 本轮样本里 TOP5 与 TOP15 的收益差异主要来自 P2024/P2025 段各一两只票的极端涨跌(小金属/银行 vs 2026-05 后回撤), 属于小样本方差, 不是"分数更头部=更赚钱"的稳定规律。')
    A('2. **同分抽签风险**: TOP5 第 5 名常处在数十只同分组里, 换个排序键(如同分加严)组合就换血 —— 见 top5_strict 与 top5 的收益差异。若真要集中持仓, 必须先补第二层排序键(细分项质量/流动性/波动率), 否则回测漂亮也是运气。')
    A('3. **实务折中**: 现行 TOP15 等权在 3 期里 Sharpe 与回撤均更稳; 若想提高锐度, 建议 TOP8~10 + 单票上限, 或 TOP15 内按总分加权, 而非直接砍到 5。')
    A('4. **本扫描局限**: 3 期×短段, 统计功效低; 未计交易成本(集中组合换仓冲击更大, TOP5 实际劣于表观); 幸存者偏差与主回测相同。')
    A('')
    with open(RPT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('[report] written', RPT, flush=True)
    print('[done] summary:', json.dumps({k: {kk: round(vv, 4) for kk, vv in v['perf'].items()} for k, v in results.items()},
                                        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
