# -*- coding: utf-8 -*-
"""C-1 第三轮: 风险侧显著性 + 分期一致性 (2026-09-11)

前两轮结论:
  - ATR 移动止损 = 决定性有害 (-21~-32pp) => 该条必须删除;
  - 固定硬止损 (网格 8~25%) 使 Sharpe 0.69 -> 0.85~1.00、MDD -26.7% -> -12.5~-20.5%,
    但总收益在 +0~+2.5pp 区间抖动, 仅 15/16 处有尖峰 => 收益提升不可信, 风险改善疑似稳健;
  - 120日时间止损 +9.77pp 且 Sharpe 0.96, 但网格 (60/80/90/100/110/120/130/150/180)
    呈锯齿状 => 亦可能是噪声。

本轮回答三个问题:
  Q1 回撤改善是"降暴露的算术结果"还是真信号? -> 与随机暴露带的 MDD 分布对比。
  Q2 收益提升是否落在噪声带内? -> 随机带总分位 (rank)。
  Q3 分期是否一致 (报告验收标准要求 4 个上行段超额全部转正)?
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import importlib
m = importlib.import_module('kuaidao_timing_backtest_v4')
bf = importlib.import_module('kuaidao_bench_framework')

OUT5 = 'data/kuaidao_test/full_backtest/ablation'
CAL = m.load_calendar()
NP = 2000


def S(atr=None, hard=None, t3=False, ts=None, cp=False):
    return dict(atr_mult=atr, hard=hard, t3=t3, time_stop=ts, cost_protect=cp)


CANDS = [
    ('B_q_bh_baseline',      'bh',  'none', None,                        False, False, (0.0, 0.0)),
    ('ATR2_full_spec',       'bh',  'c1',   S(atr=2.0, hard=0.09, t3=True, ts=60, cp=True), False, False, (0.0, 0.0)),
    ('hard15_only',          'bh',  'c1',   S(hard=0.15),                 False, False, (0.0, 0.0)),
    ('T120_only',            'bh',  'c1',   S(ts=120),                    False, False, (0.0, 0.0)),
    ('h12_T120_t3',          'bh',  'c1',   S(hard=0.12, ts=120, t3=True), False, False, (0.0, 0.0)),
    ('h15_T120_t3',          'bh',  'c1',   S(hard=0.15, ts=120, t3=True), False, False, (0.0, 0.0)),
    ('K5_full_pkg_15bp',     'bh',  'c1',   S(hard=0.15, ts=120, t3=True), True,  True,  (0.0005, 0.0010)),
    ('K5b_pkg_15bp_h12',     'bh',  'c1',   S(hard=0.12, ts=120, t3=True), True,  True,  (0.0005, 0.0010)),
    ('sig_old_timing',       'sig', 'old',  None,                          False, False, (0.0, 0.0)),
]


def one(name, entry, rule, spec, cond, re, cost):
    res = m.run_variant(name, m.QUARTERLY, entry, rule, cond, re, cost, exit_spec=spec, verbose=False)
    s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
    p = m.perf(s)
    df = pd.DataFrame(res['nav_rows'])
    expo = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
    rt = m.roundtrip_stats(res['trades'], CAL)
    band = bf.random_exposure_band(res['member_rets'], res['n_pos'], n_paths=NP, with_mdd=True)
    sb = bf.scaled_bh_nav(res['r_bh'], res['n_pos'], res['n_track'])
    sb_s = pd.Series(sb, index=s.index)
    sb_perf = m.perf(sb_s)
    v = bf.verdict(p['total'], band, mdd=p['mdd'])
    prev, per = 1.0, []
    for pr in res['period_rows']:
        nv = float(pr['nav_end']) if pr['nav_end'] is not None else np.nan
        per.append(dict(eff=pr['eff'], tag=pr['tag'], ret=nv / prev - 1))
        prev = nv
    return dict(name=name, perf=p, exposure=expo, scaled=sb_perf, band=band, verdict=v,
                n_rt=rt['n_rt'], win=rt['win'], avg_ret=rt.get('avg_ret'),
                reasons=res['sig']['sell_reasons'], periods=per, trades=res['trades'],
                nav_rows=res['nav_rows'])


def main():
    os.makedirs(OUT5, exist_ok=True)
    out = {}
    for row in CANDS:
        r = one(*row)
        out[r['name']] = r
        p, sc, v, band = r['perf'], r['scaled'], r['verdict'], r['band']
        print(f"\n### {r['name']}", flush=True)
        print(f"  total={p['total']:+.2%} sharpe={p['sharpe']:.2f} mdd={p['mdd']:.1%} expo={r['exposure']:.0%} "
              f"rt={r['n_rt']} win={None if r['win'] is None else round(r['win'],3)}", flush=True)
        print(f"  暴露等价基准(scaledBH): total={sc['total']:+.2%} sharpe={sc['sharpe']:.2f} mdd={sc['mdd']:.1%}", flush=True)
        print(f"  随机带 total p5={band['pct']['p5']:+.1%} p50={band['pct']['p50']:+.1%} p95={band['pct']['p95']:+.1%} "
              f"| 变体分位={v['rank']:.0%} z={v['z']:+.2f} {'BEAT95' if v['beat_hi'] else ('in-band' if v['inside'] else 'BELOW-p5')}", flush=True)
        print(f"  随机带 MDD p5={band['mdd_pct']['p5']:.1%} p50={band['mdd_pct']['p50']:.1%} p95={band['mdd_pct']['p95']:.1%} "
              f"| 变体MDD={p['mdd']:.1%} z={v.get('mdd_z',float('nan')):+.2f} "
              f"{'MDD显著更浅' if v.get('mdd_better') else 'MDD不显著'}", flush=True)
        print(f"  sell_reasons={r['reasons']}", flush=True)

    # 分期一致性
    bm = {x['eff']: x['ret'] for x in out['B_q_bh_baseline']['periods']}
    print('\n===== 分期超额 (vs B_q_bh 基线) =====', flush=True)
    hdr = ['period'] + [n for n in out if n != 'B_q_bh_baseline']
    print(' | '.join(f'{h[:22]:>22s}' for h in hdr), flush=True)
    for i, pr in enumerate(out['B_q_bh_baseline']['periods']):
        cells = [f"{pr['tag']}({pr['eff'][4:6]}/{pr['eff'][6:8]})".rjust(22)]
        for n in hdr[1:]:
            e = out[n]['periods'][i]
            cells.append((f"{e['ret']:+.2%} ({e['ret']-pr['ret']:+.1%}pp)").rjust(22))
        print(' | '.join(cells), flush=True)
    # 上行段判定 (按基线池子收益正负划分)
    print('\n上行段 (=基线池子收益>0) 超额是否转正:', flush=True)
    for n in hdr[1:]:
        ups = [(out[n]['periods'][i]['ret'] - pr['ret']) for i, pr in
               enumerate(out['B_q_bh_baseline']['periods']) if pr['ret'] > 0]
        print(f"  {n:24s} n_up={len(ups)} 全部转正={all(x > 0 for x in ups)} "
              f"均值={np.mean(ups):+.2%} 明细={[round(x*100,2) for x in ups]}", flush=True)

    json.dump({k: {kk: vv for kk, vv in v.items()
                   if kk not in ('trades', 'nav_rows')} for k, v in out.items()},
              open(f'{OUT5}/ablation3.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUT5}/ablation3.json', flush=True)


if __name__ == '__main__':
    main()
