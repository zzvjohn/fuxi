# -*- coding: utf-8 -*-
"""C-1 出场改造 · 第二轮网格 (2026-09-11)

第一轮消融结论: ATR 移动止损是元凶 (-21~-32pp); 固定硬止损 -15% 与 120日时间止损
是严格改进 (+6.5pp / +9.8pp, Sharpe & MDD 同时改善)。
本轮: (1) 参数稳定性扫描 —— 若是刀刃上的最优点即为噪声;
      (2) 组合与全包 (回补 / N-1 / 15bp 成本);
      (3) A-4 随机暴露带判定 + 分期一致性。
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
BASE = None


def S(atr=None, hard=None, t3=False, ts=None, cp=False):
    return dict(atr_mult=atr, hard=hard, t3=t3, time_stop=ts, cost_protect=cp)


def run(name, entry='bh', rule='c1', spec=None, cond=False, re=False, cost=(0.0, 0.0)):
    res = m.run_variant(name, m.QUARTERLY, entry, rule, cond, re, cost, exit_spec=spec, verbose=False)
    s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
    p = m.perf(s)
    df = pd.DataFrame(res['nav_rows'])
    expo = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
    rt = m.roundtrip_stats(res['trades'], CAL)
    band = bf.random_exposure_band(res['member_rets'], res['n_pos'], n_paths=2000)
    sb = float(bf.scaled_bh_nav(res['r_bh'], res['n_pos'], res['n_track'])[-1] - 1)
    v = bf.verdict(p['total'], band)
    # 分期收益
    prev, per = 1.0, []
    for pr in res['period_rows']:
        nv = float(pr['nav_end']) if pr['nav_end'] is not None else np.nan
        per.append(round(nv / prev - 1, 4))
        prev = nv
    return dict(name=name, perf=p, exposure=expo, n_rt=rt['n_rt'], win=rt['win'],
                avg_ret=rt.get('avg_ret'), n_tail50=rt.get('n_tail50'),
                reasons=res['sig']['sell_reasons'], band=band, scaled_bh=sb, verdict=v,
                periods=per, trades=res['trades'])


def show(r, base_total):
    p = r['perf']
    band = r['band']['pct']
    print(f"[{r['name']:26s}] tot={p['total']:+.2%} ({(p['total']-base_total)*100:+6.2f}pp) "
          f"sh={p['sharpe']:.2f} mdd={p['mdd']:.1%} expo={r['exposure']:.0%} "
          f"rt={r['n_rt']:3d} win={None if r['win'] is None else round(r['win'],3)} "
          f"| scalBH={r['scaled_bh']:+.2%} band[{band['p5']:+.0%},{band['p95']:+.0%}] "
          f"z={r['verdict']['z']:+.2f} {'BEAT95' if r['verdict']['beat_hi'] else ('in-band' if r['verdict']['inside'] else 'BELOW-p5')} "
          f"| {r['reasons']}", flush=True)


def main():
    global BASE
    base = run('BH_baseline', 'bh', 'none')
    BASE = base['perf']['total']
    show(base, BASE)
    print('\n--- A. 硬止损网格 (仅固定止损, 无移动) ---', flush=True)
    grid_h = {}
    for h in (0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22, 0.25):
        r = run(f'hard{h*100:.0f}', 'bh', 'c1', S(hard=h))
        grid_h[f'hard{h}'] = r
        show(r, BASE)
    print('\n--- B. 时间止损网格 ---', flush=True)
    grid_t = {}
    for ts in (60, 80, 90, 100, 110, 120, 130, 150, 180):
        r = run(f'T{ts}', 'bh', 'c1', S(ts=ts))
        grid_t[f'T{ts}'] = r
        show(r, BASE)
    print('\n--- C. 组合 (硬止损 + 时间止损 + T3) ---', flush=True)
    combos = {}
    for h, ts, t3 in ((0.15, 120, False), (0.15, 120, True), (0.15, 110, True),
                      (0.12, 120, True), (0.15, 90, True), (0.18, 120, True),
                      (0.15, 150, True), (0.20, 120, False)):
        nm = f'h{int(h*100)}_T{ts}' + ('_t3' if t3 else '')
        r = run(nm, 'bh', 'c1', S(hard=h, ts=ts, t3=t3))
        combos[nm] = r
        show(r, BASE)
    print('\n--- D. 全包 (最佳组合 + 回补 / N-1 / 成本 / 入场) ---', flush=True)
    best = S(hard=0.15, ts=120, t3=True)
    pkgs = {}
    for nm, entry, cond, re, cost in (
            ('K1_bh_h15T120t3',          'bh',  False, False, (0.0, 0.0)),
            ('K2_bh_h15T120t3_re',       'bh',  False, True,  (0.0, 0.0)),
            ('K3_bh_h15T120t3_N1',       'bh',  True,  False, (0.0, 0.0)),
            ('K4_bh_h15T120t3_N1_re',    'bh',  True,  True,  (0.0, 0.0)),
            ('K5_bh_h15T120t3_N1_re_15bp', 'bh', True, True,  (0.0005, 0.0010)),
            ('K6_sig_h15T120t3',         'sig', False, False, (0.0, 0.0)),
            ('K7_sig_h15T120t3_N1',      'sig', True,  False, (0.0, 0.0)),
    ):
        r = run(nm, entry, 'c1', best, cond, re, cost)
        pkgs[nm] = r
        show(r, BASE)
    json.dump(dict(baseline=BASE,
                   grid_hard={k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in grid_h.items()},
                   grid_time={k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in grid_t.items()},
                   combos={k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in combos.items()},
                   pkgs={k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in pkgs.items()},
                   base=base),
              open(f'{OUT5}/ablation2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUT5}/ablation2.json', flush=True)


if __name__ == '__main__':
    main()
