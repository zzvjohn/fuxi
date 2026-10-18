# -*- coding: utf-8 -*-
"""C-1 出场改造 · 逐规则消融实验 (2026-09-11)

目的: 上一轮全量回测显示 C-1 全套把 +39.14% 打到 +17.83%, 且胜率 59.5% -> 32.7%。
      本脚本把 C-1 拆成 5 条子规则逐条单独施加, 确认是哪一条在毁灭收益,
      并量化"被截断的右尾"(missed tail): 每笔卖出后若持有到本池期结束的反事实收益。

口径: 季频池, entry='bh' (换池日等权买入), reentry=False, 无成本, 无 N-1。
      —— 即以 +39.14% 的纯 BH 为唯一对照, 每条规则单独加上去。
"""
import sys, os, json, bisect
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import importlib
m = importlib.import_module('kuaidao_timing_backtest_v4')
bf = importlib.import_module('kuaidao_bench_framework')

OUT = 'data/kuaidao_test/full_backtest'
OUT5 = f'{OUT}/ablation'
CAL = m.load_calendar()
EFFS = sorted(min([d for d in CAL if d > rd]) for rd, _ in m.QUARTERLY)
CIDX = {d: i for i, d in enumerate(CAL)}


def S(atr=None, hard=None, t3=False, ts=None, cp=False):
    return dict(atr_mult=atr, hard=hard, t3=t3, time_stop=ts, cost_protect=cp)


ABLATION = [
    ('BH_baseline',          'none', None),
    ('hard9',                'c1', S(hard=0.09)),
    ('hard15',               'c1', S(hard=0.15)),
    ('hard20',               'c1', S(hard=0.20)),
    ('hard25',               'c1', S(hard=0.25)),
    ('atr2',                 'c1', S(atr=2.0)),
    ('atr3',                 'c1', S(atr=3.0)),
    ('atr4',                 'c1', S(atr=4.0)),
    ('atr6',                 'c1', S(atr=6.0)),
    ('T3_only',              'c1', S(t3=True)),
    ('T60_only',             'c1', S(ts=60)),
    ('T120_only',            'c1', S(ts=120)),
    ('atr3+T60',             'c1', S(atr=3.0, ts=60)),
    ('atr3+hard20',          'c1', S(atr=3.0, hard=0.20)),
    ('atr3+hard20+cp',       'c1', S(atr=3.0, hard=0.20, cp=True)),
    ('atr3+hard20+T120+cp',  'c1', S(atr=3.0, hard=0.20, ts=120, cp=True)),
    ('C1_full_spec',         'c1', S(atr=2.0, hard=0.09, t3=True, ts=60, cp=True)),
]


def missed_tail(rt_list):
    """右尾损失诊断: 每个回合卖出后, 若持有至本池期结束 的反事实收益."""
    rt = rt_list or []
    if not rt:
        return dict(n=0, given=0.0, fore=0.0, miss_pp=0.0, n_tail_cut=0, fore_tail=0.0)
    rows = []
    for x in rt:
        d = m.get_px(x['code'])
        if d is None:
            continue
        bd = x['buy_date']
        i = bisect.bisect_right(EFFS, bd) - 1
        pend = EFFS[i + 1] if i + 1 < len(EFFS) else CAL[-1]
        if pend not in d.index or bd not in d.index:
            continue
        held = float(d.loc[pend, 'close']) / float(x['buy_px']) - 1
        rows.append(dict(code=x['code'], actual=x['ret'], held=held, fore=held - x['ret'],
                         reason=x['sell_reason']))
    if not rows:
        return dict(n=0, given=0.0, fore=0.0, miss_pp=0.0, n_tail_cut=0, fore_tail=0.0)
    df = pd.DataFrame(rows)
    given = float(df['actual'].mean())
    fore = float(df['fore'].mean())
    tail = df[df['fore'] > 0.20]            # 卖飞 >20pp 的笔数
    return dict(n=len(df), given_mean=given, fore_mean=fore,
                given_total=float(df['actual'].sum()), fore_total=float(df['fore'].sum()),
                n_tail_cut=int(len(tail)), fore_tail_total=float(tail['fore'].sum()),
                top5_fore=float(df['fore'].nlargest(5).sum()),
                rows=rows)


def main():
    os.makedirs(OUT5, exist_ok=True)
    out, rows_txt = {}, []
    base_total = None
    for name, rule, spec in ABLATION:
        res = m.run_variant(name, m.QUARTERLY, 'bh', rule, False, False, (0.0, 0.0),
                            exit_spec=spec, verbose=False)
        s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
        p = m.perf(s)
        rt = m.roundtrip_stats(res['trades'], CAL)
        mt = missed_tail(rt.get('rts'))
        if base_total is None:
            base_total = p['total']
        out[name] = dict(spec=spec, perf=p, n_rt=rt['n_rt'], win=rt['win'],
                         avg_ret=rt.get('avg_ret'), n_tail50=rt.get('n_tail50'),
                         sell_reasons=res['sig']['sell_reasons'], missed=mt)
        reasons = res['sig']['sell_reasons'] or {}
        rows_txt.append((name, p, rt, reasons, mt))
        print(f"[{name:22s}] tot={p['total']:+.2%} ({(p['total']-base_total)*100:+6.2f}pp) "
              f"sh={p['sharpe']:.2f} mdd={p['mdd']:.1%} rt={rt['n_rt']:3d} "
              f"win={rt['win'] if rt['win'] is None else round(rt['win'],3)} "
              f"avg={rt.get('avg_ret') if rt.get('avg_ret') is None else round(rt['avg_ret'],3)} "
              f"| miss_fore={mt.get('fore_mean',0):+.1%} "
              f"tailcut={mt.get('n_tail_cut',0)} fore_tot={mt.get('fore_total',0):+.2f} "
              f"| why={reasons}", flush=True)
    # ---- 宽松组合的完整包: + 信号回补 + N-1 + 成本 ----
    print('\n===== 组合包 (宽松 C-1 + 回补 + N-1 + 15bp) =====', flush=True)
    loose = S(atr=3.0, hard=0.20, ts=120, cp=True)
    combos = [
        ('P1_bh_loose',            'bh',  loose, False, False, (0.0, 0.0)),
        ('P2_bh_loose_re',         'bh',  loose, False, True,  (0.0, 0.0)),
        ('P3_bh_loose_re_N1',      'bh',  loose, True,  True,  (0.0, 0.0)),
        ('P4_bh_loose_re_N1_net15', 'bh', loose, True,  True,  (0.0005, 0.0010)),
        ('P5_sig_loose',           'sig', loose, False, False, (0.0, 0.0)),
        ('P6_sig_loose_N1',        'sig', loose, True,  False, (0.0, 0.0)),
    ]
    pkg = {}
    for name, entry, spec, cond, re, cost in combos:
        res = m.run_variant(name, m.QUARTERLY, entry, 'c1', cond, re, cost,
                            exit_spec=spec, verbose=False)
        s = pd.DataFrame(res['nav_rows']).set_index('date')['nav']
        p = m.perf(s)
        df = pd.DataFrame(res['nav_rows'])
        expo = float((df['n_pos'] / df['n_mem'].clip(lower=1)).mean())
        band = bf.random_exposure_band(res['member_rets'], res['n_pos'], n_paths=2000)
        sb = float(bf.scaled_bh_nav(res['r_bh'], res['n_pos'], res['n_track'])[-1] - 1)
        v = bf.verdict(p['total'], band)
        rt = m.roundtrip_stats(res['trades'], CAL)
        mt = missed_tail(rt.get('rts'))
        pkg[name] = dict(perf=p, exposure=expo, band=band['pct'], scaled_bh=sb, verdict=v,
                         n_rt=rt['n_rt'], win=rt['win'], n_tail50=rt.get('n_tail50'),
                         sell_reasons=res['sig']['sell_reasons'], missed=mt)
        print(f"[{name:26s}] tot={p['total']:+.2%} ({(p['total']-base_total)*100:+6.2f}pp) "
              f"sh={p['sharpe']:.2f} mdd={p['mdd']:.1%} expo={expo:.0%} rt={rt['n_rt']} "
              f"win={None if rt['win'] is None else round(rt['win'],3)} "
              f"| scaledBH={sb:+.2%} band95={band['pct']['p95']:+.2%} z={v['z']:+.2f} "
              f"| tailcut={mt.get('n_tail_cut',0)} fore_tot={mt.get('fore_total',0):+.2f}", flush=True)
    json.dump(dict(ablation=out, packages=pkg, baseline_total=base_total),
              open(f'{OUT5}/ablation.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUT5}/ablation.json', flush=True)


if __name__ == '__main__':
    main()
