# -*- coding: utf-8 -*-
"""选股通道 Stage-2: N3_state_cash 的 A-4 判定 + 全变体暴露分解 (2026-09-11)

Stage-1 发现 N3_state_cash 三项指标全面优于基线 (+47.50% / Sharpe 0.87 / MDD −24.9%)。
但它通过"剔除 below-MA120 后资金持现金"**同时降低了 gross 暴露** —— 必须先排除
"MDD/Sharpe 改善 = 暴露算术" 的可能, 再做随机剔除安慰剂。

层 1: 每变体 平均 gross 暴露 + **同暴露 scaledBH**(该变体的 gross 路径 × 全池等权收益)
层 2: N3_state_cash 的随机剔除安慰剂 (保持每期剔除只数不变, 随机挑标的) 400 路径
       与随机权重置换安慰剂 400 路径
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import kuaidao_selection_channel_test as S  # noqa: E402

OUT = S.OUT
OUTS = f'{OUT}/selection'
N_PLACEBO = 400


def sim_ex(plan):
    """返回 nav 序列 / gross 序列 / 同暴露 scaledBH 序列 / 分期净值。"""
    ical = S.CALIDX
    idxs = [ical[p['eff']] for p in plan]
    nav = sb = 1.0
    navs_all, sb_all, gross_all = [], [], []
    per = []
    for k, p in enumerate(plan):
        i0 = idxs[k]
        i1 = idxs[k + 1] if k + 1 < len(plan) else S.NCAL
        cols = np.array([S.CIDX[c] for c in p['sel']])
        n = len(cols)
        Rb = S.MK['R'][i0:i1, cols].copy()
        Ab = S.MK['A'][i0:i1, cols]
        Bb = np.maximum.accumulate(Ab, axis=0)
        w = np.array([p['w'][c] for c in p['sel']], dtype=float)
        wnom = np.where(w > 0, w, 1.0 / max(n, 1))
        for j, c in enumerate(p['sel']):
            if p['entry'][c] == p['eff']:
                Rb[0, j] = 0.0
        Rz = np.nan_to_num(Rb, nan=0.0)
        den = (Bb @ w) if p['renorm'] else (Bb @ wnom)
        num = (Rz * Ab) @ w
        r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        denw = Bb @ wnom
        gross = np.divide(Bb @ w, denw, out=np.ones_like(num), where=denw > 1e-12)
        bden = Bb.sum(axis=1)
        rbh = np.divide((Rz * Ab).sum(axis=1), bden, out=np.zeros_like(bden), where=bden > 1e-12)
        navs = nav * np.cumprod(1 + r)
        sbs = sb * np.cumprod(1 + gross * rbh)
        nav, sb = float(navs[-1]), float(sbs[-1])
        navs_all.append(navs); sb_all.append(sbs); gross_all.append(gross)
        per.append(dict(eff=p['eff'], tag=p['tag'], nav_end=nav, sbh_end=sb))
    nav_ser = pd.Series(np.concatenate(navs_all), index=S.CAL)
    sb_ser = pd.Series(np.concatenate(sb_all), index=S.CAL)
    return nav_ser, sb_ser, np.concatenate(gross_all), per


def mdd(x):
    return float((x / x.cummax() - 1).min())


def main():
    S.MK = S.build_market()
    eff_map = {}
    for rd, _ in S.QUARTERLY:
        later = [d for d in S.CAL if d > rd]
        eff_map[rd] = min(later)

    specs = [
        ('B_eq_bh', dict(weights_mode='equal')),
        ('N3_state', dict(weights_mode='strength', basis='state')),
        ('N3_state_nocap', dict(weights_mode='strength', basis='state', cap=None)),
        ('N3_state_cash', dict(weights_mode='strength', basis='state', renorm=False)),
        ('N3_event', dict(weights_mode='strength', basis='event')),
        ('N3_filter_only', dict(weights_mode='strength', basis='filter')),
        ('N5_nd3', dict(weights_mode='equal', n_drop=3)),
        ('N5_nd5', dict(weights_mode='equal', n_drop=5)),
        ('N5_nd8', dict(weights_mode='equal', n_drop=8)),
        ('N3N5_state_nd5', dict(weights_mode='strength', basis='state', n_drop=5)),
    ]

    print('=== 层 1: 暴露分解 (实际 vs 同暴露 scaledBH) ===')
    print(f'{"变体":18s} {"总收益":>9s} {"Sharpe":>7s} {"MDD":>8s} {"平均gross":>9s} '
          f'{"scaledBH":>9s} {"Δtotal":>8s} {"scaledBH_MDD":>12s} {"Δmdd":>8s} {"Δsharpe":>8s}')
    base = {}
    out = {}
    for name, kw in specs:
        plan = S.make_plan(eff_map, **kw)
        nav, sbh, gross, per = sim_ex(plan)
        p = S.kv.perf(nav)
        ps = S.kv.perf(sbh)
        out[name] = dict(perf=p, sbh=ps, gross=float(np.mean(gross)),
                         gross_min=float(np.min(gross)), gross_max=float(np.max(gross)),
                         delta_total=float(p['total'] - ps['total']),
                         delta_mdd=float(p['mdd'] - ps['mdd']),
                         delta_sharpe=float(p['sharpe'] - ps['sharpe']))
        print(f'{name:18s} {p["total"]:>+9.2%} {p["sharpe"]:>7.2f} {p["mdd"]:>8.1%} '
              f'{np.mean(gross):>9.1%} {ps["total"]:>+9.2%} '
              f'{(p["total"]-ps["total"])*100:>+7.2f}p {ps["mdd"]:>12.1%} '
              f'{(p["mdd"]-ps["mdd"])*100:>+7.2f}p {p["sharpe"]-ps["sharpe"]:>+8.2f}')

    print('\n=== 层 2: N3_state_cash 安慰剂 (400 路径/种) ===')
    kw_cash = dict(dict(specs)['N3_state_cash'])
    real = out['N3_state_cash']
    pl_out = {}
    for mode in ('drop', 'weight'):
        tots, mdds, shs, gross = [], [], [], []
        for seed in range(N_PLACEBO):
            rng = np.random.default_rng(7000 + seed)
            pl = S.make_plan(eff_map, rng=rng, placebo=mode, **kw_cash)
            nav, sbh, g, _ = sim_ex(pl)
            pp = S.kv.perf(nav)
            tots.append(pp['total']); mdds.append(pp['mdd']); shs.append(pp['sharpe'])
            gross.append(float(np.mean(g)))
        tots, mdds, shs = np.array(tots), np.array(mdds), np.array(shs)
        q = lambda a, x: float((a < x).mean() * 100)
        pl_out[mode] = dict(
            p5=float(np.percentile(tots, 5)), p50=float(np.percentile(tots, 50)),
            p95=float(np.percentile(tots, 95)), pctile=q(tots, real['perf']['total']),
            z=float((real['perf']['total'] - tots.mean()) / tots.std(ddof=1)),
            mdd_p50=float(np.percentile(mdds, 50)),
            mdd_pctile_better=float((mdds > real['perf']['mdd']).mean() * 100),
            sharpe_p50=float(np.percentile(shs, 50)), sharpe_pctile=q(shs, real['perf']['sharpe']),
            gross_mean=float(np.mean(gross)))
        o = pl_out[mode]
        print(f'[cash|{mode:6s}] real={real["perf"]["total"]:+.2%} '
              f'| placebo p5={o["p5"]:+.2%} p50={o["p50"]:+.2%} p95={o["p95"]:+.2%} '
              f'| 分位={o["pctile"]:5.1f}% z={o["z"]:+.2f} | '
              f'MDD real={real["perf"]["mdd"]:.1%} p50={o["mdd_p50"]:.1%} 优于分位={o["mdd_pctile_better"]:.0f}% '
              f'| Sharpe real={real["perf"]["sharpe"]:.2f} p50={o["sharpe_p50"]:.2f} 分位={o["sharpe_pctile"]:.0f}% '
              f'| gross={o["gross_mean"]:.1%}')

    json.dump(dict(layer1=out, layer2_cash=pl_out), open(f'{OUTS}/stats_sel_stage2.json', 'w',
              encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] selection/stats_sel_stage2.json')


if __name__ == '__main__':
    main()
