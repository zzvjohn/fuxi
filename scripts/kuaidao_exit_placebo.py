# -*- coding: utf-8 -*-
"""C-1 第四轮: 出场时机的安慰剂检验 + 出场信息含量 (2026-09-11)

第三轮结论:
  - 全部候选的回撤改善 ≈ 暴露下降的算术结果 (变体 MDD vs 同暴露 scaledBH MDD 几乎相等),
    没有任何"风险 alpha";
  - h15/h12_T120_t3 总收益位于随机暴露带 p98~p99, 但我们共试了 45+ 个变体 =>
    多重检验下不可信;
  - 分期看是典型 short-gamma 画像: 弱段帮忙、强段伤害 (P2024 -8.4pp)。
  - 全部候选 **未通过** 报告写死的验收标准 (4 个上行段超额全部转正)。

本轮两个决定性检验:
  Q4 出场剔除的标的, 其后向收益是否系统性低于池子平均? (出场信息含量)
     - 若 ≈0 => 止损只是"降暴露", 无信息;
     - 若显著为负 => 止损确实在下跌前离场, 有信息。
  Q5 安慰剂: 保持 **(入场日, 持有期分布, 只数)** 不变, 只随机重排
     "哪个仓位持有多少天" => 得到"零出场信息"的收益分布。
     真实变体若落在该分布内, 则出场时机的贡献是噪声。
"""
import sys, os, json, bisect
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import importlib
m = importlib.import_module('kuaidao_timing_backtest_v4')

OUT5 = 'data/kuaidao_test/full_backtest/ablation'
CAL = m.load_calendar()
CIDX = {d: i for i, d in enumerate(CAL)}
EFFS = sorted(min([d for d in CAL if d > rd]) for rd, _ in m.QUARTERLY)
N_PLACEBO = 400


def S(atr=None, hard=None, t3=False, ts=None, cp=False):
    return dict(atr_mult=atr, hard=hard, t3=t3, time_stop=ts, cost_protect=cp)


def period_end_of(d):
    i = bisect.bisect_right(EFFS, d) - 1
    return EFFS[i + 1] if i + 1 < len(EFFS) else CAL[-1]


def fwd_analysis(rts, pool_mean_cache):
    """出场信息含量: 被卖出标的后向 20 日 / 至池末收益 vs 同期池子平均."""
    fwd20, fwdpe, bench20, benchpe = [], [], [], []
    for x in rts:
        d = m.get_px(x['code'])
        if d is None:
            continue
        sd = x['sell_date']
        i = CIDX.get(sd)
        if i is None:
            continue
        # 后向 20 日
        f20 = None
        if i + 20 < len(CAL) and CAL[i + 20] in d.index and sd in d.index:
            f20 = float(d.loc[CAL[i + 20], 'close']) / float(d.loc[sd, 'close']) - 1
        pend = period_end_of(sd)
        fpe = None
        if pend in d.index and sd in d.index and pend != sd:
            fpe = float(d.loc[pend, 'close']) / float(d.loc[sd, 'close']) - 1
        if f20 is None and fpe is None:
            continue
        if f20 is not None:
            fwd20.append(f20)
            bench20.append(pool_mean_cache[(sd, 20, x['sell_reason'])])
        if fpe is not None:
            fwdpe.append(fpe)
            benchpe.append(pool_mean_cache[(sd, 'pe', x['sell_reason'])])
    return dict(n=len(fwd20),
                fwd20_mean=float(np.mean(fwd20)) if fwd20 else None,
                bench20_mean=float(np.mean(bench20)) if bench20 else None,
                alpha20=float(np.mean(fwd20) - np.mean(bench20)) if fwd20 else None,
                fwdpe_mean=float(np.mean(fwdpe)) if fwdpe else None,
                benchpe_mean=float(np.mean(benchpe)) if benchpe else None,
                alphape=float(np.mean(fwdpe) - np.mean(benchpe)) if fwdpe else None)


def build_pool_mean(selldates_with_reason):
    """为每个 (卖出日, 期限, 理由) 预计算"同期全池等权"的对应收益."""
    cache = {}
    members_at = {}
    for rd, tag in m.QUARTERLY:
        eff = min([d for d in CAL if d > rd])
        members_at[eff] = set(m.load_pool(tag))
    effs_sorted = sorted(members_at)
    for sd, hz, reason in selldates_with_reason:
        i = CIDX.get(sd)
        if i is None:
            continue
        # 当日所属池
        j = bisect.bisect_right(effs_sorted, sd) - 1
        pool = members_at[effs_sorted[j]] if j >= 0 else set()
        vals = []
        for c in pool:
            d = m.get_px(c)
            if d is None or sd not in d.index:
                continue
            if hz == 'pe':
                pend = period_end_of(sd)
                if pend in d.index and pend != sd:
                    vals.append(float(d.loc[pend, 'close']) / float(d.loc[sd, 'close']) - 1)
            else:
                if i + hz < len(CAL) and CAL[i + hz] in d.index:
                    vals.append(float(d.loc[CAL[i + hz], 'close']) / float(d.loc[sd, 'close']) - 1)
        cache[(sd, hz, reason)] = float(np.mean(vals)) if vals else 0.0
    return cache


def main():
    os.makedirs(OUT5, exist_ok=True)
    spec = S(hard=0.15, ts=120, t3=True)
    real = m.run_variant('h15_T120_t3', m.QUARTERLY, 'bh', 'c1', False, False,
                         (0.0, 0.0), exit_spec=spec, verbose=False)
    s = pd.DataFrame(real['nav_rows']).set_index('date')['nav']
    rp = m.perf(s)
    rt = m.roundtrip_stats(real['trades'], CAL)
    rts = rt['rts']
    print(f"[real h15_T120_t3] total={rp['total']:+.2%} sharpe={rp['sharpe']:.2f} "
          f"mdd={rp['mdd']:.1%} n_rt={rt['n_rt']} win={rt['win']:.3f}", flush=True)

    # ---------- Q4 出场信息含量 ----------
    keys = [(x['sell_date'], 20, x['sell_reason']) for x in rts] + \
           [(x['sell_date'], 'pe', x['sell_reason']) for x in rts]
    cache = build_pool_mean(set(keys))
    fa = fwd_analysis(rts, cache)
    print('\n--- Q4 出场信息含量 (被卖出标的后向收益 vs 同期池子平均) ---', flush=True)
    print(f"  后向20日 : 出场标的={fa['fwd20_mean']:+.2%}  池子平均={fa['bench20_mean']:+.2%}  "
          f"alpha={fa['alpha20']:+.2%}  n={fa['n']}", flush=True)
    print(f"  至池末   : 出场标的={fa['fwdpe_mean']:+.2%}  池子平均={fa['benchpe_mean']:+.2%}  "
          f"alpha={fa['alphape']:+.2%}", flush=True)
    by_reason = {}
    for rsn in sorted({x['sell_reason'] for x in rts}):
        sub = [x for x in rts if x['sell_reason'] == rsn]
        f = fwd_analysis(sub, cache)
        by_reason[rsn] = f
        print(f"    [{rsn:14s}] n={f['n']:3d} fwd20={f['fwd20_mean']:+.2%} "
              f"pool={f['bench20_mean']:+.2%} alpha20={f['alpha20']:+.2%} "
              f"| fwd_pe={f['fwdpe_mean']:+.2%} alpha_pe={f['alphape']:+.2%}", flush=True)

    # ---------- Q5 安慰剂: 随机重排持有期 ----------
    holds = [CIDX[x['sell_date']] - CIDX[x['buy_date']] for x in rts
             if x['buy_date'] in CIDX and x['sell_date'] in CIDX]
    keyset = [(x['code'], x['buy_date']) for x in rts]
    print(f"\n--- Q5 安慰剂 (随机重排持有期, n_paths={N_PLACEBO}) ---", flush=True)
    print(f"  真实持有期分布: n={len(holds)} mean={np.mean(holds):.1f} "
          f"med={np.median(holds):.0f} min={min(holds)} max={max(holds)}", flush=True)
    rng = np.random.default_rng(7)
    totals, sharpes, mdds, expos = [], [], [], []
    hold_arr = np.array(holds)
    for k in range(N_PLACEBO):
        perm = rng.permutation(hold_arr)
        ph = {ky: int(v) for ky, v in zip(keyset, perm)}
        r = m.run_variant(f'placebo{k}', m.QUARTERLY, 'bh', 'placebo', False, False,
                          (0.0, 0.0), placebo_holds=ph, verbose=False)
        ns = pd.DataFrame(r['nav_rows']).set_index('date')['nav']
        pf = m.perf(ns)
        totals.append(pf['total'])
        sharpes.append(pf['sharpe'])
        mdds.append(pf['mdd'])
        dfp = pd.DataFrame(r['nav_rows'])
        expos.append(float((dfp['n_pos'] / dfp['n_mem'].clip(lower=1)).mean()))
    totals = np.array(totals); sharpes = np.array(sharpes); mdds = np.array(mdds)
    print(f"  安慰剂 total : p5={np.percentile(totals,5):+.2%} p50={np.percentile(totals,50):+.2%} "
          f"p95={np.percentile(totals,95):+.2%} mean={totals.mean():+.2%} sd={totals.std(ddof=1):.2%}", flush=True)
    print(f"  -> 真实 total={rp['total']:+.2%}  分位={float((totals < rp['total']).mean()):.0%}  "
          f"z={(rp['total']-totals.mean())/totals.std(ddof=1):+.2f}", flush=True)
    print(f"  安慰剂 sharpe: p5={np.percentile(sharpes,5):.2f} p50={np.percentile(sharpes,50):.2f} "
          f"p95={np.percentile(sharpes,95):.2f}  -> 真实={rp['sharpe']:.2f} "
          f"分位={float((sharpes < rp['sharpe']).mean()):.0%}", flush=True)
    print(f"  安慰剂 mdd   : p5={np.percentile(mdds,5):.1%} p50={np.percentile(mdds,50):.1%} "
          f"p95={np.percentile(mdds,95):.1%}  -> 真实={rp['mdd']:.1%} "
          f"分位(越浅越好)={float((mdds > rp['mdd']).mean()):.0%}", flush=True)
    print(f"  安慰剂 expo  : mean={np.mean(expos):.0%} (真实 61%)", flush=True)
    json.dump(dict(real=dict(perf=rp, n_rt=rt['n_rt'], win=rt['win']),
                   fwd=fa, fwd_by_reason=by_reason,
                   placebo=dict(n=N_PLACEBO,
                                total_pct={f'p{q}': float(np.percentile(totals, q)) for q in (5, 25, 50, 75, 95)},
                                total_mean=float(totals.mean()), total_std=float(totals.std(ddof=1)),
                                sharpe_pct={f'p{q}': float(np.percentile(sharpes, q)) for q in (5, 50, 95)},
                                mdd_pct={f'p{q}': float(np.percentile(mdds, q)) for q in (5, 50, 95)},
                                expo_mean=float(np.mean(expos)),
                                real_total=rp['total'], real_rank=float((totals < rp['total']).mean()),
                                real_sharpe=rp['sharpe'], real_mdd=rp['mdd'])),
              open(f'{OUT5}/ablation4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUT5}/ablation4.json', flush=True)


if __name__ == '__main__':
    main()
