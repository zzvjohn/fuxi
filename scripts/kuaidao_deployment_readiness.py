"""快刀斩股 · 可部署性体检 (Deployment Readiness)

回答"这套选股逻辑能不能真金白银跑"，四组测试，全部挂在已自证的
`kuaidao_selection_channel_test.py` 引擎上（B_eq_bh = +39.14% / 0.69 / -26.7%）：

  T1 脆弱性 (fragility)   : 每期剔除事后收益最好的 j 只 -> 收益还剩多少
                            (对照: 剔除最差的 j 只)  -> 收益是不是靠 2~3 只票
  T2 成本敏感性           : 单边 0/15/30/50/100bp 往返成本后的净值
  T3 执行滞后             : 建仓日整体推迟 0/1/2/3/5/10 个交易日
  T4 容量粗估             : 池内标的流通市值 / 日均成交额分布 -> 规模上限

输出: data/kuaidao_test/full_backtest/selection/deploy_ready.json
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import kuaidao_selection_channel_test as ks  # noqa: E402

OUT = 'data/kuaidao_test/full_backtest'
OUTS = f'{OUT}/selection'

_ORIG_LOAD = ks.load_pool_ordered
EXCL = {}          # tag -> set(codes) 由 T1 设置


def patched_load(tag):
    pool = _ORIG_LOAD(tag)
    ex = EXCL.get(tag)
    return [c for c in pool if c not in ex] if ex else pool


ks.load_pool_ordered = patched_load


def base_eff_map():
    return {rd: min(d for d in ks.CAL if d > rd) for rd, _ in ks.QUARTERLY}


# ---------------------------------------------------------------- T1 分段收益
def segment_returns(plan):
    """每个 (tag, code) 在其持仓期内的复利收益 (与引擎同口径: 新进者首日归零)."""
    ical = ks.CALIDX
    idxs = [ical[p['eff']] for p in plan]
    out = {}
    for k, p in enumerate(plan):
        i0 = idxs[k]
        i1 = idxs[k + 1] if k + 1 < len(plan) else ks.NCAL
        for c in p['sel']:
            j = ks.CIDX[c]
            r = ks.MK['R'][i0:i1, j].copy()
            a = ks.MK['A'][i0:i1, j]
            if len(r) and p['entry'][c] == p['eff']:
                r[0] = 0.0
            r = np.where(a > 0, np.nan_to_num(r, nan=0.0), 0.0)
            out[(p['tag'], c)] = float(np.prod(1.0 + r) - 1.0)
    return out


def main():
    ks.MK = ks.build_market()
    print(f'[market] codes={ks.NC} cal={ks.NCAL} ({ks.CAL[0]}..{ks.CAL[-1]})', flush=True)
    eff_map = base_eff_map()

    # ---------------- 基线 ----------------
    plan0, nav0, perf0, per0 = ks.make_and_run(eff_map, weights_mode='equal')
    print(f'[BASE   ] total={perf0["total"]:+.2%} sharpe={perf0["sharpe"]:.2f} '
          f'mdd={perf0["mdd"]:.1%}', flush=True)

    seg = segment_returns(plan0)
    B = perf0['total']
    out = dict(base=dict(total=B, sharpe=perf0['sharpe'], mdd=perf0['mdd'],
                         periods=per0))

    # ================================================================ T1 脆弱性
    print('\n===== T1 脆弱性: 每期剔除事后最好/最差的 j 只 =====', flush=True)
    tags = [p['tag'] for p in plan0]

    def topj_codes_by_tag(j, worst=False):
        m = {}
        for t in tags:
            items = sorted(((c, r) for (tt, c), r in seg.items() if tt == t),
                           key=lambda x: x[1], reverse=not worst)
            m[t] = {c for c, _ in items[:j]}
        return m

    frag = {}
    for mode in ('best', 'worst'):
        for j in (1, 2, 3, 5):
            EXCL.clear()
            EXCL.update(topj_codes_by_tag(j, worst=(mode == 'worst')))
            _, _, pf, _ = ks.make_and_run(eff_map, weights_mode='equal')
            frag[f'drop_{mode}_{j}'] = dict(total=pf['total'], sharpe=pf['sharpe'],
                                            mdd=pf['mdd'], delta_pp=(pf['total'] - B) * 100)
            print(f'  drop_{mode}_{j:<2d} total={pf["total"]:+8.2%} '
                  f'({(pf["total"]-B)*100:+7.2f}pp) sharpe={pf["sharpe"]:.2f} '
                  f'mdd={pf["mdd"]:7.1%}', flush=True)
    EXCL.clear()
    out['fragility'] = frag

    # ---- T1b 安慰剂: 每期随机剔除 1 只 (400 路径) -> 脆弱性到底是不是"运气" ----
    print('\n----- T1b 安慰剂: 每期随机剔除 1 只 (400 路径) -----', flush=True)
    N_PB = 400
    pb_tot, pb_mdd = [], []
    for seed in range(N_PB):
        rng = np.random.default_rng(5000 + seed)
        EXCL.clear()
        for t in tags:
            pool = _ORIG_LOAD(t)
            EXCL[t] = {pool[int(k)] for k in rng.permutation(len(pool))[:1]}
        _, _, pf, _ = ks.make_and_run(eff_map, weights_mode='equal')
        pb_tot.append(pf['total']); pb_mdd.append(pf['mdd'])
    pb_tot, pb_mdd = np.array(pb_tot), np.array(pb_mdd)
    EXCL.clear()
    q = lambda a, x: float((a < x).mean() * 100)
    best1 = frag['drop_best_1']['total']
    worst1 = frag['drop_worst_1']['total']
    out['frag_placebo'] = dict(
        n_paths=N_PB,
        p5=float(np.percentile(pb_tot, 5)), p50=float(np.percentile(pb_tot, 50)),
        p95=float(np.percentile(pb_tot, 95)), mean=float(pb_tot.mean()),
        std=float(pb_tot.std(ddof=1)),
        drop_best_1=best1, drop_best_1_pctile=q(pb_tot, best1),
        drop_best_1_z=float((best1 - pb_tot.mean()) / pb_tot.std(ddof=1)),
        drop_worst_1=worst1, drop_worst_1_pctile=q(pb_tot, worst1),
        drop_worst_1_z=float((worst1 - pb_tot.mean()) / pb_tot.std(ddof=1)),
        mdd_p50=float(np.percentile(pb_mdd, 50)),
    )
    o = out['frag_placebo']
    print(f'  随机剔除 1 只/期: p5={o["p5"]:+.2%} p50={o["p50"]:+.2%} p95={o["p95"]:+.2%} '
          f'(mean={o["mean"]:+.2%} sd={o["std"]:.2%})', flush=True)
    print(f'  事后最好 1 只 -> {best1:+.2%} = 第 {o["drop_best_1_pctile"]:.1f} 分位 '
          f'z={o["drop_best_1_z"]:+.2f}', flush=True)
    print(f'  事后最差 1 只 -> {worst1:+.2%} = 第 {o["drop_worst_1_pctile"]:.1f} 分位 '
          f'z={o["drop_worst_1_z"]:+.2f}', flush=True)

    # 单只聚合: 全样本贡献 top10
    contrib = sorted(seg.items(), key=lambda kv: kv[1], reverse=True)
    out['top_contributors'] = [dict(tag=t, code=c, ret=r) for (t, c), r in contrib[:10]]
    out['bottom_contributors'] = [dict(tag=t, code=c, ret=r) for (t, c), r in contrib[-10:]]
    print('\n  [单仓位收益 TOP5] ' + ' | '.join(
        f'{t}/{c} {r:+.1%}' for (t, c), r in contrib[:5]), flush=True)

    # ================================================================ T2 成本
    print('\n===== T2 交易成本敏感性 (往返 bp) =====', flush=True)
    def apply_cost(plan, nav, rt_bp):
        if rt_bp <= 0:
            return nav
        mult, prev, idx = 1.0, set(), nav.index
        scale = pd.Series(1.0, index=idx)
        for p in plan:
            n = len(p['sel'])
            n_new = sum(1 for c in p['sel'] if c not in prev)
            mult *= (1.0 - (n_new / max(n, 1)) * rt_bp / 1e4)
            scale.loc[scale.index >= p['eff']] = mult
            prev = set(p['sel'])
        return nav * scale

    cost_out, prev_tot = {}, B
    for bp in (0, 15, 30, 50, 100, 200):
        navc = apply_cost(plan0, nav0, bp)
        pf = ks.kv.perf(navc)
        cost_out[f'{bp}bp'] = dict(total=pf['total'], sharpe=pf['sharpe'], mdd=pf['mdd'],
                                   delta_pp=(pf['total'] - B) * 100)
        print(f'  {bp:>3d}bp  total={pf["total"]:+8.2%} ({(pf["total"]-B)*100:+7.2f}pp) '
              f'sharpe={pf["sharpe"]:.2f} mdd={pf["mdd"]:7.1%}', flush=True)
    # 换手统计
    prev, turns = set(), []
    for p in plan0:
        n = len(p['sel'])
        turns.append(sum(1 for c in p['sel'] if c not in prev) / n)
        prev = set(p['sel'])
    out['cost'] = dict(curve=cost_out, turns=[float(x) for x in turns],
                       mean_turn=float(np.mean(turns)),
                       per_year_turn=float(np.mean(turns) * 3),
                       ann_cost_30bp=float(np.mean(turns) * 3 * 0.0030))
    print(f'  平均换手率/期={np.mean(turns):.1%}  年化≈{np.mean(turns)*3:.2f} 次 '
          f'-> 30bp 下年化成本 ≈ {np.mean(turns)*3*0.0030:.2%}', flush=True)

    # ================================================================ T3 执行滞后
    print('\n===== T3 建仓日滞后 L 个交易日 =====', flush=True)
    lag_out = {}
    for L in (0, 1, 2, 3, 5, 10):
        e2 = {}
        for rd, e in eff_map.items():
            i = min(ks.CALIDX[e] + L, ks.NCAL - 1)
            e2[rd] = ks.CAL[i]
        _, navl, pfl, _ = ks.make_and_run(e2, weights_mode='equal')
        d = (pfl['total'] - B) * 100
        lag_out[f'lag{L}'] = dict(total=pfl['total'], sharpe=pfl['sharpe'], mdd=pfl['mdd'],
                                  delta_pp=d)
        print(f'  lag{L:<2d}  total={pfl["total"]:+8.2%} ({d:+7.2f}pp) '
              f'sharpe={pfl["sharpe"]:.2f} mdd={pfl["mdd"]:7.1%}', flush=True)
    out['entry_lag'] = lag_out

    # ================================================================ T4 容量
    print('\n===== T4 容量粗估 (池内标的流动性与市值) =====', flush=True)
    codes = set(ks.ALL_CODES)
    frames, path = [], 'data/raw/daily_basic.csv'
    try:
        for ch in pd.read_csv(path, usecols=['ts_code', 'trade_date', 'close',
                                             'turnover_rate_f', 'free_share',
                                             'circ_mv', 'total_mv'],
                              dtype={'ts_code': str}, chunksize=2_000_000):
            frames.append(ch[ch['ts_code'].isin(codes)])
        db = pd.concat(frames, ignore_index=True)
        db = db.drop_duplicates(['ts_code', 'trade_date'])          # 🔴 铁律: 先去重
        db['trade_date'] = pd.to_datetime(db['trade_date'].astype(str), format='%Y%m%d')
        db = db[db['trade_date'] >= '2024-05-01']
        # 日均成交额(亿元) = free_share(万股) * 1e4 * trf% * close(元) / 1e8
        db['adv_yi'] = db['free_share'] * 1e4 * (db['turnover_rate_f'] / 100.0) * db['close'] / 1e8
        g = db.groupby('ts_code').agg(adv_med=('adv_yi', 'median'),
                                      adv_p10=('adv_yi', lambda s: s.quantile(0.10)),
                                      circ_mv_med=('circ_mv', 'median'),
                                      total_mv_med=('total_mv', 'median'))
        cap = dict(
            n_codes=int(len(g)),
            adv_med_yi=float(g['adv_med'].median()),
            adv_p10_yi=float(g['adv_p10'].median()),
            adv_min_yi=float(g['adv_med'].min()),
            circ_mv_med_yi=float(g['circ_mv_med'].median() / 1e4),
            total_mv_med_yi=float(g['total_mv_med'].median() / 1e4),
            circ_mv_min_yi=float(g['circ_mv_med'].min() / 1e4),
        )
        # 组合容量: 15 只等权, 单票不超过 k% ADV -> 组合上限 = 15 * (k% * 最小 ADV 中位)
        cap['cap_yi_5pct_adv'] = float(15 * 0.05 * g['adv_p10'].median())
        cap['cap_yi_10pct_adv'] = float(15 * 0.10 * g['adv_p10'].median())
        out['capacity'] = cap
        print(f'  池内 {cap["n_codes"]} 只  中位日均成交额={cap["adv_med_yi"]:.2f}亿 '
              f'(p10={cap["adv_p10_yi"]:.2f}亿)  中位流通市值={cap["circ_mv_med_yi"]:.0f}亿 '
              f'最小={cap["circ_mv_min_yi"]:.0f}亿', flush=True)
        print(f'  组合容量粗估 (15 只等权, 按 p10 ADV 计): '
              f'5% ADV -> {cap["cap_yi_5pct_adv"]:.2f}亿 ; 10% ADV -> {cap["cap_yi_10pct_adv"]:.2f}亿',
              flush=True)
    except Exception as e:
        out['capacity'] = dict(error=str(e))
        print(f'  [skip] 容量估算失败: {e}', flush=True)

    os.makedirs(OUTS, exist_ok=True)
    json.dump(out, open(f'{OUTS}/deploy_ready.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(f'\n[saved] {OUTS}/deploy_ready.json', flush=True)


if __name__ == '__main__':
    main()
