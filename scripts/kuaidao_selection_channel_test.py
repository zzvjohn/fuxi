# -*- coding: utf-8 -*-
"""快刀斩股 选股通道改造实测: N-3 信号强度加权 + N-5 增量换池 (2026-09-11)

【Base 决策】
常数缩放暴露 k 的 Sharpe(k) = (k*m - rf_d)/(k*s)*sqrt(250) = (m - rf_d/k)/s*sqrt(250)
对 k **单调递增** ⇒ 任何降暴露都**降低** Sharpe (实测 k=0.61 -> 0.63 vs k=1.00 -> 0.69)。
报告 6.2 表的 "同暴露 61% 买入持有 Sharpe 0.82" 来自 scaledBH(p_t 路径),
是需要预知变体暴露路径的**基准**, 非可实盘策略。
⇒ 在"选 Sharpe 更大"的规则下, 可实盘选项中最大 Sharpe = 全暴露 B_quarter_bh (0.69)。

【本脚本】在 **纯 BH (无任何价格出场)** 之上只改"买什么/买多少" (选股通道):

N-3 信号强度加权 + 单票上限 (MAXW=0.10)
  basis='state' : close<=L250*1.03 -> 2.0 ; elif close<MA120 -> 0.0 ; elif 一阳穿4线 -> 1.5 ; else 1.0
  basis='event' : v3 信号事件 (B1 下跌触一年低点 -> 2.0 / B2 跌破MA120穿越 -> 0.0 / B3 一阳穿4线 -> 1.5)
  basis='filter': 仅 close<MA120 -> 0.0 (不参与), 其余等权
  renorm=True   : 0 权重资金**重分配**给其余标的 (gross 100%, 纯成分效应)
  renorm=False  : 0 权重资金**持现金** (gross<100%)

N-5 增量换池 n_drop: 换池时 newcomer/leaver 按各自池文件行序 (= 快刀 total 降序) 排序;
  最多替换 n_drop 只 —— 加入 rank 最好的 n_drop 个 newcomer, 保留 rank 最好的 (n_out-n_drop) 个 leaver。
  n_drop=None 即当前行为 (全量替换)。

【A-4 选股版安慰剂】保持"改多少"不变, 随机化"改谁":
  weight : 保持每期权重的多重集不变, 随机打乱分配给同一批标的
  drop   : 保持每期剔除只数不变, 随机挑标的剔除, 其余等权
  pool   : 保持每期替换只数不变, 随机挑标的进/出

前视: 权重/信号用**生效日当日收盘** (MOC 成交, 与 v4 口径一致); L250/R250 不含当日。
引擎自证: B_eq_bh 必须精确复现 v4 的 +39.14%。
输出: data/kuaidao_test/full_backtest/selection/{nav_sel.csv, stats_sel.json}
"""
import sys, os, json
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, 'e:/quant/scripts')
os.chdir('e:/quant')
import kuaidao_timing_backtest_v4 as kv  # noqa: E402

OUT = 'data/kuaidao_test/full_backtest'
OUTS = f'{OUT}/selection'
NEAR = kv.NEAR_TOL
MAXW = 0.10
N_PLACEBO = 400
ANNUAL = list(kv.ANNUAL)
QUARTERLY = list(kv.QUARTERLY)

CAL = kv.load_calendar()
CALIDX = {d: i for i, d in enumerate(CAL)}
NCAL = len(CAL)


# ---------------- 全样本逐日矩阵 (一次性, 供所有变体/安慰剂复用) ----------------
def load_pool_ordered(tag):
    df = pd.read_csv(f'{OUT}/pool_{tag}.csv', dtype={'code': str})
    return [c for c in df['code'].dropna().tolist() if c]


ALL_CODES = sorted({c for _, tg in QUARTERLY for c in load_pool_ordered(tg)})
CIDX = {c: i for i, c in enumerate(ALL_CODES)}
NC = len(ALL_CODES)


def build_market():
    """R[i,j] = 标的 j 在 cal[i] 的收益 (无数据=NaN); A[i,j] = 有数据; 另有 L250/MA120 供分类."""
    R = np.full((NCAL, NC), np.nan)
    A = np.zeros((NCAL, NC))
    LOW = np.zeros((NCAL, NC), dtype=bool)      # close <= L250*1.03
    BELOW = np.zeros((NCAL, NC), dtype=bool)    # close < MA120
    DROP = np.zeros((NCAL, NC), dtype=bool)     # close < prev_close
    YANG = np.zeros((NCAL, NC), dtype=bool)     # 一阳穿4线
    BEV = np.zeros((NCAL, NC), dtype=bool)      # 跌破 MA120 穿越事件(+5日下跌)
    for c in ALL_CODES:
        d, ind = kv.get_px(c), kv.get_ind(c)
        if d is None or ind is None:
            continue
        j = CIDX[c]
        close = d['close'].to_numpy(float)
        prevc = ind['prev_close'].to_numpy(float)
        l250 = ind['L250'].to_numpy(float)
        ma120 = ind['ma120'].to_numpy(float)
        ma120p = ind['ma120_prev'].to_numpy(float)
        c5 = ind['close5'].to_numpy(float)
        op = d['open'].to_numpy(float)
        mamin = ind['ma_min'].to_numpy(float)
        mamax = ind['ma_max'].to_numpy(float)
        for k, dt in enumerate(d.index):
            i = CALIDX.get(dt)
            if i is None:
                continue
            if np.isfinite(prevc[k]) and prevc[k] > 0:
                R[i, j] = close[k] / prevc[k] - 1.0
            A[i, j] = 1.0
            LOW[i, j] = np.isfinite(l250[k]) and close[k] <= l250[k] * (1 + NEAR)
            BELOW[i, j] = np.isfinite(ma120[k]) and close[k] < ma120[k]
            DROP[i, j] = np.isfinite(prevc[k]) and close[k] < prevc[k]
            YANG[i, j] = (np.isfinite(mamin[k]) and np.isfinite(mamax[k])
                          and op[k] < mamin[k] and close[k] > mamax[k])
            BEV[i, j] = (np.isfinite(ma120[k]) and np.isfinite(ma120p[k]) and np.isfinite(c5[k])
                         and close[k] < ma120[k] and prevc[k] >= ma120p[k] and close[k] < c5[k])
    return dict(R=R, A=A, LOW=LOW, BELOW=BELOW, DROP=DROP, YANG=YANG, BEV=BEV)


MK = None


def strength(c, i, basis):
    """返回 (raw_weight, label). 数据缺失 -> (1.0,'nodata') 不惩罚."""
    j = CIDX[c]
    if not MK['A'][i, j]:
        return 1.0, 'nodata'
    if basis == 'state':
        if MK['LOW'][i, j]:
            return 2.0, 'low'
        if MK['BELOW'][i, j]:
            return 0.0, 'below_ma120'
        if MK['YANG'][i, j]:
            return 1.5, 'yang4'
        return 1.0, 'normal'
    if basis == 'event':
        if MK['LOW'][i, j] and MK['DROP'][i, j]:
            return 2.0, 'low_ev'
        if MK['BEV'][i, j]:
            return 0.0, 'break_ev'
        if MK['YANG'][i, j]:
            return 1.5, 'yang4_ev'
        return 1.0, 'normal'
    if basis == 'filter':
        if MK['BELOW'][i, j]:
            return 0.0, 'below_ma120'
        return 1.0, 'normal'
    raise ValueError(basis)


def cap_weights(ws, cap, allow=None):
    """水填充: 仅在 allow (默认 权重>0) 的标的内分配, 0 权重者**永不复活**.

    ⚠ 若 cap * m < 总和 (m = 可用只数), cap 不可行 -> 抬到可行下限 1/m。
      否则超额会被灌进被剔除的 0 权重标的, "不参与"被反手取消 (2026-09-11 实测踩坑)。
    """
    if cap is None or not ws:
        return list(ws)
    ws = list(ws)
    if allow is None:
        allow = [w > 0 for w in ws]
    m = sum(allow)
    if m == 0:
        return ws
    cap = max(cap, 1.0 / m + 1e-12)
    for _ in range(100):
        over = [k for k in range(len(ws)) if allow[k] and ws[k] > cap + 1e-12]
        if not over:
            break
        excess = sum(ws[k] - cap for k in over)
        for k in over:
            ws[k] = cap
        free = [k for k in range(len(ws)) if allow[k] and ws[k] < cap - 1e-12]
        if not free:
            break
        base = sum(ws[k] for k in free)
        for k in free:
            ws[k] += excess * (ws[k] / base if base > 0 else 1.0 / len(free))
    return ws


# ---------------- 建造持仓计划 ----------------
def make_plan(eff_map, weights_mode='equal', basis='state', n_drop=None,
              cap=MAXW, renorm=True, rng=None, placebo=None, retain_once=False):
    plan = []
    members_prev = []
    entry_prev, rank_prev = {}, {}
    kept_once = set()
    for rd, tag in QUARTERLY:
        new_pool = load_pool_ordered(tag)
        eff = eff_map[rd]
        new_set = set(new_pool)
        new_rank = {c: k for k, c in enumerate(new_pool)}

        if n_drop is None or not members_prev:
            sel = list(new_pool)
        else:
            newcomers = [c for c in new_pool if c not in set(members_prev)]
            leavers = [c for c in members_prev if c not in new_set]
            m = min(len(newcomers), len(leavers))
            if placebo == 'pool':
                keep = [c for c in members_prev if c in new_set]
                pool_c = list(leavers) + list(newcomers)
                if 0 < m < len(pool_c):
                    idx = rng.permutation(len(pool_c))[:m]
                    sel = keep + [pool_c[int(k)] for k in idx]
                else:
                    sel = list(new_pool)
            else:
                n_rep = min(m, n_drop)
                add = newcomers[:n_rep]
                keep_n = m - n_rep
                if retain_once:
                    # 被保留的 leaver 只允许"延一期", 下次换池必须离场 -> 消除 rank 粘滞
                    c1 = [c for c in leavers if c not in kept_once]
                    keep_lev = sorted(c1, key=lambda c: rank_prev.get(c, 999))[:keep_n]
                    if len(keep_lev) < keep_n:
                        rest = [c for c in leavers if c not in keep_lev]
                        keep_lev += sorted(rest, key=lambda c: rank_prev.get(c, 999))[:keep_n - len(keep_lev)]
                    kept_once |= set(keep_lev)
                else:
                    keep_lev = sorted(leavers, key=lambda c: rank_prev.get(c, 999))[:keep_n]
                kept_once -= set(new_pool)
                sel = [c for c in members_prev if c in new_set] + keep_lev + add

        # ---- 权重 ----
        i_eff = CALIDX[eff]
        raw_w, labels = {}, {}
        for c in sel:
            if weights_mode == 'equal':
                raw_w[c], labels[c] = 1.0, 'equal'
            else:
                raw_w[c], labels[c] = strength(c, i_eff, basis)
        if weights_mode != 'equal' and placebo == 'weight' and rng is not None:
            vals = rng.permutation(np.array([raw_w[c] for c in sel], dtype=float))
            raw_w = {c: float(v) for c, v in zip(sel, vals)}
        elif weights_mode != 'equal' and placebo == 'drop' and rng is not None:
            k = sum(1 for c in sel if raw_w[c] == 0.0)
            if 0 < k < len(sel):
                idx = rng.permutation(len(sel))[:k]
                drop = {sel[int(v)] for v in idx}
                raw_w = {c: (0.0 if c in drop else 1.0) for c in sel}
        codes = list(raw_w)
        vals = [raw_w[c] for c in codes]
        tot = sum(vals)
        if tot > 0:
            vals = [v / tot for v in vals]
            vals = cap_weights(vals, cap)
        raw_w = {c: v for c, v in zip(codes, vals)}

        sel_set = set(sel)
        entry = {c: (entry_prev[c] if c in entry_prev else eff) for c in sel}
        plan.append(dict(eff=eff, tag=tag, i_eff=i_eff, sel=sel, w=raw_w, entry=entry,
                         labels=labels, renorm=renorm,
                         n_swap=sum(1 for c in sel if c not in sel_set or c not in entry_prev)))
        entry_prev = dict(entry)
        members_prev = list(sel)
        rank_prev = {c: new_rank.get(c, rank_prev.get(c, 99)) for c in sel}
    return plan


# ---------------- 向量化净值 ----------------
def simulate(plan, cal=None):
    cal = CAL if cal is None else cal
    ical = CALIDX
    idxs = [ical[p['eff']] for p in plan]
    nav = 1.0
    nav_rows, per_rows, bar_ret = [], [], []
    for k, p in enumerate(plan):
        i0 = idxs[k]
        i1 = idxs[k + 1] if k + 1 < len(plan) else NCAL
        cols = np.array([CIDX[c] for c in p['sel']])
        n = len(cols)
        Rb = MK['R'][i0:i1, cols].copy()
        Ab = MK['A'][i0:i1, cols]
        Bb = np.maximum.accumulate(Ab, axis=0)
        w = np.array([p['w'][c] for c in p['sel']], dtype=float)
        wnom = np.where(w > 0, w, 1.0 / max(n, 1))
        # 建仓日不计收益 (仅当日新进者; 跨期保留者按其原始建仓日)
        for j, c in enumerate(p['sel']):
            if p['entry'][c] == p['eff']:
                Rb[0, j] = 0.0
        Rz = np.nan_to_num(Rb, nan=0.0)
        if p['renorm']:
            den = Bb @ w
        else:
            den = Bb @ wnom
        num = (Rz * Ab) @ w
        r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        navs = nav * np.cumprod(1.0 + r)
        nav = float(navs[-1]) if len(navs) else nav
        for t, v, rr in zip(cal[i0:i1], navs, r):
            nav_rows.append(dict(date=t, nav=float(v), period=p['tag']))
        per_rows.append(dict(eff=p['eff'], tag=p['tag'], nav_end=nav))
        bar_ret.append(r)
    return nav_rows, per_rows


def period_returns(per_rows):
    out, prev = [], 1.0
    for p in per_rows:
        out.append(dict(eff=p['eff'], tag=p['tag'], ret=p['nav_end'] / prev - 1))
        prev = p['nav_end']
    return out


def label_mix(plan):
    cnt = Counter()
    for p in plan:
        for c in p['sel']:
            cnt[p['labels'][c] + ('(w0)' if p['w'][c] == 0 else '')] += 1
    return dict(cnt)


def make_and_run(eff_map, **kw):
    plan = make_plan(eff_map, **kw)
    nr, pr = simulate(plan)
    s = pd.DataFrame(nr).set_index('date')['nav']
    return plan, s, kv.perf(s), period_returns(pr)


# ---------------- 主流程 ----------------
def main():
    global MK
    MK = build_market()
    print(f'[market] codes={NC} cal={NCAL} ({CAL[0]}..{CAL[-1]})', flush=True)
    eff_map = {}
    for rd, _ in QUARTERLY:
        later = [d for d in CAL if d > rd]
        eff_map[rd] = min(later) if later else None

    specs = [
        ('B_eq_bh', dict(weights_mode='equal')),
        ('N3_state', dict(weights_mode='strength', basis='state')),
        ('N3_state_nocap', dict(weights_mode='strength', basis='state', cap=None)),
        ('N3_state_cash', dict(weights_mode='strength', basis='state', renorm=False)),
        ('N3_event', dict(weights_mode='strength', basis='event')),
        ('N3_filter_only', dict(weights_mode='strength', basis='filter')),
        ('N5_nd3', dict(weights_mode='equal', n_drop=3)),
        ('N5_nd5', dict(weights_mode='equal', n_drop=5)),
        ('N5_nd5_once', dict(weights_mode='equal', n_drop=5, retain_once=True)),
        ('N5_nd8', dict(weights_mode='equal', n_drop=8)),
        ('N3N5_state_nd5', dict(weights_mode='strength', basis='state', n_drop=5)),
    ]
    results, base = {}, None
    for name, kw in specs:
        plan, s, p, pr = make_and_run(eff_map, **kw)
        if base is None:
            base = p['total']
        results[name] = dict(plan=plan, nav=s, perf=p, periods=pr, kw=kw, mix=label_mix(plan))
        print(f'[{name:18s}] total={p["total"]:+8.2%} ({(p["total"]-base)*100:+6.2f}pp) '
              f'sharpe={p["sharpe"]:.2f} mdd={p["mdd"]:7.1%} 换入合计={sum(x["n_swap"] for x in plan):3d}'
              f' | {results[name]["mix"]}', flush=True)

    print('\n===== A-4 选股版安慰剂 (400 路径) =====', flush=True)
    placebo_out = {}
    for name, mode in [('N3_state', 'weight'), ('N3_state', 'drop'),
                       ('N3_event', 'weight'), ('N3_event', 'drop'),
                       ('N3_filter_only', 'drop'), ('N5_nd5', 'pool'),
                       ('N3_state', 'pool')]:
        kw = dict(dict(specs)[name])
        r_tot = results[name]['perf']['total']
        r_mdd = results[name]['perf']['mdd']
        r_sh = results[name]['perf']['sharpe']
        tots, mdds, shs = [], [], []
        for seed in range(N_PLACEBO):
            rng = np.random.default_rng(1000 + seed)
            _, _, pp, _ = make_and_run(eff_map, rng=rng, placebo=mode, **kw)
            tots.append(pp['total']); mdds.append(pp['mdd']); shs.append(pp['sharpe'])
        tots, mdds, shs = np.array(tots), np.array(mdds), np.array(shs)
        q = lambda a, x: float((a < x).mean() * 100)
        key = f'{name}|{mode}'
        placebo_out[key] = dict(
            real=r_tot, real_mdd=r_mdd, real_sharpe=r_sh,
            p5=float(np.percentile(tots, 5)), p50=float(np.percentile(tots, 50)),
            p95=float(np.percentile(tots, 95)), pctile=q(tots, r_tot),
            z=float((r_tot - tots.mean()) / tots.std(ddof=1)),
            mdd_p50=float(np.percentile(mdds, 50)),
            mdd_pctile_better=float((mdds > r_mdd).mean() * 100),
            sharpe_p50=float(np.percentile(shs, 50)), sharpe_pctile=q(shs, r_sh))
        o = placebo_out[key]
        print(f'[{key:24s}] real={o["real"]:+.2%} | placebo p5={o["p5"]:+.2%} p50={o["p50"]:+.2%} '
              f'p95={o["p95"]:+.2%} | 分位={o["pctile"]:5.1f}% z={o["z"]:+.2f} | '
              f'MDD real={o["real_mdd"]:.1%} p50={o["mdd_p50"]:.1%} 优于分位={o["mdd_pctile_better"]:.0f}% | '
              f'Sharpe real={o["real_sharpe"]:.2f} p50={o["sharpe_p50"]:.2f} 分位={o["sharpe_pctile"]:.0f}%',
              flush=True)

    bmap = {r['eff']: r['ret'] for r in results['B_eq_bh']['periods']}
    print('\n===== 分期超额 (vs B_eq_bh, 单位 pp) =====')
    hdr = f'{"period":9s} {"B_base":>8s}'
    for name, _ in specs[1:]:
        hdr += f' {name[:10]:>11s}'
    print(hdr)
    for i, r0 in enumerate(results['B_eq_bh']['periods']):
        line = f'{r0["tag"]:9s} {r0["ret"]:>8.2%}'
        for name, _ in specs[1:]:
            line += f' {(results[name]["periods"][i]["ret"]-r0["ret"])*100:>+10.2f}p'
        print(line)
    UP = ['P2024H1', 'P2024Q3', 'P2024', 'P2025Q3']
    print('\n上行段超额转正计数 / 全期跑赢期数:')
    for name, _ in specs[1:]:
        pos = sum(1 for i, r0 in enumerate(results['B_eq_bh']['periods'])
                  if r0['tag'] in UP and results[name]['periods'][i]['ret'] > r0['ret'])
        tot = sum(1 for i, r0 in enumerate(results['B_eq_bh']['periods'])
                  if results[name]['periods'][i]['ret'] > r0['ret'])
        print(f'  {name:18s} 上行段 {pos}/4   全期 {tot}/8')

    os.makedirs(OUTS, exist_ok=True)
    navs = pd.DataFrame({n: r['nav'] for n, r in results.items()})
    navs.index.name = 'date'
    navs.to_csv(f'{OUTS}/nav_sel.csv')
    json.dump(dict(
        base_choice=dict(
            note='常数缩放 Sharpe 对 k 单调递增; 可实盘 max = 全暴露 B (0.69). 0.82 来自 scaledBH(p_t 路径) 基准',
            frontier=json.load(open(f'{OUT}/timing4/base_choice.json'))['frontier']),
        engine_check=dict(B_eq_bh_total=results['B_eq_bh']['perf']['total'],
                          v4_reference='+39.14%'),
        results={n: dict(perf=r['perf'], kw=str(r['kw']), mix=r['mix'], periods=r['periods'],
                         swaps=sum(x['n_swap'] for x in r['plan']),
                         excess=[r['periods'][i]['ret'] - bmap[r['periods'][i]['eff']]
                                 for i in range(len(r['periods']))])
                 for n, r in results.items()},
        placebo=placebo_out, n_trials_plan=len(specs), n_placebo_paths=len(placebo_out) * N_PLACEBO,
    ), open(f'{OUTS}/stats_sel.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print('\n[saved] selection/nav_sel.csv / stats_sel.json')


if __name__ == '__main__':
    main()
