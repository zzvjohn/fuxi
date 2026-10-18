# -*- coding: utf-8 -*-
"""A-4 随机暴露对照框架 (2026-09-11)

用途
----
为任何"池内择时 / 出场改造"变体提供**可证伪的噪声带**。
背景: 快刀择时的全部盈亏可用代数恒等式刻画 ——
    持仓时 C 与 B 持有**相同名单相同权重**, 故
        C 日收益 ≡ B 日收益 − 未持仓成员的加权收益
    ⇒ 择时超额 ≡ −(1−暴露) × 池子涨跌
即"个股级日频择时"的盈亏几乎全部来自**暴露算术**, 与信号判断无关。
因此任何改造的收益必须与"**同暴露、零技能**"的基准对比, 否则不可证伪
(参见 Harvey-Liu-Saretto-Pontiff 多重检验批判)。

两条基准
--------
(1) 随机暴露带 (non-parametric)
    保持变体每日 持仓只数 k_t 与 跟踪只数 n_t 不变, 在每个交易日从当日
    可交易成员中**随机选 k_t 只**持有, 重复 n_paths 次得到总收益分布。
    期望上  E[r_t] = (k_t/n_t) · mean(全体成员收益) = p_t · r_bh_t。
    判定: 变体总收益须 > 95% 分位, 才认为存在超越"暴露算术"的信号贡献。

(2) 暴露等价解析基准 (parametric, scaled_bh)
    nav_scaled = Π (1 + p_t · r_bh_t),  p_t = k_t/n_t
    —— "零选股技能、仅逐日复制该变体暴露路径"的基准。
    与 (1) 在期望上一致, 但无采样噪声, 用于快速对照。

使用
----
    from kuaidao_bench_framework import random_exposure_band, scaled_bh_nav, verdict
    band = random_exposure_band(res['member_rets'], res['n_pos'], n_paths=2000)
    v = verdict(res['perf']['total'], band)
"""
import numpy as np


def random_exposure_band(member_rets, n_pos, n_paths=2000, seed=42,
                         pcts=(5, 25, 50, 75, 95), rf_daily=0.0, with_mdd=False):
    """随机暴露带.

    member_rets : List[List[float]]  每个交易日 *全部可交易跟踪标的* 的当日收益
    n_pos       : List[int]          变体当日实际持仓只数 k_t
    n_track     : 由 len(member_rets[t]) 推得 n_t
    with_mdd    : True 时同时返回随机路径的 MDD 分布 (用于检验回撤改善是否超越噪声)

    return dict(n_paths, seed, pct{p5..p95}, mean, std, totals[, mdd_pct, mdd_mean, mdds])
    """
    assert len(member_rets) == len(n_pos), 'member_rets / n_pos 长度不一致'
    rng = np.random.default_rng(seed)
    lognav = np.zeros(n_paths, dtype=float)
    nav_path = np.ones((n_paths, len(member_rets)), dtype=float) if with_mdd else None
    for t, rets in enumerate(member_rets):
        r = np.asarray(rets, dtype=float)
        n = r.size
        k = int(n_pos[t])
        if n == 0 or k <= 0:
            rr = np.zeros(n_paths)
        elif k >= n:                     # 满仓 => 无选择自由度, 直接等权
            rr = np.full(n_paths, r.mean())
        else:
            # 向量化: 每个路径独立抽 k 个下标 (无放回)
            U = rng.random((n_paths, n))
            idx = np.argpartition(U, k - 1, axis=1)[:, :k]
            rr = r[idx].sum(axis=1) / n
        lognav += np.log1p(rr)
        if with_mdd:
            nav_path[:, t] = np.exp(lognav)
    totals = np.expm1(lognav)
    out = dict(n_paths=int(n_paths), seed=int(seed),
               pct={f'p{q}': float(np.percentile(totals, q)) for q in pcts},
               mean=float(totals.mean()), std=float(totals.std(ddof=1)),
               min=float(totals.min()), max=float(totals.max()),
               totals=totals)
    if with_mdd:
        run_max = np.maximum.accumulate(nav_path, axis=1)
        mdds = (nav_path / run_max - 1).min(axis=1)
        out.update(mdd_pct={f'p{q}': float(np.percentile(mdds, q)) for q in pcts},
                   mdd_mean=float(mdds.mean()), mdd_std=float(mdds.std(ddof=1)), mdds=mdds)
    return out


def scaled_bh_nav(r_bh, n_pos, n_track):
    """暴露等价解析基准: 每日按 p_t = k_t/n_t 缩放池子等权收益 r_bh_t."""
    nav, navs = 1.0, []
    for rb, k, n in zip(r_bh, n_pos, n_track):
        p = (k / n) if n else 0.0
        nav *= (1 + p * rb)
        navs.append(nav)
    return np.asarray(navs, dtype=float)


def verdict(total, band, alpha=0.05, mdd=None):
    """变体总收益 (与可选 MDD) vs 随机暴露带."""
    lo = band['pct'][f'p{int(alpha * 100)}']
    hi = band['pct'][f'p{int((1 - alpha) * 100)}']
    z = (total - band['mean']) / band['std'] if band['std'] > 0 else float('nan')
    # 分位 (变体在随机路径分布中的位置)
    rank = float((band['totals'] < total).mean()) if 'totals' in band else float('nan')
    out = dict(lo=float(lo), hi=float(hi), mean=float(band['mean']), z=float(z),
               rank=rank, beat_hi=bool(total > hi), inside=bool(lo <= total <= hi))
    if mdd is not None and 'mdds' in band:
        mlo = band['mdd_pct'][f'p{int((1 - alpha) * 100)}']   # MDD 越浅越好 => 用反向分位
        mhi = band['mdd_pct'][f'p{int(alpha * 100)}']
        mz = (mdd - band['mdd_mean']) / band['mdd_std'] if band['mdd_std'] > 0 else float('nan')
        mrank = float((band['mdds'] > mdd).mean())
        out.update(mdd_lo=float(mlo), mdd_hi=float(mhi), mdd_z=float(mz), mdd_rank=mrank,
                   mdd_better=bool(mdd > mhi))
    return out


def fmt_band(band, total=None, alpha=0.05):
    """一行式文本摘要."""
    p = band['pct']
    s = (f"random band n={band['n_paths']}: "
         f"p5={p['p5']:+.2%} p25={p['p25']:+.2%} p50={p['p50']:+.2%} "
         f"p75={p['p75']:+.2%} p95={p['p95']:+.2%} (mean={band['mean']:+.2%} sd={band['std']:.2%})")
    if total is not None:
        v = verdict(total, band, alpha)
        tag = 'BEATS 95%' if v['beat_hi'] else ('INSIDE BAND(噪声)' if v['inside'] else 'BELOW p5')
        s += f" | variant={total:+.2%} z={v['z']:+.2f} -> {tag}"
    return s
