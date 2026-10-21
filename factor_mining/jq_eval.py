"""
JQ 候选评估流水线（端到端）
============================
跑两目标 NSGA-II + GFlowNet 两个因子挖掘模块，对**全量**候选因子做 JQ 候选资格判定
+ 多样性度量，回答两个问题：
  Q1 能不能产出 JQ candidate？（按 JQ 门判定）
  Q2 能不能实现多样化？（内部相关 / 跨方法相关 / 去重叶子因子集 / 覆盖度）

JQ 门（忠实映射生产 E1「Spearman rank IC 描述有效性」+ 用户验证线 ICIR>=0.3 且 +IC%>=55%）:
  1. n_weeks >= 15           （OOS 有效周数充足）
  2. |icir| >= 0.3           （稳定排序：用户验证线，方向由 direction 自动翻转故取绝对值）
  3. +IC% >= 0.55            （IC 与检测方向一致的周期占比 >= 55%，用户验证线）
  4. turnover <= 0.5         （可交易性：周换手不超过 50%，防伪高收益但不可落地）

🔴 注意：icir 是比值型指标，这里只用于「验收/筛选」（JQ 门），不做「搜索目标」——
   搜索目标仍是 ic_mean 主导（data_loader.ic_strength_return_score）。这符合方法论铁律。

用法:
  python -m factor_mining.jq_eval --pop 120 --gen 40 --gf_ep 300 --gf_bs 16 --gf_n 300
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .data_loader import (load_aligned, evaluate_composite, _cross_sectional_ic,
                              linear_combine, shrinkage_icir, LAMBDA_PRIOR)
    from .nsga2_biobjective import NSGA2Biobjective
    from .gflownet.grammar import Grammar
    from .gflownet.model import GFlowNetPolicy
    from .gflownet.train import train, sample_diverse
except ImportError:  # 兼容平铺脚本运行
    from data_loader import (load_aligned, evaluate_composite, _cross_sectional_ic,
                             linear_combine, shrinkage_icir, LAMBDA_PRIOR)
    from nsga2_biobjective import NSGA2Biobjective
    from gflownet.grammar import Grammar
    from gflownet.model import GFlowNetPolicy
    from gflownet.train import train, sample_diverse

RESULTS = Path(__file__).parent / "results" / "jq_eval"

# ---- JQ 门阈值（可复现口径，贝叶斯收缩 ICIR 版） ----
MIN_N_WEEKS = 15
MIN_PLUS_IC_PCT = 0.55
MAX_TURNOVER = 0.5
# 🔴 比值型 ICIR 升级为贝叶斯收缩 ICIR（替代裸 |icir|>=0.3 + 硬 ic_std>=0.04）：
#   - 裸 |icir| 会被 std→0 伪信号穿透（实测 icir 爆炸到 368325）
#   - 硬 ic_std>=0.04 是拍脑袋的一刀切
#   - 收缩 ICIR = ic_mean/(ic_std+λ_prior)：分母加数据驱动先验，std→0 时不再爆炸，
#     连续、可微、有统计意义。门 = |shrink_icir| >= 阈值（保留强度验证）。
SHRINK_ICIR_THRESHOLD = 0.2  # 等价于原始 |icir|>=0.3 在典型 ic_std=0.12 下的收缩值
LCB_ALPHA = 0.05             # LCB 作为辅助诊断（不硬卡，记录稳健下界）
LCB_BLOCK_LENGTH = 12
LCB_REPLICATES = 999


def ic_series_of(df, fr) -> np.ndarray:
    common_idx = df.index.intersection(fr.index)
    common_cols = df.columns.intersection(fr.columns)
    C = df.loc[common_idx, common_cols].astype(np.float64).values
    F = fr.loc[common_idx, common_cols].astype(np.float64).values
    return _cross_sectional_ic(C, F)


def full_metrics(fr, comp_df):
    """一个候选因子的完整指标（含 JQ 门字段）。comp_df 为截面因子值 DataFrame。"""
    r = evaluate_composite(comp_df, fr)
    if r is None:
        return None
    ic = ic_series_of(comp_df, fr)
    ic_f = ic[np.isfinite(ic)]
    if len(ic_f) == 0:
        return None
    direction = r["direction"]
    plus_ic_pct = float((ic_f * direction > 0).mean())
    ic_std = float(np.std(ic_f, ddof=1)) if len(ic_f) > 1 else 0.0
    return {**r, "ic_series": ic, "plus_ic_pct": plus_ic_pct, "ic_std": ic_std}


def block_bootstrap_icir_lcb(ic_series, lambda_prior=LAMBDA_PRIOR, alpha=LCB_ALPHA,
                             block_length=LCB_BLOCK_LENGTH, replicates=LCB_REPLICATES, seed=0):
    """LCB(收缩 ICIR)：circular moving-block bootstrap，每个 resample 算 shrinkage_icir，取 alpha 分位下界。

    - block bootstrap 保持 IC 序列的时序相关性（比独立重采样更稳健）
    - 收缩分母 ic_std+λ_prior：std→0 时 icir→ic_mean/λ 不爆炸（贝叶斯收缩替代硬阈值）
    - 返回 None 表示数据不足（n < 2*block_length）
    """
    ic = np.array([x for x in ic_series if np.isfinite(x)])
    n = len(ic)
    if n < 2 * block_length:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(np.ceil(n / block_length)))
    idx = np.arange(block_length)
    draws = np.empty(replicates)
    for r in range(replicates):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([ic[(s + idx) % n] for s in starts])[:n]
        m = sample.mean()
        sd = sample.std(ddof=1)
        draws[r] = m / (sd + lambda_prior)   # shrinkage ICIR
    return float(np.quantile(draws, alpha))


def jq_gate(m) -> dict:
    """JQ 候选资格判定（贝叶斯收缩 ICIR 版），返回 {passed, checks, shrink_icir, lcb_shrink_icir}。

    升级点：裸 |icir|>=0.3 + 硬 ic_std>=0.04 → |shrinkage_icir| >= 0.2。
      - 收缩分母 ic_std+λ_prior 替代硬阈值（std→0 不爆炸，连续、数据驱动）
      - 收缩 ICIR 点估计 + 阈值保留「强度」验证（拒绝微弱正信号）
      - LCB(shrinkage_icir) 作为辅助诊断字段，不硬卡
    """
    if m is None:
        return {"passed": False, "checks": {"candidate": False},
                "shrink_icir": None, "lcb_shrink_icir": None}
    shrink = shrinkage_icir(m["ic_mean"], m.get("ic_std", 0.0))
    lcb = block_bootstrap_icir_lcb(m["ic_series"], seed=1729)
    checks = {
        "n_weeks>=15": bool(m["n_weeks"] >= MIN_N_WEEKS),
        "|shrink_icir|>=0.2": bool(abs(shrink) >= SHRINK_ICIR_THRESHOLD),
        "+IC%>=0.55": bool(m["plus_ic_pct"] >= MIN_PLUS_IC_PCT),
        "turnover<=0.5": bool(m["turnover"] <= MAX_TURNOVER),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "shrink_icir": shrink, "lcb_shrink_icir": lcb}


def pairwise_corr(series_list) -> float:
    if len(series_list) <= 1:
        return 0.0
    M = np.nan_to_num(np.vstack(series_list), nan=0.0)
    corr = np.corrcoef(M)
    n = corr.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.abs(corr[iu])))


def distinct_leaf_sets(nsga_cands, gf_cands, grammar):
    """去重叶子因子集（结构多样性）：返回 (nsga_distinct, gf_distinct, union_leafs)。"""
    nsga_sets = set()
    for c in nsga_cands:
        s = frozenset(k for k, w in c["weights"].items() if abs(w) > 1e-6)
        if s:
            nsga_sets.add(s)
    gf_sets = set()
    for c in gf_cands:
        leaves = frozenset(
            grammar.id2token[t] for t in c["token_ids"] if t >= grammar.leaf_start
        )
        if leaves:
            gf_sets.add(leaves)
    union = set()
    for s in nsga_sets | gf_sets:
        union |= s
    return len(nsga_sets), len(gf_sets), len(union), union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--gf_ep", type=int, default=300)
    ap.add_argument("--gf_bs", type=int, default=16)
    ap.add_argument("--gf_n", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=7)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("加载数据...")
    factor_dict, fr = load_aligned()
    n_factors = len(factor_dict)
    print(f"  因子 {n_factors} 个, fr {fr.shape}, 用时 {time.time()-t0:.0f}s")

    # ---------------- 1. 两目标 NSGA-II ----------------
    print("\n" + "=" * 64)
    print(f"① 两目标 NSGA-II (pop={args.pop}, gen={args.gen})")
    print("=" * 64)
    t1 = time.time()
    ga = NSGA2Biobjective(factor_dict, fr)
    pareto = ga.run(pop_size=args.pop, n_generations=args.gen, verbose=True)
    print(f"  NSGA-II 用时 {time.time()-t1:.0f}s, Pareto 解 {len(pareto)} 个")

    # ---------------- 2. GFlowNet ----------------
    print("\n" + "=" * 64)
    print(f"② GFlowNet (ep={args.gf_ep}, bs={args.gf_bs}, max_len={args.max_len})")
    print("=" * 64)
    t2 = time.time()
    grammar = Grammar(list(factor_dict.keys()), max_len=args.max_len)
    grammar.set_factor_arrays(factor_dict)
    policy = GFlowNetPolicy(grammar.n_tokens)
    train(policy, grammar, fr, n_episodes=args.gf_ep, batch_size=args.gf_bs, verbose=True)
    gf_samples = sample_diverse(policy, grammar, fr, n_samples=args.gf_n)
    print(f"  GFlowNet 用时 {time.time()-t2:.0f}s, 采样 {len(gf_samples)} 个有效表达式")

    # ---------------- 3. 全量候选评估 + JQ 门 ----------------
    print("\n" + "=" * 64)
    print("③ 全量候选评估 + JQ 门")
    print("=" * 64)

    nsga_cands = []
    for p in pareto:
        comp = linear_combine(factor_dict, p["weights"])
        if comp.empty:
            continue
        m = full_metrics(fr, comp)
        if m is None:
            continue
        gate = jq_gate(m)
        nsga_cands.append({
            "weights": p["weights"],
            "n_factors": p["n_factors"],
            "ic_mean": m["ic_mean"],
            "icir": m["icir"],
            "ic_std": m["ic_std"],
            "gross_sharpe": m["gross_sharpe"],
            "turnover": m["turnover"],
            "direction": m["direction"],
            "plus_ic_pct": m["plus_ic_pct"],
            "n_weeks": m["n_weeks"],
            "ic_series": m["ic_series"].tolist(),
            "jq_passed": gate["passed"],
            "jq_checks": gate["checks"],
            "shrink_icir": gate.get("shrink_icir"),
            "lcb_shrink_icir": gate.get("lcb_shrink_icir"),
        })

    gf_cands = []
    for d in gf_samples:
        comp_df = grammar.evaluate(d["token_ids"])
        if comp_df is None or not np.isfinite(comp_df).any():
            continue
        import pandas as pd
        comp_df = pd.DataFrame(comp_df, index=fr.index, columns=fr.columns)
        m = full_metrics(fr, comp_df)
        if m is None:
            continue
        gate = jq_gate(m)
        gf_cands.append({
            "token_ids": d["token_ids"],
            "expr": d["expr"],
            "n_tokens": d["n_tokens"],
            "ic_mean": m["ic_mean"],
            "icir": m["icir"],
            "ic_std": m["ic_std"],
            "gross_sharpe": m["gross_sharpe"],
            "turnover": m["turnover"],
            "direction": m["direction"],
            "plus_ic_pct": m["plus_ic_pct"],
            "n_weeks": m["n_weeks"],
            "ic_series": m["ic_series"].tolist(),
            "jq_passed": gate["passed"],
            "jq_checks": gate["checks"],
            "shrink_icir": gate.get("shrink_icir"),
            "lcb_shrink_icir": gate.get("lcb_shrink_icir"),
        })

    nsga_pass = [c for c in nsga_cands if c["jq_passed"]]
    gf_pass = [c for c in gf_cands if c["jq_passed"]]
    print(f"  NSGA-II: {len(nsga_pass)}/{len(nsga_cands)} 个 JQ 候选")
    print(f"  GFlowNet: {len(gf_pass)}/{len(gf_cands)} 个 JQ 候选")

    # ---------------- 4. 多样性 ----------------
    print("\n" + "=" * 64)
    print("④ 多样性度量")
    print("=" * 64)

    nsga_ics = [np.array(c["ic_series"]) for c in nsga_cands]
    gf_ics = [np.array(c["ic_series"]) for c in gf_cands]
    nsga_corr = pairwise_corr(nsga_ics)
    gf_corr = pairwise_corr(gf_ics)
    cross_corr = pairwise_corr(nsga_ics + gf_ics) if (nsga_ics and gf_ics) else 0.0

    # 结构多样性：去重叶子因子集 + 覆盖度
    nsga_distinct, gf_distinct, union_size, union_leafs = distinct_leaf_sets(
        nsga_cands, gf_cands, grammar)
    nsga_leaf_coverage = len(set().union(*[frozenset(c["weights"]) for c in nsga_cands])) / n_factors
    gf_leaf_coverage = len(set().union(*[frozenset(
        grammar.id2token[t] for t in c["token_ids"] if t >= grammar.leaf_start
    ) for c in gf_cands])) / n_factors

    print(f"  内部相关: NSGA-II={nsga_corr:.3f}  GFlowNet={gf_corr:.3f}")
    print(f"  跨方法相关(互补性): {cross_corr:.3f}  (越低=越互补)")
    print(f"  去重叶子因子集: NSGA-II={nsga_distinct}  GFlowNet={gf_distinct}")
    print(f"  叶子覆盖度: NSGA-II={nsga_leaf_coverage:.1%}  GFlowNet={gf_leaf_coverage:.1%}  (共 {n_factors} 因子)")

    # ---------------- 5. 落盘 ----------------
    summary = {
        "jq_gate": {
            "min_n_weeks": MIN_N_WEEKS,
            "min_plus_ic_pct": MIN_PLUS_IC_PCT, "max_turnover": MAX_TURNOVER,
            "gate_method": "shrinkage_icir_threshold",
            "shrink_icir_threshold": SHRINK_ICIR_THRESHOLD, "lambda_prior": LAMBDA_PRIOR,
            "lcb_aux": {"alpha": LCB_ALPHA, "block_length": LCB_BLOCK_LENGTH,
                        "replicates": LCB_REPLICATES},
        },
        "nsga2": {
            "n_candidates": len(nsga_cands), "n_jq_pass": len(nsga_pass),
            "pass_rate": len(nsga_pass) / max(1, len(nsga_cands)),
            "mean_internal_corr": nsga_corr,
            "distinct_leaf_sets": nsga_distinct, "leaf_coverage": nsga_leaf_coverage,
            "ic_mean": [c["ic_mean"] for c in nsga_cands],
            "icir": [c["icir"] for c in nsga_cands],
            "turnover": [c["turnover"] for c in nsga_cands],
            "gross_sharpe": [c["gross_sharpe"] for c in nsga_cands],
        },
        "gflownet": {
            "n_candidates": len(gf_cands), "n_jq_pass": len(gf_pass),
            "pass_rate": len(gf_pass) / max(1, len(gf_cands)),
            "mean_internal_corr": gf_corr,
            "distinct_leaf_sets": gf_distinct, "leaf_coverage": gf_leaf_coverage,
            "ic_mean": [c["ic_mean"] for c in gf_cands],
            "icir": [c["icir"] for c in gf_cands],
            "turnover": [c["turnover"] for c in gf_cands],
            "gross_sharpe": [c["gross_sharpe"] for c in gf_cands],
            "n_tokens": [c["n_tokens"] for c in gf_cands],
        },
        "cross_method_corr": cross_corr,
        "union_leaf_size": union_size,
        "n_factors_total": n_factors,
    }

    with open(RESULTS / "nsga2_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": nsga_cands}, f, ensure_ascii=False, indent=2, default=str)
    with open(RESULTS / "gflownet_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": gf_cands}, f, ensure_ascii=False, indent=2, default=str)
    with open(RESULTS / "jq_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到 {RESULTS}/")
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
