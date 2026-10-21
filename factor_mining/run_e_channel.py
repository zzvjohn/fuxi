"""
端到端：NSGA-II + GFlowNet 产出候选 → JQ 门（收缩版 + 旧硬门对照）→ 生产 E 道（E1-E4 移植）
====================================================================================
这是 Task 18 的落地脚本（Path B）：把生产 E 道判据移植到 97 因子宇宙后，跑一次端到端闭环，
回答两个问题：
  Q1  升级后的 JQ 门（贝叶斯收缩 ICIR）与旧硬门（|icir|>=0.3 + ic_std>=0.04）通过率差多少？
  Q2  通过 JQ 门的候选，能有多少穿过生产 E1-E4 研究门（方向/稳定/双所/独立性/配对增量/压力）？

用法:
  python -m factor_mining.run_e_channel --pop 60 --gen 20 --gf_ep 150 --gf_bs 16 --gf_n 150
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import data_loader
    from . import e_channel
    from .jq_eval import jq_gate, full_metrics, shrinkage_icir
    from .nsga2_biobjective import NSGA2Biobjective
    from .gflownet.grammar import Grammar
    from .gflownet.model import GFlowNetPolicy
    from .gflownet.train import train, sample_diverse
except ImportError:
    import data_loader
    import e_channel
    from jq_eval import jq_gate, full_metrics, shrinkage_icir
    from nsga2_biobjective import NSGA2Biobjective
    from gflownet.grammar import Grammar
    from gflownet.model import GFlowNetPolicy
    from gflownet.train import train, sample_diverse

RESULTS = Path(__file__).parent / "results" / "e_channel"

# 旧硬门（对照基线，来自 reapply_gate.py 的历史口径）
OLD_MIN_ABS_ICIR = 0.3
OLD_MIN_IC_STD = 0.04
OLD_MIN_PLUS_IC_PCT = 0.55
OLD_MAX_TURNOVER = 0.5
OLD_MIN_N_WEEKS = 15


def old_hard_gate(m):
    """旧硬门：|icir|>=0.3 且 ic_std>=0.04（已被收缩版替代，这里仅作通过率对照）。"""
    checks = {
        "n_weeks>=15": bool(m["n_weeks"] >= OLD_MIN_N_WEEKS),
        "|icir|>=0.3": bool(abs(m["icir"]) >= OLD_MIN_ABS_ICIR),
        "+IC%>=0.55": bool(m["plus_ic_pct"] >= OLD_MIN_PLUS_IC_PCT),
        "turnover<=0.5": bool(m["turnover"] <= OLD_MAX_TURNOVER),
        "ic_std>=0.04": bool(m["ic_std"] >= OLD_MIN_IC_STD),
    }
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--gf_ep", type=int, default=150)
    ap.add_argument("--gf_bs", type=int, default=16)
    ap.add_argument("--gf_n", type=int, default=150)
    ap.add_argument("--max_len", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("加载数据 + 构建 97 因子池 rank...")
    fd, fr = data_loader.load_aligned()
    names = list(fd.keys())
    pool_ranks = {}
    for n in names:
        pool_ranks[n] = e_channel._cross_sectional_rank_matrix(fd[n].astype(np.float64).values)
    print(f"  因子 {len(names)} 个, fr {fr.shape}, 池 rank 构建完成, 用时 {time.time()-t0:.0f}s")

    # ---------------- ① 两目标 NSGA-II ----------------
    print("\n" + "=" * 64)
    print(f"① 两目标 NSGA-II (pop={args.pop}, gen={args.gen})")
    print("=" * 64)
    t1 = time.time()
    ga = NSGA2Biobjective(fd, fr, seed=args.seed)
    pareto = ga.run(pop_size=args.pop, n_generations=args.gen, verbose=True)
    print(f"  NSGA-II 用时 {time.time()-t1:.0f}s, Pareto 解 {len(pareto)} 个")

    # ---------------- ② GFlowNet ----------------
    print("\n" + "=" * 64)
    print(f"② GFlowNet (ep={args.gf_ep}, bs={args.gf_bs}, max_len={args.max_len})")
    print("=" * 64)
    t2 = time.time()
    grammar = Grammar(names, max_len=args.max_len)
    grammar.set_factor_arrays(fd)
    policy = GFlowNetPolicy(grammar.n_tokens)
    train(policy, grammar, fr, n_episodes=args.gf_ep, batch_size=args.gf_bs, seed=args.seed, verbose=True)
    gf_samples = sample_diverse(policy, grammar, fr, n_samples=args.gf_n)
    print(f"  GFlowNet 用时 {time.time()-t2:.0f}s, 采样 {len(gf_samples)} 个有效表达式")

    # ---------------- ③ 构建候选 + JQ 门 + E 道 ----------------
    print("\n" + "=" * 64)
    print("③ 候选评估：JQ 门（收缩/旧硬对照）+ 生产 E 道（E1-E4）")
    print("=" * 64)

    candidates = []  # 统一结构：{method, repr, comp_df, metrics, jq_gate, e_channel}

    def add_candidate(method, comp_df, meta):
        m = full_metrics(fr, comp_df)
        if m is None:
            return
        new_gate = jq_gate(m)
        old_pass, old_checks = old_hard_gate(m)
        # E 道（全样本口径，漏斗短路）
        e_res = e_channel.evaluate_candidate_e_channel(comp_df, fr, pool_ranks, names)
        candidates.append({
            "method": method,
            **meta,
            "ic_mean": m["ic_mean"], "icir": m["icir"], "ic_std": m["ic_std"],
            "gross_sharpe": m["gross_sharpe"], "turnover": m["turnover"],
            "direction": m["direction"], "plus_ic_pct": m["plus_ic_pct"], "n_weeks": m["n_weeks"],
            "ic_series": np.asarray(m["ic_series"]).tolist(),
            "jq_passed": new_gate["passed"], "jq_checks": new_gate["checks"],
            "shrink_icir": new_gate["shrink_icir"],
            "lcb_shrink_icir": new_gate["lcb_shrink_icir"],
            "old_gate_passed": old_pass, "old_gate_checks": old_checks,
            "e_channel": e_res,
        })

    # NSGA-II 候选
    for p in pareto:
        comp = data_loader.linear_combine(fd, p["weights"])
        if comp.empty:
            continue
        add_candidate("nsga2", comp, {"weights": p["weights"], "n_factors": p["n_factors"]})

    # GFlowNet 候选
    for d in gf_samples:
        comp_arr = grammar.evaluate(d["token_ids"])
        if comp_arr is None or not np.isfinite(comp_arr).any():
            continue
        comp_df = pd.DataFrame(comp_arr, index=fr.index, columns=fr.columns)
        add_candidate("gflownet", comp_df, {"token_ids": d["token_ids"], "expr": d["expr"], "n_tokens": d["n_tokens"]})

    print(f"  共 {len(candidates)} 个候选（NSGA-II {sum(1 for c in candidates if c['method']=='nsga2')} / "
          f"GFlowNet {sum(1 for c in candidates if c['method']=='gflownet')}）")

    # ---------------- ④ 汇总漏斗 ----------------
    def stat(subset, key):
        n = len(subset)
        if n == 0:
            return 0
        if callable(key):
            return sum(1 for c in subset if key(c))
        return sum(1 for c in subset if c[key])

    summary = {
        "config": {"pop": args.pop, "gen": args.gen, "gf_ep": args.gf_ep,
                   "gf_bs": args.gf_bs, "gf_n": args.gf_n, "max_len": args.max_len, "seed": args.seed},
        "jq_gate": {"method": "shrinkage_icir_threshold", "shrink_threshold": 0.2,
                    "old_hard": {"min_abs_icir": OLD_MIN_ABS_ICIR, "min_ic_std": OLD_MIN_IC_STD}},
        "totals": {"n_candidates": len(candidates)},
    }
    for method in ("nsga2", "gflownet", "all"):
        sub = [c for c in candidates if method == "all" or c["method"] == method]
        summary[method] = {
            "n": len(sub),
            "jq_pass_new": stat(sub, "jq_passed"),
            "jq_pass_old": stat(sub, "old_gate_passed"),
            "e1": stat(sub, lambda c: c["e_channel"]["funnel"]["e1"]),
            "e2": stat(sub, lambda c: c["e_channel"]["funnel"]["e2"]),
            "e3": stat(sub, lambda c: c["e_channel"]["funnel"]["e3"]),
            "e4": stat(sub, lambda c: c["e_channel"]["funnel"]["e4"]),
        }

    # ---------------- ⑤ 落盘 ----------------
    # 序列化：去掉 comp_df（大数组），保留可读字段
    out_candidates = []
    for c in candidates:
        oc = {k: v for k, v in c.items() if k not in ("comp_df",)}
        # ic_series 已在 full_metrics 里，保留但截断显示
        if "ic_series" in oc and isinstance(oc["ic_series"], (list, np.ndarray)):
            oc["ic_series"] = np.asarray(oc["ic_series"]).tolist()
        out_candidates.append(oc)

    with open(RESULTS / "e_channel_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": out_candidates, "summary": summary}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到 {RESULTS}/")
    print(f"总用时 {time.time()-t0:.0f}s")

    # ---------------- ⑥ 打印漏斗 ----------------
    print("\n" + "=" * 64)
    print("E 道通过率漏斗（全样本口径）")
    print("=" * 64)
    for method in ("nsga2", "gflownet", "all"):
        s = summary[method]
        print(f"{method:8s} n={s['n']:3d}  "
              f"JQ新={s['jq_pass_new']:3d} JQ旧={s['jq_pass_old']:3d}  "
              f"E1={s['e1']:3d} E2={s['e2']:3d} E3={s['e3']:3d} E4={s['e4']:3d}")


if __name__ == "__main__":
    main()
