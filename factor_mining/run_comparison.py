"""
对比实验：两目标 NSGA-II vs GFlowNet 因子挖掘
================================================
用同一份数据、同一评估口径跑两个方法，对比:
  1. 收益-换手分布（Pareto 前沿 vs GFlowNet 采样散点）
  2. 方法内因子相关性（多样性）—— GFlowNet 的核心卖点验证
  3. 产出因子质量（毛Sharpe / ICIR / 换手）

用法:
  python run_comparison.py --pop 120 --gen 40 --gf_ep 300 --gf_bs 16 --gf_n 300
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import load_aligned, evaluate_composite, _cross_sectional_ic, linear_combine
    from .nsga2_biobjective import NSGA2Biobjective
    from .gflownet.grammar import Grammar
    from .gflownet.model import GFlowNetPolicy
    from .gflownet.train import train, sample_diverse
except ImportError:  # 兼容平铺脚本运行（python run_comparison.py）
    from data_loader import load_aligned, evaluate_composite, _cross_sectional_ic, linear_combine
    from nsga2_biobjective import NSGA2Biobjective
    from gflownet.grammar import Grammar
    from gflownet.model import GFlowNetPolicy
    from gflownet.train import train, sample_diverse

RESULTS = Path(__file__).parent / "results"


def ic_series_of(df: pd.DataFrame, fr: pd.DataFrame) -> np.ndarray:
    """因子值 DataFrame → 截面 IC 序列。"""
    common_idx = df.index.intersection(fr.index)
    common_cols = df.columns.intersection(fr.columns)
    C = df.loc[common_idx, common_cols].astype(np.float64).values
    F = fr.loc[common_idx, common_cols].astype(np.float64).values
    return _cross_sectional_ic(C, F)


def pairwise_corr(series_list) -> float:
    """一组 IC 序列的平均两两 Pearson 相关（衡量多样性；越低越多样）。"""
    if len(series_list) <= 1:
        return 0.0
    M = np.nan_to_num(np.vstack(series_list), nan=0.0)
    corr = np.corrcoef(M)
    n = corr.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.abs(corr[iu])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--gf_ep", type=int, default=300)
    ap.add_argument("--gf_bs", type=int, default=16)
    ap.add_argument("--gf_n", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=7)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    print("加载数据...")
    factor_dict, fr = load_aligned()
    print(f"  因子 {len(factor_dict)} 个, fr {fr.shape}, 用时 {time.time()-t0:.0f}s")

    # ---------------- 1. 两目标 NSGA-II ----------------
    print("\n" + "=" * 60)
    print(f"① 两目标 NSGA-II (pop={args.pop}, gen={args.gen})")
    print("=" * 60)
    t1 = time.time()
    ga = NSGA2Biobjective(factor_dict, fr)
    pareto = ga.run(pop_size=args.pop, n_generations=args.gen, verbose=True)
    print(f"  NSGA-II 用时 {time.time()-t1:.0f}s, Pareto 解 {len(pareto)} 个")

    # ---------------- 2. GFlowNet ----------------
    print("\n" + "=" * 60)
    print(f"② GFlowNet (ep={args.gf_ep}, bs={args.gf_bs}, max_len={args.max_len})")
    print("=" * 60)
    t2 = time.time()
    grammar = Grammar(list(factor_dict.keys()), max_len=args.max_len)
    grammar.set_factor_arrays(factor_dict)
    policy = GFlowNetPolicy(grammar.n_tokens)
    train(policy, grammar, fr, n_episodes=args.gf_ep, batch_size=args.gf_bs, verbose=True)
    gf_samples = sample_diverse(policy, grammar, fr, n_samples=args.gf_n)
    print(f"  GFlowNet 用时 {time.time()-t2:.0f}s, 采样 {len(gf_samples)} 个有效表达式")

    # ---------------- 3. 对比分析 ----------------
    print("\n" + "=" * 60)
    print("③ 对比分析")
    print("=" * 60)

    # NSGA-II Pareto 的 IC 序列（每个解一个 composite）
    nsga_ics = []
    nsga_metrics = []
    for p in pareto:
        comp = None
        comp = linear_combine(factor_dict, p["weights"])
        if comp.empty:
            continue
        ic = ic_series_of(comp, fr)
        nsga_ics.append(ic)
        nsga_metrics.append({
            "weights": p["weights"],
            "gross_sharpe": p["gross_sharpe"],
            "icir": p["icir"],
            "ic_mean": float(np.nanmean(ic)),
            "turnover": p["turnover"],
        })

    # GFlowNet top-K（对齐 NSGA-II Pareto 解数，公平对比）
    k = max(len(nsga_ics), 1)
    gf_top = gf_samples[:k]
    gf_ics = []
    gf_metrics = []
    gf_depth = []
    for d in gf_top:
        comp_df = pd.DataFrame(grammar.evaluate(d["token_ids"]),
                               index=fr.index, columns=fr.columns)
        ic = ic_series_of(comp_df, fr)
        gf_ics.append(ic)
        gf_depth.append(len(d["token_ids"]))
        gf_metrics.append({
            "expr": d["expr"],
            "n_tokens": len(d["token_ids"]),
            "gross_sharpe": d["gross_sharpe"],
            "icir": d["icir"],
            "ic_mean": float(np.nanmean(ic)),
            "turnover": d["turnover"],
        })

    nsga_corr = pairwise_corr(nsga_ics) if nsga_ics else 0.0
    gf_corr = pairwise_corr(gf_ics) if gf_ics else 0.0
    # 跨方法相关（互补性：越低说明 GFlowNet 探索了 NSGA-II 未覆盖的区域）
    cross_corr = pairwise_corr(nsga_ics + gf_ics) if (nsga_ics and gf_ics) else 0.0

    summary = {
        "nsga2": {
            "n_pareto": len(pareto),
            "n_evaluated": len(nsga_metrics),
            "sharpe": [p["gross_sharpe"] for p in pareto],
            "turnover": [p["turnover"] for p in pareto],
            "icir": [p["icir"] for p in pareto],
            "ic_mean": [m["ic_mean"] for m in nsga_metrics],
            "mean_internal_corr": nsga_corr,
        },
        "gflownet": {
            "n_samples": len(gf_samples),
            "n_topk": len(gf_metrics),
            "sharpe": [d["gross_sharpe"] for d in gf_top],
            "turnover": [d["turnover"] for d in gf_top],
            "icir": [d["icir"] for d in gf_top],
            "ic_mean": [m["ic_mean"] for m in gf_metrics],
            "n_tokens": gf_depth,
            "mean_internal_corr": gf_corr,
        },
        "cross_method_corr": cross_corr,
    }

    print(f"\n  NSGA-II: Pareto {len(pareto)} 解 | 平均Sharpe={np.mean(summary['nsga2']['sharpe']):.2f} "
          f"| 平均换手={np.mean(summary['nsga2']['turnover']):.2f} | 内部相关={nsga_corr:.3f}")
    print(f"  GFlowNet: top-{len(gf_top)} 表达式 | 平均Sharpe={np.mean(summary['gflownet']['sharpe']):.2f} "
          f"| 平均换手={np.mean(summary['gflownet']['turnover']):.2f} | 内部相关={gf_corr:.3f} "
          f"| 平均token数={np.mean(gf_depth) if gf_depth else 0:.1f}")
    print(f"  跨方法相关(互补性) = {cross_corr:.3f}  (越低=GFlowNet 越互补)")

    # 保存结果
    with open(RESULTS / "nsga2_pareto.json", "w", encoding="utf-8") as f:
        json.dump({"pareto": nsga_metrics}, f, ensure_ascii=False, indent=2, default=str)
    with open(RESULTS / "gflownet_samples.json", "w", encoding="utf-8") as f:
        json.dump({"samples": gf_metrics}, f, ensure_ascii=False, indent=2, default=str)
    with open(RESULTS / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到 {RESULTS}/")
    print(f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
