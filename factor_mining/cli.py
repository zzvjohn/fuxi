"""factor_mining 统一 CLI。

用法（在 e:/quant 下，或任意位置，包以 -m 方式调用）:
  python -m factor_mining nsga2    --pop 120 --gen 40 [--seed 42] [--out DIR]
  python -m factor_mining gflownet --ep 300 --bs 16 --n 300 [--max-len 7] [--seed 0] [--out DIR]
  python -m factor_mining compare  --pop 120 --gen 40 --gf_ep 300 --gf_bs 16 --gf_n 300 [--out DIR]

数据源默认 e:/quant/research/factor_alchemy/output/{factor_cache,forward_returns}.pkl，
可用环境变量 FUXI_FACTOR_CACHE / FUXI_FORWARD_RETURNS 覆盖（供每日自动化接入不同口径）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_aligned
from .nsga2_biobjective import NSGA2Biobjective

DEFAULT_OUT = Path(__file__).parent / "results"


# ---------------------------------------------------------------- nsga2
def _run_nsga2(args) -> dict:
    factor_dict, fr = load_aligned()
    print(f"加载数据：因子 {len(factor_dict)} 个, fr {fr.shape}")
    ga = NSGA2Biobjective(factor_dict, fr, seed=args.seed)
    pareto = ga.run(pop_size=args.pop, n_generations=args.gen, verbose=True)
    print(f"\nPareto 解 {len(pareto)} 个")
    if pareto:
        sharpe = [p["gross_sharpe"] for p in pareto]
        to = [p["turnover"] for p in pareto]
        print(f"  毛Sharpe median={np.median(sharpe):.2f} max={max(sharpe):.2f}")
        print(f"  换手率  median={np.median(to):.3f}")
    payload = {"pareto": pareto, "n_pareto": len(pareto)}
    return payload


# ---------------------------------------------------------------- gflownet
def _run_gflownet(args) -> dict:
    from .gflownet.grammar import Grammar
    from .gflownet.model import GFlowNetPolicy
    from .gflownet.train import train, sample_diverse

    factor_dict, fr = load_aligned()
    print(f"加载数据：因子 {len(factor_dict)} 个, fr {fr.shape}")
    grammar = Grammar(list(factor_dict.keys()), max_len=args.max_len)
    grammar.set_factor_arrays(factor_dict)
    policy = GFlowNetPolicy(grammar.n_tokens)
    train(policy, grammar, fr, n_episodes=args.ep, batch_size=args.bs, seed=args.seed, verbose=True)
    samples = sample_diverse(policy, grammar, fr, n_samples=args.n)
    print(f"\n采样 {len(samples)} 个有效表达式")
    if samples:
        to = [d["turnover"] for d in samples]
        tk = [d["n_tokens"] for d in samples]
        print(f"  换手率 median={np.median(to):.3f} | 平均 token={np.mean(tk):.1f}")
    payload = {"samples": samples, "n_samples": len(samples)}
    return payload


# ---------------------------------------------------------------- compare
def _run_compare(args) -> dict:
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "factor_mining.run_comparison",
           "--pop", str(args.pop), "--gen", str(args.gen),
           "--gf_ep", str(args.gf_ep), "--gf_bs", str(args.gf_bs),
           "--gf_n", str(args.gf_n), "--max_len", str(args.max_len)]
    print("调用 run_comparison.py:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if r.returncode != 0:
        raise SystemExit(f"run_comparison.py 退出码 {r.returncode}")
    summary_path = Path(__file__).parent / "results" / "comparison_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------- dispatch
def main(argv=None):
    ap = argparse.ArgumentParser(prog="factor_mining",
                                 description="因子挖掘方法库：两目标 NSGA-II 与 GFlowNet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_nsga = sub.add_parser("nsga2", help="两目标 NSGA-II 组合层")
    p_nsga.add_argument("--pop", type=int, default=120)
    p_nsga.add_argument("--gen", type=int, default=40)
    p_nsga.add_argument("--seed", type=int, default=42)
    p_nsga.add_argument("--out", type=str, default=None)
    p_nsga.set_defaults(fn=_run_nsga2)

    p_gf = sub.add_parser("gflownet", help="GFlowNet 因子表达式生成器")
    p_gf.add_argument("--ep", type=int, default=300)
    p_gf.add_argument("--bs", type=int, default=16)
    p_gf.add_argument("--n", type=int, default=300)
    p_gf.add_argument("--max-len", type=int, default=7)
    p_gf.add_argument("--seed", type=int, default=0)
    p_gf.add_argument("--out", type=str, default=None)
    p_gf.set_defaults(fn=_run_gflownet)

    p_cmp = sub.add_parser("compare", help="两方法同口径对比")
    p_cmp.add_argument("--pop", type=int, default=120)
    p_cmp.add_argument("--gen", type=int, default=40)
    p_cmp.add_argument("--gf_ep", type=int, default=300)
    p_cmp.add_argument("--gf_bs", type=int, default=16)
    p_cmp.add_argument("--gf_n", type=int, default=300)
    p_cmp.add_argument("--max-len", type=int, default=7)
    p_cmp.add_argument("--out", type=str, default=None)
    p_cmp.set_defaults(fn=_run_compare)

    args = ap.parse_args(argv)

    out_dir = Path(args.out) if getattr(args, "out", None) else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    payload = args.fn(args)

    out_file = out_dir / f"{args.cmd}_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_file}  (总用时 {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
