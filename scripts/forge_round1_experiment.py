# -*- coding: utf-8 -*-
"""
Forge Round 1 手动试验 (P-025 变体)
======================================
P-025 原方案: --max_depth 6 --population_size 200 攻击流动性×微观结构单一范式。
本试验变体: 攻击 **成熟范式集合** (8 个, 含流动性×微观结构) —
依据 MAB reward>0 + JQ 已验证 breed + 王者复合构成:

  筹码分布(0.55) / 动量反转(0.35) / 流动性×微观结构(0.08+d_liq0.20) /
  下行保护(0.15) / 波动率适应(0.15) / 价量关系(JQ BEST +143%) /
  均值回复(JQ PASS +125%) / 尾部风险(王者复合)

已接入 FSA (subtree_fingerprinter):
  1. 外部库冻结骨架检查 (scan_library 280 因子, 阈值 15%)
  2. 运行内骨架集中度冻结 (某骨架 ≥15% 种群 或 ≥3 次 → 冻结, 适应度 -999)

试验流程:
  1. 重扫因子库 → FSA 外部禁止列表
  2. 加载 weekly_prices.parquet → Forge 数据 (T,N)
  3. FactorForge(pop=200, gens=5, depth=6, fsa + 8 范式 profiles)
  4. 骨架去重取 Top-K 候选 → S5JointFilter 真实回测 (两年超额+Calmar)
  5. 对照 gp_breed 基线 (通过率 ~10.4%) 输出裁决

用法: python scripts/forge_round1_experiment.py [--s5-top 15] [--pop 200]
"""
import json
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(FA_DIR))

# ═══════════════════════════════════════════════════════════
# 成熟范式 profiles — 统一从 forge 模块导入 (生产 ralph_loop 共用同源)
# ═══════════════════════════════════════════════════════════
from forge.paradigm_profiles import MATURE_PARADIGM_PROFILES  # noqa: E402


# ═══════════════════════════════════════════════════════════
# Step 1: FSA 重扫因子库
# ═══════════════════════════════════════════════════════════
def fsa_scan_library():
    from subtree_fingerprinter import SubtreeFingerprinter
    fsa = SubtreeFingerprinter()
    df = pd.read_csv(DATA_DIR / "passed_factor_pool.csv")
    factors = [
        {"factor_name": str(r["name"]), "formula": str(r.get("formula", "") or "")}
        for _, r in df.iterrows()
    ]
    report = fsa.scan_library(factors, threshold=0.15, persist=True)
    print(f"[FSA] 库扫描: {report.total_factors} 因子 → {report.total_skeletons} 骨架, "
          f"冻结 {report.forbidden_count}, 最大集中度 {report.max_concentration:.1%} "
          f"({report.max_concentration_skeleton[:50]})")
    for w in report.warnings[:5]:
        print(f"  {w}")
    return fsa, report


# ═══════════════════════════════════════════════════════════
# Step 2: 数据加载 (复刻 ralph_loop._load_forge_data)
# ═══════════════════════════════════════════════════════════
def load_forge_data():
    df = pd.read_parquet(ROOT / "output" / "ap_batch" / "cache" / "weekly_prices.parquet")
    df = df.rename(columns={"stock_code": "stock", "trade_date": "date",
                            "vol": "volume", "fwd_ret": "forward_return"})
    df["date"] = pd.to_datetime(df["date"])
    print(f"[Data] weekly_prices: shape={df.shape}, stocks={df['stock'].nunique()}, "
          f"dates={df['date'].min().date()}~{df['date'].max().date()}")

    data = {}
    pivot_cache = {}
    for fname, col in [("close", "close"), ("volume", "volume"),
                       ("open", "close"), ("high", "close"), ("low", "close")]:
        if col not in pivot_cache:
            pivot_cache[col] = df.pivot_table(
                index="date", columns="stock", values=col, aggfunc="last")
        data[fname] = pivot_cache[col].values.astype(float)

    # vwap: volume 代理 (与 to_pandas_string 的 vwap→volume_p 语义一致)
    data["vwap"] = data["volume"].copy()

    fwd = df.pivot_table(index="date", columns="stock",
                         values="forward_return", aggfunc="last")
    data["fwd_return"] = fwd.values.astype(float)

    close_arr = data["close"]
    hist_ret = np.full_like(close_arr, np.nan)
    if close_arr.shape[0] > 1:
        hist_ret[1:] = (close_arr[1:] - close_arr[:-1]) / np.where(
            close_arr[:-1] != 0, close_arr[:-1], np.nan)
    data["returns"] = hist_ret

    print(f"[Data] (T,N) = {close_arr.shape}, fields = {list(data.keys())}")
    return data


# ═══════════════════════════════════════════════════════════
# Step 3: Forge 进化
# ═══════════════════════════════════════════════════════════
def run_forge(data, fsa, pop_size, n_generations, max_depth):
    from forge import FactorForge

    forge = FactorForge(
        data=data,
        max_depth=max_depth,
        max_complexity=12.0,          # 与生产 ralph_loop 一致
        icir_threshold=0.15,
        forward_returns=data.get("fwd_return"),
        seed=42,
        fsa=fsa,
        fsa_retry=3,
        paradigm_profiles=MATURE_PARADIGM_PROFILES,
    )
    print(f"\n[Forge] 启动: pop={pop_size}, gens={n_generations}, "
          f"depth={max_depth}, paradigms={len(MATURE_PARADIGM_PROFILES)}, FSA={'on' if fsa else 'off'}")
    t0 = time.time()
    best_all = forge.evolve(
        n_generations=n_generations,
        pop_size=pop_size,
        tournament_size=3,
        elite_count=3,
        mutation_rate=0.3,
        crossover_rate=0.5,
        verbose=True,
    )
    print(f"[Forge] 完成: {time.time() - t0:.0f}s, best_all={len(best_all)}")
    return forge, best_all


# ═══════════════════════════════════════════════════════════
# Step 4: 骨架去重取 Top-K + S5 真实回测
# ═══════════════════════════════════════════════════════════
def select_top_candidates(final_pop, fsa, top_k, per_paradigm):
    """从最终种群按 fitness 骨架去重取 Top-K; 再并入每范式最佳 (保证范式覆盖)"""
    if fsa is None:
        skel_of = lambda b: b["expression"]  # noqa: E731
    else:
        skel_of = lambda b: fsa.parse_expression(b["expression"])  # noqa: E731

    seen = set()
    ranked = []
    for b in final_pop:  # 已按 fitness 降序
        skel = skel_of(b)
        if skel in seen:
            continue
        seen.add(skel)
        ranked.append(b)

    picked = ranked[:top_k]
    picked_ids = {id(p) for p in picked}

    if per_paradigm:
        for p in MATURE_PARADIGM_PROFILES:
            bests = [b for b in ranked if b.get("paradigm") == p]
            if not bests:
                continue
            top = bests[0]
            if id(top) not in picked_ids:
                picked.append(top)
                picked_ids.add(id(top))
    return picked


def evaluate_s5(picked):
    from s5_joint_filter import S5JointFilter

    sf = S5JointFilter(top_n=80, sample_stocks=200)
    candidates = [
        {
            "factor_name": f"forge_r1_{c['paradigm'] or 'u'}_{abs(hash(c['expression'])) % 100000}",
            "formula": c["pandas_expression"],
            "paradigm": c.get("paradigm"),
            "forge_icir": c.get("icir") or 0.0,
            "forge_fitness": c["fitness"],
            "expression": c["expression"],
        }
        for c in picked
    ]
    if not sf.is_ready:
        print("[S5] S5JointFilter 数据未就绪, 跳过真实回测")
        return None, [], candidates
    print(f"\n[S5] 对 {len(candidates)} 个候选做真实回测 (两年超额+Calmar)...")
    results = sf.validate_batch(candidates, verbose=True)
    return sf, results, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s5-top", type=int, default=15, help="S5 评估的 Top-K 候选数")
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--skip-forge", action="store_true",
                        help="跳过 GP 进化, 直接用已有 forge_round1_run.json 做 S5")
    args = parser.parse_args()

    # ── Step 1: FSA ──
    fsa, fsa_report = fsa_scan_library()

    # ── Step 2/3: 进化 ──
    run_path = DATA_DIR / "forge_round1_run.json"
    if args.skip_forge and run_path.exists():
        print("[Forge] 跳过进化, 复用上次运行结果")
        with open(run_path, encoding="utf-8") as f:
            saved = json.load(f)
        final_pop = saved.get("final_population") or saved.get("best_all")
        stats = saved["stats"]
    else:
        data = load_forge_data()
        forge, best_all = run_forge(data, fsa, args.pop, args.gens, args.depth)
        stats = forge.last_run_stats
        final_pop = forge.final_population
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump({"final_population": final_pop, "stats": stats,
                       "fsa_report": {"total_skeletons": fsa_report.total_skeletons,
                                      "forbidden_count": fsa_report.forbidden_count,
                                      "max_concentration": fsa_report.max_concentration}},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"[Save] {run_path}")
        print(f"[Forge] 最终种群: {len(final_pop)} 个体 (fitness 降序), "
              f"Top-5 ICIR: {[round(b.get('icir') or 0, 2) for b in final_pop[:5]]}")

    print("\n" + "=" * 90)
    print("Forge 运行统计")
    print("=" * 90)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_paradigm"},
                     ensure_ascii=False, indent=2))
    print("\n各范式最佳:")
    for p, s in stats.get("per_paradigm", {}).items():
        print(f"  [{p}] fit={s['best_fitness']:.3f} icir={s['best_icir']:+.3f} "
              f"gen={s['best_generation']} | {s['best_expression'][:60]}")

    # ── Step 4: 候选筛选 + S5 ──
    picked = select_top_candidates(final_pop, fsa, args.s5_top, per_paradigm=True)
    print(f"\n[S5] 最终种群骨架去重后选取 {len(picked)} 个候选 (Top-{args.s5_top} + 范式最佳补全)")

    sf, results, candidates = evaluate_s5(picked)
    if sf is None:
        print("[S5] S5 数据未就绪 — 本轮只产出 Forge 运行结果, 不做回测裁决")
        return

    # ── 汇总 ──
    print("\n" + "=" * 100)
    print(f"{'因子':<34} {'范式':<12} {'forge_ICIR':>9} {'S5':<6} "
          f"{'excess25':>9} {'excess26':>9} {'calmar25':>8} {'calmar26':>8}")
    print("=" * 100)
    n_pass = 0
    summary_rows = []
    for cand, r in zip(candidates, results):
        p = cand["paradigm"] or "-"
        if r.passed:
            n_pass += 1
        row = {
            "factor_name": cand["factor_name"], "paradigm": p,
            "forge_icir": cand["forge_icir"], "passed": r.passed,
            "excess_2025": getattr(r, "excess_2025", None),
            "excess_2026": getattr(r, "excess_2026", None),
            "calmar_2025": getattr(r, "calmar_2025", None),
            "calmar_2026": getattr(r, "calmar_2026", None),
            "reason": r.reason, "expression": cand["expression"],
            "pandas_expression": cand["formula"],
        }
        summary_rows.append(row)
        e25 = f"{r.excess_2025:.1%}" if getattr(r, "excess_2025", None) is not None else "-"
        e26 = f"{r.excess_2026:.1%}" if getattr(r, "excess_2026", None) is not None else "-"
        c25 = f"{r.calmar_2025:.2f}" if getattr(r, "calmar_2025", None) is not None else "-"
        c26 = f"{r.calmar_2026:.2f}" if getattr(r, "calmar_2026", None) is not None else "-"
        mark = "✅" if r.passed else "❌"
        print(f"{cand['factor_name'][:34]:<34} {p:<12} {cand['forge_icir']:>+9.3f} "
              f"{mark:<6} {e25:>9} {e26:>9} {c25:>8} {c26:>8}")

    print("=" * 100)
    pass_rate = n_pass / len(candidates) if candidates else 0
    print(f"\n裁决: S5 通过 {n_pass}/{len(candidates)} = {pass_rate:.0%} "
          f"(gp_breed 历史基线 ~10.4%)")
    print(f"FSA: 拦截 {stats.get('fsa_blocks', 0)} 次, 运行内冻结骨架 "
          f"{stats.get('n_run_frozen_skeletons', 0)} 个")

    out_path = DATA_DIR / "forge_round1_s5_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"pop": args.pop, "gens": args.gens, "depth": args.depth,
                       "s5_top": args.s5_top},
            "stats": stats,
            "n_s5_pass": n_pass, "n_evaluated": len(candidates),
            "rows": summary_rows,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
