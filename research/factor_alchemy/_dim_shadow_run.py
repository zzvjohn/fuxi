"""
_dim_shadow_run.py
==================
P-20260831: 维度一致性预剪枝 — 影子期统计脚本 (Alpha2 移植验收)

三部分:
  A. 随机树生成统计: shadow 模式下 6000 棵 grow/full 树的硬违规率/软标记率,
     按 (算子, 量纲对) 聚合 — "被剪掉的子树占比" 主证据
  B. JQ 冠军因子校准审计: CHAMPION_FACTORS (JQ 验证过的种子) 若被硬规则标记,
     说明规则过严 → 需要收紧规则再转正 (误杀冠军=不可接受)
  C. enforce 模式压力测试: 重采样成功率 + mini evolve 冒烟 (grow/full/交叉/变异
     全路径), 验证生成器在硬剪下不死循环、行为符合预期

运行 (default venv):
  cd research/factor_alchemy
  C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe _dim_shadow_run.py
"""

import os
import sys
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from forge import dimension_rules as dr
from forge.expression import grow_random_tree, full_random_tree
from forge.primitives import Primitive, INPUT_PRIMITIVES

OUT_JSON = os.path.join(SCRIPT_DIR, "..", "..", "output",
                        "dim_prune_shadow_report.json")


def part_a_generation_stats(seed=42, n_each=3000):
    print("=" * 64)
    print("A. 随机树生成统计 (shadow 模式, 只计数不拦截)")
    print("=" * 64)
    rng = np.random.RandomState(seed)
    dr.reset_counters()
    dr.set_mode("shadow")

    for kind, gen in (("grow", grow_random_tree), ("full", full_random_tree)):
        n_hard = n_soft = n_clean = 0
        for _ in range(n_each):
            t = gen(rng, max_depth=4, dim_prune="shadow")
            aud = dr.audit_forge_node(t, record=False)
            if aud is None:
                n_hard += 1  # 解析失败按违规计 (几乎不会发生)
                continue
            if aud.n_hard:
                n_hard += 1
            if aud.n_soft:
                n_soft += 1
            if aud.clean:
                n_clean += 1
        print(f"  {kind}: 生成 {n_each} 棵 | 含硬违规 {n_hard} "
              f"({n_hard / n_each:.1%}) | 含软标记 {n_soft} "
              f"({n_soft / n_each:.1%}) | 全干净 {n_clean} ({n_clean / n_each:.1%})")

    snap = dr.COUNTERS.snapshot()
    print(f"  [生成期门事件] 硬违规 {snap['n_hard_events']} | "
          f"软标记 {snap['n_soft_events']} (标记事件合计 {snap['n_flag_events']})")
    print("  Top 硬违规 (算子, 量纲对):")
    for k, v in list(snap["hard_by_op"].items())[:8]:
        print(f"    {k}: {v}")
    print("  Top 软标记:")
    for k, v in list(snap["soft_by_op"].items())[:8]:
        print(f"    {k}: {v}")
    return snap


def part_b_champion_calibration():
    print("=" * 64)
    print("B. JQ 冠军因子校准审计 (硬规则误杀冠军 = 规则过严)")
    print("=" * 64)
    try:
        from seed_injector import CHAMPION_FACTORS
    except Exception as e:
        print(f"  [!] 无法导入 CHAMPION_FACTORS: {e}")
        return None

    hard_hits, soft_hits, unparsed = [], [], []
    for c in CHAMPION_FACTORS:
        expr = c.get("expression") or c.get("formula") or ""
        name = c.get("name", "?")
        if not expr:
            continue
        aud = dr.audit_expression_string(expr, record=False)
        if aud is None:
            unparsed.append((name, expr[:60]))
            continue
        if aud.n_hard:
            hard_hits.append((name, aud.hard_violations))
        if aud.n_soft:
            soft_hits.append((name, aud.soft_marks))

    n = len(CHAMPION_FACTORS)
    print(f"  冠军因子总数: {n} | 硬违规 {len(hard_hits)} | "
          f"软标记 {len(soft_hits)} | 无法解析 {len(unparsed)}")
    if hard_hits:
        print("  ⚠️ 硬规则标记的冠军 (转正前必须处理, 否则会误杀):")
        for name, vios in hard_hits:
            for path, reason in vios[:3]:
                print(f"    - {name}: {reason} @ {path}")
    else:
        print("  ✅ 无冠军被硬规则标记 — 规则未误杀 JQ 验证因子")
    if unparsed:
        print(f"  [i] 无法解析 {len(unparsed)} 条 (字段未注册不影响: UNKNOWN 不剪枝):")
        for name, e in unparsed[:5]:
            print(f"    - {name}: {e}...")
    return {"n_champions": n, "n_hard": len(hard_hits),
            "n_soft": len(soft_hits), "n_unparsed": len(unparsed),
            "hard_hits": [h[0] for h in hard_hits],
            "soft_hits": [s[0] for s in soft_hits][:20],
            "unparsed": [u[0] for u in unparsed]}


def part_c_enforce_stress(seed=7, n_each=3000):
    print("=" * 64)
    print("C. enforce 模式压力测试 (硬剪 + 重采样)")
    print("=" * 64)
    rng = np.random.RandomState(seed)
    dr.reset_counters()
    dr.set_mode("enforce")

    for kind, gen in (("grow", grow_random_tree), ("full", full_random_tree)):
        n_hard = n_clean = 0
        t0 = time.time()
        for _ in range(n_each):
            t = gen(rng, max_depth=4, dim_prune="enforce")
            aud = dr.audit_forge_node(t, record=False)
            if aud is None or not aud.clean:
                n_hard += 1
            else:
                n_clean += 1
        dt = time.time() - t0
        print(f"  {kind}: 生成 {n_each} 棵 | 残留硬违规 {n_hard} "
              f"({n_hard / n_each:.1%}) | 干净 {n_clean} | "
              f"耗时 {dt:.1f}s ({n_each / dt:.0f} 棵/s)")
    snap = dr.COUNTERS.snapshot()
    print(f"  [生成期门事件] 硬违规拦截 {snap['n_hard_events']} 次")
    return {"residual_hard_rate": n_hard / n_each,
            "gate_hard_events": snap["n_hard_events"]}


def part_d_mini_evolve_smoke():
    print("=" * 64)
    print("D. FactorForge mini evolve 冒烟 (shadow + enforce 全路径)")
    print("=" * 64)
    from forge import FactorForge
    T, N = 80, 60
    rng = np.random.RandomState(42)
    data = {
        "open": 10 + rng.randn(T, N).cumsum(axis=0),
        "high": 10 + rng.randn(T, N).cumsum(axis=0) + 1.0,
        "low": 10 + rng.randn(T, N).cumsum(axis=0) - 1.0,
        "close": 10 + rng.randn(T, N).cumsum(axis=0),
        "volume": np.abs(1e6 + rng.randn(T, N) * 1e5),
        "vwap": 10 + rng.randn(T, N).cumsum(axis=0),
        "returns": rng.randn(T, N) / 100,
    }
    fwd = rng.randn(T, N) / 100
    for mode in ("shadow", "enforce"):
        fg = FactorForge(
            data, vault=None, chronicle=None,
            max_depth=4, icir_threshold=0.0, forward_returns=fwd,
            seed=42, fsa=None, fsa_retry=3, dim_prune=mode)
        try:
            fg.evolve(n_generations=2, pop_size=20, workers=1, verbose=False)
            s = fg.last_run_stats
            print(f"  {mode}: 完成 2 代 | dim_hard={s['dim_hard']} "
                  f"dim_soft={s['dim_soft']} | 合格 {s['n_qualified']} | "
                  f"模式 {s['dim_prune_mode']}")
        except Exception as e:
            print(f"  {mode}: ❌ 异常 {type(e).__name__}: {e}")
            return False
    return True


def part_e_breeder_smoke():
    print("=" * 64)
    print("E. GPBreeder 育种冒烟 (champion 模板, shadow + enforce)")
    print("=" * 64)
    try:
        from factor_expression_tree import GPBreeder
        from seed_injector import CHAMPION_FACTORS
    except Exception as e:
        print(f"  [!] 导入失败: {e}")
        return False
    templates = [{"formula": c.get("expression", ""), "name": c.get("name", "")}
                 for c in CHAMPION_FACTORS[:10]]
    for mode in ("shadow", "enforce"):
        br = GPBreeder(dim_prune=mode)
        try:
            kids = br.breed_from_templates(templates, n_children=30,
                                           output_format="pandas")
            print(f"  {mode}: 产出 {len(kids)} 子代 | dim_hard={br.dim_hard} "
                  f"dim_soft={br.dim_soft}")
        except Exception as e:
            print(f"  {mode}: ❌ 异常 {type(e).__name__}: {e}")
            return False
    return True


def main():
    print(f"FUXI_DIM_PRUNE env = {os.environ.get('FUXI_DIM_PRUNE', '(未设置, 默认 shadow)')}")
    report = {}
    report["A_generation_shadow"] = part_a_generation_stats()
    report["B_champion_calibration"] = part_b_champion_calibration()
    report["C_enforce_stress"] = part_c_enforce_stress()
    report["D_mini_evolve_smoke"] = part_d_mini_evolve_smoke()
    report["E_breeder_smoke"] = part_e_breeder_smoke()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("=" * 64)
    print(f"报告已写入: {OUT_JSON}")
    print("转正条件 (SOP): B 冠军硬违规=0 且 C 残留硬违规率≈0 且 "
          "D/E 冒烟通过; A 硬违规率仅作收益度量 (影子→对照→转正)")


if __name__ == "__main__":
    main()
