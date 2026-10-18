# -*- coding: utf-8 -*-
"""
P-20260831 P1 多样性折扣 — 影子验收脚本 (Alpha2 MaxCorr 移植)
================================================================================
A. 参照库加载统计 (JQ 正面验证因子来源分布)
B. 模板折扣分布 (success_templates 池 vs JQ 参照 → jq_max_corr 分布, w=0.10 折扣强度)
C. 冠军校准 (CHAMPION 126 池 vs JQ 参照 → 有多少会被打折/打多狠; 王者自命中=预期)
D. GPBreeder 冒烟 (shadow/enforce 双模式 _layered_template_weights 行为)
E. apply_discount 单元验证

转正条件 (SOP):
  B 折扣强度分布合理 (大部分模板 corr<0.4 → 折扣 <4%, 少部分高占用模板被软引导)
  C 无系统性误伤 (非 JQ 冠军 corr 分布与随机水平一致, 而非普遍高 corr)
  D 双模式冒烟通过, shadow 生产行为完全一致
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FUXI_DIV_DISCOUNT", "shadow")

from forge import diversity_discount as dd

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                        "div_discount_shadow_report.json")


# ═══════════════════════════════════════════════════════════
# A. 参照库统计
# ═══════════════════════════════════════════════════════════

def part_a_reference_stats():
    lib = dd.load_jq_reference_library()
    print(f"[A] JQ 正参照库: {len(lib)} 个因子")
    for f in lib:
        print(f"    - {f['factor_name'][:55]:57s} | {f['formula'][:60]}")
    return {"n_reference": len(lib),
            "names": [f["factor_name"] for f in lib]}


# ═══════════════════════════════════════════════════════════
# B. 模板折扣分布
# ═══════════════════════════════════════════════════════════

def _load_template_pool(limit: int = 60):
    """从 experience_memory success_templates 取 pandas 兼容模板池"""
    import re
    _forge_pat = re.compile(
        r"\b(ts_mean|ts_std|ts_pct|ts_delta|ts_zscore|ts_corr|ts_rank|ts_skew|"
        r"ts_kurtosis|ts_argmax|ts_argmin|cs_rank|neg|scale|rank|div|sub|add|mul|"
        r"if_else|rolling_min|rolling_max|pow|sigmoid)\s*\(")
    em_path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                           "experience_memory.json")
    if not os.path.exists(em_path):
        return []
    em = json.load(open(em_path, encoding="utf-8"))
    raw = em.get("success_templates", [])
    if isinstance(raw, dict):
        raw = list(raw.values())
    pool = []
    for t in raw:
        if isinstance(t, dict):
            name = t.get("pattern_id", t.get("factor_name", t.get("name", "")))
            fml = t.get("formula", t.get("expression", ""))
        elif hasattr(t, "formula"):
            name = getattr(t, "pattern_id", getattr(t, "name", ""))
            fml = getattr(t, "formula", getattr(t, "expression", ""))
        else:
            continue
        if name and fml and not _forge_pat.search(str(fml)):
            pool.append({"factor_name": str(name), "formula": str(fml)})
    return pool[:limit]


def part_b_template_discount(limit: int = 60):
    pool = _load_template_pool(limit)
    if not pool:
        print("[B] 无模板池, 跳过")
        return {"skipped": True}
    cache = dd.get_shared_cache()
    corrs = cache.get_corrs_batch(
        [p["formula"] for p in pool], [p["factor_name"] for p in pool])
    dist = {k: 0 for k in dd.COUNTERS.snapshot()["corr_bucket"]}
    disc_gt5, disc_gt10, disc_gt20 = 0, 0, 0
    rows = []
    for p in pool:
        c = corrs.get(p["factor_name"], 0.0)
        f = dd.discount_factor(c)
        rows.append({"name": p["factor_name"], "corr": round(c, 4),
                     "discount": round(1.0 - f, 4)})
        if c < 0.2: dist["0.0-0.2"] += 1
        elif c < 0.4: dist["0.2-0.4"] += 1
        elif c < 0.6: dist["0.4-0.6"] += 1
        elif c < 0.8: dist["0.6-0.8"] += 1
        else: dist["0.8-1.0"] += 1
        if 1.0 - f >= 0.05: disc_gt5 += 1
        if 1.0 - f >= 0.10: disc_gt10 += 1
        if 1.0 - f >= 0.20: disc_gt20 += 1
    rows.sort(key=lambda r: -r["corr"])
    n = len(rows)
    print(f"[B] 模板池 {n} 个 vs JQ 参照的 corr 分布: {dist}")
    print(f"    折扣>5%: {disc_gt5} ({disc_gt5/n*100:.0f}%) | >10%: {disc_gt10} "
          f"({disc_gt10/n*100:.0f}%) | >20%: {disc_gt20} ({disc_gt20/n*100:.0f}%)")
    print(f"    Top 高占用模板 (会收到最强软引导):")
    for r in rows[:8]:
        print(f"      corr={r['corr']:.2f} disc={r['discount']:.0%}  {r['name'][:50]}")
    snap = dd.COUNTERS.snapshot()
    print(f"    计数器: {snap}")
    return {"n_templates": n, "dist": dist, "disc_gt5": disc_gt5,
            "disc_gt10": disc_gt10, "disc_gt20": disc_gt20,
            "top_occupied": rows[:8], "counters": snap}


# ═══════════════════════════════════════════════════════════
# C. 冠军校准 (126 CHAMPION vs JQ 参照)
# ═══════════════════════════════════════════════════════════

def part_c_champion_calibration(limit: int = 30):
    import re
    from seed_injector import CHAMPION_FACTORS
    _forge_pat = re.compile(
        r"\b(ts_mean|ts_std|ts_pct|ts_delta|ts_zscore|ts_corr|ts_rank|ts_skew|"
        r"ts_kurtosis|ts_argmax|ts_argmin|cs_rank|neg|scale|rank|div|sub|add|mul|"
        r"if_else|rolling_min|rolling_max|pow|sigmoid)\s*\(")
    # 只保留 pandas 可求值风格冠军 (跳过 forge 原语风格 — FactorICComputer 无法转换)
    champs = []
    n_forge_style = 0
    for c in CHAMPION_FACTORS:
        expr = c.get("expression", "")
        if not expr:
            continue
        if _forge_pat.search(expr):
            n_forge_style += 1
            continue
        champs.append({"factor_name": c.get("name", "champ"), "formula": expr})
    print(f"[C] 冠军池: pandas 风格 {len(champs)} 个, "
          f"forge 原语风格 {n_forge_style} 个跳过 (求值器无法转换)")
    champs = champs[:limit]
    cache = dd.get_shared_cache()
    ref_names = [f["factor_name"] for f in dd.load_jq_reference_library()]
    corrs = cache.get_corrs_batch(
        [c["formula"] for c in champs], [c["factor_name"] for c in champs])
    n_self = sum(1 for c in champs if c["factor_name"] in ref_names)
    vals = [corrs.get(c["factor_name"], 0.0) for c in champs]
    import numpy as np
    vals = [v for v in vals if v > 0]
    print(f"[C] 冠军 {len(champs)}: 自命中(即JQ参照本身) {n_self}, "
          f"corr>0 的 {len(vals)} 个")
    if vals:
        print(f"    corr 分布: mean={np.mean(vals):.3f} median={np.median(vals):.3f} "
              f"p90={np.percentile(vals, 90):.3f} max={max(vals):.3f}")
        high = sorted([(c["factor_name"], corrs.get(c["factor_name"], 0))
                       for c in champs], key=lambda x: -x[1])[:8]
        print(f"    最高占用冠军 (w=0.10 下打折最狠):")
        for nm, cv in high:
            print(f"      corr={cv:.2f} disc={1-dd.discount_factor(cv):.0%}  {nm[:50]}")
    return {"n_champs": len(champs), "n_forge_style_skipped": n_forge_style,
            "n_self": n_self,
            "n_corr_positive": len(vals),
            "corr_mean": float(np.mean(vals)) if vals else None,
            "corr_median": float(np.median(vals)) if vals else None,
            "corr_p90": float(np.percentile(vals, 90)) if vals else None}


# ═══════════════════════════════════════════════════════════
# D. GPBreeder 冒烟
# ═══════════════════════════════════════════════════════════

def part_d_breeder_smoke():
    from factor_expression_tree import GPBreeder
    metas = [
        {"factor_name": "tpl_jq_self", "formula": "(1.0 / ((volume_p + open_p) + 1e-6))",
         "verification_level": "jq_single", "success_rate": 0.8, "occurrence_count": 1},
        {"factor_name": "tpl_plain", "formula": "(close_p - close_p.shift(1)) / close_p.shift(1)",
         "verification_level": "s5_passed", "success_rate": 0.6, "occurrence_count": 3},
        {"factor_name": "tpl_unknown", "formula": "close_p.pct_change(20)",
         "verification_level": "stage2_only", "success_rate": 0.1, "occurrence_count": 5},
    ]
    out = {}
    for mode in ("shadow", "enforce"):
        os.environ["FUXI_DIV_DISCOUNT"] = mode
        dd.COUNTERS.reset()
        dd.reset_shared_cache()
        br = GPBreeder(diversity_w=0.10, seed=7)
        ws = br._layered_template_weights(metas)
        print(f"[D] mode={mode}: weights = {[round(w, 4) for w in ws]}")
        out[mode] = {"weights": [round(w, 4) for w in ws],
                     "counters": dd.COUNTERS.snapshot()}
    os.environ["FUXI_DIV_DISCOUNT"] = "shadow"
    dd.reset_shared_cache()
    # shadow 基线: w=0 时行为完全一致
    br0 = GPBreeder(diversity_w=0.0, seed=7)
    ws0 = [round(w, 4) for w in br0._layered_template_weights(metas)]
    print(f"[D] mode=w0基线: weights = {ws0}")
    assert all(abs(a - b) < 1e-6 for a, b in zip(ws0, out["shadow"]["weights"])), \
        "shadow 模式下行为必须与 w=0 完全一致!"
    print("[D] shadow 与 w=0 基线完全一致 ✓")
    # enforce 生效验证: tpl_jq_self (corr=1) 应被折扣 10%
    ws_enf = out["enforce"]["weights"]
    ws_sh = out["shadow"]["weights"]
    assert abs(ws_sh[0] - ws_enf[0]) > 1e-4, "enforce 模式下高占用模板应被折扣"
    print(f"[D] enforce 折扣生效 ✓ (tpl_jq_self: {ws_sh[0]} → {ws_enf[0]})")
    return out


# ═══════════════════════════════════════════════════════════
# E. 单元验证
# ═══════════════════════════════════════════════════════════

def part_e_unit():
    checks = []
    checks.append(("corr=0 不打折", abs(dd.apply_discount(1.0, 0.0) - 1.0) < 1e-12))
    checks.append(("corr=1 w=0.10 → 0.90", abs(dd.apply_discount(1.0, 1.0, 0.10) - 0.90) < 1e-12))
    checks.append(("corr=1 w=0.60 floor=0.5 → 0.50",
                   abs(dd.apply_discount(1.0, 1.0, 0.60) - 0.50) < 1e-12))
    checks.append(("corr=0.5 w=0.10 → 0.95", abs(dd.apply_discount(1.0, 0.5, 0.10) - 0.95) < 1e-12))
    checks.append(("corr>1 截断", abs(dd.apply_discount(1.0, 3.0, 0.10) - 0.90) < 1e-12))
    checks.append(("w=0 直通", abs(dd.apply_discount(0.7, 0.9, 0.0) - 0.7) < 1e-12))
    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"[E] {'PASS' if v else 'FAIL'}  {name}")
    return {"all_pass": ok, "checks": [{"name": n, "pass": v} for n, v in checks]}


def main():
    print(f"FUXI_DIV_DISCOUNT env = "
          f"{os.environ.get('FUXI_DIV_DISCOUNT', '(未设置, 默认 shadow)')}")
    report = {}
    report["A_reference_stats"] = part_a_reference_stats()
    report["B_template_discount"] = part_b_template_discount(limit=60)
    report["C_champion_calibration"] = part_c_champion_calibration(limit=126)
    report["D_breeder_smoke"] = part_d_breeder_smoke()
    report["E_unit"] = part_e_unit()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("=" * 64)
    print(f"报告已写入: {OUT_JSON}")
    print("转正条件 (SOP): B 折扣>10% 的模板占比合理(高占用少数) 且 "
          "C 冠军池 corr 分布呈长尾(多数低 corr) 且 D 双模式冒烟通过 且 E 全过")


if __name__ == "__main__":
    main()
