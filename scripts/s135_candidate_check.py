"""
把 passed_factor_pool.csv 中 status=candidate 的因子重跑完整 S1-S5 管线。

复刻 RalphLoop._phase_evaluate 的验证路径:
  S1: Fast IC (预计算 ic_mean/icir vs 阈值 IC>=0.02, ICIR>=0.3)
  S2: 与 Memory 库模板的实时 Spearman 横截面相关 (阈值 <=0.5)
  S3: 批次内去重 (两两相关 >0.7 时 ICIR 低者淘汰)
  S4: OOS + 替换检查 (无预计算 OOS 回退到 icir)
  S5: S5JointFilter 真实回测 (优先) / IC 年拆分轻量过滤 (回退)

用法: python scripts/s135_candidate_check.py [--only NAMES]
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
FA_DIR = ROOT / "research" / "factor_alchemy"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(FA_DIR))

from multi_stage_validator import MultiStageValidator  # noqa: E402
from factor_ic_computer import FactorICComputer  # noqa: E402


def _is_pandas_compatible(formula: str) -> bool:
    """复刻 ralph_loop.py v0.5.3 的公式可执行性过滤"""
    import re as _re
    if not formula or str(formula).startswith("# TODO"):
        return False
    forge_patterns = [
        r'\bts_mean\(', r'\bts_std\(', r'\bts_pct\(', r'\bts_delta\(',
        r'\bts_zscore\(', r'\bts_corr\(', r'\bts_rank\(', r'\bts_skew\(',
        r'\bts_kurtosis\(', r'\bts_argmax\(', r'\bts_argmin\(',
        r'\bcs_rank\(', r'\bneg\(', r'\bscale\(', r'\breturns\b',
        r'\brank\(', r'\bdiv\(', r'\bsub\(', r'\badd\(', r'\bmul\(',
        r'\bif_else\(', r'\brolling_min\b', r'\brolling_max\b',
        r'\bpow\(', r'\blog\(', r'\bsqrt\(', r'\babs\(', r'\bsigmoid\(',
    ]
    for pat in forge_patterns:
        if _re.search(pat, str(formula)):
            return False
    fictional_cols = [
        'north_money', 'south_money',
        'rd_expense', 'rd_capitalized', 'rd_staff', 'revenue',
        'goodwill', 'intan_assets', 'concept_label',
    ]
    for fc in fictional_cols:
        if fc in str(formula):
            return False
    return True


def load_candidates(only=None):
    df = pd.read_csv(DATA_DIR / "passed_factor_pool.csv")
    cand = df[df["status"] == "candidate"].copy()
    if only:
        cand = cand[cand["name"].isin(only)]
    out = []
    for _, r in cand.iterrows():
        formula = str(r.get("formula", "") or "").strip()
        out.append({
            "factor_name": str(r["name"]),
            "formula": formula,
            "hypothesis": str(r.get("hypothesis", "") or "")[:300],
            "paradigm": str(r.get("category", "") or ""),
            "ic": float(r["ic_mean"]) if pd.notna(r.get("ic_mean")) else 0.0,
            "icir": float(r["icir"]) if pd.notna(r.get("icir")) else 0.0,
            "direction": str(r.get("direction", "long") or "long"),
        })
    return out


def load_lib_factors():
    """复刻 RalphLoop: 从 Memory success_templates + pattern_stats 收集库公式"""
    mem_path = DATA_DIR / "experience_memory.json"
    if not mem_path.exists():
        print("  [S2] ⚠️ experience_memory.json 不存在，库因子为空")
        return []
    with open(mem_path, encoding="utf-8") as f:
        mem = json.load(f)
    lib_factors = []
    seen = set()
    templates_raw = mem.get("success_templates", [])
    if isinstance(templates_raw, dict):
        templates_raw = list(templates_raw.values())
    for t in templates_raw:
        if isinstance(t, dict):
            name = t.get("pattern_id", t.get("factor_name", t.get("name", "")))
            formula = t.get("formula", t.get("expression", ""))
        else:
            continue
        if name and formula and name not in seen and _is_pandas_compatible(str(formula)):
            seen.add(name)
            lib_factors.append({"factor_name": str(name), "formula": str(formula)})
    pattern_stats = mem.get("pattern_stats", {})
    if isinstance(pattern_stats, dict):
        for pid, pdata in pattern_stats.items():
            if isinstance(pdata, dict):
                pf = pdata.get("formula", pdata.get("expression", ""))
                if pf and pid not in seen and _is_pandas_compatible(str(pf)):
                    seen.add(pid)
                    lib_factors.append({"factor_name": str(pid), "formula": str(pf)})
    return lib_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="只检查指定因子名")
    args = parser.parse_args()

    candidates = load_candidates(args.only)
    print(f"候选因子: {len(candidates)} 个")
    for c in candidates:
        print(f"  - {c['factor_name']} | {c['paradigm']} | IC={c['ic']:.4f} ICIR={c['icir']:.3f}"
              f" | formula={'有' if c['formula'] else '无'}")

    # IC 计算器
    ic_comp = FactorICComputer()
    print(f"\nFactorICComputer ready: {ic_comp.is_ready}")

    # 库因子
    lib_factors = load_lib_factors()
    print(f"库因子公式: {len(lib_factors)} 个")

    # S2 预计算: 与库相关性 (复刻 _phase_evaluate)
    n_corr = 0
    n_unavail = 0
    for cand in candidates:
        if not cand["formula"]:
            cand["max_corr"] = 0.0
            cand["max_corr_factor"] = ""
            continue
        mc, mf = ic_comp.compute_max_corr_vs_library(
            cand["formula"], lib_factors, candidate_name=cand["factor_name"])
        if mc >= 0:
            cand["max_corr"] = mc
            cand["max_corr_factor"] = mf
            n_corr += 1
        else:
            cand["max_corr"] = 0.0
            cand["max_corr_factor"] = ""
            n_unavail += 1
    print(f"[S2] 相关性计算: {n_corr}/{len(candidates)} 完成, {n_unavail} 不可算(保守放行)")

    # S5: 真实回测优先, 否则 IC 年拆分轻量过滤
    s5_filter = None
    s5_light = False
    try:
        from s5_joint_filter import S5JointFilter
        sf = S5JointFilter(top_n=80)  # P-20260909-A: 默认全量池, FUXI_S5_UNIVERSE 可回退
        if sf.is_ready:
            s5_filter = sf
            print("[S5] S5JointFilter 真实回测可用")
        else:
            print("[S5] S5JointFilter 数据未就绪 → 轻量回退")
    except Exception as e:
        print(f"[S5] S5JointFilter 不可用: {e} → 轻量回退")

    if s5_filter is None:
        s5_light = True
        n_s5 = 0
        for cand in candidates:
            if not cand["formula"]:
                continue
            yearly = ic_comp.compute_yearly_ic(cand["formula"], cand["factor_name"])
            if yearly:
                y25 = yearly.get("2025") or {}
                y26 = yearly.get("2026") or {}
                cand["excess_2025"] = abs(y25.get("ic", 0))
                cand["excess_2026"] = abs(y26.get("ic", 0))
                cand["calmar"] = abs(yearly.get("all", {}).get("icir", 0))
                n_s5 += 1
        print(f"[S5] 年拆分 IC 填充: {n_s5}/{len(candidates)}")

    validator = MultiStageValidator(data_dir=DATA_DIR)
    results, summary = validator.validate(
        candidates=candidates,
        library_signals=None,
        library_factors=lib_factors,
        s5_filter=s5_filter,
        s5_lightweight=s5_light,
    )

    # 输出明细
    print("\n" + "=" * 100)
    print(f"{'因子':<42} {'S1':<6} {'S2':<6} {'S3':<6} {'S4':<6} {'S5':<6} 结果")
    print("=" * 100)
    n_pass = [0, 0, 0, 0, 0]
    eligible = 0
    for c, r in zip(candidates, results):
        flags = [r.s1_passed, r.s2_passed, r.s3_passed, r.s4_passed, r.s5_passed]
        for i, f in enumerate(flags):
            n_pass[i] += int(f)
        marks = "".join("✅" if f else "❌" for f in flags)
        if r.eligible_for_jq:
            eligible += 1
        print(f"{c['factor_name']:<42} {marks}  {'→ JQ eligible' if r.eligible_for_jq else ''}")
        if not r.s1_passed:
            print(f"    S1: {r.s1_reason}")
        if r.s1_passed and not r.s2_passed:
            print(f"    S2: {r.s2_reason}")
        if r.s1_passed and r.s2_passed and not r.s3_passed:
            print(f"    S3: {r.s3_reason}")
        if r.s1_passed and r.s2_passed and r.s3_passed and not r.s4_passed:
            print(f"    S4: {r.s4_reason}")
        if all(r.s1_passed and r.s2_passed and r.s3_passed and r.s4_passed for r in [r]) and not r.s5_passed:
            print(f"    S5: {r.s5_reason}")

    print("=" * 100)
    print(f"漏斗: S1 剩 {n_pass[0]} → S2 剩 {n_pass[1]} → S3 剩 {n_pass[2]} "
          f"→ S4 剩 {n_pass[3]} → S5 剩 {n_pass[4]}")
    print(f"最终 JQ eligible: {eligible}/{len(candidates)}")
    print(f"S5 模式: {'真实回测 S5JointFilter' if not s5_light else 'IC年拆分轻量过滤(代理)'}")
    if summary:
        print(f"summary: {json.dumps(summary, ensure_ascii=False)[:300]}")
        # v0.2: 落盘结果 (供看板/追溯)
        import datetime as _dt
        summary["detail"] = [
            {
                "factor_name": c["factor_name"],
                "s1_passed": r.s1_passed, "s2_passed": r.s2_passed,
                "s3_passed": r.s3_passed, "s4_passed": r.s4_passed,
                "s5_passed": r.s5_passed, "eligible_for_jq": r.eligible_for_jq,
                "s5_reason": r.s5_reason if not r.s5_passed else "",
                "s5_eval_status": r.s5_eval_status,
                "s5_marking_diagnostics": r.s5_marking_diagnostics,
                "s5_economic_readings_status": r.s5_economic_readings_status,
                "s5_diagnostic_readings": {
                    "excess_2025": r.s5_year1_excess,
                    "excess_2026": r.s5_year2_excess,
                    "calmar_2025": r.s5_calmar_2025,
                    "calmar_2026": r.s5_calmar_2026,
                } if r.s5_economic_readings_status == "diagnostic_only_data_insufficient" else None,
            }
            for c, r in zip(candidates, results)
        ]
        out_path = Path(__file__).resolve().parent.parent / "data" / "s135_candidate_check_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"结果已落盘: {out_path}")


if __name__ == "__main__":
    main()
