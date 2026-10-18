# -*- coding: utf-8 -*-
"""
P-20260831 P2 top-k 双口径 ICIR 影子 — 分离力分析 (Alpha2 top-k 高光备份移植)
================================================================================
核心假设 (lane_calibration 混叠之谜):
  「pandas 周频 ICIR 与 JQ 结果相关性弱」— 实证锚点:
    FAILED  volume_dual_rank   = 0.509   ← FAILED 比 PASS 还高!
    PASS    ts_std             = 0.183
    MARGINAL square_square_tsmin = 0.230 / vol_zscore_inv = 0.013
  假设: FAILED 因子的高 ICIR 由少数尖峰周撑起 (脆弱), 稳健因子各周均匀贡献。
  探针: topk_ratio = top-k 高光均值口径 ICIR / 标准 ICIR — 比值越大越集中/越脆弱。

输出: 按 jq_outcome 分组的 标准 ICIR / topk_ratio / trim_ratio 分布 +
      卷对卷明细 → 判定「双口径能否分离 PASS vs FAILED 锚点」。
纯分析脚本, 不修改任何生产文件。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lane_calibration import load_calibration, review_pandas_full

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "output",
                        "topk_dual_caliber_report.json")


def main():
    cal = load_calibration()
    points = cal.get("points", [])
    print(f"校准集 {len(points)} 点")

    # 选分析对象: 全部有 jq_outcome 的 + pandas_icir 最高的 5 个无 outcome 点
    with_outcome = [p for p in points if p.get("jq_outcome")]
    no_outcome = [p for p in points if not p.get("jq_outcome")
                  and p.get("pandas_icir") is not None]
    no_outcome.sort(key=lambda p: -(p.get("pandas_icir") or 0))
    targets = with_outcome + no_outcome[:5]
    print(f"复评对象: {len(with_outcome)} 有 outcome + {len(no_outcome[:5])} 高 ICIR 对照")

    rows = []
    for p in targets:
        fml = p.get("formula", "")
        if not fml:
            continue
        full = review_pandas_full(fml, p.get("name", ""))
        if full is None:
            print(f"  [skip] eval 失败: {p.get('name', '?')[:45]}")
            rows.append({"name": p.get("name", ""),
                         "jq_outcome": p.get("jq_outcome"),
                         "eval_ok": False})
            continue
        rows.append({
            "name": p.get("name", ""),
            "jq_outcome": p.get("jq_outcome"),
            "eval_ok": True,
            "icir": full["weekly_icir"],
            "n": full["weekly_n"],
            "topk_icir": full["topk_icir"],
            "topk_ratio": full["topk_ratio"],
            "trim_icir": full["trim_icir"],
            "trim_ratio": full["trim_ratio"],
        })

    ok_rows = [r for r in rows if r.get("eval_ok")]
    print(f"\n成功复评 {len(ok_rows)}/{len(rows)}")

    # ── 分组统计 ──
    groups = {}
    for r in ok_rows:
        g = str(r["jq_outcome"] or "NONE")
        groups.setdefault(g, []).append(r)

    print("\n" + "=" * 78)
    print(f"{'outcome':14s} {'n':>3s} {'ICIR均值':>9s} {'topk_ratio均值':>15s} "
          f"{'trim_ratio均值':>15s} {'ratio>1.5占比':>12s}")
    print("-" * 78)
    summary = {}
    for g in sorted(groups):
        rs = groups[g]
        icirs = [r["icir"] for r in rs]
        trs = [r["topk_ratio"] for r in rs if r["topk_ratio"] is not None]
        mrs = [r["trim_ratio"] for r in rs if r["trim_ratio"] is not None]
        n_high = sum(1 for v in trs if v > 1.5)
        summary[g] = {
            "n": len(rs),
            "icir_mean": round(float(np.mean(icirs)), 4),
            "topk_ratio_mean": round(float(np.mean(trs)), 4) if trs else None,
            "trim_ratio_mean": round(float(np.mean(mrs)), 4) if mrs else None,
            "n_ratio_gt_1_5": n_high,
            "ratio_gt_1_5_pct": round(n_high / max(len(trs), 1), 3),
        }
        s = summary[g]
        print(f"{g:14s} {s['n']:>3d} {s['icir_mean']:>9.4f} "
              f"{str(s['topk_ratio_mean']):>15s} {str(s['trim_ratio_mean']):>15s} "
              f"{n_high}/{len(trs)} ({s['ratio_gt_1_5_pct']:.0%})")

    # ── 卷对卷明细 ──
    print("\n明细 (按 topk_ratio 降序):")
    for r in sorted(ok_rows, key=lambda x: -(x["topk_ratio"] or 0)):
        flag = ""
        if r["topk_ratio"] and r["topk_ratio"] > 1.5:
            flag = " ← 尖峰型?"
        print(f"  {str(r['jq_outcome'] or 'NONE'):11s} ICIR={r['icir']:.3f} "
              f"topk={r['topk_icir']:.3f}(r={str(r['topk_ratio']):>5s}) "
              f"trim={r['trim_icir']:.3f}(r={str(r['trim_ratio']):>5s}) "
              f"n={r['n']}{flag}  {r['name'][:40]}")

    # ── 分离力结论 ──
    print("\n" + "=" * 78)
    passed = groups.get("JQ_PASSED", [])
    failed = groups.get("JQ_FAILED", [])
    conclusion = {}
    if passed and failed:
        p_ratios = [r["topk_ratio"] for r in passed if r["topk_ratio"] is not None]
        f_ratios = [r["topk_ratio"] for r in failed if r["topk_ratio"] is not None]
        if p_ratios and f_ratios:
            p_m = float(np.mean(p_ratios))
            f_m = float(np.mean(f_ratios))
            separated = f_m > p_m * 1.3  # FAILED 集中度显著更高 → 有分离力
            conclusion = {
                "PASS_topk_ratio_mean": round(p_m, 3),
                "FAILED_topk_ratio_mean": round(f_m, 3),
                "separation_detected": bool(separated),
                "note": ("FAILED 组集中度显著更高 → 双口径对混叠有解释力, "
                         "可考虑纳入 S1 弱否决或 τ_w 校准旁证"
                         if separated else
                         "两组成集中度无显著差异 → 双口径无法解释混叠, 维持影子"),
            }
            print(f"PASS 组 topk_ratio 均值 = {p_m:.3f}  vs  FAILED 组 = {f_m:.3f}")
            print(f"分离力判定: {conclusion['note']}")
    else:
        conclusion = {"note": "PASS/FAILED 组样本不足, 无法判定"}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "conclusion": conclusion},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已写入: {OUT_JSON}")


if __name__ == "__main__":
    main()
