# -*- coding: utf-8 -*-
"""
FD 翻译因子全量 S1 效果检验 (与 09-05 抽样同口径)
==================================================================================
- 输入: data/_fd_tmp/translations.json
- 口径: FactorICComputer 全市场 270d fwd5 (与 09-05 抽样实证完全一致)
- 输出: data/_fd_tmp/fd_s1_results.csv (全量) + fd_s1_passed.json (过线清单)
- 阈值: |ic| >= 0.02 且 |icir| >= 0.1 且 ic 方向与 direction 标注一致 (正 IC 期望)
- 附带: 与库最大相关 max_corr (去重参考)
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "research" / "factor_alchemy"))
sys.path.insert(0, str(BASE / "scripts"))

from factor_ic_computer import FactorICComputer
from daily_factor_hypothesis import validate_and_preprocess_formula

TRANS_PATH = BASE / "data" / "_fd_tmp" / "translations.json"
OUT_CSV = BASE / "data" / "_fd_tmp" / "fd_s1_results.csv"
OUT_PASS = BASE / "data" / "_fd_tmp" / "fd_s1_passed.json"


def main():
    trans = json.load(open(TRANS_PATH, encoding="utf-8"))
    ic_comp = FactorICComputer()

    # --new-only: 跳过已有筛查记录的 slug (增量批次)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-only", action="store_true",
                    help="只筛 translations.json 中不在 fd_s1_results.csv 里的 slug")
    args = ap.parse_args()
    done_slugs = set()
    if args.new_only and OUT_CSV.exists():
        import pandas as pd
        done_slugs = set(pd.read_csv(OUT_CSV)["slug"].tolist())
        print(f"  [new-only] 已筛 {len(done_slugs)} 条, 跳过")

    results = []
    n_total = n_eval = n_precheck_fail = 0
    for slug, t in trans.items():
        n_total += 1
        if slug in done_slugs:
            continue
        formula = t.get("formula_pandas")
        if not formula:
            results.append({
                "slug": slug, "name_zh": t.get("name_zh", ""), "category": t.get("category", ""),
                "ic": None, "icir": None, "n_days": 0, "max_corr": None,
                "verdict": "SKIP_NO_FORMULA", "note": t.get("skip_reason", ""),
            })
            continue
        ok, warns, cleaned = validate_and_preprocess_formula(formula, slug)
        if not ok:
            n_precheck_fail += 1
            results.append({
                "slug": slug, "name_zh": t.get("name_zh", ""), "category": t.get("category", ""),
                "ic": None, "icir": None, "n_days": 0, "max_corr": None,
                "verdict": "PRECHECK_FAIL", "note": "; ".join(warns)[:200],
            })
            continue
        n_eval += 1
        try:
            r = ic_comp.compute(cleaned, slug)
            ic, icir, n_days = r["ic"], r["icir"], r["n_days"]
        except Exception as e:
            results.append({
                "slug": slug, "name_zh": t.get("name_zh", ""), "category": t.get("category", ""),
                "ic": None, "icir": None, "n_days": 0, "max_corr": None,
                "verdict": "EVAL_ERROR", "note": f"{type(e).__name__}: {str(e)[:120]}",
            })
            continue
        try:
            mc = ic_comp.compute_max_corr_vs_library(cleaned)
            max_corr = mc.get("max_corr") if isinstance(mc, dict) else None
        except Exception:
            max_corr = None

        # 过线判定: 期望 ic>0 (翻译时已取负), ic>=0.02 & icir>=0.1
        passed = (ic is not None and ic >= 0.02 and icir >= 0.1 and n_days >= 30)
        results.append({
            "slug": slug, "name_zh": t.get("name_zh", ""), "category": t.get("category", ""),
            "ic": ic, "icir": icir, "n_days": n_days, "max_corr": max_corr,
            "verdict": "PASS_S1" if passed else "FAIL_S1", "note": "",
        })
        print(f"[{n_eval}] {slug}: ic={ic:.4f} icir={icir:.4f} n={n_days} corr={max_corr} {'PASS' if passed else ''}")

    # 落盘 CSV (--new-only 时合并历史行)
    fieldnames = ["slug", "name_zh", "category", "ic", "icir", "n_days", "max_corr", "verdict", "note"]
    if args.new_only and OUT_CSV.exists():
        import pandas as _pd
        old_rows = _pd.read_csv(OUT_CSV).to_dict("records")
    else:
        old_rows = []
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(old_rows)
        w.writerows(results)

    passed_list = [r for r in results if r["verdict"] == "PASS_S1"]
    json.dump({
        "screened_at": datetime.now().isoformat(),
        "n_total": n_total, "n_eval": n_eval, "n_precheck_fail": n_precheck_fail,
        "n_pass": len(passed_list),
        "passed": passed_list,
    }, open(OUT_PASS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n完成: 总数 {n_total} | 可评估 {n_eval} | 预检失败 {n_precheck_fail} | S1 过线 {len(passed_list)}")
    for r in sorted(passed_list, key=lambda x: -(x["icir"] or 0))[:10]:
        print(f"  TOP {r['slug']}: ic={r['ic']:.4f} icir={r['icir']:.4f} corr={r['max_corr']}")


if __name__ == "__main__":
    main()
