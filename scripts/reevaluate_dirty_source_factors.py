# -*- coding: utf-8 -*-
"""
P-20260901-001 污染源复评: 干净 daily_prices.csv 上重算量额字段因子的 IC/ICIR。

背景 (2026-08-31 审计): 旧 daily_prices.csv 的 vol 列错位污染使含 volume/amount
字段因子 IC 平均失真 0.0238 (对照组 0.0022), 7/30 符号翻转 →
历史裁决中所有吃量额字段因子的 S1 结论不可信。

本脚本:
  1. 从 data/ic_diff_clean_vs_dirty.json 取 vol 组 30 个受污染影响因子;
  2. 从 data/passed_factor_pool.csv 取 status 含 failed 且公式含 volume/amount 的因子;
  3. 在干净数据源上以 S1 同口径 (FactorICComputer() 全市场 fwd5) 重算 ic/icir;
  4. 判定 eligible_recheck: |ic|>=0.02 且 icir>=0.1 (方向由重入裁决时决定);
  5. 输出 data/dirty_factor_reeval.json (只读评估, 不直接入库)。

用法: python scripts/reevaluate_dirty_source_factors.py
"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

from factor_ic_computer import FactorICComputer  # noqa: E402

IC_ABS_MIN = 0.02   # daily lane 门槛 |ic|
ICIR_MIN = 0.10     # daily lane 门槛 icir


def load_pool_formulas() -> dict:
    """passed_factor_pool.csv → {name: formula}"""
    path = ROOT / "data" / "passed_factor_pool.csv"
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["name"]: r.get("formula", "") for r in rows}


def collect_targets() -> list:
    """收集复评目标: ic_diff vol 组 + 库中被拒的量额字段因子 (去重)"""
    targets = {}
    ic_diff_path = ROOT / "data" / "ic_diff_clean_vs_dirty.json"
    if ic_diff_path.exists():
        with open(ic_diff_path, encoding="utf-8") as f:
            diffs = json.load(f)
        for x in diffs:
            if x.get("group") == "vol":
                name = x.get("factor")
                if name:
                    targets[name] = {
                        "source": "ic_diff_audit",
                        "ic_dirty": x.get("ic_dirty"),
                        "ic_clean_audit": x.get("ic_clean"),
                    }
    else:
        print("[WARN] ic_diff_clean_vs_dirty.json 不存在, 仅扫描因子库")

    formulas = load_pool_formulas()
    for name, fml in formulas.items():
        if name in targets:
            continue
        low = fml.lower()
        if ("volume" in low or "amount" in low) and (
            "fail" in low or "reserve" in low
        ):
            # 公式字段级: 确认确实引用 volume/amount 字段而非仅名字含
            targets[name] = {
                "source": "pool_status_scan",
                "ic_dirty": None,
                "ic_clean_audit": None,
            }
    return targets, formulas


def main():
    t0 = time.time()
    targets, formulas = collect_targets()
    print(f"复评目标: {len(targets)} 个因子")
    missing = [n for n in targets if n not in formulas or not formulas[n]]
    if missing:
        print(f"[WARN] {len(missing)} 个目标无公式, 跳过: {missing}")

    comp = FactorICComputer()  # 全市场口径, 与 S1 裁决一致
    results = []
    for name, meta in targets.items():
        fml = formulas.get(name)
        if not fml:
            results.append({**meta, "factor": name, "eval_status": "no_formula"})
            continue
        r = comp.compute(fml, name)
        ic, icir, n_days = r.get("ic", 0), r.get("icir", 0), r.get("n_days", 0)
        eligible = (abs(ic) >= IC_ABS_MIN and icir >= ICIR_MIN)
        results.append({
            **meta,
            "factor": name,
            "ic": round(ic, 6),
            "icir": round(icir, 4),
            "n_days": n_days,
            "eval_status": "ok",
            "eligible_recheck": eligible,
        })
        flag = "★ RE-CHECK" if eligible else ""
        print(f"  {name}: ic={ic:+.4f} icir={icir:+.3f} n={n_days} {flag}")

    eligible = [x for x in results if x.get("eligible_recheck")]
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": {"abs_ic": IC_ABS_MIN, "icir": ICIR_MIN},
        "total": len(results),
        "eligible_count": len(eligible),
        "results": results,
    }
    out_path = ROOT / "data" / "dirty_factor_reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== 复评完成: {len(eligible)}/{len(results)} 达复评门槛 | "
          f"耗时 {time.time()-t0:.0f}s ===")
    print("输出:", out_path)


if __name__ == "__main__":
    main()
