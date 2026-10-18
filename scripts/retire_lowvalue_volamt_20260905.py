# -*- coding: utf-8 -*-
"""
P-20260905-002: 低价值量额依赖因子批量退役

背景:
- 2026-09-05 P0-2 退役机制落地, 低价值库存亮牌 520 个
  (reserve + 无裁决验证痕迹 + |ic_mean|<0.02 + 未复评)
- 其中 39.2% 公式依赖 volume/amount 字段 — 该字段受历史数据污染
  (ic_diff_clean_vs_dirty.json: 量额因子 IC 平均失真 0.024, 7/30 符号翻转)
- 污染口径下 |ic_mean| 仍 <0.02 → 干净源真实信号只会更弱, 退役证据充分

动作 (--commit):
- 统一池 status: reserve → retired (所有同名 reserve 行)
- 落盘 data/library_retired.json (退役清单, 可追溯, 带回捞条件)
- 下游 ralph/MMR 经 load_factor_metadata 排除 retired (配套改造 P-20260905-002b)

用法:
  python scripts/retire_lowvalue_volamt_20260905.py            # 干跑预览
  python scripts/retire_lowvalue_volamt_20260905.py --commit   # 落盘退役
"""
import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
POOL_CSV = BASE / "data" / "unified_factor_pool.csv"
REEVAL_JSON = BASE / "data" / "p001_full_clean_reeval_20260905.json"
FREEZE_JSON = BASE / "data" / "library_freeze.json"
RETIRED_JSON = BASE / "data" / "library_retired.json"

IC_MEAN_WEAK = 0.02
VOLAMT_PAT = re.compile(r"\bvolume\b|\bamount\b")
TRACE_COLS = ["daily_icir", "weekly_icir", "jq_return"]


def _has(v) -> bool:
    try:
        return v is not None and not pd.isna(v)
    except (TypeError, ValueError):
        return True


def _load_reeval_names() -> set:
    if not REEVAL_JSON.exists():
        return set()
    with open(REEVAL_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return {r.get("name") for r in (d.get("results") or [])}


def _load_freeze_names() -> set:
    if not FREEZE_JSON.exists():
        return set()
    with open(FREEZE_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return {it.get("name") for it in d.get("frozen", [])}


def _load_retired_names() -> set:
    if not RETIRED_JSON.exists():
        return set()
    with open(RETIRED_JSON, encoding="utf-8") as f:
        d = json.load(f)
    return {it.get("name") for it in d.get("retired", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="落盘: 备份CSV → status=retired → 写 library_retired.json")
    args = ap.parse_args()

    pool = pd.read_csv(POOL_CSV)
    n_raw = len(pool)
    pool = pool.drop_duplicates(subset=["name"], keep="last").copy()
    reeval_names = _load_reeval_names()
    freeze_names = _load_freeze_names()
    retired_names = _load_retired_names()

    print(f"[退役筛选] 池原始 {n_raw} 行 / 去重后 {len(pool)} 因子 | "
          f"已复评 {len(reeval_names)} | 已冷冻 {len(freeze_names)} | "
          f"已退役 {len(retired_names)}")

    # ── 第一步: 低价值库存 (与 library_eviction.py 同口径) ──
    low_value = []
    for _, r in pool.iterrows():
        name = r["name"]
        status = str(r.get("status") or "reserve")
        if status != "reserve" or name in freeze_names or name in retired_names:
            continue
        # 无裁决验证痕迹
        has_trace = any(_has(r.get(c)) for c in TRACE_COLS if c in pool.columns)
        if has_trace:
            continue
        # 历史 |ic_mean| < 0.02 (非 NaN)
        im = r.get("ic_mean")
        try:
            imf = None if pd.isna(im) else float(im)
        except (TypeError, ValueError):
            imf = None
        if imf is None or abs(imf) >= IC_MEAN_WEAK:
            continue
        # 未复评
        if name in reeval_names:
            continue
        low_value.append((name, imf, r))

    print(f"\n低价值库存总数: {len(low_value)}")

    # ── 第二步: 量额依赖子集 ──
    volamt = []
    for name, imf, r in low_value:
        fx = str(r.get("formula") or "")
        if VOLAMT_PAT.search(fx):
            volamt.append((name, imf, r))

    print(f"量额依赖子集: {len(volamt)} (占低价值 {len(volamt)/max(len(low_value),1)*100:.1f}%)")

    if not volamt:
        print("\n无退役对象, 退出。")
        return

    # 画像
    src = {}
    par = {}
    signs = {"pos": 0, "neg": 0, "zero": 0}
    for name, imf, r in volamt:
        s = str(r.get("source") or r.get("origin") or "unknown")
        src[s] = src.get(s, 0) + 1
        p = str(r.get("paradigm") or "unknown")
        par[p] = par.get(p, 0) + 1
        if imf > 1e-9:
            signs["pos"] += 1
        elif imf < -1e-9:
            signs["neg"] += 1
        else:
            signs["zero"] += 1

    print("\n画像:")
    print(f"  ic_mean 符号: 正 {signs['pos']} / 负 {signs['neg']} / 零 {signs['zero']}")
    print(f"  source 分布: {dict(sorted(src.items(), key=lambda kv: -kv[1]))}")
    print(f"  paradigm 分布: {dict(sorted(par.items(), key=lambda kv: -kv[1]))}")
    print(f"\n前 20 个退役对象:")
    for name, imf, _ in sorted(volamt, key=lambda x: abs(x[1]))[:20]:
        print(f"  {name[:52]:<52} ic_mean={imf:+.4f}")

    if not args.commit:
        print("\n[dry-run] 未执行任何修改。加 --commit 落盘退役。")
        return

    # ── commit 路径 ──
    names = [v[0] for v in volamt]
    bak = POOL_CSV.with_suffix(".csv.bak_retire_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(POOL_CSV, bak)
    print(f"\n[commit] 已备份池: {bak}")

    mask = pool["name"].isin(names) & (pool["status"] == "reserve")
    n_updated = int(mask.sum())
    pool.loc[mask, "status"] = "retired"
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8")
    print(f"[commit] status 更新: reserve → retired, {n_updated} 行")

    now = datetime.now().isoformat(timespec="seconds")
    if RETIRED_JSON.exists():
        with open(RETIRED_JSON, encoding="utf-8") as f:
            rd = json.load(f)
    else:
        rd = {"version": 1, "retired": [], "updated_at": ""}
    existing = {it["name"] for it in rd["retired"]}
    new_entries = []
    for name, imf, r in volamt:
        if name in existing:
            continue
        new_entries.append({
            "name": name,
            "retired_at": now,
            "from_status": "reserve",
            "batch": "lowvalue_volamt_20260905",
            "reason": "低价值库存(从未裁决验证+|ic_mean|<0.02) 且公式依赖 volume/amount 字段(历史数据污染敏感, 污染口径下仍无信号)",
            "ic_mean": imf,
            "source": str(r.get("source") or r.get("origin") or ""),
            "paradigm": str(r.get("paradigm") or ""),
            "formula_len": len(str(r.get("formula") or "")),
            "recall_conditions": [
                "量额数据源污染修复 + 干净源复评 icir>=0.15",
                "新范式驱动需重验 (人工指定, 重新走 S1, 不进 MMR 主键)",
            ],
            "recall_audit": [],
        })
    rd["retired"].extend(new_entries)
    rd["updated_at"] = now
    with open(RETIRED_JSON, "w", encoding="utf-8") as f:
        json.dump(rd, f, ensure_ascii=False, indent=1)
    print(f"[commit] ✅ 已退役 {len(new_entries)} 个因子 → {RETIRED_JSON}")
    print("[commit] ⚠️ 下游联动: load_factor_metadata 需排除 retired "
          "(P-20260905-002b, 与 frozen 同链路)")


if __name__ == "__main__":
    main()
