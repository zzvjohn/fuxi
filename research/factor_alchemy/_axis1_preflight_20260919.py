# -*- coding: utf-8 -*-
"""2026-09-19 送 S5 前置检查: 31 条 axis=1 因子
================================================================================
背景 (9-18 审计结论, `_rank_axis_layer_audit_20260918.py`):
  含 axis=1 的公式 = 需要横截面 pivot 上下文。`_needs_cross_section()` 原用
  **字面子串白名单** ⇒ 写法变体被漏判 ⇒ 落到 `_evaluate_per_stock` (逐股 Series)
  ⇒ `Series.rank(axis=1)` ValueError ⇒ L1594 `except Exception` 静默 return None
  ⇒ **候选被静默丢掉, 从未真正评估过**。
  ⇒ 9-18 修了检测器 ⇒ 本批 31 条**首次**获得组合层读数。

本脚本 = 只读前置检查, 三件事:
  1. 取出 NOVEL_FACTOR_LIBRARY 中 axis=1 的全部条目 (期望 31);
  2. 🔴 逐条验证 `_needs_cross_section` 现在返回 True (路由修复真的生效);
     —— 若有条目仍为 False, 送检无意义, 必须停止并归因;
  3. 审计分母: 括号平衡 / 公式长度 / 名字唯一性 (排除已知死重形态, 见 S5 技能 §6b)。
输出: data/_axis1_31_candidates_20260919.json  (候选清单, 供送检脚本消费)
"""
import io
import json
import os
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r'e:/quant')
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

AX1_ANY = re.compile(r'\baxis\s*=\s*1\b')
AX1_CALL = re.compile(r'\.\s*\w+\s*\([^()]*\baxis\s*=\s*1\b')
AX1_RANK = re.compile(r'\.\s*rank\s*\([^()]*\baxis\s*=\s*1\b')
OUT = ROOT / "data" / "_axis1_31_candidates_20260919.json"


def _balance(t: str) -> bool:
    d = 0
    for ch in t:
        d += (ch == "(") - (ch == ")")
        if d < 0:
            return False
    return d == 0


def main():
    print("=" * 78)
    print("0. 并发实例检查 (S5 技能 §9.1: 同脚本多实例会互相覆盖)")
    print("=" * 78)
    try:
        import psutil
        me = os.getpid()
        hits = []
        for p in psutil.process_iter(['pid', 'cmdline', 'memory_info', 'cpu_times']):
            try:
                cl = ' '.join(p.info['cmdline'] or [])
            except Exception:
                continue
            if p.info['pid'] == me:
                continue
            if any(k in cl for k in ('s5_fullpool_rerun', 's5_joint_filter',
                                     'validate_factor', 's5_axis1')):
                try:
                    rss = round(p.info['memory_info'].rss / 1e9, 2)
                    cpu = round(sum(p.info['cpu_times'][:2]), 1)
                except Exception:
                    rss, cpu = -1, -1
                hits.append((p.info['pid'], rss, cpu, cl[:130]))
        print(f"   同族实例数 = {len(hits)}")
        for h in hits:
            print(f"     pid={h[0]} rss={h[1]}GB cpu={h[2]}s  {h[3]}")
    except ImportError:
        print("   psutil 不可用, 跳过")

    print()
    print("=" * 78)
    print("1. 取 NOVEL_FACTOR_LIBRARY 中 axis=1 条目")
    print("=" * 78)
    import daily_factor_hypothesis as dh
    raw = getattr(dh, 'NOVEL_FACTOR_LIBRARY', [])
    print(f"   库条目数 = {len(raw)}")

    entries = []
    for i, ent in enumerate(raw):
        if not isinstance(ent, dict):
            continue
        f = str(ent.get('formula') or ent.get('expression') or '')
        if not f:
            continue
        if AX1_ANY.search(f):
            entries.append((str(ent.get('id') or ent.get('name') or i), ent, f))
    print(f"   含 axis=1 = {len(entries)}")
    n_rank = sum(1 for _, _, f in entries if AX1_RANK.search(f))
    print(f"   其中 .rank 形式 = {n_rank}")

    print()
    print("=" * 78)
    print("2. 🔴 路由自证: _needs_cross_section 逐条实测 (必须全 True)")
    print("=" * 78)
    import s5_joint_filter as S
    needs = S.S5JointFilter._needs_cross_section
    dummy = object()

    rows = []
    n_true = n_false = n_err = 0
    for key, ent, f in entries:
        name = str(ent.get('name') or key)
        try:
            det = bool(needs(dummy, f))
            err = ""
        except Exception as e:
            det, err = None, f"{type(e).__name__}: {e}"[:80]
        if det is True:
            n_true += 1
        elif det is False:
            n_false += 1
        else:
            n_err += 1

        # 分母审计 (S5 技能 §6b: 排除与自变量无关的死重)
        bal = _balance(f)
        dead = None
        if not bal:
            dead = "bracket_imbalance"
        elif len(f) >= 200:
            dead = f"len_is_cap:{len(f)}"
        row = {
            "factor_name": name,
            "novel_id": key,
            "formula": f,
            "len": len(f),
            "balanced": bal,
            "needs_cross_section": det,
            "detect_error": err,
            "axis1_calls": sorted(set(AX1_CALL.findall(f)))[:4],
            "has_rank_ax1": bool(AX1_RANK.search(f)),
            "dead_weight": dead,
        }
        rows.append(row)

    print(f"   needs=True  {n_true}")
    print(f"   needs=False {n_false}   <-- 必须为 0; 非 0 则路由修复未覆盖")
    print(f"   needs=ERR   {n_err}")
    if n_false:
        print("   🔴 仍未覆盖的条目:")
        for r in rows:
            if r["needs_cross_section"] is False:
                print(f"     - {r['factor_name']}  calls={r['axis1_calls']}")
    if n_err:
        for r in rows:
            if r["needs_cross_section"] is None:
                print(f"     ERR {r['factor_name']}: {r['detect_error']}")

    print()
    print("=" * 78)
    print("3. 分母审计 / 名字唯一性")
    print("=" * 78)
    names = [r["factor_name"] for r in rows]
    dup = {n for n in names if names.count(n) > 1}
    print(f"   名字唯一 = {len(set(names))}/{len(names)}   重复 = {sorted(dup)}")
    n_dead = sum(1 for r in rows if r["dead_weight"])
    print(f"   死重形态 = {n_dead}")
    for r in rows:
        if r["dead_weight"]:
            print(f"     - {r['factor_name']} ({r['dead_weight']}, len={r['len']})")
    lens = sorted(r["len"] for r in rows)
    if lens:
        print(f"   公式长度 min/中位/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
        print(f"   长度 == 200 的条数 = {sum(1 for r in rows if r['len'] == 200)}"
              f"   == 500 的条数 = {sum(1 for r in rows if r['len'] == 500)}")

    print()
    print("   -- 31 条清单 --")
    for i, r in enumerate(rows, 1):
        flag = "RANK" if r["has_rank_ax1"] else "    "
        print(f"   {i:>2}. [{flag}] {r['factor_name'][:44]:<44} "
              f"len={r['len']:>4} needs={r['needs_cross_section']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump({
            "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "scripts/daily_factor_hypothesis.py::NOVEL_FACTOR_LIBRARY",
            "n_total_lib": len(raw),
            "n_axis1": len(rows),
            "n_rank_axis1": n_rank,
            "n_needs_true": n_true,
            "n_needs_false": n_false,
            "n_dead_weight": n_dead,
            "gate_ok": (n_false == 0 and n_err == 0),
            "candidates": rows,
        }, fh, ensure_ascii=False, indent=2)
    print()
    print("written:", OUT)
    print("DONE")


if __name__ == "__main__":
    main()
