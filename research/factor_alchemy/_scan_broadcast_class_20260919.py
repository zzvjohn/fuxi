# -*- coding: utf-8 -*-
"""全库静态扫描: 「axis=1 归约结果被 DataFrame 按列广播」同族 (9-18 只修了 5 条命名条目)。

背景:
  9-18 的 P-20260918-022 标题已写明修了 **5 条广播对齐错**
  (`novel_040/070/073/099/203`, DataFrame−Series 按列对齐 ⇒ 全 NaN)。
  2026-09-19 送 S5 首次真评估时, `co_movement_intensity` (**novel_026**) 实测
  `🧹 常数因子拦截: 有效值仅 1 个离散水平` ⇒ 它是**同族的第 6 个实例, 从未被修**。
  ⇒ 本脚本做**同族穷举**: 在 NOVEL_FACTOR_LIBRARY 全量上静态找出所有候选实例。

🔴 判据纪律 (不得违反):
  - 这是**静态启发式**, 输出的是**待验证假设**, **不是裁决**。
    不得把命中当「已确认缺陷」写进结论 (锚在实现 = 需实跑读数)。
  - 判据锚在**语法结构** (归约方法 + 相邻二元算子 + 缺 `axis=0`), **不锚命名**。
  - 必须同报**误报与漏报**风险; 必须给出可复核的逐条证据 (原始片段)。

命中定义 (三个条件同时成立):
  C1 语句内出现 `<obj>.<mean|std|sum|median|var|max|min>(axis=1)` ⇒ 归约成**日期索引 Series**;
  C2 该归约表达式**紧邻二元算子** `* / + -` 的右侧 (即被当作 DataFrame 的操作数);
  C3 该**语句**内**不含** `axis=0` (含则以正确的按行对齐方式暴露, 不属本族)。
"""
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path(r'e:/quant')
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

OUT = ROOT / "data" / "_axis1_broadcast_class_scan_20260919.json"

# C1: 归约方法 (rank 不算 —— rank 保持 DataFrame 形状, 不是归约成 Series)
RED = r"(?:mean|std|sum|median|var|max|min|sem|prod)"
# C2: 二元算子紧邻归约表达式右侧
PAT_BINOP = re.compile(r"[*\/+\-]\s*\(?\s*[A-Za-z_]\w*\.\s*" + RED +
                       r"\s*\(\s*axis\s*=\s*1\s*\)")
# 也覆盖 <obj>.<method>(...).<method2>() 型 (如 close_p.mean(axis=1).pct_change())
PAT_RED = re.compile(r"[A-Za-z_]\w*\.\s*" + RED + r"\s*\(\s*axis\s*=\s*1\s*\)")
KNOWN_FIXED = {"novel_040", "novel_070", "novel_073", "novel_099", "novel_203"}


def statements(f):
    """按行/分号切语句 (公式多为多行 DSL)。"""
    parts = re.split(r"[\n;]", f or "")
    return [p.strip() for p in parts if p.strip()]


def scan_one(formula):
    for st in statements(formula):
        if "axis=0" in st.replace(" ", "") or "axis = 0" in st:
            continue                       # C3 不成立
        if not PAT_RED.search(st):
            continue                       # C1 不成立
        m = PAT_BINOP.search(st)
        if m:
            return {"statement": st[:240], "evidence": m.group(0)[:80]}
    return None


def main():
    import daily_factor_hypothesis as dh
    lib = getattr(dh, "NOVEL_FACTOR_LIBRARY", [])
    print("[LIB] NOVEL_FACTOR_LIBRARY n=%d" % len(lib))

    hits, n_ax1 = [], 0
    for it in lib:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("factor_name") or it.get("id")
        nid = it.get("novel_id") or it.get("id")
        f = it.get("formula") or it.get("expression") or ""
        if not f:
            continue
        if re.search(r"\baxis\s*=\s*1\b", f):
            n_ax1 += 1
        r = scan_one(f)
        if r:
            hits.append({"name": name, "novel_id": nid, "len": len(f),
                         "in_known_fixed_set": nid in KNOWN_FIXED,
                         "formula": f[:400], **r})

    print("[SCAN] 含 axis=1 的条目 = %d / %d" % (n_ax1, len(lib)))
    print("[SCAN] 命中本族(候选) = %d" % len(hits))
    print()
    print("=== 命中明细 (🔴 待验证假设, 非裁决) ===")
    for h in sorted(hits, key=lambda x: str(x["novel_id"])):
        tag = "[已在9-18修复集]" if h["in_known_fixed_set"] else "[新候选]"
        print("%s %-12s %-38s len=%3d" % (tag, h["novel_id"], str(h["name"])[:38], h["len"]))
        print("      证据算子片段: %s" % h["evidence"].replace("\n", " "))
        print("      语句: %s" % h["statement"].replace("\n", " | ")[:150])
    new = [h for h in hits if not h["in_known_fixed_set"]]
    print()
    print("[SUMMARY] 本族候选 = %d (其中 %d 属 9-18 已修命名集, **%d 为新候选**) "
          "⇒ 9-18 的 5 条命名修复**未穷举该类**"
          % (len(hits), len(hits) - len(new), len(new)))

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps({
        "scanned_at": "2026-09-19",
        "library": "daily_factor_hypothesis.NOVEL_FACTOR_LIBRARY",
        "n_lib": len(lib), "n_axis1": n_ax1, "n_hits": len(hits),
        "n_new_candidates": len(new),
        "known_fixed_set": sorted(KNOWN_FIXED),
        "criterion": "C1 归约(axis=1)→日期索引Series; C2 紧邻二元算子右侧; C3 语句内无 axis=0",
        "caveat": "静态启发式, 产出为待验证假设; 须实跑读数确认 (锚在实现)",
        "hits": hits,
    }, ensure_ascii=False, indent=1))
    print("输出: %s" % OUT)


if __name__ == "__main__":
    main()
