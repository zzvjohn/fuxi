"""第 0 层：archetype 映射验证（只读，不写回源文件）。

对 stage1_factor_proposals.json 的 2776 条候选按 paradigm_archetype_map.json 做
archetype 归并，输出覆盖率 + 每个 archetype 条数 + top paradigm 映射抽查，
用于审计映射质量。select_hypotheses 运行时用同样的逻辑（本脚本仅验证/审计）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuxi_v1.archetypes import load_archetypes, archetype_of

WORKSPACE = Path(__file__).resolve().parents[1]
PROPOSALS = WORKSPACE / "data" / "stage1_factor_proposals.json"


def main():
    archetypes, unclassified = load_archetypes()
    d = json.load(open(PROPOSALS, encoding="utf-8"))
    ps = d.get("proposals", d if isinstance(d, list) else [])
    dist = Counter()
    paradigm_to_archetype = {}
    for p in ps:
        aid = archetype_of(p, archetypes, unclassified)
        dist[aid] += 1
        para = (p.get("paradigm") or "").strip() or "__none__"
        paradigm_to_archetype.setdefault(para, Counter())[aid] += 1

    print("=== 总提案数:", len(ps), "===")
    print("=== archetype 分布（24 类 + unclassified）===")
    for aid, cnt in dist.most_common():
        name = next((a["zh"] for a in archetypes if a["id"] == aid), aid)
        print(f"{cnt:5d}  {aid:16s} {name}")
    print()
    un = dist.get(unclassified, 0)
    print(f"=== unclassified（无法归类）: {un} 条，占比 {un/len(ps)*100:.1f}% ===")
    print()
    # top 30 paradigm 的映射抽查
    print("=== top 30 paradigm 映射抽查 ===")
    para_freq = Counter((p.get("paradigm") or "").strip() or "__none__" for p in ps)
    for para, _ in para_freq.most_common(30):
        arch = paradigm_to_archetype[para].most_common(1)[0][0]
        print(f"{para[:40]:42s} -> {arch}")


if __name__ == "__main__":
    main()
