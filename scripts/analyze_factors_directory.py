# -*- coding: utf-8 -*-
"""factors.directory 采集结果分析: 分类/公式形态/与 854 库重叠 (2026-09-05)"""
import json
import os
import re
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

def main():
    with open(os.path.join(DATA, "factors_directory_factors.json"), encoding="utf-8") as f:
        fd = json.load(f)
    facs = fd["factors"]
    print("总因子数:", len(facs))

    # 1. 分类分布
    cats = Counter(x["category"] for x in facs)
    print("分类分布:")
    for k, v in cats.most_common():
        print("  %-15s %d" % (k, v))

    # 2. 公式形态
    n_with_latex = sum(1 for x in facs if any(fm["latex"].strip() for fm in x["formulas"]))
    print("带 latex 公式:", n_with_latex, "/", len(facs))
    latex_ops = Counter()
    for x in facs:
        for fm in x["formulas"]:
            for op in re.findall(r"[A-Z]{2,}(?=\()", fm["latex"]):
                latex_ops[op] += 1
    print("latex 算子分布 top20:", latex_ops.most_common(20))

    # 3. 与 854 库重叠 (name 关键词)
    lib = json.load(open(os.path.join(DATA, "library_orthogonality_state.json"), encoding="utf-8"))
    lib_names = " ".join(fx.get("name", "") for fx in lib["factors"])
    lib_exprs = " ".join(fx.get("expression", "") for fx in lib["factors"])
    overlap_name = 0
    for x in facs:
        slug = x["slug"].replace("-", " ")
        # 关键词粗匹配
        if slug.split() and any(w in lib_names for w in slug.split() if len(w) > 4):
            overlap_name += 1
    print("slug 关键词与 854 库 name 粗重叠:", overlap_name, "/", len(facs))

    # 4. 输出候选清单 (name_zh + category + 公式样本)
    sample = []
    for x in facs[:10]:
        sample.append({"name_zh": x["name_zh"], "category": x["category"],
                       "formula": x["formulas"][0]["latex"] if x["formulas"] else ""})
    print("样本:")
    for s in sample:
        print("  [%s] %s :: %s" % (s["category"], s["name_zh"], s["formula"][:70]))


if __name__ == "__main__":
    main()
