#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全库因子 eval 健康诊断 (P-20260901 稳定性回归工具)

对 passed_factor_pool.csv 全库因子逐一 _eval_formula, 分类统计 eval 失败原因。
用途: 5 轮完整运行前的干净基线确认 + 运行中/后的挂起因子排查。

用法:
  python scripts/diagnose_library_eval.py            # 全库诊断
  python scripts/diagnose_library_eval.py --top 50   # 只诊断 |icir| top 50
"""
import sys, re, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "factor_alchemy"))

import pandas as pd
from factor_ic_computer import FactorICComputer
from factor_ic_computer import (
    MONEYFLOW_FIELDS, MARGIN_FIELDS, TOP_LIST_FIELDS, TOP_INST_FIELDS,
    NORTH_INST_FIELDS, NORTH_FIELDS, ANALYST_FIELDS, INDUSTRY_PEER_FIELDS,
    INDUSTRY_CODE_FIELDS, VALUATION_FIELDS, FINA_FIELDS,
)

def build_known_fields():
    """构造合法字段名全集 (与 _eval_formula field_cols + suffix_map 对齐)。"""
    base = ["close", "open", "high", "low", "volume", "amount",
            "overnight", "amplitude", "returns", "turnover", "hl_ratio"]
    known = set(base)
    for grp in [MONEYFLOW_FIELDS, MARGIN_FIELDS, TOP_LIST_FIELDS, TOP_INST_FIELDS,
                NORTH_INST_FIELDS, NORTH_FIELDS, ANALYST_FIELDS, INDUSTRY_PEER_FIELDS,
                INDUSTRY_CODE_FIELDS, VALUATION_FIELDS, FINA_FIELDS]:
        known.update(grp)
    # 带 _p 后缀别名
    known |= {f"{f}_p" for f in known}
    return known


def main():
    top = None
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    sample = None
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])

    pool_path = ROOT / "data" / "passed_factor_pool.csv"
    df = pd.read_csv(pool_path)
    print(f"因子池: {len(df)} 个因子")

    # 有公式的因子
    df = df[df["formula"].notna() & (df["formula"].astype(str).str.strip() != "")]
    if top:
        df = df.reindex(df["icir"].abs().sort_values(ascending=False).index).head(top)
    print(f"有公式可评估: {len(df)} 个{' (top %d)' % top if top else ''}\n")

    comp = FactorICComputer(sample_stocks=sample)  # None=全市场; int=抽样降本
    comp._load_data()

    known_fields = build_known_fields()
    # 已知但当前数据面加载不到的字段 (用于区分"字段缺失" vs "函数缺失")
    # 直接从 comp 内部 wide 环境探测: 用 _get_wide_field 试每个已知字段
    available = set()
    for f in known_fields:
        base = f[:-2] if f.endswith("_p") else f
        if comp._get_wide_field(base) is not None:
            available.add(f)
            available.add(base)
    print(f"已知字段 {len(known_fields)} 个, 当前数据面可用 {len(available)} 个\n")

    results = []
    for _, row in df.iterrows():
        name = row["name"]
        formula = str(row["formula"])
        try:
            fdf = comp._eval_formula(formula, name)
            if fdf is None or len(fdf) < 100:
                results.append((name, formula, "EMPTY", "返回空/样本不足", ""))
            else:
                results.append((name, formula, "OK", "", ""))
        except Exception as e:
            results.append((name, formula, type(e).__name__, str(e)[:120], ""))

    ok = [r for r in results if r[2] == "OK"]
    fail = [r for r in results if r[2] != "OK"]

    # 分类
    cat = Counter()
    detail = defaultdict(list)
    for name, formula, etype, emsg, _ in fail:
        if etype == "NameError":
            m = re.search(r"name '([^']+)' is not defined", emsg)
            varname = m.group(1) if m else "?"
            if varname in known_fields and varname not in available:
                key = "NameError-字段缺失(数据面)"
            elif varname in known_fields:
                key = "NameError-字段存在但未注入"
            else:
                key = f"NameError-未定义函数/变量"
                varname = varname
            cat[key] += 1
            detail[key].append((name, varname))
        elif etype == "SyntaxError":
            cat["SyntaxError-语法"] += 1
            detail["SyntaxError-语法"].append((name, emsg[:60]))
        elif etype == "TypeError":
            cat["TypeError-类型"] += 1
            detail["TypeError-类型"].append((name, emsg[:60]))
        elif etype == "EMPTY":
            cat["EMPTY-空结果"] += 1
            detail["EMPTY-空结果"].append((name, emsg[:60]))
        else:
            cat[f"其他-{etype}"] += 1
            detail[f"其他-{etype}"].append((name, emsg[:60]))

    print("=" * 70)
    print(f"  全库 eval 诊断结果")
    print("=" * 70)
    print(f"  总计: {len(results)} | 成功: {len(ok)} | 失败: {len(fail)}")
    print(f"\n  失败分类:")
    for k, c in cat.most_common():
        print(f"    {k:32s} {c}")
        for name, extra in detail[k][:15]:
            print(f"        - {name[:50]}  [{extra[:40]}]")
        if len(detail[k]) > 15:
            print(f"        ... 共 {len(detail[k])} 个")

    # 落盘
    out = {
        "total": len(results), "ok": len(ok), "fail": len(fail),
        "categories": {k: c for k, c in cat.items()},
        "failed_factors": [
            {"name": n, "type": t, "msg": m} for n, f, t, m, _ in fail
        ],
    }
    out_path = ROOT / "data" / "library_eval_diagnose.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果 → {out_path}")


if __name__ == "__main__":
    main()
