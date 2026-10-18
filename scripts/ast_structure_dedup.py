# -*- coding: utf-8 -*-
"""P-20260915-005: AST 级结构去重 影子 CLI (零裁决 —— 只输出读数与标签)

统一入口, 避免与 research/factor_alchemy/structure_dedup.py 同名
(🔴 同名模块铁律: run_v4_pipeline 把 scripts/ 插入 sys.path[0],
 同名会导致 ralph_loop 的 from <name> import 解析到 scripts 版 → 循环导入)。

用法:
  # 库内两两自扫 (主入口): 产出 data/structure_dedup_report.json
  python scripts/ast_structure_dedup.py --scan

  # 单条表达式检查 (对照库内最大公共子树)
  python scripts/ast_structure_dedup.py --check "close.rolling(20).mean()"

  # 批量: 从文件读表达式 (每行一条)
  python scripts/ast_structure_dedup.py --check-file exprs.txt

  # 覆盖阈值/抽象层
  python scripts/ast_structure_dedup.py --scan --threshold 10
  python scripts/ast_structure_dedup.py --scan --abstraction field

回滚: 删除本文件 + research/factor_alchemy/structure_dedup.py
      + 撤销 ralph_loop 钩子; shadow 模式零裁决。
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "research" / "factor_alchemy"))

from structure_dedup import (  # noqa: E402
    ASTStructureDeductor,
    DEFAULT_THRESHOLD,
    LIB_STATE_PATH,
    REPORT_PATH,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="AST 级结构去重 (影子)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", action="store_true", help="库内两两自扫 (主入口)")
    g.add_argument("--check", type=str, help="检查单条表达式")
    g.add_argument("--check-file", type=str, help="批量检查 (每行一条表达式)")

    ap.add_argument("--threshold", type=int, default=None,
                    help=f"最大公共子树节点数阈值 (默认 {DEFAULT_THRESHOLD}, env FUXI_STRUCT_DEDUP_THRESHOLD)")
    ap.add_argument("--abstraction", type=str, default=None,
                    choices=["literal", "field", "full"], help="enforce 判重所用抽象层")
    ap.add_argument("--min-size", type=int, default=None, help="入索引的最小子树节点数 (默认 3)")
    ap.add_argument("--max-fanout", type=int, default=None, help="单签名扇出保护 (默认 400)")
    ap.add_argument("--top", type=int, default=30, help="报告内 flagged 对展示条数")
    ap.add_argument("--state", type=str, default=str(LIB_STATE_PATH), help="因子库状态文件")
    ap.add_argument("--json", action="store_true", help="额外打印完整 JSON")
    args = ap.parse_args()

    d = ASTStructureDeductor(
        threshold=args.threshold,
        abstraction=args.abstraction,
        min_size=args.min_size,
        max_fanout=args.max_fanout,
    )

    if args.scan:
        rep = d.run_scan(lib_state_path=Path(args.state), persist=True, top_n=args.top)
        print(rep["readout"])
        print("\n【最大公共子树节点数分布 (全部有共享子树的因子对)】")
        for ab, dd in rep["max_common_size_distribution"].items():
            print(f"  {ab:8s} 对数 {dd['n_pairs_with_any_shared']:>7d} | "
                  f"p50 {dd['p50']:>3d} | p90 {dd['p90']:>3d} | p99 {dd['p99']:>3d} | "
                  f"p99.9 {dd['p999']:>3d} | max {dd['max']:>3d}")
        print("\n【阈值为 %d 时超阈值对数】" % rep["threshold"])
        for ab, n in rep["summary"]["n_pairs_ge_threshold"].items():
            print(f"  {ab:8s} {n:>6d} 对")
        x = rep["cross_read"]
        if x.get("corr_available"):
            print(f"\n【交叉读数 (数值口径口径覆盖 {x['corr_universe']} 因子)】")
            print(f"  ① 结构重复 且 |corr|<0.30 (数值去重盲区): {x['n_structural_dup_numeric_uncorr']} 对")
            for e in x["structural_dup_numeric_uncorrelated"][:8]:
                print(f"     - {e['factor_a']} × {e['factor_b']} "
                      f"(公共子树 {e['max_common_size']} 节点, |corr|={e['abs_corr']}, "
                      f"符号{'同' if e['sign_same'] else '异'})")
            print(f"  ② |corr|>0.85 但结构零重叠 (结构层盲区): {x['n_numeric_dup_structurally_clean']} 对")
            for e in x["numeric_dup_structurally_clean"][:8]:
                print(f"     - {e['factor_a']} × {e['factor_b']} (|corr|={e['abs_corr']})")
        else:
            print(f"\n[交叉读数不可用] {x.get('error')}")
        if rep["notes"]:
            print(f"\n【诊断注记 {len(rep['notes'])} 条】")
            for n in rep["notes"][:8]:
                print(f"  {n}")
        print(f"\n产物: {REPORT_PATH}")
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    # ── 单条 / 批量检查 ──
    _stp = Path(args.state)
    if not _stp.exists():
        print(f"✗ 因子库状态文件不存在: {_stp}\n"
              f"  提示: 用 --state 指定, 默认 data/library_orthogonality_state.json")
        return 2
    st = json.loads(_stp.read_text(encoding="utf-8"))
    d.index_library(st.get("factors", []) or [])
    print(f"[StructDedup] 库已索引 {len(d.factors)} 因子, 阈值 {d.threshold}, "
          f"抽象层 {d.abstraction}, mode={d.mode}")

    exprs = []
    if args.check:
        exprs = [args.check]
    else:
        _cf = Path(args.check_file)
        if not _cf.exists():
            print(f"✗ 表达式文件不存在: {_cf}  (当前工作目录 {Path.cwd()})")
            return 2
        exprs = [ln.strip() for ln in _cf.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    if not exprs:
        print("✗ 无待检查表达式 (--check 或 --check-file 需提供内容)")
        return 2

    for e in exprs:
        r = d.check_candidate(e)
        if not r.get("ok"):
            print(f"\n✗ 解析失败: {e[:100]}  ({r.get('reason')})")
            continue
        pa = r["per_abstraction"]
        print(f"\n表达式: {e[:120]}")
        print(f"  节点 {r['n_nodes']} | 数值常量 {r['n_free_args']} (ratio {r['free_args_ratio']}) | "
              f"变量 {r['n_unique_vars']} (ratio {r['unique_vars_ratio']}) | 符号 {r['sign']:+d}")
        for ab in ("literal", "field", "full"):
            v = pa[ab]
            flag = "⚠️超阈值" if v["max_common_size"] > r["threshold"] else "ok"
            print(f"  [{ab:8s}] 最大公共子树 {v['max_common_size']:>3d} 节点 {flag} | "
                  f"命中 {v['matched_factor'] or '-'} | 符号{'同' if v['sign_same'] else '异'}")
        print(f"  三条件: cond1(size≤{r['threshold']})={r['cond1_size_le_thresh']} "
              f"cond2(free_args<0.5)={r['cond2_free_args']} "
              f"cond3(unique_vars<0.5)={r['cond3_unique_vars']} | "
              f"结构重复={r['is_structural_dup']} | would_reject={r['would_reject']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
