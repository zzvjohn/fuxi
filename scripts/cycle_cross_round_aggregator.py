"""伏羲 v1.1 闭环跨轮聚合器（只读诊断）。

把多轮闭环研究的候选做跨轮聚合，回答三个问题：
  1. 因子相似度是否过高 —— structural_family 跨轮重叠度（同族在多轮重现）；
  2. 是否过度集中某小领域 —— mechanism 分布集中度（Top-N 占比 + 有效机制数）；
  3. 多轮结论是否可信 —— Holm 多重检验显著命中数 + 唯一公式率。

只读 cycle_ledger.json + 各 run 研究报告 + falsified_motifs.json，产出一份
聚合报告（JSON），不改动任何被读资产。

用法：
  python scripts/cycle_cross_round_aggregator.py            # 打印摘要 + 写报告
  python scripts/cycle_cross_round_aggregator.py --json      # 只打印完整 JSON
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
LEDGER = WORKSPACE / "data" / "fuxi_unified" / "cycle_ledger.json"
RUNS = WORKSPACE / "data" / "fuxi_unified" / "runs"
FALSIFIED_MOTIFS = WORKSPACE / "data" / "fuxi_unified" / "falsified_motifs.json"
OUT = WORKSPACE / "data" / "fuxi_unified" / "cycle_cross_round_aggregation.json"


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _report_for_run(run_id):
    """由 run_id 定位研究报告路径。"""
    if not run_id:
        return None
    receipt = RUNS / run_id / "research" / "RECEIPT.json"
    r = _load_json(receipt)
    if not r:
        return None
    return r.get("report_path")


def _herfindahl(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0
    hhi = sum((v / total) ** 2 for v in counter.values())
    return round(hhi, 4), round(1.0 / hhi if hhi else 0.0, 2)


def aggregate():
    ledger = _load_json(LEDGER, {"cycles": []})
    cycles = ledger.get("cycles", [])

    rounds = []
    family_by_round = []          # 每轮出现的 structural_family 集合
    mechanism_counter = Counter()
    family_counter = Counter()
    family_rounds = defaultdict(set)  # family -> 出现的轮次集合
    holm_hits = 0
    holm_tested = 0
    total_attempted = 0
    total_unique_formula = 0
    total_valid = 0
    rejected_counter = Counter()

    for i, cycle in enumerate(cycles):
        phases = cycle.get("phases", {})
        research = phases.get("research", {})
        if research.get("state") != "completed":
            continue
        run_id = research.get("run_id")
        report_path = _report_for_run(run_id)
        if not report_path or not Path(report_path).is_file():
            continue
        report = _load_json(report_path)
        if not report:
            continue

        fams = set()
        for c in report.get("candidate_results", []):
            if not isinstance(c, dict):
                continue
            sf = c.get("structural_family")
            if sf:
                fams.add(sf)
                family_counter[sf] += 1
                family_rounds[sf].add(i)
            mech = None
            # mechanism 从 pareto_archive 或 candidate hypothesis 反查
        for e in report.get("pareto_archive", []):
            if isinstance(e, dict) and e.get("mechanism"):
                mechanism_counter[e["mechanism"]] += 1
        family_by_round.append(fams)

        msd = report.get("multiple_search_diagnostics", {})
        if isinstance(msd, dict):
            for v in msd.values():
                if isinstance(v, dict):
                    holm_tested += 1
                    if v.get("reject_null"):
                        holm_hits += 1

        total_attempted += int(report.get("attempted_candidates", 0) or 0)
        total_unique_formula += int(report.get("unique_formula_count", 0) or 0)
        total_valid += int(report.get("valid_descriptive_factors", 0) or 0)

        rounds.append({
            "round_index": i,
            "run_id": run_id,
            "attempted_candidates": report.get("attempted_candidates"),
            "valid_descriptive_factors": report.get("valid_descriptive_factors"),
            "unique_formula_count": report.get("unique_formula_count"),
            "structural_families": len(fams),
        })

    # 跨轮相似度：同族在多轮重现
    n_rounds = len(rounds)
    recurring_families = {f: sorted(rs) for f, rs in family_rounds.items() if len(rs) > 1}
    cross_round_overlap_rate = round(len(recurring_families) / max(len(family_counter), 1), 4)

    # mechanism 集中度
    mech_hhi, mech_effective = _herfindahl(mechanism_counter)
    top_mech = mechanism_counter.most_common(5)
    top3_share = round(
        sum(v for _, v in mechanism_counter.most_common(3)) / max(sum(mechanism_counter.values()), 1), 4)

    # falsified motifs tally
    falsified = _load_json(FALSIFIED_MOTIFS, {"motifs": []})
    falsified_motifs = falsified.get("motifs", [])
    falsified_mech = Counter(m.get("mechanism") for m in falsified_motifs if m.get("mechanism"))

    return {
        "schema_version": "fuxi.cycle_cross_round.1",
        "rounds_completed": n_rounds,
        "total_attempted_candidates": total_attempted,
        "total_valid_descriptive_factors": total_valid,
        "unique_formula_rate": round(total_unique_formula / max(total_attempted, 1), 4),
        "total_unique_structural_families": len(family_counter),
        "cross_round_recurring_families": len(recurring_families),
        "cross_round_overlap_rate": cross_round_overlap_rate,
        "recurring_families": [
            {"structural_family": f, "rounds": rs, "count": family_counter[f]}
            for f, rs in sorted(recurring_families.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ],
        "mechanism_distribution": dict(mechanism_counter.most_common()),
        "mechanism_hhi": mech_hhi,
        "mechanism_effective_count": mech_effective,
        "mechanism_top3_share": top3_share,
        "mechanism_top5": [{"mechanism": m, "count": v} for m, v in top_mech],
        "holm_tested": holm_tested,
        "holm_significant_hits": holm_hits,
        "falsified_motif_count": len(falsified_motifs),
        "falsified_mechanism_distribution": dict(falsified_mech.most_common()),
        "rounds": rounds,
    }


def _summarize(r):
    print("=== 伏羲闭环跨轮聚合摘要 ===")
    print(f"完成轮数: {r['rounds_completed']}  总候选: {r['total_attempted_candidates']}  "
          f"有效描述因子: {r['total_valid_descriptive_factors']}")
    print(f"唯一公式率: {r['unique_formula_rate']:.2%}  "
          f"(低=重复公式多，相似度风险高)")
    print(f"唯一结构族: {r['total_unique_structural_families']}  "
          f"跨轮重现族: {r['cross_round_recurring_families']}  "
          f"跨轮重叠率: {r['cross_round_overlap_rate']:.2%}")
    print(f"机制集中度 HHI: {r['mechanism_hhi']}  有效机制数: {r['mechanism_effective_count']}  "
          f"Top3 占比: {r['mechanism_top3_share']:.2%}")
    print(f"Holm 检验: {r['holm_significant_hits']}/{r['holm_tested']} 显著  "
          f"已证伪 motif: {r['falsified_motif_count']}")
    if r["mechanism_top5"]:
        print("Top 机制:", ", ".join(f"{m['mechanism']}:{m['count']}" for m in r["mechanism_top5"]))
    if r["recurring_families"]:
        print("跨轮重现族（相似度信号）:")
        for fam in r["recurring_families"][:10]:
            print(f"  {fam['structural_family'][:16]}… rounds={fam['rounds']} count={fam['count']}")


def main():
    parser = argparse.ArgumentParser(description="闭环跨轮聚合诊断")
    parser.add_argument("--json", action="store_true", help="只打印完整 JSON 报告")
    parser.add_argument("--out", default=str(OUT), help="聚合报告输出路径")
    args = parser.parse_args()

    report = aggregate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp." + str(int(__import__("time").time())))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tmp.replace(out)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _summarize(report)
        print(f"\n报告已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
