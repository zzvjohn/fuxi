"""Horizon 敏感性扫描：6 formal JQ pass 因子 + 3 baseline 在 horizon=5/20/60 下的 E1 全套。

目的（用户拍板，2026-10-20）：
  用真实数据回答"日频 label 测不出单调/弱 IC，改月频(20日)是否真能提升预测力"，
  据此决定是否值得另起 20 日 horizon 的完整 GA campaign 重跑。

方法：
  * 复用 unified_153d8e 真实快照 + market（含 bars），不触网；
  * 用 fuxi_v1.datasets.materialize_labels 生成不同 horizon 的 label（execution_contract_hash
    不变，仅 horizon_sessions 变），再走 fuxi_v1.evaluation.factor_statistics 得到 E1 描述统计；
  * 指标：Rank IC 均值 / ICIR(=mean/sd，非年化) / +IC% / 有效截面数；
  * 固定 research 窗口 start=2021-05-18，end 用 research_end=2025-12-31（尾部因 label 未成熟
    自然缺样本，随 horizon 拉长尾部缺失增多，作为 horizon 效应的固有代价如实报告）。

产出：
  data/system_evolution/calibration/horizon_sensitivity.json
  reports/fuxi_horizon_sensitivity_20261020.md
"""
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuxi_v1.contracts import timestamp, load_json
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.datasets import materialize_labels
from fuxi_v1.evaluation import factor_statistics, DEFAULT_POLICY
from fuxi_v1.dsl import evaluate_ast

INPUT = "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e/research/campaign_153d8e0779d74d49e2f3606e/input.json"
POLICY = "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e/research/campaign_153d8e0779d74d49e2f3606e/policy.json"
BUNDLE_DIRS = [
    "data/fuxi_unified/runs/jq_bundles_20261019",
    "data/fuxi_unified/runs/reeval_jq_20261019",
]
HORIZONS = [5, 20, 60]


def _ast(op, *args, **kw):
    return {"op": op, "args": list(args), **kw}


def _field(name):
    return {"field": name}


def baseline_asts():
    """三个经典 A 股单因子 baseline（覆盖反转/动量/低波动三大风格）。"""
    return {
        "baseline_5d_reversal": _ast("neg", _ast("rank", _ast("return", _field("close"), window=5))),
        "baseline_20d_momentum": _ast("rank", _ast("return", _field("close"), window=20)),
        "baseline_20d_lowvol": _ast("neg", _ast("rank", _ast("std", _field("close"), window=20))),
    }


def formal_jq_asts():
    """6 个 formal jq_candidate_allowed=true 因子的 canonical_ast。"""
    out = {}
    for rundir in BUNDLE_DIRS:
        bs_path = os.path.join(rundir, "BUILD_SUMMARY.json")
        if not os.path.exists(bs_path):
            continue
        for b in load_json(bs_path):
            if not b.get("jq_candidate_allowed"):
                continue
            c = load_json(os.path.join(b["bundle"], "candidate.json"))
            cid = c["candidate_id"][:12]
            out[f"formal_{cid}"] = c["canonical_ast"]
    return out


def run_one(scores, snapshot, labels, clock, start, end):
    selected = {d: v for d, v in scores.items() if start <= d <= end}
    e1 = factor_statistics(selected, snapshot, labels, DEFAULT_POLICY, clock)
    diag = e1["diagnostics"]
    sections = diag["sections"]
    rank_ics = [r["rank_ic"] for r in sections if r["rank_ic"] is not None]
    pos = sum(1 for x in rank_ics if x > 0)
    spearman = diag["spearman"]
    return {
        "mean_rank_ic": spearman["mean"],
        "rank_ic_sd": spearman["sample_sd"],
        "icir": spearman["icir_nonannualized"],
        "positive_ic_fraction": pos / len(rank_ics) if rank_ics else None,
        "valid_sections": diag["valid_section_count"],
        "n_pairs": e1.get("n_pairs"),
    }


def main():
    payload = load_json(INPUT)
    snapshot = Snapshot(payload["snapshot"])
    market = payload["market"]
    clock = payload["research_clock"]  # factor_statistics 内部会 timestamp() 转换
    policy = load_json(POLICY)
    start = policy["request"]["research_start"]
    end = policy["request"]["research_end"]

    factors = {}
    factors.update(formal_jq_asts())
    factors.update(baseline_asts())
    print(f"因子数: {len(factors)} (formal={sum(1 for k in factors if k.startswith('formal_'))}, "
          f"baseline={sum(1 for k in factors if k.startswith('baseline_'))})")

    # 预计算各因子 score（与 horizon 无关，只与 snapshot 有关）
    scores_by_factor = {}
    for name, ast in factors.items():
        scores_by_factor[name] = evaluate_ast(ast, snapshot)
        print(f"  evaluated {name}")

    results = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        labels = Labels(materialize_labels(snapshot, market, horizon=h), snapshot)
        for name in factors:
            results[h][name] = run_one(scores_by_factor[name], snapshot, labels, clock, start, end)
        print(f"horizon={h}: 完成")

    # 汇总输出
    out = {
        "schema_version": "fuxi.horizon_sensitivity.1",
        "snapshot_hash": snapshot.hash,
        "research_start": start,
        "research_end": end,
        "horizons": HORIZONS,
        "factors": {h: results[h] for h in HORIZONS},
    }
    os.makedirs("data/system_evolution/calibration", exist_ok=True)
    with open("data/system_evolution/calibration/horizon_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写入 data/system_evolution/calibration/horizon_sensitivity.json")

    # 控制台矩阵
    print("\n=== Rank IC 均值矩阵（因子 × horizon）===")
    names = list(factors.keys())
    header = "因子".ljust(24) + "".join(f"  {h:>6}日" for h in HORIZONS)
    print(header)
    for name in names:
        row = name.ljust(24)
        for h in HORIZONS:
            v = results[h][name]["mean_rank_ic"]
            row += f"  {v:+.4f}" if v is not None else "     N/A"
        print(row)
    print("\n=== ICIR 矩阵（因子 × horizon）===")
    print(header)
    for name in names:
        row = name.ljust(24)
        for h in HORIZONS:
            v = results[h][name]["icir"]
            row += f"  {v:+.3f}" if v is not None else "     N/A"
        print(row)
    print("\n=== +IC% 矩阵（因子 × horizon）===")
    print(header)
    for name in names:
        row = name.ljust(24)
        for h in HORIZONS:
            v = results[h][name]["positive_ic_fraction"]
            row += f"  {v*100:5.1f}%" if v is not None else "     N/A"
        print(row)


if __name__ == "__main__":
    main()
