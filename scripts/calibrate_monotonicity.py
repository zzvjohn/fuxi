"""E1 单调性门阈值校准（用真实已评估候选样本）。

用最近的 unified campaign 产出（evidence.sqlite 里的候选 canonical_ast + input.json
的真实快照/labels）对每个候选重新求值 scores，跑单调性诊断，收集
``mean_monotonic_spearman`` 与 spread 显著性的分布，据此标定
``MONOTONICITY_DEFAULTS["min_monotonic_spearman"]`` 阈值（初始值 0.8）。

口径：
  · scores = evaluate_ast(canonical_ast, snapshot) 的纯因子截面（非 blended）；
  · labels = input.json 的 materialized labels（horizon=5，prediction_proxy）；
  · 窗口 = policy 的 research 窗口（fallback 全 session）。
  · 只报告分布，不改代码、不写生产阈值——标定结果由人工确认后回填。

用法：
  python scripts/calibrate_monotonicity.py [--input <input.json>] [--evidence <evidence.sqlite>]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuxi_v1.contracts import load_json  # noqa: E402
from fuxi_v1.data import Snapshot, Labels  # noqa: E402
from fuxi_v1.dsl import evaluate_ast  # noqa: E402
from fuxi_v1.monotonicity import monotonicity_diagnostics, monotonicity_gate  # noqa: E402


def _research_window(evidence_path, snapshot):
    """从 evidence 的 calibration_request fact 拿 research 窗口；取不到则用全 session。"""
    con = sqlite3.connect(evidence_path)
    cur = con.cursor()
    row = cur.execute(
        "SELECT payload FROM facts WHERE kind='calibration_request' LIMIT 1").fetchone()
    con.close()
    if row:
        req = json.loads(row[0])
        start, end = req.get("research_start"), req.get("research_end")
        if start in snapshot.sessions and end in snapshot.sessions:
            return start, end
    return snapshot.sessions[0], snapshot.sessions[-1]


def calibrate(input_path, evidence_path):
    payload = load_json(input_path)
    snapshot = Snapshot(payload["snapshot"])
    labels = Labels(payload["labels"], snapshot)
    clock = payload["research_clock"]
    start, end = _research_window(evidence_path, snapshot)

    con = sqlite3.connect(evidence_path)
    cur = con.cursor()
    cand_rows = cur.execute(
        "SELECT payload FROM facts WHERE kind='candidate'").fetchall()

    results = []
    for (raw,) in cand_rows:
        cand = json.loads(raw)
        ast = cand.get("canonical_ast")
        if not ast:
            continue
        scores = evaluate_ast(ast, snapshot)
        diag = monotonicity_diagnostics(scores, snapshot, labels, clock, start, end)
        gate = monotonicity_gate(diag)
        results.append({
            "candidate_id": cand["candidate_id"][:12],
            "formula_hash": cand.get("formula_hash", "")[:12],
            "n_sessions": diag["n_sessions"],
            "mean_monotonic_spearman": diag["mean_monotonic_spearman"],
            "mean_adjacent_monotonic_fraction": diag["mean_adjacent_monotonic_fraction"],
            "spread_lower": (diag["spread_inference"] or {}).get("lower"),
            "spread_mean": (diag["spread_inference"] or {}).get("mean"),
            "time_avg_group_returns": diag["time_averaged_group_returns"],
            "gate_decision": gate["decision"],
            "gate_reason": gate["reason_code"],
            "spread_significant": gate["spread_significant"],
            "spearman_ok": gate["monotonic_spearman_ok"],
        })

    return {
        "input": str(input_path),
        "evidence": str(evidence_path),
        "window": [start, end],
        "n_candidates": len(results),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(
        ROOT / "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e"
        "/research/campaign_153d8e0779d74d49e2f3606e/input.json"))
    ap.add_argument("--evidence", default=str(
        ROOT / "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e"
        "/research/campaign_153d8e0779d74d49e2f3606e/evidence.sqlite"))
    ap.add_argument("--out", default=str(
        ROOT / "data/system_evolution/calibration/monotonicity_calibration.json"))
    args = ap.parse_args()

    report = calibrate(args.input, args.evidence)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总分布
    rs = report["results"]
    spears = [r["mean_monotonic_spearman"] for r in rs
              if r["mean_monotonic_spearman"] is not None]
    print(f"候选数: {len(rs)}")
    print(f"窗口: {report['window'][0]} -> {report['window'][1]}")
    if spears:
        spears_sorted = sorted(spears)
        n = len(spears_sorted)
        def q(p):
            return spears_sorted[min(n - 1, int(p * n))]
        print(f"mean_monotonic_spearman 分布: n={n}")
        print(f"  min={spears_sorted[0]:.3f}  q25={q(.25):.3f}  median={q(.5):.3f}  "
              f"q75={q(.75):.3f}  max={spears_sorted[-1]:.3f}")
    print("\n逐候选明细:")
    for r in rs:
        tr = r["time_avg_group_returns"]
        tr_s = "[" + ",".join(f"{v:.4f}" for v in tr) + "]" if tr else "[]"
        print(f"  {r['candidate_id']}  n_session={r['n_sessions']}  "
              f"spearman={r['mean_monotonic_spearman']}  "
              f"adj_frac={r['mean_adjacent_monotonic_fraction']}  "
              f"spread_lo={r['spread_lower']}  "
              f"gate={r['gate_decision']}({r['gate_reason']})  "
              f"groups={tr_s}")
    print(f"\n校准报告已写入: {out}")


if __name__ == "__main__":
    main()
