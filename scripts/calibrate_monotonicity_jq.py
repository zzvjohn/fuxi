"""全量历史 JQ pass 因子单调性校准（只读诊断，不改生产阈值）。

对全部历史 JQ pass 因子跑单调性诊断，收集 ``mean_monotonic_spearman`` 与
spread 显著性的分布，据此标定 ``monotonicity_gate`` 的 ``min_monotonic_spearman``
阈值（当前生产初始值 0.8）。

两类 JQ pass 因子：
  1. formal：``data/fuxi_unified/runs/{jq_bundles_20261019,reeval_jq_20261019}``
     的 BUILD_SUMMARY 中 ``jq_candidate_allowed=true`` 的候选（canonical_ast 直接评估）；
  2. legacy：``data/jq_validated_champions.json`` 的 GP 育种 JQ 自动入库因子，
     表达式为 pandas 风格。本脚本用受控 namespace 直接对快照面板求值
     （best-effort 信号等价，非生产等价声明；只用于校准分布）。

口径：
  · snapshot/labels = unified_153d8e 真实 campaign（research_exposed，64 只，1212 session）；
  · 窗口 = research 窗口（2021-05-18 -> 2025-12-31）；
  · 只报告分布 + 多阈值 pass 率，不写生产阈值，标定结果由人工确认后回填。

用法：
  python scripts/calibrate_monotonicity_jq.py [--out <json>]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fuxi_v1.contracts import load_json  # noqa: E402
from fuxi_v1.data import Snapshot, Labels  # noqa: E402
from fuxi_v1.dsl import evaluate_ast  # noqa: E402
from fuxi_v1.monotonicity import (  # noqa: E402
    MONOTONICITY_DEFAULTS, monotonicity_diagnostics, monotonicity_gate)
from fuxi_v1.shared_catalog import FIELD_ALIASES  # noqa: E402

INPUT_DEFAULT = str(
    ROOT / "data/fuxi_unified/runs/unified_153d8e0779d74d49e2f3606e"
    "/research/campaign_153d8e0779d74d49e2f3606e/input.json")

FORMAL_RUNDIRS = [
    ROOT / "data/fuxi_unified/runs/jq_bundles_20261019",
    ROOT / "data/fuxi_unified/runs/reeval_jq_20261019",
]
LEGACY_CHAMPIONS = ROOT / "data/jq_validated_champions.json"

THRESHOLD_GRID = [0.8, 0.6, 0.4, 0.2, 0.1, 0.0, -0.1, -0.2]


# ---------------------------------------------------------------- 快照面板求值
def _build_panels(snapshot):
    """为 legacy pandas 表达式构建 {alias: DataFrame(sessions x instruments)}。"""
    import pandas as pd
    instruments = sorted({s for members in snapshot.universe.values() for s in members})
    panels = {}
    for alias, field in FIELD_ALIASES.items():
        if field not in snapshot.payload["fields"]:
            continue
        rows = {}
        for i, d in enumerate(snapshot.sessions):
            clock = snapshot.clocks[d]
            rows[d] = {s: snapshot.value(i, s, field, clock) for s in instruments}
        panels[alias] = pd.DataFrame.from_dict(rows, orient="index")
    return panels, instruments


def _eval_legacy(expression, panels, instruments):
    """在受控 namespace 下求值 pandas 风格表达式 -> {day: {code: float}}。"""
    import math
    import numpy as np
    import pandas as pd
    namespace = dict(panels)
    namespace.update({"np": np, "log": np.log, "sqrt": np.sqrt,
                      "abs": np.abs, "exp": np.exp})
    tree = compile(expression, "<jq_legacy_expression>", "eval")
    value = eval(tree, {"__builtins__": {}}, namespace)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        scores = {}
        for d in value.index:
            row = value.loc[d]
            if isinstance(row, pd.Series):
                out = {}
                for s in instruments:
                    v = row.get(s)
                    if v is not None and np.isfinite(v):
                        out[s] = float(v)
                scores[d] = out
            else:  # scalar per day (degenerate)
                if np.isfinite(row):
                    scores[d] = {s: float(row) for s in instruments}
        return scores
    # 标量/常量表达式 → 退化截面，无单调性意义
    return None


# ---------------------------------------------------------------- 收集 JQ 因子
def _collect_formal():
    """formal JQ pass 候选：BUILD_SUMMARY jq_candidate_allowed=true -> canonical_ast。"""
    factors = []
    for rundir in FORMAL_RUNDIRS:
        summary_path = rundir / "BUILD_SUMMARY.json"
        if not summary_path.is_file():
            continue
        for row in load_json(summary_path):
            if not row.get("jq_candidate_allowed"):
                continue
            bundle = Path(row["bundle"]) / "candidate.json"
            if not bundle.is_file():
                continue
            cand = load_json(bundle)
            factors.append({
                "source": "formal_jq_candidate",
                "run": rundir.name,
                "name": row["candidate_id"][:16],
                "candidate_id": row["candidate_id"],
                "canonical_ast": cand.get("canonical_ast"),
                "expression": None,
            })
    return factors


def _collect_legacy():
    """legacy JQ 自动入库冠军：expression 字符串（pandas 风格）。"""
    factors = []
    if not LEGACY_CHAMPIONS.is_file():
        return factors
    for row in load_json(LEGACY_CHAMPIONS):
        expr = row.get("expression")
        if not expr or not isinstance(expr, str) or not expr.strip():
            continue
        factors.append({
            "source": "legacy_gp_jq_validated",
            "name": row.get("name", "?"),
            "candidate_id": None,
            "canonical_ast": None,
            "expression": expr.strip(),
            "paradigm": row.get("paradigm"),
        })
    return factors


# ---------------------------------------------------------------- 主流程
def _research_window(input_path):
    """从同目录 evidence.sqlite 的 calibration_request fact 读 research 窗口。"""
    import sqlite3
    evidence = Path(input_path).parent / "evidence.sqlite"
    if evidence.is_file():
        con = sqlite3.connect(evidence)
        row = con.execute(
            "SELECT payload FROM facts WHERE kind='calibration_request' LIMIT 1").fetchone()
        con.close()
        if row:
            req = json.loads(row[0])
            return req.get("research_start"), req.get("research_end")
    return None, None


def run_calibration(input_path, out_path):
    payload = load_json(input_path)
    snapshot = Snapshot(payload["snapshot"])
    labels = Labels(payload["labels"], snapshot)
    clock = payload["research_clock"]
    start, end = _research_window(input_path)
    if start is None or start not in snapshot.sessions:
        start = snapshot.sessions[0]
    if end is None or end not in snapshot.sessions:
        end = snapshot.sessions[-1]

    formal = _collect_formal()
    legacy = _collect_legacy()
    panels, instruments = _build_panels(snapshot)

    results = []
    for f in formal:
        try:
            scores = evaluate_ast(f["canonical_ast"], snapshot)
        except Exception as exc:  # noqa: BLE001
            results.append({**f, "eval_status": "error", "error": str(exc)})
            continue
        results.append(_diagnose(f, scores, snapshot, labels, clock, start, end))

    for f in legacy:
        try:
            scores = _eval_legacy(f["expression"], panels, instruments)
        except Exception as exc:  # noqa: BLE001
            results.append({**f, "eval_status": "error", "error": str(exc)})
            continue
        if scores is None:
            results.append({**f, "eval_status": "skipped", "error": "degenerate_or_non_cross_sectional"})
            continue
        results.append(_diagnose(f, scores, snapshot, labels, clock, start, end))

    report = {
        "input": str(input_path),
        "window": [start, end],
        "n_formal": len(formal),
        "n_legacy": len(legacy),
        "n_evaluated": sum(1 for r in results if r.get("eval_status") == "ok"),
        "n_error": sum(1 for r in results if r.get("eval_status") == "error"),
        "n_skipped": sum(1 for r in results if r.get("eval_status") == "skipped"),
        "threshold_grid": THRESHOLD_GRID,
        "results": results,
    }
    report["pass_rates"] = _pass_rates(results)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(report)
    return report


def _diagnose(f, scores, snapshot, labels, clock, start, end):
    diag = monotonicity_diagnostics(scores, snapshot, labels, clock, start, end,
                                     block_length=MONOTONICITY_DEFAULTS["block_length"],
                                     alpha=MONOTONICITY_DEFAULTS["alpha"],
                                     replicates=MONOTONICITY_DEFAULTS["replicates"],
                                     seed=MONOTONICITY_DEFAULTS["seed"])
    gate = monotonicity_gate(diag)
    return {
        **f,
        "eval_status": "ok",
        "n_sessions": diag["n_sessions"],
        "mean_monotonic_spearman": diag["mean_monotonic_spearman"],
        "mean_adjacent_monotonic_fraction": diag["mean_adjacent_monotonic_fraction"],
        "spread_lower": (diag["spread_inference"] or {}).get("lower"),
        "spread_mean": (diag["spread_inference"] or {}).get("mean"),
        "spread_status": (diag["spread_inference"] or {}).get("status"),
        "time_avg_group_returns": diag["time_averaged_group_returns"],
        "gate_decision": gate["decision"],
        "gate_reason": gate["reason_code"],
        "spread_significant": gate["spread_significant"],
        "spearman_ok_at_0_8": gate["monotonic_spearman_ok"],
    }


def _pass_rates(results):
    """多阈值 pass 率：pass = spread_ok AND spearman >= threshold。"""
    ok = [r for r in results if r.get("eval_status") == "ok"
          and r.get("n_sessions", 0) >= MONOTONICITY_DEFAULTS["min_sessions"]]
    rates = {}
    for t in THRESHOLD_GRID:
        n_pass = sum(1 for r in ok
                     if r.get("spread_significant")
                     and r.get("mean_monotonic_spearman") is not None
                     and r["mean_monotonic_spearman"] >= t)
        rates[str(t)] = {"pass": n_pass, "n": len(ok),
                         "rate": round(n_pass / len(ok), 4) if ok else None}
    return rates


def _print_summary(report):
    ok = [r for r in report["results"] if r.get("eval_status") == "ok"]
    print(f"formal={report['n_formal']} legacy={report['n_legacy']} "
          f"evaluated={report['n_evaluated']} error={report['n_error']} skipped={report['n_skipped']}")
    print(f"窗口: {report['window'][0]} -> {report['window'][1]}")
    spears = sorted(r["mean_monotonic_spearman"] for r in ok
                    if r["mean_monotonic_spearman"] is not None)
    if spears:
        n = len(spears)
        def q(p):
            return spears[min(n - 1, int(p * n))]
        print(f"mean_monotonic_spearman 分布: n={n} "
              f"min={spears[0]:.3f} q25={q(.25):.3f} median={q(.5):.3f} "
              f"q75={q(.75):.3f} max={spears[-1]:.3f}")
    print(f"\n多阈值 pass 率 (spread_ok AND spearman>=t):")
    for t, r in report["pass_rates"].items():
        print(f"  spearman>={t}: {r['pass']}/{r['n']} = {r['rate']}")
    print("\n逐因子明细:")
    for r in report["results"]:
        if r.get("eval_status") != "ok":
            print(f"  [{r['source']}] {r['name']}: {r['eval_status']} {r.get('error','')[:60]}")
            continue
        tr = r["time_avg_group_returns"]
        tr_s = "[" + ",".join(f"{v:.4f}" for v in tr) + "]" if tr else "[]"
        print(f"  [{r['source']}] {r['name']}  n={r['n_sessions']} "
              f"spearman={r['mean_monotonic_spearman']:.3f} "
              f"spread_lo={r['spread_lower']:.5f} spread_ok={r['spread_significant']} "
              f"gate={r['gate_decision']}({r['gate_reason']}) groups={tr_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DEFAULT)
    ap.add_argument("--out", default=str(
        ROOT / "data/system_evolution/calibration/monotonicity_jq_calibration.json"))
    args = ap.parse_args()
    run_calibration(args.input, args.out)
    print(f"\n校准报告已写入: {args.out}")


if __name__ == "__main__":
    main()
