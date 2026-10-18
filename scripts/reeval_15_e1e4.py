# -*- coding: utf-8 -*-
"""重评估 jq_standalone_15 的 15 个因子（新 E1-E4 标准，含改造后的 E2 池冗余门）。

链路: 复用 campaign_779fe46 的 input.json(snapshot/labels/market/research_clock)，
     对每个因子 candidate.json 用当前代码 evaluate_ast 重算 scores，再调用
     evaluate_candidate 跑 E0-E4。pool 按 Rank IC 降序动态增长（与 INDEX.md 一致）。

用法:
  python scripts/reeval_15_e1e4.py            # 全量 15 个
  python scripts/reeval_15_e1e4.py --smoke    # 只跑 IC 最高的 1 个(空池)，验证链路
"""
import sys
import os
import json
import time
import glob

sys.path.insert(0, "e:/quant")

from fuxi_v1.data import Snapshot, Labels          # noqa: E402
from fuxi_v1.dsl import evaluate_ast               # noqa: E402
from fuxi_v1.search import node                    # noqa: E402
from fuxi_v1.research_evaluation import evaluate_candidate  # noqa: E402
from fuxi_v1.contracts import load_json            # noqa: E402

CAMPAIGN = "e:/quant/data/fuxi_unified/runs/unified_779fe46ab39b01018533172f/research/campaign_779fe46ab39b01018533172f"
INPUT = os.path.join(CAMPAIGN, "input.json")
BUNDLES = os.path.join(CAMPAIGN, "bundles")
OUT = "e:/quant/output/reeval_15_e1e4.json"
TOP_N = 6


def collect_targets():
    rows = []
    for ad in glob.glob(os.path.join(BUNDLES, "run_*", "admission.json")):
        d = json.load(open(ad, encoding="utf-8"))
        if sorted(d.get("blockers", [])) != ["independent_data_contract_receipt_missing"]:
            continue
        run = os.path.dirname(ad)
        cand = json.load(open(os.path.join(run, "candidate.json"), encoding="utf-8"))
        ev = json.load(open(os.path.join(run, "evaluation.json"), encoding="utf-8"))
        e1 = next(e for e in ev["evaluations"] if e["stage"] == "E1")
        sp = e1["metrics"]["spearman"]
        rows.append({"run": run, "short": d["candidate_id"][:16],
                     "candidate_id": d["candidate_id"], "candidate": cand,
                     "ic": sp["mean"], "icir": sp["icir_nonannualized"]})
    rows.sort(key=lambda r: -r["ic"])
    return rows


def main():
    smoke = "--smoke" in sys.argv
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    t0 = time.time()
    print("[1/5] 加载 payload (input.json ~249MB)...", flush=True)
    payload = load_json(INPUT)
    print("      payload 就绪，%.0fs" % (time.time() - t0), flush=True)
    snapshot = Snapshot(payload["snapshot"])
    labels = Labels(payload["labels"], snapshot)
    market = payload["market"]
    clock = payload["research_clock"]
    print("      Snapshot/Labels 构造完成，%.0fs" % (time.time() - t0), flush=True)

    print("[2/5] 计算 baseline rank(return(close,20))...", flush=True)
    baseline_ast = node("rank", node("return", {"field": "close"}, window=20))
    baseline = evaluate_ast(baseline_ast, snapshot)
    print("      baseline 完成，%.0fs" % (time.time() - t0), flush=True)

    print("[3/5] 筛选目标因子...", flush=True)
    rows = collect_targets()
    if smoke:
        rows = rows[:1]
    elif limit:
        rows = rows[:limit]
    print("      目标 %d 个因子（按 Rank IC 降序）" % len(rows), flush=True)

    print("[4/5] 依次评估（E2 pool 动态增长，入池=E1-E4 全 pass）...", flush=True)
    pool_scores = []
    out = []
    for i, r in enumerate(rows, 1):
        t_i = time.time()
        scores = evaluate_ast(r["candidate"]["canonical_ast"], snapshot)
        policy = load_json(os.path.join(r["run"], "policy.json"))
        res = evaluate_candidate(r["candidate"], scores, baseline, snapshot, labels,
                                 market, policy, clock, top_n=TOP_N, pool_scores=pool_scores)
        evs = {e["stage"]: e for e in res["evaluations"]}
        pr = evs["E2"].get("diagnostics", {}).get("pool_redundancy", {}) or {}
        summary = {
            "rank": i, "short": r["short"], "ic": round(r["ic"], 5), "icir": round(r["icir"], 3),
            "candidate_id": r["candidate_id"],
            "E1": {"decision": evs["E1"]["decision"], "reason": evs["E1"]["reason_code"],
                   "ic_mean": round((evs["E1"].get("diagnostics", {}).get("spearman", {}) or {}).get("mean", 0) or 0, 5)},
            "E2": {"decision": evs["E2"]["decision"], "reason": evs["E2"]["reason_code"],
                   "max_corr": pr.get("max_corr"), "pool_size": pr.get("pool_size"),
                   "redundant_with": pr.get("redundant_with"),
                   "residual_alpha_lower": (pr.get("residual_alpha") or {}).get("lower")},
            "E3": {"decision": evs["E3"]["decision"], "reason": evs["E3"]["reason_code"]},
            "E4": {"decision": evs["E4"]["decision"], "reason": evs["E4"]["reason_code"]},
        }
        allpass = all(evs[s]["decision"] == "pass" for s in ["E1", "E2", "E3", "E4"])
        summary["all_pass"] = allpass
        out.append(summary)
        if allpass:
            pool_scores.append({"candidate_id": r["candidate_id"], "scores": scores})
        print("  [%2d/%d] %s  E1=%s E2=%s E3=%s E4=%s all_pass=%s pool=%d  %.0fs"
              % (i, len(rows), r["short"], evs["E1"]["decision"], evs["E2"]["decision"],
                 evs["E3"]["decision"], evs["E4"]["decision"], allpass, len(pool_scores),
                 time.time() - t_i), flush=True)

    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "top_n": TOP_N, "pool_order": "rank_ic_desc", "smoke": smoke,
               "results": out}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[5/5] 完成，总耗时 %.0fs，结果写入 %s" % (time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()
