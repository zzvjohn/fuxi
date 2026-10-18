# -*- coding: utf-8 -*-
"""重评估 2026-10-18 08:00 scheduled 闭环的 6 个 attempted 因子（新 E1-E4 + 新 E2 池冗余门）。

背景：这 6 个因子原本在 fuxi_fusion_20261009（93 截面）上评估，E1 全 not_applicable。
本脚本把它们搬到 v2.0 基线的 ledger_v1（native64_20210104_20251231_ledger_v1，1125 截面）
上，用当前代码（含 P-20261018-014 的 E2 池冗余门）重跑 E0-E4。

E2 参照池 = 19:00 run（campaign_779fe46）中 E1-E4 全通过、仅缺数据契约回执的 15 个因子。

用法:
  python scripts/reeval_6_e1e4.py            # 全量 6 个
  python scripts/reeval_6_e1e4.py --smoke    # 只跑 1 个，验证链路
  python scripts/reeval_6_e1e4.py --empty-pool  # E2 用空池（看原始 E1/E3/E4）
"""
import sys
import os
import json
import time
import glob

sys.path.insert(0, "e:/quant")

from fuxi_v1.data import Snapshot, Labels                       # noqa: E402
from fuxi_v1.dsl import evaluate_ast                            # noqa: E402
from fuxi_v1.search import node                                 # noqa: E402
from fuxi_v1.research_evaluation import evaluate_candidate       # noqa: E402
from fuxi_v1.contracts import load_json                          # noqa: E402

SRC_CAMPAIGN = "e:/quant/data/fuxi_unified/runs/unified_c43180df30c92861954475de/research/campaign_c43180df30c92861954475de"
REF_CAMPAIGN = "e:/quant/data/fuxi_unified/runs/unified_779fe46ab39b01018533172f/research/campaign_779fe46ab39b01018533172f"
INPUT = os.path.join(REF_CAMPAIGN, "input.json")
POLICY = os.path.join(REF_CAMPAIGN, "policy.json")
OUT = "e:/quant/output/reeval_6_e1e4.json"
TOP_N = 6


def load_targets():
    rows = []
    for run in sorted(glob.glob(os.path.join(SRC_CAMPAIGN, "bundles", "run_*"))):
        cand_path = os.path.join(run, "candidate.json")
        if not os.path.isfile(cand_path):
            continue
        cand = json.load(open(cand_path, encoding="utf-8"))
        ev_path = os.path.join(run, "evaluation.json")
        ic = None
        if os.path.isfile(ev_path):
            ev = json.load(open(ev_path, encoding="utf-8"))
            e1 = next((e for e in ev.get("evaluations", []) if e.get("stage") == "E1"), None)
            if e1:
                ic = ((e1.get("metrics") or {}).get("spearman") or {}).get("mean")
        rows.append({"run": run, "candidate": cand, "orig_ic": ic})
    rows.sort(key=lambda r: -(r["orig_ic"] or -9e9))
    return rows


def load_pool():
    """19:00 run 中 E1-E4 全通过（仅缺数据契约）的因子 scores，作为 E2 参照池。"""
    pool = []
    for run in sorted(glob.glob(os.path.join(REF_CAMPAIGN, "bundles", "run_*"))):
        ad = os.path.join(run, "admission.json")
        if not os.path.isfile(ad):
            continue
        d = json.load(open(ad, encoding="utf-8"))
        if sorted(d.get("blockers", [])) != ["independent_data_contract_receipt_missing"]:
            continue
        sp = os.path.join(run, "scores.json")
        if not os.path.isfile(sp):
            continue
        pool.append({"candidate_id": d.get("candidate_id"), "scores": load_json(sp)})
    return pool


def main():
    smoke = "--smoke" in sys.argv
    empty_pool = "--empty-pool" in sys.argv
    t0 = time.time()

    print("[1/5] 加载 ledger_v1 payload (input.json ~264MB)...", flush=True)
    payload = load_json(INPUT)
    snapshot = Snapshot(payload["snapshot"])
    labels = Labels(payload["labels"], snapshot)
    market = payload["market"]
    clock = payload["research_clock"]
    policy = load_json(POLICY)
    print("      payload 就绪，%.0fs；snapshot=%s" % (time.time() - t0, snapshot.hash[:12]), flush=True)

    print("[2/5] 计算 baseline rank(return(close,20))...", flush=True)
    baseline = evaluate_ast(node("rank", node("return", {"field": "close"}, window=20)), snapshot)
    print("      baseline 完成，%.0fs" % (time.time() - t0), flush=True)

    print("[3/5] 装载 E2 参照池...", flush=True)
    pool = [] if empty_pool else load_pool()
    print("      池大小 = %d（%s）" % (len(pool), "空池" if empty_pool else "19:00 已通过 15 因子"), flush=True)

    rows = load_targets()
    if smoke:
        rows = rows[:1]
    print("[4/5] 目标 %d 个因子（按原 93 截面 IC 降序）" % len(rows), flush=True)

    out = []
    for i, r in enumerate(rows, 1):
        t_i = time.time()
        cand = r["candidate"]
        scores = evaluate_ast(cand["canonical_ast"], snapshot)
        res = evaluate_candidate(cand, scores, baseline, snapshot, labels, market, policy, clock,
                                 top_n=TOP_N, pool_scores=pool)
        evs = {e["stage"]: e for e in res["evaluations"]}
        e1_sp = ((evs["E1"].get("diagnostics") or {}).get("spearman") or {})
        pr = (evs["E2"].get("diagnostics") or {}).get("pool_redundancy", {}) or {}
        summary = {
            "rank": i,
            "short": cand["candidate_id"][:16],
            "candidate_id": cand["candidate_id"],
            "orig_ic_93": r["orig_ic"],
            "E1": {"decision": evs["E1"]["decision"], "reason": evs["E1"].get("reason_code"),
                   "ic_mean": e1_sp.get("mean"), "icir": e1_sp.get("icir_nonannualized"),
                   "n_sections": ((evs["E1"].get("diagnostics") or {}).get("valid_section_count"))},
            "E2": {"decision": evs["E2"]["decision"], "reason": evs["E2"].get("reason_code"),
                   "max_corr": pr.get("max_corr"), "pool_size": pr.get("pool_size"),
                   "redundant_with": pr.get("redundant_with"),
                   "residual_alpha_lower": (pr.get("residual_alpha") or {}).get("lower")},
            "E3": {"decision": evs["E3"]["decision"], "reason": evs["E3"].get("reason_code")},
            "E4": {"decision": evs["E4"]["decision"], "reason": evs["E4"].get("reason_code")},
        }
        allpass = all(evs[s]["decision"] == "pass" for s in ["E1", "E2", "E3", "E4"])
        summary["all_pass"] = allpass
        summary["blockers"] = ([] if allpass else [s + "_not_accepted" for s in ["E1", "E2", "E3", "E4"]
                                                    if evs[s]["decision"] != "pass"])
        summary["jq_precondition"] = allpass  # 仍缺独立数据契约回执
        out.append(summary)
        print("  [%d/%d] %s  E1=%s E2=%s E3=%s E4=%s all_pass=%s  IC=%s  %.0fs"
              % (i, len(rows), summary["short"], evs["E1"]["decision"], evs["E2"]["decision"],
                 evs["E3"]["decision"], evs["E4"]["decision"], allpass,
                 round(e1_sp.get("mean") or 0, 5), time.time() - t_i), flush=True)

    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "source_campaign": os.path.basename(SRC_CAMPAIGN),
               "ref_campaign": os.path.basename(REF_CAMPAIGN),
               "snapshot_hash": snapshot.hash,
               "policy_hash": policy["policy_hash"],
               "max_trials": policy["request"]["max_trials"],
               "research_window": [policy["request"]["research_start"], policy["request"]["research_end"]],
               "pool_size": len(pool), "empty_pool": empty_pool, "smoke": smoke,
               "results": out}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[5/5] 完成，总耗时 %.0fs，结果写入 %s" % (time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()
