"""A/B test: refine/combine arms under FUXI_V1_GA_OPS=off vs =on on the 1212-section snapshot.

Runs the unified_runner campaign twice with an identical frozen request (same seed,
same snapshot, no memory, LLM off) and only the FUXI_V1_GA_OPS environment knob
differing. The two runs get distinct run_ids (GA_OPS enters effective_config ->
extra_resolved -> run identity), so they never collide on the same cache.

Output: a side-by-side comparison of refine/combine candidates (mean_rank_ic,
research_p_one_sided, n_signal_pairs, economic_reward_count).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "fuxi_unified" / "snapshots" / "native64_20210104_20251231" / "campaign_input.json"
PYTHON = sys.executable


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(tag, request_path, ga_on):
    env = dict(os.environ)
    if ga_on:
        env["FUXI_V1_GA_OPS"] = "on"
    else:
        # P-20261018 修正：GA_OPS 的代码默认值已翻为 "on"，此处**必须显式写 off**。
        # 旧写法 env.pop(...) 只是删除变量，代码仍回落到默认 on，导致两臂都跑 GA、
        # 候选序列 100% 相同（已实测：80/80 candidate_id 一致，mutation axis 均为
        # one_node_edit/combination）。显式 off 才能恢复真正的 baseline 对照。
        env["FUXI_V1_GA_OPS"] = "off"
    # P-20261018 提速（实测：4 并行 3.84x + 快照复用 1.36x，result/scores bit-exact）：
    #   M1 轮内并行——每轮 4 个候选同时评估；
    #   M2 持久池——按 (input_path,input_hash) 复用 249MB 快照，免每候选重解析。
    # 内存实测 4 worker 峰值 5.27GB（本机可用 13.5GB），pool_size 4 安全。
    # 回退：FUXI_AB_SPEEDUP=0 → 恢复逐候选 spawn + 串行（旧行为）。
    if os.environ.get("FUXI_AB_SPEEDUP", "1") == "1":
        env["FUXI_V1_PERSISTENT_POOL"] = "1"
        env["FUXI_V1_POOL_SIZE"] = os.environ.get("FUXI_AB_POOL_SIZE", "4")
        # 每 slot 处理约 candidates_per_round 个请求/轮；40 留足余量，避免 recycle 重载快照。
        env["FUXI_V1_POOL_MAX_REQUESTS"] = os.environ.get("FUXI_AB_POOL_MAX_REQUESTS", "40")
        env["FUXI_V1_PARALLEL_CANDIDATES"] = "1"
    cmd = [PYTHON, "-B", "-X", "utf8", "-m", "fuxi_v1.unified_runner", "run",
           "--workspace", str(ROOT), "--request", str(request_path), "--trigger", "manual"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    run_id = None
    try:
        summary = json.loads(out)
        run_id = summary.get("run_id")
        state = summary.get("state")
    except Exception:
        state = "parse_error"
    if proc.returncode != 0 or state != "completed":
        print(f"[{tag}] FAILED rc={proc.returncode} state={state}")
        print(f"[{tag}] stdout tail: {out[-500:]}")
        print(f"[{tag}] stderr tail: {err[-500:]}")
        if "run identity collision" in err:
            print(f"[{tag}] HINT: 同配置已有 run。配置未变时可直接复用其 run_id；"
                  f"确需重跑请加 --tag <批次名>（会改变 run identity）。")
        return None
    print(f"[{tag}] completed run_id={run_id} elapsed={elapsed:.1f}s")
    return run_id


def load_report(run_id):
    rec = json.load(open(ROOT / "data" / "fuxi_unified" / "runs" / run_id / "research" / "RECEIPT.json", encoding="utf-8"))
    report = json.load(open(rec["report_path"], encoding="utf-8"))
    return rec, report


def arm_stats(report):
    cr = report.get("candidate_results", [])
    arms = {}
    for c in cr:
        arm = c.get("arm")
        if arm not in ("refine", "combine"):
            continue
        arms.setdefault(arm, []).append(c)
    out = {}
    for arm, rows in arms.items():
        ics = [abs(r.get("mean_rank_ic") or 0.0) for r in rows]
        ps = [r.get("research_p_one_sided") for r in rows if r.get("research_p_one_sided") is not None]
        ns = [r.get("n_signal_pairs") for r in rows if r.get("n_signal_pairs")]
        out[arm] = {
            "count": len(rows),
            "abs_mean_rank_ic": round(sum(ics) / len(ics), 6) if ics else None,
            "max_abs_ic": round(max(ics), 6) if ics else None,
            "min_p_one_sided": round(min(ps), 6) if ps else None,
            "n_signal_pairs": ns[0] if ns else None,
            # e1_decision 的实际取值是 "pass"/"reject"/"not_applicable"（不是 "accept"）。
            # P-20261018 修正口径：旧写 "accept" 使 e1_pass 恒为 0，掩盖了真实通过数。
            "e1_pass": sum(1 for r in rows if r.get("e1_decision") == "pass"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    # P-20261017: 默认预算按 ga_ops 的统计约束常量设定——20 rounds × 4 candidates
    # = max_trials 80 -> t 门槛 3.42 / 最低可检测 ICIR 0.102；GA 臂占四臂之半，故
    # 实际 GA 代数 ≈ 10 代、每代种群 4。旧默认 4×2=8（max_trials 8, t≈2.7）代数不足。
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--seed", type=int, default=77777)
    # P-20261018: run identity 由 request 全字段派生。配置不变时重跑会撞上已存在的
    # run（ContractError: run identity collision）。加 --tag 写入 purpose 即可开新批次；
    # 不传则保持原行为（同配置可复用已完成 run，但重跑会明确报 collision）。
    ap.add_argument("--tag", default="", help="批次标记，拼入 purpose 以区分同配置的不同批次")
    args = ap.parse_args()

    purpose = "GA operator A/B" + (f" [{args.tag}]" if args.tag else "")
    request = {
        "schema_version": "fuxi.unified_run.1",
        "input": str(SNAPSHOT),
        "input_sha256": file_sha(SNAPSHOT),
        "rounds": args.rounds,
        "candidates_per_round": args.candidates,
        "seed": args.seed,
        "mode": "adaptive",
        "scheduler_policy": "legacy",
        "include_catalog": True,
        "purpose": purpose,
        "llm_enabled": False,
        "model_algorithms": [],
        "external_hypotheses": [],
        # 1212-section x 64-universe snapshot load exceeds the 2 GiB governance default;
        # manual trigger may explicitly raise the worker memory ceiling (design-sanctioned).
        "limits": {
            # 20 rounds × 4 candidates = 80 恰好顶满旧上限；抬到 120 留出余量，
            # 同时仍守住 ga_ops.GA_TRIALS_BUDGET=(60,120) 的 max_trials 上限。
            "max_rounds": 40, "max_candidates": 120, "max_worker_seconds": 120,
            "max_total_seconds": 14400, "max_memory_mb": 8192,
            "max_external_requests": 30, "max_llm_calls": 12, "max_output_tokens": 24000,
        },
    }
    req_path = ROOT / "data" / "fuxi_unified" / "_ab_request.json"
    req_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

    base_id = run_one("baseline(off)", req_path, ga_on=False)
    ga_id = run_one("ga(on)", req_path, ga_on=True)

    if not base_id or not ga_id:
        print("A/B aborted: one arm failed")
        return 1

    _, base_report = load_report(base_id)
    _, ga_report = load_report(ga_id)
    print("\n=== baseline (GA off)  refine/combine ===")
    print(json.dumps(arm_stats(base_report), ensure_ascii=False, indent=2))
    print("economic_reward_count:", base_report.get("economic_reward_count"),
          "| attempted:", base_report.get("attempted_candidates"))
    print("\n=== ga (GA on)  refine/combine ===")
    print(json.dumps(arm_stats(ga_report), ensure_ascii=False, indent=2))
    print("economic_reward_count:", ga_report.get("economic_reward_count"),
          "| attempted:", ga_report.get("attempted_candidates"))

    # dump full candidate_results for both arms for deeper inspection
    out = {"baseline_run_id": base_id, "ga_run_id": ga_id,
           "baseline": {"refine_combine": arm_stats(base_report), "economic_reward_count": base_report.get("economic_reward_count"),
                        "attempted": base_report.get("attempted_candidates")},
           "ga": {"refine_combine": arm_stats(ga_report), "economic_reward_count": ga_report.get("economic_reward_count"),
                  "attempted": ga_report.get("attempted_candidates")}}
    out_path = ROOT / "data" / "fuxi_unified" / "_ab_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
