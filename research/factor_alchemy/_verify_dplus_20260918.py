# -*- coding: utf-8 -*-
"""Verify scoped D+ closure result for forge_gen4_gp_gen4_r1_ts_rank (read-only)."""
import json, os

REPO = r"e:\quant"

def load(p):
    with open(p, "r", encoding="utf-8", newline="") as f:
        return json.load(f)

print("=== 1) trajectory entry ===")
tj = load(os.path.join(REPO, "data", "trajectory_log.json"))
entries = tj.get("trajectories", tj) if isinstance(tj, dict) else tj
print("type:", type(tj).__name__, "| n_entries:", len(entries))
target = None
n_dpa = 0
backlog = 0
for e in entries:
    if e.get("trajectory_id") == "traj_2561b7d3c712":
        target = e
    if e.get("d_plus_applied"):
        n_dpa += 1
    if e.get("jq_validated") and not e.get("d_plus_applied"):
        backlog += 1
if target:
    for k in sorted(target.keys()):
        if k.startswith("jq_") or k in ("d_plus_applied", "final_outcome", "final_ic", "final_icir"):
            print("  %-18s = %r" % (k, target.get(k)))
else:
    print("  !! target entry NOT FOUND")
print("  d_plus_applied=True count:", n_dpa)
print("  jq_validated=True & not d_plus_applied (backlog):", backlog)

print()
print("=== 2) lane_calibration ===")
lc = load(os.path.join(REPO, "data", "lane_calibration.json"))
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print("  %s%s : list len=%d" % (path, k, len(v)))
                if len(v) > 1:
                    print("    last =", {kk: v[-1].get(kk) for kk in list(v[-1].keys())[:8]})
            elif isinstance(v, dict):
                walk(v, path + k + ".")
walk(lc)
for k, v in lc.items():
    if not isinstance(v, (dict, list)):
        print("  %s = %r" % (k, v))

print()
print("=== 3) jq_codegen_queue target ===")
q = load(os.path.join(REPO, "data", "jq_codegen_queue.json"))
items = q.get("items", q) if isinstance(q, dict) else q
hit = 0
for it in (items if isinstance(items, list) else []):
    nm = it.get("name") or it.get("key") or it.get("factor_name") or ""
    if "forge_gen4_gp_gen4_r1_ts_rank" in str(nm):
        hit += 1
        print("  [%d] status=%r" % (hit, it.get("status")))
        for kk in ("jq_return", "jq_sharpe", "jq_maxdd", "jq_ic", "jq_icir", "jq_rating"):
            if kk in it:
                print("       %s = %r" % (kk, it.get(kk)))
print("  total matched keys:", hit)

# CRLF check
print()
print("=== 4) newline audit ===")
for rel in ["data/stage1_optimization_proposals.json", "data/jq_codegen_queue.json"]:
    p = os.path.join(REPO, rel)
    with open(p, "rb") as f:
        b = f.read()
    crlf = b.count(b"\r\n")
    lf = b.count(b"\n") - crlf
    print("  %-45s CRLF=%d bare_LF=%d" % (rel, crlf, lf))
