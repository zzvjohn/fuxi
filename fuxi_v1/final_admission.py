"""Canonical final JQ admission (2026-10-19).

Why this exists: ``jq_candidate_allowed`` is produced while the campaign runs,
against a pool that grows in *settle order*. E2 is a greedy redundancy gate, so
the same factor can pass or fail depending on which peers were admitted before
it. The delivery pipeline re-evaluated in IC-descending order and produced a
different count (7 flagged vs 4 deliverable), which is confusing and makes the
daily report disagree with the bundles.

This module defines one deterministic rule used by both sides:

    sort every E0–E4-pass candidate by mean rank IC descending (ties by
    candidate_id), then greedily re-run the E2 pool gate against the accepted
    pool. Only the accepted members are deliverable JQ candidates.

The campaign records the result as a ``final_admission`` fact and reports it as
``jq_candidate_count``; the provisional settle-order flag is preserved separately
as ``jq_candidate_count_provisional``.
"""
from .contracts import digest
from .dsl import evaluate_ast
from .research_evaluation import pool_redundancy

RULE = "ic_desc_greedy_v1"


def canonical_final_admission(results, candidates_all, snapshot, labels, clock, request,
                              *, score_fn=None, top_n=None):
    eligible = [r for r in results
                if r.get("jq_candidate_allowed") and r.get("candidate_id") in candidates_all]
    eligible.sort(key=lambda r: (-(r.get("mean_rank_ic") if r.get("mean_rank_ic") is not None else -9.0),
                                 r["candidate_id"]))
    start, end = request["research_start"], request["research_end"]
    pool = []
    members = []
    for row in eligible:
        cid = row["candidate_id"]
        ast = candidates_all[cid]["canonical_ast"]
        scores = (score_fn or evaluate_ast)(ast, snapshot)
        verdict = pool_redundancy(scores, pool, snapshot, labels, clock, start, end, request)
        accepted = verdict.get("decision") == "pass"
        members.append({
            "candidate_id": cid,
            "mean_rank_ic": row.get("mean_rank_ic"),
            "accepted": accepted,
            "reason_code": verdict.get("reason_code"),
            "max_corr": verdict.get("max_corr"),
            "redundant_with": verdict.get("redundant_with"),
            "pool_size_before": len(pool),
        })
        if accepted:
            pool.append({"candidate_id": cid, "scores": scores})
    return {
        "schema_version": "fuxi.final_admission.1",
        "rule": RULE,
        "eligible_count": len(eligible),
        "accepted_count": sum(1 for m in members if m["accepted"]),
        "members": members,
        "accepted_ids": [m["candidate_id"] for m in members if m["accepted"]],
        "rule_hash": digest({"rule": RULE, "members": members}),
    }
