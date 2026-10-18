"""G0/G1/G2 route records and five-state status handling (P0.3, v2.0 landing).

Lands the CONTRACTS.md "一次求值与筛选接口" ScreenResult and the
"状态和生命周期" five-state discipline:

* The five evaluation states ``valid / data_insufficient / precheck_rejected /
  error / not_evaluated`` describe "can it be measured"; the route decision
  ``shortlist / park / duplicate`` describes "how is it routed"; evidence_scope
  describes "what conclusion does it support". The three never collapse into
  one pass boolean.
* ``unsupported`` stays a capability tag, NOT a sixth evaluation state.
* Old E0-E5 evaluation objects are preserved verbatim under ``legacy_refs``
  (adapted, referenced, never stuffed into the new enums). Old pass/reject
  booleans never grant new qualification: ``formal_qualification`` stays null
  unless a versioned AdmissionService grants it later.
* An old economic stage that was never evaluated maps to ``not_evaluated``, not
  to a rejection.
"""

from .contracts import ContractError, digest, keys, require, validate_evaluation

SCREEN_RESULT_SCHEMA = "fuxi.screen_result.1"

FIVE_STATES = ("valid", "data_insufficient", "precheck_rejected", "error", "not_evaluated")
ROUTE_DECISIONS = ("shortlist", "park", "duplicate")
CAPABILITY_TAGS = ("unsupported",)  # tags only; never appended to FIVE_STATES
EVIDENCE_SCOPES = ("diagnostic", "research", "account", "forward_acceptance")
EXPOSURE_LEVELS = ("research_exposed", "unexposed")


def validate_screen_result(result):
    keys(result, ("schema_version", "candidate_id", "eval_status", "decision",
                  "reason_code", "evidence_scope", "exposure", "support_ref",
                  "n_sections", "n_pairs", "rank_ic_mean", "icir_unannualized",
                  "signal_cluster_id", "selection_route", "formal_qualification",
                  "legacy_refs"),
        ("capability_status",))
    require(result["schema_version"] == SCREEN_RESULT_SCHEMA, "unsupported screen result schema")
    require(result["eval_status"] in FIVE_STATES, "status outside the five evaluation states")
    require(result["decision"] in ROUTE_DECISIONS, "invalid route decision")
    require(result["formal_qualification"] is None,
            "router must not grant qualification directly")
    require(result["evidence_scope"] in EVIDENCE_SCOPES, "invalid evidence scope")
    require(result["exposure"] in EXPOSURE_LEVELS, "invalid exposure level")
    require(result["reason_code"] is None or isinstance(result["reason_code"], str),
            "reason_code must be a string or null")
    require(result["selection_route"] is None or isinstance(result["selection_route"], str),
            "selection_route must be a string or null")
    require(result["signal_cluster_id"] is None or isinstance(result["signal_cluster_id"], str),
            "signal_cluster_id must be a string or null")
    if "capability_status" in result:
        require(result["capability_status"] in CAPABILITY_TAGS,
                "unknown capability tag")
    require(type(result["n_sections"]) is int and result["n_sections"] >= 0, "invalid n_sections")
    require(type(result["n_pairs"]) is int and result["n_pairs"] >= 0, "invalid n_pairs")
    for metric in ("rank_ic_mean", "icir_unannualized"):
        value = result[metric]
        require(value is None or (type(value) in (int, float) and value == value),
                "invalid " + metric)
    for ref in result["legacy_refs"]:
        keys(ref, ("evaluation", "digest"))
        validate_evaluation(ref["evaluation"])
        require(ref["digest"] == digest(ref["evaluation"]), "legacy ref digest mismatch")
    digest(result)
    return result


def _status_from_legacy(evaluations):
    """Derive the five-state measurement status from legacy E0-E5 objects.

    Old economic rejects do NOT become an evaluation status: status describes
    measurability, not disposition.
    """
    statuses = [e["eval_status"] for e in evaluations]
    if any(s == "error" for s in statuses):
        return "error"
    if any(s == "precheck_rejected" for s in statuses):
        return "precheck_rejected"
    if any(s == "data_insufficient" for s in statuses) and not any(s == "valid" for s in statuses):
        return "data_insufficient"
    if any(s == "valid" for s in statuses):
        return "valid"
    return "not_evaluated"


def route_from_legacy(candidate_id, legacy_evaluations, *, diagnostics=None,
                      evidence_scope="diagnostic", exposure="research_exposed",
                      capability_status=None):
    """Build a new ScreenResult that references (never rewrites) old evaluations.

    ``legacy_evaluations`` is a list of dicts produced by ``contracts.evaluation``
    (E0-E5). The adapter keeps them verbatim; nothing old is silently imported
    into the new enums. ``diagnostics`` may carry descriptive readings
    (n_sections/n_pairs/rank_ic_mean/icir_unannualized/signal_cluster_id/
    selection_route) already computed by the new G1 diagnostic pass.
    """
    require(isinstance(legacy_evaluations, list), "legacy_evaluations must be a list")
    for evaluation in legacy_evaluations:
        validate_evaluation(evaluation)
    diagnostics = diagnostics or {}
    status = _status_from_legacy(legacy_evaluations)
    decision = ("shortlist" if status == "valid"
                and not any(e.get("decision") == "reject" for e in legacy_evaluations)
                and any(e.get("decision") == "pass" for e in legacy_evaluations)
                else "park")
    reason_code = None
    if status == "precheck_rejected":
        reasons = sorted({e.get("reason_code") for e in legacy_evaluations
                          if e.get("eval_status") == "precheck_rejected" and e.get("reason_code")})
        reason_code = ";".join(reasons) if reasons else None
    if capability_status is not None:
        require(capability_status in CAPABILITY_TAGS, "unknown capability tag")
    result = {"schema_version": SCREEN_RESULT_SCHEMA,
              "candidate_id": candidate_id,
              "eval_status": status,
              "decision": decision,
              "reason_code": reason_code,
              "evidence_scope": evidence_scope,
              "exposure": exposure,
              "support_ref": diagnostics.get("support_ref"),
              "n_sections": diagnostics.get("n_sections", 0),
              "n_pairs": diagnostics.get("n_pairs", 0),
              "rank_ic_mean": diagnostics.get("rank_ic_mean"),
              "icir_unannualized": diagnostics.get("icir_unannualized"),
              "signal_cluster_id": diagnostics.get("signal_cluster_id"),
              "selection_route": diagnostics.get("selection_route"),
              "formal_qualification": None,
              "legacy_refs": [{"evaluation": e, "digest": digest(e)} for e in legacy_evaluations]}
    if capability_status is not None:
        result["capability_status"] = capability_status
    validate_screen_result(result)
    return result


def route_duplicate(candidate_id, signal_cluster_id, *, diagnostics=None,
                    evidence_scope="research", exposure="research_exposed"):
    """Mark a same-support near-neighbour as duplicate (routing only)."""
    diagnostics = diagnostics or {}
    result = {"schema_version": SCREEN_RESULT_SCHEMA,
              "candidate_id": candidate_id,
              "eval_status": "valid",
              "decision": "duplicate",
              "reason_code": "same_support_neighbour",
              "evidence_scope": evidence_scope,
              "exposure": exposure,
              "support_ref": diagnostics.get("support_ref"),
              "n_sections": diagnostics.get("n_sections", 0),
              "n_pairs": diagnostics.get("n_pairs", 0),
              "rank_ic_mean": diagnostics.get("rank_ic_mean"),
              "icir_unannualized": diagnostics.get("icir_unannualized"),
              "signal_cluster_id": signal_cluster_id,
              "selection_route": "duplicate",
              "formal_qualification": None,
              "legacy_refs": []}
    validate_screen_result(result)
    return result
