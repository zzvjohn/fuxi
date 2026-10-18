"""StrategyIR freeze/export, real JQ receipts and frozen forward records (P4, v2.0 landing).

R2 blueprint "冻结交付与真实未来" and CONTRACTS.md lifecycle:

* ``freeze_export`` fixes code/config/model/input contract references into a
  JQExportBundle. File existence is NOT a success signal.
* ``jq_verified`` requires a REAL platform run_id, a frozen strategy version and
  a per-field comparison receipt; screenshots or local simulation are not
  receipts.
* ``forward_observed`` only records that an observation happened; whether the
  target was met is a separate decision field, and release_review never grants
  real brokerage trading permission by itself.
* The release state machine validates transitions with required evidence; no
  path auto-promotes to production defaults.
"""

from .contracts import ContractError, digest, keys, require

STRATEGY_IR_SCHEMA = "fuxi.strategy_ir.1"
EXPORT_BUNDLE_SCHEMA = "fuxi.jq_export_bundle.1"
JQ_RECEIPT_SCHEMA = "fuxi.jq_receipt.1"
FORWARD_RECORD_SCHEMA = "fuxi.forward_record.1"

RELEASE_STATES = ("proposed", "screened", "portfolio_candidate", "parked",
                  "research_selected", "jq_verified", "forward_pending",
                  "forward_observed", "release_review", "monitored", "retired")


def validate_strategy_ir(ir):
    keys(ir, ("schema_version", "candidate_ids", "train_scope", "model_artifacts",
              "prediction_inputs", "rebalance_calendar", "ranking_and_tie_rules",
              "weights", "holdings_count", "constraints", "turnover_buffer",
              "capital", "execution_contract", "risk_conditions"),
         ("score_origin", "native_ir_ref"))
    require(ir["schema_version"] == STRATEGY_IR_SCHEMA, "unsupported StrategyIR schema")
    require(isinstance(ir["candidate_ids"], list) and ir["candidate_ids"], "candidate_ids required")
    require(isinstance(ir["execution_contract"], dict) and ir["execution_contract"],
            "execution contract required")
    require(isinstance(ir["risk_conditions"], list), "risk_conditions must be a list")
    digest(ir)
    return ir


def is_native_ir(ir):
    """True when ``ir`` is the native execution StrategyIR produced by
    ``portfolio.compile_strategy`` (``strategy`` spec + ``decisions``), which
    shares the ``fuxi.strategy_ir.1`` schema string but NOT the R2 delivery
    field set. Such IRs must be bridged before contract validation."""
    return (isinstance(ir, dict) and "strategy" in ir and "decisions" in ir
            and "snapshot_hash" in ir)


def delivery_strategy_ir(native_ir, *, train_scope, capital):
    """Bridge a native execution StrategyIR onto the R2 delivery contract shape
    (CONTRACTS.md: candidate_ids/train_scope/model_artifacts/prediction_inputs/
    rebalance_calendar/ranking_and_tie_rules/weights/holdings_count/constraints/
    turnover_buffer/capital/execution_contract/risk_conditions).

    Field mapping is declarative and lossless with respect to the strategy spec:

    * candidate_ids        <- strategy.candidate_id (single-candidate IR)
    * train_scope          <- caller-provided research window [start, end]
    * model_artifacts      <- [model_hash] when a model produced the scores, else []
    * prediction_inputs    <- the score stream identity ("score_<candidate_id>")
    * rebalance_calendar   <- {type: session_interval, sessions: rebalance_sessions}
    * ranking_and_tie_rules<- strategy.weight_rule
    * weights              <- {candidate_id: 1.0} (single-candidate weight)
    * holdings_count       <- strategy.top_n
    * constraints          <- max_weight / gross_exposure / industry_cap
    * turnover_buffer      <- strategy.turnover_budget
    * capital              <- caller-provided market execution initial_cash
    * execution_contract   <- venue jq + strategy execution/signal/unfilled rules
    * risk_conditions      <- [] (no risk rules exist in the native IR)

    ``train_scope`` and ``capital`` are NOT carried by the native IR; the
    caller must supply them from the campaign request and the market execution
    contract. Missing context raises ContractError (fail-closed, no fabrication).
    """
    require(is_native_ir(native_ir), "native execution StrategyIR required")
    spec = native_ir["strategy"]
    require(isinstance(spec, dict) and spec.get("schema_version") == "fuxi.strategy.1",
            "native IR strategy spec required")
    require(isinstance(train_scope, (list, tuple)) and len(train_scope) == 2
            and all(isinstance(x, str) for x in train_scope),
            "train_scope must be [start, end] session strings")
    require(capital is not None, "capital (market execution initial_cash) required")
    candidate_id = spec["candidate_id"]
    require(isinstance(candidate_id, str) and candidate_id, "candidate_id required")
    return {
        "schema_version": STRATEGY_IR_SCHEMA,
        "candidate_ids": [candidate_id],
        "train_scope": list(train_scope),
        "model_artifacts": [native_ir["model_hash"]] if native_ir.get("model_hash") else [],
        "prediction_inputs": ["score_" + candidate_id],
        "rebalance_calendar": {"type": "session_interval",
                               "sessions": spec["rebalance_sessions"]},
        "ranking_and_tie_rules": spec["weight_rule"],
        "weights": {candidate_id: 1.0},
        "holdings_count": spec["top_n"],
        "constraints": [{"kind": "max_weight", "value": spec["max_weight"]},
                        {"kind": "gross_exposure", "value": spec["gross_exposure"]},
                        {"kind": "industry_cap", "value": spec["industry_cap"]}],
        "turnover_buffer": spec["turnover_budget"],
        "capital": capital,
        "execution_contract": {"venue": "jq",
                               "execution_rule": spec["execution_rule"],
                               "signal_rule": spec["signal_rule"],
                               "unfilled_order_policy": spec["unfilled_order_policy"]},
        "risk_conditions": [],
        "score_origin": native_ir.get("score_origin"),
        "native_ir_ref": native_ir.get("ir_hash"),
    }


def freeze_export(strategy_ir, *, code_ref, config_ref, input_contract_ref):
    """Freeze a research-selected StrategyIR into a JQ export bundle."""
    validate_strategy_ir(strategy_ir)
    for field, value in (("code_ref", code_ref), ("config_ref", config_ref),
                         ("input_contract_ref", input_contract_ref)):
        require(isinstance(value, str) and len(value) == 64, field + " must be sha256 hex")
    bundle = {"schema_version": EXPORT_BUNDLE_SCHEMA,
              "strategy_ir": strategy_ir,
              "code_ref": code_ref,
              "config_ref": config_ref,
              "input_contract_ref": input_contract_ref,
              "frozen_digest": None}
    bundle["frozen_digest"] = digest({"strategy_ir": strategy_ir, "code_ref": code_ref,
                                      "config_ref": config_ref,
                                      "input_contract_ref": input_contract_ref})
    return bundle


def jq_receipt(*, export_bundle, platform, platform_run_id, per_field_comparison,
               verified_by):
    """A REAL platform receipt. File existence is not acceptable evidence: the
    receipt requires a platform run id and a per-field comparison table."""
    require(export_bundle["schema_version"] == EXPORT_BUNDLE_SCHEMA, "invalid export bundle")
    require(isinstance(platform, str) and platform, "platform required")
    require(isinstance(platform_run_id, str) and platform_run_id, "real platform run id required")
    require(isinstance(per_field_comparison, dict) and per_field_comparison,
            "per-field comparison table required")
    for field, record in per_field_comparison.items():
        keys(record, ("platform_value_digest", "local_value_digest", "match"))
        require(record["match"] is True, "field comparison must match to certify: " + field)
    require(isinstance(verified_by, str) and verified_by, "verified_by required")
    receipt = {"schema_version": JQ_RECEIPT_SCHEMA,
               "export_digest": export_bundle["frozen_digest"],
               "platform": platform,
               "platform_run_id": platform_run_id,
               "per_field_comparison": per_field_comparison,
               "verified_by": verified_by,
               "note": "real platform evidence; local simulation or file existence is not a receipt"}
    digest(receipt)
    return receipt


def forward_record(*, strategy_ref, frozen_version, look_plan, observed_sessions,
                   met_target=None):
    """Frozen forward observation record. ``met_target`` is None until decided;
    observation does not imply target met, and never grants trading permission."""
    require(isinstance(strategy_ref, str) and len(strategy_ref) == 64, "strategy_ref must be sha256 hex")
    require(isinstance(frozen_version, str) and frozen_version, "frozen_version required")
    require(isinstance(look_plan, list) and look_plan, "look_plan required")
    require(isinstance(observed_sessions, int) and observed_sessions >= 0, "invalid session count")
    if met_target is not None:
        require(type(met_target) is bool, "met_target must be a boolean or None")
    record = {"schema_version": FORWARD_RECORD_SCHEMA,
              "strategy_ref": strategy_ref,
              "frozen_version": frozen_version,
              "look_plan": list(look_plan),
              "observed_sessions": observed_sessions,
              "met_target": met_target,
              "note": "observation only; decision fields and release_review are separate"}
    digest(record)
    return record


# The transition table from the CONTRACTS.md state diagram.
RELEASE_TRANSITIONS = {
    ("proposed", "screened"): (),
    ("screened", "portfolio_candidate"): (),
    ("screened", "parked"): (),
    ("portfolio_candidate", "research_selected"): (),
    ("portfolio_candidate", "parked"): (),
    ("research_selected", "jq_verified"): ("receipt",),
    ("jq_verified", "forward_pending"): ("frozen_export",),
    ("forward_pending", "forward_observed"): ("observation",),
    ("forward_observed", "release_review"): (),
    ("release_review", "monitored"): (),
    ("monitored", "retired"): (),
}


def validate_transition(from_state, to_state, evidence):
    """Check a release-state transition with required evidence."""
    require(from_state in RELEASE_STATES, "unknown from_state")
    require(to_state in RELEASE_STATES, "unknown to_state")
    require(isinstance(evidence, dict), "evidence required")
    transition = RELEASE_TRANSITIONS.get((from_state, to_state))
    if transition is None:
        raise ContractError("transition not allowed: %s -> %s" % (from_state, to_state))
    missing = [k for k in transition if not evidence.get(k)]
    if missing:
        raise ContractError("missing transition evidence: %s" % ", ".join(missing))
    return {"from": from_state, "to": to_state, "evidence": sorted(transition)}
