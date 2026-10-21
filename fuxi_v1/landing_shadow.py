"""P5 production wiring: shadow hooks for the P1-P4 contract artefacts (v2.0 landing).

Every hook is shadow-only: it writes ``landing_shadow`` / ``landing_shadow_error``
facts and NEVER modifies production decision objects (chosen arm, proposals,
summaries, archive all stay on the historical path). Every hook is fail-closed:
any exception lands an error fact and never propagates into the campaign loop.
Controlled by ``FUXI_V1_P5_SHADOW`` (default on since v0.7.186, four-gate verified;
explicit off restores the shadow-disabled historical path; see effective_config).

Production-arm -> five-arm mapping is explicit and complete (tested); the five
arms replace the fixed 40/30/20/10 split only as a shadow reading for now.
"""

import os

from .contracts import ContractError, digest, require

PRODUCTION_ARM_TO_V2 = {
    "fixed": "A_template",
    "refine": "B_edit",
    "combine": "C_combine",
    "regime_combine": "C_combine",
    "llm": "D_knowledge",
    "random": "E_explore",
    "random_ast": "E_explore",
    "gflownet": "E_explore",
}

# Production emitter set covered by the mapping (completeness is asserted in tests).
PRODUCTION_ARMS = ("fixed", "random", "random_ast", "refine", "combine", "regime_combine", "gflownet", "llm")


def p5_shadow_mode():
    mode = os.environ.get("FUXI_V1_P5_SHADOW", "on")
    require(mode in {"off", "on"}, "unknown p5 shadow mode")
    return mode


# ----------------------------------------------------------------- AST surface

def ast_surface(ast):
    """Extract (required_fields, operators, max_window) from a canonical AST."""
    fields, ops, max_window = set(), set(), None

    def walk(node):
        nonlocal max_window
        if "field" in node:
            fields.add(node["field"])
        elif "op" in node:
            ops.add(node["op"])
        if "window" in node:
            max_window = node["window"] if max_window is None \
                else max(max_window, node["window"])
        for arg in node.get("args", []):
            walk(arg)
    walk(ast)
    return sorted(fields), sorted(ops), max_window


# ----------------------------------------------------------------- shadow state

class ShadowState:
    """Cross-round shadow ledger state, replayed from facts on resume.

    P5.1 (v0.7.187): the two shadow selectors each carry their own persistent
    ``arms_state`` accumulator (dispatch/resolve counters + r_dev sums). Keeping
    them separate means round_robin dispatches never pollute the UCB cold-start
    logic, and hook 3 closes the microbatches hook 1 opened so the UCB branch
    actually reaches ``ucb_priority`` after the cold rounds.
    """

    def __init__(self, enabled):
        from .queue import CooldownTracker, DedupLedger
        from .complement import FoldIsolatedFeedback
        self.enabled = enabled
        self.dedup = DedupLedger()
        self.cooldown = CooldownTracker()
        self.feedback = FoldIsolatedFeedback()
        self.error_count = 0
        self.arms_state_rr = {}
        self.arms_state_ucb = {}
        self.pending_shadow = {}

    def replay(self, database):
        """Rebuild dedup/cooldown/arms state from previously recorded shadow facts."""
        for _, fact in database.facts("landing_shadow"):
            if fact.get("kind") == "dedup_cooldown":
                for entry in fact.get("proposals", []):
                    self.dedup.record(entry["ast_identity"], entry["contract_identity"],
                                      entry["candidate_id"])
                self.cooldown.record(fact.get("arm"), fact.get("new_behavior_count", 0))
            # Selector ledgers restore from full snapshots. Facts replay in
            # rowid order, so a round's later dedup_cooldown (post-resolve)
            # snapshot overrides its earlier arm_selection (post-dispatch) one.
            if fact.get("kind") in ("arm_selection", "dedup_cooldown"):
                if fact.get("arms_state_rr") is not None:
                    self.arms_state_rr = fact["arms_state_rr"]
                if fact.get("arms_state_ucb") is not None:
                    self.arms_state_ucb = fact["arms_state_ucb"]


def _append_once(database, fact_type, fact_id, payload):
    """Idempotent append: resume never writes the same shadow fact twice."""
    if not database.db.execute("SELECT 1 FROM facts WHERE id=?",
                               (fact_id,)).fetchone():
        database.append(fact_type, payload, fact_id)


def _safe(state, database, campaign_id, kind, round_index, build):
    """Fail-closed wrapper: exceptions land landing_shadow_error, never raise."""
    fact_id = campaign_id + ":landing_shadow:" + kind + ":" + str(round_index)
    try:
        payload = build()
        _append_once(database, "landing_shadow", fact_id, payload)
    except Exception as exc:
        state.error_count += 1
        try:
            database.append("landing_shadow_error",
                            {"kind": kind, "round": round_index,
                             "error_type": type(exc).__name__, "reason": str(exc)},
                            fact_id + ":error:" + str(state.error_count))
        except Exception:
            pass  # evidence write must never disturb the production run


# ----------------------------------------------------------------- hooks

def hook_arm_selection(state, database, campaign_id, round_index, production_arm,
                       history, budget_remaining, llm_authorized, valid_parents,
                       computable_parents, template_slots_remaining):
    """Shadow reading: five-arm eligibility mask + round_robin / ucb selections."""
    from .arms import arm_eligibility, select_action
    if production_arm not in PRODUCTION_ARM_TO_V2:
        raise ContractError("production arm unmapped: " + str(production_arm))
    base_context = {"budget_remaining": int(budget_remaining),
                    "supported_fields": ["close"],  # eligibility only checks emptiness
                    "supported_operators": ["rank"],
                    "pending_microbatch": False,
                    "template_slots_remaining": bool(template_slots_remaining),
                    "valid_parents": bool(valid_parents),
                    "computable_parents": list(computable_parents or ()),
                    "combination_already_tried": False,
                    "new_knowledge_available": False,
                    "llm_authorized": bool(llm_authorized),
                    "cooldown": False,
                    "unexplored_legal_space": True}

    def run(selector, arms_state):
        context = dict(base_context)
        receipt = select_action(selector=selector, arms_state=arms_state,
                                context=context, seed=round_index,
                                selection_index=round_index)
        return receipt

    rr = run("round_robin", state.arms_state_rr)
    ucb = run("ucb_shadow", state.arms_state_ucb)
    state.pending_shadow[round_index] = {"rr": rr["chosen_arm"], "ucb": ucb["chosen_arm"]}
    return {"kind": "arm_selection", "round": round_index,
            "production_arm": production_arm,
            "production_arm_v2": PRODUCTION_ARM_TO_V2[production_arm],
            "round_robin": {"chosen": rr["chosen_arm"], "reason": rr["reason"],
                            "eligible": rr["eligible_arms"],
                            "masked": rr["masked_reasons"]},
            "ucb_shadow": {"chosen": ucb["chosen_arm"], "reason": ucb["reason"],
                           "eligible": ucb["eligible_arms"],
                           "masked": ucb["masked_reasons"]},
            "divergence_rr_vs_production": rr["chosen_arm"] != PRODUCTION_ARM_TO_V2[production_arm],
            "history_rounds": len(history),
            "arms_state_rr": {arm: dict(st) for arm, st in state.arms_state_rr.items()},
            "arms_state_ucb": {arm: dict(st) for arm, st in state.arms_state_ucb.items()}}


def hook_g0_precheck(state, database, campaign_id, round_index, proposals,
                     fields, operators, session_count):
    """Shadow reading: G0 legality check per proposal (never withholds)."""
    from .gates import g0_precheck
    surface = {"fields": set(fields), "operators": set(operators),
               "min_session_count": session_count, "units": {}}
    per_candidate = []
    for proposal in proposals:
        required, ops, lookback = ast_surface(proposal["canonical_ast"])
        spec = {"required_fields": required, "operators": ops, "lookback": lookback}
        g0 = g0_precheck(spec, surface)
        per_candidate.append({"candidate_id": proposal["candidate_id"],
                              "formula_hash": proposal["formula_hash"],
                              "eval_status": g0["eval_status"],
                              "reason_code": g0["reason_code"],
                              "capability_status": g0["capability_status"]})
    would_block = [c for c in per_candidate if c["eval_status"] != "valid"]
    return {"kind": "g0_precheck", "round": round_index,
            "checked": len(per_candidate), "would_block": len(would_block),
            "would_block_candidates": would_block, "per_candidate": per_candidate}


def hook_dedup_cooldown(state, database, campaign_id, round_index, production_arm,
                        proposals, contract_ref):
    """Shadow reading: dedup ledger + arm cooldown (production ``seen`` still rules)."""
    entries = []
    for proposal in proposals:
        verdict = state.dedup.record(proposal["formula_hash"], contract_ref,
                                     proposal["candidate_id"])
        entries.append({"candidate_id": proposal["candidate_id"],
                        "formula_hash": proposal["formula_hash"],
                        "verdict": verdict["verdict"],
                        "ast_identity": proposal["formula_hash"],
                        "contract_identity": contract_ref})
    new_behavior_count = sum(1 for e in entries if e["verdict"] == "new")
    cooldown = state.cooldown.record(production_arm, new_behavior_count)
    # P5.1: close the shadow microbatches opened by hook 1. r_dev is the
    # production round's share of new-behaviour proposals — the offline
    # counterfactual reading attributed to the arm each shadow selector chose
    # (production outcome, shadow arm). Zero-output rounds score 0.0; the
    # resolve loop drains any pending left over by a crash-resume replay.
    from .arms import resolve_microbatch
    r_dev = float(new_behavior_count) / max(1, len(entries))
    resolve_info = {}
    for selector, chosen_arm in (("round_robin", state.pending_shadow.get(round_index, {}).get("rr")),
                                 ("ucb_shadow", state.pending_shadow.get(round_index, {}).get("ucb"))):
        ledger = state.arms_state_rr if selector == "round_robin" else state.arms_state_ucb
        resolved = 0
        arm_state = ledger.get(chosen_arm, {}) if chosen_arm else {}
        while chosen_arm and arm_state.get("n_pending", 0) > 0:
            arm_state = resolve_microbatch(chosen_arm, r_dev, ledger)
            resolved += 1
        resolve_info[selector] = {"arm": chosen_arm, "r_dev": r_dev,
                                  "resolved_count": resolved,
                                  "arm_state": dict(arm_state) if chosen_arm else None}
    return {"kind": "dedup_cooldown", "round": round_index, "arm": production_arm,
            "proposals": entries,
            "new_attempts": state.dedup.new_attempts,
            "duplicate_attempts": state.dedup.duplicate_attempts,
            "new_behavior_count": new_behavior_count,
            "cooldown": cooldown,
            "resolve": resolve_info,
            "arms_state_rr": {arm: dict(st) for arm, st in state.arms_state_rr.items()},
            "arms_state_ucb": {arm: dict(st) for arm, st in state.arms_state_ucb.items()}}


def hook_g1_complement(state, database, campaign_id, round_index, fold_segment,
                       per_candidate):
    """Shadow reading: G1 diagnostics + complement routing + fold-isolated feedback.

    ``per_candidate``: list of dicts with candidate_id, hypothesis_id, parent_ids,
    source_refs, generation_action_id, daily_ic, daily_support, evaluation_bundle.
    """
    from .gates import g1_diagnostics
    from .complement import record_feedback, task_role_route
    rows = []
    for item in per_candidate:
        g1 = g1_diagnostics(item["daily_ic"], daily_support=item["daily_support"])
        route, reason = task_role_route(role="standalone", g1=g1)
        event = record_feedback(
            generation_action_id=item["generation_action_id"],
            candidate_id=item["candidate_id"],
            hypothesis_id=item["hypothesis_id"],
            parent_ids=item["parent_ids"], source_refs=item["source_refs"],
            evaluation_bundle=item["evaluation_bundle"],
            fold_segment=list(fold_segment),
            objective_kind="predictive",
            outcome_summary={"route": route, "reason_code": reason,
                             "rank_ic_mean": g1["rank_ic_mean"],
                             "icir_unannualized": g1["icir_unannualized"]})
        state.feedback.ingest(event)
        rows.append({"candidate_id": item["candidate_id"],
                     "rank_ic_mean": g1["rank_ic_mean"],
                     "icir_unannualized": g1["icir_unannualized"],
                     "n_sections": g1["n_sections"], "route": route,
                     "reason_code": reason})
    visible = len(state.feedback.view(fold_segment))
    return {"kind": "g1_complement", "round": round_index,
            "fold_segment": list(fold_segment), "rows": rows,
            "fold_isolated_view_count": visible,
            "fold_isolation_holds": visible == len(state.feedback.events)}


def hook_export(state, database, campaign_id, round_index, items, code_ref,
                config_ref, input_contract_ref, market=None, research_window=None):
    """Shadow reading: StrategyIR contract check + frozen export (no auto-promotion).

    ``items``: list of (evaluation_id, candidate_id, strategy_ir_or_None) for the
    round; one shadow fact per round.

    Native execution IRs (portfolio.compile_strategy shape) are bridged onto the
    R2 delivery contract before validation, using ``market`` (execution
    initial_cash -> capital) and ``research_window`` ([start, end] -> train_scope).
    Missing bridge context is recorded as ``contract_gap``, never fabricated.
    """
    from .export import delivery_strategy_ir, freeze_export, is_native_ir, validate_strategy_ir
    rows = []
    for evaluation_id, candidate_id, strategy_ir in items:
        if not isinstance(strategy_ir, dict):
            rows.append({"evaluation_id": evaluation_id, "candidate_id": candidate_id,
                         "contract_check": "no_strategy_ir", "frozen_digest": None})
            continue
        ir = strategy_ir
        try:
            if is_native_ir(strategy_ir):
                require(market is not None and isinstance(market.get("execution"), dict)
                        and market["execution"].get("initial_cash") is not None,
                        "bridge context market.execution.initial_cash required")
                require(research_window is not None and len(research_window) == 2
                        and all(isinstance(x, str) for x in research_window),
                        "bridge context research_window [start, end] required")
                ir = delivery_strategy_ir(
                    strategy_ir,
                    train_scope=research_window,
                    capital=market["execution"]["initial_cash"])
            validate_strategy_ir(ir)
            ir_ok, reason = True, None
        except Exception as exc:
            ir_ok, reason = False, "contract_gap:" + type(exc).__name__ + ":" + str(exc)[:200]
        bundle = None
        if ir_ok:
            bundle = freeze_export(ir, code_ref=code_ref, config_ref=config_ref,
                                   input_contract_ref=input_contract_ref)
        rows.append({"evaluation_id": evaluation_id, "candidate_id": candidate_id,
                     "contract_check": "valid" if ir_ok else reason,
                     "bridged": ir is not strategy_ir,
                     "frozen_digest": bundle["frozen_digest"] if bundle else None})
    return {"kind": "export", "round": round_index, "rows": rows,
            "contract_valid": sum(1 for r in rows if r["contract_check"] == "valid")}
