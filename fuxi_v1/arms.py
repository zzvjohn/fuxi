"""Five generation arms: eligibility mask, unified counting and selectors (P2, v2.0 landing).

R2 blueprint sections 4-5. The five arms replace the fixed 40/30/20/10 split
routing: A template instantiation, B local edit, C complementary combine,
D restricted knowledge proposal, E typed random exploration.

* ``arm_eligibility`` computes the qualification mask with concrete reasons
  (empty parents, missing fields, duplicates, in-flight microbatches, budget
  exhaustion) — eligibility precedes exploration, so a masked arm is never
  forced to run.
* ``select_action`` is one function with two selector branches:
  - ``round_robin`` (P2 baseline): fixed rotation over eligible arms; every 5th
    executable selection goes to the least-dispatched eligible arm.
  - ``ucb_shadow`` (shadow): cold start for newly eligible arms, the same 5th-
    selection forced exploration, otherwise the interpretable UCB priority
    ``mean(r_dev,a) + sqrt(2 ln(max(2,N)) / n_resolved(a))``. r_dev is the
    per-microbatch share of task-relevant, G0-valid, G1-diagnosed candidates
    with new behaviour over declared slots, in [0,1]; null (external failure)
    blocks and never enters the mean. Zero-output deterministic failures count
    as 0. Auer et al. regret guarantees do NOT transfer (non-i.i.d. arms).
* Dispatch counts immediately (``n_dispatched``/``n_pending``); an arm has at
  most one in-flight microbatch; no reward is ever disguised as a return.
"""

from math import log, sqrt
from random import Random

from .contracts import ContractError, digest, keys, require

ARMS = ("A_template", "B_edit", "C_combine", "D_knowledge", "E_explore")
SELECTORS = ("round_robin", "ucb_shadow")
RECEIPT_SCHEMA = "fuxi.action_receipt.1"


def arm_eligibility(arm, context):
    """Qualification mask for one arm. Returns (eligible, [masked_reasons])."""
    require(arm in ARMS, "unknown arm")
    require(isinstance(context, dict), "context required")
    reasons = []
    if context.get("pending_microbatch"):
        reasons.append("in_flight")
    if context.get("budget_remaining", 0) <= 0:
        reasons.append("budget_exhausted")
    supported_fields = set(context.get("supported_fields") or ())
    supported_operators = set(context.get("supported_operators") or ())
    if not supported_fields:
        reasons.append("field_surface_empty")
    if arm == "A_template":
        if not context.get("template_slots_remaining"):
            reasons.append("template_space_exhausted")
    elif arm == "B_edit":
        if not context.get("valid_parents"):
            reasons.append("no_valid_parent")
    elif arm == "C_combine":
        if len(context.get("computable_parents", ())) < 2:
            reasons.append("needs_two_parents")
        if context.get("combination_already_tried"):
            reasons.append("duplicate_combination")
    elif arm == "D_knowledge":
        if not context.get("new_knowledge_available"):
            reasons.append("no_new_knowledge")
        if not context.get("llm_authorized"):
            reasons.append("llm_not_authorized")
    elif arm == "E_explore":
        if context.get("cooldown"):
            reasons.append("cooldown")
        if not context.get("unexplored_legal_space"):
            reasons.append("exploration_space_exhausted")
    eligible = not reasons
    return eligible, sorted(set(reasons))


def arm_priority(arm, state, total_resolved):
    """Interpretable UCB priority; None when the arm has no resolved reward."""
    st = state.get(arm, {})
    n_resolved = st.get("n_resolved", 0)
    r_sum = st.get("r_dev_sum", 0.0)
    r_count = st.get("r_dev_count", 0)
    if n_resolved <= 0 or r_count <= 0:
        return None
    mean = r_sum / r_count
    bonus = sqrt(2 * log(max(2, total_resolved)) / n_resolved)
    return mean + bonus


def select_action(*, selector, arms_state, context, seed, selection_index):
    """One selection step; writes the ActionReceipt and updates dispatch counts.

    ``arms_state``: {arm: {n_dispatched, n_resolved, n_pending, r_dev_sum,
    r_dev_count}}. ``context``: eligibility context (see arm_eligibility).
    """
    require(selector in SELECTORS, "unknown selector")
    require(type(selection_index) is int and selection_index >= 0, "invalid selection index")
    eligible, masked = [], {}
    for arm in ARMS:
        ok, reasons = arm_eligibility(arm, context)
        if ok:
            eligible.append(arm)
        else:
            masked[arm] = reasons
    if not eligible:
        return _receipt(None, "idle_exhausted", eligible, masked, arms_state,
                        context, seed, selection_index)
    if selector == "round_robin":
        if selection_index % 5 == 4:
            chosen = min(eligible, key=lambda a: arms_state.get(a, {}).get("n_dispatched", 0))
            reason = "forced_explore"
        else:
            chosen = eligible[selection_index % len(eligible)]
            reason = "fixed_rotation"
    else:  # ucb_shadow
        cold = [a for a in eligible if arms_state.get(a, {}).get("n_dispatched", 0) == 0]
        if cold:
            chosen, reason = cold[0], "cold_start"
        elif selection_index % 5 == 4:
            chosen = min(eligible, key=lambda a: arms_state.get(a, {}).get("n_dispatched", 0))
            reason = "forced_explore"
        else:
            total = sum(st.get("n_resolved", 0) for st in arms_state.values())
            ranked = [(arm_priority(a, arms_state, total), a) for a in eligible]
            ranked = [(p, a) for p, a in ranked if p is not None]
            if not ranked:
                chosen, reason = cold_choice_fallback(eligible, arms_state), "no_resolved_priority"
            else:
                chosen = max(ranked, key=lambda pair: pair[0])[1]
                reason = "ucb_priority"
    _dispatch(chosen, arms_state, context)
    return _receipt(chosen, reason, eligible, masked, arms_state, context,
                    seed, selection_index)


def cold_choice_fallback(eligible, arms_state):
    # No arm has a resolved reward yet: use least-dispatched (fixed seed order
    # is unnecessary here since ties are broken by ARMS order).
    return min(eligible, key=lambda a: arms_state.get(a, {}).get("n_dispatched", 0))


def resolve_microbatch(arm, r_dev, arms_state):
    """Close an in-flight microbatch. r_dev is a float in [0,1] or None."""
    require(arm in ARMS, "unknown arm")
    st = arms_state.setdefault(arm, {"n_dispatched": 0, "n_resolved": 0, "n_pending": 0,
                                     "r_dev_sum": 0.0, "r_dev_count": 0})
    require(st.get("n_pending", 0) > 0, "no in-flight microbatch to resolve")
    st["n_pending"] -= 1
    st["n_resolved"] += 1
    if r_dev is not None:
        require(type(r_dev) is float and 0.0 <= r_dev <= 1.0, "r_dev must be in [0,1] or None")
        st["r_dev_sum"] += r_dev
        st["r_dev_count"] += 1
    return dict(st)


def _dispatch(arm, arms_state, context):
    st = arms_state.setdefault(arm, {"n_dispatched": 0, "n_resolved": 0, "n_pending": 0,
                                     "r_dev_sum": 0.0, "r_dev_count": 0})
    st["n_dispatched"] += 1
    st["n_pending"] += 1
    context["budget_remaining"] = max(0, context.get("budget_remaining", 0) - 1)


def _receipt(chosen, reason, eligible, masked, arms_state, context, seed, selection_index):
    receipt = {"schema_version": RECEIPT_SCHEMA, "chosen_arm": chosen, "reason": reason,
               "eligible_arms": list(eligible), "masked_reasons": dict(masked),
               "arms_state": {arm: dict(st) for arm, st in arms_state.items()},
               "budget_reserved": context.get("budget_remaining", 0),
               "seed": seed, "selection_index": selection_index,
               "candidates": context.get("candidates", [])}
    require(receipt["reason"] in ("cold_start", "forced_explore", "ucb_priority",
                                  "fixed_rotation", "no_resolved_priority", "idle_exhausted"),
            "unknown selection reason")
    digest(receipt)
    return receipt
