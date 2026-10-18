"""Complement routing, risk tasks, channel pruning and fold-isolated feedback (P3, v2.0 landing).

R2 blueprint sections 3 / 6 / 8:

* Task roles decide the objective BEFORE results are seen: standalone asks
  "can it be a useful independent strategy", complement asks "does B+candidate
  add paired net increment (marginal IC tolerated)", risk_condition asks "does
  it cut the pre-declared risk metric within the pre-declared loss budget".
  A task role is never rewritten after seeing results.
* Multi-channel forms are pruned on the shortlist boundary with single-channel
  stripping on the same support (B+P, B+V, B+PV); an interaction claim beyond
  simple two-feature addition must additionally show B+P+V.
* Feedback updates only real participants (lineage), records engineering /
  predictive / account objectives separately, and is fold-isolated: an event
  from one fold's validation segment never touches another fold's memory view.
"""

from .contracts import ContractError, digest, keys, require
from .routing import validate_screen_result

FEEDBACK_SCHEMA = "fuxi.feedback_event.1"
TASK_ROLES = ("standalone", "complement", "risk_condition")
OBJECTIVE_KINDS = ("engineering", "predictive", "account")

# Same-support similarity above this is flagged as a near neighbour (G1 hint).
NEAR_NEIGHBOUR_ABS_CORR = 0.95


def task_role_route(*, role, g1, similarity=None, risk_metric=None):
    """Routing hint per declared task role. Returns (route, reason_code)."""
    require(role in TASK_ROLES, "unknown task role")
    if role == "risk_condition":
        require(risk_metric is not None, "risk tasks must carry the pre-declared risk metric")
        improved = risk_metric.get("improved")
        within_budget = risk_metric.get("loss_within_budget")
        if improved is None or within_budget is None:
            return ("park", "risk_objective_not_yet_measurable")
        return ("shortlist", "risk_objective_met") if (improved and within_budget) \
            else ("park", "risk_objective_not_met")
    if similarity is not None and abs(similarity) >= NEAR_NEIGHBOUR_ABS_CORR:
        return ("duplicate", "same_support_neighbour")
    if role == "complement":
        # Marginal IC is tolerated; complement needs its own evaluation channel,
        # so a G0/G1-valid candidate routes on even when IC hints are low.
        return ("shortlist", "complement_channel")
    # standalone: weak IC parks (routing hint only, never an economic rejection)
    ic = g1.get("rank_ic_mean")
    icir = g1.get("icir_unannualized")
    if ic is None or icir is None:
        return ("park", "signal_not_diagnosable")
    return ("shortlist", "standalone_priority") if ic >= 0.01 or icir >= 0.10 \
        else ("park", "weak_ic_routing_hint")


def channel_ablation_plan(channels, *, interaction_claim=False):
    """Comparisons needed before a multi-channel form enters research assets.

    Channels is a non-empty list of channel ids (e.g. ["P", "V"]). The plan
    always contains B+each single channel and B+all channels; an interaction
    claim (product interaction beyond simple addition) additionally requires
    B+P+V on top of B+PV.
    """
    require(isinstance(channels, list) and channels and all(isinstance(c, str) for c in channels),
            "channels must be a non-empty list of ids")
    channels = list(dict.fromkeys(channels))  # dedupe, keep order
    plan = [tuple([channel]) for channel in channels]
    if len(channels) >= 2:
        plan.append(tuple(channels))
        if interaction_claim:
            # explicit interaction contrast on top of the simple combination
            plan.append(tuple(["interaction"] + channels))
    return [("baseline",) + combo for combo in plan]


def prune_multi_channel(comparisons, target_metric):
    """If the complex form does not improve the target, keep the simpler
    representative. ``comparisons`` maps combo-tuples to {target_metric: value}.
    Returns (kept_combo, reason)."""
    require(isinstance(comparisons, dict) and comparisons, "comparisons required")
    best = None
    for combo, metrics in comparisons.items():
        require(target_metric in metrics, "missing target metric")
        value = metrics[target_metric]
        if best is None or value > comparisons[best][target_metric]:
            best = combo
    return (best, "best_on_target" if best else "no_data")


def record_feedback(*, generation_action_id, candidate_id, hypothesis_id, parent_ids,
                    source_refs, evaluation_bundle, fold_segment, objective_kind,
                    outcome_summary):
    """Build + validate a feedback event over ACTUAL lineage only.

    ``parent_ids``/``source_refs`` must be the real participants of
    ``generation_action_id`` (caller verifies provenance before calling); the
    event records engineering / predictive / account objectives separately.
    """
    require(isinstance(generation_action_id, str) and generation_action_id, "action id required")
    require(isinstance(candidate_id, str) and candidate_id, "candidate id required")
    require(isinstance(hypothesis_id, str) and hypothesis_id, "hypothesis id required")
    require(isinstance(parent_ids, list) and all(isinstance(p, str) for p in parent_ids),
            "parent_ids must be a list of ids")
    require(isinstance(source_refs, list) and all(isinstance(s, str) for s in source_refs),
            "source_refs must be a list of ids")
    require(isinstance(evaluation_bundle, dict) and evaluation_bundle, "evaluation_bundle required")
    require(isinstance(fold_segment, list) and len(fold_segment) == 2
            and all(isinstance(s, str) for s in fold_segment), "fold_segment must be [start, end]")
    require(objective_kind in OBJECTIVE_KINDS, "unknown objective kind")
    event = {"schema_version": FEEDBACK_SCHEMA,
             "generation_action_id": generation_action_id,
             "candidate_id": candidate_id,
             "hypothesis_id": hypothesis_id,
             "parent_ids": sorted(parent_ids),
             "source_refs": sorted(source_refs),
             "fold_segment": list(fold_segment),
             "objective_kind": objective_kind,
             "outcome_summary": outcome_summary,
             "evaluation_digest": digest(evaluation_bundle)}
    digest(event)
    return event


class FoldIsolatedFeedback:
    """Feedback ledger with fold isolation.

    An event carrying fold_segment F may only update the memory view of F's own
    scope (its train/validate context). A view built for a different fold never
    sees it — this is the "折内隔离" acceptance.
    """

    def __init__(self):
        self.events = []

    def ingest(self, event):
        require(event["schema_version"] == FEEDBACK_SCHEMA, "invalid feedback event")
        self.events.append(event)
        return len(self.events)

    def view(self, fold_segment):
        require(isinstance(fold_segment, list) and len(fold_segment) == 2, "fold_segment must be a pair")
        return [event for event in self.events if event["fold_segment"] == list(fold_segment)]

    def lineage_covered(self, fold_segment):
        """All source/parent ids actually updated inside this fold."""
        ids = set()
        for event in self.view(fold_segment):
            ids.update(event["parent_ids"])
            ids.update(event["source_refs"])
        return ids
