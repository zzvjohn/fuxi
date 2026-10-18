"""Source lineage and bounded MemoryView retrieval (P0.4, v2.0 landing).

Lands the CONTRACTS.md "知识与生成接口" requirements for provenance:

* SourceCard carries real availability time; retrieval filters strictly on
  ``as_of`` so past tasks can never be answered with future winner cards
  (the R2 counterexample row: replaying a historical task must not see cards
  that did not exist yet).
* Rolling folds exclude validation/future segments: results and derived
  rankings computed on a validation segment are invisible to the training
  scope that precedes them.
* Candidate lineage (hypothesis_id / parent_ids / edit_spec /
  generation_action_id / source_refs) is a first-class record; feedback only
  updates real participants.
* The MemoryView reports what was omitted and why (bounded retrieval), instead
  of silently truncating.
"""

from .contracts import ContractError, digest, keys, require, timestamp

SOURCE_CARD_SCHEMA = "fuxi.source_card.1"
MEMORY_VIEW_SCHEMA = "fuxi.memory_view.1"
CANDIDATE_LINEAGE_SCHEMA = "fuxi.candidate_lineage.1"

EVIDENCE_KINDS = ("positive", "negative", "neutral", "revoked")
SOURCE_TYPES = ("paper", "internal_study", "market_observation", "generator_feedback",
                "mechanism_card", "vendor_data")


def source_card(source_id, *, available_at, source_date, source_type, mechanism_tags,
                evidence_kind, recent_status, evidence_window, data_contract_ref=None,
                derived_from=None, revoked_dependencies=()):
    """Build + validate a SourceCard."""
    card = {"schema_version": SOURCE_CARD_SCHEMA, "source_id": source_id,
            "available_at": available_at, "source_date": source_date,
            "source_type": source_type, "mechanism_tags": sorted(mechanism_tags),
            "evidence_kind": evidence_kind, "recent_status": recent_status,
            "evidence_window": evidence_window, "data_contract_ref": data_contract_ref,
            "derived_from": derived_from,
            "revoked_dependencies": sorted(revoked_dependencies)}
    validate_source_card(card)
    return card


def validate_source_card(card):
    keys(card, ("schema_version", "source_id", "available_at", "source_date", "source_type",
                "mechanism_tags", "evidence_kind", "recent_status", "evidence_window",
                "data_contract_ref", "derived_from", "revoked_dependencies"))
    require(card["schema_version"] == SOURCE_CARD_SCHEMA, "unsupported source card schema")
    require(isinstance(card["source_id"], str) and card["source_id"], "source_id required")
    timestamp(card["available_at"])
    timestamp(card["source_date"])
    require(card["source_type"] in SOURCE_TYPES, "invalid source type")
    require(card["evidence_kind"] in EVIDENCE_KINDS, "invalid evidence kind")
    require(isinstance(card["mechanism_tags"], list), "mechanism_tags must be a list")
    for window in card["evidence_window"]:
        require(isinstance(window, list) and len(window) == 2
                and all(isinstance(s, str) for s in window), "evidence_window segments must be [start, end] pairs")
    require(isinstance(card["revoked_dependencies"], list), "revoked_dependencies must be a list")
    if card["derived_from"] is not None:
        keys(card["derived_from"], ("parent_source_id", "segment", "ranking_kind"))
    digest(card)
    return card


def visible_cards(cards, as_of, *, allowed_exposure="research", excluded_segments=(),
                  select=None, max_cards=None):
    """Bounded, as_of-correct MemoryView over source cards.

    Exclusion rules (each one recorded, never silently dropped):

    1. ``available_at > as_of``: the card did not exist yet at the replay clock
       (no future winners leaking into historical tasks).
    2. ``evidence_window`` intersects an excluded segment (rolling-fold
       validation/future exclusion).
    3. ``derived_from.segment`` intersects an excluded segment (derived
       rankings inherit their window).
    4. ``evidence_kind == "revoked"`` or non-empty ``revoked_dependencies``:
       revoked evidence is kept in the ledger but not offered as evidence.

    Returns a MemoryView dict with the selected cards, per-card selection
    reason, the as_of clock, allowed exposure and the omitted accounting
    (count + reasons), so callers can disclose truncation instead of hiding it.
    """
    require(isinstance(cards, list), "cards must be a list")
    clock = timestamp(as_of)
    excluded = set(excluded_segments)
    selected, omitted = [], {}
    for card in cards:
        validate_source_card(card)
        reasons = []
        if timestamp(card["available_at"]) > clock:
            reasons.append("not_available_at_as_of")
        if any(_intersects(window, excluded) for window in card["evidence_window"]):
            reasons.append("segment_excluded")
        derived = card["derived_from"]
        if derived is not None and _intersects(derived["segment"], excluded):
            reasons.append("derived_ranking_segment_excluded")
        if card["evidence_kind"] == "revoked" or card["revoked_dependencies"]:
            reasons.append("revoked")
        if reasons:
            omitted.setdefault(card["source_id"], reasons)
        else:
            selected.append((card, ["as_of_visible"]))
    if select is not None:
        require(callable(select), "select must be callable")
        selected = [(card, reasons + ["selected_by_policy"]) for card, reasons in selected
                    if select(card)]
    if max_cards is not None:
        require(type(max_cards) is int and max_cards >= 0, "invalid max_cards")
        overflow = selected[max_cards:]
        selected = selected[:max_cards]
        for card, _ in overflow:
            omitted.setdefault(card["source_id"], []).append("over_budget")
    return {"schema_version": MEMORY_VIEW_SCHEMA,
            "as_of": as_of,
            "allowed_exposure": allowed_exposure,
            "excluded_segments": sorted(excluded),
            "cards": [{"source_id": card["source_id"], "card": card,
                       "selection_reasons": reasons} for card, reasons in selected],
            "omitted_count": len(omitted),
            "omitted_reasons": omitted}


def _intersects(window, excluded):
    for start, end in excluded:
        if not (end < window[0] or start > window[1]):
            return True
    return False


def candidate_lineage(*, hypothesis_id, parent_ids, edit_spec, generation_action_id,
                      source_refs):
    """Build + validate a candidate lineage record (CandidateSpec provenance)."""
    record = {"schema_version": CANDIDATE_LINEAGE_SCHEMA,
              "hypothesis_id": hypothesis_id,
              "parent_ids": sorted(parent_ids),
              "edit_spec": edit_spec,
              "generation_action_id": generation_action_id,
              "source_refs": sorted(source_refs)}
    validate_candidate_lineage(record)
    return record


def validate_candidate_lineage(record):
    keys(record, ("schema_version", "hypothesis_id", "parent_ids", "edit_spec",
                  "generation_action_id", "source_refs"))
    require(record["schema_version"] == CANDIDATE_LINEAGE_SCHEMA, "unsupported lineage schema")
    for field in ("hypothesis_id", "generation_action_id"):
        require(isinstance(record[field], str) and record[field], field + " required")
    for field in ("parent_ids", "source_refs"):
        require(isinstance(record[field], list) and all(isinstance(s, str) for s in record[field]),
                field + " must be a list of strings")
    require(record["edit_spec"] is None or isinstance(record["edit_spec"], str),
            "edit_spec must be a string or null")
    digest(record)
    return record
