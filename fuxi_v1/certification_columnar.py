"""Columnar-schema certification: the same derivation, vectorised over full cross sections.

The row-form certifiers in ``certification.py`` consume ``snapshot['universe']``
(day -> member list), ``market['bars']`` (row dicts) and ``snapshot['columns']``
(row matrices).  Columnar payloads keep the same facts as hash-bound memory-mapped
arrays; this module derives the identical flag vocabulary without materialising
rows.  The refusal rule is unchanged: a blocking gap makes the flag ``False`` and
no columnar path grants an admission the formal gates would otherwise deny.

Columnar-specific claim narrowing (all recorded in the certificate basis):

* The per-day membership matrix is a *frozen build fact*; the snapshot carries no
  ``universe_audit.reasons_by_session``, so ``alive_missing`` (a listed name absent
  from a day's panel) is not derivable and is recorded as a basis note instead of
  a refusal.  Cohort reproduction from the declared rule remains mandatory.
* The frozen snapshot embeds the built cohort in ``cohort_declaration.members``;
  when present it must equal the frozen instrument axis exactly, and the vendor
  security master is re-derived as a second witness whose coverage gaps are
  residual gaps, not refusals (the master's own completeness is outside the
  claim).
* ``event_coverage`` is a categorical array; ``missing_evidence`` cells on the
  member panel are the columnar analogue of the row-form ``unspecified`` refusal.
"""
from bisect import bisect_left
from collections import Counter

import numpy as np

from .certification import (SCHEMA, SUPPORTED_ACTION_STATUS, _flag,
                            certify_observation_clock, certify_execution_rules)
from .columnar import ColumnarSnapshot, ColumnarMarket
from .contracts import digest, MARKET_CONTENT_EXCLUSIONS
from .pit_universe import (canonical_code, reproduce_cohort, neutrality_audit,
                           ALL_A_FRAME_PREDICATE, FRAME_PREDICATE)


def _int_days(sessions):
    return np.array([int(d.replace("-", "")) for d in sessions], dtype=np.int64)


def _lifecycle_edges(life, sessions, instruments):
    """Per-instrument listed/delisted session edges plus master coverage flags.

    Mirrors ``certification.listed_at`` semantics: a name is listed from its
    list date (inclusive) until the day before its delist date; BJ names are
    additionally not listed before 2021-11-15.
    """
    days = _int_days(sessions)
    n = len(instruments)
    list_edge = np.zeros(n, dtype=np.int64)
    delist_edge = np.full(n, len(sessions), dtype=np.int64)
    known = np.zeros(n, dtype=bool)
    for j, code in enumerate(instruments):
        row = life.get(canonical_code(code))
        if row is None:
            continue
        known[j] = True
        listed = (row.get("list_date") or "").strip().replace("-", "")
        delisted = (row.get("delist_date") or "").strip().replace("-", "")
        if listed and listed.isdigit():
            list_edge[j] = bisect_left(days, int(listed))
        if delisted and delisted.isdigit():
            delist_edge[j] = bisect_left(days, int(delisted))
        if canonical_code(code).endswith(".BJ"):
            list_edge[j] = max(list_edge[j], bisect_left(days, 20211115))
    return days, list_edge, delist_edge, known


def certify_universe_columnar(snap, declaration, lifecycle):
    """PIT claim over a columnar snapshot: the frozen cohort equals the declared rule's output.

    Returns ``(universe_flag, representativeness_flag)`` exactly like
    ``certification.certify_universe``.
    """
    cohort = declaration.get("cohort") or {}
    rule_kind = cohort.get("rule_kind")
    frozen = {canonical_code(c) for c in snap.instruments}
    declared_members = cohort.get("members")
    basis, blocking, residual = {}, [], []
    reproduction = None
    if declared_members:
        declared = {canonical_code(str(m.get("ts_code"))) for m in declared_members}
        if declared != frozen:
            blocking.append("cohort_not_reproducible_from_declared_rule")
            basis.update(declared_only=sorted(declared - frozen)[:5],
                         frozen_only=sorted(frozen - declared)[:5])
        reproduced, reproduction = reproduce_cohort(lifecycle, cohort, snap.sessions)
        if reproduced is not None:
            missing = sorted(declared - set(reproduced))
            extra = sorted(set(reproduced) - declared)
            if missing or extra:
                residual.append("vendor_master_coverage_gap:%d_missing,%d_extra" % (len(missing), len(extra)))
                basis["vendor_master_gap_examples"] = (missing + extra)[:5]
            basis["reproduction_frame_size"] = reproduction.get("frame_size")
            basis["reproduction_permutation_invariant"] = reproduction.get("permutation_invariant")
        else:
            residual.append("cohort_rule_not_reproducible_from_current_vendor_master:"
                            + ",".join(reproduction.get("blocking_gaps", [])))
            basis["reproduction_blocking_gaps"] = reproduction.get("blocking_gaps", [])
    else:
        reproduced, reproduction = reproduce_cohort(lifecycle, cohort, snap.sessions)
        if reproduced is None:
            blocking += reproduction.get("blocking_gaps", [])
        elif {canonical_code(c) for c in reproduced} != frozen:
            blocking.append("cohort_not_reproducible_from_declared_rule")
            basis.update(reproduction_only_in_rule=sorted(set(reproduced) - frozen)[:5],
                         reproduction_only_in_frozen=sorted(frozen - set(reproduced))[:5])
        basis.update(reproduction)
    # Per-day membership against the master: only derivable where the master
    # covers the name; coverage gaps are counted, never treated as refusals.
    _, list_edge, delist_edge, known = _lifecycle_edges(lifecycle, snap.sessions, snap.instruments)
    day_idx = np.arange(len(snap.sessions))[:, None]
    listed = known[None, :] & (day_idx >= list_edge[None, :]) & (day_idx < delist_edge[None, :])
    member = snap.membership
    outside = int((member & known[None, :] & ~listed).sum())
    if outside:
        blocking.append("member_outside_declared_lifecycle:%d" % outside)
    sizes = [int(v) for v in member.sum(axis=1)]
    if not sizes or min(sizes) < 10:
        blocking.append("auditable_universe_below_10_instruments")
    unknown_master_cells = int((member & ~known[None, :]).sum())
    posthoc = cohort.get("posthoc_exclusion")
    if posthoc is True:
        blocking.append("posthoc_selection_not_pit_neutral")
    elif posthoc is not False:
        blocking.append("selection_rule_posthoc_status_undeclared")
    keys = [str(k).lower() for k in (cohort.get("sampling_key_fields") or [])]
    if not keys:
        blocking.append("sampling_key_undeclared")
    all_a = rule_kind == "all_a_share_census"
    neutrality = neutrality_audit(lifecycle, snap.sessions, sorted(frozen), all_a=all_a)
    if neutrality.get("survivor_filter_detected") is True:
        blocking.append("survivor_filter_detected_in_cohort")
    basis.update(sessions=len(snap.sessions), instruments=len(snap.instruments),
                 size_min=min(sizes) if sizes else 0, size_max=max(sizes) if sizes else 0,
                 member_cells_outside_lifecycle=outside,
                 member_cells_not_in_vendor_master=unknown_master_cells,
                 removal_reason_audit="not_derivable_from_columnar_snapshot_membership_is_frozen_fact",
                 neutrality=neutrality,
                 claim="frozen columnar cohort equals the declared rule's members and daily membership "
                       "stays inside the lifecycle table wherever the vendor master covers the name",
                 selection_rule=cohort.get("rule"))
    if rule_kind != "all_a_share_census":
        residual.append("cohort_size_is_a_research_choice_not_a_market_census")
    declared_frame = cohort.get("frame_predicate")
    derived_frame = ALL_A_FRAME_PREDICATE if all_a else FRAME_PREDICATE
    frame_gaps = []
    if not declared_frame:
        frame_gaps.append("sampling_frame_predicate_undeclared")
    elif declared_frame != derived_frame:
        frame_gaps.append("declared_frame_predicate_does_not_match_derived")
    representative = (not blocking) and bool(reproduction is not None and reproduction.get("permutation_invariant")) \
        and not frame_gaps
    if not representative and not frame_gaps and not blocking:
        frame_gaps.append("panel_is_not_a_certified_market_sample")
    representativeness = _flag(representative, {
        "sampling_frame": cohort.get("sampling_frame"),
        "frame_size": reproduction.get("frame_size") if reproduction else None,
        "declared_frame_predicate": declared_frame,
        "derived_frame_predicate": derived_frame,
        "permutation_invariant": reproduction.get("permutation_invariant") if reproduction else None,
        "cohort_share_of_frame": neutrality.get("cohort_share_of_frame"),
        "cohort_names_delisting_inside_window": neutrality.get("cohort_names_delisting_inside_window"),
        "expected_cohort_delisting_count": neutrality.get("expected_cohort_delisting_count"),
        "claim": ("common A-share census of the declared lifecycle window; external historical master revisions still require independent evidence"
                  if all_a else
                  "the panel is a declared columnar cohort; delisting names retained; it is not a market census")},
        frame_gaps,
        ["breadth_and_level_claims_remain_out_of_scope"] if representative else [])
    return _flag(not blocking, basis, blocking, residual), representativeness


def _missing_category_codes(category_values, names):
    return [int(k) for k, v in category_values.items() if v in names]


def certify_account_events_columnar(snap, market, lifecycle=None):
    """Account claim (columnar): no member cell the account can act on has an unknown action."""
    member = snap.membership
    ca = market.status["corporate_action_known"].astype(np.int8)
    unknown_member = int((member & (ca != 1)).sum())
    unknown_outside = int((~member & (ca != 1)).sum())
    missing = _missing_category_codes(market.category_values["event_coverage"], {"missing_evidence"})
    ec = market.categorical["event_coverage"]
    missing_coverage = int((member & np.isin(ec, missing)).sum()) if missing else 0
    terms = market["action_terms"]
    # P-009 (a3): a term carrying a hash-bound exemption binding stays a declared
    # residual gap; only unsupported terms WITHOUT a binding block certification.
    # Mirrors certification.certify_account_events so the columnar and row-form
    # refusals stay identical.
    unsupported = sorted({str(t.get("status")) for t in terms
                          if str(t.get("status")) not in SUPPORTED_ACTION_STATUS and not t.get("exemption")})
    exempted = [t for t in terms if t.get("status") not in SUPPORTED_ACTION_STATUS and t.get("exemption")]
    close = market.numeric["close"]
    missing_bars = int((member & ~np.isfinite(close)).sum())
    blocking = []
    if missing_bars:
        blocking.append("possible_holdings_missing_bars:%d" % missing_bars)
    if unknown_member:
        blocking.append("member_cells_with_unknown_corporate_action:%d" % unknown_member)
    if unsupported:
        blocking.append("unsupported_action_terms:" + ",".join(unsupported))
    if missing_coverage:
        blocking.append("member_cells_without_event_classification:%d" % missing_coverage)
    basis = {"member_unknown_cells": unknown_member, "nonmember_unknown_cells": unknown_outside,
             "action_terms": len(terms), "unsupported_terms": len(terms) - sum(
                 t.get("status") in SUPPORTED_ACTION_STATUS or bool(t.get("exemption")) for t in terms),
             "exempted_terms": len(exempted),
             "claim": "every tradeable member cell carries a resolved corporate-action status",
             "nonmember_unknown_meaning": "never eligible before this session; prior members remain in scope until the effective delisting date"}
    residual = ["public_source_events_are_not_broker_settlement_receipts",
                "investor_specific_dividend_tax_not_certified"]
    if exempted:
        residual.append("declared_action_term_exemptions:%d" % len(exempted))
    return _flag(not blocking, basis, blocking, residual)


def certify_event_coverage_columnar(snap, market, lifecycle=None):
    """Market claim (columnar): action coverage is complete wherever the account can act."""
    member = snap.membership
    ca = market.status["corporate_action_known"].astype(np.int8)
    unknown_in_scope = int((member & (ca != 1)).sum())
    categories = market.category_values["event_coverage"]
    ec = market.categorical["event_coverage"]
    counts = Counter()
    for value, label in categories.items():
        counts[label] += int((member & (ec == int(value))).sum())
    missing = counts.get("missing_evidence", 0)
    close = market.numeric["close"]
    missing_bars = int((member & ~np.isfinite(close)).sum())
    blocking = []
    if missing_bars:
        blocking.append("possible_holdings_missing_bars:%d" % missing_bars)
    if unknown_in_scope:
        blocking.append("member_cells_with_unknown_event_coverage:%d" % unknown_in_scope)
    if missing:
        blocking.append("unspecified_event_coverage_cells:%d" % missing)
    basis = {"coverage_class_counts": dict(sorted(counts.items())),
             "member_unknown_cells": unknown_in_scope,
             "claim": "corporate actions over the member panel are covered by cited evidence",
             "nonmember_unknown_meaning": "only never-eligible cells excluded; selection removal is not liquidation, but delisting force-liquidates the account"}
    return _flag(not blocking, basis, blocking,
                 ["coverage_is_evidence_based_and_not_vendor_settlement_certified"])


def certify_tradability_columnar(snap, market, lifecycle=None):
    """Account claim (columnar): pause state and quote/limit fields are unambiguous on the member panel."""
    member = snap.membership
    paused = market.status["paused"].astype(np.int8)
    values = market.category_values["mark_source"]
    frozen_marks = _missing_category_codes(values, {"declared_frozen_last_close_on_vendor_halt",
                                                    "declared_ex_cash_reference_on_vendor_halt"})
    partial_marks = _missing_category_codes(values, {"observed_close_on_vendor_partial_halt"})
    observed_marks = _missing_category_codes(values, {"observed_close", "halt_action_reference_not_identifiable"})
    mark = market.categorical["mark_source"]
    close, open_, volume = market.numeric["close"], market.numeric["open"], market.numeric["volume"]
    up, down = market.numeric["up_limit"], market.numeric["down_limit"]
    ambiguous = int((member & (paused == -1)).sum())
    halted = member & (paused == 1)
    fabricated = int((halted & np.isfinite(close) & np.isin(mark, observed_marks)).sum())
    partial_halted = int((halted & np.isin(mark, partial_marks)).sum())
    frozen_halted = int((halted & np.isin(mark, frozen_marks)).sum())
    active = member & (paused == 0)
    missing_quote = int((active & (~np.isfinite(open_) | ~np.isfinite(close) | ~np.isfinite(volume))).sum())
    missing_limit = int((active & (~np.isfinite(up) | ~np.isfinite(down))).sum())
    bracket = (open_ < down - 1e-9) | (open_ > up + 1e-9)
    bad_bracket = int((active & bracket).sum())
    missing_bars = int((member & ~np.isfinite(close)).sum())
    blocking = []
    if missing_bars:
        blocking.append("possible_holdings_missing_bars:%d" % missing_bars)
    if ambiguous:
        blocking.append("member_cells_with_ambiguous_pause_state:%d" % ambiguous)
    if missing_quote:
        blocking.append("member_cells_missing_quote_or_volume:%d" % missing_quote)
    if missing_limit:
        blocking.append("member_cells_missing_price_limits:%d" % missing_limit)
    if bad_bracket:
        blocking.append("member_cells_where_open_outside_limits:%d" % bad_bracket)
    if fabricated:
        blocking.append("halted_member_cells_with_observed_mark:%d" % fabricated)
    basis = {"pause_state_counts": {"True": int((member & (paused == 1)).sum()),
                                    "False": int((member & (paused == 0)).sum()),
                                    "None": ambiguous},
             "halted_cells_with_frozen_mark": frozen_halted,
             "halted_cells_with_partial_halt_mark": partial_halted,
             "missing_quote_cells": missing_quote, "missing_limit_cells": missing_limit,
             "open_outside_limit_cells": bad_bracket, "fabricated_mark_cells": fabricated,
             "claim": "pause state and order-time quotes/limits are unambiguous on the member panel"}
    return _flag(not blocking, basis, blocking,
                 ["pause_state_is_a_vendor_halt_record_not_an_exchange_receipt"])


def certify_columnar(payload, *, declaration=None, lifecycle=None, certified_at=None):
    """Derive the full certificate from a frozen columnar payload.

    The certificate structure, schema version, claim scope and the
    one-directional refusal rule are identical to ``certification.certify``.
    The embedded ``cohort_declaration`` fills the declaration when the caller
    did not supply one, exactly like the row-form entry point.
    """
    from .certification import DEFAULT_DECLARATION, cohort_declaration
    merged = dict(DEFAULT_DECLARATION)
    merged.update(declaration or {})
    declaration = merged
    if declaration.get("cohort") is None:
        declaration["cohort"] = cohort_declaration(declaration.pop("_plan", None), payload)
    if lifecycle is None:
        lifecycle = declaration.pop("_lifecycle", None)
    snap = ColumnarSnapshot(payload["snapshot"])
    market = ColumnarMarket(payload["market"])
    try:
        flags = {}
        flags["pit_certified"] = certify_observation_clock(payload["snapshot"], declaration)
        if lifecycle is None:
            flags["universe_certified"] = _flag(False, {"claim": "panel reproducibility"},
                                                ["lifecycle_table_not_supplied"])
            flags["market_representativeness_certified"] = _flag(False, {}, ["lifecycle_table_not_supplied"])
        else:
            flags["universe_certified"], flags["market_representativeness_certified"] = \
                certify_universe_columnar(snap, declaration, lifecycle)
        flags["account_events_certified"] = certify_account_events_columnar(snap, market, lifecycle)
        flags["event_coverage_certified"] = certify_event_coverage_columnar(snap, market, lifecycle)
        flags["tradability_certified"] = certify_tradability_columnar(snap, market, lifecycle)
        flags["execution_rules_certified"] = certify_execution_rules(
            market, declaration, instruments=market["instruments"])
        blocking = sorted({g for f in flags.values() for g in f["blocking_gaps"]})
        residual = sorted({g for f in flags.values() for g in f["residual_gaps"]})
        certificate = {
            "schema_version": SCHEMA,
            "snapshot_hash": digest(payload["snapshot"]),
            "market_digest": digest({k: v for k, v in payload["market"].items()
                                     if k not in MARKET_CONTENT_EXCLUSIONS}),
            "market_hash": payload["market"].get("market_hash"),
            "declaration": {k: v for k, v in declaration.items() if not k.startswith("_")},
            "lifecycle_digest": digest({k: v for k, v in sorted((lifecycle or {}).items())}),
            "flags": flags,
            "blocking_gaps": blocking,
            "residual_gaps": residual,
            "claim_scope": {
                "kind": "derived_research_contract_validation",
                "investment_eligibility_certified": False,
                "certified": sorted(k for k, f in flags.items() if f["value"]),
                "refused": sorted(k for k, f in flags.items() if not f["value"]),
                "not_covered": ["absolute_return_level", "market_breadth",
                                "vendor_revision_history", "broker_settlement_receipts"],
            },
            "direction": "certification can only refuse; no certificate grants an admission",
        }
        authority = dict(certificate)
        certificate = dict(authority, certified_at=certified_at)
        certificate["certification_hash"] = digest(authority)
        return certificate
    finally:
        snap.close()
        market.close()
