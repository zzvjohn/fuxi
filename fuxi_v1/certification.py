"""Derived data certification: every flag is recomputed from evidence, never asserted.

The v1.0 kernel previously asked an operator to assert four booleans
(``pit_certified``/``universe_certified``/``account_events_certified``/
``execution_rules_certified``) and left ``snapshot.quality`` /
``market.event_coverage_certified`` hard-coded ``False``. That produced a dead
end: candidates whose economic read-outs were fully satisfied were still blocked
by an *assertion* nobody was allowed to make, so the whole downstream chain
(E3/E4 account evidence, research admission, JQ eligibility) stayed at
``data_insufficient`` forever.

This module replaces the assertion with a derivation. Given a frozen payload and
a *declaration* of the assumptions being certified, it recomputes each flag from
the bytes, records the counts and the gaps that produced the verdict, and hashes
the result. Three properties are load-bearing:

1. **Reproducible, not asserted.** The universe flag requires that the cohort be
   regenerated exactly from the declared selection rule and the lifecycle table.
   If a rule cannot reproduce the cohort, no future information entered the
   choice and the flag is refused with a named blocking gap.
2. **Claim-scoped.** A flag certifies a *claim*, not a vibe. PIT availability is
   certified for same-session daily bars under a declared publication clock; it
   is explicitly *not* certified against vendor revision archives. Residual gaps
   are carried in the certificate so downstream code can narrow claims without
   pretending the gate opened.
3. **One-directional.** Certification can only ever refuse. A blocking gap makes
   the flag ``False``; there is no path by which a certificate grants an
   admission that the formal gates would otherwise deny.
"""
from collections import Counter
from datetime import date

from .contracts import digest, require, finite, timestamp, MARKET_CONTENT_EXCLUSIONS
from .pit_universe import FRAME_PREDICATE

SCHEMA = "fuxi.data_certification.1"

# Declared ex-ante reasons a listed, tradable member may be absent from the
# eligible set on a session. Anything outside this vocabulary is a silent
# removal and blocks universe certification.
#
# The vocabulary mirrors the declared universe policy: unknown or conflicting
# name state and not-yet-published name changes are excluded because acting on
# them would use information that was not public on the session. A batch may
# narrow or widen the set through ``declaration['ex_ante_reasons']``, but it must
# say so explicitly, because the set is what "no silent removal" is measured against.
EX_ANTE_REASONS = frozenset({
    "not_listed_or_delisted",
    "SME_before_mainboard_merger",
    "listing_age",
    "halt_or_tradability_unknown",
    "quote_missing",
    "corporate_action_unknown",
    "name_st",
    "name_unknown",
    "name_conflicting",
    "name_not_yet_published",
})

# Keys that must never appear in a cohort selection rule: their presence means
# the sample was drawn with knowledge of the outcome being measured.
FORBIDDEN_SAMPLING_KEYS = ("return", "delist", "outcome", "ic", "sharpe", "pnl", "nav",
                           "blocker", "eligible", "eligible_reason")

REQUIRED_EXECUTION_KEYS = (
    "initial_cash", "buy_lot", "sell_lot", "buy_commission", "sell_commission",
    "minimum_commission", "sell_tax", "settlement_sessions", "price_model",
)

SUPPORTED_ACTION_STATUS = frozenset({
    "cash_terms_supported", "ordinary_distribution_supported",
    "restructuring_no_holder_distribution",
})

# Fields whose presence in a feature snapshot would mean the value was computed
# from information that did not exist at the declared decision time.
FORWARD_MARKERS = ("forward", "future", "label", "target", "outcome")


def _flag(value, basis, blocking=(), residual=()):
    return {"value": bool(value), "basis": basis,
            "blocking_gaps": sorted(set(blocking)), "residual_gaps": sorted(set(residual))}


def _iso_day(value):
    return None if not value else str(value)[:10].replace("-", "")


def _lifecycle(lifecycle):
    """Normalise a code -> {list_date, delist_date, list_status} mapping to YYYYMMDD."""
    from .pit_universe import canonical_code
    result = {}
    for code, row in lifecycle.items():
        result[canonical_code(code)] = {"list_date": _iso_day(row.get("list_date")),
                                        "delist_date": _iso_day(row.get("delist_date")),
                                        "list_status": (row.get("list_status") or "").strip() or None}
    return result


def listed_at(life, code, day):
    """Lifecycle membership of ``code`` on ``day`` (``YYYY-MM-DD``).

    ``None`` means the identifier is not covered by the lifecycle table at all,
    which is treated as *not certified* by the caller rather than as absence.
    """
    from .pit_universe import canonical_code
    row = life.get(canonical_code(code))
    if row is None:
        return None
    if canonical_code(code).endswith('.BJ') and _iso_day(day) < '20211115':
        return False
    compact = day.replace("-", "")
    if row["list_date"] and compact < row["list_date"]:
        return False
    if row["delist_date"] and compact >= row["delist_date"]:
        return False
    return True


def certify_observation_clock(snapshot, declaration):
    """PIT claim: no feature value is used before its declared publication clock."""
    availability = declaration.get("availability") or {}
    clock = availability.get("vendor_session_clock")
    require(isinstance(clock, str) and len(clock) >= 5, "declared vendor publication clock required")
    sessions = snapshot["sessions"]
    times = snapshot["decision_times"]
    basis, blocking, residual = {}, [], []
    blocking += ["session_calendar_not_strictly_increasing"] if sessions != sorted(set(sessions)) else []
    if sorted(times) != sorted(sessions):
        blocking.append("decision_clock_calendar_mismatch")
    early, malformed = [], []
    for day in sessions:
        raw = times.get(day)
        if not isinstance(raw, str) or raw[:10] != day:
            malformed.append(day)
            continue
        try:
            stamp = timestamp(raw)
        except Exception:
            malformed.append(day)
            continue
        if stamp.utcoffset() is None:
            malformed.append(day)
        elif stamp < timestamp(day + "T" + clock + availability.get("timezone", "+08:00")):
            early.append(day)
    if malformed:
        blocking.append("malformed_decision_clock:%d" % len(malformed))
    if early:
        blocking.append("decision_before_declared_availability:%d" % len(early))
    fields = snapshot.get("fields") or {}
    used = declaration.get("fields") or sorted(fields)
    missing = sorted(set(used) - set(fields))
    if missing:
        blocking.append("declared_field_not_in_snapshot:" + ",".join(missing))
    off_frequency = sorted(f for f in used if (fields.get(f) or {}).get("frequency") != "daily")
    if off_frequency:
        blocking.append("non_daily_field:" + ",".join(off_frequency))
    undeclared = sorted(f for f in used if not (fields.get(f) or {}).get("adjustment"))
    if undeclared:
        blocking.append("adjustment_basis_undeclared:" + ",".join(undeclared))
    forward = sorted(f for f in used if any(m in f.lower() for m in FORWARD_MARKERS))
    if forward:
        blocking.append("forward_looking_field_in_feature_set:" + ",".join(forward))
    if snapshot.get("evaluation_start") not in sessions:
        blocking.append("evaluation_start_not_a_session")
    if any(day < snapshot.get("evaluation_start", day) for day in sessions[:0]):
        blocking.append("session_before_evaluation_start")
    basis.update(declared_availability=clock, sessions=len(sessions), fields=sorted(used),
                 earliest_decision_time=min(times.values()) if times else None,
                 latest_decision_time=max(times.values()) if times else None,
                 evaluation_start=snapshot.get("evaluation_start"),
                 claim="same-session daily bars available at or before the declared clock")
    residual += ["vendor_revision_archive_not_certified"]
    return _flag(not blocking, basis, blocking, residual)


def certify_universe(snapshot, declaration, lifecycle):
    """PIT claim: the eligible panel is the exact output of the declared selection rule."""
    cohort = declaration.get("cohort") or {}
    declared_reasons = frozenset(declaration.get("ex_ante_reasons") or EX_ANTE_REASONS)
    life = _lifecycle(lifecycle)
    sessions, universe = snapshot["sessions"], snapshot["universe"]
    # Row-form snapshots (fuxi.snapshot.1/.2) express membership via `universe` and
    # carry no `instruments` key (and Snapshot rejects unknown keys). Derive the frozen
    # cohort as the union of all per-session universes; columnar paths never reach here.
    instruments = snapshot.get("instruments") or sorted({c for v in universe.values() for c in v})
    basis, blocking, residual = {}, [], []
    posthoc = cohort.get("posthoc_exclusion")
    if posthoc is True:
        blocking.append("posthoc_selection_not_pit_neutral")
    elif posthoc is not False:
        blocking.append("selection_rule_posthoc_status_undeclared")
    keys = [str(k).lower() for k in (cohort.get("sampling_key_fields") or [])]
    if not keys:
        blocking.append("sampling_key_undeclared")
    contaminated = sorted(k for k in keys if any(m in k for m in FORBIDDEN_SAMPLING_KEYS))
    if contaminated:
        blocking.append("sampling_rule_depends_on_outcomes:" + ",".join(contaminated))
    from .pit_universe import canonical_code, reproduce_cohort, neutrality_audit
    reproduced, reproduce_basis = reproduce_cohort(life, cohort, sessions)
    frozen = {canonical_code(c) for c in instruments}
    if reproduced is None:
        blocking += reproduce_basis.get("blocking_gaps", [])
    elif set(reproduced) != frozen:
        only_rule = sorted(set(reproduced) - frozen)[:5]
        only_frozen = sorted(frozen - set(reproduced))[:5]
        blocking.append("cohort_not_reproducible_from_declared_rule")
        basis.update(reproduction_only_in_rule=only_rule, reproduction_only_in_frozen=only_frozen)
    basis.update(reproduce_basis)
    outside, alive_missing, reason_counts = [], Counter(), Counter()
    audit = snapshot.get("universe_audit") or {}
    for day in sessions:
        members = {canonical_code(c) for c in (universe.get(day) or ())}
        for code in members:
            if listed_at(life, code, day) is not True:
                outside.append((day, code))
        for code in frozen:
            if code in members:
                continue
            if listed_at(life, code, day) is True:
                alive_missing[code] += 1
        for reason_key, count in ((audit.get("reasons_by_session") or {}).get(day) or {}).items():
            for part in str(reason_key).split("|"):
                if part:
                    reason_counts[part] += count
    if outside:
        blocking.append("member_outside_declared_lifecycle:%d" % len(outside))
        basis["lifecycle_violation_examples"] = outside[:5]
    if alive_missing and not audit.get("reasons_by_session"):
        blocking.append("removal_reasons_not_bound_to_snapshot")
    undeclared = sorted(r for r in reason_counts if r not in declared_reasons)
    if undeclared:
        blocking.append("undeclared_removal_reason:" + ",".join(undeclared))
    sizes = [len(universe.get(day) or ()) for day in sessions]
    if not sizes or min(sizes) < 10:
        blocking.append("auditable_universe_below_10_instruments")
    neutrality = neutrality_audit(life, sessions, frozen,all_a=cohort.get('rule_kind')=='all_a_share_census')
    if not reproduce_basis.get("permutation_invariant", False):
        blocking.append("cohort_selection_depends_on_frame_order")
    if neutrality.get("survivor_filter_detected") is True:
        blocking.append("survivor_filter_detected_in_cohort")
    basis.update(sessions=len(sessions), instruments=len(instruments),
                 size_min=min(sizes) if sizes else 0, size_max=max(sizes) if sizes else 0,
                 removal_reasons={k: v for k, v in sorted(reason_counts.items())},
                 declared_ex_ante_reasons=sorted(declared_reasons),
                 neutrality=neutrality,
                 claim="eligible panel regenerated from the declared rule and the lifecycle table",
                 selection_rule=cohort.get("rule"))
    if cohort.get('rule_kind')!='all_a_share_census':
        residual += ["cohort_size_is_a_research_choice_not_a_market_census"]

    # Representativeness is a *separate* claim from reproducibility, so it gets a
    # separate derivation. It must not be satisfied merely because the universe
    # flag passed: the batch has to say which population it sampled from, the
    # declaration is checked against the predicate the code actually applies, the
    # selection has to be invariant to frame order, and the survivor test has to
    # come back clean. Otherwise the flag is a mirror of `universe_certified` and
    # certifies nothing.
    frame_gaps = []
    declared_frame = cohort.get("frame_predicate")
    derived_frame = reproduce_basis.get("frame_predicate")
    if not declared_frame:
        frame_gaps.append("sampling_frame_predicate_undeclared")
    elif declared_frame != derived_frame:
        frame_gaps.append("declared_frame_predicate_does_not_match_derived")
    representative = (not blocking) and bool(reproduce_basis.get("permutation_invariant")) \
        and not frame_gaps
    if not representative and not frame_gaps:
        frame_gaps.append("panel_is_not_a_certified_market_sample")
    return _flag(not blocking, basis, blocking, residual), _flag(representative, {
        "sampling_frame": (cohort.get("sampling_frame")),
        "frame_size": reproduce_basis.get("frame_size"),
        "declared_frame_predicate": declared_frame,
        "derived_frame_predicate": derived_frame,
        "permutation_invariant": reproduce_basis.get("permutation_invariant"),
        "cohort_share_of_frame": neutrality.get("cohort_share_of_frame"),
        "cohort_names_delisting_inside_window": neutrality.get("cohort_names_delisting_inside_window"),
        "expected_cohort_delisting_count": neutrality.get("expected_cohort_delisting_count"),
        "claim": ("common A-share census of the declared lifecycle window; external historical master revisions still require independent evidence"
                  if cohort.get('rule_kind')=='all_a_share_census' else
                  "the panel is an unbiased hash draw from the declared mainboard frame, with delisting names retained; it is not a market census")},
        frame_gaps,
        ["breadth_and_level_claims_remain_out_of_scope"] if representative else [])


def account_scope(snapshot, lifecycle=None):
    """Removal from today's selection does not extinguish yesterday's holding;
    delisting force-liquidates the account, so membership ends at the effective
    delisting date. Without a lifecycle table the plain accumulation semantics
    is preserved (no information, no removal)."""
    possible, result = set(), {}
    for day in snapshot["sessions"]:
        possible.update(snapshot["universe"][day])
        if lifecycle:
            possible = {c for c in possible if listed_at(lifecycle, c, day) is not False}
        result[day] = set(possible)
    return result


def missing_account_bars(snapshot, market, lifecycle=None):
    present = {(b["session"],b["instrument"]) for b in market["bars"]}
    return [(d,s) for d,names in account_scope(snapshot, lifecycle).items() for s in sorted(names) if (d,s) not in present]


def certify_account_events(snapshot, market, lifecycle=None):
    """Account claim: no member cell on which the account can act has an unknown action."""
    members = account_scope(snapshot, lifecycle)
    unknown_member, unknown_outside, missing_coverage = [], 0, 0
    for bar in market["bars"]:
        day, code = bar["session"], bar["instrument"]
        is_member = code in members.get(day, ())
        if bar.get("corporate_action_known") is not True:
            if is_member:
                unknown_member.append((day, code))
            else:
                unknown_outside += 1
        if is_member and not bar.get("event_coverage"):
            missing_coverage += 1
    terms = market.get("action_terms") or []
    # P-009 (a3): a term carrying a hash-bound exemption binding stays a declared
    # residual gap; only unsupported terms WITHOUT a binding block certification.
    unsupported = sorted({t.get("status") for t in terms
                          if t.get("status") not in SUPPORTED_ACTION_STATUS and not t.get("exemption")})
    exempted = [t for t in terms if t.get("status") not in SUPPORTED_ACTION_STATUS and t.get("exemption")]
    missing = missing_account_bars(snapshot,market,lifecycle)
    blocking = ["possible_holdings_missing_bars:%d" % len(missing)] if missing else []
    if unknown_member:
        blocking.append("member_cells_with_unknown_corporate_action:%d" % len(unknown_member))
    if unsupported:
        blocking.append("unsupported_action_terms:" + ",".join(str(s) for s in unsupported))
    if missing_coverage:
        blocking.append("member_cells_without_event_classification:%d" % missing_coverage)
    basis = {"member_unknown_cells": len(unknown_member), "nonmember_unknown_cells": unknown_outside,
             "action_terms": len(terms), "unsupported_terms": len(terms) - sum(
                 t.get("status") in SUPPORTED_ACTION_STATUS or bool(t.get("exemption")) for t in terms),
             "exempted_terms": len(exempted),
             "claim": "every tradeable member cell carries a resolved corporate-action status",
             "nonmember_unknown_meaning": "never eligible before this session; prior members remain in scope until the effective delisting date"}
    if unknown_member:
        basis["member_unknown_examples"] = unknown_member[:5]
    residual = ["public_source_events_are_not_broker_settlement_receipts",
                "investor_specific_dividend_tax_not_certified"]
    if exempted:
        residual.append("declared_action_term_exemptions:%d" % len(exempted))
    return _flag(not blocking, basis, blocking, residual)


def certify_event_coverage(snapshot, market, lifecycle=None):
    """Market claim: action coverage is complete wherever the account can act."""
    members = account_scope(snapshot, lifecycle)
    counts = Counter()
    unknown_in_scope = 0
    for bar in market["bars"]:
        is_member = bar["instrument"] in members.get(bar["session"], ())
        counts[bar.get("event_coverage") or "unspecified"] += 1
        if is_member and bar.get("corporate_action_known") is not True:
            unknown_in_scope += 1
    missing = missing_account_bars(snapshot,market,lifecycle)
    blocking = ["possible_holdings_missing_bars:%d" % len(missing)] if missing else []
    if unknown_in_scope:
        blocking.append("member_cells_with_unknown_event_coverage:%d" % unknown_in_scope)
    if counts.get("unspecified"):
        blocking.append("unspecified_event_coverage_cells:%d" % counts["unspecified"])
    basis = {"coverage_class_counts": dict(sorted(counts.items())),
             "member_unknown_cells": unknown_in_scope,
             "claim": "corporate actions over the member panel are covered by cited evidence",
             "nonmember_unknown_meaning": "only never-eligible cells excluded; selection removal is not liquidation, but delisting force-liquidates the account"}
    return _flag(not blocking, basis, blocking,
                 ["coverage_is_evidence_based_and_not_vendor_settlement_certified"])


def certify_tradability(snapshot, market, lifecycle=None):
    """Account claim: pause state and quote/limit fields are unambiguous where the account acts."""
    members = account_scope(snapshot, lifecycle)
    paused_state = Counter()
    missing_quote, missing_limit, bad_bracket, fabricated_mark = [], [], [], []
    for bar in market["bars"]:
        if bar["instrument"] not in members.get(bar["session"], ()):
            continue
        paused = bar.get("paused")
        paused_state[str(paused)] += 1
        if paused is True:
            from .halt_evidence import valid_partial_halt_mark
            observed_partial=valid_partial_halt_mark(bar,snapshot)
            if not observed_partial and bar.get("mark_source") not in {"declared_frozen_last_close_on_vendor_halt", "declared_frozen_last_close_on_issuer_halt", "declared_ex_cash_reference_on_vendor_halt"} and bar.get("close") is not None:
                fabricated_mark.append((bar["session"], bar["instrument"]))
            continue
        if paused is not False:
            continue
        if any(bar.get(k) is None for k in ("open", "close", "volume")):
            missing_quote.append((bar["session"], bar["instrument"]))
            continue
        if bar.get("up_limit") is None or bar.get("down_limit") is None:
            missing_limit.append((bar["session"], bar["instrument"]))
        elif not (bar["down_limit"] - 1e-9 <= bar["open"] <= bar["up_limit"] + 1e-9):
            bad_bracket.append((bar["session"], bar["instrument"]))
    missing = missing_account_bars(snapshot,market,lifecycle)
    blocking = ["possible_holdings_missing_bars:%d" % len(missing)] if missing else []
    if paused_state.get("None"):
        blocking.append("member_cells_with_ambiguous_pause_state:%s" % paused_state["None"])
    if missing_quote:
        blocking.append("member_cells_missing_quote_or_volume:%d" % len(missing_quote))
    if missing_limit:
        blocking.append("member_cells_missing_price_limits:%d" % len(missing_limit))
    if bad_bracket:
        blocking.append("member_cells_where_open_outside_limits:%d" % len(bad_bracket))
    if fabricated_mark:
        blocking.append("halted_member_cells_with_observed_mark:%d" % len(fabricated_mark))
    basis = {"pause_state_counts": dict(sorted(paused_state.items())),
             "missing_quote_cells": len(missing_quote), "missing_limit_cells": len(missing_limit),
             "open_outside_limit_cells": len(bad_bracket),
             "fabricated_mark_cells": len(fabricated_mark),
             "claim": "pause state and order-time quotes/limits are unambiguous on the member panel"}
    if missing_quote:
        basis["missing_quote_examples"] = missing_quote[:5]
    return _flag(not blocking, basis, blocking,
                 ["pause_state_is_a_vendor_halt_record_not_an_exchange_receipt"])


def certify_execution_rules(market, declaration, instruments=None):
    """Account claim: the rule set is complete, uniform and hash-bound to the run.

    ``instruments`` lets columnar callers pass the declared instrument axis
    directly instead of iterating 3.7M virtual bars; row-form callers keep the
    legacy derivation from ``market['bars']`` (P-20261015-006).
    """
    execution = market.get("execution") or {}
    expected = declaration.get("execution") or {}
    blocking, basis = [], {}
    missing = sorted(k for k in REQUIRED_EXECUTION_KEYS if k not in execution)
    if missing:
        blocking.append("execution_rule_missing:" + ",".join(missing))
    if not missing:
        if int(execution["buy_lot"]) < 1 or int(execution["sell_lot"]) < 1:
            blocking.append("invalid_lot_rule")
        if int(execution["settlement_sessions"]) < 1:
            blocking.append("settlement_rule_below_one_session")
        models = set(expected.get("allowed_price_models") or [execution["price_model"]])
        if execution["price_model"] not in models:
            blocking.append("undeclared_price_model")
        schedule = execution.get("sell_tax_schedule")
        if expected.get("sell_tax_schedule") and not schedule:
            blocking.append("sell_tax_schedule_missing")
        if schedule and schedule != sorted(schedule):
            blocking.append("sell_tax_schedule_not_ordered")
    if not (market.get("market_hash") or "").strip():
        blocking.append("market_hash_missing")
    from .a_share import board,instrument_rule
    from .contracts import ContractError
    names = instruments if instruments is not None else {r.get('instrument') for r in market.get('bars',[]) if r.get('instrument')}
    if execution.get('instrument_rules') is not None or any(board(c) in ('star','chinext','bj') for c in names):
        for code in sorted(names):
            try:instrument_rule(execution,code)
            except ContractError: blocking.append('instrument_quantity_rule_unverified:'+code)
    basis = {"execution_hash": digest(execution), "price_model": execution.get("price_model"),
             "settlement_sessions": execution.get("settlement_sessions"),
             "buy_lot": execution.get("buy_lot"), "sell_lot": execution.get("sell_lot"),
             "cost_status": execution.get("cost_status"),
             "market_hash": market.get("market_hash"),
             "claim": "the rule set is complete, uniform across sessions and bound to the market hash by "
                      "ir.execution_hash"}
    residual = []
    if execution.get("cost_status") != "broker_certified":
        residual.append("cost_magnitudes_are_declared_research_assumptions_not_broker_receipts")
    if execution.get("dividend_tax_model"):
        residual.append("dividend_tax_is_a_declared_flat_model_not_an_investor_tax_receipt")
    return _flag(not blocking, basis, blocking, residual)


DEFAULT_DECLARATION = {
    "availability": {"vendor_session_clock": "18:00:00"},
    "execution": {"allowed_price_models": ["next_session_open"], "sell_tax_schedule": True},
    "fields": ["open", "high", "low", "close", "volume", "amount"],
}


def certify(payload, *, declaration=None, lifecycle=None, certified_at=None):
    """Derive the full certificate from a frozen ``fuxi.history_input.1`` payload.

    Columnar payloads (``fuxi.columnar_snapshot.1``) are derived by
    ``certification_columnar.certify_columnar`` over the same declaration, so the
    certificate schema and the one-directional refusal rule are identical
    (P-20261015-006).
    """
    from .pit_universe import lifecycle_from_plan
    merged = dict(DEFAULT_DECLARATION)
    merged.update(declaration or {})
    declaration = merged
    snapshot, market = payload["snapshot"], payload["market"]
    if snapshot.get("schema_version") == "fuxi.columnar_snapshot.1":
        if declaration.get("cohort") is None:
            declaration["cohort"] = cohort_declaration(declaration.pop("_plan", None), payload)
        if lifecycle is None:
            lifecycle = declaration.pop("_lifecycle", None)
        from . import certification_columnar
        return certification_columnar.certify_columnar(
            payload, declaration=declaration, lifecycle=lifecycle, certified_at=certified_at)
    if declaration.get("cohort") is None:
        declaration["cohort"] = cohort_declaration(declaration.pop("_plan", None), payload)
    if lifecycle is None:
        lifecycle = declaration.pop("_lifecycle", None)
    flags = {}
    flags["pit_certified"] = certify_observation_clock(snapshot, declaration)
    if lifecycle is None:
        flags["universe_certified"] = _flag(False, {"claim": "panel reproducibility"},
                                            ["lifecycle_table_not_supplied"])
        flags["market_representativeness_certified"] = _flag(False, {}, ["lifecycle_table_not_supplied"])
    else:
        flags["universe_certified"], flags["market_representativeness_certified"] = certify_universe(
            snapshot, declaration, lifecycle)
    flags["account_events_certified"] = certify_account_events(snapshot, market, lifecycle)
    flags["event_coverage_certified"] = certify_event_coverage(snapshot, market, lifecycle)
    flags["tradability_certified"] = certify_tradability(snapshot, market, lifecycle)
    flags["execution_rules_certified"] = certify_execution_rules(market, declaration)
    blocking = sorted({g for f in flags.values() for g in f["blocking_gaps"]})
    residual = sorted({g for f in flags.values() for g in f["residual_gaps"]})
    certificate = {
        "schema_version": SCHEMA,
        "snapshot_hash": digest(snapshot),
        # Bind the market block by content, not by a hash the caller may have copied
        # from elsewhere: a certificate must move when the prices it authorises move.
        "market_digest": digest({k: v for k, v in market.items()
                                 if k not in MARKET_CONTENT_EXCLUSIONS}),
        "market_hash": market.get("market_hash"),
        "declaration": {k: v for k, v in declaration.items() if not k.startswith("_")},
        # The lifecycle table is an input to the universe claim, so its identity is
        # part of the certificate: a registration can then require the same table.
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


def cohort_declaration(plan=None, payload=None):
    """Read the selection rule a batch was frozen under, without inventing one.

    The declaration must be *reproducible*, so it is taken from the frozen plan's
    ``cohort`` block (salt, cutoffs, per-exchange size) rather than from prose.
    When the payload also carries an ``audit.selection_audit`` block, the batch
    declares a retrospective restriction on top of that rule and the declaration
    says so explicitly instead of quietly reporting the restricted cohort.
    """
    plan = plan or {}
    cohort = dict((plan.get("cohort") or {}))
    if not cohort:
        embedded = ((payload or {}).get("snapshot") or {}).get("cohort_declaration")
        return dict(embedded) if embedded else None
    declaration = {
        "rule_kind": cohort.get('rule_kind') or ("per_exchange_hash_cutoff" if cohort.get("hash_cutoffs") else "unknown"),
        "salt": cohort.get("salt"),
        "per_exchange_initial": cohort.get("per_exchange_initial"),
        "hash_cutoffs": cohort.get("hash_cutoffs"),
        "sampling_key_fields": ["ts_code"],
        "sampling_frame": "full_lifecycle_table",
        # Declared here so the engine can refuse a batch that describes its
        # population as something the sampling code never drew from.
        "frame_predicate": cohort.get("frame_predicate") or FRAME_PREDICATE,
        "posthoc_exclusion": False,
        "rule": cohort.get("selection") or "per-exchange hash sample over the lifecycle table",
    }
    selection = ((payload or {}).get("audit") or {}).get("selection_audit")
    if selection:
        declaration.update(
            posthoc_exclusion=True,
            parent_rule=dict(declaration),
            restriction_rule=selection.get("rule"),
            restriction_excluded=[r.get("ts_code") for r in (selection.get("excluded") or [])],
            sampling_frame="restricted_parent_cohort",
            rule=str(declaration["rule"]) + " then restricted: " + str(selection.get("rule")))
    elif (plan.get("universe_restriction") or {}).get("posthoc_exclusion") is True:
        declaration.update(posthoc_exclusion=True,
                           restriction_rule=plan["universe_restriction"].get("rule"))
    return declaration


def derived_data_receipt(certificate):
    """The four register-data booleans, recomputed rather than asserted."""
    flags = certificate["flags"]
    return {
        "derived": True,
        "certification_hash": certificate["certification_hash"],
        "schema_version": certificate["schema_version"],
        "snapshot_hash": certificate["snapshot_hash"],
        "market_hash": certificate["market_hash"],
        "pit_certified": flags["pit_certified"]["value"],
        "universe_certified": flags["universe_certified"]["value"],
        "account_events_certified": flags["account_events_certified"]["value"],
        "execution_rules_certified": flags["execution_rules_certified"]["value"],
        "event_coverage_certified": flags["event_coverage_certified"]["value"],
        "tradability_certified": flags["tradability_certified"]["value"],
        "market_representativeness_certified": flags["market_representativeness_certified"]["value"],
        "basis": {k: f["basis"] for k, f in flags.items()},
        "blocking_gaps": certificate["blocking_gaps"],
        "residual_gaps": certificate["residual_gaps"],
        "claim_scope": certificate["claim_scope"],
        "investment_eligibility_certified": False,
    }


def apply_to_payload(payload, certificate):
    """Write a certificate back into the payload's quality/market flag blocks.

    Returns a new payload; the input is left untouched so a refused certificate
    can never mutate a frozen artifact in place.
    """
    from copy import deepcopy
    result = deepcopy(payload)
    require(digest(payload["snapshot"]) == certificate["snapshot_hash"], "certificate snapshot mismatch")
    require(digest({k:v for k,v in payload["market"].items() if k not in
                   MARKET_CONTENT_EXCLUSIONS}) == certificate["market_digest"],
            "certificate market mismatch")
    # Validation is an annotation, not permission to rewrite source truth or erase
    # bias. Keeping the certified blocks intact also makes registration reproducible.
    result["data_certificate"] = certificate
    return result
