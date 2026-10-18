"""Certification must derive flags from bytes and refuse anything it cannot regenerate."""
from copy import deepcopy
import unittest

from fuxi_v1.certification import (SCHEMA, apply_to_payload, certify, cohort_declaration,
                                   derived_data_receipt, listed_at)
from fuxi_v1.contracts import ContractError, digest, require
from fuxi_v1.pit_universe import (FRAME_PREDICATE, build_pit_cohort, canonical_code,
                                  full_population_hash_sample, frame_rows, jq_code,
                                  lifecycle_from_rows, neutrality_audit, reproduce_cohort)

SESSIONS = ["2021-01-04", "2021-01-05", "2021-01-06"]
CODES = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ",
         "000007.SZ", "000008.SZ", "600001.SH", "600002.SH", "600003.SH", "600004.SH"]
LIVE = CODES[:10]
SALT = "fixture-salt"


def life_table(extra=None):
    rows = [{"ts_code": c, "list_date": "19990104", "delist_date": "", "list_status": "L"}
            for c in CODES]
    if extra:
        rows.extend(extra)
    return lifecycle_from_rows(rows)


def fixture_payload(*, universe_overrides=None, bars_overrides=None, quality=None,
                    audit=None, fields=None):
    instruments = [jq_code(c) for c in CODES]
    members = {day: [jq_code(c) for c in LIVE] for day in SESSIONS}
    for day, codes in (universe_overrides or {}).items():
        members[day] = [jq_code(c) for c in codes]
    bars = []
    for index, day in enumerate(SESSIONS):
        for code in instruments:
            bar = {"session": day, "instrument": code, "open": 10.0, "high": 10.5, "low": 9.5,
                   "close": 10.0, "volume": 100000.0, "up_limit": 11.0, "down_limit": 9.0,
                   "paused": False, "mark_source": "observed_close",
                   "corporate_action_known": True,
                   "event_coverage": "unchanged_reference_and_factor",
                   "evidence_id": "ev_%d_%s" % (index, code)}
            bars.append(bar)
    for override in (bars_overrides or []):
        for bar in bars:
            if bar["session"] == override.get("session") and bar["instrument"] == jq_code(
                    override.get("instrument", "")):
                bar.update(override.get("patch") or {})
    return {
        "schema_version": "fuxi.history_input.1",
        "snapshot": {
            "schema_version": "fuxi.historical_snapshot.1", "source_id": "fixture",
            "exposure": "research_exposed", "sessions": list(SESSIONS),
            "evaluation_start": SESSIONS[0], "instruments": instruments, "universe": members,
            "decision_times": {d: d + "T18:30:00+08:00" for d in SESSIONS},
            "columns": {f: [[1.0] * len(instruments) for _ in SESSIONS]
                        for f in ("open", "high", "low", "close", "volume", "amount")},
            "barriers": [[-1] * len(instruments) for _ in SESSIONS],
            "fields": {f: {"unit": "price", "frequency": "daily", "adjustment": "raw"}
                       for f in ("open", "high", "low", "close", "volume", "amount")},
            "quality": quality or {"pit_certified": False, "universe_certified": False,
                                   "account_events_certified": False, "gaps": []},
            "universe_audit": audit if audit is not None else {
                "reasons_by_session": {d: {"listing_age": 2} for d in SESSIONS},
                "cells": 36, "reason_vocabulary": ["listing_age"]},
            "provenance": {"plan_hash": "fixture"},
        },
        "market": {
            "sessions": list(SESSIONS), "bars": bars, "actions": [], "action_terms": [],
            "execution": {"initial_cash": "1000000", "buy_lot": 100, "sell_lot": 100,
                          "buy_commission": "0.0003", "sell_commission": "0.0003",
                          "minimum_commission": "5", "sell_tax": "0.0005",
                          "settlement_sessions": 1, "price_model": "next_session_open",
                          "cost_status": "declared_research_assumptions_not_broker_certified",
                          "sell_tax_schedule": [["0001-01-01", "0.001"],
                                                ["2023-08-28", "0.0005"]]},
            "event_coverage_certified": False, "tradability_certified": False,
            "market_hash": "fixture_market_hash",
        },
        "research_clock": SESSIONS[-1] + "T20:00:00+08:00",
        "audit": audit if isinstance(audit, dict) and "reasons_by_session" in audit else {},
    }


COHORT = {"rule_kind": "full_population_hash_sample", "salt": SALT, "target_size": len(CODES),
          "sampling_key_fields": ["ts_code"], "sampling_frame": "full_lifecycle_table",
          "frame_predicate": FRAME_PREDICATE,
          "posthoc_exclusion": False, "rule": "identifier-keyed hash sample"}

LIFECYCLE_CSV = "ts_code,list_date,delist_date,list_status\n" + "".join(
    "%s,19990104,,L\n" % code for code in CODES)


def reviewed_certification_fixture(root):
    """Write a payload, a lifecycle table and a *derived* certificate to ``root``.

    Returns a ``register-data`` specification whose reviewed evidence files are the
    three artifacts. Tests that only need a valid data receipt use this instead of
    asserting the four booleans by hand, which the operator path no longer accepts.
    """
    import hashlib
    import json
    from pathlib import Path
    from fuxi_v1.pit_universe import read_lifecycle_file
    root = Path(root)
    payload = fixture_payload()
    lifecycle_path = root / "delist_dates.csv"
    lifecycle_path.write_text(LIFECYCLE_CSV, encoding="utf-8", newline="\n")
    certificate = certify(payload, declaration={"cohort": dict(COHORT)},
                          lifecycle=read_lifecycle_file(lifecycle_path))
    require(certificate["blocking_gaps"] == [], "fixture does not certify: " + str(
        certificate["blocking_gaps"]))
    payload_path, certificate_path = root / "payload.json", root / "certificate.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8", newline="\n")
    files = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (payload_path, certificate_path, lifecycle_path)}
    return {"reviewer": "test_fixture_only", "reviewed_at": "2026-01-01T00:00:00+00:00",
            "evidence_files": files, "certificate_path": str(certificate_path.resolve()),
            "snapshot_path": str(payload_path.resolve()),
            "lifecycle_path": str(lifecycle_path.resolve()),
            "snapshot_hash": certificate["snapshot_hash"]}


def run(payload=None, declaration=None, **cohort_overrides):
    """Certify the fixture; keyword arguments refine the cohort rule block."""
    block = dict(COHORT, **cohort_overrides)
    settings = dict(declaration or {})
    settings["cohort"] = block
    return certify(payload or fixture_payload(), declaration=settings, lifecycle=life_table())


class CertificationTests(unittest.TestCase):
    def test_clean_payload_certifies_every_flag(self):
        certificate = run()
        self.assertEqual(certificate["schema_version"], SCHEMA)
        self.assertEqual(certificate["blocking_gaps"], [])
        for name, flag in certificate["flags"].items():
            self.assertTrue(flag["value"], name + " " + str(flag["blocking_gaps"]))

    def test_decision_time_before_declared_clock_refuses_pit(self):
        payload = fixture_payload()
        payload["snapshot"]["decision_times"] = {d: d + "T09:00:00+08:00" for d in SESSIONS}
        certificate = run(payload)
        self.assertFalse(certificate["flags"]["pit_certified"]["value"])
        self.assertIn("decision_before_declared_availability:3",
                      certificate["flags"]["pit_certified"]["blocking_gaps"])

    def test_naive_decision_clock_refused(self):
        payload = fixture_payload()
        payload["snapshot"]["decision_times"] = {d: d + "T18:30:00" for d in SESSIONS}
        self.assertFalse(run(payload)["flags"]["pit_certified"]["value"])

    def test_forward_looking_field_refused(self):
        payload = fixture_payload()
        payload["snapshot"]["fields"]["forward_return"] = {"unit": "ratio", "frequency": "daily",
                                                          "adjustment": "raw"}
        certificate = run(payload, declaration={"fields": ["close", "forward_return"]})
        self.assertFalse(certificate["flags"]["pit_certified"]["value"])
        self.assertTrue(any(g.startswith("forward_looking_field_in_feature_set")
                            for g in certificate["flags"]["pit_certified"]["blocking_gaps"]))

    def test_non_daily_field_refused(self):
        payload = fixture_payload()
        payload["snapshot"]["fields"]["close"]["frequency"] = "monthly"
        certificate = run(payload)
        self.assertIn("non_daily_field:close",
                      certificate["flags"]["pit_certified"]["blocking_gaps"])

    def test_posthoc_exclusion_refuses_universe(self):
        certificate = run(posthoc_exclusion=True)
        flag = certificate["flags"]["universe_certified"]
        self.assertFalse(flag["value"])
        self.assertIn("posthoc_selection_not_pit_neutral", flag["blocking_gaps"])

    def test_undeclared_posthoc_status_refused(self):
        declaration = dict(COHORT)
        declaration.pop("posthoc_exclusion")
        certificate = certify(fixture_payload(), declaration=declaration, lifecycle=life_table())
        self.assertIn("selection_rule_posthoc_status_undeclared",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])

    def test_outcome_keyed_sampling_rule_refused(self):
        certificate = run(sampling_key_fields=["ts_code", "delist_date"])
        self.assertIn("sampling_rule_depends_on_outcomes:delist_date",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])

    def test_retrospective_deletion_is_detected_by_reproduction(self):
        payload = fixture_payload()
        payload["snapshot"]["instruments"] = payload["snapshot"]["instruments"][:-1]
        certificate = run(payload)
        flag = certificate["flags"]["universe_certified"]
        self.assertFalse(flag["value"])
        self.assertIn("cohort_not_reproducible_from_declared_rule", flag["blocking_gaps"])
        self.assertEqual(flag["basis"]["reproduction_only_in_rule"], ["600004.SH"])

    def test_member_outside_lifecycle_refused(self):
        payload = fixture_payload(universe_overrides={SESSIONS[0]: LIVE})
        payload["snapshot"]["universe"][SESSIONS[0]] = payload["snapshot"]["universe"][SESSIONS[0]] + [
            jq_code("000009.SZ")]
        payload["snapshot"]["instruments"] = payload["snapshot"]["instruments"] + [jq_code("000009.SZ")]
        life = life_table([{"ts_code": "000009.SZ", "list_date": "20210105",
                            "delist_date": "", "list_status": "L"}])
        certificate = certify(payload, declaration=dict(COHORT, target_size=len(CODES) + 1),
                              lifecycle=life)
        self.assertIn("member_outside_declared_lifecycle:1",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])

    def test_undeclared_removal_reason_refused(self):
        payload = fixture_payload()
        payload["snapshot"]["universe_audit"]["reasons_by_session"] = {
            d: {"candidate_was_weak": 2} for d in SESSIONS}
        certificate = run(payload)
        self.assertIn("undeclared_removal_reason:candidate_was_weak",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])

    def test_removal_reasons_must_be_bound_to_the_snapshot(self):
        payload = fixture_payload()
        payload["snapshot"].pop("universe_audit")
        certificate = run(payload)
        self.assertIn("removal_reasons_not_bound_to_snapshot",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])

    def test_every_reason_the_adapter_emits_is_declared(self):
        from fuxi_v1.certification import EX_ANTE_REASONS
        from fuxi_v1.history import name_state
        emitted = {"not_listed_or_delisted", "SME_before_mainboard_merger", "listing_age",
                   "halt_or_tradability_unknown", "quote_missing", "corporate_action_unknown"}
        for state in ("st", "unknown", "conflicting", "not_yet_published"):
            emitted.add("name_" + state)
        self.assertEqual(name_state([], "2021-01-04")[0], "unknown")
        self.assertEqual(sorted(emitted - EX_ANTE_REASONS), [],
                         "adapter emits a removal reason the certificate would call silent")

    def test_declared_ex_ante_vocabulary_can_be_narrowed_explicitly(self):
        payload = fixture_payload()
        certificate = run(payload, declaration={"cohort": dict(COHORT),
                                                "ex_ante_reasons": ["quote_missing"]})
        self.assertIn("undeclared_removal_reason:listing_age", certificate["blocking_gaps"])
        self.assertEqual(certificate["flags"]["universe_certified"]["basis"][
            "declared_ex_ante_reasons"], ["quote_missing"])

    def test_lifecycle_membership_is_off_identifier_form(self):
        life = life_table()
        self.assertTrue(listed_at(life, "000001.SZ", "2021-01-04"))
        self.assertTrue(listed_at(life, "000001.XSHE", "2021-01-04"))
        self.assertEqual(canonical_code("600001.XSHG"), "600001.SH")

    def test_unknown_action_on_member_cell_refuses_account_evidence(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[1], "instrument": "000002.SZ",
                                                  "patch": {"corporate_action_known": False}}])
        certificate = run(payload)
        self.assertIn("member_cells_with_unknown_corporate_action:1",
                      certificate["flags"]["account_events_certified"]["blocking_gaps"])
        self.assertIn("member_cells_with_unknown_event_coverage:1",
                      certificate["flags"]["event_coverage_certified"]["blocking_gaps"])

    def test_unknown_action_outside_members_does_not_refuse(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[1], "instrument": "600004.SH",
                                                  "patch": {"corporate_action_known": False}}])
        certificate = run(payload)
        self.assertTrue(certificate["flags"]["account_events_certified"]["value"])
        self.assertEqual(certificate["flags"]["account_events_certified"]["basis"][
            "nonmember_unknown_cells"], 1)

    def test_ambiguous_pause_state_refuses_tradability(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[0], "instrument": "000003.SZ",
                                                  "patch": {"paused": None}}])
        certificate = run(payload)
        self.assertIn("member_cells_with_ambiguous_pause_state:1",
                      certificate["flags"]["tradability_certified"]["blocking_gaps"])

    def test_missing_quote_and_limit_refused(self):
        payload = fixture_payload(bars_overrides=[
            {"session": SESSIONS[0], "instrument": "000003.SZ", "patch": {"open": None}},
            {"session": SESSIONS[1], "instrument": "000003.SZ", "patch": {"up_limit": None}}])
        flag = run(payload)["flags"]["tradability_certified"]
        self.assertIn("member_cells_missing_quote_or_volume:1", flag["blocking_gaps"])
        self.assertIn("member_cells_missing_price_limits:1", flag["blocking_gaps"])

    def test_open_outside_limits_refused(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[0], "instrument": "000003.SZ",
                                                  "patch": {"open": 12.0}}])
        self.assertIn("member_cells_where_open_outside_limits:1",
                      run(payload)["flags"]["tradability_certified"]["blocking_gaps"])

    def test_halted_member_with_observed_mark_refused(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[0], "instrument": "000003.SZ",
                                                  "patch": {"paused": True,
                                                            "mark_source": "observed_close"}}])
        self.assertIn("halted_member_cells_with_observed_mark:1",
                      run(payload)["flags"]["tradability_certified"]["blocking_gaps"])

    def test_incomplete_execution_rules_refused(self):
        payload = fixture_payload()
        payload["market"]["execution"].pop("settlement_sessions")
        certificate = run(payload)
        self.assertIn("execution_rule_missing:settlement_sessions",
                      certificate["flags"]["execution_rules_certified"]["blocking_gaps"])

    def test_undeclared_price_model_refused(self):
        payload = fixture_payload()
        payload["market"]["execution"]["price_model"] = "same_session_close"
        self.assertIn("undeclared_price_model",
                      run(payload)["flags"]["execution_rules_certified"]["blocking_gaps"])

    def test_missing_lifecycle_table_refuses_universe_only(self):
        certificate = certify(fixture_payload(), declaration=dict(COHORT), lifecycle=None)
        self.assertIn("lifecycle_table_not_supplied",
                      certificate["flags"]["universe_certified"]["blocking_gaps"])
        self.assertTrue(certificate["flags"]["pit_certified"]["value"])

    def test_receipt_booleans_are_the_flag_values(self):
        certificate = run()
        receipt = derived_data_receipt(certificate)
        for name in ("pit_certified", "universe_certified", "account_events_certified",
                     "execution_rules_certified", "event_coverage_certified",
                     "tradability_certified"):
            self.assertEqual(receipt[name], certificate["flags"][name]["value"], name)
        self.assertTrue(receipt["derived"])
        self.assertEqual(receipt["certification_hash"], certificate["certification_hash"])

    def test_certificate_hash_ignores_wall_clock_but_tracks_content(self):
        first = certify(fixture_payload(), declaration=dict(COHORT), lifecycle=life_table(),
                        certified_at="2026-10-02T00:00:00+00:00")
        second = certify(fixture_payload(), declaration=dict(COHORT), lifecycle=life_table(),
                         certified_at="2026-10-03T00:00:00+00:00")
        self.assertEqual(first["certification_hash"], second["certification_hash"])
        moved = fixture_payload()
        moved["market"]["bars"][0]["close"] = 10.5
        third = certify(moved, declaration=dict(COHORT), lifecycle=life_table())
        self.assertNotEqual(first["certification_hash"], third["certification_hash"])

    def test_apply_keeps_source_truth_and_is_reproducible(self):
        certificate = run()
        payload = fixture_payload()
        original = deepcopy(payload)
        result = apply_to_payload(payload, certificate)
        self.assertEqual(payload, original)
        self.assertEqual(result["snapshot"], original["snapshot"])
        self.assertEqual(result["market"], original["market"])
        self.assertFalse(result["data_certificate"]["claim_scope"]["investment_eligibility_certified"])
        regenerated = run(result)
        self.assertEqual(regenerated["certification_hash"], certificate["certification_hash"])

    def test_apply_preserves_selection_bias_history(self):
        payload = fixture_payload(quality={"pit_certified": False, "universe_certified": False,
                                           "account_events_certified": False,
                                           "gaps": ["posthoc_universe_exclusion_selection_bias",
                                                    "kept_gap"]})
        result = apply_to_payload(payload, run(payload))
        self.assertIn("kept_gap", result["snapshot"]["quality"]["gaps"])
        self.assertIn("posthoc_universe_exclusion_selection_bias",
                         result["snapshot"]["quality"]["gaps"])

    def test_refused_flag_stays_refused_through_apply(self):
        payload = fixture_payload(bars_overrides=[{"session": SESSIONS[0], "instrument": "000003.SZ",
                                                  "patch": {"paused": None}}])
        certificate = run(payload)
        result = apply_to_payload(payload, certificate)
        self.assertFalse(certificate["flags"]["tradability_certified"]["value"])
        self.assertFalse(result["market"]["tradability_certified"])

    def test_direction_is_one_way(self):
        self.assertIn("only refuse", run()["direction"])

    def test_cohort_declaration_reads_the_plan_rule(self):
        plan = {"cohort": {"members": [{"ts_code": "000001.SZ"}], "salt": "s",
                           "hash_cutoffs": {"SH": "a", "SZ": "b"}, "per_exchange_initial": 48,
                           "selection": "per-exchange hash sample"}}
        declaration = cohort_declaration(plan)
        self.assertEqual(declaration["rule_kind"], "per_exchange_hash_cutoff")
        self.assertFalse(declaration["posthoc_exclusion"])
        plan["universe_restriction"] = {"posthoc_exclusion": True, "rule": "drop delisted names"}
        self.assertTrue(cohort_declaration(plan)["posthoc_exclusion"])

    def test_cohort_declaration_records_a_retrospective_restriction(self):
        plan = {"cohort": {"salt": "s", "hash_cutoffs": {"SH": "a", "SZ": "b"},
                           "per_exchange_initial": 48}}
        payload = {"audit": {"selection_audit": {"posthoc_exclusion": True, "rule": "drop 7 names",
                                                 "excluded": [{"ts_code": "000638.SZ"}]}}}
        declaration = cohort_declaration(plan, payload)
        self.assertTrue(declaration["posthoc_exclusion"])
        self.assertEqual(declaration["restriction_excluded"], ["000638.SZ"])
        self.assertEqual(declaration["parent_rule"]["salt"], "s")

    def test_claim_scope_never_lists_a_refused_flag(self):
        certificate = run(posthoc_exclusion=True)
        self.assertNotIn("universe_certified", certificate["claim_scope"]["certified"])
        self.assertIn("universe_certified", certificate["claim_scope"]["refused"])
        self.assertIn("market_breadth", certificate["claim_scope"]["not_covered"])


class PitUniverseTests(unittest.TestCase):
    def test_frame_retains_names_that_delist_inside_the_window(self):
        life = life_table([{"ts_code": "000100.SZ", "list_date": "19990104",
                            "delist_date": "20210105", "list_status": "D"}])
        rows = frame_rows(life, SESSIONS[0], SESSIONS[-1])
        self.assertIn("000100.SZ", {r["ts_code"] for r in rows})

    def test_frame_drops_names_that_delisted_before_the_window(self):
        life = life_table([{"ts_code": "000100.SZ", "list_date": "19990104",
                            "delist_date": "20200105", "list_status": "D"}])
        rows = frame_rows(life, SESSIONS[0], SESSIONS[-1])
        self.assertNotIn("000100.SZ", {r["ts_code"] for r in rows})

    def test_frame_drops_names_listed_after_the_window(self):
        life = life_table([{"ts_code": "000100.SZ", "list_date": "20220105",
                            "delist_date": "", "list_status": "L"}])
        rows = frame_rows(life, SESSIONS[0], SESSIONS[-1])
        self.assertNotIn("000100.SZ", {r["ts_code"] for r in rows})

    def test_reproduction_is_order_independent(self):
        life = life_table()
        codes, _ = reproduce_cohort(life, COHORT, SESSIONS)
        reversed_life = dict(reversed(list(life.items())))
        again, _ = reproduce_cohort(reversed_life, COHORT, SESSIONS)
        self.assertEqual(codes, again)

    def test_reproduction_requires_a_salt(self):
        codes, basis = reproduce_cohort(life_table(), {"rule_kind": "full_population_hash_sample",
                                                       "target_size": 1}, SESSIONS)
        self.assertIsNone(codes)
        self.assertIn("sampling_salt_undeclared", basis["blocking_gaps"])

    def test_unsupported_rule_is_refused_not_guessed(self):
        codes, basis = reproduce_cohort(life_table(), {"rule_kind": "hand_picked", "salt": SALT},
                                       SESSIONS)
        self.assertIsNone(codes)
        self.assertIn("unsupported_cohort_rule:hand_picked", basis["blocking_gaps"])

    def test_declared_cutoffs_must_match_the_rule(self):
        declaration = {"rule_kind": "per_exchange_hash_cutoff", "salt": SALT,
                       "per_exchange_initial": 2, "hash_cutoffs": {"SH": "wrong", "SZ": "wrong"}}
        codes, basis = reproduce_cohort(life_table(), declaration, SESSIONS)
        self.assertIsNone(codes)
        self.assertIn("declared_hash_cutoffs_do_not_match_rule", basis["blocking_gaps"])

    def test_survivor_test_does_not_fire_on_a_cohort_too_small_to_expect_one(self):
        """Two delisting names in a twelve-name frame cannot convict a ten-name cohort.

        The expected count is 1.67, so drawing none has probability 0.19: routine,
        not evidence. The v0.7-era rule (``kept == 0 and dying > 0``) called this a
        survivor filter, which would have refused every small panel for free.
        """
        life = life_table([{"ts_code": "000100.SZ", "list_date": "19990104",
                            "delist_date": "20210105", "list_status": "D"},
                           {"ts_code": "000101.SZ", "list_date": "19990104",
                            "delist_date": "20210106", "list_status": "D"}])
        dying = ["000100.SZ", "000101.SZ"]
        clean = neutrality_audit(life, SESSIONS, CODES + dying)
        self.assertFalse(clean["survivor_filter_detected"])
        self.assertEqual(clean["population_names_delisting_inside_window"], 2)
        self.assertEqual(clean["cohort_names_delisting_inside_window"], 2)
        filtered = neutrality_audit(life, SESSIONS, CODES)
        self.assertEqual(filtered["cohort_names_delisting_inside_window"], 0)
        self.assertGreater(filtered["observed_zero_probability"], filtered["alpha"])
        self.assertFalse(filtered["survivor_filter_detected"])

    def test_neutrality_audit_convicts_when_zero_is_implausible(self):
        """Half the frame delists inside the window and the cohort is the other half.

        Expected count is 5.0 and the probability of drawing none is 0.0067, below
        the declared alpha. Note the expected count peaks when the cohort covers
        half the frame: a *tiny* cohort can never be convicted, which is the point.
        """
        rows = [{"ts_code": "0001%02d.SZ" % i, "list_date": "19990104"} for i in range(10)]
        rows += [{"ts_code": "0002%02d.SZ" % i, "list_date": "19990104",
                  "delist_date": "20210105", "list_status": "D"} for i in range(10)]
        life = lifecycle_from_rows(rows)
        frame = [r["ts_code"] for r in frame_rows(life, SESSIONS[0], SESSIONS[-1])]
        audit = neutrality_audit(life, SESSIONS, frame[:10])
        self.assertEqual(len(frame), 20)
        self.assertEqual(audit["population_names_delisting_inside_window"], 10)
        self.assertEqual(audit["cohort_names_delisting_inside_window"], 0)
        self.assertEqual(audit["expected_cohort_delisting_count"], 5.0)
        self.assertLess(audit["observed_zero_probability"], audit["alpha"])
        self.assertTrue(audit["survivor_filter_detected"])

    def test_cohort_selection_must_not_depend_on_frame_order(self):
        """Reproduction alone would accept a positional rule; order invariance refuses it.

        ``_pick`` is exercised on a reversed frame by ``reproduce_cohort``, so a rule
        that secretly took "the first N rows" cannot also satisfy this basis field.
        """
        reversed_life = list(life_table().items())[::-1]
        codes, basis = reproduce_cohort(dict(reversed_life),
                                        dict(COHORT, target_size=5), SESSIONS)
        self.assertTrue(basis["permutation_invariant"])
        straight, _ = reproduce_cohort(life_table(), dict(COHORT, target_size=5), SESSIONS)
        self.assertEqual(codes, straight)

    def test_undeclared_frame_predicate_refuses_only_representativeness(self):
        certificate = run(**{"frame_predicate": None})
        self.assertTrue(certificate["flags"]["universe_certified"]["value"])
        self.assertFalse(certificate["flags"]["market_representativeness_certified"]["value"])
        self.assertIn("sampling_frame_predicate_undeclared",
                      certificate["flags"]["market_representativeness_certified"]["blocking_gaps"])

    def test_inflated_frame_predicate_is_refused(self):
        certificate = run(**{"frame_predicate": "all A-share listings including STAR and ChiNext"})
        self.assertTrue(certificate["flags"]["universe_certified"]["value"])
        self.assertFalse(certificate["flags"]["market_representativeness_certified"]["value"])
        self.assertIn("declared_frame_predicate_does_not_match_derived",
                      certificate["flags"]["market_representativeness_certified"]["blocking_gaps"])

    def test_representativeness_is_not_a_mirror_of_reproducibility(self):
        """The two claims must be able to disagree, or one of them certifies nothing."""
        certificate = run(**{"frame_predicate": None})
        basis = certificate["flags"]["market_representativeness_certified"]["basis"]
        self.assertIsNotNone(basis["derived_frame_predicate"])
        self.assertIn("cohort_names_delisting_inside_window", basis)
        self.assertIn("expected_cohort_delisting_count", basis)

    def test_survivor_filtered_cohort_blocks_the_universe_flag(self):
        """The rc5.2 shape, with enough delisting names for the test to have power.

        Frame is 24 names of which half delist inside the window; the frozen panel is
        the 12 survivors. Expected count is 6 and the probability of drawing none is
        0.0025, so the audit convicts -- and the panel is also not reproducible from
        the declared rule, because that rule draws all 24.
        """
        dying = ["0003%02d.SZ" % i for i in range(12)]
        life = life_table([{"ts_code": c, "list_date": "19990104",
                            "delist_date": "20210105", "list_status": "D"} for c in dying])
        certificate = certify(fixture_payload(),
                              declaration={"cohort": dict(COHORT, target_size=24)},
                              lifecycle=life)
        basis = certificate["flags"]["universe_certified"]["basis"]["neutrality"]
        self.assertEqual(basis["population_names_delisting_inside_window"], 12)
        self.assertEqual(basis["cohort_names_delisting_inside_window"], 0)
        self.assertEqual(basis["expected_cohort_delisting_count"], 6.0)
        self.assertTrue(basis["survivor_filter_detected"])
        self.assertFalse(certificate["flags"]["universe_certified"]["value"])
        gaps = certificate["flags"]["universe_certified"]["blocking_gaps"]
        self.assertIn("cohort_not_reproducible_from_declared_rule", gaps)
        self.assertIn("survivor_filter_detected_in_cohort", gaps)
        self.assertFalse(certificate["flags"]["market_representativeness_certified"]["value"])

    def test_build_pit_cohort_declares_and_audits(self):
        codes, declaration, audit = build_pit_cohort(
            life_table(), SESSIONS, salt=SALT, target_size=len(CODES),
            rule_kind="full_population_hash_sample")
        self.assertEqual(len(codes), len(CODES))
        self.assertFalse(declaration["posthoc_exclusion"])
        self.assertEqual(declaration["sampling_key_fields"], ["ts_code"])
        self.assertIn("window", audit)

    def test_full_population_sample_requires_enough_frame(self):
        rows = frame_rows(life_table(), SESSIONS[0], SESSIONS[-1])
        picked, basis = full_population_hash_sample(rows, SALT, len(rows) + 1)
        self.assertIsNone(picked)
        self.assertIn("frame_smaller_than_target_size", basis["blocking_gaps"])

    def test_frame_predicate_excludes_non_mainboard(self):
        life = lifecycle_from_rows([{"ts_code": "300001.SZ", "list_date": "19990104"},
                                    {"ts_code": "688001.SH", "list_date": "19990104"},
                                    {"ts_code": "000001.SZ", "list_date": "19990104"}])
        self.assertEqual([r["ts_code"] for r in frame_rows(life, SESSIONS[0], SESSIONS[-1])],
                         ["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
