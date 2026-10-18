"""M4 fitness triage contracts: economic distance positions readings, grants nothing."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fuxi_v1.calibration import (BREAKEVEN_IC, BREAKEVEN_SOURCE, breakeven_for,
                                 economic_distance, percentile_of)
from fuxi_v1.campaign import run_campaign
from fuxi_v1.contracts import ContractError, canonical, evaluation
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.learning import feedback_card, fitness_key, fitness_triage
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_kernel import scratch_directory

POLICY = {"policy_hash": "p", "request": {"research_start": "a", "research_end": "b"}}


def _result(mean_ic):
    e1 = evaluation("E1", "valid", "research", diagnostics={"spearman": {"mean": mean_ic}})
    e2 = evaluation("E2", "valid", "similarity", metrics={})
    e3 = evaluation("E3", "data_insufficient", "market_evidence_not_certified")
    return {"snapshot_hash": "s", "evaluations": [e1, e2, e3]}


def _card(cid, mean_ic, distance, comparisons=0, gaps=0):
    return {"candidate_id": cid, "mean_rank_ic": mean_ic, "economic_distance": distance,
            "parent_comparisons": [{} for _ in range(comparisons)],
            "gap_counts": [{} for _ in range(gaps)]}


def _campaign_store(base, name, report):
    return EvidenceStore(Path(base) / name / "runs" / report["campaign_id"] / "evidence.sqlite")


def _campaign_facts(base, name, report, kind):
    store = _campaign_store(base, name, report)
    try:
        return [p for _, p in store.facts(kind)]
    finally:
        store.close()


def _campaign_config(base, name, report):
    store = _campaign_store(base, name, report)
    try:
        row = store.db.execute("SELECT config FROM campaigns WHERE id=?", (report["campaign_id"],)).fetchone()
        return json.loads(row[0])
    finally:
        store.close()


class EconomicDistanceTests(unittest.TestCase):
    def test_frozen_anchor_values_and_source(self):
        self.assertEqual(BREAKEVEN_IC, {"equal_weight_weekly_15bps": 0.108, "r4_reference_weekly_15bps": 0.083})
        self.assertEqual(breakeven_for(), 0.108)
        self.assertEqual(breakeven_for("r4_reference_weekly_15bps"), 0.083)
        self.assertIn("fuxi_ic_ceiling_20260924", BREAKEVEN_SOURCE["report"])
        self.assertEqual(BREAKEVEN_SOURCE["usage"], "descriptive_triage_positioning_not_reward")
        with self.assertRaisesRegex(ContractError, "unknown breakeven key"):
            breakeven_for("nope")

    def test_distance_is_a_plain_ratio_and_missing_propagates(self):
        self.assertAlmostEqual(economic_distance(0.054, 0.108), 0.5)
        self.assertAlmostEqual(economic_distance(-0.054, 0.108), -0.5)
        self.assertIsNone(economic_distance(None, 0.108))
        self.assertIsNone(economic_distance(float("nan"), 0.108))

    def test_non_positive_anchor_is_rejected(self):
        for bad in (0.0, -0.1, float("nan"), float("inf"), None, True):
            with self.assertRaisesRegex(ContractError, "positive finite breakeven"):
                economic_distance(0.05, bad)


class FitnessKeyTests(unittest.TestCase):
    def test_ordering_priority_and_ties(self):
        cards = [_card("b", 0.054, 0.5), _card("a", 0.054, 0.5), _card("c", 0.097, 0.9),
                 {"candidate_id": "d", "mean_rank_ic": 0.08}, {"candidate_id": "e"}]
        ordered = [c["candidate_id"] for c in sorted(cards, key=fitness_key)]
        self.assertEqual(ordered, ["c", "a", "b", "d", "e"])

    def test_key_is_deterministic(self):
        card = _card("x", 0.05, 0.46)
        self.assertEqual(fitness_key(card), fitness_key(dict(card)))

    def test_legacy_cards_fall_back_to_raw_ic(self):
        legacy = {"candidate_id": "L", "mean_rank_ic": 0.08}
        self.assertEqual(fitness_key(legacy)[0], 1)
        newer = _card("N", None, None)
        self.assertEqual(fitness_key(newer)[0], 2)


class TriageListTests(unittest.TestCase):
    def test_top_k_swap_is_disclosed(self):
        cards = [_card("hi_comparisons", 0.01, 0.1, comparisons=2),
                 _card("hi_distance", 0.09, 0.9)]
        triage = fitness_triage(cards, limit=1)
        self.assertEqual(triage["top_k_baseline"], ["hi_comparisons"])
        self.assertEqual(triage["top_k_triage"], ["hi_distance"])
        self.assertEqual(triage["entered_top_k"], ["hi_distance"])
        self.assertEqual(triage["left_top_k"], ["hi_comparisons"])
        self.assertEqual(triage["n_cards"], 2)
        self.assertEqual([row["fit_rank"] for row in triage["rows"]], [0, 1])

    def test_digest_reproduces_under_input_permutation(self):
        cards = [_card("a", 0.01, 0.1), _card("b", 0.09, 0.9), _card("c", 0.05, 0.5)]
        first = fitness_triage(cards)
        second = fitness_triage(list(reversed(cards)))
        self.assertEqual(first["triage_digest"], second["triage_digest"])
        self.assertEqual(first["rows"], second["rows"])

    def test_empty_card_set_is_a_zero_count_not_a_fabricated_order(self):
        triage = fitness_triage([])
        self.assertEqual(triage["n_cards"], 0)
        self.assertEqual(triage["rows"], [])
        with self.assertRaisesRegex(ContractError, "positive triage display limit"):
            fitness_triage([], limit=0)


class FeedbackCardFieldTests(unittest.TestCase):
    def test_additive_fields_with_vehicle_anchor(self):
        card = feedback_card({"candidate_id": "c1"}, _result(0.054), POLICY, {}, "e", "state",
                             vehicle_ic_values=[0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(card["economic_distance"], 0.054 / 0.108)
        self.assertEqual(card["vehicle_percentile"], 1.0)
        self.assertEqual(card["economic_distance_context"]["vehicle_n"], 4)
        self.assertEqual(card["economic_distance_context"]["breakeven_key"], "equal_weight_weekly_15bps")
        self.assertEqual(card["economic_distance_context"]["usage"], "descriptive_triage_positioning_not_reward")
        self.assertFalse(card["qualification_transfer_allowed"])

    def test_missing_ic_and_missing_vehicle_stay_missing(self):
        card = feedback_card({"candidate_id": "c2"}, _result(None), POLICY, {}, "e", "state")
        self.assertIsNone(card["economic_distance"])
        self.assertIsNone(card["vehicle_percentile"])
        self.assertEqual(card["economic_distance_context"]["vehicle_n"], 0)


class CampaignIntegrationTests(unittest.TestCase):
    def _run_at(self, base, name, *, env=None, rounds=2, candidates_per_round=1, protocol=None,
                pause_after_round=None, campaign_id=None, reuse=False):
        root = Path(base) / name
        if not reuse:
            root.mkdir()
        source = root / "input.json"
        if not source.exists():
            source.write_bytes(canonical(synthetic_campaign(130, 4)))
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUXI_V1_")}
        clean.update(env or {})
        with patch.dict(os.environ, clean, clear=True):
            return run_campaign(source, root / "runs", rounds=rounds,
                                candidates_per_round=candidates_per_round, model_algorithms=(),
                                study_protocol=protocol, pause_after_round=pause_after_round,
                                campaign_id=campaign_id)

    def test_default_run_records_triage_facts(self):
        with scratch_directory() as directory:
            _, report = self._run_at(directory, "arm")
            self.assertEqual(report["status"], "completed")
            facts = _campaign_facts(directory, "arm", report, "fitness_triage")
            self.assertEqual(len(facts), 2)
            self.assertTrue(all(row["mode"] == "enforce" for row in facts))
            self.assertEqual(_campaign_config(directory, "arm", report)["fitness_triage"], "enforce")

    def test_shadow_records_display_without_changing_behavior(self):
        protocol = {"schema_version": "fuxi.controlled_study.1", "arm_schedule": ["fixed", "refine"],
                    "max_trials": 2, "checkpoint_every": 2, "memory_sources": []}
        with scratch_directory() as directory:
            _, off_report = self._run_at(directory, "off", env={"FUXI_V1_FITNESS_TRIAGE": "off"},
                                         protocol=protocol)
            _, shadow_report = self._run_at(directory, "shadow",
                                            env={"FUXI_V1_FITNESS_TRIAGE": "shadow"}, protocol=protocol)
            facts = _campaign_facts(directory, "shadow", shadow_report, "fitness_triage")
            self.assertEqual(len(facts), 2)
            for row in facts:
                self.assertEqual(row["mode"], "shadow")
                self.assertGreater(row["n_cards"], 0)
                self.assertEqual(row["enforce_consumer"], "imported_memory_ordering")
                self.assertEqual(row["usage"], "descriptive_display_only_never_acceptance")
            self.assertEqual(_campaign_facts(directory, "off", off_report, "fitness_triage"), [])
            signature = lambda report: [(r.get("formula_hash"), r.get("mean_rank_ic"), r.get("eval_status"))
                                        for r in report["candidate_results"]]
            self.assertEqual(signature(off_report), signature(shadow_report))
            self.assertEqual(_campaign_config(directory, "shadow", shadow_report)["fitness_triage"], "shadow")

    def test_resume_with_changed_fitness_mode_is_rejected(self):
        with scratch_directory() as directory:
            _, partial = self._run_at(directory, "resume", env={"FUXI_V1_FITNESS_TRIAGE": "shadow"},
                                      pause_after_round=1, campaign_id="campaign_fitness_resume")
            self.assertEqual(partial["status"], "checkpoint_paused")
            with self.assertRaisesRegex(ContractError, "fitness triage mode changed"):
                self._run_at(directory, "resume", env={"FUXI_V1_FITNESS_TRIAGE": "off"}, rounds=1,
                             pause_after_round=1, campaign_id="campaign_fitness_resume", reuse=True)

    def test_vehicle_pool_feeds_percentiles_from_same_batch_random_arm(self):
        protocol = {"schema_version": "fuxi.controlled_study.1", "arm_schedule": ["random_ast", "random_ast"],
                    "max_trials": 2, "checkpoint_every": 2, "memory_sources": []}
        with scratch_directory() as directory:
            _, report = self._run_at(directory, "vehicles", env={"FUXI_V1_FITNESS_TRIAGE": "shadow"},
                                     protocol=protocol)
            self.assertEqual(report["status"], "completed")
            facts = _campaign_facts(directory, "vehicles", report, "fitness_triage")
            self.assertEqual(facts[0]["vehicle_n"], 1)
            self.assertEqual(facts[1]["vehicle_n"], 2)
            self.assertEqual(facts[0]["rows"][0]["vehicle_percentile"], None)
            cards = _campaign_facts(directory, "vehicles", report, "feedback_v2")
            self.assertIsNone(cards[0]["vehicle_percentile"])
            self.assertIsNotNone(cards[1]["vehicle_percentile"])
            self.assertEqual(cards[1]["economic_distance_context"]["vehicle_n"], 1)


if __name__ == "__main__":
    unittest.main()
