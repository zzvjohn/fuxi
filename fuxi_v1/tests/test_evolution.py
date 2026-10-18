"""M2/M3 evolution upgrade contracts: niches, typed null arm and edit motifs grant no eligibility."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fuxi_v1.campaign import run_campaign
from fuxi_v1.contracts import ContractError, canonical
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.learning import (ast_diff_motifs, motif_gate, motif_ledger, motif_observations, motif_vetoes)
from fuxi_v1.monitoring import diversity_representatives, niche_descriptors, pareto_archive, select_parents
from fuxi_v1.search import generate
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.tests.test_kernel import scratch_directory

FIELDS = {"close": {"unit": "price"}, "volume": {"unit": "shares"}}


def _member(cid, quality, complexity, family, behavior):
    return {"candidate_id": cid, "quality": quality, "complexity": complexity, "valid_reading": True,
            "structural_family": family, "behavior_hash": behavior}


def _campaign_store(root, report):
    return EvidenceStore(Path(root)/"runs"/report["campaign_id"]/"evidence.sqlite")


def _campaign_facts(root, report, kind):
    store = _campaign_store(root, report)
    try:
        return [p for _, p in store.facts(kind)]
    finally:
        store.close()


def _campaign_config(root, report):
    store = _campaign_store(root, report)
    try:
        row = store.db.execute("SELECT config FROM campaigns WHERE id=?", (report["campaign_id"],)).fetchone()
        return json.loads(row[0])
    finally:
        store.close()


class NicheTests(unittest.TestCase):
    def test_pareto_policy_reproduces_frontier(self):
        archive = [_member("a", .5, 6, "F1", "B1"), _member("b", .4, 4, "F2", "B2"),
                   _member("c", .4, 8, "F3", "B3")]
        self.assertEqual([m["candidate_id"] for m in pareto_archive(archive)], ["a", "b"])
        self.assertEqual([m["candidate_id"] for m in select_parents(archive, "pareto")],
                         [m["candidate_id"] for m in pareto_archive(archive)])

    def test_niche_representatives_cover_each_nonempty_niche_once(self):
        archive = [_member("a", .5, 4, "F1", "B1"), _member("b", .6, 4, "F1", "B1"),
                   _member("c", .1, 6, "F2", "B2")]
        reps = diversity_representatives(archive, 8)
        self.assertEqual({m["candidate_id"] for m in reps}, {"b", "c"})

    def test_niche_sampling_spreads_deterministically_when_over_budget(self):
        archive = [_member("m%02d" % i, 1 - i * .01, 3 + i, "F%02d" % i, "B%02d" % i) for i in range(16)]
        first = [m["candidate_id"] for m in diversity_representatives(archive, 8)]
        second = [m["candidate_id"] for m in diversity_representatives(archive, 8)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(first[0], "m00")
        self.assertEqual(first[-1], "m15")

    def test_descriptor_folds_window_variants_into_one_niche(self):
        self.assertEqual(niche_descriptors(_member("x", .5, 3, "F", "B")),
                         niche_descriptors(_member("y", .9, 5, "F", "B")))
        self.assertNotEqual(niche_descriptors(_member("x", .5, 3, "F", "B")),
                            niche_descriptors(_member("y", .9, 9, "F", "B")))

    def test_niche_mix_unions_frontier_with_unique_structures(self):
        archive = [_member("a", .5, 4, "F1", "B1"), _member("b", .4, 6, "F2", "B2"),
                   _member("c", .3, 6, "F3", "B3")]
        self.assertEqual([m["candidate_id"] for m in select_parents(archive, "pareto")], ["a"])
        self.assertEqual([m["candidate_id"] for m in select_parents(archive, "niche_mix")], ["a", "b", "c"])

    def test_empty_archive_and_unknown_policy(self):
        self.assertEqual(select_parents([], "niche_mix"), [])
        self.assertEqual(diversity_representatives([], 8), [])
        with self.assertRaisesRegex(ContractError, "unknown parent policy"):
            select_parents([], "greedy")


class RandomAstArmTests(unittest.TestCase):
    def test_arm_is_pure_null_and_deterministic(self):
        first = generate(FIELDS, arm="random_ast", count=4, seed=7)
        second = generate(FIELDS, arm="random_ast", count=4, seed=7)
        third = generate(FIELDS, arm="random_ast", count=4, seed=8)
        self.assertEqual(len(first), 4)
        self.assertEqual([c["formula_hash"] for c in first], [c["formula_hash"] for c in second])
        self.assertNotEqual([c["formula_hash"] for c in first], [c["formula_hash"] for c in third])
        for candidate in first:
            self.assertEqual(candidate["action"], "calibration_random_null")
            self.assertEqual(candidate["hypothesis"]["mechanism"], "typed_random_null")
            self.assertNotEqual(candidate["action"], "typed_random_template")

    def test_seen_entries_are_withheld(self):
        first = generate(FIELDS, arm="random_ast", count=4, seed=7)
        known = {first[0]["formula_hash"]}
        filtered = generate(FIELDS, arm="random_ast", count=4, seed=7, seen=known)
        self.assertEqual(len(filtered), 3)
        self.assertNotIn(first[0]["formula_hash"], [c["formula_hash"] for c in filtered])


class MotifTests(unittest.TestCase):
    PARENT = {"op": "rank", "args": [{"op": "return", "args": [{"field": "close"}], "window": 5}]}

    def test_known_edit_forms(self):
        window_child = {"op": "rank", "args": [{"op": "return", "args": [{"field": "close"}], "window": 10}]}
        self.assertEqual(ast_diff_motifs(self.PARENT, window_child), ["swap_window@0:5->10"])
        self.assertEqual(ast_diff_motifs(self.PARENT, {"op": "neg", "args": [self.PARENT]}), ["wrap:neg@"])
        a = {"op": "rank", "args": [{"field": "close"}]}
        b = {"op": "rank", "args": [{"field": "volume"}]}
        self.assertEqual(ast_diff_motifs(a, {"op": "add", "args": [a, b]}), ["binop:add@"])
        self.assertEqual(ast_diff_motifs(a, b), ["swap_field@0:close->volume"])
        self.assertEqual(ast_diff_motifs(self.PARENT, self.PARENT), [])

    def test_ledger_and_veto_asymmetry(self):
        ledger = motif_ledger([("bad@0", False)] * 4 + [("good@0", True)] * 4)
        self.assertEqual(motif_vetoes(ledger), [])
        ledger = motif_ledger([("bad@0", False)] * 5 + [("good@0", True)] * 4)
        self.assertEqual(motif_vetoes(ledger), ["bad@0"])
        self.assertEqual(ledger["bad@0"]["n"], 5)
        self.assertLess(ledger["bad@0"]["upper95"], .45)

    def test_gate_shadow_counts_enforce_withholds(self):
        window_child = {"op": "rank", "args": [{"op": "return", "args": [{"field": "close"}], "window": 10}]}
        hit = {"candidate_id": "hit", "canonical_ast": window_child, "parent_ids": ["p"]}
        ok = {"candidate_id": "ok", "canonical_ast": self.PARENT, "parent_ids": ["p"]}
        ast_of = {"p": self.PARENT}.get
        shadow = motif_gate([hit, ok], ast_of, ["swap_window@0:5->10"], mode="shadow")
        self.assertEqual([p["candidate_id"] for p in shadow["kept"]], ["hit", "ok"])
        self.assertEqual((shadow["checked"], shadow["hit_count"]), (2, 1))
        enforce = motif_gate([hit, ok], ast_of, ["swap_window@0:5->10"], mode="enforce")
        self.assertEqual([p["candidate_id"] for p in enforce["kept"]], ["ok"])
        self.assertEqual([h["candidate_id"] for h in enforce["hits"]], ["hit"])

    def test_observations_from_feedback_cards(self):
        window_child = {"op": "rank", "args": [{"op": "return", "args": [{"field": "close"}], "window": 10}]}
        card = {"candidate": {"canonical_ast": window_child},
                "parent_comparisons": [{"parent_id": "p", "status": "valid", "mean_ic_increment": .01},
                                       {"parent_id": "p", "status": "data_insufficient", "mean_ic_increment": None}]}
        observations = motif_observations([card], {"p": self.PARENT}.get)
        self.assertEqual(observations, [("swap_window@0:5->10", True)])
        self.assertEqual(motif_observations([card], {}.get), [])


class CampaignIntegrationTests(unittest.TestCase):
    def _run(self, root, *, env=None, rounds=2, candidates_per_round=1, protocol=None,
             pause_after_round=None, campaign_id=None):
        payload = synthetic_campaign(130, 4)
        source = Path(root)/"input.json"
        if not source.exists():
            source.write_bytes(canonical(payload))
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUXI_V1_")}
        clean.update(env or {})
        with patch.dict(os.environ, clean, clear=True):
            return run_campaign(source, Path(root)/"runs", rounds=rounds,
                                candidates_per_round=candidates_per_round, model_algorithms=(),
                                study_protocol=protocol, pause_after_round=pause_after_round,
                                campaign_id=campaign_id)

    def test_default_run_records_no_new_facts_and_config_defaults(self):
        with scratch_directory() as directory:
            _, report = self._run(directory)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(_campaign_facts(directory, report, "parent_policy_shadow"), [])
            self.assertEqual(_campaign_facts(directory, report, "motif_gate"), [])
            config = _campaign_config(directory, report)
            self.assertEqual(config["parent_policy"], "pareto")
            self.assertEqual(config["motif_memory"], "off")

    def test_shadow_parent_policy_records_without_replacing_selection(self):
        with scratch_directory() as directory:
            _, report = self._run(directory, env={"FUXI_V1_PARENT_POLICY": "niche_mix_shadow"})
            self.assertEqual(report["status"], "completed")
            facts = _campaign_facts(directory, report, "parent_policy_shadow")
            self.assertEqual(len(facts), 2)
            for row in facts:
                self.assertEqual(row["actual_policy"], "pareto")
                self.assertIn("niche_mix_adds_over_pareto", row)
                self.assertIn("pareto_frontier_count", row)
            self.assertEqual(_campaign_config(directory, report)["parent_policy"], "niche_mix_shadow")

    def test_motif_shadow_records_gate_on_parent_derived_rounds(self):
        protocol = {"schema_version": "fuxi.controlled_study.1", "arm_schedule": ["fixed", "refine"],
                    "max_trials": 2, "checkpoint_every": 2, "memory_sources": []}
        with scratch_directory() as directory:
            _, report = self._run(directory, env={"FUXI_V1_MOTIF_MEMORY": "shadow"}, protocol=protocol)
            self.assertEqual(report["status"], "completed")
            facts = _campaign_facts(directory, report, "motif_gate")
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0]["arm"], "refine")
            self.assertEqual(facts[0]["mode"], "shadow")
            self.assertEqual(_campaign_config(directory, report)["motif_memory"], "shadow")

    def test_random_ast_arm_runs_through_campaign(self):
        protocol = {"schema_version": "fuxi.controlled_study.1", "arm_schedule": ["random_ast"],
                    "max_trials": 1, "checkpoint_every": 1, "memory_sources": []}
        with scratch_directory() as directory:
            _, report = self._run(directory, rounds=1, protocol=protocol)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["attempted_candidates"], 1)
            self.assertEqual(report["candidate_results"][0]["actual_action"], "calibration_random_null")

    def test_resume_with_changed_parent_policy_is_rejected(self):
        with scratch_directory() as directory:
            _, partial = self._run(directory, env={"FUXI_V1_PARENT_POLICY": "niche_mix_shadow"},
                                   pause_after_round=1, campaign_id="campaign_evolution_resume")
            self.assertEqual(partial["status"], "checkpoint_paused")
            with self.assertRaisesRegex(ContractError, "parent policy changed"):
                self._run(directory, env={"FUXI_V1_PARENT_POLICY": "pareto"}, rounds=1,
                          pause_after_round=1, campaign_id="campaign_evolution_resume")


if __name__ == "__main__":
    unittest.main()
