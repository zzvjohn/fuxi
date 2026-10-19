"""Behavioral checks for the shared system-change gate."""

import json
from pathlib import Path
import tempfile
import unittest

from fuxi_v1.contracts import ContractError, digest
from research.system_evolution.gate import assess, freeze, history, inspect, _file_hash


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


class SharedSystemChangeGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fuxi_system_gate_", dir=ROOT)
        self.root = Path(self.tmp.name).resolve()
        self.assertTrue(self.root.is_relative_to(ROOT.resolve()))
        write(self.root / "codex_automations/fuxi_runtime_policy.json", {
            "single_writer_confirmed": True, "current_mode": "controlled_manual",
            "autonomous_writes_enabled": False})
        (self.root / "data").mkdir()
        for name, value in (("base", "base"), ("current", "current"), ("next", "next")):
            (self.root / (name + ".py")).write_text(value, encoding="utf-8")
        for name in ("judge", "data", "labels", "cost"):
            write(self.root / (name + ".json"), {"source": name, "version": "frozen"})
        tasks = [{"engine": engine, "task_id": cohort, "cohort": cohort,
                  "support_hash": engine + ":" + cohort, "window": "2024-01..2024-09"}
                 for engine in ("legacy_v07", "native_v10")
                 for cohort in ("replay_exposed", "transfer_exposed")]
        self.suite = {"schema_version": "fuxi.system_evolution.1.suite", "tasks": tasks}
        write(self.root / "suite.json", self.suite)
        write(self.root / "noise.json", {"schema_version": "fuxi.system_evolution.1.aa", "pairs": [
            {"engine": "legacy_v07", "task_id": "replay_exposed", "repeat_id": "s1",
             "first": .50, "second": .502},
            {"engine": "native_v10", "task_id": "transfer_exposed", "repeat_id": "s2",
             "first": .50, "second": .503}]})
        self.plan = {"schema_version": "fuxi.system_evolution.1", "change_id": "proposal_A",
                     "title": "shared method change", "kind": "research_method",
                     "engines": ["legacy_v07", "native_v10"], "component": "memory_retrieval",
                     "hypothesis": "same task improves", "falsification": "no transfer gain",
                     "assets": {"baseline_code": "base.py", "incumbent_code": "current.py",
                                "challenger_code": "next.py", "suite": "suite.json",
                                "judge": "judge.json", "data": "data.json",
                                "labels": "labels.json", "cost": "cost.json", "aa_noise": "noise.json"},
                     "thresholds": {"noise_floor": .003, "minimum_mean_gain": .01,
                                    "maximum_case_loss": .10,
                                    "quality_noninferiority_margin": .02, "minimum_wall_saving": 1},
                     "budgets": {"wall_seconds": 30, "memory_mb": 100,
                                 "external_requests": 0, "fee_cny": 0},
                     "required_cohorts": ["transfer_exposed"]}
        write(self.root / "plan.json", self.plan)

    def tearDown(self):
        self.tmp.cleanup()

    def observe(self, role, score, *, state="valid", support=None, actual_wall=10):
        facts = write(self.root / "evidence" / (role + ".json"),
                      {"engine_results": [{"task": task["task_id"], "engine": task["engine"],
                                           "score": score} for task in self.suite["tasks"]]})
        rows = [{"engine": task["engine"], "task_id": task["task_id"],
                 "support_hash": support or task["support_hash"], "eval_status": state,
                 "score": score if state == "valid" else None,
                 "usage": {"wall_seconds": actual_wall, "memory_mb": 12,
                           "external_requests": 0, "fee_cny": 0},
                 "source_fact": {"path": str(facts), "sha256": _file_hash(facts)}}
                for task in self.suite["tasks"]]
        document = {"schema_version": "fuxi.system_evolution.1.observations", "role": role,
                    "suite_hash": digest(self.suite),
                    **{name + "_sha256": _file_hash(self.root / (name + ".json"))
                       for name in ("judge", "data", "labels", "cost")},
                    "cases": rows}
        return write(self.root / (role + ".json"), document)

    def complete(self, *, challenger_score=.56, challenger_state="valid", challenger_wall=10):
        anchor = self.observe("anchor", .5)
        incumbent = self.observe("incumbent", .5)
        challenger = self.observe("challenger", challenger_score,
                                  state=challenger_state, actual_wall=challenger_wall)
        return assess(self.root, "proposal_A", anchor, incumbent, challenger)

    def test_both_engines_same_gate_and_no_qualification_transfer(self):
        frozen = freeze(self.root, "plan.json")
        self.assertEqual(frozen["task_count"], 4)
        self.assertAlmostEqual(frozen["aa_max_absolute_delta"], .003)
        receipt = self.complete()
        self.assertEqual(receipt["eval_status"], "valid")
        self.assertTrue(receipt["review_eligible"])
        self.assertEqual(set(receipt["effects"]["groups"]),
                         {"legacy_v07:transfer_exposed", "native_v10:transfer_exposed"})
        self.assertFalse(receipt["production_activation_allowed"])
        self.assertFalse(receipt["factor_qualification_transfer_allowed"])
        self.assertFalse(receipt["economic_winner_confirmed"])
        self.assertEqual(receipt["decision"], "review_shadow")
        self.assertEqual(inspect(self.root, "proposal_A"), receipt)
        self.assertEqual(history(self.root), [receipt])
        self.assertEqual(self.complete(), receipt)

    def test_missing_or_failed_pair_never_becomes_zero_economic_reward(self):
        freeze(self.root, "plan.json")
        receipt = self.complete(challenger_state="data_insufficient")
        self.assertEqual(receipt["eval_status"], "data_insufficient")
        self.assertFalse(receipt["review_eligible"])
        self.assertEqual(receipt["decision"], "await_evidence")

    def test_efficiency_requires_measured_savings_on_same_tasks(self):
        self.plan["kind"] = "efficiency"
        write(self.root / "plan.json", self.plan)
        freeze(self.root, "plan.json")
        receipt = self.complete(challenger_score=.5, challenger_wall=8)
        self.assertEqual(receipt["decision"], "review_shadow")
        self.assertEqual(receipt["effects"]["wall_seconds_saved"], 8)

    def test_declared_forward_observations_still_cannot_confirm_economic_winner(self):
        self.plan["kind"] = "economic_logic"
        self.plan["required_cohorts"] = ["declared_forward"]
        self.suite["tasks"] = [{**task, "cohort": "declared_forward"}
                               if task["cohort"] == "transfer_exposed" else task
                               for task in self.suite["tasks"]]
        write(self.root / "suite.json", self.suite)
        write(self.root / "plan.json", self.plan)
        freeze(self.root, "plan.json")
        receipt = self.complete()
        self.assertTrue(receipt["review_eligible"])
        self.assertFalse(receipt["economic_winner_confirmed"])
        self.assertFalse(receipt["production_activation_allowed"])

    def test_exposed_research_only_cannot_claim_economic_confirmation(self):
        self.plan["kind"] = "economic_logic"
        write(self.root / "plan.json", self.plan)
        with self.assertRaisesRegex(ContractError, "forward"):
            freeze(self.root, "plan.json")

    def test_noise_tolerance_cannot_be_inflated(self):
        self.plan["thresholds"]["noise_floor"] = .5
        write(self.root / "plan.json", self.plan)
        with self.assertRaisesRegex(ContractError, "noise floor"):
            freeze(self.root, "plan.json")

    def test_resource_overrun_and_material_regression(self):
        freeze(self.root, "plan.json")
        first = self.complete(challenger_wall=35)
        self.assertEqual((first["eval_status"], first["reason_code"]),
                         ("precheck_rejected", "budget_exceeded_in_challenger"))
        self.assertFalse(first["review_eligible"])
        self.assertEqual(first["decision"], "await_evidence")

    def test_material_case_regression_retains_incumbent(self):
        freeze(self.root, "plan.json")
        receipt = self.complete(challenger_score=.30)
        self.assertEqual(receipt["reason_code"], "material_case_regression")
        self.assertFalse(receipt["review_eligible"])
        self.assertEqual(receipt["decision"], "retain_incumbent")

    def test_code_source_drift_rejected_before_comparison(self):
        freeze(self.root, "plan.json")
        (self.root / "next.py").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "challenger_code source drift"):
            self.complete()

    def test_prior_assessment_survives_subsequent_live_code_revision(self):
        freeze(self.root, "plan.json")
        receipt = self.complete()
        (self.root / "next.py").write_text("later-version", encoding="utf-8")
        self.assertEqual(inspect(self.root, "proposal_A"), receipt)
        self.assertEqual(history(self.root), [receipt])

    def test_sealed_source_tamper_is_detected(self):
        frozen = freeze(self.root, "plan.json")
        Path(frozen["assets"]["judge"]["sealed_path"]).write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "sealed source drift"):
            inspect(self.root, "proposal_A")

    def test_underlying_fact_drift_invalidates_a_published_assessment(self):
        freeze(self.root, "plan.json")
        self.complete()
        write(self.root / "evidence/challenger.json", {"changed": True})
        with self.assertRaisesRegex(ContractError, "fact source drift"):
            inspect(self.root, "proposal_A")

    def test_empty_or_changed_support_cannot_be_compared(self):
        freeze(self.root, "plan.json")
        a = self.observe("anchor", .5)
        b = self.observe("incumbent", .5)
        c = self.observe("challenger", .56, support="wrong")
        with self.assertRaisesRegex(ContractError, "task/support"):
            assess(self.root, "proposal_A", a, b, c)

    def test_repeated_candidate_change_cannot_rewrite_prior_receipt(self):
        freeze(self.root, "plan.json")
        original = self.complete()
        self.observe("challenger", .51)
        with self.assertRaisesRegex(ContractError, "immutable system-evolution artifact changed"):
            self.complete(challenger_score=.51)
        with self.assertRaisesRegex(ContractError, "assessment source drift"):
            inspect(self.root, "proposal_A")
        self.assertEqual(original["reason_code"], "eligible_for_independent_review")


if __name__ == "__main__":
    unittest.main()
