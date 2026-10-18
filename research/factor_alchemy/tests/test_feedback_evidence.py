"""Tests for the feedback evidence gates.

Every case here pins a way the consumer could silently treat an unevaluated record as a
verdict, which is the failure mode this module exists to stop.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/factor_alchemy'))

from feedback_evidence import (  # noqa: E402
    ALPHA, DEFAULT_MIN_RULE_SAMPLES, FRI_EVALUATED, FRI_UNEVALUATED,
    all_success_lower_bound, binomial_upper_bound, classify_motif_rules, fri_evidence_status,
    fri_score_for_consumption, gate_enabled, min_samples_for_bound,
    parse_rule_evidence, summarise_fri_evidence, zero_failure_upper_bound,
)


class BoundTests(unittest.TestCase):

    def test_zero_failure_bound_matches_the_definition(self):
        self.assertAlmostEqual(zero_failure_upper_bound(0), 1.0)
        self.assertAlmostEqual(zero_failure_upper_bound(1),
                               1 - ALPHA, places=12)
        self.assertAlmostEqual(zero_failure_upper_bound(3), 1 - 0.05 ** (1 / 3), places=12)
        self.assertGreater(zero_failure_upper_bound(3), 0.6)

    def test_bound_shrinks_monotonically_with_the_sample(self):
        bounds = [zero_failure_upper_bound(n) for n in range(1, 40)]
        self.assertTrue(all(a > b for a, b in zip(bounds, bounds[1:])))

    def test_eleven_samples_are_needed_to_exclude_a_quarter(self):
        """The default bar is derived, so it must come out of the formula itself."""
        self.assertEqual(min_samples_for_bound(0.25), 11)
        self.assertEqual(min_samples_for_bound(0.25), DEFAULT_MIN_RULE_SAMPLES)
        self.assertLessEqual(zero_failure_upper_bound(11), 0.25)
        self.assertGreater(zero_failure_upper_bound(10), 0.25)

    def test_twenty_nine_samples_are_needed_to_exclude_a_tenth(self):
        self.assertEqual(min_samples_for_bound(0.10), 29)

    def test_a_rate_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.1, 1.2):
            with self.assertRaises(ValueError):
                min_samples_for_bound(bad)

    def test_all_success_lower_bound_matches_the_definition(self):
        self.assertAlmostEqual(all_success_lower_bound(0), 0.0)
        self.assertAlmostEqual(all_success_lower_bound(3), 0.05 ** (1 / 3), places=12)
        self.assertLess(all_success_lower_bound(3), 0.4)


class ParseTests(unittest.TestCase):

    def test_reason_with_n_and_rate_is_read(self):
        self.assertEqual(parse_rule_evidence({"reason": "JQ成功率0% (n=3)"}), (3, 0.0))
        self.assertAlmostEqual(parse_rule_evidence({"reason": "JQ成功率67% (n=9)"})[1], 0.67)

    def test_a_rule_without_a_sample_size_is_unreadable(self):
        """A claim with no stated n cannot be checked, so it must not count as evidence."""
        self.assertIsNone(parse_rule_evidence({"reason": "JQ验证失败率过高"}))
        self.assertIsNone(parse_rule_evidence({"reason": "success rate 0%"}))
        self.assertIsNone(parse_rule_evidence({}))

    def test_a_zero_sample_is_refused(self):
        self.assertIsNone(parse_rule_evidence({"reason": "JQ成功率0% (n=0)"}))


class ClassifyTests(unittest.TestCase):

    @staticmethod
    def rule(key, kind, reason):
        return {"motif_key": key, "rule_type": kind, "reason": reason}

    def test_a_three_sample_ban_is_demoted_to_soft(self):
        """0-of-3 is consistent with a 63% true rate, so it cannot be a hard constraint."""
        rules = [self.rule("rolling|mean", "forbid", "JQ成功率0% (n=5)")]
        out = classify_motif_rules(rules)
        self.assertEqual(len(out["hard"]), 0)
        self.assertEqual(len(out["soft"]), 1)
        self.assertEqual(out["soft"][0]["samples"], 5)

    def test_a_twenty_sample_nonzero_success_ban_is_soft(self):
        rules = [self.rule("rolling|std|rolling|mean", "forbid", "JQ成功率10% (n=20)")]
        out = classify_motif_rules(rules)
        self.assertEqual(len(out["hard"]), 0)
        self.assertEqual(out["soft"][0]["observed_successes"], 2)
        self.assertAlmostEqual(out["soft"][0]["binomial_upper_bound"], 0.282618, places=5)

    def test_actual_zero_success_can_still_earn_hard(self):
        out = classify_motif_rules([self.rule("x", "forbid", "JQ成功率0% (n=20)")])
        self.assertEqual(len(out["hard"]), 1)
        self.assertAlmostEqual(binomial_upper_bound(0, 20), zero_failure_upper_bound(20))

    def test_inconsistent_rounded_rate_cannot_be_hard(self):
        out = classify_motif_rules([self.rule("x", "forbid", "JQ成功率1% (n=20)")])
        self.assertEqual(len(out["hard"]), 0)
        self.assertFalse(out["soft"][0]["success_count_consistent"])

    def test_a_key_both_forbidden_and_preferred_is_removed_from_both(self):
        """The memory layer asserting two opposite things is a rule-generator defect."""
        rules = [self.rule("rolling|mean", "forbid", "JQ成功率0% (n=5)"),
                 self.rule("rolling|mean", "prefer", "JQ成功率100% (n=30)")]
        out = classify_motif_rules(rules)
        self.assertEqual(len(out["hard"]), 0)
        self.assertEqual(len(out["soft"]), 0)
        self.assertEqual(len(out["contradictory"]), 2)
        self.assertEqual(out["stats"]["contradictory_keys"], ["rolling|mean"])

    def test_hard_rules_are_ordered_strongest_first(self):
        """A truncating consumer must drop the weakest claim, not the oldest one."""
        rules = [self.rule("a", "forbid", "JQ成功率0% (n=11)"),
                 self.rule("b", "forbid", "JQ成功率0% (n=40)"),
                 self.rule("c", "forbid", "JQ成功率0% (n=25)")]
        out = classify_motif_rules(rules)
        self.assertEqual([r["motif_key"] for r in out["hard"]], ["b", "c", "a"])

    def test_unreadable_rules_are_reported_not_promoted(self):
        rules = [self.rule("x", "forbid", "no sample here")]
        out = classify_motif_rules(rules)
        self.assertEqual(len(out["hard"]), 0)
        self.assertEqual(len(out["soft"]), 0)
        self.assertEqual(len(out["unreadable"]), 1)

    def test_a_preference_also_needs_its_sample(self):
        """Symmetry: a 100% claim from six observations is as weak as a 0% one."""
        rules = [self.rule("rough::x", "prefer", "JQ成功率100% (n=6)")]
        out = classify_motif_rules(rules)
        self.assertEqual(len(out["hard"]), 0)
        self.assertEqual(len(out["soft"]), 1)

    def test_unknown_rule_type_is_ignored(self):
        rules = [{"motif_key": "z", "rule_type": "something", "reason": "JQ成功率0% (n=99)"}]
        out = classify_motif_rules(rules)
        self.assertEqual(out["stats"]["rules_total"], 1)
        self.assertEqual(len(out["hard"]) + len(out["soft"]), 0)

    def test_the_gate_switch_reads_its_environment(self):
        original = os.environ.get("FUXI_MOTIF_EVIDENCE_GATE")
        try:
            os.environ["FUXI_MOTIF_EVIDENCE_GATE"] = "shadow"
            self.assertFalse(gate_enabled())
            os.environ["FUXI_MOTIF_EVIDENCE_GATE"] = "enforce"
            self.assertTrue(gate_enabled())
            del os.environ["FUXI_MOTIF_EVIDENCE_GATE"]
            self.assertTrue(gate_enabled())
        finally:
            if original is None:
                os.environ.pop("FUXI_MOTIF_EVIDENCE_GATE", None)
            else:
                os.environ["FUXI_MOTIF_EVIDENCE_GATE"] = original


class FriTests(unittest.TestCase):

    def test_an_early_return_record_is_not_a_score(self):
        """The signature a computed FRI can never produce: all-zero with an empty grade."""
        record = {"score": 0.0, "grade": "", "precision": 0.0, "persistence": 0.0,
                  "consistency": 0.0, "novelty": 0.0}
        self.assertEqual(fri_evidence_status(record), FRI_UNEVALUATED)
        self.assertIsNone(fri_score_for_consumption(record))

    def test_a_computed_zero_grade_d_still_counts_as_evaluated(self):
        record = {"score": 0.0, "grade": "D", "precision": 0.0, "persistence": 0.0,
                  "consistency": 0.0, "novelty": 0.0}
        self.assertEqual(fri_evidence_status(record), FRI_EVALUATED)
        self.assertAlmostEqual(fri_score_for_consumption(record), 0.0)

    def test_a_partial_record_is_evaluated(self):
        record = {"score": 0.28, "grade": "", "precision": 0.1, "persistence": 0.0,
                  "consistency": 0.0, "novelty": 0.0}
        self.assertEqual(fri_evidence_status(record), FRI_EVALUATED)

    def test_a_missing_record_is_unevaluated(self):
        self.assertEqual(fri_evidence_status(None), FRI_UNEVALUATED)
        self.assertEqual(fri_evidence_status({}), FRI_UNEVALUATED)

    def test_an_explicit_status_overrides_the_signature(self):
        """New records carry the status directly, so the fallback is not consulted."""
        computed = {"status": "computed", "score": 0.0, "grade": "",
                    "precision": 0.0, "persistence": 0.0, "consistency": 0.0, "novelty": 0.0}
        self.assertEqual(fri_evidence_status(computed), FRI_EVALUATED)
        self.assertAlmostEqual(fri_score_for_consumption(computed), 0.0)
        starved = {"status": "insufficient_ic_points", "score": 0.0, "grade": "D"}
        self.assertEqual(fri_evidence_status(starved), FRI_UNEVALUATED)
        self.assertIsNone(fri_score_for_consumption(starved))

    def test_summary_counts_the_two_populations_apart(self):
        attempts = [
            {"paradigm": "p", "fri": {"score": 0.0, "grade": "", "precision": 0.0,
                                      "persistence": 0.0, "consistency": 0.0, "novelty": 0.0}},
            {"paradigm": "p", "fri": {"score": 0.65, "grade": "A", "precision": 0.7,
                                      "persistence": 0.6, "consistency": 0.6, "novelty": 0.7}},
        ]
        out = summarise_fri_evidence(attempts)
        self.assertEqual(out["attempts_total"], 2)
        self.assertEqual(out["fri_evaluated"], 1)
        self.assertEqual(out["fri_unevaluated"], 1)
        self.assertEqual(out["evaluated_by_paradigm"], {"p": 1})


class LiveDataTests(unittest.TestCase):
    """The counts the report quotes must be reproducible from the stored files."""

    @classmethod
    def setUpClass(cls):
        import json
        path = ROOT / 'data/experience_memory.json'
        if not path.exists():
            raise unittest.SkipTest('experience memory not present')
        with open(path, encoding='utf-8') as handle:
            cls.memory = json.load(handle)

    def test_the_live_rules_are_mostly_soft(self):
        out = classify_motif_rules(self.memory['motif_rules'])
        stats = out["stats"]
        self.assertGreater(stats["rules_total"], 0)
        self.assertGreater(stats["soft"], 0)
        self.assertEqual(stats["hard"] + stats["soft"] + stats["contradictory"]
                         + stats["unreadable"], stats["rules_total"])
        self.assertTrue(stats["contradictory_keys"])

    def test_the_live_attempts_are_mostly_unevaluated(self):
        out = summarise_fri_evidence(self.memory['attempts'])
        self.assertEqual(out["attempts_total"] + 0, len(self.memory['attempts']))
        self.assertGreater(out["fri_unevaluated"], out["fri_evaluated"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
