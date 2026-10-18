import unittest
from fuxi_v1.contracts import ContractError, digest, evaluation
from fuxi_v1.routing import (FIVE_STATES, ROUTE_DECISIONS, route_from_legacy,
                             route_duplicate, validate_screen_result)


def e1_pass():
    return evaluation("E1", "valid", "calibrated_research_statistical_screen",
                      decision="pass", metrics={"rank_ic_mean": 0.02},
                      scope="research", n_pairs=0)


def e1_reject():
    return evaluation("E1", "valid", "calibrated_research_statistical_screen",
                      decision="reject", reject_class="statistical",
                      scope="research", n_pairs=0)


def e2_pass():
    return evaluation("E2", "valid", "channel_ablation_positive", decision="pass",
                      metrics={"channel_delta": 0.01}, scope="research", n_pairs=0)


def e3_economic_reject():
    return evaluation("E3", "valid", "paired_account_delta_negative", decision="reject",
                      reject_class="economic", metrics={"delta_bps": -3.2},
                      scope="account", n_pairs=120)


def e3_not_evaluated():
    return evaluation("E3", "not_evaluated", "account_quota_not_allocated",
                      decision="not_applicable")


def e0_precheck_unsupported():
    return evaluation("E0", "precheck_rejected", "capability_unsupported",
                      decision="not_applicable")


def e0_data_insufficient():
    return evaluation("E0", "data_insufficient", "field_missing_wait_for_data",
                      decision="not_applicable")


class RoutingStateTests(unittest.TestCase):
    def test_case_normal_factor(self):
        r = route_from_legacy("c_normal", [e1_pass()])
        self.assertEqual((r["eval_status"], r["decision"]), ("valid", "shortlist"))
        self.assertIsNone(r["formal_qualification"])

    def test_case_multi_channel(self):
        r = route_from_legacy("c_multi", [e1_pass(), e2_pass()])
        self.assertEqual((r["eval_status"], r["decision"]), ("valid", "shortlist"))
        self.assertEqual(len(r["legacy_refs"]), 2)

    def test_case_sparse_event(self):
        r = route_from_legacy("c_sparse", [e0_data_insufficient()])
        self.assertEqual((r["eval_status"], r["decision"]), ("data_insufficient", "park"))

    def test_case_suspended_account(self):
        # Economic reject with valid measurements: status stays valid, routing
        # parks it. The old reject is preserved verbatim under legacy_refs and
        # must NOT be re-imported as a new rejection decision.
        r = route_from_legacy("c_halt", [e1_pass(), e3_economic_reject()])
        self.assertEqual((r["eval_status"], r["decision"]), ("valid", "park"))
        self.assertIsNone(r["formal_qualification"])
        legacy_decisions = {ref["evaluation"]["decision"] for ref in r["legacy_refs"]}
        self.assertIn("reject", legacy_decisions)
        self.assertEqual(r["reason_code"], None)

    def test_case_unsupported_expression(self):
        r = route_from_legacy("c_unsup", [e0_precheck_unsupported()],
                              capability_status="unsupported")
        self.assertEqual(r["eval_status"], "precheck_rejected")
        self.assertEqual(r["capability_status"], "unsupported")
        # unsupported is a tag, never a sixth state
        self.assertNotIn("unsupported", FIVE_STATES)
        self.assertIn("precheck_rejected", FIVE_STATES)

    def test_case_missing_state(self):
        # No evaluations at all -> not_evaluated
        r = route_from_legacy("c_missing", [])
        self.assertEqual((r["eval_status"], r["decision"]), ("not_evaluated", "park"))
        # Old unevaluated economic stage -> not_evaluated, NOT a rejection
        r2 = route_from_legacy("c_missing2", [e3_not_evaluated()])
        self.assertEqual((r2["eval_status"], r2["decision"]), ("not_evaluated", "park"))
        self.assertIsNone(r2["formal_qualification"])

    def test_old_reject_never_grants_qualification(self):
        r = route_from_legacy("c_rej", [e1_reject()])
        self.assertIsNone(r["formal_qualification"])
        self.assertEqual(r["decision"], "park")

    def test_legacy_objects_preserved_by_reference(self):
        legacy = [e1_pass()]
        r = route_from_legacy("c_ref", legacy)
        self.assertIs(r["legacy_refs"][0]["evaluation"], legacy[0])
        self.assertEqual(r["legacy_refs"][0]["digest"], digest(legacy[0]))

    def test_duplicate_route(self):
        r = route_duplicate("c_dup", "cluster_7", diagnostics={"n_sections": 200})
        self.assertEqual((r["eval_status"], r["decision"]), ("valid", "duplicate"))
        self.assertEqual(r["signal_cluster_id"], "cluster_7")

    def test_validation_guards(self):
        r = route_from_legacy("c_g", [e1_pass()])
        with self.assertRaises(ContractError):
            validate_screen_result({**r, "eval_status": "sixth_state"})
        with self.assertRaises(ContractError):
            validate_screen_result({**r, "formal_qualification": "research_selected"})
        with self.assertRaises(ContractError):
            validate_screen_result({**r, "unknown_key": 1})
        with self.assertRaises(ContractError):
            validate_screen_result({**r, "capability_status": "bogus"})


if __name__ == "__main__":
    unittest.main()
