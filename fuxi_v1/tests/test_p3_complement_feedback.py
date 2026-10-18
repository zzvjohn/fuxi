import unittest
from fuxi_v1.contracts import ContractError
from fuxi_v1.complement import (FoldIsolatedFeedback, channel_ablation_plan,
                                prune_multi_channel, record_feedback, task_role_route)


class TaskRoleRouteTests(unittest.TestCase):
    def test_weak_ic_standalone_parks(self):
        g1 = {"rank_ic_mean": 0.004, "icir_unannualized": 0.05}
        route, reason = task_role_route(role="standalone", g1=g1)
        self.assertEqual((route, reason), ("park", "weak_ic_routing_hint"))

    def test_weak_ic_complement_routes(self):
        # Marginal IC is tolerated on the complement channel
        g1 = {"rank_ic_mean": 0.004, "icir_unannualized": 0.05}
        route, reason = task_role_route(role="complement", g1=g1)
        self.assertEqual((route, reason), ("shortlist", "complement_channel"))

    def test_near_neighbour_is_duplicate(self):
        g1 = {"rank_ic_mean": 0.02, "icir_unannualized": 0.3}
        route, reason = task_role_route(role="complement", g1=g1, similarity=0.97)
        self.assertEqual((route, reason), ("duplicate", "same_support_neighbour"))

    def test_similarity_only_on_same_support(self):
        # similarity=None (unsupported comparison) must never trigger duplicate
        g1 = {"rank_ic_mean": 0.02, "icir_unannualized": 0.3}
        route, _ = task_role_route(role="complement", g1=g1, similarity=None)
        self.assertEqual(route, "shortlist")

    def test_risk_task_uses_risk_objective(self):
        g1 = {"rank_ic_mean": -0.01, "icir_unannualized": -0.2}
        route, reason = task_role_route(role="risk_condition", g1=g1,
                                        risk_metric={"improved": True,
                                                     "loss_within_budget": True})
        self.assertEqual((route, reason), ("shortlist", "risk_objective_met"))
        route2, reason2 = task_role_route(role="risk_condition", g1=g1,
                                          risk_metric={"improved": False,
                                                       "loss_within_budget": True})
        self.assertEqual((route2, reason2), ("park", "risk_objective_not_met"))

    def test_risk_task_needs_metric(self):
        with self.assertRaises(ContractError):
            task_role_route(role="risk_condition", g1={"rank_ic_mean": 0.01})

    def test_standalone_signal_not_diagnosable(self):
        g1 = {"rank_ic_mean": None, "icir_unannualized": None}
        route, reason = task_role_route(role="standalone", g1=g1)
        self.assertEqual((route, reason), ("park", "signal_not_diagnosable"))


class ChannelPruningTests(unittest.TestCase):
    def test_ablation_plan_two_channels(self):
        plan = channel_ablation_plan(["P", "V"])
        self.assertEqual(plan, [("baseline", "P"), ("baseline", "V"), ("baseline", "P", "V")])

    def test_ablation_plan_interaction_claim(self):
        plan = channel_ablation_plan(["P", "V"], interaction_claim=True)
        self.assertEqual(plan[-1], ("baseline", "interaction", "P", "V"))

    def test_single_channel_plan(self):
        self.assertEqual(channel_ablation_plan(["P"]), [("baseline", "P")])

    def test_prune_keeps_best_on_target(self):
        comparisons = {("baseline", "P"): {"net_delta_bps": 1.2},
                       ("baseline", "V"): {"net_delta_bps": 0.6},
                       ("baseline", "P", "V"): {"net_delta_bps": 0.9}}
        kept, reason = prune_multi_channel(comparisons, "net_delta_bps")
        self.assertEqual((kept, reason), (("baseline", "P"), "best_on_target"))

    def test_prune_complex_not_improving(self):
        # Complex form does not beat the simpler representative -> simple wins
        comparisons = {("baseline", "P"): {"net_delta_bps": 2.0},
                       ("baseline", "P", "V"): {"net_delta_bps": 1.1}}
        kept, _ = prune_multi_channel(comparisons, "net_delta_bps")
        self.assertEqual(kept, ("baseline", "P"))


class FeedbackTests(unittest.TestCase):
    BUNDLE = {"decision": "research_shortlist", "metrics": {"delta_bps": 1.0}}

    def test_event_records_actual_lineage(self):
        event = record_feedback(generation_action_id="a1", candidate_id="c1",
                                hypothesis_id="h1", parent_ids=["p1", "p2"],
                                source_refs=["s1"], evaluation_bundle=self.BUNDLE,
                                fold_segment=["2024-01-01", "2024-12-31"],
                                objective_kind="account", outcome_summary="cost_failure")
        self.assertEqual(event["parent_ids"], ["p1", "p2"])
        self.assertEqual(event["objective_kind"], "account")
        self.assertEqual(len(event["evaluation_digest"]), 64)

    def test_fold_isolation(self):
        ledger = FoldIsolatedFeedback()
        f1 = ["2024-01-01", "2024-12-31"]
        f2 = ["2025-01-01", "2025-06-30"]
        ledger.ingest(record_feedback(generation_action_id="a1", candidate_id="c1",
                                      hypothesis_id="h1", parent_ids=["p1"], source_refs=["s1"],
                                      evaluation_bundle=self.BUNDLE, fold_segment=f1,
                                      objective_kind="predictive", outcome_summary="ok"))
        ledger.ingest(record_feedback(generation_action_id="a2", candidate_id="c2",
                                      hypothesis_id="h2", parent_ids=["p2"], source_refs=["s2"],
                                      evaluation_bundle=self.BUNDLE, fold_segment=f2,
                                      objective_kind="account", outcome_summary="ok"))
        self.assertEqual([e["generation_action_id"] for e in ledger.view(f1)], ["a1"])
        self.assertEqual([e["generation_action_id"] for e in ledger.view(f2)], ["a2"])
        self.assertEqual(ledger.lineage_covered(f1), {"p1", "s1"})
        self.assertEqual(ledger.lineage_covered(f2), {"p2", "s2"})

    def test_objectives_recorded_separately(self):
        ledger = FoldIsolatedFeedback()
        fold = ["2024-01-01", "2024-12-31"]
        for kind in ("engineering", "predictive", "account"):
            ledger.ingest(record_feedback(generation_action_id="a_" + kind, candidate_id="c",
                                          hypothesis_id="h", parent_ids=[], source_refs=[],
                                          evaluation_bundle=self.BUNDLE, fold_segment=fold,
                                          objective_kind=kind, outcome_summary="ok"))
        kinds = {e["objective_kind"] for e in ledger.view(fold)}
        self.assertEqual(kinds, {"engineering", "predictive", "account"})

    def test_bad_objective_rejected(self):
        with self.assertRaises(ContractError):
            record_feedback(generation_action_id="a", candidate_id="c", hypothesis_id="h",
                            parent_ids=[], source_refs=[], evaluation_bundle=self.BUNDLE,
                            fold_segment=["2024-01-01", "2024-12-31"],
                            objective_kind="returns", outcome_summary="x")


if __name__ == "__main__":
    unittest.main()
