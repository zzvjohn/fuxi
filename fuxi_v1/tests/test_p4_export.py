import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.export import (EXPORT_BUNDLE_SCHEMA, delivery_strategy_ir,
                            freeze_export, forward_record, is_native_ir,
                            jq_receipt, validate_strategy_ir, validate_transition)


def ir(**overrides):
    base = {"schema_version": "fuxi.strategy_ir.1", "candidate_ids": ["c1"],
            "train_scope": ["2024-01-01", "2025-06-30"], "model_artifacts": [],
            "prediction_inputs": ["score_c1"], "rebalance_calendar": "monthly_first",
            "ranking_and_tie_rules": "rank_desc_stable", "weights": {"c1": 1.0},
            "holdings_count": 80, "constraints": [], "turnover_buffer": 0.1,
            "capital": 10_000_000.0, "execution_contract": {"venue": "jq"},
            "risk_conditions": [{"kind": "max_drawdown_guard", "value": 0.25}]}
    base.update(overrides)
    return base


def bundle():
    return freeze_export(ir(), code_ref="1" * 64, config_ref="2" * 64,
                         input_contract_ref="3" * 64)


class DeliveryBridgeTests(unittest.TestCase):
    """v0.7.188: the native execution StrategyIR (portfolio.compile_strategy)
    bridges onto the R2 delivery contract at the export boundary. The native
    shape shares fuxi.strategy_ir.1 but carries strategy/decisions instead of
    the delivery field set; the bridge is the single authorised translation."""

    @staticmethod
    def native_ir(**spec_overrides):
        from fuxi_v1.portfolio import strategy_spec
        spec = strategy_spec("c42", top_n=8, rebalance_sessions=5)
        spec.update(spec_overrides)
        return {"schema_version": "fuxi.strategy_ir.1", "strategy": spec,
                "snapshot_hash": "s" * 64, "execution_hash": "e" * 64,
                "model_hash": None, "score_origin": "factor",
                "decisions": [{"decision_time": "2024-01-02T15:00:00+08:00",
                               "signal_session": "2024-01-02",
                               "execution_session": "2024-01-03",
                               "first_order_time": "2024-01-03T09:30:00+08:00",
                               "scores": {"000001": 1.0}, "universe": ["000001"]}],
                "ir_hash": "i" * 64}

    def test_is_native_ir_detection(self):
        self.assertTrue(is_native_ir(self.native_ir()))
        self.assertFalse(is_native_ir(ir()))

    def test_bridge_maps_all_contract_fields(self):
        delivery = delivery_strategy_ir(self.native_ir(),
                                        train_scope=["2024-01-02", "2025-06-30"],
                                        capital=1_000_000.0)
        validate_strategy_ir(delivery)  # must pass the P4 contract as-is
        self.assertEqual(delivery["candidate_ids"], ["c42"])
        self.assertEqual(delivery["train_scope"], ["2024-01-02", "2025-06-30"])
        self.assertEqual(delivery["model_artifacts"], [])
        self.assertEqual(delivery["prediction_inputs"], ["score_c42"])
        self.assertEqual(delivery["rebalance_calendar"],
                         {"type": "session_interval", "sessions": 5})
        self.assertEqual(delivery["ranking_and_tie_rules"], "equal_rank_ties_prorata")
        self.assertEqual(delivery["weights"], {"c42": 1.0})
        self.assertEqual(delivery["holdings_count"], 8)
        self.assertEqual(delivery["turnover_buffer"], 1.0)
        self.assertEqual(delivery["capital"], 1_000_000.0)
        self.assertEqual(delivery["execution_contract"]["venue"], "jq")
        self.assertEqual(delivery["execution_contract"]["execution_rule"],
                         "next_session_open")
        self.assertEqual(delivery["risk_conditions"], [])
        self.assertEqual(delivery["native_ir_ref"], "i" * 64)

    def test_model_artifacts_and_constraints_carry_over(self):
        native = self.native_ir()
        native["model_hash"] = "m" * 64
        delivery = delivery_strategy_ir(native,
                                        train_scope=["2024-01-02", "2025-06-30"],
                                        capital=1)
        self.assertEqual(delivery["model_artifacts"], ["m" * 64])
        kinds = {c["kind"]: c["value"] for c in delivery["constraints"]}
        self.assertEqual(kinds, {"max_weight": 0.2, "gross_exposure": 1.0,
                                 "industry_cap": 1.0})

    def test_missing_context_refused_no_fabrication(self):
        with self.assertRaises(ContractError):
            delivery_strategy_ir(self.native_ir(), train_scope=None, capital=1)
        with self.assertRaises(ContractError):
            delivery_strategy_ir(self.native_ir(),
                                 train_scope=["2024-01-02", "2025-06-30"],
                                 capital=None)
        with self.assertRaises(ContractError):
            delivery_strategy_ir(ir(),  # delivery shape is not native
                                 train_scope=["2024-01-02", "2025-06-30"],
                                 capital=1)

    def test_bridged_ir_freezes_deterministically(self):
        a = freeze_export(delivery_strategy_ir(
                              self.native_ir(),
                              train_scope=["2024-01-02", "2025-06-30"],
                              capital=1_000_000.0),
                          code_ref="1" * 64, config_ref="2" * 64,
                          input_contract_ref="3" * 64)
        b = freeze_export(delivery_strategy_ir(
                              self.native_ir(),
                              train_scope=["2024-01-02", "2025-06-30"],
                              capital=1_000_000.0),
                          code_ref="1" * 64, config_ref="2" * 64,
                          input_contract_ref="3" * 64)
        self.assertEqual(a["frozen_digest"], b["frozen_digest"])
        self.assertIsNotNone(a["frozen_digest"])


class StrategyIrTests(unittest.TestCase):
    def test_valid(self):
        validate_strategy_ir(ir())

    def test_missing_execution_contract_rejected(self):
        bad = ir(); del bad["execution_contract"]
        with self.assertRaises(ContractError):
            validate_strategy_ir(bad)

    def test_freeze_deterministic(self):
        a, b = bundle(), bundle()
        self.assertEqual(a["frozen_digest"], b["frozen_digest"])
        self.assertEqual(a["schema_version"], EXPORT_BUNDLE_SCHEMA)


class JQReceiptTests(unittest.TestCase):
    def test_real_receipt(self):
        receipt = jq_receipt(export_bundle=bundle(), platform="joinquant",
                             platform_run_id="backtest_12345",
                             per_field_comparison={"c1": {"platform_value_digest": digest([1.0]),
                                                          "local_value_digest": digest([1.0]),
                                                          "match": True}},
                             verified_by="operator")
        self.assertEqual(receipt["platform_run_id"], "backtest_12345")

    def test_file_existence_is_not_a_receipt(self):
        # No platform run id / no per-field comparison -> refused, regardless of
        # any bundle file sitting on disk.
        with self.assertRaises(ContractError):
            jq_receipt(export_bundle=bundle(), platform="joinquant", platform_run_id="",
                       per_field_comparison={}, verified_by="operator")
        with self.assertRaises(ContractError):
            jq_receipt(export_bundle=bundle(), platform="joinquant",
                       platform_run_id="backtest_1",
                       per_field_comparison={"c1": {"platform_value_digest": digest([1.0]),
                                                    "local_value_digest": digest([2.0]),
                                                    "match": False}},
                       verified_by="operator")


class ForwardRecordTests(unittest.TestCase):
    def test_observation_is_not_decision(self):
        record = forward_record(strategy_ref="a" * 64, frozen_version="v1",
                                look_plan=[1, 2, 3], observed_sessions=60)
        self.assertIsNone(record["met_target"])
        self.assertEqual(record["observed_sessions"], 60)
        decided = forward_record(strategy_ref="a" * 64, frozen_version="v1",
                                 look_plan=[1, 2, 3], observed_sessions=120,
                                 met_target=True)
        self.assertTrue(decided["met_target"])

    def test_negative_sessions_rejected(self):
        with self.assertRaises(ContractError):
            forward_record(strategy_ref="a" * 64, frozen_version="v1", look_plan=[1],
                           observed_sessions=-1)


class ReleaseTransitionTests(unittest.TestCase):
    def test_research_selected_requires_receipt(self):
        with self.assertRaises(ContractError):
            validate_transition("research_selected", "jq_verified", {})
        ok = validate_transition("research_selected", "jq_verified",
                                 {"receipt": {"platform_run_id": "x"}})
        self.assertEqual(ok["evidence"], ["receipt"])

    def test_jq_verified_requires_frozen_export(self):
        with self.assertRaises(ContractError):
            validate_transition("jq_verified", "forward_pending", {})
        validate_transition("jq_verified", "forward_pending",
                            {"frozen_export": bundle()})

    def test_forward_requires_observation(self):
        with self.assertRaises(ContractError):
            validate_transition("forward_pending", "forward_observed", {})
        validate_transition("forward_pending", "forward_observed",
                            {"observation": {"sessions": 60}})

    def test_illegal_skip_rejected(self):
        # research_selected may not jump straight to forward_pending
        with self.assertRaises(ContractError):
            validate_transition("research_selected", "forward_pending", {})

    def test_legal_chain(self):
        validate_transition("proposed", "screened", {})
        validate_transition("screened", "portfolio_candidate", {})
        validate_transition("portfolio_candidate", "research_selected", {})


if __name__ == "__main__":
    unittest.main()
