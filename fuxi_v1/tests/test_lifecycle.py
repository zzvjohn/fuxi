"""Contract fixtures exercise authority state transitions; none is external economic evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
from unittest.mock import patch
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.contracts import ContractError, canonical, digest, evaluation
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.acceptance import AcceptanceService, ReleaseService
from fuxi_v1.execution import PaperGateway, reconcile_rows
from fuxi_v1.campaign import run_campaign, choose_arm
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.data import Snapshot
from fuxi_v1.datasets import scope_of
from fuxi_v1.policy import register_confirmatory_protocol
from fuxi_v1.portfolio import portfolio_experiment, compile_strategy, strategy_spec
from fuxi_v1.dsl import evaluate_ast
from fuxi_v1.search import node
from fuxi_v1.research_evaluation import single_field_channels
from fuxi_v1.jq_strategy import compile_jq_strategy
from fuxi_v1.simulation import run_account


class LifecycleTests(unittest.TestCase):
    def test_positive_forward_paper_rollback_and_data_revocation(self):
        with scratch_directory() as directory:
            store=EvidenceStore(Path(directory)/"authority.sqlite")
            try:
                # Privileged test receipts; synthetic clock and hand-authored observations.
                data=store.receipt("data_contract",{"test_only":True})
                research=store.receipt("research_acceptance",{"candidate_id":"c","ir_hash":"ir",
                                       "policy_hash":"p","data_receipt_id":data})
                scope={"sources":["test_forward"],"instruments":["*"],"resolved":True,"label_intervals":[],
                    "intervals":[["2030-01-01T00:00:00+00:00","2030-01-31T23:59:59+00:00"]]}
                cal=store.append("calibration",{"kind":"confirmatory_calibration","verified":True,"policy_hash":"p"})
                spec={"schema_version":"fuxi.confirmatory_protocol.1","policy_hash":"p","candidate_ids":["c"],
                     "baseline_hash":"base","scope":scope,"frozen_at":"2029-12-01T00:00:00+00:00",
                     "minimum_sections":10,"alpha":.05,"block_length":2,"bootstrap_replicates":199,
                     "look_schedule":["2030-02-03T00:00:00+00:00"],"cost_profile_hash":"cost",
                     "minimum_net_increment_daily":0,"calibration_evidence_id":cal}
                protocol=register_confirmatory_protocol(store,spec)
                parity=store.receipt("jq_parity",{"ir_hash":"ir","accepted":True,"n_pairs":20,"observed_on_jq":True})
                service=AcceptanceService(store)
                pending=service.freeze_forward(research,protocol,parity)
                rows=[]
                for i in range(20):
                    day=datetime(2030,1,1,tzinfo=timezone.utc)+timedelta(days=i)
                    rows.append({"decision_time":day.isoformat(),"mature_at":(day+timedelta(days=1)).isoformat(),
                                 "paired_net_increment":.01+i*.00001})
                observation=store.append("fixture_forward",{"ir_hash":"ir","baseline_hash":"base","cost_profile_hash":"cost",
                      "scope":scope,"rows":rows,"mature_at":"2030-02-01T00:00:00+00:00","account_identifiable":True})
                accepted=service.forward(pending,observation,spec["look_schedule"][0])
                self.assertEqual(accepted["decision"],"pass")
                release=ReleaseService(store)
                artifacts={k:digest(["fixture",k]) for k in ("code","model","strategy","data_contract","policy","state_schema")}
                base=release.bundle(accepted["acceptance_receipt"],artifacts)
                update=release.bundle(accepted["acceptance_receipt"],dict(artifacts,code=digest("fixture_update")),base)
                active=release.activate_paper(update)
                reconciliation=store.append("reconciliation",{"unknown_orders":0,"account_matched":True})
                restored=release.rollback(active,reconciliation)
                self.assertEqual(store.consume_receipt(restored,"paper_activation")["bundle_receipt"],base)
                with self.assertRaises(ContractError):store.consume_receipt(active,"paper_activation")
                store.revoke(data,"fixture correction")
                with self.assertRaises(ContractError):store.consume_receipt(restored,"paper_activation")
            finally:store.close()

    def test_research_receipt_keeps_revocable_data_dependency(self):
        with scratch_directory() as directory:
            store=EvidenceStore(Path(directory)/"db.sqlite")
            try:
                data=store.receipt("data_contract",{"snapshot_hash":"s","pit_certified":True,"universe_certified":True,
                      "account_events_certified":True,"execution_rules_certified":True,
                      "schema_version":"fuxi.independent_data_assurance.1","investment_eligibility_certified":True})
                result={"candidate_id":"c","snapshot_hash":"s","policy_hash":"p","strategy_ir":{"ir_hash":"i"},
                        "source_scope":"test_only","evaluations":[evaluation(s,"valid","fixture",decision="pass")
                                                                  for s in ("E0","E1","E2","E3","E4")]}
                event=store.append("fixture_evaluation",result)
                receipt=AcceptanceService(store).research(event,data)["receipt_id"]
                self.assertIsNotNone(receipt)
                store.revoke(data,"correction")
                with self.assertRaises(ContractError):store.consume_receipt(receipt,"research_acceptance")
            finally:store.close()

    def test_shared_exposure_blocks_other_output_and_renamed_source(self):
        with scratch_directory() as directory:
            root=Path(directory); payload=synthetic_campaign(130,6)
            authority=root/"global.sqlite"; store=EvidenceStore(authority)
            try:store.protect("sealed",scope_of(Snapshot(payload["snapshot"])),"2022-01-01T00:00:00+00:00")
            finally:store.close()
            payload["snapshot"]["source_id"]="renamed_vendor"
            payload["labels"]["feature_snapshot_hash"]=Snapshot(payload["snapshot"]).hash
            source=root/"input.json"; source.write_bytes(canonical(payload))
            with self.assertRaisesRegex(ContractError,"protected"):
                run_campaign(source,root/"different_output",rounds=1,candidates_per_round=1,
                             model_algorithms=(),authority_path=authority)

    def test_crash_after_settlement_recovers_feedback_once(self):
        with scratch_directory() as directory:
            root=Path(directory);source=root/"input.json"
            source.write_bytes(canonical(synthetic_campaign(130,6)))
            kwargs=dict(rounds=1,candidates_per_round=1,model_algorithms=(),campaign_id="campaign_crash")
            with patch("fuxi_v1.campaign.record_candidate_feedback",side_effect=RuntimeError("injected after settlement")):
                with self.assertRaisesRegex(RuntimeError,"injected"):
                    run_campaign(source,root/"runs",**kwargs)
            _,report=run_campaign(source,root/"runs",**kwargs)
            store=EvidenceStore(root/"runs/campaign_crash/evidence.sqlite")
            try:
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],1)
                self.assertEqual(report["attempted_candidates"],1)
                self.assertEqual(report["capability_reward_count"],1)
                self.assertEqual(len(store.facts("experience")),1)
            finally:store.close()

    def test_economic_feedback_changes_action_without_imputing_missing(self):
        history=[{"arm":a,"evaluated_count":2,"valid_e1_count":2,"elapsed_seconds":10,
                  "economic_reward_count":1 if a=="refine" else 0,"economic_reward_sum":.2 if a=="refine" else 0}
                 for a in ("fixed","random","refine","combine")]
        arm,decision=choose_arm(5,"adaptive",history,False)
        self.assertEqual(arm,"refine")
        self.assertIsNone(decision["economic_means"]["fixed"])

    def test_nested_multichannel_ablation_does_not_reuse_full_factor(self):
        ast=node("rank",node("add",node("rank",{"field":"close"}),node("rank",{"field":"volume"})))
        channels=single_field_channels(ast)
        self.assertEqual(len(channels),2)
        self.assertEqual({f for c in channels for f in c["fields"]},{"close","volume"})
        self.assertNotIn(ast,[c["ast"] for c in channels])

    def test_portfolio_axis_runs_accounts_and_preserves_scores(self):
        payload=synthetic_campaign(130,6);snapshot=Snapshot(payload["snapshot"])
        scores=evaluate_ast(node("rank",node("return",{"field":"close"},window=5)),snapshot)
        policy={"request":{"research_start":snapshot.sessions[80],"research_end":snapshot.sessions[-1],
                   "block_length":6,"alpha":.05,"max_trials":1,"bootstrap_replicates":99,"minimum_net_increment_daily":0}}
        result=portfolio_experiment(payload,policy,scores,3)
        self.assertEqual(result["changed_keys"],["rebalance_sessions"])
        self.assertEqual(result["scores_hash"],digest(scores))
        self.assertGreater(len(result["parent_account"]["orders"]),0)
        self.assertGreater(len(result["child_account"]["orders"]),0)
        self.assertFalse(result["comparison"]["economic_reward_eligible"])

    def test_jq_venue_mapping_and_no_authority_upgrade(self):
        payload=synthetic_campaign(130,6); snapshot=Snapshot(payload["snapshot"])
        scores=evaluate_ast(node("rank",{"field":"close"}),snapshot)
        ir=compile_strategy(strategy_spec("x"),scores,snapshot,digest(payload["market"]["execution"]))
        for row in ir["decisions"]:
            row["scores"]={s.replace(".XSHG",".SH"):v for s,v in row["scores"].items()}
        ir["ir_hash"]=digest({k:v for k,v in ir.items() if k != "ir_hash"})
        script,meta=compile_jq_strategy(ir,payload["market"])
        namespace={};exec(compile(script,"jq.py","exec"),namespace)
        self.assertEqual(namespace["FROZEN"]["instrument_mapping"]["600000.SH"],"600000.XSHG")
        self.assertFalse(meta["backtest_orders_allowed"])

    def test_duplicate_parity_keys_and_empty_field_bypass_rejected(self):
        rows=[{"key":"d","cash":1},{"key":"d","cash":1}]
        with self.assertRaises(ContractError):reconcile_rows(rows,rows,["cash"])
        with self.assertRaises(ContractError):reconcile_rows(rows[:1],rows[:1],[])

    def test_completed_duplicate_fill_idempotent_but_wrong_order_rejected(self):
        with scratch_directory() as directory:
            store=EvidenceStore(Path(directory)/"db.sqlite")
            try:
                receipt=store.receipt("paper_activation",{"mode":"paper"})
                gateway=PaperGateway(store);token=gateway.take_ownership("test")
                order=gateway.intent("test",token,{"mode":"paper","shares":100},receipt)
                gateway.transition(order,token,"submitted");gateway.transition(order,token,"acknowledged")
                gateway.fill_receipt("f",order,{"shares":100});gateway.transition(order,token,"filled")
                self.assertFalse(gateway.fill_receipt("f",order,{"shares":100}))
                with self.assertRaises(ContractError):gateway.fill_receipt("f","another_order",{"shares":100})
            finally:store.close()


if __name__ == "__main__":unittest.main()
