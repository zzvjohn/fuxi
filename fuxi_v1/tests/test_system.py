from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import unittest
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.contracts import ContractError, digest, evaluation, canonical
from fuxi_v1.data import Snapshot, Labels
from fuxi_v1.datasets import validate_label_calendar, scope_of, read_calendar
from fuxi_v1.dsl import evaluate_ast, compile_ast, CATALOG
from fuxi_v1.evidence import EvidenceStore, overlaps
from fuxi_v1.models import walk_forward, validate_axis
from fuxi_v1.statistics import block_inference, holm, required_sections
from fuxi_v1.search import node, generate
from fuxi_v1.portfolio import strategy_spec, target_weights, compile_strategy
from fuxi_v1.simulation import run_account
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.execution import PaperGateway, reconcile_rows
from fuxi_v1.acceptance import AcceptanceService, ReleaseService
from fuxi_v1.jq_strategy import compile_jq_strategy
from fuxi_v1.plugins import meta_compare
from fuxi_v1.campaign import run_campaign, replay_feedback


class PITModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = synthetic_campaign(130, 6)

    def test_bitemporal_revision_arrives_later(self):
        p = deepcopy(self.payload["snapshot"])
        p["schema_version"] = "fuxi.snapshot.2"
        for row in p["observations"]:
            row.update(event_time=row["session"]+"T15:00:00+08:00", published_at=row["available_at"],
                       ingested_at=row["available_at"], quality_status="historically_certified", source_id="fixture")
        old = p["observations"][40*6*2]
        revised = deepcopy(old)
        revised.update(value=999, available_at=p["sessions"][45]+"T18:00:00+08:00",
                       published_at=p["sessions"][45]+"T18:00:00+08:00", ingested_at=p["sessions"][45]+"T18:00:00+08:00",
                       revision_id="revised")
        p["observations"].append(revised)
        snapshot = Snapshot(p)
        self.assertEqual(snapshot.value(40, old["instrument"], "close", snapshot.clocks[p["sessions"][44]]), old["value"])
        self.assertEqual(snapshot.value(40, old["instrument"], "close", snapshot.clocks[p["sessions"][45]]), 999)

    def test_label_calendar_not_just_timestamp_order(self):
        payload = deepcopy(self.payload["labels"])
        payload["label_spec"]["horizon_sessions"] = 2
        with self.assertRaisesRegex(ContractError, "horizon"):
            validate_label_calendar(payload, self.payload["snapshot"]["sessions"])

    def test_embedded_csv_calendar_header_is_audited_not_zero_sessions(self):
        with scratch_directory() as directory:
            path=Path(directory)/"calendar.csv"
            path.write_text("exchange,cal_date,is_open\nSSE,20240102,1\nexchange,cal_date,is_open\nSSE,20240103,1\nSSE,20240106,0\n",encoding="utf-8")
            days,audit=read_calendar(path,"2024-01-01","2024-01-10")
            self.assertEqual(days,["2024-01-02","2024-01-03"])
            self.assertEqual(audit["embedded_header_rows_ignored"],1)

    def test_walk_forward_future_label_perturbation(self):
        snapshot = Snapshot(self.payload["snapshot"])
        labels = Labels(self.payload["labels"], snapshot)
        ast = node("rank", node("return", {"field":"close"}, window=5))
        features = {"f": evaluate_ast(ast, snapshot)}
        first = snapshot.sessions[70]
        original = walk_forward(features, snapshot, labels, predict_start=first, min_train_sessions=20, retrain_every=100)
        changed = deepcopy(self.payload["labels"])
        for row in changed["rows"]:
            if row["mature_at"] > snapshot.payload["decision_times"][first]:
                row["value"] *= -500
        after = walk_forward(features, snapshot, Labels(changed,snapshot), predict_start=first, min_train_sessions=20, retrain_every=100)
        self.assertEqual(original["scores"], after["scores"])
        self.assertGreater(original["trained_model_count"], 0)
        self.assertTrue(all(p["score_origin"] == "walk_forward" for p in original["prediction_provenance"] if p["status"] == "valid"))

    def test_single_axis_change_rejects_rename_and_unregistered_keys(self):
        baseline = {"features":["a"],"model":"ridge"}
        validate_axis(baseline,{"features":["a","b"],"model":"ridge"},"features")
        for bad in ({"features":["b"],"model":"ridge"},{"features":["a","b"],"model":"xgb"},
                    {"features":["a","b"],"model":"ridge","extra":1}):
            with self.assertRaises(ContractError):
                validate_axis(baseline,bad,"features")

    def test_catalog_all_operators_have_actual_consumers(self):
        snapshot = Snapshot(self.payload["snapshot"])
        close = {"field":"close"}
        used = 0
        for op, arity in CATALOG["operators"].items():
            temporal = op in {"lag","return","mean","std","min","max","delta"}
            expression = node(op, *([node("rank",close)] if op == "log" else [close]*arity), window=2 if temporal else None)
            compile_ast(expression,snapshot.payload["fields"])
            values = evaluate_ast(expression,snapshot)
            self.assertEqual(set(values),set(snapshot.sessions[snapshot.start:]))
            used += 1
        self.assertEqual(used,len(CATALOG["operators"]))


class EvidenceTests(unittest.TestCase):
    def test_budget_retries_and_recovery_costs(self):
        with scratch_directory() as path:
            store = EvidenceStore(Path(path)/"evidence.sqlite")
            try:
                store.campaign("c",{"budget_units":3})
                one=store.reserve("c","experiment",2)
                self.assertEqual(store.recover("c"),1)
                with self.assertRaisesRegex(ContractError,"budget"):
                    store.reserve("c","experiment",2)
                two=store.reserve("c","experiment",1)
                store.finish(two,"completed",.5,{"value":1})
                self.assertEqual(store.cached("experiment"),{"value":1})
                self.assertEqual(store.db.execute("SELECT SUM(actual) FROM attempts").fetchone()[0],2.5)
            finally:
                store.close()

    def test_immutable_facts_and_frozen_campaign(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                event=store.append("test",{"a":1},"same")
                self.assertEqual(store.append("test",{"a":1},"same"),event)
                with self.assertRaises(ContractError): store.append("test",{"a":2},"same")
                store.campaign("c",{"budget_units":3})
                with self.assertRaises(ContractError): store.campaign("c",{"budget_units":4})
            finally: store.close()

    def test_reward_revision_invalidates_all_downstream_states(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                a=store.append("evaluation",evaluation("E1","valid","descriptive"))
                slot=["decision","candidate","capability","window","baseline"]
                first=store.reward(slot,"capability",1.,a)
                self.assertEqual(store.reward(slot,"capability",1.,a),first)
                store.register_consumer("scheduler",[first],"checkpoint0")
                store.register_consumer("prompt",["scheduler"],"checkpoint0")
                b=store.append("evaluation",evaluation("E1","error","correction"))
                store.reward(slot,"capability",None,b,supersedes=first)
                self.assertEqual(len(store.effective_rewards("capability")),1)
                self.assertIsNone(store.effective_rewards("capability")[0]["value"])
                self.assertFalse(store.state_usable("scheduler"))
                self.assertFalse(store.state_usable("prompt"))
                with self.assertRaisesRegex(ContractError,"economic"):
                    store.reward(["d","c","economic","w","b"],"economic",0.,b)
            finally: store.close()

    def test_exposure_detects_label_support_overlap_across_hashes(self):
        def scope(a,b):
            return {"sources":["prices"],"instruments":["*"],"intervals":[[a,b]],"label_intervals":[],"resolved":True}
        protected=scope("2030-01-10T00:00:00+00:00","2030-02-10T00:00:00+00:00")
        research=scope("2029-01-01T00:00:00+00:00","2029-12-20T00:00:00+00:00")
        research["label_intervals"]=[["2029-12-20T00:00:00+00:00","2030-01-15T00:00:00+00:00"]]
        self.assertTrue(overlaps(protected,research))
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                store.protect("cohort",protected,"2029-12-31T00:00:00+00:00")
                with self.assertRaisesRegex(ContractError,"protected"):
                    store.expose("another",research,"derived")
                store.expose("cohort",protected,"acceptance",acceptance=True)
                with self.assertRaisesRegex(ContractError,"exposed"):
                    store.protect("renamed",protected,"2029-12-31T00:00:00+00:00")
            finally: store.close()


class PortfolioExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=synthetic_campaign(130,6)
        cls.snapshot=Snapshot(cls.payload["snapshot"])
        cls.scores=evaluate_ast(node("rank",node("return",{"field":"close"},window=5)),cls.snapshot)

    def ir(self):
        return compile_strategy(strategy_spec("test",top_n=2,max_weight=.5),self.scores,self.snapshot,
                                digest(self.payload["market"]["execution"]))

    def test_ties_caps_turnover_cash(self):
        spec=strategy_spec("x",top_n=2,max_weight=.4,turnover_budget=.2)
        weights=target_weights({"a":1,"b":1,"c":1},spec)
        self.assertEqual(len(weights),3)
        self.assertAlmostEqual(sum(weights.values()),1.)
        moved=target_weights({"a":2,"b":1},spec,{"c":1.})
        self.assertLessEqual(sum(abs(moved.get(s,0)-{"c":1}.get(s,0)) for s in moved),.20000001)

    def test_next_session_execution_and_no_current_volume_sizing(self):
        ir=self.ir()
        a=run_account(ir,self.payload["market"])
        market=deepcopy(self.payload["market"])
        day=ir["decisions"][0]["execution_session"]
        for bar in market["bars"]:
            if bar["session"] == day: bar["volume"] *= 100
        b=run_account(ir,market)
        self.assertEqual([r for r in a["orders"] if r["session"] == day],
                         [r for r in b["orders"] if r["session"] == day])
        self.assertGreater(len(a["orders"]),0)
        self.assertTrue(all(o["signal_time"][:10] < o["session"] for o in a["orders"]))

    def test_missing_quote_or_event_never_drops_held_denominator(self):
        market=deepcopy(self.payload["market"])
        baseline=run_account(self.ir(),market)
        held_day=next(r for r in baseline["daily"] if r["holdings"])
        code=next(iter(held_day["holdings"]))
        for bar in market["bars"]:
            if bar["session"] == held_day["session"] and bar["instrument"] == code: bar["close"]=None
        result=run_account(self.ir(),market)
        row=next(r for r in result["daily"] if r["session"] == held_day["session"])
        self.assertIsNone(row["nav"])
        self.assertIn(code,row["holdings"])
        self.assertEqual(result["eval_status"],"data_insufficient")

    def test_JQ_compiler_targets_match_without_trading_permission(self):
        ir=self.ir()
        script,meta=compile_jq_strategy(ir,self.payload["market"])
        namespace={}
        exec(compile(script,"jq_strategy.py","exec"),namespace)
        count=0
        for row in ir["decisions"]:
            self.assertEqual(namespace["planned_targets"](row["signal_session"]),
                             target_weights(row["scores"],ir["strategy"]))
            count+=1
        self.assertGreater(count,0)
        self.assertFalse(meta["backtest_orders_allowed"])
        self.assertNotIn("order",namespace)

    def test_parity_requires_nonzero_same_keys_and_each_account_field(self):
        with self.assertRaises(ContractError): reconcile_rows([],[],["nav"])
        result=reconcile_rows([{"key":"d","nav":100,"cash":5}],[{"key":"d","nav":100,"cash":4}],["nav","cash"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["differences"][0]["field"],"cash")


class AcceptanceOperationTests(unittest.TestCase):
    def test_gateway_unknown_outcome_fence_and_duplicate_fill(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                # Test-only privileged authority; no external broker is called.
                activation=store.receipt("paper_activation",{"mode":"paper","live_authorized":False})
                gateway=PaperGateway(store)
                token=gateway.take_ownership("paper_account")
                order=gateway.intent("paper_account",token,{"mode":"paper","shares":100},activation)
                gateway.transition(order,token,"submitted")
                gateway.transition(order,token,"unknown")
                with self.assertRaisesRegex(ContractError,"reconciliation"):
                    gateway.transition(order,token,"acknowledged")
                next_token=gateway.take_ownership("paper_account")
                with self.assertRaisesRegex(ContractError,"fencing"):
                    gateway.transition(order,token,"acknowledged",reconciliation=True)
                gateway.transition(order,next_token,"acknowledged",reconciliation=True)
                self.assertTrue(gateway.fill_receipt("fill",order,{"shares":50}))
                self.assertFalse(gateway.fill_receipt("fill",order,{"shares":50}))
                gateway.transition(order,next_token,"partially_filled")
            finally: store.close()

    def test_upstream_revocation_invalidates_downstream_consumption(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                forward=store.receipt("forward_acceptance",{"accepted":True})
                bundle=store.receipt("release_bundle",{"forward_receipt":forward})
                paper=store.receipt("paper_activation",{"bundle_receipt":bundle})
                store.consume_receipt(paper,"paper_activation")
                store.revoke(forward,"data corrected")
                with self.assertRaisesRegex(ContractError,"inactive"):
                    store.consume_receipt(paper,"paper_activation")
            finally: store.close()

    def test_forward_look_schedule_maturity_and_nonzero_pair_control(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                scope={"sources":["forward_prices"],"instruments":["*"],
                       "intervals":[["2030-01-01T00:00:00+00:00","2030-02-01T00:00:00+00:00"]],"label_intervals":[],"resolved":True}
                protocol={"scope":scope,"candidate_ids":["c"],"frozen_at":"2029-12-01T00:00:00+00:00",
                          "look_schedule":["2030-03-01T00:00:00+00:00"],"baseline_hash":"b","cost_profile_hash":"cost",
                          "minimum_sections":10,"alpha":.05,"block_length":2,"bootstrap_replicates":199,"minimum_net_increment_daily":0}
                research=store.receipt("research_acceptance",{"candidate_id":"c","ir_hash":"ir"})
                policy=store.receipt("confirmatory_protocol",protocol)
                parity=store.receipt("jq_parity",{"ir_hash":"ir","accepted":True,"n_pairs":20,"observed_on_jq":True})
                service=AcceptanceService(store)
                pending=service.freeze_forward(research,policy,parity)
                obs=store.append("observation",{"ir_hash":"ir","baseline_hash":"b","cost_profile_hash":"cost",
                     "scope":scope,"mature_at":"2030-02-02T00:00:00+00:00","rows":[],"account_identifiable":False})
                with self.assertRaisesRegex(ContractError,"look"):
                    service.forward(pending,obs,"2030-02-03T00:00:00+00:00")
                result=service.forward(pending,obs,protocol["look_schedule"][0])
                self.assertEqual(result["eval_status"],"data_insufficient")
                with self.assertRaisesRegex(ContractError,"spent"):
                    service.forward(pending,obs,protocol["look_schedule"][0])
            finally: store.close()

    def test_multiple_testing_and_power_are_explicit(self):
        result=holm({"a":.01,"b":.03,"c":None},.05)
        self.assertTrue(result["a"]["reject_null"])
        self.assertFalse(result["b"]["reject_null"])
        self.assertEqual(result["c"]["family_size"],3)
        small=block_inference([.01]*3,block_length=2,alpha=.05,replicates=99)
        self.assertEqual(small["status"],"data_insufficient")
        self.assertGreater(required_sections(.2,.02),required_sections(.1,.02))

    def test_meta_optimizer_cannot_modify_judge(self):
        parent={"prompt_template":"a","cost_profile":"frozen"}
        with self.assertRaisesRegex(ContractError,"immutable"):
            meta_compare(parent,{"prompt_template":"a","cost_profile":"cheap"},[])

    def test_corrected_rewards_rebuild_actual_scheduler_statistics(self):
        with scratch_directory() as path:
            store=EvidenceStore(Path(path)/"db.sqlite")
            try:
                old=store.append("evaluation",evaluation("E1","valid","first"))
                slot=["campaign_test:decision:0","candidate","capability","window","baseline"]
                event=store.reward(slot,"capability",1.,old)
                rows=[{"round":0,"valid_e1_count":1,"arm":"fixed","evaluated_count":1,"elapsed_seconds":1}]
                before,state=replay_feedback(store,"campaign_test",rows)
                self.assertEqual(before[0]["valid_e1_count"],1.)
                corrected=store.append("evaluation",evaluation("E1","error","corrected"))
                store.reward(slot,"capability",None,corrected,supersedes=event)
                self.assertFalse(store.state_usable(state))
                after,new_state=replay_feedback(store,"campaign_test",rows)
                self.assertEqual(after[0]["valid_e1_count"],0)
                self.assertTrue(store.state_usable(new_state))
                self.assertEqual(rows[0]["valid_e1_count"],1)
            finally: store.close()

    def test_complete_campaign_resume_does_not_repeat_attempt_or_reward(self):
        with scratch_directory() as path:
            root=Path(path)
            source=root/"input.json"
            source.write_bytes(canonical(synthetic_campaign(130,6)))
            _,first=run_campaign(source,root/"runs",rounds=1,candidates_per_round=1,model_algorithms=(),campaign_id="campaign_resume")
            store=EvidenceStore(root/"runs/campaign_resume/evidence.sqlite")
            before=store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            store.close()
            _,second=run_campaign(source,root/"runs",rounds=1,candidates_per_round=1,model_algorithms=(),campaign_id="campaign_resume")
            store=EvidenceStore(root/"runs/campaign_resume/evidence.sqlite")
            try:
                self.assertEqual(before,store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
                self.assertEqual(first["capability_reward_count"],second["capability_reward_count"])
                self.assertEqual(second["economic_reward_count"],0)
                self.assertEqual(second["jq_candidate_count"],0)
            finally: store.close()


if __name__ == "__main__": unittest.main()
