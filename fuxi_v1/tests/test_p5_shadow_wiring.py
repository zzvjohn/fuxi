"""P5 production wiring: shadow hooks + score-ref cache key A/B tests (v2.0 landing).

Acceptance anchors (P5_WIRING_PLAN.md section 5):
1. Score-ref nine-component key: on/off produce byte-identical scores on the same
   snapshot/AST batch, with explainable compute/hit counters.
2. Shadow switch A/B: a mini campaign with FUXI_V1_P5_SHADOW on vs off produces
   byte-identical production facts and the on-run additionally carries
   landing_shadow facts only.
3. Production-arm -> five-arm mapping covers the full production emitter set.
4. Fail-closed: every hook survives broken inputs (error fact, never raises).
5. Shadow readings are self-consistent (mask reasons, G0 would-block, G1 routing).
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fuxi_v1.contracts import ContractError, canonical, digest
from fuxi_v1.evidence import EvidenceStore
from fuxi_v1.tests.test_kernel import scratch_directory
from fuxi_v1.system_fixture import synthetic_campaign
from fuxi_v1.campaign import run_campaign
from fuxi_v1.data import Snapshot
from fuxi_v1.datasets import scope_of
from fuxi_v1.dsl import evaluate_ast, score_ref_mode
from fuxi_v1.search import node
from fuxi_v1.landing_shadow import (PRODUCTION_ARM_TO_V2, PRODUCTION_ARMS,
                                    ShadowState, ast_surface, hook_arm_selection,
                                    hook_dedup_cooldown, hook_export,
                                    hook_g0_precheck, hook_g1_complement)


def _fresh_snapshot():
    payload = synthetic_campaign(130, 6)
    return Snapshot(payload["snapshot"])


ASTS = [node("rank", node("return", {"field": "close"}, window=20)),
        node("rank", node("mean", {"field": "volume"}, window=5)),
        node("add", node("rank", node("return", {"field": "close"}, window=10)),
             node("rank", node("return", {"field": "close"}, window=2))),
        node("neg", node("rank", node("return", {"field": "close"}, window=5)))]


class ScoreRefKeyABTests(unittest.TestCase):
    """Acceptance 1: nine-component key on/off -> identical values, sane counters."""

    def _run_batch(self):
        snapshot = _fresh_snapshot()
        snapshot._score_cache = {}  # enable the per-snapshot cache like architecture layers do
        first = {}
        for ast in ASTS:
            first[digest(ast)] = evaluate_ast(ast, snapshot)
        stats = snapshot._score_ref_stats
        return first, stats

    def test_on_off_identical_values_and_counter_semantics(self):
        with patch.dict(os.environ, {"FUXI_V1_SCORE_REF_KEY": "off"}):
            off_values, off_stats = self._run_batch()
        with patch.dict(os.environ, {"FUXI_V1_SCORE_REF_KEY": "on"}):
            on_values, on_stats = self._run_batch()
        self.assertEqual(set(off_values), set(on_values))
        for key in off_values:
            self.assertEqual(off_values[key], on_values[key],
                             "nine-component key must not change score values")
        # Counter semantics: same batch, same cache size -> identical hit pattern.
        self.assertEqual(off_stats["computes"], on_stats["computes"])
        self.assertEqual(off_stats["hits"], on_stats["hits"])
        self.assertEqual(on_stats["mode"], "on")
        self.assertEqual(off_stats["mode"], "off")
        # 4 unique ASTs: first pass computes all, cached repeats would be hits.
        self.assertGreaterEqual(off_stats["computes"], len(ASTS))

    def test_key_mode_getter_and_unknown_mode_rejected(self):
        with patch.dict(os.environ, {"FUXI_V1_SCORE_REF_KEY": "on"}):
            self.assertEqual(score_ref_mode(), "on")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(score_ref_mode(), "on")  # default materialised (v0.7.186 flipped)
        with patch.dict(os.environ, {"FUXI_V1_SCORE_REF_KEY": "bogus"}):
            from fuxi_v1.contracts import ContractError
            with self.assertRaises(ContractError):
                score_ref_mode()

    def test_cache_hit_is_value_identical_under_on_key(self):
        with patch.dict(os.environ, {"FUXI_V1_SCORE_REF_KEY": "on"}):
            snapshot = _fresh_snapshot()
            snapshot._score_cache = {}
            a1 = evaluate_ast(ASTS[0], snapshot)
            a2 = evaluate_ast(ASTS[0], snapshot)
            self.assertEqual(a1, a2)
            self.assertEqual(snapshot._score_ref_stats["computes"], 1)
            self.assertEqual(snapshot._score_ref_stats["hits"], 1)


class ArmMappingTests(unittest.TestCase):
    """Acceptance 3: the production emitter set maps onto the five arms."""

    def test_mapping_is_complete(self):
        self.assertEqual(set(PRODUCTION_ARM_TO_V2), set(PRODUCTION_ARMS))
        for arm in PRODUCTION_ARMS:
            self.assertIn(PRODUCTION_ARM_TO_V2[arm],
                          ("A_template", "B_edit", "C_combine", "D_knowledge", "E_explore"))


class ShadowHookUnitTests(unittest.TestCase):
    """Acceptance 4+5: hooks are fail-closed and self-consistent on fake inputs."""

    def setUp(self):
        self.root = None

    def _store(self):
        import tempfile
        self.root = tempfile.mkdtemp()
        store = EvidenceStore(Path(self.root) / "evidence.sqlite")
        return store

    def tearDown(self):
        import shutil
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)

    def test_ast_surface_extraction(self):
        fields, ops, lookback = ast_surface(ASTS[2])
        self.assertEqual(set(fields), {"close"})
        self.assertEqual(set(ops), {"add", "rank", "return"})
        self.assertEqual(lookback, 10)

    def test_arm_selection_hook_records_divergence_and_masks(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            with store.transaction():
                scope = scope_of(_fresh_snapshot())
                store.expose("campaign_x", scope, "research")
                payload = hook_arm_selection(
                    state, store, "campaign_x", 0, "llm", [], budget_remaining=3,
                    llm_authorized=False, valid_parents=False, computable_parents=[],
                    template_slots_remaining=True)
            self.assertEqual(payload["production_arm_v2"], "D_knowledge")
            # llm_not_authorized must mask D_knowledge in both shadow selectors.
            self.assertIn("llm_not_authorized",
                          payload["round_robin"]["masked"].get("D_knowledge", []))
            self.assertIn("llm_not_authorized",
                          payload["ucb_shadow"]["masked"].get("D_knowledge", []))
            self.assertIn("D_knowledge", payload["round_robin"]["masked"])
            facts = [p for _, p in store.facts("landing_shadow")]
            self.assertEqual(len(facts), 0)  # direct call writes nothing; _safe writes
        finally:
            store.close()

    def test_g0_hook_would_block_on_unknown_field(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            proposal = {"candidate_id": "c1", "formula_hash": "f" * 64,
                        "canonical_ast": node("rank", node("return", {"field": "nope"}, window=5))}
            payload = hook_g0_precheck(state, store, "campaign_x", 0, [proposal],
                                       fields={"close", "volume"},
                                       operators={"rank", "return"}, session_count=130)
            self.assertEqual(payload["would_block"], 1)
            self.assertEqual(payload["would_block_candidates"][0]["reason_code"],
                             "field_missing_wait_for_data")
        finally:
            store.close()

    def test_dedup_hook_counts_duplicates_across_rounds(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            proposals = [{"candidate_id": "c1", "formula_hash": "f" * 64}]
            p1 = hook_dedup_cooldown(state, store, "campaign_x", 0, "fixed", proposals, "c" * 64)
            self.assertEqual(p1["new_attempts"], 1)
            p2 = hook_dedup_cooldown(state, store, "campaign_x", 1, "fixed", proposals, "c" * 64)
            self.assertEqual(p2["duplicate_attempts"], 1)
            self.assertEqual(p2["new_behavior_count"], 0)
            self.assertFalse(p2["cooldown"]["cooldown"])  # streak 1 < threshold 2
            p3 = hook_dedup_cooldown(state, store, "campaign_x", 2, "fixed", proposals, "c" * 64)
            # Two consecutive zero-behaviour microbatches cool the arm down.
            self.assertTrue(p3["cooldown"]["cooldown"])
        finally:
            store.close()

    def test_g1_complement_hook_fold_isolation(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            items = [{"candidate_id": "c1", "hypothesis_id": "h1", "parent_ids": ["p1"],
                      "source_refs": ["s1"], "generation_action_id": "g1",
                      "daily_ic": [0.1, 0.2, 0.3], "daily_support": [5, 6, 7],
                      "evaluation_bundle": {"e1_status": "valid", "e3_status": "valid"}}]
            payload = hook_g1_complement(state, store, "campaign_x", 0,
                                         fold_segment=["2025-01-01", "2025-12-31"],
                                         per_candidate=items)
            self.assertTrue(payload["fold_isolation_holds"])
            self.assertEqual(payload["fold_isolated_view_count"], 1)
            self.assertEqual(payload["rows"][0]["route"], "shortlist")
            # A different fold's view must not see this event.
            other = state.feedback.view(["2024-01-01", "2024-12-31"])
            self.assertEqual(other, [])
        finally:
            store.close()

    def test_export_hook_records_contract_gap_not_fake_receipt(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            # Production result wraps strategy_ir differently; the shadow must
            # report the contract gap honestly, never fabricate a frozen digest.
            fake_ir = {"schema_version": "fuxi.strategy_ir.1"}  # missing required keys
            payload = hook_export(state, store, "campaign_x", 0,
                                  [("e1", "c1", fake_ir)], code_ref="a" * 64,
                                  config_ref="b" * 64, input_contract_ref="c" * 64)
            self.assertEqual(payload["contract_valid"], 0)
            self.assertTrue(payload["rows"][0]["contract_check"].startswith("contract_gap:"))
            self.assertIsNone(payload["rows"][0]["frozen_digest"])
        finally:
            store.close()

    def test_export_hook_bridges_native_ir_to_valid_row(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            from fuxi_v1.portfolio import strategy_spec
            native = {"schema_version": "fuxi.strategy_ir.1",
                      "strategy": strategy_spec("c42", top_n=8, rebalance_sessions=5),
                      "snapshot_hash": "s" * 64, "execution_hash": "e" * 64,
                      "model_hash": None, "score_origin": "factor",
                      "decisions": [{"decision_time": "2024-01-02T15:00:00+08:00",
                                     "signal_session": "2024-01-02",
                                     "execution_session": "2024-01-03",
                                     "first_order_time": "2024-01-03T09:30:00+08:00",
                                     "scores": {"000001": 1.0},
                                     "universe": ["000001"]}],
                      "ir_hash": "i" * 64}
            market = {"execution": {"initial_cash": 1_000_000.0}}
            payload = hook_export(state, store, "campaign_x", 0,
                                  [("e1", "c42", native)], code_ref="a" * 64,
                                  config_ref="b" * 64, input_contract_ref="c" * 64,
                                  market=market,
                                  research_window=["2024-01-02", "2025-06-30"])
            self.assertEqual(payload["contract_valid"], 1)
            self.assertEqual(payload["rows"][0]["contract_check"], "valid")
            self.assertTrue(payload["rows"][0]["bridged"])
            self.assertIsNotNone(payload["rows"][0]["frozen_digest"])
        finally:
            store.close()

    def test_export_hook_native_ir_without_context_stays_gap(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            from fuxi_v1.portfolio import strategy_spec
            native = {"schema_version": "fuxi.strategy_ir.1",
                      "strategy": strategy_spec("c42"), "snapshot_hash": "s" * 64,
                      "execution_hash": "e" * 64, "model_hash": None,
                      "score_origin": "factor", "decisions": [], "ir_hash": "i" * 64}
            payload = hook_export(state, store, "campaign_x", 0,
                                  [("e1", "c42", native)], code_ref="a" * 64,
                                  config_ref="b" * 64, input_contract_ref="c" * 64)
            self.assertEqual(payload["contract_valid"], 0)
            self.assertIn("bridge context", payload["rows"][0]["contract_check"])
            self.assertIsNone(payload["rows"][0]["frozen_digest"])
        finally:
            store.close()

    def test_fail_closed_broken_inputs_never_raise(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            from fuxi_v1.landing_shadow import _safe
            from fuxi_v1.datasets import scope_of
            with store.transaction():
                store.expose("campaign_x", scope_of(_fresh_snapshot()), "research")
                # arm selection with an unmapped arm -> error fact, no exception.
                _safe(state, store, "campaign_x", "arm_selection", 0,
                      lambda: hook_arm_selection(state, store, "campaign_x", 0, "bogus_arm",
                                                 [], budget_remaining=1, llm_authorized=False,
                                                 valid_parents=False, computable_parents=[],
                                                 template_slots_remaining=True))
            errors = [p for _, p in store.facts("landing_shadow_error")]
            self.assertEqual(len(errors), 1)
            self.assertEqual(state.error_count, 1)
        finally:
            store.close()


class StatefulUcbTests(unittest.TestCase):
    """P5.1 (v0.7.187): the shadow UCB ledger accumulates across rounds,
    resolves microbatches (reaching non-cold reasons), replays on resume, and
    the campaign rounds bound admits statistical 100+ round runs."""

    def setUp(self):
        self.root = None

    def _store(self):
        import tempfile
        self.root = tempfile.mkdtemp()
        return EvidenceStore(Path(self.root) / "evidence.sqlite")

    def tearDown(self):
        import shutil
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)

    def _round_pair(self, state, store, round_index, contract_ref):
        """One arm_selection + dedup_cooldown pair with a fresh proposal set."""
        payload = hook_arm_selection(
            state, store, "campaign_x", round_index, "fixed", [],
            budget_remaining=100, llm_authorized=False, valid_parents=True,
            computable_parents=["p1", "p2"], template_slots_remaining=True)
        proposals = [{"candidate_id": "c" + str(round_index),
                      "formula_hash": digest({"round": round_index})}]
        settle = hook_dedup_cooldown(state, store, "campaign_x", round_index,
                                     "fixed", proposals, contract_ref)
        return payload, settle

    def test_ucb_accumulates_and_leaves_cold_start(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            contract_ref = "c" * 64
            chosen, reasons = [], []
            for round_index in range(12):
                payload, settle = self._round_pair(state, store, round_index, contract_ref)
                chosen.append(payload["ucb_shadow"]["chosen"])
                reasons.append(payload["ucb_shadow"]["reason"])
                self.assertEqual(settle["resolve"]["ucb_shadow"]["resolved_count"], 1)
                self.assertEqual(settle["resolve"]["ucb_shadow"]["arm"],
                                 payload["ucb_shadow"]["chosen"])
                self.assertEqual(settle["resolve"]["ucb_shadow"]["r_dev"], 1.0)
            # Cold start only rules the first eligible rounds; afterwards the
            # warm ledger must drive ucb_priority selections.
            self.assertEqual(reasons[0], "cold_start")
            self.assertIn("ucb_priority", reasons)
            self.assertIn("forced_explore", reasons)
            self.assertGreaterEqual(len(set(chosen)), 3)
            # Ledgers are per-selector: dispatches cross rounds, resolves close
            # every microbatch, and rr/ucb books never share state.
            for arm, st in state.arms_state_ucb.items():
                self.assertEqual(st["n_pending"], 0)
                self.assertGreaterEqual(st["n_resolved"], 1)
            self.assertEqual(sum(st["n_dispatched"] for st in state.arms_state_ucb.values()), 12)
            self.assertEqual(sum(st["n_resolved"] for st in state.arms_state_ucb.values()), 12)
            self.assertIsNot(state.arms_state_rr, state.arms_state_ucb)
            # round_robin book is independently driven: 12 dispatches, all
            # settled, fixed_rotation sequence covered by arms.py tests.
            self.assertEqual(sum(st["n_dispatched"] for st in state.arms_state_rr.values()), 12)
            self.assertEqual(sum(st["n_pending"] for st in state.arms_state_rr.values()), 0)
        finally:
            store.close()

    def test_replay_restores_selector_ledgers(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            from fuxi_v1.landing_shadow import _safe
            from fuxi_v1.datasets import scope_of
            with store.transaction():
                store.expose("campaign_x", scope_of(_fresh_snapshot()), "research")
                for round_index in range(8):
                    _safe(state, store, "campaign_x", "arm_selection", round_index,
                          lambda r=round_index: hook_arm_selection(
                              state, store, "campaign_x", r, "fixed", [],
                              budget_remaining=100, llm_authorized=False,
                              valid_parents=True, computable_parents=["p1", "p2"],
                              template_slots_remaining=True))
                    _safe(state, store, "campaign_x", "dedup_cooldown", round_index,
                          lambda r=round_index: hook_dedup_cooldown(
                              state, store, "campaign_x", r, "fixed",
                              [{"candidate_id": "c" + str(r),
                                "formula_hash": digest({"round": r})}], "c" * 64))
            restored = ShadowState(enabled=True)
            restored.replay(store)
            self.assertEqual(restored.arms_state_rr, state.arms_state_rr)
            self.assertEqual(restored.arms_state_ucb, state.arms_state_ucb)
            self.assertEqual(restored.dedup.new_attempts, state.dedup.new_attempts)
            self.assertEqual(restored.dedup.duplicate_attempts, state.dedup.duplicate_attempts)
            # The settled ledger must be warm, not a cold-start blank.
            self.assertGreaterEqual(sum(st["n_resolved"]
                                        for st in restored.arms_state_ucb.values()), 8)
        finally:
            store.close()

    def test_dedup_without_pending_selection_resolves_nothing(self):
        store = self._store()
        try:
            state = ShadowState(enabled=True)
            payload = hook_dedup_cooldown(
                state, store, "campaign_x", 0, "fixed",
                [{"candidate_id": "c1", "formula_hash": "f" * 64}], "c" * 64)
            self.assertEqual(payload["resolve"]["round_robin"]["arm"], None)
            self.assertEqual(payload["resolve"]["round_robin"]["resolved_count"], 0)
            self.assertEqual(payload["resolve"]["ucb_shadow"]["arm"], None)
            self.assertEqual(payload["resolve"]["ucb_shadow"]["resolved_count"], 0)
            self.assertEqual(payload["resolve"]["ucb_shadow"]["r_dev"], 1.0)
        finally:
            store.close()

    def test_rounds_bound_admits_statistical_runs(self):
        # 201 rounds is rejected on bounds; 120 passes bounds and fails later
        # on the missing input (never with "campaign bounds invalid").
        with self.assertRaises(ContractError):
            run_campaign(Path("missing_input.json"), Path("missing_out"), rounds=201)
        with self.assertRaises(Exception) as ctx:
            run_campaign(Path("missing_input.json"), Path("missing_out"), rounds=120)
        self.assertNotIn("campaign bounds invalid", str(ctx.exception))


class ShadowCampaignABTests(unittest.TestCase):
    """Acceptance 2: P5_SHADOW on vs off -> identical production facts + extra shadow facts.

    Non-semantic runtime fields are normalised away before comparison:
    paths / attempt ids / evaluation ids (uuid), wall-clock timestamps, and
    policy_hash — the latter is verified pre-existing non-determinism (two
    off-runs differ there too, independent of P5). The decision-semantic
    fields (candidate specs, results, counts) must be byte-identical.
    """

    _SKIP_KEYS = ("bundle_path", "evaluation_id", "process", "result_path",
                  "elapsed_seconds", "at", "policy_hash")

    @staticmethod
    def _normalize(fact):
        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()
                        if k not in ShadowCampaignABTests._SKIP_KEYS}
            if isinstance(node, list):
                return [walk(v) for v in node]
            return node
        return walk(fact)

    def _run(self, root, shadow):
        source = root / "input.json"
        source.write_bytes(canonical(synthetic_campaign(130, 6)))
        env = {"FUXI_V1_P5_SHADOW": shadow}
        with patch.dict(os.environ, env):
            _, report = run_campaign(source, root / ("runs_" + shadow), rounds=1,
                                     candidates_per_round=1, model_algorithms=(),
                                     campaign_id="campaign_p5")
        return report

    def test_shadow_on_off_production_facts_identical(self):
        with scratch_directory() as directory:
            root = Path(directory)
            off = self._run(root, "off")
            on = self._run(root, "on")
            self.assertEqual(off["attempted_candidates"], on["attempted_candidates"])
            off_db = EvidenceStore(root / "runs_off" / "campaign_p5" / "evidence.sqlite")
            on_db = EvidenceStore(root / "runs_on" / "campaign_p5" / "evidence.sqlite")
            try:
                for kind in ("candidate", "candidate_result", "round_completed",
                             "research_decision"):
                    off_facts = [self._normalize(p) for _, p in off_db.facts(kind)]
                    on_facts = [self._normalize(p) for _, p in on_db.facts(kind)]
                    self.assertEqual(off_facts, on_facts,
                                     kind + " decision-semantic fields must be identical across shadow on/off")
                # candidate facts carry no runtime identity at all: byte-identical raw.
                self.assertEqual([canonical(p) for _, p in off_db.facts("candidate")],
                                 [canonical(p) for _, p in on_db.facts("candidate")])
                # The on-run additionally carries shadow facts only.
                shadow_facts = [p for _, p in on_db.facts("landing_shadow")]
                self.assertGreaterEqual(len(shadow_facts), 1)
                self.assertEqual(len(list(off_db.facts("landing_shadow"))), 0)
                kinds = {p.get("kind") for p in shadow_facts}
                self.assertTrue(kinds, "shadow facts must be present on the on-run")
                for fact in shadow_facts:
                    self.assertIn("kind", fact)
            finally:
                off_db.close()
                on_db.close()


if __name__ == "__main__":
    unittest.main()
