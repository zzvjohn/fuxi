import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.score_ref import ScoreCache, score_key, score_ref
from fuxi_v1.gates import G2ShortlistQueue, g0_precheck, g1_diagnostics, g2_seats
from fuxi_v1.queue import CooldownTracker, DedupLedger, DependencyQueue
from fuxi_v1.routing import route_from_legacy


def key_dict(**overrides):
    base = {"canonical_ast": {"op": "rank", "args": [{"op": "return", "window": 15}]},
            "operator_catalog_hash": "a" * 64, "snapshot_ref": "b" * 64,
            "universe_ref": "c" * 64, "eval_window": ["2024-01-01", "2024-12-31"],
            "point_in_time_rule": "asof_tolerance_180d",
            "adjustment_rule": "qfq", "missing_rule": "exclude_section",
            "eval_impl_ref": "d" * 64}
    base.update(overrides)
    return base


def make_ref(score_id, key):
    return score_ref(candidate_id="c1", score_id=score_id, key=key, eval_status="valid",
                     n_sections=100, support_daily=[50] * 100, rank_ic_mean=0.02)


class ScoreCacheTests(unittest.TestCase):
    def test_shared_values_computed_once(self):
        cache = ScoreCache()
        key = key_dict()
        _, score_id = score_key(**key)
        calls = []

        def compute(k):
            calls.append(1)
            return make_ref(score_id, k)

        a = cache.get_or_compute(key, compute)
        b = cache.get_or_compute(key, compute)
        self.assertIs(a, b)
        self.assertEqual(cache.computes, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(a["rank_ic_mean"], b["rank_ic_mean"])

    def test_every_key_component_changes_score_id(self):
        _, base_id = score_key(**key_dict())
        changes = {"canonical_ast": {"op": "rank", "args": []},
                   "operator_catalog_hash": "e" * 64, "snapshot_ref": "f" * 64,
                   "universe_ref": "11" * 32, "eval_window": ["2024-02-01", "2024-12-31"],
                   "point_in_time_rule": "exact", "adjustment_rule": "raw",
                   "missing_rule": "fill_prev", "eval_impl_ref": "22" * 32}
        for field, value in changes.items():
            _, changed = score_key(**key_dict(**{field: value}))
            self.assertNotEqual(base_id, changed, field)

    def test_computed_id_mismatch_rejected(self):
        cache = ScoreCache()
        key = key_dict()
        _, score_id = score_key(**key)
        other = score_ref(candidate_id="c1", score_id="3" * 64, key=key, eval_status="valid",
                          n_sections=1, support_daily=[1])
        with self.assertRaises(ContractError):
            cache.get_or_compute(key, lambda k: other)

    def test_no_overwrite(self):
        cache = ScoreCache()
        key = key_dict()
        _, score_id = score_key(**key)
        cache.put(make_ref(score_id, key))
        with self.assertRaises(ContractError):
            cache.put(make_ref(score_id, key))


class G0Tests(unittest.TestCase):
    SURFACE = {"fields": {"close", "volume"}, "operators": {"rank", "return"},
               "units": {"close": "price"}, "min_session_count": 250,
               "decision_time": "2024-06-03T18:00:00+08:00",
               "label_mature_at": "2024-06-10T18:00:00+08:00"}

    def test_normal_valid(self):
        spec = {"required_fields": ["close"], "field_units": {"close": "price"},
                "operators": ["rank"], "lookback": 20}
        self.assertEqual(g0_precheck(spec, self.SURFACE)["eval_status"], "valid")

    def test_unsupported_operator(self):
        spec = {"required_fields": ["close"], "operators": ["quantile"]}
        out = g0_precheck(spec, self.SURFACE)
        self.assertEqual(out["eval_status"], "precheck_rejected")
        self.assertEqual(out["capability_status"], "unsupported")

    def test_missing_field(self):
        spec = {"required_fields": ["turnover_rate"]}
        out = g0_precheck(spec, self.SURFACE)
        self.assertEqual(out["eval_status"], "data_insufficient")
        self.assertEqual(out["reason_code"], "field_missing_wait_for_data")

    def test_lookback_exceeds_depth(self):
        spec = {"required_fields": ["close"], "lookback": 400}
        self.assertEqual(g0_precheck(spec, self.SURFACE)["reason_code"],
                         "lookback_exceeds_data_depth")

    def test_unit_mismatch(self):
        spec = {"required_fields": ["close"], "field_units": {"close": "shares"}}
        self.assertEqual(g0_precheck(spec, self.SURFACE)["reason_code"], "unit_mismatch")

    def test_label_not_mature(self):
        spec = {"required_fields": ["close"], "label_horizon_end": "2024-06-12T18:00:00+08:00"}
        self.assertEqual(g0_precheck(spec, self.SURFACE)["reason_code"], "label_not_mature")

    def test_statuses_stay_in_five(self):
        for spec in ({}, {"required_fields": ["close"]}):
            self.assertIn(g0_precheck(spec, self.SURFACE)["eval_status"],
                          ("valid", "data_insufficient", "precheck_rejected", "error",
                           "not_evaluated"))


class G1Tests(unittest.TestCase):
    def test_icir_null_on_constant_sections(self):
        out = g1_diagnostics([], daily_support=[])
        self.assertIsNone(out["rank_ic_mean"])
        self.assertIsNone(out["icir_unannualized"])
        self.assertEqual(out["reason_code"], "constant_cross_section")

    def test_mean_and_icir(self):
        out = g1_diagnostics([0.01, 0.02, 0.03], daily_support=[100, 100, 100])
        self.assertAlmostEqual(out["rank_ic_mean"], 0.02, places=10)
        sd = ((0.01 - 0.02) ** 2 + 0 + (0.03 - 0.02) ** 2) / 2
        self.assertAlmostEqual(out["icir_unannualized"], 0.02 / sd ** 0.5, places=10)

    def test_zero_variance_null(self):
        out = g1_diagnostics([0.02, 0.02, 0.02], daily_support=[100, 100, 100])
        self.assertIsNone(out["icir_unannualized"])


class G2Tests(unittest.TestCase):
    def test_quota_formula(self):
        self.assertEqual(g2_seats(0, 0, 100)["seats"], 0)
        self.assertEqual(g2_seats(1, 0, 100)["seats"], 1)
        self.assertEqual(g2_seats(6, 0, 100)["seats"], 2)
        self.assertEqual(g2_seats(45, 0, 100)["seats"], 8)
        self.assertEqual(g2_seats(45, 8, 100)["seats"], 0)     # exhausted
        self.assertEqual(g2_seats(45, 0, 3)["seats"], 3)       # budget capped
        self.assertTrue(g2_seats(45, 0, 3)["notes"]["budget_capped"])

    def test_exploration_carryover(self):
        # Cumulative seats reach 5 -> one exploration seat is granted
        first = g2_seats(25, 0, 100, exploration_state={})
        self.assertEqual(first["seats"], 5)
        self.assertEqual(first["exploration_seats"], 1)
        # Carryover state: cohort grows to Q=8, cumulative 8 -> target still 1,
        # so no further exploration grant from the carried state
        second = g2_seats(40, 5, 100, exploration_state=first["state"])
        self.assertEqual(second["seats"], 3)
        self.assertEqual(second["exploration_seats"], 0)
        # No new seats, no new exploration
        third = g2_seats(40, 8, 100, exploration_state=second["state"])
        self.assertEqual(third["seats"], 0)
        self.assertEqual(third["exploration_seats"], 0)
        # A fresh cohort restarts its own carryover at its own fifth seat
        fresh = g2_seats(25, 0, 100, exploration_state={})
        self.assertEqual(fresh["exploration_seats"], 1)

    def test_shortlist_queue_explicit_not_evaluated(self):
        queue = G2ShortlistQueue()
        legacy = [{"schema_version": "fuxi.evaluation.1", "stage": "E1", "eval_status": "valid",
                   "decision": "pass", "reject_class": None, "reason_code": "x",
                   "metrics": None, "diagnostics": {}, "evidence_scope": "research",
                   "n_pairs": 0, "economic_reward_eligible": False, "promotable": False}]
        for i in range(5):
            queue.submit("c%d" % i, route_from_legacy("c%d" % i, legacy))
        seated = queue.allocate(2)
        self.assertEqual(len(seated), 2)
        unseated = queue.unseated()
        self.assertEqual(len(unseated), 3)
        for record in unseated:
            self.assertEqual(record["eval_status"], "not_evaluated")
            self.assertEqual(record["reason_code"], "account_quota_not_allocated")

    def test_baseline_cache(self):
        queue = G2ShortlistQueue()
        queue.cache_baseline("bs_ref", {"nav": [1.0, 1.01]})
        self.assertEqual(queue.get_baseline("bs_ref")["nav"][-1], 1.01)
        self.assertIsNone(queue.get_baseline("other_ref"))


class QueueTests(unittest.TestCase):
    def test_dedup(self):
        ledger = DedupLedger()
        self.assertEqual(ledger.record("ast1", "contract1", "c1")["verdict"], "new")
        out = ledger.record("ast1", "contract1", "c2")
        self.assertEqual(out["verdict"], "duplicate")
        self.assertEqual(out["first_candidate_id"], "c1")
        self.assertFalse(out["new_output"])
        self.assertEqual((ledger.new_attempts, ledger.duplicate_attempts), (1, 1))

    def test_cooldown(self):
        tracker = CooldownTracker(threshold=2)
        self.assertFalse(tracker.record("arm_a", 0)["cooldown"])
        self.assertTrue(tracker.record("arm_a", 0)["cooldown"])
        self.assertFalse(tracker.record("arm_a", 1)["cooldown"])
        self.assertFalse(tracker.record("arm_a", 0)["cooldown"])

    def test_dependency_queue(self):
        queue = DependencyQueue()
        self.assertEqual(queue.request("need_1", {"field": "turnover"}, "c1")["verdict"],
                         "task_created")
        self.assertEqual(queue.request("need_1", {"field": "turnover"}, "c2")["verdict"],
                         "waiting")
        self.assertEqual(queue.request("need_1", {"field": "turnover"}, "c2")["verdict"],
                         "already_waiting")
        out = queue.resolve("need_1", {"data_ready": True})
        self.assertEqual(sorted(out["unblocked"]), ["c1", "c2"])
        self.assertEqual(queue.request("need_1", {"field": "turnover"}, "c3")["verdict"],
                         "resolved")
        self.assertEqual(queue.pending_tasks(), [])


if __name__ == "__main__":
    unittest.main()
