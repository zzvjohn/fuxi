import unittest
from fuxi_v1.arms import (ARMS, arm_eligibility, arm_priority, resolve_microbatch,
                          select_action)


def base_context(**overrides):
    ctx = {"pending_microbatch": False, "budget_remaining": 30,
           "supported_fields": {"close", "volume"}, "supported_operators": {"rank", "add"},
           "template_slots_remaining": 5, "valid_parents": ["p1"],
           "computable_parents": ["p1", "p2"], "combination_already_tried": False,
           "new_knowledge_available": False, "llm_authorized": False,
           "cooldown": False, "unexplored_legal_space": True, "candidates": []}
    ctx.update(overrides)
    return ctx


def empty_state():
    return {}


class EligibilityTests(unittest.TestCase):
    def test_a_eligible_and_e_eligible(self):
        ctx = base_context()
        self.assertTrue(arm_eligibility("A_template", ctx)[0])
        self.assertTrue(arm_eligibility("E_explore", ctx)[0])

    def test_b_masked_without_parent(self):
        ok, reasons = arm_eligibility("B_edit", base_context(valid_parents=[]))
        self.assertFalse(ok)
        self.assertIn("no_valid_parent", reasons)

    def test_c_masked_with_single_parent(self):
        ok, reasons = arm_eligibility("C_combine", base_context(computable_parents=["p1"]))
        self.assertFalse(ok)
        self.assertIn("needs_two_parents", reasons)

    def test_c_masked_duplicate(self):
        ok, reasons = arm_eligibility("C_combine", base_context(combination_already_tried=True))
        self.assertFalse(ok)
        self.assertIn("duplicate_combination", reasons)

    def test_d_masked_without_knowledge_or_llm(self):
        ok, reasons = arm_eligibility("D_knowledge", base_context())
        self.assertFalse(ok)
        self.assertIn("no_new_knowledge", reasons)
        self.assertIn("llm_not_authorized", reasons)
        ok2, _ = arm_eligibility("D_knowledge", base_context(new_knowledge_available=True,
                                                             llm_authorized=True))
        self.assertTrue(ok2)

    def test_e_masked_cooldown(self):
        ok, reasons = arm_eligibility("E_explore", base_context(cooldown=True))
        self.assertFalse(ok)
        self.assertIn("cooldown", reasons)

    def test_in_flight_and_budget_masks_everything(self):
        ok, reasons = arm_eligibility("A_template", base_context(pending_microbatch=True))
        self.assertIn("in_flight", reasons)
        ok2, reasons2 = arm_eligibility("A_template", base_context(budget_remaining=0))
        self.assertIn("budget_exhausted", reasons2)

    def test_template_exhausted(self):
        ok, reasons = arm_eligibility("A_template", base_context(template_slots_remaining=0))
        self.assertIn("template_space_exhausted", reasons)


class SelectorTests(unittest.TestCase):
    def test_idle_when_all_masked(self):
        ctx = base_context(supported_fields=set(), template_slots_remaining=0,
                           valid_parents=[], computable_parents=[], cooldown=True,
                           unexplored_legal_space=False, budget_remaining=0)
        receipt = select_action(selector="round_robin", arms_state=empty_state(),
                                context=ctx, seed=7, selection_index=0)
        self.assertIsNone(receipt["chosen_arm"])
        self.assertEqual(receipt["reason"], "idle_exhausted")

    def test_round_robin_fixed_rotation(self):
        ctx = base_context()
        state = empty_state()
        chosen = []
        for i in range(4):
            receipt = select_action(selector="round_robin", arms_state=state,
                                    context=ctx, seed=7, selection_index=i)
            chosen.append(receipt["chosen_arm"])
            resolve_microbatch(receipt["chosen_arm"], 0.5, state)
        # base_context leaves A/B/C/E eligible (D lacks knowledge authorization)
        self.assertEqual(chosen, ["A_template", "B_edit", "C_combine", "E_explore"])

    def test_forced_explore_every_fifth(self):
        ctx = base_context()
        # A already dispatched once; forced explore picks the least-dispatched
        # eligible arm -> B_edit (ties break in ARMS order).
        state = {"A_template": {"n_dispatched": 1, "n_resolved": 1, "n_pending": 0,
                                "r_dev_sum": 0.5, "r_dev_count": 1}}
        receipt = select_action(selector="round_robin", arms_state=state,
                                context=ctx, seed=7, selection_index=4)
        self.assertEqual(receipt["reason"], "forced_explore")
        self.assertEqual(receipt["chosen_arm"], "B_edit")

    def test_ucb_cold_start(self):
        ctx = base_context()
        receipt = select_action(selector="ucb_shadow", arms_state=empty_state(),
                                context=ctx, seed=7, selection_index=0)
        self.assertEqual(receipt["reason"], "cold_start")
        self.assertEqual(receipt["chosen_arm"], "A_template")

    def test_ucb_prefers_better_arm(self):
        # Deterministic scenario: all eligible arms have equal n_resolved so the
        # exploration bonus cancels; A's mean r_dev (0.8) must win.
        state = empty_state()
        state["A_template"] = {"n_dispatched": 2, "n_resolved": 2, "n_pending": 0,
                               "r_dev_sum": 1.6, "r_dev_count": 2}
        state["E_explore"] = {"n_dispatched": 2, "n_resolved": 2, "n_pending": 0,
                              "r_dev_sum": 0.4, "r_dev_count": 2}
        for arm in ("B_edit", "C_combine"):
            state[arm] = {"n_dispatched": 2, "n_resolved": 2, "n_pending": 0,
                          "r_dev_sum": 1.0, "r_dev_count": 2}
        ctx = base_context()
        receipt = select_action(selector="ucb_shadow", arms_state=state,
                                context=ctx, seed=7, selection_index=1)
        self.assertEqual(receipt["reason"], "ucb_priority")
        self.assertEqual(receipt["chosen_arm"], "A_template")

    def test_priority_formula(self):
        state = {"A_template": {"n_dispatched": 3, "n_resolved": 3, "n_pending": 0,
                                "r_dev_sum": 0.9, "r_dev_count": 3}}
        total = 3
        expected = 0.3 + (2 * __import__("math").log(3) / 3) ** 0.5
        self.assertAlmostEqual(arm_priority("A_template", state, total), expected, places=10)

    def test_null_reward_never_enters_mean(self):
        state = empty_state()
        ctx = base_context()
        receipt = select_action(selector="ucb_shadow", arms_state=state, context=ctx,
                                seed=7, selection_index=0)
        arm = receipt["chosen_arm"]
        resolve_microbatch(arm, None, state)  # external failure: null r_dev
        self.assertEqual(state[arm]["n_resolved"], 1)
        self.assertEqual(state[arm]["r_dev_count"], 0)
        self.assertIsNone(arm_priority(arm, state, 1))

    def test_zero_output_counts_as_zero(self):
        state = empty_state()
        ctx = base_context()
        receipt = select_action(selector="round_robin", arms_state=state, context=ctx,
                                seed=7, selection_index=0)
        arm = receipt["chosen_arm"]
        resolve_microbatch(arm, 0.0, state)  # deterministic zero-output
        self.assertEqual(state[arm]["r_dev_count"], 1)
        self.assertAlmostEqual(state[arm]["r_dev_sum"], 0.0)

    def test_one_inflight_microbatch_per_arm(self):
        ctx = base_context()
        state = empty_state()
        receipt = select_action(selector="round_robin", arms_state=state, context=ctx,
                                seed=7, selection_index=0)
        arm = receipt["chosen_arm"]
        ctx["pending_microbatch"] = True
        ok, reasons = arm_eligibility(arm, ctx)
        self.assertIn("in_flight", reasons)

    def test_dispatch_counts_immediately(self):
        ctx = base_context()
        state = empty_state()
        receipt = select_action(selector="round_robin", arms_state=state, context=ctx,
                                seed=7, selection_index=0)
        arm = receipt["chosen_arm"]
        self.assertEqual(receipt["arms_state"][arm]["n_dispatched"], 1)
        self.assertEqual(receipt["arms_state"][arm]["n_pending"], 1)
        self.assertEqual(ctx["budget_remaining"], 29)


if __name__ == "__main__":
    unittest.main()
