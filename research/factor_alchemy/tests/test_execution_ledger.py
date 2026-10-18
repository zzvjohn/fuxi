import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio_execution_ledger import CorporateAction, LotRule, run_ledger, reconcile_journal


class ExecutionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
        self.bars = {(d, c): dict(state="trading", open=10., close=10., pre_close=10.,
                                  up_limit=11., down_limit=9., action_check_available=True)
                     for d in ["2024-12-31"] + self.dates for c in ["A", "B"]}
        self.target = {"2024-12-31": {"A": 1.}}

    def run_case(self, targets=None, **kwargs):
        result = run_ledger(self.dates, self.bars, self.target if targets is None else targets,
                            initial_cash=10000., cost_bps=kwargs.pop("cost_bps", 0.), **kwargs)
        reconcile_journal(result, 10000.)
        return result

    def test_signal_close_sizes_next_open_and_cash_not_negative(self):
        self.bars[(self.dates[0], "A")].update(open=12., close=12., up_limit=15.)
        out = self.run_case(cost_bps=15.)
        trade = out.journal[0]
        self.assertEqual(trade["quantity"], 800)
        self.assertAlmostEqual(out.days[0]["cash"], 385.6)
        self.assertAlmostEqual(out.days[0]["nav"], 9985.6)
        self.assertLess(trade["signal_date"], trade["date"])

    def test_fixed_shares_not_daily_equal_weight(self):
        self.bars[(self.dates[1], "A")].update(open=10., close=20.)
        self.bars[(self.dates[2], "A")].update(open=20., close=30., pre_close=20., up_limit=40.)
        out = self.run_case({"2024-12-31": {"A": .5, "B": .5}})
        self.assertEqual(out.days[1]["nav"], 15000.)
        self.assertEqual(out.days[2]["nav"], 20000.)
        self.assertEqual(out.days[2]["positions"]["A"]["shares"], 500)

    def test_halt_blocks_exit_then_resumption_gap_and_fee_counted(self):
        self.bars[(self.dates[1], "A")] = dict(state="halted", evidence="fixture", action_check_available=True)
        self.bars[(self.dates[2], "A")].update(open=8., close=8., down_limit=7.)
        out = self.run_case({**self.target, self.dates[0]: {}}, cost_bps=15.)
        self.assertEqual(out.days[1]["positions"]["A"]["shares"], 900)
        self.assertEqual(out.days[1]["positions"]["A"]["mark"], 10.)
        self.assertEqual(out.days[2]["positions"], {})
        self.assertAlmostEqual(out.days[2]["nav"], 8175.7)
        self.assertTrue(any(o["reason"] == "halted" for o in out.orders))

    def test_limit_down_exit_keeps_position_then_sells(self):
        self.bars[(self.dates[1], "A")].update(open=9., close=9.)
        self.bars[(self.dates[2], "A")].update(open=9.5, close=9.5, pre_close=9.)
        out = self.run_case({**self.target, self.dates[0]: {}})
        self.assertEqual(out.days[1]["nav"], 9000.)
        self.assertEqual(out.days[2]["positions"], {})
        self.assertEqual(out.days[2]["nav"], 9500.)

    def test_limit_up_entry_retries_without_reranking(self):
        self.bars[(self.dates[0], "A")].update(open=11., close=11.)
        self.bars[(self.dates[1], "A")].update(open=10., close=10., pre_close=11.)
        out = self.run_case()
        self.assertEqual(out.days[0]["positions"], {})
        self.assertEqual(out.days[1]["positions"]["A"]["shares"], 1000)

    def test_expired_signal_cancels_pending_entry_not_existing_holding(self):
        self.bars[(self.dates[0], "A")].update(open=11., close=11.)
        self.bars[(self.dates[1], "A")].update(open=10., close=10., pre_close=11.)
        out = self.run_case({**self.target, self.dates[0]: None})
        self.assertEqual(out.journal, [])
        self.assertEqual(out.days[-1]['nav'], 10000.)
        self.bars[(self.dates[0], "A")].update(open=10., close=10.)
        self.bars[(self.dates[1], "A")]['pre_close'] = 10.
        out = self.run_case({**self.target, self.dates[0]: None})
        self.assertEqual(len(out.journal), 1)
        self.assertEqual(out.days[-1]['positions']['A']['shares'], 1000)

    def test_unknown_quote_or_delisting_is_not_zero(self):
        del self.bars[(self.dates[1], "A")]
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertEqual(len(out.days), 2)
        self.assertEqual(out.days[-1]["positions"]["A"]["shares"], 1000)
        self.assertNotEqual(out.days[-1]["nav"], out.days[-1]["nav"])

    def test_missing_limits_not_board_percentage_guess(self):
        self.bars[(self.dates[0], "A")].pop("up_limit")
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertEqual(out.journal, [])

    def test_missing_close_does_not_retroactively_change_open_fill(self):
        self.bars[(self.dates[0], "A")].pop("close")
        out = self.run_case()
        self.assertEqual(out.journal[0]["quantity"], 1000)
        self.assertEqual(out.status, "data_insufficient")

    def test_future_close_cannot_change_today_orders(self):
        before = self.run_case()
        self.bars[(self.dates[-1], "A")]["close"] = 100.
        after = self.run_case()
        self.assertEqual(before.journal, after.journal)
        self.assertEqual(before.days[:-1], after.days[:-1])

    def test_missing_event_blocks_held_and_pending_entry(self):
        self.bars[(self.dates[0], "A")]["action_required"] = True
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertEqual(out.journal, [])

    def test_reference_break_is_not_adjusted_return_shortcut(self):
        self.bars[(self.dates[1], "A")].update(pre_close=5., open=5., close=5.)
        out = self.run_case()
        self.assertIn("corporate_action_or_reference_break", {e["reason"] for e in out.errors})

    def test_reference_only_restructuring_changes_price_without_holder_shares(self):
        self.bars[(self.dates[1], "A")].update(pre_close=5., open=5., close=5.,
                                                action_required=True)
        self.bars[(self.dates[2], "A")].update(pre_close=5., open=5., close=5.)
        self.bars[(self.dates[3], "A")].update(pre_close=5., open=5., close=5.)
        out = self.run_case(reference_only_events={("A", self.dates[1])})
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual(out.days[1]["positions"]["A"]["shares"], 1000)
        self.assertEqual(out.days[1]["nav"], 5000.)

    def test_model_tax_and_adjacent_fractional_allotments(self):
        action = CorporateAction("e1", "A", self.dates[0], self.dates[1],
                                 self.dates[1], cash_per_share_net=1.,
                                 share_multiplier=1.001,
                                 new_shares_date=self.dates[1], evidence="fixture")
        targets = {"2024-12-31": {"A": .1}}
        floor_book = self.run_case(targets, actions=[action],
                                   fractional_share_policy="floor",
                                   gross_cash_tax_rate=.2)
        ceil_book = self.run_case(targets, actions=[action],
                                  fractional_share_policy="ceil",
                                  gross_cash_tax_rate=.2)
        self.assertEqual(floor_book.days[1]["positions"]["A"]["shares"], 100)
        self.assertEqual(ceil_book.days[1]["positions"]["A"]["shares"], 101)
        self.assertEqual(floor_book.days[1]["cash"], 9080.)
        self.assertEqual(ceil_book.days[1]["cash"], 9080.)

    def test_share_exchange_marks_receivable_then_delivers_listed_shares(self):
        for day in self.dates:
            self.bars[(day, "B")].update(open=100., close=100., pre_close=100.,
                                         up_limit=110., down_limit=90.)
        exchange = {"old_code": "A", "new_code": "B", "date": self.dates[2],
                    "ratio": .1, "evidence": "fixture notice"}
        out = self.run_case(
            delist_evidence={"A": {"date": self.dates[1], "evidence": "fixture notice"}},
            share_exchanges=[exchange])
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual(out.days[1]["positions"]["A"]["mark"], 10.)
        self.assertEqual(out.days[2]["positions"], {})
        self.assertEqual(len([e for e in out.journal if e["kind"] == "share_exchange_in"]), 1)

    def test_reacquisition_does_not_compare_with_former_holding_mark(self):
        self.bars[(self.dates[1], "A")]["close"] = 12.
        self.bars[(self.dates[2], "A")].update(pre_close=12., open=12., close=12.,
                                                up_limit=13., down_limit=11.)
        self.bars[(self.dates[3], "A")].update(pre_close=12., open=12., close=12.)
        out = self.run_case({"2024-12-31": {"A": 1.},
                             self.dates[0]: {}, self.dates[1]: {"A": 1.}})
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertGreater(out.days[2]["positions"]["A"]["shares"], 0)

    def test_dividend_receivable_and_locked_bonus_from_record_holdings(self):
        action = CorporateAction("e1", "A", self.dates[0], self.dates[1], self.dates[3],
                                 cash_per_share_net=1., share_multiplier=2.,
                                 new_shares_date=self.dates[3], evidence="synthetic-confirmed")
        for d in self.dates[1:]:
            self.bars[(d, "A")].update(open=4.5, close=4.5, pre_close=4.5, down_limit=4.)
        out = self.run_case({**self.target, self.dates[0]: {}}, actions=[action])
        self.assertEqual(out.days[1]["positions"]["A"]["shares"], 1000)
        self.assertEqual(out.days[1]["positions"]["A"]["locked"], 1000)
        self.assertEqual(out.days[1]["receivable"], 1000.)
        self.assertEqual(out.days[1]["nav"], 10000.)
        self.assertEqual(out.days[-1]["positions"], {})
        self.assertEqual(out.days[-1]["cash"], 10000.)

    def test_board_minimum_and_increment(self):
        self.assertEqual(LotRule(200, 1).round_order(199), 0)
        self.assertEqual(LotRule(200, 1).round_order(201.9), 201)
        self.assertEqual(LotRule(100, 100).round_order(201.9), 200)
        self.assertEqual(LotRule(100, 1).round_order(101.9), 101)

    def test_reconciler_catches_corrupted_cash_and_shares(self):
        out = self.run_case()
        broken = copy.deepcopy(out)
        broken.days[-1]["cash"] += 1
        with self.assertRaises(AssertionError):
            reconcile_journal(broken, 10000.)
        broken = copy.deepcopy(out)
        broken.days[-1]["positions"]["A"]["shares"] -= 1
        with self.assertRaises(AssertionError):
            reconcile_journal(broken, 10000.)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            run_ledger([], {}, {})
        with self.assertRaises(ValueError):
            self.run_case({"2024-12-31": {"A": 1.1}})
        with self.assertRaises(ValueError):
            self.run_case(actions=[CorporateAction("e", "A", self.dates[0], self.dates[1], self.dates[2])])

    def test_market_loader_mixed_dates_are_filtered_after_parsing(self):
        import pandas as pd
        path = Path(__file__).resolve().parents[3] / 'scripts/fuxi_execution_lab_20260923.py'
        spec = importlib.util.spec_from_file_location('execution_test_harness', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = pd.DataFrame({'ts_code': ['A'] * 3,
                               'trade_date': ['20241227', '2025-01-02', '2026-09-21'],
                               'close': [10., 11., 12.]})
        with patch('pandas.read_csv', return_value=iter([source])):
            frame, inventory = module.load_panel('fixture.csv', ['ts_code', 'trade_date', 'close'])
        self.assertEqual(frame.trade_date.tolist(), ['2024-12-27', '2025-01-02'])
        self.assertEqual(inventory['source_max'], '20260921')


    def duplicate_pair(self, **overrides):
        """Two names for one entitlement, differing only where asked."""
        specs = [("a1", overrides.get("first_schedule", "")),
                 ("a2", overrides.get("second_schedule", ""))]
        return tuple(
            CorporateAction(event_id=name, code="A", record_date=self.dates[0],
                            ex_date=self.dates[1], pay_date=self.dates[2],
                            cash_per_share_net=overrides.get("cash", 0.5),
                            share_multiplier=overrides.get("multiplier", 1.0),
                            new_shares_date=schedule, evidence=overrides.get("evidence", "fixture"))
            for name, schedule in specs)

    def test_duplicate_same_day_events_block_by_default(self):
        first, second = self.duplicate_pair()
        with self.assertRaises(ValueError):
            self.run_case(actions=[first, second])

    def test_collapse_identical_grants_one_entitlement(self):
        first, second = self.duplicate_pair()
        out = self.run_case(actions=[first, second], duplicate_event_policy="collapse_identical")
        self.assertEqual([row["kept_id"] for row in out.collapsed_events], ["a1"])
        self.assertEqual(out.collapsed_events[0]["collapsed_ids"], ["a1", "a2"])
        payments = [entry for entry in out.journal if entry["kind"] == "payment"]
        self.assertEqual(len(payments), 1)
        self.assertAlmostEqual(payments[0]["cash_delta"], 500.)
        self.assertAlmostEqual(out.days[-1]["nav"], 10500.)

    def test_collapse_never_merges_differing_entitlements(self):
        first, second = self.duplicate_pair(cash=0.5)
        second = CorporateAction("a2", "A", self.dates[0], self.dates[1], self.dates[2],
                                 cash_per_share_net=0.7, evidence="fixture")
        with self.assertRaises(ValueError):
            self.run_case(actions=[first, second], duplicate_event_policy="collapse_identical")

    def test_collapse_tolerates_inert_schedule_difference(self):
        first, second = self.duplicate_pair(second_schedule=self.dates[1])
        out = self.run_case(actions=[first, second], duplicate_event_policy="collapse_identical")
        self.assertEqual(out.collapsed_events[0]["differing_fields"], ["new_shares_date"])
        self.assertEqual(len(out.collapsed_events), 1)

    def test_collapse_refuses_differing_delivery_when_shares_issued(self):
        first = CorporateAction("a1", "A", self.dates[0], self.dates[1], self.dates[2],
                                share_multiplier=1.5, new_shares_date=self.dates[1],
                                evidence="fixture")
        second = CorporateAction("a2", "A", self.dates[0], self.dates[1], self.dates[2],
                                 share_multiplier=1.5, new_shares_date=self.dates[2],
                                 evidence="fixture")
        with self.assertRaises(ValueError):
            self.run_case(actions=[first, second], duplicate_event_policy="collapse_identical")

    def test_distinct_day_events_are_untouched_by_collapse(self):
        first, _ = self.duplicate_pair()
        later = CorporateAction("a3", "A", self.dates[1], self.dates[2], self.dates[3],
                                cash_per_share_net=0.2, evidence="fixture")
        out = self.run_case(actions=[first, later], duplicate_event_policy="collapse_identical")
        self.assertEqual(out.collapsed_events, [])
        self.assertEqual(len([e for e in out.journal if e["kind"] == "entitlement"]), 2)

    def test_invalid_duplicate_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(duplicate_event_policy="merge_all")


    def test_defer_refuses_execution_when_limits_are_missing(self):
        bar = self.bars[(self.dates[0], "A")]
        bar.pop("up_limit")
        bar.pop("down_limit")
        blocked = self.run_case()
        self.assertEqual(blocked.status, "data_insufficient")
        self.assertEqual(blocked.errors[0]["reason"], "missing_exact_limits")
        deferred = self.run_case(missing_limit_policy="defer")
        self.assertEqual(deferred.status, "complete_under_declared_assumptions")
        self.assertEqual(deferred.errors, [])
        first = [o for o in deferred.orders if o["date"] == self.dates[0]]
        self.assertEqual([o["reason"] for o in first], ["no_limit_evidence"])
        self.assertEqual(first[0]["filled"], 0)
        self.assertEqual(deferred.days[0]["positions"], {})
        self.assertAlmostEqual(deferred.days[0]["cash"], 10000.)

    def test_defer_retries_the_pending_target_on_the_next_session(self):
        bar = self.bars[(self.dates[0], "A")]
        bar.pop("up_limit")
        bar.pop("down_limit")
        out = self.run_case(missing_limit_policy="defer")
        self.assertEqual(out.orders[0]["reason"], "no_limit_evidence")
        fills = [e for e in out.journal if e["kind"] == "fill"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["date"], self.dates[1])
        self.assertEqual(fills[0]["quantity"], 1000)
        self.assertLess(fills[0]["signal_date"], fills[0]["date"])

    def test_invalid_missing_limit_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(missing_limit_policy="invent")


    def test_partial_halt_refuses_a_new_open_entry(self):
        # 600804.SH 2025-06-10: an intraday suspension covered the open. A new
        # entry cannot be priced at an open that the halt hid.
        self.bars[(self.dates[0], "A")].update(state="partial_halt", open=0.31, close=10.)
        out = self.run_case()
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual(out.orders[0]["reason"], "partial_halt_open")
        self.assertEqual(out.orders[0]["filled"], 0)
        fills = [e for e in out.journal if e["kind"] == "fill"]
        self.assertEqual([f["date"] for f in fills], [self.dates[1]])

    def test_partial_halt_marks_a_held_position_from_its_close(self):
        # The same session still printed a close, so a held position keeps a real
        # mark instead of losing the session.
        self.bars[(self.dates[1], "A")].update(state="partial_halt", open=0.31,
                                               close=12., pre_close=10.)
        for later in self.dates[2:]:
            self.bars[(later, "A")].update(close=12., pre_close=12., up_limit=13.)
        out = self.run_case()
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual(out.errors, [])
        self.assertAlmostEqual(out.days[1]["nav"], 12000.)
        self.assertAlmostEqual(out.days[-1]["nav"], 12000.)

    def test_partial_halt_without_a_close_still_fails_closed(self):
        bar = self.bars[(self.dates[1], "A")]
        bar.update(state="partial_halt")
        bar.pop("close")
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertEqual(out.errors[0]["reason"], "partial_halt_without_quote")

    def test_unknown_held_state_still_blocks(self):
        self.bars[(self.dates[1], "A")].update(state="unknown")
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertEqual(out.errors[0]["reason"], "unknown_held_state")

    def test_full_day_halt_without_an_action_row_blocks_by_default(self):
        # A published full-day halt carries no quote and, in the vendor panel, no
        # adjustment row either. Left at the default the unreadable check stops
        # the arm, which is the strict direction.
        self.bars[(self.dates[1], "A")] = {"state": "halted",
                                           "evidence": "published_full_day_S"}
        out = self.run_case()
        self.assertEqual(out.status, "data_insufficient")
        self.assertIn("missing_action_check", {e["reason"] for e in out.errors})
        self.assertNotIn("halted_session_check_vacuous", {e["kind"] for e in out.journal})

    def test_full_day_halt_check_is_vacuous_only_when_opted_in(self):
        self.bars[(self.dates[1], "A")] = {"state": "halted",
                                           "evidence": "published_full_day_S"}
        out = self.run_case(halted_session_check_policy="vacuous")
        self.assertEqual(out.status, "complete_under_declared_assumptions")
        self.assertEqual(out.errors, [])
        relaxed = [e for e in out.journal if e["kind"] == "halted_session_check_vacuous"]
        self.assertEqual(len(relaxed), 1)
        self.assertEqual(relaxed[0]["evidence"], "published_full_day_S")
        self.assertFalse(relaxed[0]["settled_by_supplied_action"])
        self.assertEqual(out.days[1]["positions"]["A"]["shares"], 1000)
        self.assertEqual(out.days[1]["nav"], out.days[0]["nav"])

    def test_unreadable_action_check_on_a_trading_session_is_never_relaxed(self):
        self.bars[(self.dates[1], "A")]["action_check_available"] = False
        out = self.run_case(halted_session_check_policy="vacuous")
        self.assertEqual(out.status, "data_insufficient")
        self.assertIn("missing_action_check", {e["reason"] for e in out.errors})
        self.assertNotIn("halted_session_check_vacuous", {e["kind"] for e in out.journal})

    def test_unevidenced_halt_is_never_relaxed(self):
        self.bars[(self.dates[1], "A")] = {"state": "halted"}
        out = self.run_case(halted_session_check_policy="vacuous")
        self.assertEqual(out.status, "data_insufficient")
        self.assertIn("missing_action_check", {e["reason"] for e in out.errors})

    def test_invalid_halted_session_check_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(halted_session_check_policy="relax")

    def test_book_separates_an_evidenced_partial_halt_from_a_conflict(self):
        path = Path(__file__).resolve().parents[3] / 'scripts/fuxi_window_replay_20260923.py'
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('window_replay_book', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quote = {('2025-06-10', 'A'): {'open': 0.31, 'close': 0.23, 'pre_close': 0.30}}
        day = '2025-06-10'
        strict = module.Book(quote, {}, {}, set(), {(day, 'A')})
        self.assertEqual(strict.get((day, 'A'))['state'], 'unknown')
        self.assertNotIn('evidence', strict.get((day, 'A')))
        opted = module.Book(quote, {}, {}, set(), {(day, 'A')}, partial_halt_state=True)
        bar = opted.get((day, 'A'))
        self.assertEqual(bar['state'], 'partial_halt')
        self.assertEqual(bar['evidence'], 'intraday_suspension_window_with_published_quote')
        # A full-day suspension alongside a quote stays a source disagreement.
        clash = module.Book(quote, {}, {}, {(day, 'A')}, set(), partial_halt_state=True)
        self.assertEqual(clash.get((day, 'A'))['state'], 'unknown')


if __name__ == "__main__":
    unittest.main()
