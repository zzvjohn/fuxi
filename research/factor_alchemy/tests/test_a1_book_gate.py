import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio_execution_ledger import (
    assess_economic_book, run_ledger,
)


class EconomicBookGateTests(unittest.TestCase):
    def setUp(self):
        self.days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        self.bars = {(day, "A"): {
            "state": "trading", "open": 10.0, "close": 10.0,
            "pre_close": 10.0, "up_limit": 11.0, "down_limit": 9.0,
            "action_check_available": True,
        } for day in ["2024-12-31", *self.days]}
        self.targets = {"2024-12-31": {"A": 1.0}}

    def book(self, **kwargs):
        return run_ledger(self.days, self.bars, self.targets,
                          initial_cash=10_000.0, cost_bps=15.0, **kwargs)

    def test_complete_arithmetic_book_still_has_no_independent_validation(self):
        verdict = assess_economic_book(self.book(), self.days, 10_000.0)
        self.assertEqual(verdict["status"], "book_complete_under_declared_assumptions")
        self.assertEqual(verdict["parity"]["finite_nav_days"], 3)
        self.assertFalse(verdict["independent_execution_or_data_validation"])
        self.assertFalse(verdict["formal_eligibility"])

    def test_missing_event_and_incomplete_window_fail_closed(self):
        self.bars[(self.days[1], "A")]["action_required"] = True
        verdict = assess_economic_book(self.book(), self.days, 10_000.0)
        self.assertEqual(verdict["status"], "data_insufficient")
        self.assertIn("incomplete_window", verdict["reasons"])
        self.assertIn("ledger_data_insufficient", verdict["reasons"])

    def test_stale_exclusion_never_becomes_economic_label(self):
        self.bars[(self.days[1], "A")].update(
            action_required=True, pre_close=9.0, close=9.0, open=9.0)
        self.bars[(self.days[2], "A")].update(
            pre_close=9.0, close=9.0, open=9.0)
        result = self.book(unresolved_action_policy="exclude_held_security_day")
        self.assertEqual(len(result.days), 3)
        self.assertEqual(result.status, "complete_under_declared_assumptions")
        verdict = assess_economic_book(result, self.days, 10_000.0)
        self.assertEqual(verdict["status"], "data_insufficient")
        self.assertIn("unresolved_corporate_action_or_stale_mark", verdict["reasons"])


if __name__ == "__main__":
    unittest.main()
