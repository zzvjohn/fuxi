import copy
import unittest

from fuxi_a6_validation_gate_20260926 import assess_candidate_freeze, compare_execution_books


H = "a" * 64


def freeze():
    return {"candidate_id": "candidate-1", "frozen_at": "2026-09-26T08:00:00+08:00",
            "freeze_data_ceiling": "2026-09-25", "first_signal_date": "2026-09-28",
            "first_signal_cutoff_at": "2026-09-28T15:00:00+08:00",
            "formula_sha256": H, "plan_sha256": H, "code_sha256": H,
            "evaluation_contract_sha256": H, "independent_book_sha256": H,
            "economic_label_status": "valid", "research_exposed": False,
            "sections_required": 2, "selection_changed_after_freeze": False,
            "section_schedule": [{"signal_date": "2026-09-28", "label_matured_on": "2026-10-06"},
                                 {"signal_date": "2026-10-09", "label_matured_on": "2026-10-19"}]}


def sections():
    return [{"candidate_id": "candidate-1", "signal_date": "2026-09-28",
             "label_matured_on": "2026-10-06", "data_as_of": "2026-10-06",
             "evaluation_contract_sha256": H, "economic_status": "valid",
             "independent_book_sha256": H, "access_log_sequence": 3},
            {"candidate_id": "candidate-1", "signal_date": "2026-10-09",
             "label_matured_on": "2026-10-19", "data_as_of": "2026-10-19",
             "evaluation_contract_sha256": H, "economic_status": "valid",
             "independent_book_sha256": H, "access_log_sequence": 4}]


def books():
    contract = {"candidate_id": "candidate-1", "strategy_sha256": H,
                "calendar_sha256": H, "cost_bps": 15, "start_date": "2026-09-28",
                "end_date": "2026-09-29", "session_dates": ["2026-09-28", "2026-09-29"],
                "initial_cash": 1000.0, "price_adjustment": "raw_with_actions",
                "execution_timing": "next_open"}
    base = {k: v for k, v in contract.items() if k != "session_dates"}
    base.update({"daily": [{"date": "2026-09-28", "cash": 499.25, "receivables": 0,
                             "fees": 0.75, "nav": 999.25},
                            {"date": "2026-09-29", "cash": 509.25, "receivables": 0,
                             "fees": 0.75, "nav": 1009.25}],
                 "trades": [{"date": "2026-09-28", "code": "600000.SH", "side": "buy",
                             "sequence": 1, "shares": 50, "price": 10, "fee": 0.75,
                             "cash_delta": -500.75}],
                 "positions": [{"date": "2026-09-28", "code": "600000.SH", "shares": 50,
                                "mark_price": 10, "market_value": 500},
                               {"date": "2026-09-29", "code": "600000.SH", "shares": 50,
                                "mark_price": 10, "market_value": 500}]})
    local, jq = copy.deepcopy(base), copy.deepcopy(base)
    local.update(engine="local_fixed_share", run_id="local-1")
    jq.update(engine="jq_platform", run_id="jq-2")
    return local, jq, contract


class FutureGateTests(unittest.TestCase):
    def test_valid_synthetic_two_sections_only_ready_for_adjudication(self):
        result = assess_candidate_freeze(freeze(), "2026-10-19", sections())
        self.assertEqual(result["status"], "ready_for_independent_adjudication")
        self.assertEqual(result["matured_sections"], 2)

    def test_existing_a1_data_insufficient_cannot_enter_future_verdict(self):
        f = freeze()
        f["economic_label_status"] = "data_insufficient"
        self.assertEqual(assess_candidate_freeze(f, "2026-10-19", sections())["status"], "precheck_rejected")

    def test_unmatured_or_unlogged_future_rejected(self):
        rows = sections()
        rows[1]["access_log_sequence"] = None
        self.assertEqual(assess_candidate_freeze(freeze(), "2026-10-18", rows)["status"], "precheck_rejected")

    def test_changed_selection_restarts_clock(self):
        f = freeze()
        f["selection_changed_after_freeze"] = True
        self.assertIn("selection_change_requires_new_freeze",
                      assess_candidate_freeze(f, "2026-10-19", sections())["reasons"])

    def test_freezing_after_signal_cutoff_is_not_preregistration(self):
        f = freeze()
        f["frozen_at"] = "2026-09-28T15:01:00+08:00"
        self.assertIn("freeze_not_before_first_signal_cutoff",
                      assess_candidate_freeze(f, "2026-10-19", sections())["reasons"])

    def test_future_count_cannot_shrink_to_available_rows(self):
        result = assess_candidate_freeze(freeze(), "2026-10-19", sections()[:1])
        self.assertEqual((result["status"], result["sections_required"]), ("not_yet_matured", 2))

    def test_losing_a_bad_week_does_not_make_later_week_count(self):
        result = assess_candidate_freeze(freeze(), "2026-10-19", sections()[1:])
        self.assertIn("section_not_next_in_frozen_schedule", result["reasons"])

    def test_full_books_structurally_match(self):
        result = compare_execution_books(*books())
        self.assertEqual((result["status"], result["compared_days"]), ("structural_parity_pass", 2))

    def test_missing_day_is_not_parity(self):
        local, jq, contract = books()
        jq["daily"].pop()
        self.assertEqual(compare_execution_books(local, jq, contract)["status"], "data_insufficient")

    def test_self_asserted_parity_does_not_override_mismatched_shares(self):
        local, jq, contract = books()
        jq["parity_passed"] = True
        jq["positions"][0]["shares"] = 40
        self.assertEqual(compare_execution_books(local, jq, contract)["status"], "parity_failed")

    def test_same_engine_or_run_id_is_not_independent(self):
        local, jq, contract = books()
        jq["run_id"] = local["run_id"]
        self.assertEqual(compare_execution_books(local, jq, contract)["status"], "precheck_rejected")

    def test_identical_but_wrong_nav_is_not_parity(self):
        local, jq, contract = books()
        local["daily"][0]["nav"] = 1000
        jq["daily"][0]["nav"] = 1000
        self.assertIn("local:nav_arithmetic_mismatch",
                      compare_execution_books(local, jq, contract)["reasons"])


if __name__ == "__main__":
    unittest.main()
