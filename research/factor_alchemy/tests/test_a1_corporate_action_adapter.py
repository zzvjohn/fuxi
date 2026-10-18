import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a1_corporate_action_adapter import adjudicate_bundle


class A1BundleTests(unittest.TestCase):
    def test_two_cash_rows_are_added_only_with_issuer_notice(self):
        shared = dict(ts_code="300760.SZ", div_proc="实施", ex_date="20250529",
                      record_date="20250528", pay_date="20250529",
                      div_listdate=None, stk_div=0.0, stk_bo_rate=None,
                      stk_co_rate=None, imp_ann_date="20250522")
        rows = [dict(shared, end_date="20241231", cash_div=.56, cash_div_tax=.56),
                dict(shared, end_date="20250331", cash_div=1.41, cash_div_tax=1.41)]
        market = dict(code="300760.SZ", ex_date="2025-05-29",
                      pre_close=230.51, adj_prev=1.0, adj_ex=230.51 / 228.54)
        self.assertEqual(adjudicate_bundle(rows, **market).reason,
                         "unproven_same_day_aggregation")
        notice = dict(code="300760.SZ", ex_date="2025-05-29",
                      notice_date="2025-05-22", notice_url="https://issuer.example/notice",
                      reason="two_same_day_cash_implementations_0.56_plus_1.41",
                      holder_cash_pre_tax=1.97, holder_share_rate=0.0,
                      reference_cash=1.97, reference_share_rate=0.0)
        result = adjudicate_bundle(rows + [dict(rows[0])], issuer_exception=notice, **market)
        self.assertEqual(result.status, "approved_for_gross_cash_stress_only")
        self.assertEqual(result.detail["economically_unique_implementations"], 2)
        self.assertAlmostEqual(result.action["cash_per_share_net"], 1.97)
        self.assertFalse(result.detail["account_cash_receipt_known"])

    def test_differential_reference_and_holder_rights_stay_separate(self):
        row = dict(ts_code="300100.SZ", div_proc="实施", ex_date="20250515",
                   record_date="20250514", pay_date="20250515", div_listdate="20250515",
                   stk_div=.4, cash_div=.2, cash_div_tax=.2)
        market = dict(code="300100.SZ", ex_date="2025-05-15",
                      pre_close=76.92, adj_prev=1.0, adj_ex=76.92 / 54.94)
        self.assertEqual(adjudicate_bundle([row], **market).reason,
                         "reference_identity_failed")
        notice = dict(code="300100.SZ", ex_date="2025-05-15",
                      notice_date="2025-05-09", notice_url="https://issuer.example/notice",
                      reason="repurchased_shares_excluded_from_distribution",
                      holder_cash_pre_tax=.2, holder_share_rate=.4,
                      reference_cash=.1982171, reference_share_rate=.3964343)
        result = adjudicate_bundle([row], issuer_exception=notice, **market)
        self.assertEqual(result.status, "approved_for_gross_cash_stress_only")
        self.assertAlmostEqual(result.action["cash_per_share_net"], .2)
        self.assertAlmostEqual(result.action["share_multiplier"], 1.4)
        self.assertAlmostEqual(result.detail["reference_cash"], .1982171)
        self.assertAlmostEqual(result.detail["reference_share_rate"], .3964343)
        self.assertEqual(adjudicate_bundle([row], issuer_exception={**notice,
            "holder_share_rate": .3964343}, **market).reason,
            "issuer_holder_rights_disagree_with_vendor")

    def test_pure_transfer_requires_issuer_no_cash_statement(self):
        row = dict(ts_code="300803.SZ", div_proc="实施", ex_date="20250317",
                   record_date="20250314", pay_date=None, div_listdate="20250317",
                   stk_div=.45, cash_div=None, cash_div_tax=None)
        market = dict(code="300803.SZ", ex_date="2025-03-17",
                      pre_close=92.86, adj_prev=1.0, adj_ex=92.86 / 64.04)
        self.assertEqual(adjudicate_bundle([row], **market).reason,
                         "missing_pre_tax_cash_component")
        notice = dict(code="300803.SZ", ex_date="2025-03-17",
                      notice_date="2025-03-07", notice_url="https://issuer.example/notice",
                      reason="issuer_explicit_no_cash_pure_transfer",
                      holder_cash_pre_tax=0.0, holder_share_rate=.45,
                      reference_cash=0.0, reference_share_rate=.45)
        result = adjudicate_bundle([row], issuer_exception=notice, **market)
        self.assertEqual(result.status, "approved_for_gross_cash_stress_only")
        self.assertEqual(result.action["cash_per_share_net"], 0.0)

    def test_restructuring_capital_increase_never_grants_old_holder_shares(self):
        row = dict(ts_code="002822.SZ", div_proc="实施", ex_date="20251230",
                   record_date="20251229", pay_date=None,
                   div_listdate="20251230", stk_div=1.031,
                   cash_div=None, cash_div_tax=None)
        notice = dict(code="002822.SZ", ex_date="2025-12-30",
                      notice_date="2025-12-23",
                      notice_url="https://disc.static.szse.cn/issuer-notice.pdf",
                      reason="restructuring_shares_not_distributed_to_existing_holders",
                      holder_cash_pre_tax=0.0, holder_share_rate=0.0)
        result = adjudicate_bundle(
            [row], code="002822.SZ", ex_date="2025-12-30",
            pre_close=4.07, adj_prev=1.0, adj_ex=1.0,
            issuer_exception=notice)
        self.assertEqual(result.status, "not_holder_distribution")
        self.assertIsNone(result.action)
        invalid = adjudicate_bundle(
            [row], code="002822.SZ", ex_date="2025-12-30",
            pre_close=4.07, adj_prev=1.0, adj_ex=1.0,
            issuer_exception={**notice, "holder_share_rate": 1.031})
        self.assertEqual(invalid.reason, "invalid_non_holder_rights")


if __name__ == "__main__":
    unittest.main()
