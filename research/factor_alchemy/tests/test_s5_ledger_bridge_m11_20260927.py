"""M11 定向测试：S5 固定股数账本桥接 + 代理口径复刻 (2026-09-27)。

契约 (reports/fuxi_alpha_generation_overhaul_20260926.md §M11):
  1. FUXI_S5_LEDGER = proxy(默认)|fixed_shares; 非法值 fail-fast;
  2. proxy_book 复刻 S5 enforce 语义: 日收益 = close/pre_close-1;
     缺报价**不缩分母不补零** (整组 None); 板块 adj 守卫; 成本在首个有效日扣;
  3. fixed_shares_book fail-closed: 未闭合 ⇒ data_insufficient + 缺数可枚举;
     complete ⇒ 日收益/费用/五态可读; 任何路径 formal_eligibility=False;
  4. compare_books: 逐日 NAV 差 / 费用差 / 五态分类;
  5. 未决陈旧 mark 不授 valid (账本不变量, 经本桥接原样传递)。
"""
import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

import s5_ledger_bridge as B


def _bar(open_=None, close=None, pre=None, state="trading", avail=True,
         required=False, up=None, down=None):
    d = {"state": state, "action_check_available": avail,
         "action_required": required}
    if open_ is not None:
        d["open"] = open_
    if close is not None:
        d["close"] = close
    if pre is not None:
        d["pre_close"] = pre
    if up is not None:
        d["up_limit"] = up
    if down is not None:
        d["down_limit"] = down
    return d


class LedgerModeTests(unittest.TestCase):

    def tearDown(self):
        os.environ.pop("FUXI_S5_LEDGER", None)

    def test_default_proxy(self):
        os.environ.pop("FUXI_S5_LEDGER", None)
        self.assertEqual(B.ledger_mode(), "proxy")

    def test_override(self):
        os.environ["FUXI_S5_LEDGER"] = "fixed_shares"
        self.assertEqual(B.ledger_mode(), "fixed_shares")

    def test_invalid_fail_fast(self):
        os.environ["FUXI_S5_LEDGER"] = "ledger"
        with self.assertRaises(ValueError):
            B.ledger_mode()


class ProxyBookTests(unittest.TestCase):

    DAYS = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]

    def _bars(self, a_0103=(10.2, 10.0), b_0103=(20.4, 20.0)):
        return {
            ("2025-01-03", "A"): _bar(close=a_0103[0], pre=a_0103[1]),
            ("2025-01-03", "B"): _bar(close=b_0103[0], pre=b_0103[1]),
            ("2025-01-06", "A"): _bar(close=10.0, pre=10.2),
            ("2025-01-06", "B"): _bar(close=20.0, pre=20.4),
            ("2025-01-07", "A"): _bar(close=10.1, pre=10.0),
            ("2025-01-07", "B"): _bar(close=20.2, pre=20.0),
        }

    def test_two_periods_cost_and_nav(self):
        targets = {"2025-01-02": {"A": 0.5, "B": 0.5},
                   "2025-01-06": {"A": 0.5, "B": 0.5},
                   "2025-01-07": {"A": 0.5, "B": 0.5}}
        book = B.proxy_book(self.DAYS, self._bars(), targets, top_n=2,
                            cost_bps=15.0)
        self.assertEqual(book["status"], "proxy_ok")
        self.assertEqual(book["n_periods"], 2)
        self.assertAlmostEqual(book["total_cost_frac"], 0.0015, places=12)
        rets = dict(book["daily_returns"])
        # 期1: 01-03 = +2% - 0.15% (成本首个有效日); 01-06 = -1.9608%
        self.assertAlmostEqual(rets["2025-01-03"], 0.02 - 0.0015, places=10)
        self.assertAlmostEqual(rets["2025-01-06"],
                               (10.0 / 10.2 + 20.0 / 20.4) / 2 - 1.0, places=10)
        # 期2: 01-07 = +1% (成员未换, 成本 0)
        self.assertAlmostEqual(rets["2025-01-07"], 0.01, places=10)
        expect = (1 + 0.0185) * (10.0 / 10.2) * 1.01
        self.assertAlmostEqual(book["final_nav"], expect, places=10)
        self.assertFalse(book["formal_eligibility"])

    def test_missing_quote_not_shrunk(self):
        bars = self._bars()
        del bars[("2025-01-03", "B")]          # B 缺报价 ⇒ 整组不产出
        targets = {"2025-01-02": {"A": 0.5, "B": 0.5},
                   "2025-01-06": {"A": 0.5, "B": 0.5}}
        book = B.proxy_book(self.DAYS, bars, targets, top_n=2, cost_bps=15.0)
        rets = dict(book["daily_returns"])
        self.assertIsNone(rets["2025-01-03"])           # 不得按 1/2 缩分母
        # 成本顺延到首个有效日 (01-06)
        self.assertAlmostEqual(rets["2025-01-06"],
                               (10.0 / 10.2 + 20.0 / 20.4) / 2 - 1.0 - 0.0015,
                               places=10)

    def test_adj_guard_discards_beyond_board_limit(self):
        bars = self._bars(a_0103=(13.0, 10.0))   # 主板 +30% ⇒ 守卫丢弃
        targets = {"2025-01-02": {"A": 0.5, "B": 0.5},
                   "2025-01-06": {"A": 0.5, "B": 0.5}}
        book = B.proxy_book(self.DAYS, bars, targets, top_n=2, cost_bps=15.0)
        rets = dict(book["daily_returns"])
        self.assertIsNone(rets["2025-01-03"])   # A 被丢弃 ⇒ 有效数 <2 ⇒ None

    def test_board_limit_prefixes(self):
        self.assertAlmostEqual(B.board_limit("600000.SH"), 0.10)
        self.assertAlmostEqual(B.board_limit("300750.SZ"), 0.20)
        self.assertAlmostEqual(B.board_limit("688981.SH"), 0.20)
        self.assertAlmostEqual(B.board_limit("830799.BJ"), 0.30)
        self.assertAlmostEqual(B.board_limit("430047.BJ"), 0.30)

    def test_entry_mode_next_open(self):
        days = ["2025-01-02", "2025-01-03", "2025-01-06"]
        bars = {
            ("2025-01-03", "A"): _bar(open_=10.1, close=10.2, pre=10.0),
            ("2025-01-06", "A"): _bar(open_=10.2, close=10.2, pre=10.2),
        }
        targets = {"2025-01-02": {"A": 1.0}, "2025-01-06": {"A": 1.0}}
        sc = B.proxy_book(days, bars, targets, top_n=1, cost_bps=15.0)
        no = B.proxy_book(days, bars, targets, top_n=1, cost_bps=15.0,
                          entry_mode="next_open")
        self.assertEqual(sc["entry_mode"], "signal_close")
        self.assertAlmostEqual(dict(sc["daily_returns"])["2025-01-03"],
                               0.02 - 0.0015, places=10)
        self.assertAlmostEqual(dict(no["daily_returns"])["2025-01-03"],
                               10.2 / 10.1 - 1 - 0.0015, places=10)
        with self.assertRaises(ValueError):
            B.proxy_book(days, bars, targets, top_n=1, entry_mode="close")


class FixedSharesBookTests(unittest.TestCase):

    DAYS = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]

    def _bars(self, bad_0106=False):
        limits = dict(up=11.0, down=9.0)
        bars = {
            ("2025-01-02", "600000.SH"): _bar(open_=10.0, close=10.0, pre=10.0,
                                              **limits),
            ("2025-01-03", "600000.SH"): _bar(open_=10.1, close=10.2, pre=10.0,
                                              **limits),
            ("2025-01-06", "600000.SH"): _bar(open_=10.2, close=10.1, pre=10.2,
                                              **limits),
            ("2025-01-07", "600000.SH"): _bar(open_=10.1, close=10.4, pre=10.1,
                                              **limits),
        }
        if bad_0106:
            bars[("2025-01-06", "600000.SH")] = _bar(state="unknown")
        return bars

    def test_complete_book_with_fees(self):
        targets = {"2025-01-02": {"600000.SH": 0.5}}
        book = B.fixed_shares_book(self.DAYS, self._bars(), targets,
                                   initial_cash=1_000_000.0, cost_bps=15.0)
        self.assertEqual(book["status"],
                         "book_complete_under_declared_assumptions")
        self.assertEqual(book["n_fills"], 1)
        # 目标股数 = floor(1e6*0.5/(1.0015)/10 /100)*100 = 49,900
        fee = 49_900 * 10.1 * 15 / 10000
        self.assertAlmostEqual(book["total_fees"], fee, places=6)
        nav = (1_000_000.0 - 49_900 * 10.1 - fee) + 49_900 * 10.4
        self.assertAlmostEqual(book["final_nav"], nav, places=6)
        self.assertTrue(len(book["daily_returns"]) == 4)
        self.assertFalse(book["formal_eligibility"])

    def test_incomplete_enumerates_missing(self):
        targets = {"2025-01-02": {"600000.SH": 0.5}}
        book = B.fixed_shares_book(self.DAYS, self._bars(bad_0106=True),
                                   targets,
                                   initial_cash=1_000_000.0, cost_bps=15.0)
        self.assertEqual(book["status"], "data_insufficient")
        self.assertIsNone(book["daily_returns"])
        missing = book["missing"]
        self.assertGreater(missing["n_errors"], 0)
        self.assertTrue(missing["error_counts"])
        self.assertTrue(missing["error_samples"])
        self.assertTrue(book["reasons"])

    def test_unresolved_reference_break_fails_closed(self):
        bars = self._bars()
        # 除权式前收断裂 (pre_close 10.0 → 9.0) 且未供 action
        bars[("2025-01-06", "600000.SH")] = _bar(open_=9.9, close=9.8,
                                                 pre=9.0, **{"up": 11.0, "down": 9.0})
        targets = {"2025-01-02": {"600000.SH": 0.5}}
        book = B.fixed_shares_book(self.DAYS, bars, targets,
                                   initial_cash=1_000_000.0, cost_bps=15.0)
        self.assertEqual(book["status"], "data_insufficient")
        counts = book["missing"]["error_counts"]
        self.assertTrue(any("corporate_action" in (k or "") or "break" in (k or "")
                            for k in counts), counts)

    def test_include_ledger_object(self):
        targets = {"2025-01-02": {"600000.SH": 0.5}}
        book = B.fixed_shares_book(self.DAYS, self._bars(), targets,
                                   initial_cash=1_000_000.0,
                                   include_ledger=True)
        self.assertIn("ledger_object", book)
        self.assertEqual(len(book["ledger_object"].days), 4)


class CompareTests(unittest.TestCase):

    def test_compare_and_five_state(self):
        proxy = {"status": "proxy_ok", "weighting_contract": "w1",
                 "nav_by_day": {"2025-01-02": 1.0, "2025-01-03": 1.02},
                 "total_cost_frac": 0.0015}
        ledger = {"status": "book_complete_under_declared_assumptions",
                  "weighting_contract": "w2",
                  "nav_by_day": {"2025-01-02": 1.0, "2025-01-03": 1.015},
                  "total_fees_frac": 0.0008, "reasons": []}
        out = B.compare_books(proxy, ledger, label="t")
        self.assertEqual(out["n_days_compared"], 2)
        self.assertAlmostEqual(out["max_abs_nav_diff"], 0.005, places=12)
        self.assertAlmostEqual(out["final_nav_diff"], 0.005, places=12)
        self.assertAlmostEqual(out["cost"]["diff_frac"], 0.0007, places=12)
        self.assertEqual(out["five_state"],
                         {"proxy": "valid", "ledger": "valid"})

    def test_five_state_mapping(self):
        self.assertEqual(B.classify_five_state({"status": "proxy_ok"}), "valid")
        self.assertEqual(
            B.classify_five_state(
                {"status": "book_complete_under_declared_assumptions"}),
            "valid")
        self.assertEqual(
            B.classify_five_state({"status": "data_insufficient"}),
            "data_insufficient")
        self.assertEqual(B.classify_five_state({"status": "error"}), "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
