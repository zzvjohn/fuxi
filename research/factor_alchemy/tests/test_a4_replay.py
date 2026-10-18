import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
sys.path.insert(0, str(ROOT / "scripts"))
from fuxi_a4_mechanism_replay_20260926 import replay_complete_matrix


class A4ReplayTests(unittest.TestCase):
    def setUp(self):
        self.plans = [{"plan_id": pid, "search_axes": {
            "event": pid, "context": pid, "quality": pid,
            "direction": pid, "output": pid}}
            for pid in ("a", "b")]

    def row(self, decision, maturity):
        return {"decision_at": decision, "outcomes": {
            "a": {"status": "valid", "verified_trading_label": True,
                  "economic_lcb_bps": 2.0,
                  "mature_at": maturity},
            "b": {"status": "valid", "verified_trading_label": True,
                  "economic_lcb_bps": -1.0,
                  "mature_at": maturity},
        }}

    def test_missing_counterfactual_rejected(self):
        row = self.row("2026-01-01T10:00:00+08:00",
                       "2026-01-02T10:00:00+08:00")
        row["outcomes"].pop("b")
        with self.assertRaisesRegex(ValueError, "complete outcome matrix"):
            replay_complete_matrix(self.plans, [row], seed=17)

    def test_future_result_hidden_until_mature(self):
        first = self.row("2026-01-01T10:00:00+08:00",
                         "2026-01-05T10:00:00+08:00")
        second = self.row("2026-01-02T10:00:00+08:00",
                          "2026-01-06T10:00:00+08:00")
        third = self.row("2026-01-07T10:00:00+08:00",
                         "2026-01-08T10:00:00+08:00")
        out = replay_complete_matrix(self.plans, [first, second, third], seed=17)
        self.assertEqual([r["visible_valid_before_decision"] for r in out["rounds"]],
                         [0, 0, 2])
        self.assertEqual(out["paired_rounds"], 3)

    def test_invalid_or_premature_economics_rejected(self):
        row = self.row("2026-01-01T10:00:00+08:00",
                       "2026-01-01T09:00:00+08:00")
        with self.assertRaisesRegex(ValueError, "prematurely mature"):
            replay_complete_matrix(self.plans, [row], seed=17)
        row["outcomes"]["a"]["mature_at"] = "2026-01-02T10:00:00+08:00"
        row["outcomes"]["b"]["mature_at"] = "2026-01-02T10:00:00+08:00"
        row["outcomes"]["a"]["economic_lcb_bps"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid or prematurely"):
            replay_complete_matrix(self.plans, [row], seed=17)


if __name__ == "__main__":
    unittest.main()
