"""Tests for the beta-exposure monitor (industry decomposition + drift)."""

import unittest

from fuxi_v1.contracts import digest
from fuxi_v1.dsl import generate_candidates
from fuxi_v1.exposure_monitor import (
    normalize_code, industry_exposure, exposure_snapshot, period_exposures,
    render_report,
)
from fuxi_v1.portfolio import strategy_spec, compile_strategy


class TestExposureMonitor(unittest.TestCase):
    def setUp(self):
        self.industry_map = {
            "000001.SZ": "银行",
            "000002.SZ": "房地产",
            "000003.SZ": "银行",
            "000004.SZ": "医药",
            "000005.SZ": "医药",
        }

    def test_normalize_code(self):
        self.assertEqual(normalize_code("000001.SZ"), "000001.SZ")
        self.assertEqual(normalize_code("sz000001"), "000001.SZ")
        self.assertEqual(normalize_code("000001"), "000001")

    def test_industry_exposure_aggregates_by_weight(self):
        w = {"000001.SZ": 0.2, "000002.SZ": 0.2, "000003.SZ": 0.2, "999999.SZ": 0.4}
        out = industry_exposure(w, self.industry_map)
        self.assertAlmostEqual(out["银行"], 0.4)
        self.assertAlmostEqual(out["房地产"], 0.2)
        self.assertAlmostEqual(out["未分类"], 0.4)

    def test_snapshot_hashes_industry(self):
        w = {"000001.SZ": 1.0}
        snap = exposure_snapshot(w, self.industry_map)
        self.assertEqual(snap["industry"], {"银行": 1.0})
        self.assertIsNone(snap["market_cap"])
        self.assertFalse(snap["market_cap_available"])

    def test_market_cap_buckets(self):
        cap_map = {"000001.SZ": 500e8, "000002.SZ": 200e8, "000004.SZ": 50e8}
        w = {"000001.SZ": 0.4, "000002.SZ": 0.3, "000004.SZ": 0.3}
        out = exposure_snapshot(w, self.industry_map, cap_map)
        self.assertEqual(out["market_cap"], {"大盘": 0.4, "中盘": 0.3, "小盘": 0.3})

    def test_period_exposures_and_drift(self):
        # Build a minimal strategy IR over a synthetic snapshot is heavy; instead
        # exercise the drift math through a fake IR with two decisions whose
        # scores flip the top names so industry exposure changes.
        spec = strategy_spec("demo", top_n=2, rebalance_sessions=1)
        decisions = [
            {"signal_session": "2024-01-02",
             "scores": {"000001.SZ": 1.0, "000002.SZ": 0.9, "000003.SZ": 0.8, "000004.SZ": 0.7}},
            {"signal_session": "2024-01-03",
             "scores": {"000004.SZ": 1.0, "000005.SZ": 0.9, "000001.SZ": 0.8, "000002.SZ": 0.7}},
        ]
        ir = {"schema_version": "fuxi.strategy_ir.1", "strategy": spec,
              "ir_hash": digest(spec), "decisions": decisions}
        report = period_exposures(ir, self.industry_map)
        self.assertEqual(len(report["periods"]), 2)
        p1, p2 = report["periods"]
        # max_weight=0.2 caps each name at 20% (gross 0.4, 60% cash).
        # period 1: top2 = 000001(银行), 000002(房地产) -> 银行0.2 房地产0.2
        self.assertEqual(p1["industry"], {"银行": 0.2, "房地产": 0.2})
        self.assertIsNone(p1["industry_drift"])
        self.assertEqual(p1["gross_weight"], 0.4)
        # period 2: top2 = 000004(医药), 000005(医药) -> 医药0.4
        self.assertEqual(p2["industry"], {"医药": 0.4})
        self.assertEqual(p2["industry_drift"], {"银行": -0.2, "房地产": -0.2, "医药": 0.4})

    def test_render_report(self):
        spec = strategy_spec("demo", top_n=2)
        ir = {"schema_version": "fuxi.strategy_ir.1", "strategy": spec,
              "ir_hash": "x", "decisions": [{"signal_session": "2024-01-02",
              "scores": {"000001.SZ": 1.0, "000002.SZ": 0.9}}]}
        md = render_report(period_exposures(ir, self.industry_map))
        self.assertIn("组合 beta 暴露监控", md)
        self.assertIn("银行", md)

    def test_period_exposures_dated_cap_map(self):
        # Two-layer {trade_date: {code: cap}} should be matched per session
        # (no look-ahead): day 1 both names large-cap, day 2 both small-cap.
        spec = strategy_spec("demo", top_n=2)
        decisions = [
            {"signal_session": "2024-01-02",
             "scores": {"000001.SZ": 1.0, "000002.SZ": 0.9}},
            {"signal_session": "2024-01-09",
             "scores": {"000001.SZ": 1.0, "000002.SZ": 0.9}},
        ]
        ir = {"schema_version": "fuxi.strategy_ir.1", "strategy": spec,
              "ir_hash": "x", "decisions": decisions}
        cap_map = {
            "20240102": {"000001.SZ": 500e8, "000002.SZ": 400e8},  # both 大盘
            "20240109": {"000001.SZ": 50e8, "000002.SZ": 40e8},    # both 小盘
        }
        report = period_exposures(ir, self.industry_map, cap_map)
        p1, p2 = report["periods"]
        self.assertEqual(p1["market_cap"], {"大盘": 0.4})
        self.assertEqual(p2["market_cap"], {"小盘": 0.4})
        self.assertEqual(p2["market_cap_drift"], {"大盘": -0.4, "小盘": 0.4})
        self.assertTrue(report["market_cap_available"])

    def test_period_exposures_single_layer_cap_map(self):
        # Single-layer {code: cap} treated as static across all sessions.
        spec = strategy_spec("demo", top_n=2)
        decisions = [{"signal_session": "2024-01-02",
                      "scores": {"000001.SZ": 1.0, "000002.SZ": 0.9}}]
        ir = {"schema_version": "fuxi.strategy_ir.1", "strategy": spec,
              "ir_hash": "x", "decisions": decisions}
        cap_map = {"000001.SZ": 500e8, "000002.SZ": 50e8}
        report = period_exposures(ir, self.industry_map, cap_map)
        self.assertEqual(report["periods"][0]["market_cap"], {"大盘": 0.2, "小盘": 0.2})


if __name__ == "__main__":
    unittest.main()
