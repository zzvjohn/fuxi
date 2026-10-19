"""E3 absolute-excess gate (2026-10-19).

The paired increment is measured against a straw-man baseline; on its own it can
be positive while the strategy still loses money. The gate requires, for
non-positive absolute returns, a positive excess on BOTH CSI300 and CSI500.
"""
import unittest

from fuxi_v1.benchmarks import absolute_excess, excess_gate


def excess(c300, c500, status="available"):
    return {"status": status,
            "csi300": {"status": "available", "excess": c300},
            "csi500": {"status": "available", "excess": c500}}


class ExcessGateTests(unittest.TestCase):
    def test_positive_absolute_return_needs_no_excess(self):
        ok, reason = excess_gate(excess(-0.9, -0.9), 0.05)
        self.assertTrue(ok)
        self.assertEqual(reason, "absolute_return_positive")

    def test_nonpositive_absolute_return_requires_both_indices(self):
        ok, reason = excess_gate(excess(0.10, -0.01), -0.05)
        self.assertFalse(ok)
        self.assertEqual(reason, "absolute_return_nonpositive_without_excess")
        ok, reason = excess_gate(excess(0.10, 0.02), -0.05)
        self.assertTrue(ok)
        self.assertEqual(reason, "absolute_return_nonpositive_but_excess_positive")

    def test_missing_benchmark_never_silently_passes_the_excess_check(self):
        ok, reason = excess_gate({"status": "unavailable"}, -0.05)
        self.assertTrue(ok)  # historical paired-only behaviour is preserved ...
        self.assertEqual(reason, "benchmark_unavailable_paired_only")
        ok, reason = excess_gate({"status": "partial", "csi300": {"status": "available", "excess": 0.1},
                                  "csi500": {"status": "insufficient_coverage"}}, -0.05)
        self.assertFalse(ok)  # ... but partial evidence cannot certify excess
        self.assertEqual(reason, "benchmark_evidence_incomplete")


class RealIndexTests(unittest.TestCase):
    def test_certified_window_has_both_indices(self):
        daily = [{"session": s, "nav": str(1000000 + i * 1000)}
                 for i, s in enumerate(["2021-05-18", "2022-01-04", "2023-01-03",
                                        "2024-01-02", "2025-01-02", "2025-12-31"])]
        result = absolute_excess(daily)
        self.assertEqual(result.get("status"), "available")
        for name in ("csi300", "csi500"):
            block = result[name]
            self.assertEqual(block["status"], "available")
            self.assertGreaterEqual(block["coverage"], 0.98)
            self.assertAlmostEqual(block["candidate_total_return"], 0.005, places=6)


if __name__ == "__main__":
    unittest.main()
