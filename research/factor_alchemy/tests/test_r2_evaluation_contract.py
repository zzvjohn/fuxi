from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from factor_ic_computer import FactorICComputer
from multi_stage_validator import MultiStageValidator, ValidationResult


class _PriceOnlyComputer(FactorICComputer):
    """Small deterministic loader that keeps the production window semantics."""

    def _load_data(self):
        frame = pd.read_csv(self.price_path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
        if self.lookback_days is not None:
            start = frame["trade_date"].max() - pd.Timedelta(days=self.lookback_days)
            frame = frame[frame["trade_date"] >= start]
        for name in ("open", "high", "low", "close", "volume", "amount"):
            frame[f"{name}_p"] = frame[name]
        self._price_df = frame
        self._universe = sorted(frame["ts_code"].unique())
        self._all_dates = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
        self._wide_cache.clear()
        self._loaded = True


class R2EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / "reports" / "remediation_20260922" / "_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / uuid.uuid4().hex
        self.tmp.mkdir()
        self.old_s4_mode = os.environ.get("FUXI_S4_OOS_MODE")
        self.old_yearly_mode = os.environ.get("FUXI_YEARLY_IC_WINDOW")
        os.environ["FUXI_YEARLY_IC_WINDOW"] = "enforce"

    def tearDown(self):
        if self.old_s4_mode is None:
            os.environ.pop("FUXI_S4_OOS_MODE", None)
        else:
            os.environ["FUXI_S4_OOS_MODE"] = self.old_s4_mode
        if self.old_yearly_mode is None:
            os.environ.pop("FUXI_YEARLY_IC_WINDOW", None)
        else:
            os.environ["FUXI_YEARLY_IC_WINDOW"] = self.old_yearly_mode
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _s123_result():
        return ValidationResult(
            "candidate", s1_passed=True, s2_passed=True, s3_passed=True,
            s1_ic=0.03, s1_icir=0.2,
        )

    def test_s4_audit_does_not_claim_independent_oos(self):
        os.environ["FUXI_S4_OOS_MODE"] = "audit"
        result = self._s123_result()
        MultiStageValidator().stage4_oos_replacement(
            [{"factor_name": "candidate", "ic": 0.03, "icir": 0.2}], [result], {}
        )
        self.assertTrue(result.s4_passed)
        self.assertEqual(result.s4_eval_status, "not_evaluated")
        self.assertEqual(result.s4_decision_basis, "legacy_s1_proxy")
        self.assertIn("不得表述为 OOS 通过", result.s4_reason)

    def test_s4_strict_rejects_missing_independent_oos(self):
        os.environ["FUXI_S4_OOS_MODE"] = "strict"
        result = self._s123_result()
        MultiStageValidator().stage4_oos_replacement(
            [{"factor_name": "candidate", "ic": 0.03, "icir": 0.2}], [result], {}
        )
        self.assertFalse(result.s4_passed)
        self.assertEqual(result.s4_eval_status, "not_evaluated")

    def test_s5_missing_measurement_is_not_counted_as_factor_failure(self):
        result = self._s123_result()
        result.s4_passed = True
        validator = MultiStageValidator()
        validator.stage5_joint_filter([{"factor_name": "candidate"}], [result], None)
        self.assertFalse(result.s5_passed)
        self.assertEqual(result.s5_eval_status, "not_evaluated")
        self.assertIn("不计作因子失败", result.s5_reason)

    def test_s5_formula_precheck_rejection_is_distinct_from_not_evaluated(self):
        class S5Precheck:
            elapsed_sec = 0.0
            window_days_2025 = 0
            window_days_2026 = 0
            reason = "🧹 公式合理性拦截: 价格代理"
            error = ""
            excess_2025 = 0.0
            excess_2026 = 0.0
            calmar_2025 = 0.0
            calmar_2026 = 0.0
            passed = False

        result = self._s123_result()
        MultiStageValidator()._apply_stage5_from_s5result(S5Precheck(), [result], 0)
        self.assertEqual(result.s5_eval_status, "precheck_rejected")

    def test_yearly_ic_loads_warmup_before_evaluation_window(self):
        dates = pd.bdate_range("2024-01-02", "2026-06-30")
        rows = []
        for stock in range(8):
            for t, date in enumerate(dates):
                close = 20.0 + stock + (t * (0.001 + stock * 0.0002)) + np.sin(t / 17 + stock)
                rows.append({
                    "ts_code": f"{stock:06d}.SZ",
                    "trade_date": int(date.strftime("%Y%m%d")),
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000 + stock * 10 + t,
                    "amount": (1000 + stock * 10 + t) * close,
                })
        path = self.tmp / "daily_prices.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        computer = _PriceOnlyComputer(
            price_path=path,
            lookback_days=30,
            min_stocks_per_day=5,
            forward_period=5,
        )
        result = computer.compute_yearly_ic(
            "close_p.rolling(20).mean()", "warmup_contract"
        )
        self.assertEqual(result["_meta"]["eval_status"], "valid")
        self.assertIsNotNone(result["2025"])
        self.assertGreaterEqual(result["2025"]["n_days"], 100)
        self.assertLess(result["_meta"]["calculation_window"][0], "2025-01-01")

        os.environ["FUXI_YEARLY_IC_WINDOW"] = "legacy"
        legacy = _PriceOnlyComputer(
            price_path=path,
            lookback_days=30,
            min_stocks_per_day=5,
            forward_period=5,
        ).compute_yearly_ic("close_p.rolling(20).mean()", "legacy_window")
        self.assertIsNone(legacy.get("2025"))


if __name__ == "__main__":
    unittest.main()
