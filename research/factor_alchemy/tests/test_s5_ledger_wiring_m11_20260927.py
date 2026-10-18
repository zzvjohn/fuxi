# -*- coding: utf-8 -*-
"""M11 接线测试 (2026-09-27): FUXI_S5_LEDGER 分派 + 账本短路 + fail-closed。

覆盖:
  · _s5_ledger_mode 默认/覆盖/非法 fail-fast;
  · backtest_factor 默认(proxy)不触碰账本路径 (diag 无 ledger_route);
  · fixed_shares 路径完整跑通 (合成 raw + 假实例): valid + 非空收益;
  · 持仓缺报价 ⇒ 未闭合 ⇒ data_insufficient + 缺数可枚举 + 空序列;
  · _backtest_and_metrics 对未闭合账本短路为 ledger_data_insufficient
    (不得归因成"两项均不达标")。

运行 (cwd = research/factor_alchemy):
    python -X utf8 -m unittest tests.test_s5_ledger_wiring_m11_20260927 -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

import s5_joint_filter as sjf  # noqa: E402

CODES = ("600001.SH", "600002.SH", "600003.SH")
DAYS = [f"2026-01-{d:02d}" for d in
        (5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23)]
BASE = {"600001.SH": 10.0, "600002.SH": 20.0, "600003.SH": 30.0}


def _write_raw(tmp: Path, *, drop_key=None):
    """合成 data/raw 四表 (恒定价格: 无涨跌停/无除权/无停牌 ⇒ 账本应闭合)。"""
    price_rows, limit_rows, adj_rows = [], [], []
    prev_close = {}
    for d in DAYS:
        for c in CODES:
            if drop_key == (d, c):
                continue
            close = BASE[c]
            pre = prev_close.get(c, close)
            price_rows.append((c, d.replace("-", ""), close, close, pre,
                               1000, close * 1000 * 100))
            limit_rows.append((c, d.replace("-", ""), close * 1.1, close * 0.9))
            adj_rows.append((c, d.replace("-", ""), 1.0))
            prev_close[c] = close

    def _csv(name, header, rows):
        lines = [",".join(header)]
        lines += [",".join(str(x) for x in r) for r in rows]
        (tmp / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    _csv("daily_prices.csv",
         ("ts_code", "trade_date", "open", "close", "pre_close", "vol",
          "amount"), price_rows)
    _csv("stk_limit.csv",
         ("ts_code", "trade_date", "up_limit", "down_limit"), limit_rows)
    _csv("adj_factor_daily.csv",
         ("ts_code", "trade_date", "adj_factor"), adj_rows)
    (tmp / "suspend_d.csv").write_text(
        "ts_code,trade_date,suspend_type,suspend_timing\n", encoding="utf-8")


def _factor_df(*, flip_at=9):
    """因子面板: 高值做多; flip_at 之后截面顺序翻转 ⇒ 第二期换仓。"""
    rows = []
    for i, d in enumerate(DAYS):
        for j, c in enumerate(CODES):
            v = float(len(CODES) - j) if i < flip_at else float(j + 1)
            rows.append({"ts_code": c, "trade_date": pd.Timestamp(d),
                         "factor_value": v, "amount": 1e6})
    return pd.DataFrame(rows)


def _price_df(*, drop_key=None):
    rows = []
    for d in DAYS:
        for c in CODES:
            if drop_key == (d, c):
                continue
            rows.append({"ts_code": c, "trade_date": pd.Timestamp(d),
                         "close": BASE[c], "amount": 1e6})
    return pd.DataFrame(rows)


def _fake_s5(price_df, *, top_n=2):
    s5 = sjf.S5JointFilter.__new__(sjf.S5JointFilter)
    s5._price_df = price_df
    s5._universe = list(CODES)
    s5.top_n = top_n
    s5._last_backtest_diag = None
    s5._limit_panel = None
    s5._loaded = True
    return s5


class LedgerModeTests(unittest.TestCase):
    def test_default_proxy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FUXI_S5_LEDGER", None)
            self.assertEqual(sjf._s5_ledger_mode(), "proxy")

    def test_override(self):
        with mock.patch.dict(os.environ, {"FUXI_S5_LEDGER": "fixed_shares"}):
            self.assertEqual(sjf._s5_ledger_mode(), "fixed_shares")

    def test_invalid_fail_fast(self):
        with mock.patch.dict(os.environ, {"FUXI_S5_LEDGER": "bogus"}):
            with self.assertRaises(ValueError):
                sjf._s5_ledger_mode()


class DispatchTests(unittest.TestCase):
    def test_proxy_default_untouched(self):
        """默认 proxy: 空信号早退, diag 不得出现 ledger_route (原路径未触碰)。"""
        s5 = _fake_s5(_price_df())
        with mock.patch.dict(os.environ, {"FUXI_S5_LEDGER": "proxy"}):
            port, _ = s5.backtest_factor(pd.DataFrame(
                columns=["ts_code", "trade_date", "factor_value"]))
        self.assertEqual(len(port), 0)
        diag = s5._last_backtest_diag or {}
        self.assertNotIn("ledger_route", diag)


class FixedSharesRouteTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)
        _write_raw(self.raw)
        self._env = mock.patch.dict(os.environ, {
            "FUXI_S5_LEDGER": "fixed_shares",
            "FUXI_S5_TRADABLE_GATE": "off",   # 合成数据无涨停冲突, 简化
        })
        self._env.start()
        self._raw_patch = mock.patch.object(sjf, "_LEDGER_RAW_DIR", self.raw)
        self._raw_patch.start()

    def tearDown(self):
        self._raw_patch.stop()
        self._env.stop()
        self._tmp.cleanup()

    def test_complete_route(self):
        s5 = _fake_s5(_price_df())
        port, _ = s5.backtest_factor(_factor_df())
        diag = s5._last_backtest_diag or {}
        self.assertTrue(diag.get("ledger_route"))
        self.assertEqual(diag.get("ledger_status"), "valid",
                         msg=str(diag.get("ledger_reasons")))
        self.assertGreater(len(port), 0)
        self.assertGreater(int(diag.get("ledger_n_fills") or 0), 0)
        self.assertEqual(diag.get("n_rebalance_used"), 2)
        # 换仓确实发生: 两期 targets 不同 ⇒ 卖 ≥1 买 ≥1
        self.assertGreaterEqual(int(diag.get("ledger_n_orders") or 0), 4)
        # 收益日期应落在市场日历内
        self.assertGreaterEqual(port.index.min(), pd.Timestamp(DAYS[0]))
        self.assertLessEqual(port.index.max(), pd.Timestamp(DAYS[-1]))

    def test_missing_quote_fails_closed(self):
        """持仓日缺报价 (且无停牌证据) ⇒ 未闭合, 缺数可枚举, 空序列。"""
        _write_raw(self.raw, drop_key=("2026-01-14", "600001.SH"))
        s5 = _fake_s5(_price_df(drop_key=("2026-01-14", "600001.SH")))
        port, _ = s5.backtest_factor(_factor_df())
        diag = s5._last_backtest_diag or {}
        self.assertTrue(diag.get("ledger_route"))
        self.assertEqual(diag.get("ledger_status"), "data_insufficient")
        self.assertTrue(diag.get("ledger_reasons"))
        self.assertEqual(len(port), 0)
        missing = diag.get("ledger_missing") or {}
        self.assertGreater(int(missing.get("n_errors") or 0), 0)


class ShortCircuitTests(unittest.TestCase):
    def test_metrics_short_circuit(self):
        """未闭合账本 ⇒ _backtest_and_metrics 直接给 error, 不走"不达标"归因。"""
        s5 = _fake_s5(_price_df())
        s5._last_backtest_diag = {
            "ledger_status": "data_insufficient",
            "ledger_reasons": ["missing_corporate_action_event", "halted"],
        }
        with mock.patch.object(s5, "backtest_factor",
                               return_value=(pd.Series(dtype=float),
                                             pd.Series(dtype=float))):
            port, m25, m26 = s5._backtest_and_metrics(_factor_df())
        self.assertEqual(m25.get("error"), "ledger_data_insufficient")
        self.assertEqual(m26.get("error"), "ledger_data_insufficient")
        self.assertEqual(m25.get("evaluation_status"), "data_insufficient")
        self.assertIn("missing_corporate_action_event",
                      m25.get("ledger_reasons") or [])
        self.assertEqual(len(port), 0)

    def test_verdict_reports_uncallable(self):
        s5 = _fake_s5(_price_df())
        passed, reason = s5._verdict(
            {"error": "ledger_data_insufficient"}, {"error": "ledger_data_insufficient"})
        self.assertFalse(passed)
        self.assertIn("不可裁决", reason)


if __name__ == "__main__":
    unittest.main()
