# -*- coding: utf-8 -*-
"""M18 数据新鲜度守卫单元测试 (2026-09-28)

覆盖: 日期归一化 / 交易日 lag / cutoff 期望语义 / tail vs full 一致性 /
kind 分类行为 / simulate-stale 注入 / fail-closed 闸门 / 开关解析 / 报告落盘。
运行: C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe -X utf8
      research/factor_alchemy/tests/test_freshness_guard_20260928.py
"""
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import data_freshness_guard as G  # noqa: E402


class TestDateNorm(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(G.norm_date("20260924"), "20260924")
        self.assertEqual(G.norm_date("20260924.0"), "20260924")
        self.assertEqual(G.norm_date("2026-09-24"), "20260924")
        self.assertEqual(G.norm_date("2026/09/24"), "20260924")
        self.assertIsNone(G.norm_date(""))
        self.assertIsNone(G.norm_date("nan"))
        self.assertIsNone(G.norm_date("abc"))
        self.assertIsNone(G.norm_date("2026092"))     # 7 位
        self.assertIsNone(G.norm_date("202609245"))   # 9 位


class TestCalendarLag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opens = G.load_calendar()
        assert len(cls.opens) > 1000, "日历加载异常"

    def test_expected_before_cutoff(self):
        # 交易日 09-28 早 07:00 (cutoff 前) ⇒ 期望 = 上一开市日 09-24
        now = datetime(2026, 9, 28, 7, 0)
        self.assertEqual(G.expected_open_day(self.opens, now), "20260924")

    def test_expected_after_cutoff(self):
        # 交易日 09-28 晚 22:00 (cutoff 后) ⇒ 期望 = 当日
        now = datetime(2026, 9, 28, 22, 0)
        self.assertEqual(G.expected_open_day(self.opens, now), "20260928")

    def test_expected_on_weekend(self):
        # 周日 09-27 ⇒ 期望 = 09-24
        now = datetime(2026, 9, 27, 12, 0)
        self.assertEqual(G.expected_open_day(self.opens, now), "20260924")

    def test_lag_zero_when_current(self):
        self.assertEqual(G.lag_trading_days(self.opens, "20260924", "20260924"), 0)
        # 末端超前也不为负
        self.assertEqual(G.lag_trading_days(self.opens, "20260925", "20260924"), 0)

    def test_lag_counts_open_days_only(self):
        # 09-24 -> 09-28: 09-25/26/27 休市 ⇒ lag = 1 (仅 09-28)
        self.assertEqual(G.lag_trading_days(self.opens, "20260924", "20260928"), 1)
        # 09-18 -> 09-24: 09-21/22/23/24 四个开市日 ⇒ lag = 4
        self.assertEqual(G.lag_trading_days(self.opens, "20260918", "20260924"), 4)
        # 09-23 -> 09-24: 1
        self.assertEqual(G.lag_trading_days(self.opens, "20260923", "20260924"), 1)


class TestTailVsFull(unittest.TestCase):
    """tail 快读必须与全表扫描一致 (按时间追加前提的实证)。"""

    CASES = [
        ("data/raw/index_000300_daily.csv", "trade_date"),
        ("data/raw/suspend_d.csv", "trade_date"),
        ("data/raw/kuaidao_gaps/dividend.csv", "ann_date"),
        ("data/raw/sw_daily.csv", "trade_date"),
    ]

    def test_consistency(self):
        for rel, col in self.CASES:
            p = ROOT / rel
            if not p.exists():
                continue
            with self.subTest(channel=rel):
                tail = G.tail_max_date(p, col)
                full = G.full_max_date(p, col)
                self.assertIsNotNone(tail, f"{rel} tail 读取失败")
                self.assertEqual(tail, full, f"{rel} tail={tail} != full={full}")


class TestFullModeChannels(unittest.TestCase):
    """read_mode=full 的事件表: full_max_date 必须与 pandas 全读一致。

    实测背景: forecast.csv 尾行 ann_date=20260130 而全表 max=20260807 ⇒
    事件表非时间序追加, tail 快读前提不成立 ⇒ 强制 full。
    """

    def _pandas_max(self, rel, col):
        import pandas as pd
        df = pd.read_csv(ROOT / rel, usecols=[col], dtype=str, low_memory=False)
        return df[col].map(G.norm_date).dropna().max()

    def test_forecast_full_matches_pandas(self):
        rel, col = "data/raw/forecast.csv", "ann_date"
        self.assertEqual(G.full_max_date(ROOT / rel, col), self._pandas_max(rel, col))

    def test_fina_indicator_full_matches_pandas(self):
        rel, col = "data/raw/fina_indicator.csv", "ann_date"
        self.assertEqual(G.full_max_date(ROOT / rel, col), self._pandas_max(rel, col))

    def test_event_channels_use_full_mode(self):
        for name in ("forecast", "fina_indicator", "namechange", "dividend"):
            ch = next(c for c in G.CHANNELS if c["channel"] == name)
            self.assertEqual(ch.get("read_mode"), "full", f"{name} 须 full 模式")


class TestChannelReads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opens = G.load_calendar()
        cls.by_name = {c["channel"]: c for c in G.CHANNELS}

    def test_retired_not_stale(self):
        r = G.read_channel(self.by_name["hk_hold"], self.opens,
                           datetime(2026, 9, 28, 7, 0))
        self.assertEqual(r["level"], "retired")
        self.assertIsNotNone(r["max_date"])
        # retired 不参与 gate
        gate = G.evaluate_gate([r])
        self.assertFalse(gate["blocked"])

    def test_on_demand_info(self):
        r = G.read_channel(self.by_name["fina_indicator"], self.opens,
                           datetime(2026, 9, 28, 7, 0))
        self.assertEqual(r["level"], "info")

    def test_event_ingest_age(self):
        r = G.read_channel(self.by_name["forecast"], self.opens,
                           datetime(2026, 9, 28, 7, 0))
        self.assertIn(r["level"], ("ok", "warn", "block"))
        self.assertIn("ingest_age_days", r)

    def test_trading_day_current_is_ok(self):
        r = G.read_channel(self.by_name["daily_prices"], self.opens,
                           datetime(2026, 9, 28, 7, 0))
        if r["max_date"] == "20260924":
            self.assertEqual(r["level"], "ok")
            self.assertEqual(r["lag"], 0)


class TestSimulateAndGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opens = G.load_calendar()

    def test_simulate_stale_block(self):
        r = G.read_channel({**next(c for c in G.CHANNELS if c["channel"] == "daily_prices")},
                           self.opens, datetime(2026, 9, 28, 7, 0))
        sim = G.apply_simulation([r], {"daily_prices": 4}, self.opens)
        self.assertEqual(sim[0]["level"], "block")
        self.assertEqual(sim[0]["lag"], 4)
        self.assertTrue(sim[0].get("simulated"))
        gate = G.evaluate_gate(sim)
        self.assertTrue(gate["blocked"])
        self.assertIn("daily_prices", gate["blocked_channels"])
        self.assertIn("ralph_pipeline", gate["blocked_chains"])

    def test_simulate_warn_level(self):
        r = G.read_channel(next(c for c in G.CHANNELS if c["channel"] == "daily_prices"),
                           self.opens, datetime(2026, 9, 28, 7, 0))
        sim = G.apply_simulation([r], {"daily_prices": 2}, self.opens)
        self.assertEqual(sim[0]["level"], "warn")
        gate = G.evaluate_gate(sim)
        self.assertFalse(gate["blocked"])
        self.assertIn("daily_prices", gate["warn_channels"])

    def test_retired_never_blocked_under_simulation(self):
        # retired 通道即便模拟落后也不得进入 gate (on_stale=info)
        r = G.read_channel(next(c for c in G.CHANNELS if c["channel"] == "hk_hold"),
                           self.opens, datetime(2026, 9, 28, 7, 0))
        sim = G.apply_simulation([r], {"hk_hold": 99}, self.opens)
        gate = G.evaluate_gate(sim)
        self.assertFalse(gate["blocked"])


class TestGateMode(unittest.TestCase):
    def test_default_shadow(self):
        self.assertEqual(G.gate_mode({}), "shadow")

    def test_explicit(self):
        self.assertEqual(G.gate_mode({"FUXI_FRESHNESS_GATE": "off"}), "off")
        self.assertEqual(G.gate_mode({"FUXI_FRESHNESS_GATE": "on"}), "on")
        self.assertEqual(G.gate_mode({"FUXI_FRESHNESS_GATE": "ON "}), "on")

    def test_invalid_falls_back_shadow(self):
        self.assertEqual(G.gate_mode({"FUXI_FRESHNESS_GATE": "zzz"}), "shadow")


class TestReportRoundTrip(unittest.TestCase):
    def test_write_and_reload(self):
        report = G.run_check(now=datetime(2026, 9, 28, 7, 0), quiet=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            G.write_report(report, p)
            with io.open(p, encoding="utf-8") as f:
                loaded = json.loads(f.read())
            self.assertEqual(loaded["schema"], "fuxi_freshness_guard/v1")
            self.assertEqual(len(loaded["readings"]), len(G.CHANNELS))
            self.assertIn("gate", loaded)


class TestRealStateNoBlock(unittest.TestCase):
    """当前真实状态下: 主链通道不得为 block 级 (除模拟外)。"""

    def test_no_block_now(self):
        report = G.run_check(now=datetime(2026, 9, 28, 7, 0), quiet=True)
        blocked = report["gate"]["blocked_channels"]
        self.assertEqual(blocked, [], f"存在 block 级通道: {blocked}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
