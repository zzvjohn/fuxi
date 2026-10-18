# -*- coding: utf-8 -*-
"""M6-B 周频计划库与频率匹配契约测试 (2026-09-28)

覆盖: 分频加载 / 频率匹配前置 / 周频契约 (cutoff/fields) / select_plan 空证据探索。
运行: C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe -X utf8
      research/factor_alchemy/tests/test_semantic_plan_weekly_20260928.py
"""
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from semantic_plan import validate_plan  # noqa: E402
from mechanism_search import load_catalog, select_plan  # noqa: E402

NEW = BASE / "mechanism_catalog_20260928.json"
OLD = BASE / "mechanism_catalog_20260925.json"


class TestWeeklyCatalog(unittest.TestCase):
    def test_daily_not_modified(self):
        d = load_catalog(OLD, "daily")
        self.assertEqual(len(d), 4)

    def test_old_catalog_weekly_still_raises(self):
        # 旧库无 weekly 卡 ⇒ 保持抛错 (记录现状; 这正是 M6 修的洞)
        with self.assertRaises(ValueError) as cm:
            load_catalog(OLD, "weekly")
        self.assertIn("no plans for requested frequency", str(cm.exception))

    def test_new_catalog_split_by_frequency(self):
        d = load_catalog(NEW, "daily")
        w = load_catalog(NEW, "weekly")
        self.assertEqual(len(d), 4)
        self.assertEqual(len(w), 3)
        self.assertTrue(all(p["frequency"] == "weekly" for p in w))
        self.assertTrue(all(p["frequency"] == "daily" for p in d))

    def test_new_catalog_no_unknown_frequency(self):
        with self.assertRaises(ValueError):
            load_catalog(NEW, "hourly")

    def test_weekly_plan_ids(self):
        w = load_catalog(NEW, "weekly")
        ids = {p["plan_id"] for p in w}
        self.assertEqual(ids, {"pw_weekly_reversal_v1", "pw_compression_breakout_v1",
                               "pw_effort_result_divergence_v1"})


class TestWeeklyContract(unittest.TestCase):
    def test_cutoff_contract(self):
        w = load_catalog(NEW, "weekly")
        for p in w:
            self.assertEqual(p["signal_cutoff"], "week_end_post_close")
            self.assertEqual(p["earliest_trade"], "next_week")

    def test_fields_within_weekly_available(self):
        allowed = {"open", "high", "low", "close", "volume", "amount"}
        w = load_catalog(NEW, "weekly")
        for p in w:
            self.assertTrue(set(p["fields"]) <= allowed, p["plan_id"])

    def test_wrong_cutoff_rejected(self):
        bad = {
            "plan_id": "pw_bad_cutoff", "frequency": "weekly",
            "mechanism": "x", "prediction": "y", "falsifier": "z", "baseline": "b",
            "target_use": "research_feature", "fields": ["close"],
            "signal_cutoff": "post_close", "earliest_trade": "next_session",
        }
        with self.assertRaises(ValueError):
            validate_plan(bad)

    def test_daily_plan_not_loadable_as_weekly(self):
        from semantic_plan import load_plan
        with self.assertRaises(ValueError) as cm:
            load_plan(NEW, "pv_gap_rejection_v1", "weekly")
        self.assertIn("frequency disagrees", str(cm.exception))

    def test_weekly_plan_loadable(self):
        from semantic_plan import load_plan
        p = load_plan(NEW, "pw_weekly_reversal_v1", "weekly")
        self.assertEqual(p["frequency"], "weekly")
        with self.assertRaises(ValueError):
            load_plan(NEW, "pw_weekly_reversal_v1", "daily")


class TestSelectPlanWeekly(unittest.TestCase):
    def test_explore_with_empty_evidence(self):
        w = load_catalog(NEW, "weekly")
        plan, audit = select_plan(w, [], seed=7, round_index=0)
        self.assertEqual(audit["mode"], "explore")
        self.assertEqual(audit["valid_economic_observations"], 0)
        self.assertFalse(audit["grants_promotion"])
        self.assertIn(plan["plan_id"], {p["plan_id"] for p in w})


class TestFreqPrecheckInRalphLoop(unittest.TestCase):
    """M6-B 频率匹配前置: 直接执行 ralph_loop.py 中补丁后的真实源码段验证行为。

    提取 `if _plan_mode == "auto":` 到 `if _plan_mode == "on":` 之间的真实文本,
    在受控 env 中 exec:
      · 旧库(无 weekly) + freq=weekly ⇒ _plan_mode 降级 "off" + set_semantic_plan(None)
      · 新库(有 weekly) + freq=weekly ⇒ _plan_mode 保持 "auto" (不降级)
      · 新库 + freq=daily           ⇒ 保持 "auto"
    """

    SRC = BASE / "ralph_loop.py"

    @classmethod
    def setUpClass(cls):
        import textwrap
        src = cls.SRC.read_text(encoding="utf-8")
        start = src.find('            if _plan_mode == "auto":\n')
        end = src.find('            if _plan_mode == "on":\n')
        assert start != -1 and end != -1 and start < end, "预检段定位失败"
        cls.seg = textwrap.dedent(src[start:end])
        assert "_lane_freq_pre" in cls.seg, "预检段缺补丁标记"

    def _exec_seg(self, plan_file, freq):
        import os
        from types import SimpleNamespace

        class FakeLLM:
            def __init__(self):
                self.plan = "UNSET"

            def set_semantic_plan(self, p):
                self.plan = p

        fake = SimpleNamespace(llm_gen=FakeLLM())
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {
                "FUXI_SEMANTIC_PLAN_FILE": str(plan_file),
                "FUXI_SEMANTIC_EVIDENCE_FILE": "x-evidence.jsonl"}):
            env = {"os": os, "freq": freq, "_plan_mode": "auto",
                   "print": lambda *a, **k: None, "self": fake}
            exec(compile(self.seg, "<freq_precheck_seg>", "exec"), env)
        return env["_plan_mode"], fake.llm_gen.plan

    def test_old_catalog_degrades_off(self):
        # 降级后由链尾 else: set_semantic_plan(None) 处理 (不在本段内) ⇒
        # 段内只验证 _plan_mode 变为 "off" 且未注入任何计划。
        mode, plan = self._exec_seg(OLD, "weekly")
        self.assertEqual(mode, "off")
        self.assertEqual(plan, "UNSET")

    def test_new_catalog_keeps_auto(self):
        mode, plan = self._exec_seg(NEW, "weekly")
        self.assertEqual(mode, "auto")
        self.assertEqual(plan, "UNSET")

    def test_new_catalog_daily_keeps_auto(self):
        mode, plan = self._exec_seg(NEW, "daily")
        self.assertEqual(mode, "auto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
