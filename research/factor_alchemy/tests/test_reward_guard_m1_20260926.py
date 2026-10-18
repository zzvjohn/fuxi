"""M1 定向测试：奖励守卫（影子半程）。

契约（reports/fuxi_alpha_generation_overhaul_20260926.md §M1）:
  1. 五态守卫: 仅 eval_status="valid" 且数值有限的事件可奖惩;
     data_insufficient / error / not_evaluated / precheck_rejected 永不奖惩;
  2. NaN / 缺失 jq_return 视为无读数(not_evaluated), 不得当 0 或负值;
  3. 默认 shadow: 只落台账, MAB 臂零改动 (只读数不生效);
  4. enforce: 写前 ensure_reward_source(换源重置) + legacy 记账(失败阈值/冷却);
  5. ralph_loop._mab_record_result 薄封装走同一通道 (唯一通道)。
"""
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

import reward_guard


class FakeArm:
    def __init__(self, direction_id, name="", reward_source="jq"):
        self.direction_id = direction_id
        self.name = name or direction_id
        self.reward_source = reward_source
        self.expected_reward = 0.0
        self.n_rewards = 0
        self.failures = 0
        self.successes = 0
        self.jq_results = []
        self.status = "active"


class FakeMab:
    def __init__(self, arms):
        self._arms = list(arms)
        self.saved = 0
        self.resets = []

    def resolve_arms(self, direction_name):
        return list(self._arms)

    def ensure_reward_source(self, d, source):
        if d.reward_source and d.reward_source != source:
            self.resets.append(d.direction_id)
            d.expected_reward = 0.0
            d.n_rewards = 0
            d.reward_source = source
            return True
        d.reward_source = source
        return False

    def save_state(self):
        self.saved += 1


def _ledger_lines(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


class RewardGuardM1Tests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self._tmp.name) / "reward_guard_ledger.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _consume(self, arm, *, env=None, **kw):
        mab = FakeMab([arm])
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": "shadow", **(env or {})}):
            res = reward_guard.consume_verified_reward(
                mab, "资金流", evidence=kw.pop("evidence", {}),
                ledger_path=self.ledger, **kw)
        return mab, res

    # ── 1) 五态守卫 ──────────────────────────────────────────
    def test_non_valid_states_never_rewarded(self):
        for status in ("data_insufficient", "error", "not_evaluated", "precheck_rejected"):
            arm = FakeArm("资金流__lightgbm_rank")
            mab, res = self._consume(
                arm, evidence={"eval_status": status, "value": 100.0},
                jq_return=100.0, jq_sharpe=1.0)
            self.assertEqual(res["status"], "no_reward", status)
            self.assertIsNone(res["reward"], status)
            self.assertFalse(res["applied"], status)
            self.assertEqual(arm.n_rewards, 0, status)
            self.assertEqual(arm.expected_reward, 0.0, status)
            self.assertEqual(mab.saved, 0, status)

    def test_nan_or_missing_value_is_no_reading(self):
        for jr in (float("nan"), None):
            arm = FakeArm("a")
            mab, res = self._consume(arm, evidence={}, jq_return=jr, jq_sharpe=None)
            self.assertEqual(res["eval_status"], "not_evaluated", repr(jr))
            self.assertEqual(res["status"], "no_reward")
            self.assertIsNone(res["reward"])
        arm = FakeArm("a")
        mab, res = self._consume(arm, evidence={"eval_status": "valid", "value": None})
        self.assertEqual(res["status"], "no_reward")
        self.assertIn("缺失", res["reason"])

    # ── 3) shadow 只读数不生效 ───────────────────────────────
    def test_shadow_default_is_read_only(self):
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": ""}):
            self.assertEqual(reward_guard.guard_mode(), "shadow")
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": "bogus"}):
            self.assertEqual(reward_guard.guard_mode(), "shadow")
        arm = FakeArm("资金流__lightgbm_rank", reward_source="model_proxy")
        arm.expected_reward = 0.5
        arm.n_rewards = 2
        mab, res = self._consume(
            arm, env={"FUXI_REWARD_GUARD": "shadow"},
            evidence={"eval_status": "valid", "value": 190.0, "ref": "f1"},
            jq_return=190.0, jq_sharpe=1.2)
        self.assertEqual(res["status"], "shadow_only")
        self.assertFalse(res["applied"])
        self.assertAlmostEqual(res["reward"], reward_guard.jq_reward_from_return(190.0), places=6)
        self.assertEqual(res["source_switch_needed"], ["资金流__lightgbm_rank"])
        # 臂零改动（只读数不生效）
        self.assertEqual(arm.n_rewards, 2)
        self.assertEqual(arm.expected_reward, 0.5)
        self.assertEqual(arm.reward_source, "model_proxy")
        self.assertEqual(arm.successes, 0)
        self.assertEqual(mab.saved, 0)
        self.assertEqual(mab.resets, [])
        recs = _ledger_lines(self.ledger)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "shadow_only")
        self.assertEqual(recs[0]["ref"], "f1")

    # ── 4) enforce 生效 + 量纲守卫 + legacy 记账 ────────────
    def test_enforce_applies_with_source_reset(self):
        arm = FakeArm("资金流__lightgbm_rank", reward_source="model_proxy")
        arm.expected_reward = 0.5
        arm.n_rewards = 2
        mab, res = self._consume(
            arm, env={"FUXI_REWARD_GUARD": "enforce"},
            evidence={"eval_status": "valid", "value": 190.0},
            jq_return=190.0, jq_sharpe=1.2)
        self.assertEqual(res["status"], "applied")
        self.assertTrue(res["applied"])
        self.assertEqual(mab.resets, ["资金流__lightgbm_rank"])  # 换源重置留痕
        self.assertEqual(arm.reward_source, "jq")
        self.assertEqual(arm.n_rewards, 1)  # reset 后首写
        self.assertAlmostEqual(arm.expected_reward, res["reward"], places=6)
        self.assertEqual(arm.successes, 1)
        self.assertEqual(len(arm.jq_results), 1)
        self.assertEqual(mab.saved, 1)

    def test_enforce_failure_rule_and_cooling(self):
        arm = FakeArm("a")
        for _ in range(3):
            mab, res = self._consume(
                arm, env={"FUXI_REWARD_GUARD": "enforce"},
                evidence={"eval_status": "valid", "value": -30.0},
                jq_return=-30.0, jq_sharpe=-1.0)
            self.assertEqual(res["status"], "applied")
        self.assertEqual(arm.failures, 3)
        self.assertEqual(arm.status, "cooling")
        self.assertEqual(arm.successes, 0)
        # sharpe 缺失 → 不猜成败（缺数不得当 0）
        arm2 = FakeArm("b")
        mab2, res2 = self._consume(
            arm2, env={"FUXI_REWARD_GUARD": "enforce"},
            evidence={"eval_status": "valid", "value": 10.0}, jq_return=10.0)
        self.assertEqual(res2["status"], "applied")
        self.assertEqual(arm2.successes, 0)
        self.assertEqual(arm2.failures, 0)

    # ── off / no_arms ───────────────────────────────────────
    def test_off_mode_takes_no_action_and_no_ledger(self):
        arm = FakeArm("a")
        mab, res = self._consume(
            arm, env={"FUXI_REWARD_GUARD": "off"},
            evidence={"eval_status": "valid", "value": 10.0},
            jq_return=10.0, jq_sharpe=1.0)
        self.assertEqual(res["status"], "off")
        self.assertFalse(self.ledger.exists())
        self.assertEqual(mab.saved, 0)

    def test_no_registered_arm_logged_not_rewarded(self):
        mab = FakeMab([])
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": "shadow"}):
            res = reward_guard.consume_verified_reward(
                mab, "未注册范式", evidence={"eval_status": "valid", "value": 10.0},
                jq_return=10.0, jq_sharpe=1.0, ledger_path=self.ledger)
        self.assertEqual(res["status"], "no_arms")
        self.assertEqual(mab.saved, 0)
        recs = _ledger_lines(self.ledger)
        self.assertEqual(recs[0]["status"], "no_arms")

    # ── 5) ralph 唯一通道 ───────────────────────────────────
    def test_step34_wired_into_jq_feedback(self):
        src = (FA / "ralph_loop.py").read_text(encoding="utf-8")
        self.assertIn("Step 3.4: (M1 影子半程) 奖励守卫", src)
        self.assertIn('"reward_guard": reward_guard_report', src)
        # 死函数复活为唯一通道: 定义 1 处 + jq_feedback 调用 1 处
        self.assertEqual(src.count("self._mab_record_result("), 1)

    def test_ralph_wrapper_is_the_single_channel(self):
        ralph_loop = importlib.import_module("ralph_loop")
        arm = FakeArm("资金流__lightgbm_rank")
        fake_self = SimpleNamespace(mab=FakeMab([arm]))
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": "shadow"}), \
             patch.object(reward_guard, "_DEFAULT_LEDGER", self.ledger):
            res = ralph_loop.RalphLoop._mab_record_result(fake_self, "资金流", 190.0, 1.2)
        self.assertEqual(res["status"], "shadow_only")
        self.assertEqual(res["status_source"], "inferred_from_value")
        self.assertEqual(arm.n_rewards, 0)  # shadow 不生效
        recs = _ledger_lines(self.ledger)
        self.assertEqual(len(recs), 1)
        # 显式非 valid 状态 → 五态守卫拦截
        with patch.dict(os.environ, {"FUXI_REWARD_GUARD": "shadow"}), \
             patch.object(reward_guard, "_DEFAULT_LEDGER", self.ledger):
            res2 = ralph_loop.RalphLoop._mab_record_result(
                fake_self, "资金流", 190.0, 1.2,
                evidence={"eval_status": "data_insufficient"})
        self.assertEqual(res2["status"], "no_reward")


if __name__ == "__main__":
    unittest.main()
