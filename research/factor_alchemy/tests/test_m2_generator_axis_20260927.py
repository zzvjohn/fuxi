"""M2 (2026-09-27, P-20260927-005) 定向测试：生成器轴学习「写+读双开关」+ 离线回放。

契约（reports/fuxi_alpha_generation_overhaul_20260926.md §M2）:
  1. 写入端三态 off | shadow(默认) | on; shadow 只写观测账本, 绝不动 MAB 状态;
     旧的 raise+静默吞异常模式被「显式返回 + lines」制取代（任何路径都可见）;
  2. 读端开关 FUXI_GENERATOR_AXIS 与 FUXI_EVO_REFINER 解耦: EVO_REFINER=on
     不再启用生成器轴学习; 轴学习仅在 GENERATOR_AXIS=on 时应用; 默认
     shadow×shadow 时返回值与历史逐位一致（pulls 分档）;
  3. 回放引擎: 确定性; 策略只见已消费前缀（未选动作不猜收益）; 等预算同槽位;
     valid = fitness > -999 哨兵; 簇 = 骨架 AST 距离 >= 0.74 判独立;
     预注册判定规则可复现（无差异 → 保留均匀调度）。
"""
import importlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FA = ROOT / "research" / "factor_alchemy"
sys.path.insert(0, str(FA))

import generator_axis  # noqa: E402
import generator_axis_replay as gar  # noqa: E402

_ENV_KEYS = ["FUXI_GENERATOR_AXIS", "FUXI_EVO_REFINER",
             "FUXI_GENERATOR_AXIS_MIN_PULLS", "FUXI_EVO_REFINER_MIN_PULLS",
             "FUXI_GENERATOR_AXIS_EVERY", "FUXI_EVO_REFINER_EVERY",
             "FUXI_GENERATOR_AXIS_WRITE", "FUXI_LEGACY_GENERATOR_PROXY_REWARD"]


@contextmanager
def env_ctx(**kv):
    """隔离的 os.environ 上下文: 先清空 M2 相关键, 再设入给定值。"""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in kv.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Res:
    def __init__(self, icir):
        self.s1_icir = icir


class FakeArm:
    def __init__(self, name):
        self.name = name
        self.generator_rewards = {}


class FakeMab:
    def __init__(self, arms=None, simulate_missing=False, raise_on_record=False):
        self.arms = list(arms or [])
        self.calls = []
        self.simulate_missing = simulate_missing
        self.raise_on_record = raise_on_record

    def resolve_arms(self, paradigm):
        return list(self.arms)

    def record_generator_reward(self, identifier, generator, reward,
                                success=True, source="model_proxy"):
        if self.raise_on_record:
            raise RuntimeError("boom-record")
        self.calls.append({"identifier": identifier, "generator": generator,
                           "reward": reward, "success": success, "source": source})
        targets = self.arms[:1] if self.simulate_missing else self.arms
        for a in targets:
            a.generator_rewards.setdefault(generator, []).append(reward)
        return len(self.arms)


def _mk_obs(valid, expr="", positive=None):
    if positive is None:
        positive = valid
    return {"fitness": (0.1 if valid else -1000.0), "valid": bool(valid),
            "positive": bool(positive), "expression": expr}


# ══════════════════════════════════════════════════════════════════
# 1. 开关解析
# ══════════════════════════════════════════════════════════════════

class TestSwitchResolution(unittest.TestCase):
    def test_write_mode_default_and_explicit(self):
        self.assertEqual(generator_axis.resolve_write_mode({}), "shadow")
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_GENERATOR_AXIS_WRITE": "off"}), "off")
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_GENERATOR_AXIS_WRITE": "on"}), "on")
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_GENERATOR_AXIS_WRITE": "shadow"}), "shadow")

    def test_write_mode_legacy_alias_and_precedence(self):
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_LEGACY_GENERATOR_PROXY_REWARD": "on"}), "on")
        # 新开关优先于旧别名
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_GENERATOR_AXIS_WRITE": "off",
             "FUXI_LEGACY_GENERATOR_PROXY_REWARD": "on"}), "off")

    def test_write_mode_invalid_falls_back_shadow(self):
        self.assertEqual(generator_axis.resolve_write_mode(
            {"FUXI_GENERATOR_AXIS_WRITE": "banana"}), "shadow")

    def test_read_mode_resolution(self):
        self.assertEqual(generator_axis.resolve_read_mode({}), "shadow")
        self.assertEqual(generator_axis.resolve_read_mode(
            {"FUXI_GENERATOR_AXIS": "off"}), "off")
        self.assertEqual(generator_axis.resolve_read_mode(
            {"FUXI_GENERATOR_AXIS": "on"}), "on")
        # 解耦: 即使 EVO_REFINER=on, 读开关未被设置也必须是 shadow
        self.assertEqual(generator_axis.resolve_read_mode(
            {"FUXI_EVO_REFINER": "on"}), "shadow")
        self.assertEqual(generator_axis.resolve_read_mode(
            {"FUXI_GENERATOR_AXIS": "weird"}), "shadow")


# ══════════════════════════════════════════════════════════════════
# 2. 写入端三态
# ══════════════════════════════════════════════════════════════════

class TestWriteSide(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self._tmp.name) / "axis_shadow.jsonl"
        self.er = {"results": [Res(0.4), Res(0.2)], "stage5_passed": 0,
                   "eligible_for_jq": 0}

    def tearDown(self):
        self._tmp.cleanup()

    def test_off_explicit_skip_visible(self):
        mab = FakeMab([FakeArm("a1")])
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "off"}, ledger_path=self.ledger)
        self.assertEqual(rep["mode"], "off")
        self.assertEqual(rep["status"], "skipped")
        self.assertTrue(rep["lines"] and "off" in rep["lines"][0])
        self.assertFalse(self.ledger.exists())
        self.assertEqual(mab.calls, [])

    def test_shadow_writes_ledger_only(self):
        mab = FakeMab([FakeArm("a1"), FakeArm("a2")])
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            loop_id="run-1", env={}, ledger_path=self.ledger)
        self.assertEqual(rep["mode"], "shadow")
        self.assertEqual(rep["status"], "shadow_logged")
        self.assertEqual(mab.calls, [])           # 绝不动 MAB
        self.assertTrue(self.ledger.exists())
        line = self.ledger.read_text(encoding="utf-8")
        self.assertTrue(line.endswith("\n"))
        entry = json.loads(line.strip())
        exp_reward = round(math.tanh(0.4 * 3.0) * 0.3, 6)
        self.assertAlmostEqual(entry["reward"], exp_reward, places=6)
        self.assertEqual(entry["icir_max"], 0.4)
        self.assertEqual(entry["n_icir"], 2)
        self.assertEqual(entry["n_arms_expected"], 2)
        self.assertEqual(entry["mode"], "shadow")
        self.assertEqual(entry["run_id"], "run-1")
        self.assertEqual(entry["source"], "model_proxy")
        self.assertFalse(entry["success"])

    def test_shadow_append_only_two_lines(self):
        mab = FakeMab([])
        for i in range(2):
            generator_axis.s2a_generator_axis_write(
                mab, "forge", "动量反转", self.er, env={},
                ledger_path=self.ledger)
        lines = self.ledger.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        for ln in lines:
            json.loads(ln)  # 每行都是合法 JSON

    def test_shadow_no_icir_no_write(self):
        mab = FakeMab([])
        er = {"results": [Res(0.0), Res(None), Res("bad")], "stage5_passed": 0}
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", er, env={}, ledger_path=self.ledger)
        self.assertEqual(rep["status"], "no_data")
        self.assertFalse(self.ledger.exists())
        self.assertEqual(mab.calls, [])

    def test_on_records_and_broadcast_check(self):
        arms = [FakeArm("a1"), FakeArm("a2")]
        mab = FakeMab(arms)
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "on"}, ledger_path=self.ledger)
        self.assertEqual(rep["status"], "recorded")
        self.assertEqual(rep["n_arms"], 2)
        self.assertEqual(len(mab.calls), 1)
        c = mab.calls[0]
        self.assertEqual(c["identifier"], "动量反转")
        self.assertEqual(c["generator"], "forge")
        self.assertAlmostEqual(c["reward"], math.tanh(1.2) * 0.3, places=9)
        self.assertEqual(c["source"], "model_proxy")
        joined = "\n".join(rep["lines"])
        self.assertIn("生成器轴记账", joined)
        self.assertIn("范式广播: 本应 2 臂 / 实际 2 臂", joined)
        self.assertFalse(self.ledger.exists())    # on 模式不写影子账本

    def test_on_detects_missing_arm(self):
        arms = [FakeArm("a1"), FakeArm("a2")]
        mab = FakeMab(arms, simulate_missing=True)
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "on"})
        self.assertEqual(rep["status"], "recorded")
        self.assertEqual(rep["missing_arms"], ["a2"])
        self.assertIn("未落臂", "\n".join(rep["lines"]))

    def test_on_no_arms_hit_warns(self):
        mab = FakeMab([])
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "on"})
        self.assertEqual(rep["status"], "recorded")
        self.assertEqual(rep["n_arms"], 0)
        self.assertIn("未命中臂", "\n".join(rep["lines"]))

    def test_on_record_exception_returns_error_not_raise(self):
        mab = FakeMab([FakeArm("a1")], raise_on_record=True)
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "on"})
        self.assertEqual(rep["status"], "error")
        self.assertIn("boom-record", rep["error"])
        self.assertTrue(rep["lines"])            # 不静默

    def test_legacy_alias_enables_mab_write(self):
        mab = FakeMab([FakeArm("a1")])
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_LEGACY_GENERATOR_PROXY_REWARD": "on"})
        self.assertEqual(rep["mode"], "on")
        self.assertEqual(len(mab.calls), 1)

    def test_invalid_env_warns_but_defaults_shadow(self):
        mab = FakeMab([])
        rep = generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", self.er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "banana"},
            ledger_path=self.ledger)
        self.assertEqual(rep["mode"], "shadow")
        self.assertTrue(any("非法" in ln for ln in rep["lines"]))

    def test_write_never_touches_decision_fields(self):
        mab = FakeMab([FakeArm("a1")])
        er = {"results": [Res(0.4)], "stage5_passed": 2, "eligible_for_jq": 1,
              "candidates": [{"factor_name": "x"}]}
        snapshot = json.dumps(er, default=str, sort_keys=True)
        generator_axis.s2a_generator_axis_write(
            mab, "forge", "动量反转", er,
            env={"FUXI_GENERATOR_AXIS_WRITE": "on"}, ledger_path=self.ledger)
        self.assertEqual(json.dumps(er, default=str, sort_keys=True), snapshot)

    def test_module_source_has_no_delete_calls(self):
        src = (FA / "generator_axis.py").read_text(encoding="utf-8")
        for bad in ("unlink(", "rmtree(", "os.remove", "shutil.move"):
            self.assertNotIn(bad, src)


# ══════════════════════════════════════════════════════════════════
# 3. 读端（_mab_select_generator 解耦; 免实例化 unbound 调用）
# ══════════════════════════════════════════════════════════════════

class TestReadSide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ralph_loop = importlib.import_module("ralph_loop")
        cls.fn = cls.ralph_loop.RalphLoop._mab_select_generator

    def _call(self, direction):
        buf = io.StringIO()
        with redirect_stdout(buf):
            # 注意: 必须经类访问取裸函数 —— 若写成 self.fn(...)，
            # 函数被当作本测试类的方法而自动绑定，实测为 3 参调用 (test 单例成 self)。
            ret = type(self).fn(None, direction)   # 该函数零 self 引用
        return ret, buf.getvalue()

    @staticmethod
    def _dir(pulls=4, rewards=None):
        return SimpleNamespace(
            name="动量反转",
            pulls=pulls,
            generator_rewards=rewards if rewards is not None else {
                "gp_breed": [0.2], "llm": [0.9], "forge": [0.1],
                "gp_evolve": [0.3]},
        )

    def test_default_shadow_returns_legacy_tier_with_readout(self):
        with env_ctx():
            ret, out = self._call(self._dir(pulls=5))
        self.assertEqual(ret, "forge")            # pulls>3 分档
        self.assertIn("[S2a/shadow]", out)
        self.assertIn("FUXI_GENERATOR_AXIS 当前=shadow", out)

    def test_decoupled_from_evo_refiner(self):
        # EVO_REFINER=on 不得再启用生成器轴 (历史耦合点): 返回值仍为 pulls 分档
        with env_ctx(FUXI_EVO_REFINER="on"):
            ret, out = self._call(self._dir(pulls=4))
        self.assertEqual(ret, "forge")
        self.assertIn("FUXI_GENERATOR_AXIS 当前=shadow", out)

    def test_axis_on_applies_without_evo_refiner(self):
        with env_ctx(FUXI_GENERATOR_AXIS="on", FUXI_EVO_REFINER="off"), \
                patch("random.random", return_value=0.99):
            ret, _ = self._call(self._dir(pulls=4))
        self.assertEqual(ret, "llm")              # 学习支生效: llm 均值最优

    def test_axis_on_probe_routes_gp_evolve(self):
        with env_ctx(FUXI_GENERATOR_AXIS="on", FUXI_GENERATOR_AXIS_EVERY="5"):
            ret, out = self._call(self._dir(pulls=5))
        self.assertEqual(ret, "gp_evolve")        # 周期探针
        self.assertIn("周期精炼探针", out)

    def test_axis_off_silent_legacy(self):
        with env_ctx(FUXI_GENERATOR_AXIS="off"):
            ret, out = self._call(self._dir(pulls=5))
        self.assertEqual(ret, "forge")
        self.assertNotIn("[S2a/shadow]", out)

    def test_min_pulls_new_name_and_old_name_fallback(self):
        d = self._dir(pulls=4)
        with env_ctx(FUXI_GENERATOR_AXIS="on"):
            with patch("random.random", return_value=0.99):
                ret_new_default, _ = self._call(d)
        self.assertEqual(ret_new_default, "llm")  # 默认 min_pulls=3 → 成熟
        with env_ctx(FUXI_GENERATOR_AXIS="on", FUXI_GENERATOR_AXIS_MIN_PULLS="10"):
            ret_new, _ = self._call(d)
        self.assertEqual(ret_new, "forge")        # 未成熟 → pulls 分档
        with env_ctx(FUXI_GENERATOR_AXIS="on", FUXI_EVO_REFINER_MIN_PULLS="10"):
            ret_old, _ = self._call(d)
        self.assertEqual(ret_old, "forge")        # 旧名回落同样生效

    def test_shadow_return_equals_off_across_pulls(self):
        # 行为保持: 默认 (shadow) 与显式 off 的返回值在所有 pulls 档位逐一相同
        for pulls in (0, 1, 2, 3, 4, 5, 10, 15):
            with env_ctx():
                r_default, _ = self._call(self._dir(pulls=pulls))
            with env_ctx(FUXI_GENERATOR_AXIS="off"):
                r_off, _ = self._call(self._dir(pulls=pulls))
            self.assertEqual(r_default, r_off, msg=f"pulls={pulls}")

    def test_invalid_read_mode_falls_back_shadow(self):
        with env_ctx(FUXI_GENERATOR_AXIS="weird"):
            ret, out = self._call(self._dir(pulls=4))
        self.assertEqual(ret, "forge")
        self.assertIn("FUXI_GENERATOR_AXIS 当前=shadow", out)


# ══════════════════════════════════════════════════════════════════
# 4. 回放引擎
# ══════════════════════════════════════════════════════════════════

class TestReplayEngine(unittest.TestCase):
    def test_obs_sentinel_semantics(self):
        f = gar._obs_from_candidate
        self.assertTrue(f({"fitness": 0.1})["valid"])
        self.assertTrue(f({"fitness": 0.1})["positive"])
        self.assertTrue(f({"fitness": -0.2})["valid"])
        self.assertFalse(f({"fitness": -0.2})["positive"])
        self.assertFalse(f({"fitness": -1000.0})["valid"])
        self.assertFalse(f({"fitness": None})["valid"])
        self.assertFalse(f({"fitness": "abc"})["valid"])
        self.assertFalse(f({"fitness": float("nan")})["valid"])

    def test_load_and_validate_synthetic(self):
        data = {"rows": [
            {"seed": 1, "unique_budget": 2,
             "forge": {"candidates": [{"fitness": 0.1, "expression": "a(x, 1)"},
                                      {"fitness": -1000, "expression": "b"}]},
             "quantum": {"candidates": [{"fitness": 0.2, "expression": "c(x)"},
                                        {"fitness": 0.3, "expression": "d(x)"}]}},
            {"seed": 2, "unique_budget": 2,
             "forge": {"candidates": [{"fitness": 0.0, "expression": "e(x)"},
                                      {"fitness": 0.1, "expression": "f(x)"}]},
             "quantum": {"candidates": [{"fitness": 0.2, "expression": "g(x)"},
                                        {"fitness": 0.3, "expression": "h(x)"}]}},
        ]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fb.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            m = gar.load_full_batch(p)
        self.assertEqual(m["arms"], ["forge", "quantum"])
        self.assertEqual(len(m["seeds"]), 2)
        self.assertTrue(gar.validate_matrix(m)["ok"])
        # 人为破坏配对 → 校验须报 unpaired
        m2 = json.loads(json.dumps(m))
        m2["seeds"][0]["arms"]["quantum"] = m2["seeds"][0]["arms"]["quantum"][:1]
        issues = gar.validate_matrix(m2)
        self.assertFalse(issues["ok"])
        self.assertEqual(issues["issues"][0]["kind"], "unpaired_arm_lengths")

    def test_uniform_round_robin_and_cap_skip(self):
        arms = {"A": [_mk_obs(True, f"A{i}") for i in range(2)],
                "B": [_mk_obs(True, f"B{i}") for i in range(4)]}
        pol = gar.make_uniform_policy(list(arms.keys()))
        res = gar.simulate(arms, pol, 6)
        self.assertEqual(res["consumed_total"], 6)
        self.assertEqual(res["per_arm_slots"], {"A": 2, "B": 4})
        self.assertEqual([a for _, a in res["trace"]], ["A", "B", "A", "B", "B", "B"])

    def test_mature_warmup_then_exploit(self):
        arms = {"A": [_mk_obs(i < 7, f"A{i}") for i in range(10)],
                "B": [_mk_obs(False, f"B{i}") for i in range(10)]}
        pol = gar.make_mature_policy(list(arms.keys()), warmup=3)
        res = gar.simulate(arms, pol, 10)
        # warmup: A,B,A,B,A,B (3 臂序轮转); exploit: A 率 1.0 恒选至配额边界
        self.assertEqual([a for _, a in res["trace"][:6]], ["A", "B", "A", "B", "A", "B"])
        self.assertEqual(res["per_arm_slots"], {"A": 7, "B": 3})
        self.assertEqual(res["valid"], 7)

    def test_mature_fallback_when_best_exhausted(self):
        arms = {"X": [_mk_obs(True, f"X{i}") for i in range(2)],
                "Y": [_mk_obs(False, f"Y{i}") for i in range(5)]}
        pol = gar.make_mature_policy(list(arms.keys()), warmup=1)
        res = gar.simulate(arms, pol, 7)
        self.assertEqual(res["per_arm_slots"], {"X": 2, "Y": 5})
        self.assertEqual(res["valid"], 2)

    def test_no_lookahead_prefix_only(self):
        arms = {"A": [_mk_obs(i % 2 == 0, f"A{i}") for i in range(8)],
                "B": [_mk_obs(i % 3 == 0, f"B{i}") for i in range(8)]}
        seen = []

        def spy(state):
            t = state["t"]
            tot = sum(len(v) for v in state["consumed"].values())
            self.assertEqual(tot, t)                       # 只见已消费前缀
            for a, seq in arms.items():
                k = len(state["consumed"][a])
                self.assertEqual(state["consumed"][a], seq[:k])  # 严格前缀
            seen.append(t)
            return "A" if t % 2 == 0 else "B"

        res = gar.simulate(arms, spy, 8)
        self.assertEqual(seen, list(range(8)))
        self.assertEqual(res["consumed_total"], 8)

    def test_cluster_index_threshold_semantics(self):
        ci = gar._ClusterIndex(0.74)
        self.assertTrue(ci.add("ts_mean(close, 5)"))
        self.assertFalse(ci.add("ts_mean(close, 20)"))   # 同骨架参数变体 → 同簇
        self.assertTrue(ci.add("sqrt(ts_std(ts_sum(returns, 5), 5))"))
        self.assertFalse(ci.add("ts_mean(close, 5)"))    # 重复 → 同簇
        self.assertEqual(ci.count, 2)
        # 空表达式不计入, 但留痕
        self.assertFalse(ci.add(""))
        self.assertEqual(ci.n_empty, 1)

    def test_budget_integrity_frac1_full_consumption(self):
        arms = {"A": [_mk_obs(True, f"A{i}") for i in range(4)],
                "B": [_mk_obs(True, f"B{i}") for i in range(4)]}
        matrix = {"seeds": [{"seed": 7, "arms": arms}], "arms": ["A", "B"]}
        policies = {"uniform": gar.make_uniform_policy(["A", "B"]),
                    "mature": gar.make_mature_policy(["A", "B"], warmup=2)}
        res = gar.evaluate_matrix(matrix, policies=policies,
                                  budget_fracs=(1.0,), primary_frac=1.0)
        s = res["per_seed"][0]["by_frac"]["1.0"]
        self.assertEqual(s["budget"], 8)
        for name in ("uniform", "mature"):
            r = s["policies"][name]
            self.assertEqual(r["consumed_total"], 8)
            self.assertEqual(r["per_arm_slots"], {"A": 4, "B": 4})
            self.assertEqual(r["valid"], 8)
        self.assertEqual(res["aggregate"]["1.0"]["delta_valid"]["median"], 0)

    def test_verdict_advantage_scenario(self):
        arms = {"A": [_mk_obs(i < 7, f"A{i}") for i in range(10)],
                "B": [_mk_obs(False, f"B{i}") for i in range(10)]}
        matrix = {"seeds": [{"seed": 1, "arms": arms}], "arms": ["A", "B"]}
        policies = {"uniform": gar.make_uniform_policy(["A", "B"]),
                    "mature": gar.make_mature_policy(["A", "B"], warmup=3)}
        res = gar.evaluate_matrix(matrix, policies=policies,
                                  budget_fracs=(0.5,), primary_frac=0.5)
        v = res["verdict"]
        self.assertTrue(v["mature_advantage"])
        self.assertEqual(v["action"], "mature_allocation")
        self.assertFalse(v["stop_rule_triggered"])
        agg = res["aggregate"]["0.5"]
        self.assertGreater(agg["delta_valid"]["median"], 0)
        self.assertGreaterEqual(agg["delta_clusters"]["median"], 0)

    def test_verdict_keep_uniform_when_no_difference(self):
        arms = {"A": [_mk_obs(False, f"A{i}") for i in range(10)],
                "B": [_mk_obs(False, f"B{i}") for i in range(10)]}
        matrix = {"seeds": [{"seed": 1, "arms": arms}], "arms": ["A", "B"]}
        policies = {"uniform": gar.make_uniform_policy(["A", "B"]),
                    "mature": gar.make_mature_policy(["A", "B"], warmup=3)}
        res = gar.evaluate_matrix(matrix, policies=policies,
                                  budget_fracs=(0.5,), primary_frac=0.5)
        v = res["verdict"]
        self.assertFalse(v["mature_advantage"])
        self.assertEqual(v["action"], "keep_uniform")
        self.assertTrue(v["stop_rule_triggered"])   # 停止规则: 无差异 → 保留均匀

    def test_simulate_deterministic(self):
        arms = {"A": [_mk_obs(i % 2 == 0, f"A{i}") for i in range(6)],
                "B": [_mk_obs(i % 3 == 0, f"B{i}") for i in range(6)]}
        pol = gar.make_mature_policy(["A", "B"], warmup=2)
        r1 = gar.simulate(arms, pol, 9)
        r2 = gar.simulate(arms, pol, 9)
        self.assertEqual(r1, r2)

    def test_preregistered_constants_frozen(self):
        self.assertEqual(gar.INDEP_MIN_DIST, 0.74)
        self.assertEqual(gar.WARMUP_PER_ARM, 3)
        self.assertEqual(gar.PRIMARY_FRAC, 0.5)
        self.assertEqual(tuple(gar.BUDGET_FRACS), (0.25, 0.5, 0.75, 1.0))
        self.assertIn("未选动作不猜收益", gar.DECISION_RULE)

    def test_real_matrix_smoke(self):
        p = ROOT / "reports" / "fuxi_generator_full_batch_20260926" / "full_batch.json"
        if not p.exists():
            self.skipTest("full_batch.json 不存在")
        m = gar.load_full_batch(p)
        self.assertTrue(gar.validate_matrix(m)["ok"])
        self.assertEqual(len(m["seeds"]), 5)
        self.assertEqual(len(m["arms"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
