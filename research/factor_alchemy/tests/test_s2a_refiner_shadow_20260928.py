# -*- coding: utf-8 -*-
"""M6-A 精炼器影子遥测契约测试 (2026-09-28)

覆盖: 开关解析 / 8 事件分类 / would_fire 口径 / 结构签名双抽象 /
      子代归因 (主路 vs 子代, 五态分母 read+refused) / 账本 append+回读 /
      LOG=off 零写入 / 异常不抛 / 汇总三态与触发率分母 / 生产接线存在性 /
      "遥测不碰 MAB" 边界。

运行: C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe -X utf8
      research/factor_alchemy/tests/test_s2a_refiner_shadow_20260928.py
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import s2a_refiner_shadow as S  # noqa: E402

RALPH = BASE / "ralph_loop.py"


class _Res:
    """最小 ValidationResult 桩 (object 形态, 与生产同类)。"""

    def __init__(self, passed=False, status="valid", n_days=300,
                 forbidden=False, struct_dup=False):
        self.s1_passed = passed
        self.s1_eval_status = status
        self.s1_n_days = n_days
        self.s1_forbidden_hit = forbidden
        self.s1_struct_dup_hit = struct_dup


def _tmp_ledger(name="ledger.jsonl"):
    d = tempfile.mkdtemp(prefix="m6a_")
    return Path(d) / name


# ══════════════════════════════════════════════════════════════════
class TestSwitchResolution(unittest.TestCase):
    def test_log_default_is_on(self):
        self.assertEqual(S.resolve_log_mode({}), "on")
        self.assertEqual(S.DEFAULT_LOG_MODE, "on")

    def test_log_off(self):
        self.assertEqual(S.resolve_log_mode({S.LOG_ENV: "off"}), "off")
        self.assertEqual(S.resolve_log_mode({S.LOG_ENV: "OFF"}), "off")

    def test_log_illegal_falls_back_on(self):
        self.assertEqual(S.resolve_log_mode({S.LOG_ENV: "bogus"}), "on")

    def test_mode_default_shadow(self):
        self.assertEqual(S.resolve_mode_lenient({}), "shadow")
        self.assertEqual(S.resolve_mode_lenient({S.MODE_ENV: "on"}), "on")
        self.assertEqual(S.resolve_mode_lenient({S.MODE_ENV: "zzz"}), "shadow")

    def test_gate_env_numbers(self):
        self.assertEqual(S._gate_env_numbers({}), (3, 5))
        self.assertEqual(S._gate_env_numbers({S.EVERY_ENV: "0"}), (3, 1))
        self.assertEqual(S._gate_env_numbers({S.MIN_PULLS_ENV: "x"}), (3, 5))


# ══════════════════════════════════════════════════════════════════
class TestEventClassification(unittest.TestCase):
    def test_all_eight_events_reachable(self):
        g_off = {"mode": "off", "mature": False, "probe": False}
        g_shadow_miss = {"mode": "shadow", "mature": False, "probe": False, "every": 5}
        g_shadow_fire = {"mode": "shadow", "mature": True, "probe": False, "every": 5}
        g_on_miss = {"mode": "on", "mature": False, "probe": False}
        g_on = {"mode": "on", "mature": True, "probe": True}
        seen = {
            S.classify_refine_event(g_off),
            S.classify_refine_event(g_on_miss),
            S.classify_refine_event(g_shadow_miss),
            S.classify_refine_event(g_shadow_fire, pulls=25),
            S.classify_refine_event(g_on, n_seeds=0),
            S.classify_refine_event(g_on, n_seeds=2, n_kids=0),
            S.classify_refine_event(g_on, n_seeds=2, n_kids=3),
            S.classify_refine_event(g_on, error="boom"),
        }
        self.assertEqual(seen, set(S.REFINE_EVENTS))

    def test_period_gate_required_for_would_fire(self):
        # ★ 生产事实: gate["mature"] 不含周期门 ⇒ 非周期轮不得报 would_fire
        g = {"mode": "shadow", "mature": True, "probe": False, "every": 5}
        self.assertEqual(S.classify_refine_event(g, pulls=12), "shadow_gate_miss")
        self.assertEqual(S.classify_refine_event(g, pulls=25), "shadow_would_fire")

    def test_unknown_pulls_never_claims_would_fire(self):
        # pulls 未知 ⇒ 保守: 不得声称会触发 (防假读数)
        g = {"mode": "shadow", "mature": True, "probe": False, "every": 5}
        self.assertEqual(S.classify_refine_event(g), "shadow_gate_miss")
        self.assertEqual(S.fire_readiness(g, None), (True, None))
        self.assertEqual(S.fire_readiness(g, 25), (True, True))
        self.assertEqual(S.fire_readiness(g, 12), (True, False))

    def test_every_zero_clamps_like_production(self):
        # 生产 _s2a_refiner_gate 用 max(1, int(env or 5)) ⇒ 0 钳到 1 (每轮命中)
        g = {"mode": "shadow", "mature": True, "every": 0}
        self.assertEqual(S.fire_readiness(g, 7), (True, True))
        self.assertEqual(S.fire_readiness({"mode": "shadow", "mature": True}, 7),
                         (True, False))   # 缺 every ⇒ 默认 5, 7%5!=0
        self.assertEqual(S.fire_readiness({"mode": "shadow", "mature": True}, 5),
                         (True, True))

    def test_none_gate_is_not_off(self):
        # 门未求值 ⇒ 不得误读成"开关关闭"
        self.assertEqual(S.classify_refine_event(None, pulls=5), "shadow_gate_miss")
        self.assertEqual(S.classify_refine_event({"mode": "x"}, pulls=5),
                         "shadow_gate_miss")

    def test_error_wins_over_mode(self):
        g = {"mode": "off", "mature": True, "probe": True}
        self.assertEqual(S.classify_refine_event(g, error="e"), "error")


# ══════════════════════════════════════════════════════════════════
class TestRefineRecord(unittest.TestCase):
    def _rec(self, gate, **kw):
        return S.build_refine_record(gate=gate, env={}, **kw)

    def test_shadow_would_fire_requires_mature(self):
        r = self._rec({"mode": "shadow", "mature": True, "every": 5}, pulls=25)
        self.assertTrue(r["would_fire"])
        self.assertEqual(r["event"], "shadow_would_fire")
        r2 = self._rec({"mode": "shadow", "mature": False, "every": 5}, pulls=25)
        self.assertFalse(r2["would_fire"])

    def test_shadow_would_fire_requires_period(self):
        r = self._rec({"mode": "shadow", "mature": True, "every": 5}, pulls=12)
        self.assertFalse(r["would_fire"])
        self.assertEqual(r["event"], "shadow_gate_miss")

    def test_on_is_never_would_fire(self):
        r = self._rec({"mode": "on", "mature": True, "probe": True}, pulls=25,
                      n_seeds=2, n_kids=3)
        self.assertFalse(r["would_fire"])
        self.assertEqual(r["event"], "on_refined")
        self.assertEqual(r["arm"], S.ARM_ON)

    def test_off_still_logged_with_arm_off(self):
        r = self._rec({"mode": "off"}, pulls=0)
        self.assertEqual(r["event"], "off_skipped")
        self.assertEqual(r["arm"], "off")

    def test_every_one_fires_always(self):
        r = self._rec({"mode": "shadow", "mature": True, "every": 1}, pulls=7)
        self.assertTrue(r["would_fire"])

    def test_record_carries_direction_and_run_id(self):
        r = self._rec({"mode": "shadow", "mature": True, "every": 5}, pulls=25,
                      run_id="ralph_x", direction="筹码分布__x", paradigm="筹码分布",
                      generator="forge", n_main_candidates=10,
                      seed_sources="main:3|retr:2|dedup:4")
        for k in ("run_id", "direction", "paradigm", "generator",
                  "n_main_candidates", "seed_sources"):
            self.assertIn(k, r)
        self.assertEqual(r["run_id"], "ralph_x")
        self.assertEqual(r["schema"], S.SCHEMA)
        self.assertEqual(r["stage"], "refine")


# ══════════════════════════════════════════════════════════════════
class TestStructureKey(unittest.TestCase):
    def test_literal_distinguishes_field(self):
        a = S.structure_key("ts_mean(close,20)/ts_std(close,20)", "literal")
        b = S.structure_key("ts_mean(volume,20)/ts_std(volume,20)", "literal")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a, b)

    def test_field_abstracts_field(self):
        a = S.structure_key("ts_mean(close,20)/ts_std(close,20)", "field")
        b = S.structure_key("ts_mean(volume,20)/ts_std(volume,20)", "field")
        self.assertEqual(a, b)          # 骨架相同
        c = S.structure_key("ts_mean(close,5)/ts_std(close,5)", "field")
        self.assertNotEqual(a, c)       # 参数不同 ⇒ 骨架不同

    def test_invalid_returns_none_not_fallback(self):
        self.assertIsNone(S.structure_key("not python ((", "literal"))
        self.assertIsNone(S.structure_key("", "literal"))
        self.assertIsNone(S.structure_key(None, "literal"))


# ══════════════════════════════════════════════════════════════════
class TestEvalRecord(unittest.TestCase):
    def test_no_child_returns_none(self):
        rec = S.build_eval_record(
            [{"formula": "ts_mean(close,5)"}], [_Res()], env={})
        self.assertIsNone(rec)

    def test_main_child_split_and_novelty(self):
        cands = [
            {"formula": "ts_mean(close,20)/ts_std(close,20)"},
            {"formula": "ts_mean(volume,20)/ts_std(volume,20)", "_source": "gp_evolve"},
            {"formula": "ts_mean(close,20)/ts_std(close,20)", "_source": "gp_evolve"},
        ]
        res = [_Res(False, "valid"), _Res(True, "valid"),
               _Res(False, "data_insufficient", n_days=-1)]
        rec = S.build_eval_record(cands, res, env={})
        self.assertEqual(rec["n_main"], 1)
        self.assertEqual(rec["n_child"], 2)
        self.assertEqual(rec["child"]["s1_pass"], 1)
        self.assertEqual(rec["child"]["s1_read"], 1)
        self.assertEqual(rec["child"]["s1_refused"], 1)
        self.assertEqual(rec["child"]["s1_data_insufficient"], 1)
        self.assertEqual(rec["child"]["n_days_missing"], 1)
        # literal: 子代里 volume 版为新 ⇒ 1; close 版与主路同 ⇒ 不计
        self.assertEqual(rec["novel_literal"], 1)
        # field: 子代骨架与主路骨架相同 ⇒ 0
        self.assertEqual(rec["novel_field"], 0)
        self.assertIsNotNone(rec["sig_mode"])

    def test_dict_and_object_results_both_supported(self):
        cands = [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"}]
        r_obj = S.build_eval_record(cands, [_Res(True, "valid")], env={})
        r_dict = S.build_eval_record(
            cands, [{"s1_passed": True, "s1_eval_status": "valid"}], env={})
        self.assertEqual(r_obj["child"]["s1_pass"], 1)
        self.assertEqual(r_dict["child"]["s1_pass"], 1)

    def test_missing_result_counts_as_refused_not_pass(self):
        cands = [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"}]
        rec = S.build_eval_record(cands, [], env={})
        self.assertEqual(rec["child"]["s1_pass"], 0)
        self.assertEqual(rec["child"]["s1_refused"], 1)

    def test_rate_none_when_denominator_zero(self):
        cands = [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"}]
        rec = S.build_eval_record(cands, [], env={})
        self.assertIsNone(rec["s1_rate_main"])   # 主路 0 条 ⇒ 无分母, 不得写 0.0

    def test_shape_matches_production_zip(self):
        # 与 learning_feedback 同口径: 候选与结果按索引 zip
        cands = [{"formula": "a", "_source": "forge"},
                 {"formula": "b", "_source": "gp_evolve"}]
        res = [_Res(True, "valid"), _Res(False, "valid")]
        rec = S.build_eval_record(cands, res, env={})
        self.assertEqual(rec["main"]["s1_pass"], 1)     # 第 1 条是主路且 pass
        self.assertEqual(rec["child"]["s1_pass"], 0)    # 第 2 条是子代且 fail


# ══════════════════════════════════════════════════════════════════
class TestLedgerIO(unittest.TestCase):
    def test_refine_telemetry_writes_and_reads_back(self):
        p = _tmp_ledger()
        rep = S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 5},
            run_id="r1", direction="d", paradigm="p", generator="forge",
            pulls=25, ledger_path=p, env={})
        self.assertEqual(rep["status"], "logged")
        self.assertEqual(rep["event"], "shadow_would_fire")
        with io.open(p, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "r1")

    def test_log_off_writes_nothing(self):
        p = _tmp_ledger()
        rep = S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 5}, pulls=25,
            ledger_path=p, env={S.LOG_ENV: "off"})
        self.assertEqual(rep["status"], "skipped_switch_off")
        self.assertFalse(p.exists())

    def test_illegal_log_switch_reports_line(self):
        p = _tmp_ledger()
        rep = S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 5}, pulls=25,
            ledger_path=p, env={S.LOG_ENV: "wat"})
        self.assertEqual(rep["status"], "logged")
        self.assertTrue(any("非法" in l for l in rep["lines"]))

    def test_eval_telemetry_no_child_no_line(self):
        p = _tmp_ledger()
        rep = S.s2a_refiner_eval_telemetry(
            [{"formula": "ts_mean(close,5)"}], [_Res()], ledger_path=p, env={})
        self.assertEqual(rep["status"], "no_child")
        self.assertFalse(p.exists())

    def test_eval_telemetry_with_child_writes_line(self):
        p = _tmp_ledger()
        rep = S.s2a_refiner_eval_telemetry(
            [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"}],
            [_Res(True, "valid")], run_id="r2", ledger_path=p, env={})
        self.assertEqual(rep["status"], "logged")
        self.assertTrue(any("子代归因" in l for l in rep["lines"]))

    def test_append_never_raises_on_bad_entry(self):
        # 不可序列化值 → error 状态, 不抛出 (不静默: 有 lines)
        p = _tmp_ledger()
        bad = S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True}, pulls=5,
            ledger_path=p, env={})
        self.assertIn(bad["status"], ("logged", "error"))
        # 强制造错: 传入不可 JSON 化的 gate 字段
        rep = {"status": "x"}
        try:
            S.append_shadow_entry({"k": object()}, p)
            self.fail("应抛 TypeError (由调用方 try 包住)")
        except TypeError:
            pass

    def test_appended_lines_are_single_lf(self):
        p = _tmp_ledger()
        for i in range(3):
            S.s2a_refiner_refine_telemetry(
                gate={"mode": "shadow", "mature": True, "every": 1}, pulls=i + 1,
                run_id=f"r{i}", ledger_path=p, env={})
        b = io.open(p, "rb").read()
        self.assertEqual(b.count(b"\r\n"), 0)
        self.assertEqual(b.count(b"\n"), 3)


# ══════════════════════════════════════════════════════════════════
class TestSummarize(unittest.TestCase):
    def test_no_ledger_state(self):
        s = S.summarize_ledger(_tmp_ledger("absent.jsonl"))
        self.assertEqual(s["status"], "no_ledger")

    def test_empty_state(self):
        p = _tmp_ledger()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        s = S.summarize_ledger(p)
        self.assertEqual(s["status"], "empty")

    def test_valid_with_denominator_and_error_excluded(self):
        p = _tmp_ledger()
        S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 5}, pulls=25,
            paradigm="A", ledger_path=p, env={})              # would_fire
        S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 5}, pulls=26,
            paradigm="A", ledger_path=p, env={})              # gate_miss
        S.s2a_refiner_refine_telemetry(
            gate={"mode": "off"}, pulls=27, paradigm="B",
            ledger_path=p, env={})                            # off_skipped
        # 人为写一条 error 行
        S.append_shadow_entry(S.build_refine_record(
            gate={"mode": "shadow"}, pulls=29, error="boom", env={}), p)
        s = S.summarize_ledger(p)
        self.assertEqual(s["status"], "valid")
        self.assertEqual(s["n_refine_rows"], 4)
        self.assertEqual(s["fire"]["n_would_fire"], 1)
        # 分母 = 门被求值的轮次 (error 轮排除)
        self.assertEqual(s["fire"]["n_gated_rounds"], 3)
        self.assertAlmostEqual(s["fire"]["would_fire_rate"], 1 / 3)
        self.assertEqual(s["fire"]["by_paradigm"]["A"]["would_fire"], 1)

    def test_children_aggregation_and_rates(self):
        p = _tmp_ledger()
        S.s2a_refiner_eval_telemetry(
            [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"},
             {"formula": "ts_mean(close,9)", "_source": "gp_evolve"}],
            [_Res(True, "valid"), _Res(False, "data_insufficient", n_days=-1)],
            ledger_path=p, env={})
        s = S.summarize_ledger(p)
        c = s["children"]
        self.assertEqual(c["n_total"], 2)
        self.assertEqual(c["n_s1_pass"], 1)
        self.assertEqual(c["n_s1_read"], 1)
        self.assertEqual(c["n_s1_refused"], 1)
        self.assertAlmostEqual(c["s1_pass_rate_over_n"], 0.5)
        self.assertAlmostEqual(c["s1_pass_rate_over_read"], 1.0)

    def test_stage_ladder_recorded_for_children(self):
        # R5: 子代是否推进到 S2/S4/S5 (S5 分母带 read/refused)
        cands = [{"formula": "ts_mean(close,5)", "_source": "gp_evolve"},
                 {"formula": "ts_mean(open,5)", "_source": "gp_evolve"}]
        res = [SimpleNamespace(s1_passed=True, s1_eval_status="valid", s1_n_days=300,
                               s2_passed=True, s4_passed=True, s5_passed=True,
                               s5_eval_status="valid"),
               SimpleNamespace(s1_passed=False, s1_eval_status="valid", s1_n_days=300,
                               s2_passed=False, s4_passed=False, s5_passed=False,
                               s5_eval_status="data_insufficient")]
        rec = S.build_eval_record(cands, res, env={})
        self.assertEqual(rec["child"]["s2_pass"], 1)
        self.assertEqual(rec["child"]["s4_pass"], 1)
        self.assertEqual(rec["child"]["s5_pass"], 1)
        self.assertEqual(rec["child"]["s5_read"], 1)
        self.assertEqual(rec["child"]["s5_refused"], 1)

    def test_arm_filter(self):
        p = _tmp_ledger()
        S.s2a_refiner_refine_telemetry(
            gate={"mode": "shadow", "mature": True, "every": 1}, pulls=1,
            ledger_path=p, env={})
        S.s2a_refiner_refine_telemetry(
            gate={"mode": "on", "mature": True, "probe": True}, pulls=1,
            n_seeds=2, n_kids=3, ledger_path=p, env={S.MODE_ENV: "on"})
        self.assertEqual(S.summarize_ledger(p)["n_refine_rows"], 2)
        self.assertEqual(S.summarize_ledger(p, arm="on")["n_refine_rows"], 1)
        self.assertEqual(S.summarize_ledger(p, arm="shadow")["n_refine_rows"], 1)


# ══════════════════════════════════════════════════════════════════
class TestProductionWiring(unittest.TestCase):
    """测试形状须与生产一致: 直接读生产源码断言接线存在且成对。"""

    @classmethod
    def setUpClass(cls):
        cls.src = io.open(RALPH, "r", encoding="utf-8", newline="").read()

    def test_refine_telemetry_wired_twice(self):
        n = self.src.count("s2a_refiner_refine_telemetry")
        self.assertEqual(n, 2, f"门控遥测接线数 {n} (正常路径 + 异常路径)")

    def test_eval_telemetry_wired_once(self):
        self.assertEqual(self.src.count("s2a_refiner_eval_telemetry"), 1)

    def test_eval_wired_after_evaluate(self):
        i_eval = self.src.index("evaluate_result = self._phase_evaluate(")
        i_telem = self.src.index("s2a_refiner_eval_telemetry")
        i_distill = self.src.index("distill_result = self._phase_distill(")
        self.assertLess(i_eval, i_telem)
        self.assertLess(i_telem, i_distill)     # 必须在 D 阶段之前 (候选未被改写)

    def test_telemetry_does_not_touch_mab(self):
        # 遥测调用块内不得出现 mab / MAB 相关写入
        i = self.src.index("s2a_refiner_eval_telemetry")
        block = self.src[i - 200:i + 900]
        for bad in ("self.mab", "record_reward", "record_generator_reward"):
            self.assertNotIn(bad, block)

    def test_ledger_path_default_is_data_dir(self):
        self.assertEqual(S.DEFAULT_LEDGER.name, "s2a_refiner_shadow.jsonl")
        self.assertEqual(S.DEFAULT_LEDGER.parent.name, "data")

    def test_telemetry_signature_has_no_mab_param(self):
        import inspect
        sig = inspect.signature(S.s2a_refiner_eval_telemetry)
        self.assertNotIn("mab", sig.parameters)
        sig2 = inspect.signature(S.s2a_refiner_refine_telemetry)
        self.assertNotIn("mab", sig2.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
