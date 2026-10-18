# -*- coding: utf-8 -*-
"""M10 定向测试 (2026-09-27): forbidden 豁免机制 + 翻默认 + struct_dedup 接线 + 顺序修复。

四组:
  1. TestForbiddenExempt   — 豁免白名单机制 (真实清单 + 合成变体 + 关闭/缺失容错)
  2. TestForbiddenDefault  — 默认翻 enforce + 显式回退 + 路径解析
  3. TestStructDupWiring   — ralph_loop 打标 (源码锚) + validator 消费 (合成候选)
  4. TestOrderFix          — 符号守卫→forbidden→struct 顺序断言 (源码锚)

跑法: python -X utf8 -m unittest tests.test_m10_gates_20260927 -v
"""
import contextlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

RA = Path(__file__).resolve().parent.parent
if str(RA) not in sys.path:
    sys.path.insert(0, str(RA))

import forbidden_gate as fg                                  # noqa: E402
import multi_stage_validator as msv                          # noqa: E402

_ENV_KEYS = ("FUXI_FORBIDDEN_GATE", "FUXI_FORBIDDEN_EXEMPT",
             "FUXI_STRUCT_DEDUP", "FUXI_EVO_REFINER",
             "FUXI_STRUCT_DEDUP_THRESHOLD", "FUXI_STRUCT_DEDUP_ABSTRACTION")

DVS_EXPR = "-(volume_p * close_p).rolling(20).std() / (volume_p * close_p).rolling(20).mean()"
MDD_EXPR = "(close_p / close_p.rolling(120).max()).rolling(20).mean()"
# dollar_vol_stability 变体: 窗口不同 (10/40), 应仍命中豁免 (近似)
DVS_VARIANT = "-(volume_p * close_p).rolling(10).std() / (volume_p * close_p).rolling(40).mean()"


@contextlib.contextmanager
def env_ctx(**kv):
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _gate(**kw):
    g = fg.ForbiddenGate(**kw)
    g.load()
    return g


CORPUS = [
    # 命中黑名单 + 豁免
    {"factor_name": "dollar_vol_stability", "formula": DVS_EXPR},
    # 命中黑名单 + 不豁免
    {"factor_name": "max_drawdown_duration", "formula": MDD_EXPR},
    # 干净
    {"factor_name": "clean_one", "formula": "(close / close.shift(20) - 1).rolling(5).mean()"},
]


class TestForbiddenExempt(unittest.TestCase):
    def test_real_exempt_manifest_loads(self):
        with env_ctx():
            g = _gate()
        self.assertEqual(len(g.exempt_entries), 1)
        self.assertIn("dollar_vol_stability",
                      json.dumps(list(g.exempt_entries.values()), ensure_ascii=False))

    def test_exempted_semantics_dvs(self):
        with env_ctx():                       # 默认 enforce
            g = _gate()
        r = g.check(DVS_EXPR)
        self.assertTrue(r["blocked"])          # 命中黑名单是事实
        self.assertTrue(r["exempted"])         # 但豁免
        self.assertFalse(r["enforced"])        # 不硬拒
        self.assertIn("豁免", r["reason"])
        self.assertEqual(r["exempt_factor"], "dollar_vol_stability")

    def test_not_exempted_for_jq_falsified(self):
        with env_ctx():
            g = _gate()
        r = g.check(MDD_EXPR)
        self.assertTrue(r["blocked"])
        self.assertFalse(r["exempted"])
        self.assertTrue(r["enforced"])         # enforce 默认 ⇒ 硬拒

    def test_exempt_variant_matches_approx(self):
        with env_ctx():
            g = _gate()
        r = g.check(DVS_VARIANT)
        self.assertTrue(r["exempted"])
        self.assertFalse(r["enforced"])

    def test_exempt_off_disables(self):
        with env_ctx(FUXI_FORBIDDEN_EXEMPT="off"):
            g = _gate()
        r = g.check(DVS_EXPR)
        self.assertTrue(r["blocked"])
        self.assertFalse(r["exempted"])
        self.assertTrue(r["enforced"])         # 无豁免 ⇒ enforce 下硬拒

    def test_exempt_missing_file_no_error(self):
        with env_ctx(FUXI_FORBIDDEN_EXEMPT=str(Path(tempfile.gettempdir()) / "no_such_exempt.json")):
            g = _gate()
        self.assertEqual(len(g.exempt_entries), 0)
        self.assertEqual(g.exempt_error, "")
        r = g.check(DVS_EXPR)
        self.assertTrue(r["enforced"])

    def test_scan_library_exempt_split(self):
        with env_ctx():
            g = _gate()
        rep = g.scan_library(CORPUS)
        self.assertEqual(rep["n_scanned"], 3)
        self.assertEqual(rep["n_conflict"], 2)
        self.assertEqual(rep["n_exempted"], 1)
        self.assertEqual(rep["n_blocking"], 1)
        names = {c["factor_name"]: c["exempted"] for c in rep["conflicts"]}
        self.assertTrue(names["dollar_vol_stability"])
        self.assertFalse(names["max_drawdown_duration"])

    def test_exempt_keeps_info_visible(self):
        # 豁免不隐藏: matched / direction_id / similarity 全部保留
        with env_ctx():
            g = _gate()
        r = g.check(DVS_EXPR)
        self.assertTrue(r["matched"])
        self.assertTrue(r["direction_id"].startswith("jq_failure::"))
        self.assertGreaterEqual(r["similarity"], 0.95)
        self.assertGreater(r["exempt_similarity"], 0.9)

    def test_stats_exempted_counter(self):
        with env_ctx():
            g = _gate()
        g.check(DVS_EXPR)
        self.assertEqual(g._stats["exempted"], 1)
        self.assertEqual(g._stats["checked"], 1)


class TestForbiddenDefault(unittest.TestCase):
    def test_default_mode_enforce(self):
        with env_ctx():
            g = _gate()
        self.assertEqual(g.mode, "enforce")
        self.assertEqual(fg._resolve_mode(), "enforce")

    def test_explicit_shadow_fallback(self):
        with env_ctx(FUXI_FORBIDDEN_GATE="shadow"):
            g = _gate()
        self.assertEqual(g.mode, "shadow")
        r = g.check(MDD_EXPR)
        self.assertTrue(r["blocked"])
        self.assertFalse(r["enforced"])        # shadow: 标而不拒

    def test_explicit_off_fallback(self):
        with env_ctx(FUXI_FORBIDDEN_GATE="off"):
            g = _gate()
        self.assertEqual(g.mode, "off")

    def test_invalid_falls_back_enforce(self):
        with env_ctx(FUXI_FORBIDDEN_GATE="weird"):
            g = _gate()
        self.assertEqual(g.mode, "enforce")

    def test_exempt_path_resolution(self):
        with env_ctx():
            self.assertEqual(fg._resolve_exempt_path(), fg.DEFAULT_EXEMPT_PATH)
        with env_ctx(FUXI_FORBIDDEN_EXEMPT="off"):
            self.assertIsNone(fg._resolve_exempt_path())
        with env_ctx(FUXI_FORBIDDEN_EXEMPT=r"x:\custom.json"):
            self.assertEqual(fg._resolve_exempt_path(), Path(r"x:\custom.json"))
        self.assertEqual(fg._resolve_exempt_path(Path("y.json")), Path("y.json"))


class TestStructDupWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rl_src = (RA / "ralph_loop.py").read_text(encoding="utf-8")

    def test_ralph_loop_has_struct_marker(self):
        self.assertIn('cand["_struct_dup_hit"] = {', self.rl_src)
        self.assertIn('cand.pop("_struct_dup_hit", None)', self.rl_src)

    def test_ralph_loop_uses_public_wrapper(self):
        self.assertIn("get_indexed_deductor()", self.rl_src)
        self.assertIn('os.environ.get("FUXI_STRUCT_DEDUP", "shadow")', self.rl_src)

    def _run_stage1(self, cand, hit):
        c = {"formula": "close / close.shift(1) - 1", "natural_freq": "daily"}
        c.update(cand or {})
        if hit is not None:
            c["_struct_dup_hit"] = hit
        v = msv.MultiStageValidator()
        results = [msv.ValidationResult(factor_name="c0", formula=c["formula"])]
        out = v.stage1_fast_ic([c], results)
        return v, out[0]

    def test_validator_rejects_on_would_reject(self):
        _, r = self._run_stage1(None, {"would_reject": True, "reason": "dup-test"})
        self.assertTrue(r.s1_struct_dup_hit)
        self.assertFalse(r.s1_passed)
        self.assertEqual(r.s1_eval_status, "not_evaluated")
        self.assertIn("AST 结构重复硬拒", r.s1_reason)

    def test_validator_shadow_only_marks(self):
        # would_reject=False (shadow) → 只标记, 不因它拒
        _, r = self._run_stage1(None, {"would_reject": False, "reason": "shadow-dup"})
        self.assertTrue(r.s1_struct_dup_hit)
        self.assertIn("shadow-dup", r.s1_struct_dup_reason)
        self.assertNotIn("AST 结构重复硬拒", r.s1_reason)

    def test_validator_counters(self):
        v, _ = self._run_stage1(None, {"would_reject": True, "reason": "x"})
        self.assertEqual(v._s1_struct_dup_hits, 1)
        self.assertEqual(v._s1_struct_dup_rejected, 1)
        v2, _ = self._run_stage1(None, {"would_reject": False, "reason": "x"})
        self.assertEqual(v2._s1_struct_dup_hits, 1)
        self.assertEqual(v2._s1_struct_dup_rejected, 0)

    def test_validator_no_hit_untouched(self):
        _, r = self._run_stage1(None, None)
        self.assertFalse(r.s1_struct_dup_hit)
        self.assertEqual(r.s1_struct_dup_reason, "")

    def test_validator_to_dict_has_fields(self):
        r = msv.ValidationResult(factor_name="t", formula="x")
        d = r.to_dict()["s1"]
        self.assertIn("struct_dup_hit", d)
        self.assertIn("struct_dup_reason", d)

    def test_structure_dedup_public_wrapper(self):
        import structure_dedup as sd
        self.assertTrue(hasattr(sd, "get_indexed_deductor"))
        d = sd.get_indexed_deductor()
        self.assertIsNotNone(d)                # 真实库状态文件存在, 应可建索引


class TestOrderFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (RA / "ralph_loop.py").read_text(encoding="utf-8")

    def test_order_fix_marker_present(self):
        self.assertIn("M10 (2026-09-27) 顺序修复", self.src)
        self.assertNotIn("W3 (2026-09-12): forbidden_directions", self.src)

    def test_forbidden_after_guard(self):
        i_guard = self.src.index("P-20260907-006: S5 方向符号守卫")
        i_fb = self.src.index('cand["_forbidden_hit"] = _fr')
        self.assertLess(i_guard, i_fb, "forbidden 打标必须在符号守卫之后")

    def test_forbidden_before_struct(self):
        i_fb = self.src.index('cand["_forbidden_hit"] = _fr')
        i_sd = self.src.index('cand["_struct_dup_hit"] = {')
        self.assertLess(i_fb, i_sd, "struct_dedup 打标在 forbidden 之后 (同区, 顺序无硬要求但保持稳定)")

    def test_struct_block_default_shadow(self):
        # struct 块开关默认 shadow (接线批不翻默认)
        self.assertIn('os.environ.get("FUXI_STRUCT_DEDUP", "shadow")', self.src)
        self.assertNotIn('os.environ.get("FUXI_STRUCT_DEDUP", "enforce")', self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
