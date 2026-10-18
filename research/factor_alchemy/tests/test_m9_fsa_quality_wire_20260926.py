# -*- coding: utf-8 -*-
"""M9 (2026-09-26): FSA 阈值降级为观察 + 统一质量门接线 — 定向夹具与接线断言。

对应 reports/fuxi_alpha_generation_overhaul_20260926.md M9:
  (1) 库级 FSA share 判据不可达 ⇒ 明确降级为观察: 不修阈值不翻默认,
      把"不可达"从静默 0 变成显式读数 (get_context_status reachable=False);
  (2) 统一质量门 `_gate_fatal` 接消费方: FUXI_QUALITY_GATE_WIRE =
      off | shadow(默认) | enforce — shadow 记账零行为变化, enforce 拒绝
      且证据留档 (不无声消失)。

契约:
  - quality_gate_wire_mode: 默认 shadow; 非法值 fail-fast (ValueError);
  - quality_gate_wire_step (纯函数): 只按 `_gate_result` 判定分级;
    无 `_gate_result` 的候选视为**未评估** — 不计 n_fatal, enforce 不误删;
  - get_context_status: 只读; share 不可达 ⇒ reachable=False + diag 可读;
    dominance ⇒ reachable 保持 True; 空状态 ⇒ reachable=None;
  - 生产主线 (ralph_loop.py) 接 wire 调用与 return 留档键;
    forbidden_gate.py docstring 更新为 M9 复测事实。
"""
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FA_DIR = ROOT / "research" / "factor_alchemy"
if str(FA_DIR) not in sys.path:
    sys.path.insert(0, str(FA_DIR))

from factor_quality_gate import FactorQualityGate, GateResult  # noqa: E402
from subtree_fingerprinter import SubtreeFingerprinter  # noqa: E402
from ralph_loop import quality_gate_wire_mode, quality_gate_wire_step  # noqa: E402


@contextlib.contextmanager
def _no_env():
    """确保 FUXI_QUALITY_GATE_WIRE 未设置 (验证默认值)。"""
    saved = os.environ.pop("FUXI_QUALITY_GATE_WIRE", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["FUXI_QUALITY_GATE_WIRE"] = saved


@contextlib.contextmanager
def _env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── 骨架夹具: 10 个结构互异的表达式 (share 不可达场景) ──
DISTINCT_10 = [
    "close.rolling(20).mean()",
    "ts_delta(close, 5)",
    "ts_rank(volume, 10)",
    "ts_corr(close, volume, 20)",
    "ts_skewness(close, 60)",
    "ts_min(low, 5)",
    "ts_max(high, 5)",
    "ts_sum(amount, 20)",
    "ts_std(close, 30)",
    "ts_decay_linear(close, 10)",
]

FIELDS = ["close", "open", "high", "low", "volume"]
WINDOWS = [5, 10]
# 10 个同骨架因子 (FIELD.rolling(N1).mean()) — share 可达场景
SAME_SKEL_10 = [f"{f}.rolling({w}).mean()" for f in FIELDS for w in WINDOWS]


def _factors(exprs, tag="f"):
    return [{"factor_name": f"{tag}{i:02d}", "formula": e}
            for i, e in enumerate(exprs)]


class QualityGateWireModeTests(unittest.TestCase):
    def test_default_is_shadow(self):
        with _no_env():
            self.assertEqual(quality_gate_wire_mode(), "shadow")

    def test_env_values(self):
        for v in ("off", "shadow", "enforce", " Shadow ", "ENFORCE"):
            with _env(FUXI_QUALITY_GATE_WIRE=v):
                self.assertEqual(quality_gate_wire_mode(), v.strip().lower())

    def test_invalid_value_fail_fast(self):
        for bad in ("1", "on", "true", "enabled"):
            with _env(FUXI_QUALITY_GATE_WIRE=bad):
                with self.assertRaises(ValueError):
                    quality_gate_wire_mode()


class QualityGateWireStepTests(unittest.TestCase):
    @staticmethod
    def _mk(name, passed, fatal=None):
        c = {"factor_name": name, "formula": f"formula_{name}"}
        c["_gate_result"] = GateResult(passed=passed, score=0.5,
                                       fatal_issues=list(fatal or []))
        if fatal:
            c["_gate_fatal"] = list(fatal)
        return c

    def test_shadow_records_but_keeps(self):
        ok = self._mk("ok", True)
        bad = self._mk("bad", False, ["[语法] 括号不匹配"])
        r = quality_gate_wire_step([ok, bad], "shadow", stage="unit")
        self.assertEqual(r["n_checked"], 2)
        self.assertEqual(r["n_fatal"], 1)
        self.assertEqual(r["n_removed"], 0)
        self.assertEqual(len(r["kept"]), 2)  # shadow 不删
        self.assertEqual(len(r["records"]), 1)
        rec = r["records"][0]
        self.assertEqual(rec["name"], "bad")
        self.assertEqual(rec["stage"], "unit")
        self.assertIn("括号不匹配", rec["fatal_issues"][0])
        self.assertFalse(rec["removed"])

    def test_enforce_removes_only_failed(self):
        ok = self._mk("ok", True)
        bad = self._mk("bad", False, ["[风险] 未来函数"])
        r = quality_gate_wire_step([ok, bad], "enforce", stage="unit")
        self.assertEqual(r["n_fatal"], 1)
        self.assertEqual(r["n_removed"], 1)
        self.assertEqual([c["factor_name"] for c in r["kept"]], ["ok"])
        self.assertTrue(r["records"][0]["removed"])

    def test_unevaluated_candidate_is_not_fatal(self):
        """无 _gate_result = 未评估 (≠拒绝): 不计 fatal, enforce 不误删。"""
        ok = self._mk("ok", True)
        no_gate = {"factor_name": "no_gate"}  # 未评估
        r = quality_gate_wire_step([ok, no_gate], "enforce")
        self.assertEqual(r["n_fatal"], 0)
        self.assertEqual(r["n_removed"], 0)
        self.assertEqual(len(r["kept"]), 2)

    def test_pure_function_no_side_effect(self):
        bad = self._mk("bad", False, ["[经济] 未引用任何价量字段"])
        snapshot = dict(bad)
        quality_gate_wire_step([bad], "enforce")
        self.assertEqual(bad, snapshot)  # 入参未被修改

    def test_max_records_caps_evidence(self):
        cands = [self._mk(f"b{i}", False, ["[语法] x"]) for i in range(5)]
        r = quality_gate_wire_step(cands, "shadow", max_records=2)
        self.assertEqual(r["n_fatal"], 5)   # 分母不截断
        self.assertEqual(len(r["records"]), 2)  # 仅证据截断

    def test_real_gate_consistency(self):
        """真实 FactorQualityGate: wire 的 n_fatal 与 verify 结果一致。"""
        gate = FactorQualityGate()
        c_empty = {"factor_name": "g_empty", "formula": ""}
        c_ok = {"factor_name": "g_ok", "formula": "close.rolling(20).mean()"}
        for c in (c_empty, c_ok):
            c["_gate_result"] = gate.verify(c)
        self.assertFalse(c_empty["_gate_result"].passed)  # 空表达式必然 fatal
        r = quality_gate_wire_step([c_empty, c_ok], "shadow")
        expect = sum(1 for c in (c_empty, c_ok) if not c["_gate_result"].passed)
        self.assertEqual(r["n_fatal"], expect)
        self.assertGreaterEqual(expect, 1)


class FsaContextStatusTests(unittest.TestCase):
    def _mk_fsa(self, mode="share"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return SubtreeFingerprinter(
            state_path=Path(tmp.name) / "fsa_state.json", mode=mode)

    def test_share_unreachable_explicit(self):
        fsa = self._mk_fsa("share")
        report = fsa.scan_library(_factors(DISTINCT_10), persist=True)
        self.assertEqual(report.forbidden_count, 0)
        st = fsa.get_context_status()
        self.assertEqual(st["mode"], "share")
        self.assertFalse(st["reachable"])
        self.assertEqual(st["n_forbidden"], 0)
        self.assertIn("不可达", st["diag"])
        self.assertIn("空转", st["diag"])
        self.assertIn("FUXI_FSA_MODE=dominance", st["diag"])
        self.assertGreaterEqual(st["threshold_count"], 2)

    def test_share_reachable_when_concentrated(self):
        fsa = self._mk_fsa("share")
        report = fsa.scan_library(_factors(SAME_SKEL_10), persist=True)
        self.assertGreaterEqual(report.forbidden_count, 1)
        st = fsa.get_context_status()
        self.assertTrue(st["reachable"])
        self.assertEqual(st["diag"], "")
        self.assertEqual(st["n_forbidden"], 1)
        self.assertEqual(st["max_count"], 10)

    def test_dominance_mode_stays_reachable(self):
        fsa = self._mk_fsa("dominance")
        fsa.scan_library(_factors(DISTINCT_10), persist=True)
        st = fsa.get_context_status()
        self.assertEqual(st["mode"], "dominance")
        self.assertTrue(st["reachable"])  # 相对判据不因库规模失效
        self.assertEqual(st["diag"], "")

    def test_empty_state_reachable_none(self):
        fsa = self._mk_fsa("share")
        st = fsa.get_context_status()
        self.assertIsNone(st["reachable"])
        self.assertIn("为空", st["diag"])

    def test_read_only(self):
        fsa = self._mk_fsa("share")
        fsa.scan_library(_factors(SAME_SKEL_10), persist=True)
        sp = fsa.state_path
        before = sp.read_bytes()
        fsa.get_context_status()
        self.assertEqual(sp.read_bytes(), before)  # 诊断不改状态文件


class WiringAnchorTests(unittest.TestCase):
    """生产文件接线锚点 (防未来重构静默断线)。"""

    def _read(self, rel):
        return (ROOT / rel).read_text(encoding="utf-8")

    def test_ralph_loop_main_wiring(self):
        text = self._read("research/factor_alchemy/ralph_loop.py")
        self.assertIn("def quality_gate_wire_mode() -> str:", text)
        self.assertIn("def quality_gate_wire_step(", text)
        self.assertIn("_qgw_mode = quality_gate_wire_mode()", text)
        self.assertIn("stage=\"ralph_phase_generate\"", text)
        self.assertIn("\"quality_gate_wire\": {", text)
        self.assertIn("\"fsa_context\": retrieve_result.get(\"fsa_context_status\", {})", text)
        self.assertIn("retrieve_result[\"fsa_context_status\"] = _fsa_ctx_st", text)
        self.assertIn("_qgw_records = []", text)  # 公共初始化 (全分支 return 安全)

    def test_forbidden_gate_doc_updated(self):
        text = self._read("research/factor_alchemy/forbidden_gate.py")
        self.assertIn("M9 (2026-09-26) 复测", text)
        self.assertIn("get_context_status()", text)
        self.assertNotIn("永不可达", text)  # 过时措辞已移除

    def test_subtree_method_exists_and_wired_consumer(self):
        text = self._read("research/factor_alchemy/subtree_fingerprinter.py")
        self.assertIn("def get_context_status(self) -> Dict:", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
