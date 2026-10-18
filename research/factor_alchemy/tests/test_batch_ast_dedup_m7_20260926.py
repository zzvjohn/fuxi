# -*- coding: utf-8 -*-
"""M7 (2026-09-26): 批内 AST 去重公共阶段 — 定向夹具与接线断言。

对应 reports/fuxi_alpha_generation_overhaul_20260926.md M7:
  把 `_batch_dedup_ast` 提为公共阶段, 覆盖 Ralph LLM 分支与 reviewer 二轮。

契约:
  - enforce (默认): 批内结构重复保留先出现者; 命中项全部留档 (去重不删证据);
  - shadow: 只留档不移除 (供误杀率 A/B); off: 原样返回;
  - 非法开关值 fail-fast (ValueError);
  - 生产主线 (ralph_loop.py) 两处 parse 后均接 batch_dedup_ast,
    结果字典带 "batch_ast_dedup" 留档键。

夹具骨架 (已用生产指纹器实测):
  dup_a/dup_b → 'FIELD.rolling(N1).mean()/FIELD.rolling(N1).std()' (同骨架)
  uniq_c      → 'delta(FIELD,N1).rank(pct=True)'
  uniq_d      → 'net_mf_vol.rolling(N1).sum()'
"""
import contextlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FA_DIR = ROOT / "research" / "factor_alchemy"
if str(FA_DIR) not in sys.path:
    sys.path.insert(0, str(FA_DIR))

import llm_generator  # noqa: E402
from llm_generator import LLMGenerator, dedup_ast_candidates  # noqa: E402


FP_DUP = "FIELD.rolling(N1).mean()/FIELD.rolling(N1).std()"

DUP_A = {"factor_name": "dup_a",
         "formula": "close.rolling(20).mean() / close.rolling(20).std()"}
DUP_B = {"factor_name": "dup_b",
         "formula": "volume.rolling(40).mean() / volume.rolling(40).std()"}
UNIQ_C = {"factor_name": "uniq_c", "formula": "ts_delta(close, 5).rank(pct=True)"}
UNIQ_D = {"factor_name": "uniq_d", "formula": "net_mf_vol.rolling(10).sum()"}


@contextlib.contextmanager
def _no_env():
    """确保 FUXI_BATCH_AST_DEDUP 未设置 (验证默认值)。"""
    saved = os.environ.pop("FUXI_BATCH_AST_DEDUP", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["FUXI_BATCH_AST_DEDUP"] = saved


def _batch(*items):
    return [dict(c) for c in items]


class BatchAstDedupStageTests(unittest.TestCase):
    def test_enforce_removes_batch_duplicate_and_keeps_evidence(self):
        ev = []
        out = dedup_ast_candidates(_batch(DUP_A, DUP_B, UNIQ_C),
                                   evidence_out=ev, mode="enforce")
        self.assertEqual([c["factor_name"] for c in out], ["dup_a", "uniq_c"])
        self.assertEqual(len(ev), 1)
        rec = ev[0]
        self.assertEqual(rec["factor_name"], "dup_b")
        self.assertEqual(rec["dup_of"], "dup_a")
        self.assertEqual(rec["fingerprint"], FP_DUP)
        self.assertTrue(rec["removed"])
        self.assertFalse(rec["round2"])
        self.assertEqual(rec["stage"], "")

    def test_distinct_skeletons_untouched(self):
        ev = []
        out = dedup_ast_candidates(_batch(UNIQ_C, UNIQ_D), evidence_out=ev, mode="enforce")
        self.assertEqual(len(out), 2)
        self.assertEqual(ev, [])

    def test_shadow_records_without_removing(self):
        ev = []
        out = dedup_ast_candidates(_batch(DUP_A, DUP_B), evidence_out=ev, mode="shadow")
        self.assertEqual([c["factor_name"] for c in out], ["dup_a", "dup_b"])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["dup_of"], "dup_a")
        self.assertFalse(ev[0]["removed"])

    def test_off_is_noop(self):
        ev = []
        out = dedup_ast_candidates(_batch(DUP_A, DUP_B), evidence_out=ev, mode="off")
        self.assertEqual(len(out), 2)
        self.assertEqual(ev, [])

    def test_default_mode_is_enforce_when_env_unset(self):
        with _no_env():
            self.assertEqual(llm_generator.batch_dedup_mode(), "enforce")
            out = dedup_ast_candidates(_batch(DUP_A, DUP_B), mode=None)
        self.assertEqual(len(out), 1)

    def test_env_switch_governs_public_stage(self):
        with patch.dict(os.environ, {"FUXI_BATCH_AST_DEDUP": "shadow"}):
            ev = []
            out = dedup_ast_candidates(_batch(DUP_A, DUP_B), evidence_out=ev)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0]["removed"])

    def test_invalid_mode_fails_fast(self):
        with patch.dict(os.environ, {"FUXI_BATCH_AST_DEDUP": "yes"}):
            with self.assertRaises(ValueError):
                llm_generator.batch_dedup_mode()

    def test_round2_marker_and_stage_recorded(self):
        r2a = dict(DUP_B)
        r2a["_llm_reviewer_round2"] = True
        r2b = dict(r2a)
        r2b["factor_name"] = "dup_b2"
        ev = []
        out = dedup_ast_candidates([r2a, r2b], evidence_out=ev,
                                   mode="enforce", stage="llm_round2")
        self.assertEqual(len(out), 1)
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0]["round2"])
        self.assertEqual(ev[0]["stage"], "llm_round2")

    def test_formula_less_candidates_pass_through(self):
        ev = []
        out = dedup_ast_candidates([{"factor_name": "x"},
                                    {"factor_name": "y", "formula": ""}],
                                   evidence_out=ev, mode="enforce")
        self.assertEqual([c["factor_name"] for c in out], ["x", "y"])
        self.assertEqual(ev, [])

    def test_single_candidate_short_circuits(self):
        ev = []
        out = dedup_ast_candidates(_batch(DUP_A), evidence_out=ev, mode="enforce")
        self.assertEqual(len(out), 1)
        self.assertEqual(ev, [])

    def test_instance_entry_and_history_alias_delegate(self):
        gen = LLMGenerator()
        ev = []
        out = gen.batch_dedup_ast(_batch(DUP_A, DUP_B), evidence=ev,
                                  mode="enforce", stage="llm_round1")
        self.assertEqual(len(out), 1)
        self.assertEqual(ev[0]["stage"], "llm_round1")
        with patch.dict(os.environ, {"FUXI_BATCH_AST_DEDUP": "enforce"}):
            out2 = gen._batch_dedup_ast(_batch(DUP_A, DUP_B))
        self.assertEqual(len(out2), 1)

    def test_mainline_wiring_anchors_present(self):
        """接线断言: 主线两处 parse 后均接去重, 结果字典带留档键。"""
        src = (FA_DIR / "ralph_loop.py").read_text(encoding="utf-8")
        self.assertIn('stage="llm_round1"', src)
        self.assertIn('stage="llm_round2"', src)
        self.assertIn('"batch_ast_dedup"', src)
        self.assertIn("from llm_generator import batch_dedup_mode", src)
        # 二轮不做跨轮合并去重 (reviewer 本质是同骨架定向改进)
        self.assertNotIn('stage="llm_merge"', src)


if __name__ == "__main__":
    unittest.main()
