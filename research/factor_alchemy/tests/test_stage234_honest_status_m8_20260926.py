# -*- coding: utf-8 -*-
"""M8 (2026-09-26): S2/S3/S4 空通过诚实化 + 行为门影子语义 — 定向夹具。

对应 reports/fuxi_alpha_generation_overhaul_20260926.md M8:
  - S2 缺 max_corr 不得被读作"相关低": passed=True 但 status=data_insufficient;
  - S3 未计算/部分未计算不得写"批次内无高相关冗余 ✅": 五态状态 + 配对分母逐候选可见;
  - S4 替换检查空转与判定分离 (replacement_eval_status, 缺数≠通过);
  - summary 五态覆盖 s2/s3, M8 影子拦截分母可见 (接线锚点);
  - 行为同质性门影子语义: shadow 只记账不剥夺 eligible, on 才剥夺。

判定纪律: 本批不改变任何 pass/fail 结果 (annotate-only), 仅状态/分母/措辞诚实化。
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FA_DIR = ROOT / "research" / "factor_alchemy"
if str(FA_DIR) not in sys.path:
    sys.path.insert(0, str(FA_DIR))

from multi_stage_validator import MultiStageValidator, ValidationResult  # noqa: E402


def _base(n):
    rs = [ValidationResult(factor_name=f"c{i}") for i in range(n)]
    for r in rs:
        r.s1_passed = True
        r.s1_icir = 0.5
        r.s2_passed = True
    return rs


class S2HonestStatusTests(unittest.TestCase):
    def test_missing_max_corr_is_data_insufficient_not_low_corr(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "forge_dsl", "formula": "ts_rank(close, 20)"}]
        results = _base(1)
        v.stage2_correlation(cands, results)
        self.assertTrue(results[0].s2_passed)              # 判定不变 (annotate)
        self.assertEqual(results[0].s2_eval_status, "data_insufficient")
        self.assertIn("status=data_insufficient", results[0].s2_reason)
        self.assertEqual(v._s2_default_zero, 1)

    def test_precomputed_corr_is_valid(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "c", "formula": "close.rank(pct=True)", "max_corr": 0.2}]
        results = _base(1)
        v.stage2_correlation(cands, results)
        self.assertEqual(results[0].s2_eval_status, "valid")
        self.assertEqual(results[0].s2_corr_source, "precomputed")
        self.assertNotIn("空通过", results[0].s2_reason)

    def test_seed_recheck_exempt_is_not_evaluated(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "seed", "formula": "close", "_seed_recheck": True}]
        results = _base(1)
        v.stage2_correlation(cands, results)
        self.assertEqual(results[0].s2_eval_status, "not_evaluated")
        self.assertIn("种子重检豁免", results[0].s2_reason)


class S3HonestDenominatorTests(unittest.TestCase):
    def _prep(self, n):
        v = MultiStageValidator()
        cands = [{"factor_name": f"c{i}",
                  "formula": f"close.rolling({10 + i}).mean()"} for i in range(n)]
        return v, cands, _base(n)

    def test_all_missing_never_claims_clean(self):
        v, cands, results = self._prep(3)          # 全部无 signal
        v.stage3_batch_dedup(cands, results)
        for r in results:
            self.assertTrue(r.s3_passed)                            # 判定不变
            self.assertEqual(r.s3_eval_status, "not_evaluated")
            self.assertNotIn("无高相关冗余", r.s3_reason)            # 禁止误报
            self.assertGreater(r.s3_pairs_missing, 0)
            self.assertEqual(r.s3_pairs_computed, 0)

    def test_computed_pairs_allow_clean_claim(self):
        v, cands, results = self._prep(2)
        rng = np.random.default_rng(0)
        cands[0]["signal"] = rng.normal(size=60)
        cands[1]["signal"] = rng.normal(size=60)
        v.stage3_batch_dedup(cands, results)
        for r in results:
            self.assertEqual(r.s3_eval_status, "valid")
            self.assertEqual(r.s3_pairs_computed, 1)
            self.assertEqual(r.s3_pairs_missing, 0)

    def test_partial_missing_is_data_insufficient(self):
        v, cands, results = self._prep(3)
        rng = np.random.default_rng(1)
        cands[0]["signal"] = rng.normal(size=60)
        cands[1]["signal"] = rng.normal(size=60)
        # c2 无 signal ⇒ (0,2)/(1,2) 缺数
        v.stage3_batch_dedup(cands, results)
        r0, r1, r2 = results
        self.assertEqual((r0.s3_pairs_computed, r0.s3_pairs_missing), (1, 1))
        self.assertEqual(r0.s3_eval_status, "data_insufficient")
        self.assertIn("部分未计算", r0.s3_reason)
        self.assertEqual((r2.s3_pairs_computed, r2.s3_pairs_missing), (0, 2))
        self.assertEqual(r2.s3_eval_status, "not_evaluated")

    def test_single_candidate_no_pairs_no_gap(self):
        v, cands, results = self._prep(1)
        v.stage3_batch_dedup(cands, results)
        self.assertTrue(results[0].s3_passed)
        self.assertEqual(results[0].s3_pairs_missing, 0)   # 非数据缺口
        self.assertIn("无需去重", results[0].s3_reason)

    def test_high_corr_still_rejects_and_status_valid(self):
        v, cands, results = self._prep(2)
        base = np.random.default_rng(2).normal(size=60)
        cands[0]["signal"] = base
        cands[1]["signal"] = base * 1.0 + 1e-9     # 近似同信号 ⇒ corr ≈ 1 > 0.7
        v.stage3_batch_dedup(cands, results)
        self.assertFalse(results[1].s3_passed)      # 裁决逻辑不变 (实测计算)
        self.assertEqual(results[1].s3_eval_status, "valid")
        self.assertIn("高相关", results[1].s3_reason)


class S4ReplacementHonestyTests(unittest.TestCase):
    def test_replace_compare_marked_valid(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "new", "formula": "x", "oos_ic": 0.05, "oos_icir": 0.6}]
        results = _base(1)
        results[0].s2_max_corr_factor = "libold"
        results[0].s3_passed = True
        v.stage4_oos_replacement(cands, results, library_factors={"libold": {"icir": 0.2}})
        self.assertEqual(results[0].s4_replacement_eval_status, "valid")
        self.assertTrue(results[0].s4_passed)

    def test_lib_meta_missing_marked_not_evaluated(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "new", "formula": "x", "oos_ic": 0.05, "oos_icir": 0.6}]
        results = _base(1)
        results[0].s2_max_corr_factor = "libold"
        results[0].s2_eval_status = "valid"
        results[0].s3_passed = True
        v.stage4_oos_replacement(cands, results, library_factors=None)
        self.assertEqual(results[0].s4_replacement_eval_status, "not_evaluated")
        self.assertTrue(results[0].s4_lib_meta_missing)
        self.assertTrue(results[0].s4_passed)       # 判定不变, 仅状态分离

    def test_s2_missing_corr_blocks_replacement_claim(self):
        v = MultiStageValidator()
        cands = [{"factor_name": "new", "formula": "x", "oos_ic": 0.05, "oos_icir": 0.6}]
        results = _base(1)
        results[0].s2_max_corr_factor = ""           # S2 未计算
        results[0].s2_eval_status = "data_insufficient"
        results[0].s3_passed = True
        v.stage4_oos_replacement(cands, results, library_factors={"libold": {"icir": 0.2}})
        self.assertEqual(results[0].s4_replacement_eval_status, "data_insufficient")
        self.assertIn("S2 相关性未计算", results[0].s4_reason)


class ContractAndWiringTests(unittest.TestCase):
    def test_to_dict_exposes_new_status_fields(self):
        r = ValidationResult(factor_name="x")
        r.s2_eval_status = "data_insufficient"
        r.s3_eval_status = "not_evaluated"
        r.s3_pairs_computed = 2
        r.s3_pairs_missing = 1
        r.s4_replacement_eval_status = "valid"
        d = r.to_dict()
        self.assertEqual(d["s2"]["eval_status"], "data_insufficient")
        self.assertEqual(d["s3"]["eval_status"], "not_evaluated")
        self.assertEqual(d["s3"]["pairs_computed"], 2)
        self.assertEqual(d["s3"]["pairs_missing"], 1)
        self.assertEqual(d["s4"]["replacement_eval_status"], "valid")

    def test_summary_wiring_anchors_present(self):
        src = (FA_DIR / "multi_stage_validator.py").read_text(encoding="utf-8")
        self.assertIn('for stage in ("s1", "s2", "s3", "s4", "s5", "s6")', src)
        self.assertIn('"stage2_valid_passed"', src)
        self.assertIn('"stage3_valid_passed"', src)
        self.assertIn('"m8_shadow_reject_s2"', src)
        self.assertIn('"m8_shadow_reject_s3"', src)
        self.assertIn('"m8_shadow_reject_s4_replacement"', src)


class BehaviorGateShadowSemanticsTests(unittest.TestCase):
    """FUXI_BEHAVIOR_GATE_WIRE: shadow=只记账不剥夺 / on=剥夺 (翻默认前置证据)。"""

    def _fake_bh_module(self):
        mod = types.ModuleType("behavior_homogeneity")

        class _Verdict:
            redundancy_label = "NEAR_DUPLICATE"
            nearest_similarity = 0.99
            nearest_factor_id = "libA"
            gate_passed = False

        class _BH:
            def __init__(self, **_kw):
                pass

            def load(self):
                return 1

            def build_library_from_pool(self, pool):
                return len(pool)

            def evaluate_candidate(self, factor_id, signal_series, daily_returns=None):
                return _Verdict()

            def save(self):
                return 1

        mod.BehaviorHomogeneity = _BH
        return mod

    def _run(self, wire_mode):
        import config
        import ralph_loop
        from unittest.mock import patch as _p

        cands = [{"factor_name": "cand_new", "signal": np.ones(60)}]
        results = _base(1)
        results[0].eligible_for_jq = True
        stub_self = types.SimpleNamespace(stats={"total_loops": 0})
        v06 = {"behavior_homogeneity_enabled": True}
        with _p.dict(os.environ, {"FUXI_BEHAVIOR_GATE_WIRE": wire_mode}), \
             _p.dict(sys.modules, {"behavior_homogeneity": self._fake_bh_module()}), \
             _p.object(config, "V06_EXPERIMENTAL", v06):
            out = ralph_loop.RalphLoop._v06_evaluate_guards(
                stub_self, cands, results,
                library_signals={"libA": np.ones(60)})
        return out, cands, results

    def test_shadow_counts_without_stripping_eligible(self):
        out, cands, results = self._run("shadow")
        self.assertEqual(out["behavior_would_reject"], 1)
        self.assertEqual(out["behavior_rejected"], 0)
        self.assertTrue(results[0].eligible_for_jq)           # 未剥夺
        self.assertTrue(cands[0].get("_behavior_would_reject"))

    def test_on_strips_eligible(self):
        out, cands, results = self._run("on")
        self.assertEqual(out["behavior_rejected"], 1)
        self.assertEqual(out["behavior_would_reject"], 0)
        self.assertFalse(results[0].eligible_for_jq)          # 剥夺生效
        self.assertTrue(cands[0].get("_behavior_rejected"))

    def test_dual_fingerprint_thresholds_reject_for_real_module(self):
        """0.95/0.90 双指纹硬拒路径 (真实模块, 非 stub)。"""
        from behavior_homogeneity import BehaviorHomogeneity
        rng = np.random.default_rng(7)
        sig = rng.normal(size=80)
        ret = rng.normal(size=80)
        bh = BehaviorHomogeneity(enabled=True)
        bh.register("libX", sig, residual_returns=ret)
        v = bh.evaluate_candidate("candY", sig, residual_returns=ret)
        self.assertFalse(v.gate_passed)
        self.assertEqual(v.redundancy_label, "NEAR_DUPLICATE")
        self.assertGreaterEqual(v.signal_correlation, 0.95)
        self.assertGreaterEqual(v.residual_correlation, 0.90)


if __name__ == "__main__":
    unittest.main()
