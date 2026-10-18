# -*- coding: utf-8 -*-
"""M13 增量门修复测试 (2026-09-27): control 快照 / 五态 / 分母 / 异常路径。

覆盖:
  A) IncrementalMarginGate 五态
     · 未启用 ⇒ not_evaluated + deferred (不得记通过)
     · 期数不足 ⇒ data_insufficient + deferred (**回归**: 旧版 gate_passed=True "豁免")
     · 空序列 / 零方差 ⇒ data_insufficient
     · valid pass / valid reject; 分母 (n_control/n_treatment/n_paired/dropped) 可见
  B) control_snapshot
     · 选择规则与遍历序无关 (同一 sha); explicit 规则保序去重截断
     · ensure: created → frozen → drift (漂移**不切换** control, 只记 drift_events)
     · rotate 必须给 reason; revision+1; update_log 落盘
  C) run_incremental_pairs
     · 门关 = 零行为变化 (不触碰候选)
     · 依赖缺失 / control 空 / control 序列短 ⇒ 五态 + 补数队列 (不剥夺 eligible)
     · 预算可见: 超出预算的候选落 not_evaluated/budget_exhausted, 不进黑洞
     · 异常 ⇒ error + 计数 + 队列, **不剥夺** eligible
     · 仅 valid+reject 才剥夺 eligible
     · 每个 eligible 候选恰好落进一个桶 (分桶完备性)
  D) ralph_loop 适配器
     · 端到端: out 带 incremental_control/coverage/deferred, 候选带状态
     · 适配器整体异常 ⇒ not_evaluated + 队列 (不静默放行)
     · 源码守卫: 旧的"遍历序前 8 条"control 与 max_pair=3 不得复活

运行 (cwd = research/factor_alchemy):
    python -X utf8 -m unittest tests.test_m13_incremental_gate_20260927 -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

import config  # noqa: E402
import control_snapshot as cs  # noqa: E402
import incremental_pairing as ip  # noqa: E402
from incremental_margin import IncrementalMarginGate  # noqa: E402

RALPH = _HERE.parents[1] / "ralph_loop.py"
MIN_PERIODS = 8


def _series(seed=11, n=120, drift=0.003, vol=0.02):
    rng = np.random.RandomState(seed)
    return drift + rng.normal(0, vol, n)


def _lib(n=10):
    return [{"formula": f"lib_formula_{i}"} for i in range(n)]


class _Res:
    def __init__(self, eligible=True):
        self.eligible_for_jq = eligible


class _FakeS5:
    """rank_sum_combo_returns: control(短名单) 用 base, treatment(长名单) 用 target。"""

    def __init__(self, base, target=None, *, control_len=None):
        self.base = list(base)
        self.target = list(target) if target is not None else list(base)
        self.control_len = control_len
        self.calls = []

    def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
        self.calls.append(list(formulas))
        if stats is not None:
            stats.update({"n_formulas": len(formulas), "n_evaluated": len(formulas),
                          "n_contributed": len(formulas), "contribution_gap": 0})
        if self.control_len is not None and len(formulas) <= self.control_len:
            return list(self.base)
        return list(self.target)


class TestIncrementalMarginFiveState(unittest.TestCase):
    def setUp(self):
        self.c = _series()
        self.t_better = self.c + 0.004                      # 共享噪声 + 确定增量
        rng = np.random.RandomState(5)
        self.t_noise = self.c + rng.normal(0, 0.02, len(self.c))   # 无真实增量

    def test_disabled_is_not_evaluated(self):
        v = IncrementalMarginGate(enabled=False).evaluate(self.c, self.t_better, factor_id="f")
        self.assertEqual(v.evaluation_status, "not_evaluated")
        self.assertEqual(v.decision, "deferred")
        self.assertFalse(v.gate_passed)
        self.assertIn("未启用", v.reason)

    def test_short_series_not_exempt_pass(self):
        """回归: 旧实现 n<8 时 gate_passed=True + "豁免"; 现在必须 deferred。"""
        v = IncrementalMarginGate(enabled=True).evaluate(
            self.c[:5], self.t_better[:5], factor_id="short")
        self.assertEqual(v.evaluation_status, "data_insufficient")
        self.assertEqual(v.decision, "deferred")
        self.assertFalse(v.gate_passed, "期数不足不得记为通过")
        self.assertIn("不豁免", v.reason)

    def test_empty_series_data_insufficient(self):
        v = IncrementalMarginGate(enabled=True).evaluate([], [], factor_id="empty")
        self.assertEqual(v.evaluation_status, "data_insufficient")
        self.assertEqual(v.decision, "deferred")
        self.assertEqual(v.n_paired, 0)

    def test_zero_variance_data_insufficient(self):
        v = IncrementalMarginGate(enabled=True).evaluate(
            np.zeros(60), np.zeros(60), factor_id="zero")
        self.assertEqual(v.evaluation_status, "data_insufficient")
        self.assertIn("零方差", v.reason)

    def test_valid_pass(self):
        v = IncrementalMarginGate(enabled=True).evaluate(self.c, self.t_better, factor_id="p")
        self.assertEqual(v.evaluation_status, "valid")
        self.assertEqual(v.decision, "pass")
        self.assertTrue(v.gate_passed)
        self.assertGreater(v.incremental_net_ir, 0)

    def test_valid_reject(self):
        v = IncrementalMarginGate(enabled=True).evaluate(self.c, self.t_noise, factor_id="r")
        self.assertEqual(v.evaluation_status, "valid")
        self.assertEqual(v.decision, "reject")
        self.assertFalse(v.gate_passed)

    def test_denominators_visible(self):
        c = list(self.c) + [np.nan] * 4          # control 更长且含 NaN
        t = list(self.t_better)[:100]            # treatment 更短
        v = IncrementalMarginGate(enabled=True).evaluate(c, t, factor_id="den")
        self.assertEqual(v.n_control, len(c))
        self.assertEqual(v.n_treatment, len(t))
        self.assertEqual(v.n_paired, 100)
        self.assertEqual(v.n_dropped_to_min_len, 24)
        d = v.to_dict()
        for k in ("evaluation_status", "decision", "n_control", "n_treatment",
                  "n_paired", "n_dropped_nonfinite", "n_dropped_to_min_len", "min_periods"):
            self.assertIn(k, d)

    def test_min_periods_floor(self):
        """min_periods 低于 4 (即 _ir 内部下限) 会被抬到 4, 免得静默落进 0.0 分支。"""
        g = IncrementalMarginGate(enabled=True, min_incremental_periods=1)
        self.assertEqual(g.min_incremental_periods, 4)


class TestControlSnapshot(unittest.TestCase):
    def test_selection_order_independent(self):
        lib = _lib(12)
        a, ma = cs.select_formulas(lib)
        b, mb = cs.select_formulas(list(reversed(lib)))
        self.assertEqual(a, b, "选择必须与遍历序无关")
        self.assertEqual(ma["n_selected"], 8)
        self.assertEqual(cs.build_snapshot(lib)["snapshot_sha256"],
                         cs.build_snapshot(list(reversed(lib)))["snapshot_sha256"])

    def test_selection_explicit_rule(self):
        picked, meta = cs.select_formulas(
            _lib(4), rule="explicit",
            explicit=["z1", "z2", "z1", "", "z3"], top_k=2)
        self.assertEqual(picked, ["z1", "z2"], "保序 + 去重 + 截断")
        self.assertEqual(meta["rule"], "explicit")

    def test_rule_validation(self):
        with self.assertRaises(ValueError):
            cs.select_formulas(_lib(2), rule="nope")
        with self.assertRaises(ValueError):
            cs.select_formulas(_lib(2), top_k=0)

    def test_registry_created_then_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            reg = cs.ControlSnapshotRegistry(Path(td) / "snap.json")
            r1 = reg.ensure(_lib(10))
            r2 = reg.ensure(_lib(10))
            self.assertEqual(r1["status"], "created")
            self.assertEqual(r2["status"], "frozen")
            self.assertFalse(r2["drift_detected"])
            self.assertEqual(r1["snapshot"]["snapshot_sha256"],
                             r2["snapshot"]["snapshot_sha256"])

    def test_drift_detected_but_control_frozen(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "snap.json"
            reg = cs.ControlSnapshotRegistry(p)
            first = reg.ensure(_lib(10))["snapshot"]
            r = reg.ensure(_lib(40))           # 库大幅变化
            self.assertEqual(r["status"], "drift")
            self.assertTrue(r["drift_detected"])
            self.assertEqual(r["snapshot"]["formulas"], first["formulas"],
                             "漂移时必须保持旧 control, 不得静默切换")
            self.assertEqual(r["snapshot"]["snapshot_sha256"],
                             first["snapshot_sha256"])
            self.assertTrue(r["diff"]["n_added"] > 0 or r["diff"]["n_removed"] > 0)
            # drift 事件落盘 (可审计的漂移计数)
            stored = cs.ControlSnapshotRegistry(p).load()
            self.assertEqual(len(stored["drift_events"]), 1)
            self.assertIn("stored_sha256", stored["drift_events"][0])

    def test_rotate_requires_reason_and_bumps_revision(self):
        with tempfile.TemporaryDirectory() as td:
            reg = cs.ControlSnapshotRegistry(Path(td) / "snap.json")
            reg.ensure(_lib(10))
            with self.assertRaises(ValueError):
                reg.rotate(_lib(10), reason="   ")
            r = reg.rotate(_lib(10), reason="smoke")
            self.assertEqual(r["snapshot"]["revision"], 2)
            self.assertEqual(r["snapshot"]["rotation_reason"], "smoke")
            self.assertEqual(len(r["snapshot"]["update_log"]), 1)
            self.assertIn("frozen_until_explicit_rotate", r["snapshot"]["rotation_rule"])

    def test_empty_library_status(self):
        with tempfile.TemporaryDirectory() as td:
            reg = cs.ControlSnapshotRegistry(Path(td) / "snap.json")
            r = reg.ensure([])
            self.assertEqual(r["status"], "empty_library")
            self.assertEqual(r["snapshot"]["formulas"], [])

    def test_snapshot_path_resolution(self):
        p = ip.resolve_snapshot_path({}, None)
        self.assertTrue(str(p).endswith("data\\ralph_control_snapshot.json")
                        or str(p).endswith("data/ralph_control_snapshot.json"))
        override = ip.resolve_snapshot_path({}, "X:/tmp/abc.json")
        self.assertEqual(str(override).replace("\\", "/"), "X:/tmp/abc.json")


class TestRunIncrementalPairs(unittest.TestCase):
    def _tmp_snap(self, td):
        return Path(td) / "snap.json"

    def test_disabled_zero_touch(self):
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(cands, res, lib_factors=_lib(),
                                         s5_filter=_FakeS5(_series()),
                                         cfg={}, snapshot_path=self._tmp_snap(td))
            self.assertEqual(r["status"], "not_evaluated")
            self.assertEqual(r["checked"], 0)
            self.assertFalse(Path(td, "snap.json").exists(), "门关不得落盘快照")
        self.assertNotIn("_incremental_eval_status", cands[0], "门关必须零行为变化")

    def test_dependency_missing_not_evaluated_queue(self):
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(cands, res, lib_factors=_lib(),
                                         s5_filter=None,
                                         cfg={"incremental_margin_enabled": True},
                                         snapshot_path=self._tmp_snap(td))
        self.assertEqual(r["status"], "not_evaluated")
        self.assertEqual(cands[0]["_incremental_eval_status"], "not_evaluated")
        self.assertEqual(r["deferred"]["n_deferred"], 1)
        self.assertTrue(res[0].eligible_for_jq, "依赖缺失不得剥夺 eligible")

    def test_budget_visible(self):
        cands = [{"factor_name": f"c{i}", "formula": f"x{i}"} for i in range(5)]
        res = [_Res() for _ in cands]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(
                cands, res, lib_factors=_lib(),
                s5_filter=_FakeS5(_series(), target=_series()),
                cfg={"incremental_margin_enabled": True, "incremental_max_pair": 2},
                snapshot_path=self._tmp_snap(td))
        cov = r["coverage"]
        self.assertEqual(cov["budget"], 2)
        self.assertEqual(cov["n_pairs_attempted"], 2)
        self.assertEqual(cov["deferred_by_reason"].get("budget_exhausted"), 3)
        self.assertEqual(cands[2]["_incremental_eval_status"], "not_evaluated",
                         "被预算挡下的候选必须有显式状态 (不得从输出消失)")
        self.assertEqual(sum(cov["status_distribution"].values()), cov["n_eligible"])

    def test_exception_marks_error_not_deprived(self):
        class _Boom(_FakeS5):
            def rank_sum_combo_returns(self, formulas, *, top_n=None, names=None, stats=None):
                if len(formulas) > 8:
                    raise RuntimeError("boom")
                return super().rank_sum_combo_returns(formulas, top_n=top_n, names=names, stats=stats)

        cands = [{"factor_name": f"c{i}", "formula": f"x{i}"} for i in range(3)]
        res = [_Res() for _ in cands]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(cands, res, lib_factors=_lib(),
                                         s5_filter=_Boom(_series()),
                                         cfg={"incremental_margin_enabled": True},
                                         snapshot_path=self._tmp_snap(td))
        self.assertEqual(r["coverage"]["n_error"], 3)
        self.assertEqual(r["exceptions"], 3)
        self.assertEqual(r["rejected"], 0)
        self.assertTrue(all(c["_incremental_eval_status"] == "error" for c in cands))
        self.assertTrue(all(x.eligible_for_jq for x in res), "异常不得剥夺 eligible")
        self.assertEqual(r["deferred"]["n_deferred"], 3)

    def test_short_control_series_data_insufficient(self):
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(cands, res, lib_factors=_lib(),
                                         s5_filter=_FakeS5(_series(n=5)),
                                         cfg={"incremental_margin_enabled": True},
                                         snapshot_path=self._tmp_snap(td))
        self.assertEqual(r["status"], "data_insufficient")
        self.assertEqual(cands[0]["_incremental_eval_status"], "data_insufficient")
        self.assertTrue(res[0].eligible_for_jq, "缺数不得惩罚")

    def test_valid_reject_revokes_eligible(self):
        base = _series(seed=3)
        worse = list(np.asarray(base) - 0.004)      # treatment 严格劣于 control
        cands = [{"factor_name": "bad", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(cands, res, lib_factors=_lib(),
                                         s5_filter=_FakeS5(base, target=worse),
                                         cfg={"incremental_margin_enabled": True},
                                         snapshot_path=self._tmp_snap(td))
        self.assertEqual(r["checked"], 1)
        self.assertEqual(r["rejected"], 1)
        self.assertFalse(res[0].eligible_for_jq, "valid+reject 才剥夺 eligible")
        self.assertTrue(cands[0]["_incremental_rejected"])
        self.assertEqual(cands[0]["_incremental_eval_status"], "valid")
        self.assertEqual(cands[0]["_incremental_verdict"]["decision"], "reject")

    def test_control_recorded_and_frozen_across_runs(self):
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        cfg = {"incremental_margin_enabled": True}
        with tempfile.TemporaryDirectory() as td:
            snap = self._tmp_snap(td)
            r1 = ip.run_incremental_pairs(cands, res, lib_factors=_lib(10),
                                          s5_filter=_FakeS5(_series()),
                                          cfg=cfg, snapshot_path=snap)
            r2 = ip.run_incremental_pairs(cands, [_Res()], lib_factors=_lib(30),
                                          s5_filter=_FakeS5(_series()),
                                          cfg=cfg, snapshot_path=snap)
        c1, c2 = r1["control"], r2["control"]
        self.assertEqual(c1["snapshot_sha256"], c2["snapshot_sha256"],
                         "库变了 control 也必须保持冻结 (漂移计数应为 0 切换)")
        self.assertTrue(c1["frozen"] and c2["frozen"])
        self.assertTrue(c1["rule"] and isinstance(c1["revision"], int))
        self.assertEqual(c1["composition"].get("n_evaluated"), 8)
        self.assertEqual(c2["drift_detected"], True, "漂移必须被记录 (只是不切换)")
        self.assertIsNotNone(c2["drift_diff"])

    def test_bucket_partition_complete(self):
        cands = [{"factor_name": f"c{i}", "formula": ("" if i == 1 else f"x{i}")}
                 for i in range(4)]
        res = [_Res() for _ in cands]
        with tempfile.TemporaryDirectory() as td:
            r = ip.run_incremental_pairs(
                cands, res, lib_factors=_lib(),
                s5_filter=_FakeS5(_series(n=5)),   # control 短 ⇒ 全落 data_insufficient
                cfg={"incremental_margin_enabled": True, "incremental_max_pair": 2},
                snapshot_path=self._tmp_snap(td))
        cov = r["coverage"]
        self.assertEqual(sum(cov["status_distribution"].values()), cov["n_eligible"])
        self.assertEqual(cov["n_eligible"], 4)

    # ── 回退通道（反事实） ────────────────────────────────────────────
    def test_control_mode_getter(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("FUXI_INCREMENTAL_CONTROL", None)
            self.assertEqual(ip.control_mode(), "snapshot", "默认必须 snapshot")
        with mock.patch.dict("os.environ", {"FUXI_INCREMENTAL_CONTROL": "traversal_legacy"}):
            self.assertEqual(ip.control_mode(), "traversal_legacy")
        with mock.patch.dict("os.environ", {"FUXI_INCREMENTAL_CONTROL": "bogus"}):
            self.assertEqual(ip.control_mode(), "snapshot", "非法值 fail-safe 回 snapshot")

    def test_traversal_legacy_counterfactual(self):
        lib = _lib(12)
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            snap = self._tmp_snap(td)
            with mock.patch.dict("os.environ",
                                 {"FUXI_INCREMENTAL_CONTROL": "traversal_legacy"}):
                r = ip.run_incremental_pairs(
                    cands, res, lib_factors=lib, s5_filter=_FakeS5(_series()),
                    cfg={"incremental_margin_enabled": True}, snapshot_path=snap)
            self.assertFalse(snap.exists(), "反事实口径不得落盘快照")
        self.assertEqual(r["control"]["mode"], "traversal_legacy")
        self.assertIsNone(r["control"]["snapshot_sha256"])
        # control = 遍历序前 8 条（= M13 前语义）
        self.assertEqual(r["control"]["n_formulas"], 8)
        self.assertTrue(any("反事实口径" in n for n in r["notes"]))


class TestRalphLoopAdapter(unittest.TestCase):
    def _call(self, cands, res, *, s5, cfg_extra=None, lib=None):
        extra = {
            "direction_campaign_enabled": False,
            "budget_ledger_monitor_enabled": False,
            "behavior_homogeneity_enabled": False,
            "holdout_enabled": False,
            "incremental_margin_enabled": True,
        }
        extra.update(cfg_extra or {})
        with mock.patch.dict(config.V06_EXPERIMENTAL, extra, clear=False):
            from ralph_loop import RalphLoop
            return RalphLoop._v06_evaluate_guards(
                None, cands, res, lib_factors=lib or _lib(10), s5_filter=s5)

    def test_adapter_records_control_and_coverage(self):
        cands = [{"factor_name": "a", "formula": "x"}]
        res = [_Res()]
        with tempfile.TemporaryDirectory() as td:
            out = self._call(cands, res, s5=_FakeS5(_series()),
                             cfg_extra={"incremental_control_snapshot_path":
                                        str(Path(td) / "s.json")})
        self.assertIn("incremental_control", out)
        self.assertTrue(out["incremental_control"]["snapshot_sha256"])
        self.assertEqual(out["incremental_coverage"]["n_eligible"], 1)
        self.assertIn("incremental_deferred", out)
        self.assertEqual(out["incremental_checked"], 1)
        self.assertEqual(cands[0]["_incremental_eval_status"], "valid")

    def test_adapter_exception_not_silent(self):
        cands = [{"factor_name": "a", "formula": "x"}, {"factor_name": "b", "formula": "y"}]
        res = [_Res(), _Res()]
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(ip, "run_incremental_pairs",
                                   side_effect=RuntimeError("adapter boom")):
                out = self._call(cands, res, s5=_FakeS5(_series()),
                                 cfg_extra={"incremental_control_snapshot_path":
                                            str(Path(td) / "s.json")})
        self.assertEqual(out["incremental_status"], "error")
        self.assertEqual(out["incremental_exceptions"], 1)
        self.assertEqual(out["incremental_deferred"]["n_deferred"], 2)
        self.assertTrue(all(c["_incremental_eval_status"] == "not_evaluated" for c in cands))
        self.assertTrue(all(x.eligible_for_jq for x in res), "异常不得剥夺 eligible")
        self.assertTrue(any("not_evaluated" in n for n in out["notes"]))

    def test_source_guard_no_traversal_control(self):
        """源码守卫: 旧的遍历序 control 与 max_pair=3 不得复活。"""
        with RALPH.open("rb") as fh:
            text = fh.read().decode("utf-8").replace("\r\n", "\n")
        self.assertNotIn('[f.get("formula") for f in lib_factors if f.get("formula")][:8]',
                         text, "旧 control 选择 (遍历序前 8 条) 不得复活")
        self.assertNotIn("max_pair = 3", text, "旧硬编码预算不得复活")
        # 注意: 行为簇门 (M10 范围) 仍有自己的"降级放行"; 这里只守增量门那一条
        self.assertNotIn("[v0.6-增量] 门禁异常(降级放行)", text,
                         "增量门不得再有静默降级放行")
        self.assertIn("run_incremental_pairs", text)


class TestManifestWiring(unittest.TestCase):
    """快照哈希/分母必须真的进 run manifest (不是只进内存 summary)。"""

    def test_record_evaluation_persists_incremental_block(self):
        import json as _json

        from run_contract import RunContract

        payload = {
            "incremental_status": "valid",
            "incremental_control": {"snapshot_sha256": "deadbeef" * 8, "revision": 1,
                                    "rule": "sha256_prefix", "drift_detected": False},
            "incremental_coverage": {"n_eligible": 3, "budget": 12,
                                     "status_distribution": {"valid": 3}},
            "incremental_deferred": {"n_deferred": 0, "items": []},
            "incremental_exceptions": 0,
        }
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with RunContract(parameters={"m13": True}, run_id="m13_manifest_probe",
                             runs_dir=tmp / "runs", lock_path=tmp / "lock.json") as rc:
                rc.record_evaluation(payload, [])
            manifest = _json.loads(
                (tmp / "runs" / "m13_manifest_probe" / "manifest.json").read_text("utf-8"))
        got = manifest["evaluation"]["summary"]
        self.assertEqual(got["incremental_control"]["snapshot_sha256"], "deadbeef" * 8)
        self.assertEqual(got["incremental_coverage"]["status_distribution"], {"valid": 3})
        self.assertIn("incremental_exceptions", got)

    def test_pipeline_passes_evaluate_summary_to_contract(self):
        """run_v4_pipeline 必须把 evaluate 阶段 summary 交给 record_evaluation。"""
        with (RALPH.parent / "run_v4_pipeline.py").open("rb") as fh:
            text = fh.read().decode("utf-8").replace("\r\n", "\n")
        self.assertIn('_evaluation.get("summary", {})', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
