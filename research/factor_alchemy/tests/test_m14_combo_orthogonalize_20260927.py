# -*- coding: utf-8 -*-
"""M14 (2026-09-27) 组合层正交化诊断 —— 单元测试。

覆盖四组:
  A. 核心算法 (合成世界, 可控 base_r2 / 三口径 / 零方差 / 期数)
  B. 零裁决防护 (不改候选 / 只出标量 / off 空读数 / forbidden_use / 不 import neutralize)
  C. control 同源 (M13 快照读写 / 缺失不兜底 / 注入路径)
  D. 轮级聚合 (读法分布 / 单候选异常隔离 / 候选优先级)

合成世界设计 (fwd=1, close 递推 ⇒ fwd_return ≈ beta * X1 + noise):
  · cand_copy   = X1 的复制       ⇒ base_r2 ≈ 1  ⇒ replicated_exposure
  · cand_indep  = 独立噪声        ⇒ base_r2 ≈ 0  ⇒ increment_candidate
  · cand_partial= 0.6*X1+0.4噪声  ⇒ base_r2 ≈ 0.36, decay ≈ 1 ⇒ weak_increment
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FA = HERE.parent
sys.path.insert(0, str(FA))

import combo_orthogonalize as co  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 合成世界
# ═══════════════════════════════════════════════════════════

def _panel_from_series(s: pd.Series) -> pd.DataFrame:
    return (s.rename("factor_value").reset_index()
            [["ts_code", "trade_date", "factor_value"]])


def build_world(*, n_dates: int = 40, n_codes: int = 60, seed: int = 3,
                beta_fwd: float = 0.01):
    """返回 (panels, price_df, meta)。

    panels: {formula: DataFrame[ts_code, trade_date, factor_value]}
    price_df: DataFrame[ts_code, trade_date, close] (close 递推使 fwd=1 的
              fwd_return ≈ beta_fwd * X1 + 噪声)
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2026-01-05", periods=n_dates, freq="B")
    codes = [f"S{i:03d}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([codes, dates], names=["ts_code", "trade_date"])

    X1 = pd.Series(rng.normal(size=len(idx)), index=idx, name="X1")
    X2 = pd.Series(rng.normal(size=len(idx)), index=idx, name="X2")
    NOISE = pd.Series(rng.normal(size=len(idx)), index=idx, name="NOISE")

    # close 递推: r_t = close_{t+1}/close_t - 1 = beta * X1_t + eps
    close_vals = {}
    for c in codes:
        px = 10.0
        for d in dates:
            close_vals[(c, d)] = px
            r = beta_fwd * float(X1.loc[(c, d)]) + rng.normal(0, 0.002)
            px = px * (1.0 + r)
    close_s = pd.Series(close_vals).reindex(idx).astype("float64")
    price = close_s.rename("close").reset_index()

    panels = {
        "F_X1": _panel_from_series(X1),
        "F_X2": _panel_from_series(X2),
        "F_ZERO": _panel_from_series(pd.Series(1.0, index=idx)),        # 零方差
        "F_COPY": _panel_from_series(X1.copy()),                        # 复制 X1
        "F_INDEP": _panel_from_series(NOISE.copy()),                    # 独立
        "F_PARTIAL": _panel_from_series(0.6 * X1 + 0.4 * NOISE),        # 部分
        "F_GROUPED": _panel_from_series(                                # 组内恒定
            pd.Series(np.where([c < "S030" for c in
                                idx.get_level_values("ts_code")], 1.0, 2.0), index=idx)),
    }
    return panels, price, {"n_dates": n_dates, "n_codes": n_codes}


class _StubComp:
    def __init__(self, panels, price):
        self._panels = panels
        self._price_df = price

    def _eval_formula(self, formula, factor_name=None):
        df = self._panels.get(str(formula))
        return None if df is None else df.copy()


class _Res:
    def __init__(self, eligible=False, s1=False):
        self.eligible_for_jq = eligible
        self.s1_passed = s1


class _Base(unittest.TestCase):
    """每个测试: 清缓存 + 装 stub + fwd=1 + 关行业读数 (独立于 data/ 文件)。"""

    def setUp(self):
        self._saved_env = dict(os.environ)
        os.environ["FUXI_COMBO_ORTHO_FWD"] = "1"
        co._CTX.clear()
        self.panels, self.price, self.meta = build_world()
        co.attach_computer(_StubComp(self.panels, self.price))
        co._CTX["ind_map"] = {}            # 关行业读数 (保持测试独立)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        co._CTX.clear()

    def _check(self, formula, name, ctl=("F_X1", "F_X2"), **kw):
        return co.factor_check(formula, name, list(ctl), ctl_sha="test", **kw)


# ═══════════════════════════════════════════════════════════
# A. 核心算法
# ═══════════════════════════════════════════════════════════

class TestCoreAlgorithm(_Base):

    def test_replica_base_r2_near_one(self):
        row = self._check("F_COPY", "copy")
        self.assertEqual(row["status"], "valid")
        self.assertGreater(row["base_r2_median"], 0.95)
        self.assertEqual(row["read"], "replicated_exposure")

    def test_independent_base_r2_low(self):
        row = self._check("F_INDEP", "indep")
        self.assertEqual(row["status"], "valid")
        self.assertLess(row["base_r2_median"], 0.10)
        self.assertEqual(row["read"], "increment_candidate")

    def test_partial_candidate_weak_increment(self):
        row = self._check("F_PARTIAL", "partial")
        self.assertEqual(row["status"], "valid")
        # 理论值: corr(0.6X1+0.4N, X1)=0.6/sqrt(0.52)≈0.832 ⇒ R²≈0.69 (秩域略低 ~0.65)
        self.assertGreater(row["base_r2_median"], 0.20)
        self.assertLess(row["base_r2_median"], 0.75)
        # 表观 IC 主要来自 X1 (可被基组合解释) ⇒ 残差后坍缩
        self.assertGreater(abs(row["rank_raw"]), co.MIN_IC_ABS)
        self.assertGreater(row["ic_decay_ratio"], co.DECAY_RISK)
        self.assertEqual(row["read"], "weak_increment")

    def test_three_calibers_present_and_numeric(self):
        row = self._check("F_PARTIAL", "partial")
        for k in ("rank_raw", "rank_resid", "pearson_raw", "ic_decay_ratio"):
            self.assertIn(k, row)
            self.assertIsInstance(row[k], float)
        # 残差口径 IC 应远小于表观 (因为增量信号可被基组合解释)
        self.assertLess(abs(row["rank_resid"]), abs(row["rank_raw"]))

    def test_ic_decay_high_for_replica(self):
        row = self._check("F_COPY", "copy")
        self.assertGreater(row["ic_decay_ratio"], co.DECAY_RISK)

    def test_zero_variance_ctl_dropped_and_counted(self):
        row = self._check("F_COPY", "copy", ctl=("F_X1", "F_ZERO"))
        self.assertEqual(row["status"], "valid")
        self.assertGreater(row["n_zero_var_col_dropped"], 0)
        # F_X1 仍被使用 (n_ctl_used_median == 1)
        self.assertEqual(row["n_ctl_used_median"], 1)
        self.assertGreater(row["base_r2_median"], 0.95)

    def test_group_constant_candidate_no_crash(self):
        """组内恒定候选 (每期仅两个取值): 三口径仍能并列报出, 不崩。"""
        row = self._check("F_GROUPED", "grouped")
        self.assertEqual(row["status"], "valid")
        self.assertIn("rank_raw", row)
        self.assertIn("pearson_raw", row)
        self.assertIsInstance(row["pearson_raw"], float)

    def test_min_stocks_skip_causes_data_insufficient(self):
        os.environ["FUXI_COMBO_ORTHO_MIN_STOCKS"] = "100000"
        row = self._check("F_COPY", "copy")
        self.assertEqual(row["status"], "data_insufficient")
        self.assertIn("periods", str(row.get("reason")))

    def test_periods_insufficient_flag(self):
        # 控制列几乎全常数 ⇒ 每期无可回归列 ⇒ 期数不足
        panels, price, _ = build_world(n_dates=6, n_codes=60)
        co.attach_computer(_StubComp(panels, price))
        row = self._check("F_COPY", "copy")
        self.assertEqual(row["status"], "data_insufficient")

    def test_fwd_nan_tail_excluded_from_ic(self):
        """最后 1 天 fwd_return 为 NaN (fwd=1) ⇒ base_r2 期数 > IC 期数, 不崩。"""
        row = self._check("F_COPY", "copy")
        self.assertEqual(row["n_periods"], self.meta["n_dates"])
        self.assertTrue(np.isfinite(row["rank_raw"]))

    def test_no_overlap_returns_data_insufficient(self):
        panels = dict(self.panels)
        co.attach_computer(_StubComp(panels, self.price))
        row = co.factor_check("F_NOT_EXIST", "missing", ["F_X1"], ctl_sha="t")
        self.assertIsNone(row)          # 面板拿不到 ⇒ None (由 evaluate 转 data_insufficient)


# ═══════════════════════════════════════════════════════════
# B. 零裁决防护
# ═══════════════════════════════════════════════════════════

class TestNoRulingGuards(_Base):

    def _evaluate(self, cands, results, ctl=("F_X1", "F_X2"), **kw):
        with tempfile.TemporaryDirectory() as td:
            return co.evaluate(cands, results, ctl_formulas=list(ctl),
                               out_path=Path(td) / "o.json", verbose=False, **kw)

    def test_evaluate_does_not_modify_candidates(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        results = [_Res(eligible=True)]
        snapshot = json.dumps(cands, ensure_ascii=False)
        p = self._evaluate(cands, results)
        self.assertEqual(json.dumps(cands, ensure_ascii=False), snapshot)
        self.assertTrue(results[0].eligible_for_jq, "零裁决: 不得剥夺 eligible")
        self.assertEqual(p["status"], "valid")

    def test_payload_scalars_only(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"},
                 {"factor_name": "indep", "formula": "F_INDEP"}]
        p = self._evaluate(cands, [_Res(), _Res()])
        allowed = (int, float, str, bool, type(None))

        def _walk(v, path=""):
            if isinstance(v, dict):
                for k, vv in v.items():
                    _walk(vv, f"{path}.{k}")
            elif isinstance(v, list):
                for i, vv in enumerate(v):
                    _walk(vv, f"{path}[{i}]")
            else:
                self.assertIsInstance(v, allowed, f"非标量输出: {path} = {type(v)}")

        _walk(p)

    def test_off_mode_not_evaluated(self):
        os.environ["FUXI_COMBO_ORTHO"] = "off"
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        with tempfile.TemporaryDirectory() as td:
            p = co.evaluate(cands, [_Res()], ctl_formulas=["F_X1"],
                            out_path=Path(td) / "o.json", verbose=False)
        self.assertEqual(p["mode"], "off")
        self.assertEqual(p["status"], "not_evaluated")
        self.assertEqual(p["n_checked"], 0)

    def test_forbidden_use_field_present(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        p = self._evaluate(cands, [_Res()])
        fu = p.get("forbidden_use", "")
        self.assertIn("中性化", fu)
        self.assertIn("V5", fu)
        self.assertIn("裁决输入", fu)

    def test_source_has_no_neutralize_import(self):
        src = (FA / "combo_orthogonalize.py").read_text(encoding="utf-8")
        self.assertNotIn("import neutralize", src)
        self.assertNotIn("from neutralize", src)

    def test_output_json_has_no_residual_payload(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.json"
            co.evaluate(cands, [_Res()], ctl_formulas=["F_X1"],
                        out_path=out, verbose=False)
            raw = out.read_text(encoding="utf-8")
        d = json.loads(raw)
        self.assertNotIn("residuals", d)
        for r in d.get("rows", []):
            self.assertNotIn("resid_panel", r)
        # 文件应为紧凑标量读数 (非面板): 行数远小于股票×期数
        self.assertLess(raw.count("\n"), 60)


# ═══════════════════════════════════════════════════════════
# C. control 同源 (M13 快照)
# ═══════════════════════════════════════════════════════════

class TestControlSource(_Base):

    def test_load_control_missing(self):
        with tempfile.TemporaryDirectory() as td:
            formulas, meta = co.load_control(Path(td) / "nope.json")
        self.assertEqual(formulas, [])
        self.assertEqual(meta["status"], "snapshot_missing")

    def test_load_control_corrupt(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            formulas, meta = co.load_control(p)
        self.assertEqual(formulas, [])
        self.assertTrue(str(meta["status"]).startswith("snapshot_unreadable"))

    def test_load_control_roundtrip_matches_m13_sha(self):
        """与 M13 control_snapshot 同源: 写出的 sha256 必须被 load_control 读到且一致。"""
        import control_snapshot as cs
        items = [{"name": f"t{i}", "formula": f"F{i}_{i * 7 + 3}"} for i in range(20)]
        with tempfile.TemporaryDirectory() as td:
            snap_path = Path(td) / "ctl.json"
            reg = cs.ControlSnapshotRegistry(snap_path)
            built = reg.ensure(items)
            formulas, meta = co.load_control(snap_path)
            expect_sha = cs.snapshot_sha256("sha256_prefix", 8, formulas)
            self.assertEqual(meta["status"], "ok")
            self.assertEqual(meta["snapshot_sha256"], expect_sha)
            self.assertEqual(len(formulas), 8)
            self.assertEqual(meta["snapshot_sha256"],
                             built["snapshot"]["snapshot_sha256"])

    def test_control_unavailable_data_insufficient(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        with tempfile.TemporaryDirectory() as td:
            p = co.evaluate(cands, [_Res()], snapshot_path=Path(td) / "nope.json",
                            out_path=Path(td) / "o.json", verbose=False)
        self.assertEqual(p["status"], "data_insufficient")
        self.assertIn("control_unavailable", str(p.get("reason")))
        self.assertEqual(p["n_checked"], 0)

    def test_injected_ctl_used(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        with tempfile.TemporaryDirectory() as td:
            p = co.evaluate(cands, [_Res()], ctl_formulas=["F_X1"],
                            out_path=Path(td) / "o.json", verbose=False)
        self.assertEqual(p["control"]["status"], "injected")
        self.assertEqual(p["n_checked"], 1)
        self.assertGreater(p["rows"][0]["base_r2_median"], 0.95)


# ═══════════════════════════════════════════════════════════
# D. 轮级聚合
# ═══════════════════════════════════════════════════════════

class TestEvaluateAggregation(_Base):

    def _evaluate(self, cands, results, **kw):
        with tempfile.TemporaryDirectory() as td:
            return co.evaluate(cands, results, out_path=Path(td) / "o.json",
                               verbose=False, **kw)

    def test_read_distribution_counts(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"},
                 {"factor_name": "indep", "formula": "F_INDEP"},
                 {"factor_name": "partial", "formula": "F_PARTIAL"}]
        p = self._evaluate(cands, [_Res()] * 3, ctl_formulas=["F_X1", "F_X2"])
        self.assertEqual(p["n_checked"], 3)
        self.assertEqual(p["n_replicated"], 1)
        self.assertEqual(p["n_increment_candidate"], 1)
        self.assertEqual(p["n_weak_increment"], 1)
        self.assertEqual(sum(p["read_distribution"].values()), 3)

    def test_single_candidate_error_isolated(self):
        """单候选面板缺失 ⇒ 不产生 row (None), 其余候选照常; 计数器完备。"""
        cands = [{"factor_name": "missing", "formula": "F_NOT_EXIST"},
                 {"factor_name": "copy", "formula": "F_COPY"}]
        p = self._evaluate(cands, [_Res()] * 2, ctl_formulas=["F_X1"])
        self.assertEqual(p["n_checked"], 1)
        self.assertEqual(p["rows"][0]["factor_name"], "copy")
        self.assertEqual(p["status"], "valid")

    def test_pick_priority_jq_first(self):
        cands = [{"factor_name": "a", "formula": "F_INDEP"},
                 {"factor_name": "b", "formula": "F_COPY"}]
        results = [_Res(eligible=False, s1=True), _Res(eligible=True)]
        p = self._evaluate(cands, results, ctl_formulas=["F_X1"], )
        # b 是 jq eligible ⇒ 优先层只含 b
        self.assertEqual([r["factor_name"] for r in p["rows"]], ["b"])

    def test_schema_and_status_fields(self):
        cands = [{"factor_name": "copy", "formula": "F_COPY"}]
        p = self._evaluate(cands, [_Res()], ctl_formulas=["F_X1"])
        self.assertEqual(p["schema"], co.SCHEMA)
        self.assertIn("updated_at", p)
        self.assertIn("params", p)
        self.assertEqual(p["params"]["fwd"], 1)
        self.assertEqual(p["n_data_insufficient"], 0)
        self.assertEqual(p["n_error"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
