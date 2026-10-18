# -*- coding: utf-8 -*-
"""M4 定向测试 (2026-09-27): 行为空间 QD 选择器挑战引擎。

四组:
  1. TestDescriptor     — 描述符分桶 (op_family / cx_bucket 边界)
  2. TestPoolLoad       — 真实矩阵池加载 (大小 / 字段 / valid 哨兵)
  3. TestSelectors      — legacy / build_grid / qd_pure / qd_fill (合成池硬断言)
  4. TestEvaluate       — 判定规则 (合成占优/不占优) + frac=1.0 收敛 + 结构

跑法: python -X utf8 -m unittest tests.test_m4_qd_selector_20260927 -v
"""
import json
import sys
import unittest
from pathlib import Path

RA = Path(__file__).resolve().parent.parent
if str(RA) not in sys.path:
    sys.path.insert(0, str(RA))

import behavior_qd_selector as qd                              # noqa: E402

MATRIX = (Path(__file__).resolve().parents[3]
          / "reports" / "fuxi_generator_full_batch_20260926" / "full_batch.json")


def synth(pool):
    """合成池条目: (expression, fitness, root_op, complexity) 元组列表。"""
    out = []
    for expr, fv, op, cx in pool:
        out.append(qd._cell_of({"expression": expr, "fitness": fv,
                                "root_op": op, "complexity": cx, "_arm": "synth"}))
    return out


class TestDescriptor(unittest.TestCase):
    def test_op_family_ts(self):
        self.assertEqual(qd.op_family("ts_delta"), "ts")
        self.assertEqual(qd.op_family("ts_std"), "ts")

    def test_op_family_unary_binary_other_unknown(self):
        self.assertEqual(qd.op_family("zscore"), "unary")
        self.assertEqual(qd.op_family("sqrt"), "unary")
        self.assertEqual(qd.op_family("div"), "binary")
        self.assertEqual(qd.op_family("rank"), "other")
        self.assertEqual(qd.op_family(""), "unknown")
        self.assertEqual(qd.op_family(None), "unknown")

    def test_cx_bucket_boundaries(self):
        self.assertEqual(qd.cx_bucket(0.3), "low")
        self.assertEqual(qd.cx_bucket(1.0), "low")     # 闭区间界
        self.assertEqual(qd.cx_bucket(1.01), "mid")
        self.assertEqual(qd.cx_bucket(3.5), "mid")     # 闭区间界
        self.assertEqual(qd.cx_bucket(3.51), "high")
        self.assertEqual(qd.cx_bucket(None), "unknown")

    def test_descriptor_pair(self):
        self.assertEqual(qd.descriptor({"root_op": "ts_rank", "complexity": 2.0}),
                         ("ts", "mid"))


class TestPoolLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = qd.load_qd_pool(MATRIX)

    def test_pool_total_and_sizes(self):
        sizes = [len(s["pool"]) for s in self.m["seeds"]]
        self.assertEqual(sum(sizes), 252)
        self.assertEqual(sizes.count(52), 3)
        self.assertEqual(sizes.count(48), 2)

    def test_fields_complete(self):
        c = self.m["seeds"][0]["pool"][0]
        for k in ("arm", "expression", "fitness", "valid", "positive",
                  "root_op", "complexity", "cell"):
            self.assertIn(k, c)

    def test_valid_sentinel(self):
        cells = synth([("a", -1000.0, "div", 1.0), ("b", 0.5, "div", 1.0)])
        self.assertFalse(cells[0]["valid"])
        self.assertTrue(cells[1]["valid"])
        self.assertTrue(cells[1]["positive"])

    def test_negative_fitness_is_valid_not_positive(self):
        cells = synth([("a", -0.2, "div", 1.0)])
        self.assertTrue(cells[0]["valid"])
        self.assertFalse(cells[0]["positive"])


class TestSelectors(unittest.TestCase):
    def test_legacy_topk_valid_first(self):
        pool = synth([("e1", 0.9, "div", 5.0),          # valid
                      ("e2", -999.5, "div", 5.0),       # invalid (哨兵)
                      ("e3", 0.5, "mul", 1.0)])         # valid
        top2 = qd.select_legacy(pool, 2)
        self.assertEqual([c["expression"] for c in top2], ["e1", "e3"])

    def test_legacy_k_zero_and_over(self):
        pool = synth([("e1", 0.9, "div", 5.0)])
        self.assertEqual(qd.select_legacy(pool, 0), [])
        self.assertEqual(len(qd.select_legacy(pool, 99)), 1)

    def test_legacy_deterministic_tiebreak(self):
        pool = synth([("b", 0.5, "div", 1.0), ("a", 0.5, "div", 1.0)])
        # 并列 fitness → expression 升序时 reverse=True 取 'b' 在前（降序）
        got = [c["expression"] for c in qd.select_legacy(pool, 2)]
        self.assertEqual(got, ["b", "a"])

    def test_build_grid_keeps_best_per_cell(self):
        pool = synth([("hi", 0.9, "div", 5.0),        # binary/high
                      ("lo", 0.1, "div", 5.0),        # binary/high 同格
                      ("other", 0.5, "mul", 1.0)])    # binary/low
        grid = qd.build_grid(pool)
        self.assertEqual(len(grid), 2)
        best = {tuple(pool[i]["cell"]): pool[i]["expression"] for i in grid.values()}
        self.assertEqual(best[("binary", "high")], "hi")

    def test_qd_pure_no_fill(self):
        # 3 个不同格: div->binary, ts_std->ts, rank->other（mul/div 同归 binary 族!）
        pool = synth([("a", 0.9, "div", 5.0), ("b", 0.8, "ts_std", 5.0),
                      ("c", 0.7, "rank", 1.0)])
        sel = qd.select_qd_pure(pool, 10)
        self.assertEqual(len(sel), 3)                 # 格数=3 < K=10 不填充

    def test_qd_fill_strict_k(self):
        pool = synth([("a", 0.9, "div", 5.0), ("b", 0.8, "div", 1.0),
                      ("c", 0.7, "mul", 5.0), ("d", 0.6, "mul", 1.0)])
        sel = qd.select_qd_fill(pool, 3)
        self.assertEqual(len(sel), 3)
        # 前两席 = 格代表（质量序），第三席 = 非代表填充
        self.assertEqual({sel[0]["cell"][0], sel[1]["cell"][0]}, {"binary"})
        self.assertEqual(len({tuple(c["cell"]) for c in sel[:2]}), 2)

    def test_qd_fill_reps_priority(self):
        # legacy 用槽位全给单格高分; qd_fill 用槽位换覆盖（另一格代表占席）
        pool = synth([("h1", 0.90, "div", 5.0),      # binary/high 代表
                      ("h2", 0.85, "div", 5.0),      # binary/high 非代表
                      ("t1", 0.20, "ts_std", 1.0),   # ts/low 代表
                      ("d1", 0.15, "div", 1.0)])     # binary/low 代表
        leg = qd.select_legacy(pool, 2)
        fil = qd.select_qd_fill(pool, 2)
        self.assertEqual([c["expression"] for c in leg], ["h1", "h2"])
        self.assertEqual([c["expression"] for c in fil], ["h1", "t1"])  # t1 代表占席

    def test_selectors_deterministic(self):
        pool = synth([("a", 0.9, "div", 5.0), ("b", 0.8, "div", 1.0),
                      ("c", 0.7, "mul", 5.0), ("d", 0.6, "ts_std", 1.0)])
        for fn in (qd.select_legacy, qd.select_qd_pure, qd.select_qd_fill):
            r1 = [c["expression"] for c in fn(pool, 3)]
            r2 = [c["expression"] for c in fn(pool, 3)]
            self.assertEqual(r1, r2)


class TestEvaluate(unittest.TestCase):
    def _synth_matrix(self, seed_pools):
        return {"schema": qd.POOL_SCHEMA,
                "seeds": [{"seed": i, "pool": p} for i, p in enumerate(seed_pools)],
                "arms": list(qd.DEFAULT_ARMS)}

    def test_structure(self):
        m = qd.load_qd_pool(MATRIX)
        rep = qd.evaluate_qd_matrix(m)
        self.assertEqual(rep["schema"], qd.SCHEMA)
        self.assertIn("per_seed", rep)
        self.assertIn("aggregate", rep)
        self.assertIn("verdict", rep)
        self.assertIn("limitations", rep)
        self.assertEqual(len(rep["per_seed"]), 5)

    def test_primary_frac_and_rule_constants(self):
        m = qd.load_qd_pool(MATRIX)
        rep = qd.evaluate_qd_matrix(m)
        v = rep["verdict"]
        self.assertEqual(v["primary_frac"], 0.5)
        self.assertEqual(v["primary_pair"], ["legacy", "qd_fill"])
        self.assertEqual(v["pos_seeds_required"], 4)  # ceil(0.8*5)
        self.assertIn("停止规则", v["decision_rule"])  # 规则文本为中文
        self.assertTrue(v["decision_rule"].startswith("预注册判定"))

    def test_frac1_converges_same_set(self):
        # 全预算: qd_fill 填满 -> 与 legacy 同为全池, 指标恒等
        m = qd.load_qd_pool(MATRIX)
        rep = qd.evaluate_qd_matrix(m, budget_fracs=(1.0,))
        for ps in rep["per_seed"]:
            s = ps["by_frac"]["1.0"]["selectors"]
            self.assertEqual(s["legacy"]["clusters"], s["qd_fill"]["clusters"])
            self.assertEqual(s["legacy"]["valid"], s["qd_fill"]["valid"])

    def test_verdict_stop_on_flat_pool(self):
        # 平池（单格全同结构）: QD 无覆盖优势 -> stop
        flat = synth([("x%d" % i, 0.5 - 0.01 * i, "div", 5.0) for i in range(20)])
        rep = qd.evaluate_qd_matrix(self._synth_matrix([flat] * 5))
        self.assertFalse(rep["verdict"]["qd_advantage"])
        self.assertTrue(rep["verdict"]["stop_rule_triggered"])

    def test_verdict_adv_on_constructed_advantage(self):
        # 构造占优: 6 条同格**同骨架**高质 (骨架剔除常数 ⇒ token 集相同 ⇒ 同簇)
        # + 3 条跨格代表 (各自独立骨架) + 16 条同格低质噪声。池 25 条, K=round(6.25)=6。
        # legacy top6 = h0..h5 (同簇 ⇒ clusters=1);
        # qd_fill = [h0,m_ts,m_mul,m_rank] + rest[:2]=[h1,h2] ⇒ clusters=4;
        # ⇒ delta_clusters=+3 (每 seed 恒正), delta_valid=0。
        import random
        rnd = random.Random(7)
        seeds = []
        for s in range(5):
            pool = []
            for j, win in enumerate((5, 10, 20, 60, 120, 250)):
                pool.append(("close.rolling(%d).mean() / open.rolling(%d).mean()"
                             % (win, win), 0.90 - 0.01 * j, "div", 5.0))  # binary/high
            pool.append(("ts_std(dollar_volume, 5)", 0.50, "ts_std", 5.0))  # ts/high
            pool.append(("turnover * volume", 0.45, "mul", 1.0))           # binary/low
            pool.append(("rank(vwap)", 0.40, "rank", 1.0))                 # other/low
            for j in range(16):
                pool.append(("n%d" % j, rnd.uniform(-0.5, 0.2), "div", 5.0))
            seeds.append(synth(pool))
        rep = qd.evaluate_qd_matrix(self._synth_matrix(seeds),
                                    budget_fracs=(0.25,), primary_frac=0.25)
        v = rep["verdict"]
        agg = rep["aggregate"]["0.25"]
        self.assertEqual(rep["per_seed"][0]["by_frac"]["0.25"]["K"], 6)
        self.assertGreater(agg["delta_clusters"]["median"], 0)
        self.assertGreaterEqual(agg["delta_clusters"]["pos_seeds"], 4)
        self.assertTrue(v["qd_advantage"])
        self.assertEqual(v["action"], "qd_selector_adopt_candidate")

    def test_evaluate_never_mutates_pool(self):
        m = qd.load_qd_pool(MATRIX)
        snapshot = json.dumps(m["seeds"][0]["pool"][:5], sort_keys=True, default=str)
        qd.evaluate_qd_matrix(m, budget_fracs=(0.25,))
        after = json.dumps(m["seeds"][0]["pool"][:5], sort_keys=True, default=str)
        self.assertEqual(snapshot, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
