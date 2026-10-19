"""慢循环全流程合成测试：freeze → assess → 评分卡（四类 kind 三态建议）。

复用 gate 的合成资产构造，扩展为 8-task 比较套件（满足评分卡 P3 样本门槛），
验证 scorecard.build_scorecard 对 correctness/efficiency/research_method/economic_logic
四类系统进化的转正建议正确。
"""
import json
import tempfile
import unittest
from pathlib import Path

from fuxi_v1.contracts import digest
from research.system_evolution.gate import assess, freeze, _file_hash
from research.system_evolution.scorecard import build_scorecard

ROOT = Path(__file__).resolve().parents[2]

ENGINES = ("legacy_v07", "native_v10")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def make_tasks():
    """8-task 套件：每引擎 4 任务（2 replay + 2 transfer）。"""
    tasks = []
    for engine in ENGINES:
        for i, cohort in enumerate(["replay_exposed", "replay_exposed",
                                    "transfer_exposed", "transfer_exposed"]):
            task_id = f"{cohort}_{i}"
            tasks.append({"engine": engine, "task_id": task_id, "cohort": cohort,
                          "support_hash": f"{engine}:{task_id}", "window": "2024-01..2024-09"})
    return tasks


class SlowLoopFullFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fuxi_slow_loop_", dir=ROOT)
        self.root = Path(self.tmp.name).resolve()
        write(self.root / "codex_automations/fuxi_runtime_policy.json", {
            "single_writer_confirmed": True, "current_mode": "controlled_manual",
            "autonomous_writes_enabled": False})
        (self.root / "data").mkdir()
        for name, value in (("base", "base"), ("current", "current"), ("next", "next")):
            (self.root / (name + ".py")).write_text(value, encoding="utf-8")
        for name in ("judge", "data", "labels", "cost"):
            write(self.root / (name + ".json"), {"source": name, "version": "frozen"})
        self.tasks = make_tasks()
        self.suite = {"schema_version": "fuxi.system_evolution.1.suite", "tasks": self.tasks}
        write(self.root / "suite.json", self.suite)
        write(self.root / "noise.json", {"schema_version": "fuxi.system_evolution.1.aa", "pairs": [
            {"engine": "legacy_v07", "task_id": self.tasks[0]["task_id"], "repeat_id": "s1",
             "first": .50, "second": .502},
            {"engine": "legacy_v07", "task_id": self.tasks[1]["task_id"], "repeat_id": "s2",
             "first": .50, "second": .503}]})
        self.change_id = "proposal_slow"

    def tearDown(self):
        self.tmp.cleanup()

    def _plan(self, kind, cohorts):
        return {"schema_version": "fuxi.system_evolution.1", "change_id": self.change_id,
                "title": "shared change", "kind": kind,
                "engines": list(ENGINES), "component": "system_component",
                "hypothesis": "improves the system", "falsification": "no measured gain",
                "assets": {"baseline_code": "base.py", "incumbent_code": "current.py",
                           "challenger_code": "next.py", "suite": "suite.json",
                           "judge": "judge.json", "data": "data.json",
                           "labels": "labels.json", "cost": "cost.json", "aa_noise": "noise.json"},
                "thresholds": {"noise_floor": .003, "minimum_mean_gain": .01,
                               "maximum_case_loss": .10,
                               "quality_noninferiority_margin": .02, "minimum_wall_saving": 1},
                "budgets": {"wall_seconds": 30, "memory_mb": 100,
                            "external_requests": 0, "fee_cny": 0},
                "required_cohorts": cohorts}

    def _observe(self, role, scores, wall=10):
        """scores: dict {task_id: score}（缺省用 .5），wall: 每 task 秒数。"""
        facts = write(self.root / "evidence" / (role + ".json"),
                      {"engine_results": [{"task": t["task_id"], "engine": t["engine"],
                                           "score": scores.get(t["task_id"], .5)}
                                          for t in self.tasks]})
        rows = [{"engine": t["engine"], "task_id": t["task_id"],
                 "support_hash": t["support_hash"], "eval_status": "valid",
                 "score": scores.get(t["task_id"], .5),
                 "usage": {"wall_seconds": wall, "memory_mb": 12,
                           "external_requests": 0, "fee_cny": 0},
                 "source_fact": {"path": str(facts), "sha256": _file_hash(facts)}}
                for t in self.tasks]
        document = {"schema_version": "fuxi.system_evolution.1.observations", "role": role,
                    "suite_hash": digest(self.suite),
                    **{name + "_sha256": _file_hash(self.root / (name + ".json"))
                       for name in ("judge", "data", "labels", "cost")},
                    "cases": rows}
        return write(self.root / (role + ".json"), document)

    def _assess(self, kind, cohorts, challenger_scores, *, challenger_wall=10, incumbent_wall=10):
        write(self.root / "plan.json", self._plan(kind, cohorts))
        freeze(self.root, "plan.json")
        anchor = self._observe("anchor", {})
        incumbent = self._observe("incumbent", {}, wall=incumbent_wall)
        challenger = self._observe("challenger", challenger_scores, wall=challenger_wall)
        return assess(self.root, self.change_id, anchor, incumbent, challenger)

    def test_efficiency_recommend_promote(self):
        # wall 节省 20%（80→64s），score 全非劣（delta=0）
        self._assess("efficiency", ["replay_exposed", "transfer_exposed"],
                     {}, challenger_wall=8, incumbent_wall=10)
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "RECOMMEND_PROMOTE")
        p2 = next(g for g in card["gates"] if g["id"] == "P2")
        self.assertTrue(p2["met"])
        self.assertIn("20%", p2["value"])

    def test_efficiency_wall_saving_below_10pct_rejects(self):
        # wall 节省仅 5%（80→76s），gate 仍 review_shadow，但 P2 比例门槛不过 → REJECT
        self._assess("efficiency", ["replay_exposed", "transfer_exposed"],
                     {}, challenger_wall=9.5, incumbent_wall=10)
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "REJECT")
        p2 = next(g for g in card["gates"] if g["id"] == "P2")
        self.assertFalse(p2["met"])

    def test_correctness_sparse_fix_promotes(self):
        # 稀疏修复：仅 1 个 case 从 .5 → .55，其余保持 .5
        first = self.tasks[0]["task_id"]
        self._assess("correctness", ["replay_exposed"], {first: .55})
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "RECOMMEND_PROMOTE")
        p2 = next(g for g in card["gates"] if g["id"] == "P2")
        self.assertTrue(p2["met"])  # maxΔ>噪声，不被 0-case 稀释
        s_c1 = next(g for g in card["gates"] if g["id"] == "S-C1")
        self.assertTrue(s_c1["met"])

    def test_research_method_transfer_gain_promotes(self):
        # transfer cohort 全正向增益（.5 → .52），跨 2 引擎
        transfer_ids = [t["task_id"] for t in self.tasks if t["cohort"] == "transfer_exposed"]
        self._assess("research_method", ["transfer_exposed"], {t: .52 for t in transfer_ids})
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "RECOMMEND_PROMOTE")
        s_r2 = next(g for g in card["gates"] if g["id"] == "S-R2")
        self.assertTrue(s_r2["met"])

    def test_economic_logic_hold_without_forward_meta(self):
        # declared_forward 增益 beyond noise（效应 .013/.003≈4.3× ≥3×），
        # 但无 FORWARD_META → S-G1 不过 → HOLD
        self.tasks = [dict(t, cohort="declared_forward") for t in self.tasks]
        self.suite = {"schema_version": "fuxi.system_evolution.1.suite", "tasks": self.tasks}
        write(self.root / "suite.json", self.suite)
        scores = {t["task_id"]: .513 for t in self.tasks}  # 全 8 任务 delta=.013
        write(self.root / "plan.json", self._plan("economic_logic", ["declared_forward"]))
        freeze(self.root, "plan.json")
        anchor = self._observe("anchor", {})
        incumbent = self._observe("incumbent", {})
        challenger = self._observe("challenger", scores)
        assess(self.root, self.change_id, anchor, incumbent, challenger)
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "HOLD")
        s_g1 = next(g for g in card["gates"] if g["id"] == "S-G1")
        self.assertFalse(s_g1["met"])

    def test_material_regression_rejects(self):
        # challenger 显著退化（.5 → .3）→ gate 判 retain_incumbent → REJECT
        first = self.tasks[0]["task_id"]
        self._assess("research_method", ["transfer_exposed"], {first: .30})
        card = build_scorecard(self.root, self.change_id)
        self.assertEqual(card["recommendation"], "REJECT")
        p1 = next(g for g in card["gates"] if g["id"] == "P1")
        self.assertFalse(p1["met"])


if __name__ == "__main__":
    unittest.main()
