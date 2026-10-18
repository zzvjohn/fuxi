# -*- coding: utf-8 -*-
"""M6-A 生产接线整合测试 (2026-09-28)

方法: 从 `ralph_loop.py` **提取真实接线段**（textwrap.dedent 后受控 exec），
      用假 `self`/`generate_result` 驱动，断言账本行与零回归行为。

为什么不用复制逻辑: **测试形状须与生产一致** —— 复制会在生产改动后静默失配。
（与 M6-B `TestFreqPrecheckInRalphLoop` 同一纪律。）

覆盖: on+命中→on_refined / shadow→would_fire 且不动候选 / off→off_skipped /
      on 未命中周期→on_gate_miss 且不动候选 / 无种子→on_no_seed /
      evo 抛异常→error 行仍落（不静默）/ LOG=off→零行 / eval 段归因落行。

运行: C:/Users/Zhong/.workbuddy/binaries/python/envs/default/Scripts/python.exe -X utf8
      research/factor_alchemy/tests/test_s2a_refiner_wiring_20260928.py
"""
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import s2a_refiner_shadow as S  # noqa: E402
from ralph_loop import _s2a_refiner_gate  # noqa: E402

RALPH = BASE / "ralph_loop.py"

M_REFINE = "        # S2-a (2026-09-12): gp_evolve 精炼器补路 (可达性修复)"
M_REFINE_END = "        # ── P-20260907-004: 周频原生模板注入 (weekly 补给轮) ──"
M_EVAL = "        # ── M6-A (2026-09-28): S2-a 精炼器子代归因遥测 ──────────────"
M_EVAL_END = "        # ── Phase D: Distill ──────────────────────────────"


def _extract(start: str, end: str) -> str:
    src = io.open(RALPH, "r", encoding="utf-8", newline="").read()
    i = src.index(start)
    j = src.index(end, i)
    seg = src[i:j].replace("\r\n", "\n")
    # 精炼段的 `try:` 在 S2-a 注释块之后 ⇒ 从 try 起截
    if start == M_REFINE:
        head = "        try:\n"
        seg = head + seg[seg.index(head, seg.index(M_REFINE)) + len(head):]
    return textwrap.dedent(seg)


class _Arm:
    def __init__(self, name="筹码分布__ensemble", pulls=25, status="active"):
        self.name = name
        self.direction_id = name
        self.paradigm = "筹码分布"
        self.pulls = pulls
        self.status = status


class _MAB:
    def __init__(self, arm):
        self._arm = arm

    def resolve_arms(self, identifier):
        return [self._arm]


def _self_fake(kids=2, raise_evo=False, downgrade=None):
    def gen_evo(**kw):
        if raise_evo:
            raise RuntimeError("boom-evo")
        cands = [{"formula": f"ts_mean(close,{i + 7})", "_source": "gp_evolve",
                  "factor_name": f"kid{i}"} for i in range(kids)]
        return ("gp_evolve", cands, [])

    return SimpleNamespace(
        mab=None, _last_evo_downgrade=downgrade,
        _phase_generate_evo=gen_evo)


class TestRefineSegment(unittest.TestCase):
    """驱动真实精炼器段（含门控遥测落盘）。"""

    @classmethod
    def setUpClass(cls):
        cls.seg = _extract(M_REFINE, M_REFINE_END)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="m6a_wire_"))
        self.ledger = self.tmp / "ledger.jsonl"
        self._orig = S.DEFAULT_LEDGER
        S.DEFAULT_LEDGER = self.ledger          # 生产账本零污染

    def tearDown(self):
        S.DEFAULT_LEDGER = self._orig

    def _run(self, *, mode, pulls=25, every="5", min_pulls="3", kids=2,
             main_n=3, raise_evo=False, log="on", seeds=True):
        import os
        arm = _Arm(pulls=pulls)
        slf = _self_fake(kids=kids, raise_evo=raise_evo)
        slf.mab = _MAB(arm)
        gen = {"candidates": [{"formula": f"ts_mean(vol,{i + 3})",
                               "_source": "forge"} for i in range(main_n)],
               "n_candidates": main_n}
        env = {"os": os, "print": lambda *a, **k: None,
               "self": slf, "_refine_dir": arm, "paradigm": "筹码分布",
               "generator": "forge", "generate_result": gen,
               "gp_templates": [{"formula": "ts_std(amount,10)"}] if seeds else [],
               "retrieve_result": {"success_templates": []},
               "max_candidates": 10, "evo_turns": 5, "use_fsa": False,
               "loop_id": "test_loop_1", "_s2a_refiner_gate": _s2a_refiner_gate}
        old = {k: os.environ.get(k) for k in
               ("FUXI_EVO_REFINER", "FUXI_EVO_REFINER_EVERY",
                "FUXI_EVO_REFINER_MIN_PULLS", "FUXI_REFINER_SHADOW_LOG")}
        os.environ["FUXI_EVO_REFINER"] = mode
        os.environ["FUXI_EVO_REFINER_EVERY"] = every
        os.environ["FUXI_EVO_REFINER_MIN_PULLS"] = min_pulls
        os.environ["FUXI_REFINER_SHADOW_LOG"] = log
        try:
            exec(compile(self.seg, "<refine_seg>", "exec"), env)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        rows = []
        if self.ledger.exists():
            rows = [json.loads(x) for x in
                    io.open(self.ledger, encoding="utf-8") if x.strip()]
        return rows, gen

    # ── 主路径 ─────────────────────────────────────────────
    def test_on_hit_writes_on_refined_and_appends_kids(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1", kids=2)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["event"], "on_refined")
        self.assertEqual(r["arm"], "on")
        self.assertEqual(r["n_kids"], 2)
        self.assertEqual(r["n_main_candidates"], 3)
        self.assertEqual(len(gen["candidates"]), 5)        # 只追加
        self.assertEqual(gen["n_candidates"], 5)
        self.assertTrue(r["seed_sources"].startswith("main:3|retr:"))

    def test_shadow_would_fire_and_candidates_untouched(self):
        rows, gen = self._run(mode="shadow", every="1", min_pulls="1")
        self.assertEqual(rows[0]["event"], "shadow_would_fire")
        self.assertTrue(rows[0]["would_fire"])
        self.assertEqual(rows[0]["arm"], "shadow")
        self.assertEqual(len(gen["candidates"]), 3)        # 零回归
        self.assertIsNone(rows[0]["n_kids"])

    def test_off_still_logged(self):
        rows, gen = self._run(mode="off")
        self.assertEqual(rows[0]["event"], "off_skipped")
        self.assertEqual(rows[0]["arm"], "off")
        self.assertEqual(len(gen["candidates"]), 3)

    def test_on_period_miss_no_append(self):
        rows, gen = self._run(mode="on", every="5", min_pulls="3", pulls=12)
        self.assertEqual(rows[0]["event"], "on_gate_miss")
        self.assertFalse(rows[0]["probe"])
        self.assertEqual(len(gen["candidates"]), 3)

    def test_on_hit_without_seeds(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1", seeds=False)
        # 主路候选仍作种子（3 条）⇒ 仍能产出；用 0 主路 + 0 模板测无种子
        self.assertIn(rows[0]["event"], ("on_refined",))

    def test_on_no_seed_when_both_sources_empty(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1",
                              main_n=0, seeds=False)
        self.assertEqual(rows[0]["event"], "on_no_seed")
        self.assertEqual(rows[0]["n_seeds"], 0)
        self.assertEqual(len(gen["candidates"]), 0)

    def test_evo_exception_logged_as_error(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1", raise_evo=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "error")
        self.assertIn("boom-evo", rows[0]["error"])
        self.assertEqual(len(gen["candidates"]), 3)        # 不阻塞主路

    def test_log_off_writes_nothing(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1", log="off")
        self.assertEqual(rows, [])
        self.assertEqual(len(gen["candidates"]), 5)        # 行为不受账本开关影响

    def test_no_kids_records_downgrade(self):
        rows, gen = self._run(mode="on", every="1", min_pulls="1", kids=0)
        self.assertEqual(rows[0]["event"], "on_no_kids")
        self.assertEqual(rows[0]["n_kids"], 0)
        self.assertEqual(len(gen["candidates"]), 3)


class TestEvalSegment(unittest.TestCase):
    """驱动真实子代归因段。"""

    @classmethod
    def setUpClass(cls):
        cls.seg = _extract(M_EVAL, M_EVAL_END)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="m6a_ev_"))
        self.ledger = self.tmp / "ledger.jsonl"
        self._orig = S.DEFAULT_LEDGER
        S.DEFAULT_LEDGER = self.ledger

    def tearDown(self):
        S.DEFAULT_LEDGER = self._orig

    def _run(self, cands, results):
        import os
        env = {"os": os, "print": lambda *a, **k: None,
               "generate_result": {"candidates": cands},
               "evaluate_result": {"results": results},
               "loop_id": "test_loop_2", "paradigm": "筹码分布",
               "generator": "forge"}
        exec(compile(self.seg, "<eval_seg>", "exec"), env)
        rows = []
        if self.ledger.exists():
            rows = [json.loads(x) for x in
                    io.open(self.ledger, encoding="utf-8") if x.strip()]
        return rows

    def test_child_rows_written(self):
        cands = [{"formula": "ts_mean(vol,3)", "_source": "forge"},
                 {"formula": "ts_mean(close,7)", "_source": "gp_evolve"},
                 {"formula": "ts_std(amount,10)", "_source": "gp_evolve"}]
        res = [SimpleNamespace(s1_passed=False, s1_eval_status="valid", s1_n_days=300,
                               s2_passed=False, s5_passed=False,
                               s5_eval_status="valid"),
               SimpleNamespace(s1_passed=True, s1_eval_status="valid", s1_n_days=300,
                               s2_passed=True, s5_passed=False,
                               s5_eval_status="valid"),
               SimpleNamespace(s1_passed=False, s1_eval_status="data_insufficient",
                               s1_n_days=-1, s2_passed=False, s5_passed=False,
                               s5_eval_status="data_insufficient")]
        rows = self._run(cands, res)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["stage"], "eval")
        self.assertEqual(r["n_main"], 1)
        self.assertEqual(r["n_child"], 2)
        self.assertEqual(r["child"]["s1_pass"], 1)
        self.assertEqual(r["child"]["s1_read"], 1)
        self.assertEqual(r["child"]["s1_refused"], 1)
        self.assertEqual(r["main"]["s1_pass"], 0)
        self.assertEqual(r["run_id"], "test_loop_2")

    def test_no_child_no_row(self):
        rows = self._run([{"formula": "ts_mean(vol,3)", "_source": "forge"}],
                         [SimpleNamespace(s1_passed=False, s1_eval_status="valid")])
        self.assertEqual(rows, [])


class TestSignatureContract(unittest.TestCase):
    """回归守卫 (2026-09-28 实证缺陷)。

    `s2a_refiner_refine_telemetry` 曾漏 `error` 形参, 而生产 except 段以
    `error=` 调用它 ⇒ TypeError 被内层静默吞掉 ⇒ 「异常轮也落一行」永不为真。
    两条防线: (1) 行为级 —— error 入参被接受且落行;
             (2) 静态级 —— ralph_loop.py 全部调用点 kwarg ⊆ 模块签名。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="m6a_sig_"))
        self.ledger = self.tmp / "ledger.jsonl"

    def test_refine_telemetry_accepts_error_kwarg(self):
        rep = S.s2a_refiner_refine_telemetry(
            gate={"mode": "on", "mature": True, "every": 1,
                  "min_pulls": 1, "probe": True, "reason": "on_probe"},
            run_id="sig_loop", pulls=5, error="boom-evo",
            ledger_path=self.ledger, env={})
        self.assertEqual(rep["status"], "logged")
        self.assertEqual(rep["record"]["event"], "error")
        self.assertEqual(rep["record"]["error"], "boom-evo")

    def test_no_unexpected_kwargs_in_ralph_calls(self):
        import ast
        msrc = io.open(BASE / "s2a_refiner_shadow.py", "r",
                       encoding="utf-8", newline="").read()
        sigs = {}
        for n in ast.parse(msrc).body:
            if isinstance(n, ast.FunctionDef):
                allowed = {a.arg for a in n.args.args}
                allowed |= {a.arg for a in n.args.kwonlyargs}
                sigs[n.name] = (allowed, n.args.kwarg is not None)
        rsrc = io.open(RALPH, "r", encoding="utf-8", newline="").read()
        rtree = ast.parse(rsrc)
        aliases = {}
        for n in ast.walk(rtree):
            if (isinstance(n, ast.ImportFrom)
                    and (n.module or "").endswith("s2a_refiner_shadow")):
                for al in n.names:
                    aliases[al.asname or al.name] = al.name
        problems, checked = [], 0
        for n in ast.walk(rtree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name):
                name = aliases.get(f.id)
            elif isinstance(f, ast.Attribute):
                name = f.attr if f.attr in sigs else None
            else:
                name = None
            if name not in sigs:
                continue
            checked += 1
            allowed, has_kwargs = sigs[name]
            if has_kwargs:
                continue
            for kw in n.keywords:
                if kw.arg is not None and kw.arg not in allowed:
                    problems.append((n.lineno, name, kw.arg))
        self.assertEqual(checked, 3, f"wired calls expected 3, saw {checked}")
        self.assertEqual(problems, [], f"unexpected kwargs: {problems}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
