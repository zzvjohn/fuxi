"""M15 定向测试：Stage2 队列五态分流 + 滞留台账 + 模式闸门。

被测契约（reports/fuxi_alpha_generation_overhaul_20260926.md §M15）:
  1. annotate 输出真实逐条五态与分母；「尚未发生」不得从输出消失
  2. 已有尝试记录 -> 拒绝/不足必须被记为已尝试，而不是「尚未发生」
  3. 滞留 >14 天触发 review_due（只报不阻塞）；未知年龄不得当 0 岁
  4. 模式闸门：annotate/audit 一律不回测；enforce 执行；run 为历史别名
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "stage2_s5_chain_m15_test", ROOT / "scripts" / "stage2_s5_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


class QueueTriageM15Tests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data = root / "data"
        self.data.mkdir()
        self.reports = root / "reports"
        self.reports.mkdir()
        now = datetime.now()
        self.pool = self.data / "stage1_factor_proposals.json"
        self.pool.write_text(json.dumps({"proposals": [
            {"factor_name": "f_old", "formula": "close.rank(pct=True)", "status": "PASS",
             "icir": 0.42, "generated_at": (now - timedelta(days=46)).isoformat(timespec="seconds")},
            {"factor_name": "f_new", "formula": "volume.rank(pct=True)", "status": "PASS",
             "icir": 0.36, "generated_at": (now - timedelta(days=1)).isoformat(timespec="seconds")},
            {"factor_name": "f_unknown_age", "formula": "open.rank(pct=True)", "status": "PASS",
             "icir": 0.31, "generated_at": "unknown-date"},
            {"factor_name": "f_bad", "formula": "close - close", "status": "PASS",
             "icir": 0.33, "generated_at": (now - timedelta(days=2)).isoformat(timespec="seconds")},
        ]}, ensure_ascii=False), encoding="utf-8")
        self.queue = self.data / "stage2_s5_pending.json"
        self.ledger = self.data / "stage2_s5_queue_age.json"
        self.verdicts = self.data / "stage2_s5_verdicts.json"
        self._patches = [
            patch.object(chain, "FACTOR_POOL", self.pool),
            patch.object(chain, "PENDING_QUEUE", self.queue),
            patch.object(chain, "AGE_LEDGER", self.ledger),
            patch.object(chain, "VERDICT_INDEX", self.verdicts),
            patch.object(chain, "REPORT_DIR", self.reports),
            patch.object(chain, "POOL_CSV", self.data / "passed_factor_pool.csv"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _run_main(self, mode, argv=(), **env_extra):
        env = {"FUXI_STAGE2_S5_CHAIN": mode}
        env.update(env_extra)
        buf = io.StringIO()
        with patch.dict(os.environ, env), \
             patch.object(sys, "argv", ["stage2_s5_chain.py", *argv]), \
             contextlib.redirect_stdout(buf):
            rc = chain._main_under_lock()
        return rc, buf.getvalue()

    def _snapshot(self):
        return json.loads(self.queue.read_text(encoding="utf-8"))

    def _ledger(self):
        return json.loads(self.ledger.read_text(encoding="utf-8"))

    def test_annotate_keeps_never_attempted_visible_with_denominator(self):
        rc, out = self._run_main("annotate", ["--queue-only"])
        self.assertEqual(rc, 0)
        snap = self._snapshot()
        self.assertEqual(snap["schema"], "stage2_s5_pending/v2")
        d = snap["denominator"]
        self.assertEqual(d["tracked"], 4)
        self.assertEqual(d["read"], 0)
        self.assertEqual(d["refused"], 0)
        self.assertEqual(d["not_yet_attempted"], 4)
        self.assertTrue(d["assert_split_ok"])
        self.assertEqual(d["by_state"], {"not_evaluated": 4})
        self.assertEqual(len(snap["pending"]), 4)  # 「尚未发生」不消失
        self.assertTrue(all(x["triage_state"] == "not_evaluated" for x in snap["pending"]))
        self.assertEqual(snap["preflight_projection"]["mode"], "not_run")
        self.assertIn("not_yet_attempted=4", out)

    def test_attempt_history_marks_prior_refusal_not_never_attempted(self):
        run_a = self.reports / "20260901_000000"
        run_a.mkdir()
        (run_a / "readings.json").write_text(json.dumps([
            {"factor_name": "f_old", "s5_eval_status": "data_insufficient", "s5_passed": False,
             "run_id": "20260901_000000", "checked_at": "2026-09-01T00:00:00",
             "s5_reason": "held_price_observations_missing"}]), encoding="utf-8")
        run_b = self.reports / "20260902_000000"
        run_b.mkdir()
        (run_b / "readings.json").write_text(json.dumps([
            {"factor_name": "f_new", "s5_eval_status": "valid", "s5_passed": True,
             "run_id": "20260902_000000", "checked_at": "2026-09-02T00:00:00",
             "s5_reason": "ok"}]), encoding="utf-8")
        rc, _ = self._run_main("annotate", ["--queue-only"])
        self.assertEqual(rc, 0)
        snap = self._snapshot()
        d = snap["denominator"]
        self.assertEqual(d["read"], 1)
        self.assertEqual(d["read_passed"], 1)
        self.assertEqual(d["refused"], 1)
        self.assertEqual(d["not_yet_attempted"], 2)
        self.assertTrue(d["assert_split_ok"])
        items = {x["factor_name"]: x for x in snap["pending"]}
        self.assertEqual(items["f_old"]["triage_state"], "data_insufficient")
        self.assertEqual(items["f_old"]["triage_source"], "attempt_history")
        self.assertEqual(items["f_old"]["n_attempts"], 1)
        self.assertEqual(items["f_new"]["triage_state"], "valid")
        self.assertTrue(items["f_new"]["triage_evidence"]["last_passed"])

    def test_retention_ledger_flags_two_week_backlog_without_blocking(self):
        rc, out = self._run_main("annotate")
        self.assertEqual(rc, 0)  # review_due 只报不阻塞
        led = self._ledger()
        self.assertEqual(led["threshold_days"], 14)
        self.assertEqual(led["n_tracked"], 4)
        self.assertEqual(led["n_review_due"], 1)
        self.assertEqual(led["review_due"], ["f_old"])
        self.assertGreater(led["items"]["f_old"]["age_days"], 40)
        self.assertEqual(led["items"]["f_old"]["age_source"], "generated_at")
        # 未知年龄 != 0 岁：不得计入 review_due，单独计 unknown
        self.assertIsNone(led["items"]["f_unknown_age"]["age_days"])
        self.assertIsNone(led["items"]["f_unknown_age"]["review_due"])
        self.assertEqual(led["n_age_unknown"], 1)
        self.assertIn("滞留评审触发", out)
        # 第二次运行：first_seen 从账本延续（f_unknown_age 不再 fallback）
        rc2, _ = self._run_main("annotate")
        self.assertEqual(rc2, 0)
        led2 = self._ledger()
        self.assertEqual(led2["items"]["f_unknown_age"]["age_source"], "ledger_first_seen")
        self.assertIsNotNone(led2["items"]["f_unknown_age"]["age_days"])

    def test_annotate_and_audit_never_execute_s5(self):
        for mode in ("annotate", "audit"):
            with patch.object(chain, "run_s5",
                              side_effect=AssertionError("run_s5 must not be called")):
                rc, out = self._run_main(mode, ["--limit", "2"])
            self.assertEqual(rc, 0, mode)
            self.assertIn("只标注不执行", out)

    def test_enforce_executes_and_run_remains_alias(self):
        self.assertEqual(chain.canonical_mode("run"), "enforce")
        for mode in ("enforce", "run"):
            calls = []

            def _capture(pending, limit, run_id, dry_run=False):
                calls.append({"n": len(pending), "limit": limit, "dry_run": dry_run})
                return [], {"s5_mode": "stub"}

            with patch.object(chain, "run_s5", side_effect=_capture):
                rc, _ = self._run_main(mode, ["--limit", "2"])
            self.assertEqual(rc, 0, mode)
            self.assertEqual(len(calls), 1, mode)
            self.assertFalse(calls[0]["dry_run"], mode)
            self.assertEqual(calls[0]["limit"], 2, mode)

    def test_audit_adds_shadow_projection_without_changing_state(self):
        rc, _ = self._run_main("audit", ["--queue-only"])
        self.assertEqual(rc, 0)
        snap = self._snapshot()
        pf = snap["preflight_projection"]
        self.assertEqual(pf["mode"], "shadow")
        self.assertEqual(pf["would_reject_under_enforce"], 1)
        d = snap["denominator"]
        self.assertEqual(d["not_yet_attempted"], 4)  # 投影不改写五态/分母
        items = {x["factor_name"]: x for x in snap["pending"]}
        self.assertTrue(items["f_bad"]["preflight_shadow"]["would_reject_under_enforce"])
        self.assertEqual(items["f_bad"]["triage_state"], "not_evaluated")
        self.assertFalse(items["f_old"]["preflight_shadow"]["would_reject_under_enforce"])

    def test_dry_run_preview_still_available_in_enforce_mode(self):
        rc, _ = self._run_main("enforce", ["--limit", "2", "--dry-run"],
                               FUXI_STAGE2_PREFLIGHT="off")
        self.assertEqual(rc, 0)
        self.assertFalse(self.verdicts.exists() and
                         json.loads(self.verdicts.read_text(encoding="utf-8"))["n_verdicts"] > 0)


if __name__ == "__main__":
    unittest.main()
