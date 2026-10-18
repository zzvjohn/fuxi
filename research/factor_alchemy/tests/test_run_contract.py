from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from run_contract import (
    ActiveFuxiRunError,
    FuxiWriterLock,
    RunContract,
    RunReuseError,
    candidate_identity,
)


class RunContractTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / "reports" / "remediation_20260922" / "_test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / uuid.uuid4().hex
        self.tmp.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_identity_uses_full_formula(self):
        prefix = "x" * 200
        a = candidate_identity({"formula": prefix + "A", "source": "forge"})
        b = candidate_identity({"formula": prefix + "B", "source": "forge"})
        self.assertNotEqual(a["formula_sha256"], b["formula_sha256"])
        self.assertNotEqual(a["candidate_id"], b["candidate_id"])

    def test_live_lock_is_fail_closed(self):
        path = self.tmp / "runtime.lock"
        first = FuxiWriterLock(path, "one")
        first.acquire()
        try:
            with self.assertRaises(ActiveFuxiRunError):
                FuxiWriterLock(path, "two").acquire()
        finally:
            first.release()

    def test_manifest_is_atomic_and_run_id_is_immutable(self):
        runs = self.tmp / "runs"
        lock = self.tmp / "runtime.lock"
        with RunContract(parameters={"generator": "forge"}, run_id="r1",
                         runs_dir=runs, lock_path=lock) as contract:
            contract.phase("generate", "completed", n_candidates=1)
            contract.record_candidates([{"factor_name": "f", "formula": "rank(close)"}])

        manifest = json.loads((runs / "r1" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["phases"]["generate"]["n_candidates"], 1)
        self.assertEqual(len(manifest["candidates"][0]["formula_sha256"]), 64)
        self.assertFalse(list((runs / "r1").glob("*.tmp.*")))

        with self.assertRaises(RunReuseError):
            with RunContract(parameters={"generator": "forge"}, run_id="r1",
                             runs_dir=runs, lock_path=lock):
                pass

    def test_evaluation_status_is_persisted(self):
        class Result:
            def to_dict(self):
                return {"factor_name": "candidate", "s5": {"eval_status": "not_evaluated"}}

        runs = self.tmp / "runs"
        lock = self.tmp / "runtime.lock"
        with RunContract(parameters={}, run_id="evaluation",
                         runs_dir=runs, lock_path=lock) as contract:
            contract.record_evaluation(
                {"evaluation_status": {"s5": {"not_evaluated": 1}}},
                [Result()],
            )
        manifest = json.loads(
            (runs / "evaluation" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["evaluation"]["summary"]["evaluation_status"]["s5"]["not_evaluated"],
            1,
        )

    # ---- P4 (2026-09-25): manifest 覆盖面 ----

    def _manifest(self, name, **kwargs):
        runs = self.tmp / ("runs_" + name)
        lock = self.tmp / ("runtime_" + name + ".lock")
        with RunContract(parameters={}, run_id=name, runs_dir=runs, lock_path=lock,
                         **kwargs) as contract:
            return json.loads(contract.manifest_path.read_text(encoding="utf-8"))

    def test_data_sentinels_cover_the_decision_inputs(self):
        """停牌表 / 退市通道 / 证券主表 / 交易日历都是判据输入, 必须在标准集里。

        旧标准集只有 daily_prices / moneyflow_daily / 三个 json ⇒ manifest 声称覆盖
        「数据」, 却对「整日停牌」和「退市日期」这两条判据的输入一无所知。
        """
        manifest = self._manifest("coverage")
        keys = set(manifest["data_fingerprints"])
        for required in ("data/raw/daily_prices.csv",
                         "data/raw/suspend_d.csv",
                         "data/raw/delist_dates.csv",
                         "data/raw/stock_list.csv",
                         "data/raw/trade_calendar.csv"):
            self.assertIn(required, keys)
        self.assertEqual(manifest["fingerprint_coverage"]["standard_data"],
                         len(keys) - len(manifest["fingerprint_coverage"]
                                         ["extra_data_newly_added"]))

    def test_declared_extra_inputs_are_fingerprinted_and_attributed(self):
        """批次可以声明自己的脚本与数据, 并能在 manifest 里被认出来。"""
        code = self.tmp / "batch_script.py"
        code.write_text("# batch\n", encoding="utf-8")
        data = self.tmp / "batch_table.csv"
        data.write_text("trade_date\n20260104\n", encoding="utf-8")

        manifest = self._manifest("declared", extra_code=[code], extra_data=[data])
        code_keys = manifest["code_fingerprints"]
        data_keys = manifest["data_fingerprints"]

        declared_code = [k for k in code_keys if k.endswith("batch_script.py")]
        declared_data = [k for k in data_keys if k.endswith("batch_table.csv")]
        self.assertEqual(len(declared_code), 1)
        self.assertEqual(len(declared_data), 1)
        self.assertEqual(len(code_keys[declared_code[0]]["sha256"]), 64)
        self.assertEqual(manifest["fingerprint_coverage"]["extra_code"], declared_code)
        self.assertEqual(manifest["fingerprint_coverage"]["extra_data"], declared_data)

    def test_declaring_a_standard_input_again_is_not_a_new_key(self):
        """🔴 重复声明 ≠ 扩展标准集。

        批次常需要「显式声明某个标准输入」来表达「本批确实读了它」(例如为了证明
        停牌表属于判据输入而声明 suspend_d.csv)。若把这种声明也算进「新增键」,
        读者就会把「本来就有的键」误读成本批扩进来的键 —— 标准集大小与实际键数
        也会对不上。故已标准者归 already_standard, 只有真正的新键归 newly_added。
        """
        standard = Path(__file__).resolve().parents[3] / "data" / "raw" / "suspend_d.csv"
        self.assertTrue(standard.exists(), "标准输入应存在于仓库内")
        manifest = self._manifest("redeclare", extra_data=[standard])
        cover = manifest["fingerprint_coverage"]
        self.assertEqual(cover["extra_data"],
                         ["data/raw/suspend_d.csv"])
        self.assertEqual(cover["extra_data_already_standard"],
                         ["data/raw/suspend_d.csv"])
        self.assertEqual(cover["extra_data_newly_added"], [])
        self.assertEqual(len(manifest["data_fingerprints"]), cover["standard_data"])

    def test_a_declared_input_that_does_not_exist_is_refused(self):
        """声明了却不存在 ⇒ 抛错。允许它记一行 exists=False 会把声明变成不必兑现的话。"""
        with self.assertRaises(ValueError):
            RunContract(parameters={}, run_id="missing",
                        runs_dir=self.tmp / "runs_missing",
                        lock_path=self.tmp / "runtime_missing.lock",
                        extra_code=[self.tmp / "nope.py"])

    def test_an_input_outside_the_project_keeps_an_absolute_key(self):
        """项目外路径不能被 relative_to 静默拒掉。"""
        outside = Path(tempfile.mkdtemp(prefix="fuxi_p4_"))
        try:
            blob = outside / "outside_script.py"
            blob.write_text("# outside\n", encoding="utf-8")
            manifest = self._manifest("outside", extra_code=[blob])
            self.assertIn(blob.as_posix(), manifest["code_fingerprints"])
        finally:
            shutil.rmtree(outside, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
