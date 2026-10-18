"""Real process-tree lifecycle checks; these fixtures carry no financial evidence."""

from pathlib import Path
import json
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch

import psutil
from fuxi_v1.campaign import worker, _WorkerTreeGuard
from fuxi_v1.contracts import ContractError


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"reports/fuxi_v1_ten_rounds_20261009/worker_guard"
REAL_POPEN = subprocess.Popen
CHILD = '''from pathlib import Path
import json,sys,time,psutil
block=bytearray(int(sys.argv[2])*1024*1024)
for index in range(0,len(block),4096): block[index]=1
current=psutil.Process()
Path(sys.argv[1]).write_text(json.dumps({"pid":current.pid,"create_time":current.create_time(),"rss_bytes":current.memory_info().rss}),encoding="utf8")
time.sleep(float(sys.argv[3]))
'''
LAUNCHER = '''from pathlib import Path
import subprocess,sys
if sys.argv[1]=="failure":
 print("fixture failure",file=sys.stderr)
 raise SystemExit(7)
child=subprocess.Popen([sys.executable,"-B","-X","utf8",sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]])
child.wait()
if sys.argv[1]=="success":
 Path(sys.argv[6]).write_text('{"process_fixture":true}',encoding="utf8")
'''


def same_live(pid, birth):
    try:
        current = psutil.Process(pid)
        return current.create_time() == birth and current.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class WorkerTreeGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = EVIDENCE/"fixtures"/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.child_script, self.launcher_script = self.directory/"child.py", self.directory/"launcher.py"
        self.child_script.write_text(CHILD, encoding="utf8")
        self.launcher_script.write_text(LAUNCHER, encoding="utf8")
        # This unrelated live process is a sibling of the worker, not its child.
        self.unrelated = REAL_POPEN([sys._base_executable, "-B", "-c", "import time;time.sleep(90)"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.unrelated_birth = psutil.Process(self.unrelated.pid).create_time()
        self.launched = []
        self.fixture_known = []

    def tearDown(self):
        # Only this fixture's exact process identities can be cleaned up here.
        for pid, birth in self.fixture_known:
            try:
                if same_live(pid, birth):
                    psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
        for process in self.launched+[self.unrelated]:
            try:
                current = psutil.Process(process.pid)
                owned = [(child.pid, child.create_time()) for child in current.children(recursive=True)]
                owned.append((current.pid, current.create_time()))
                for pid, birth in owned:
                    if same_live(pid, birth):
                        psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
            try:
                process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.fail("fixture process was not reclaimed")

    def run_worker(self, mode, allocation=1, duration=30., timeout=8., limit=2048, monitor_failure=False):
        receipt = self.directory/"child_identity.json"
        output_directory = self.directory/"worker"

        def launch(original_command, **kwargs):
            self.assertEqual(original_command[3:5], ["-m", "fuxi_v1.workers"])
            command = [sys.executable, "-B", "-X", "utf8", str(self.launcher_script), mode,
                       str(self.child_script), str(receipt), str(allocation), str(duration),
                       str(output_directory/"result.json")]
            actual = REAL_POPEN(command, **kwargs)
            self.launched.append(actual)
            return actual

        original_sample = _WorkerTreeGuard.sample

        def failing_sample(guard):
            measured = original_sample(guard)
            if receipt.exists():
                raise psutil.AccessDenied(guard.process.pid)
            return measured

        with patch("fuxi_v1.campaign.subprocess.Popen", side_effect=launch):
            if monitor_failure:
                with patch.object(_WorkerTreeGuard, "sample", failing_sample):
                    result, state = worker({"process_fixture": True}, output_directory,
                                           timeout=timeout, memory_mb=limit)
            else:
                result, state = worker({"process_fixture": True}, output_directory,
                                       timeout=timeout, memory_mb=limit)
        self.assertTrue(same_live(self.unrelated.pid, self.unrelated_birth))
        for observed in state["process_tree"]["observed_processes"]:
            self.assertFalse(same_live(observed["pid"], observed["create_time"]))
        observed_child = json.loads(receipt.read_text(encoding="utf8")) if receipt.exists() else None
        if observed_child:
            self.assertFalse(same_live(observed_child["pid"], observed_child["create_time"]))
        with (EVIDENCE/"TEST_OBSERVATIONS.jsonl").open("ab") as stream:
            stream.write((json.dumps({"test": self.id(), "fixture_only": True,
                                      "unrelated": {"pid": self.unrelated.pid, "create_time": self.unrelated_birth,
                                                    "still_alive": True},
                                      "child_receipt": observed_child, "state": state})+"\n").encode("utf8"))
        return result, state, observed_child

    def test_memory_budget_observes_real_descendants_and_reclaims_only_tree(self):
        result, state, child = self.run_worker("waiting", allocation=96, limit=70)
        self.assertIsNone(result)
        self.assertEqual(state["reason"], "worker_memory_budget")
        # A fast guard may kill the allocated child before its independent
        # receipt is written. The actual psutil sample must still contain that
        # child's RSS, while the launcher's own RSS remains below the limit.
        self.assertGreater(max(member["rss_bytes"] for row in state["process_tree"]["resource_samples"]
                               for member in row["children"]), 32*1024*1024)
        self.assertLess(state["process_tree"]["peak_launcher_rss_bytes"], 70*1024*1024)
        self.assertGreater(state["process_tree"]["peak_tree_rss_bytes"], 70*1024*1024)
        self.assertGreaterEqual(state["process_tree"]["max_children"], 1)
        self.assertTrue(state["process_tree"]["cleanup"]["complete"])
        self.assertTrue(any(row["child_count"] and row["children"] for row in state["process_tree"]["resource_samples"]))

    def test_timeout_reclaims_descendants_without_touching_sibling(self):
        result, state, child = self.run_worker("waiting", timeout=2.)
        self.assertIsNotNone(child)
        self.assertIsNone(result)
        self.assertEqual(state["reason"], "worker_timeout")
        self.assertTrue(state["process_tree"]["cleanup"]["complete"])
        self.assertLess(state["elapsed_seconds"], 7.)

    def test_monitor_failure_is_closed_and_reclaims_known_tree(self):
        result, state, child = self.run_worker("waiting", monitor_failure=True)
        self.assertIsNotNone(child)
        self.assertIsNone(result)
        self.assertEqual(state["reason"], "worker_monitor_failure")
        self.assertEqual(state["process_tree"]["monitor_error"]["error_type"], "AccessDenied")
        self.assertTrue(state["process_tree"]["cleanup"]["complete"])

    def test_success_preserves_result_and_records_nonempty_tree_samples(self):
        result, state, child = self.run_worker("success", duration=.5)
        self.assertEqual(result, {"process_fixture": True})
        self.assertEqual(state["eval_status"], "valid")
        self.assertIsNotNone(child)
        self.assertGreaterEqual(state["process_tree"]["max_children"], 1)
        self.assertGreater(state["process_tree"]["peak_tree_rss_bytes"], state["process_tree"]["peak_launcher_rss_bytes"])

    def test_failure_reason_remains_worker_failure(self):
        result, state, child = self.run_worker("failure")
        self.assertIsNone(result)
        self.assertIsNone(child)
        self.assertEqual(state["reason"], "worker_failure")
        self.assertEqual(state["exit_code"], 7)

    def test_creation_time_mismatch_never_signals_unrelated_process(self):
        guard = _WorkerTreeGuard(self.unrelated)
        guard.known[self.unrelated.pid] = self.unrelated_birth+1.
        cleanup = guard.stop(seconds=.1)
        self.assertTrue(same_live(self.unrelated.pid, self.unrelated_birth))
        self.assertFalse(cleanup["complete"])
        self.assertEqual(cleanup["signalled"], [])
        self.assertEqual(cleanup["unverified"][0]["pid"], self.unrelated.pid)

    def test_initial_monitor_failure_stops_owned_launcher_and_blocks_unverified_tree(self):
        receipt = self.directory/"child_identity.json"
        directory = self.directory/"worker"

        def launch(command, **kwargs):
            actual = REAL_POPEN([sys.executable, "-B", "-X", "utf8", str(self.launcher_script), "waiting",
                                str(self.child_script), str(receipt), "1", "30", str(directory/"result.json")], **kwargs)
            self.launched.append(actual)
            return actual

        def cannot_initialize(guard, process):
            deadline = time.monotonic()+5.
            while not receipt.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(receipt.exists())
            # Test-only ownership inventory permits cleanup of the deliberate
            # orphan fixture. The failed production guard receives none of it.
            descendants = psutil.Process(process.pid).children(recursive=True)
            self.fixture_known = [(child.pid, child.create_time()) for child in descendants]
            raise psutil.AccessDenied(process.pid)

        with patch("fuxi_v1.campaign.subprocess.Popen", side_effect=launch), \
                patch.object(_WorkerTreeGuard, "__init__", cannot_initialize):
            with self.assertRaisesRegex(ContractError, "cleanup incomplete"):
                worker({"process_fixture": True}, directory, timeout=8., memory_mb=2048)
        guard = json.loads((directory/"process_guard.json").read_text(encoding="utf8"))
        self.assertFalse(guard["cleanup"]["complete"])
        self.assertTrue(guard["cleanup"]["launcher_exit_known"])
        self.assertTrue(guard["cleanup"]["launcher_stopped_via_owned_handle"])
        self.assertTrue(guard["cleanup"]["descendants_unverified"])
        self.assertTrue(same_live(self.unrelated.pid, self.unrelated_birth))
        with (EVIDENCE/"TEST_OBSERVATIONS.jsonl").open("ab") as stream:
            stream.write((json.dumps({"test": self.id(), "fixture_only": True,
                                      "campaign_blocked": True, "guard": guard,
                                      "unrelated_still_alive": True})+"\n").encode("utf8"))

    def test_cleanup_discovery_failure_never_claims_complete_inventory(self):
        receipt = self.directory/"child_identity.json"
        process = REAL_POPEN([sys.executable, "-B", "-X", "utf8", str(self.launcher_script), "waiting",
                              str(self.child_script), str(receipt), "1", "30", str(self.directory/"result.json")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.launched.append(process)
        deadline = time.monotonic()+5.
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(receipt.exists())
        self.fixture_known = [(child.pid, child.create_time()) for child in psutil.Process(process.pid).children(recursive=True)]
        guard = _WorkerTreeGuard(process)
        with patch.object(guard, "_discover", side_effect=psutil.AccessDenied(process.pid)):
            cleanup = guard.stop(seconds=.3)
        self.assertFalse(cleanup["complete"])
        self.assertTrue(cleanup["descendants_unverified"])
        self.assertTrue(same_live(self.unrelated.pid, self.unrelated_birth))


if __name__ == "__main__":
    unittest.main()
