"""Controller identity regressions, including PID reuse and a real orphan tree."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
from fuxi_v1.controller_processes import ControllerTreeGuard


class FakeProcess:
    def __init__(self, system, pid):
        self.system, self.pid = system, pid
        self.birth = self._state()["birth"]

    def _state(self):
        state = self.system.states.get(self.pid)
        if state is None or not state["alive"]:
            raise psutil.NoSuchProcess(self.pid)
        return state

    def _same(self):
        state = self._state()
        if state["birth"] != self.birth:
            raise psutil.NoSuchProcess(self.pid)
        return state

    def create_time(self):
        return self.birth

    def status(self):
        self._same()
        return "running"

    def ppid(self):
        return self._same()["parent"]

    def children(self, recursive=False):
        self._same()
        if self.pid in self.system.denied:
            raise psutil.AccessDenied(self.pid)
        result = [FakeProcess(self.system, pid) for pid, state in self.system.states.items()
                  if state["alive"] and state["parent"] == self.pid]
        hook = self.system.discovery_hooks.pop(self.pid, None)
        if hook:
            hook()
        return result

    def memory_info(self):
        return SimpleNamespace(rss=self._same()["rss"])

    def kill(self):
        hook = self.system.signal_hooks.pop(self.pid, None)
        if hook:
            hook()
        self._same()  # psutil signals reject a reused identity as well.
        self.system.signals.append((self.pid, self.birth))
        self.system.states[self.pid]["alive"] = False


class FakeSystem:
    NoSuchProcess = psutil.NoSuchProcess
    STATUS_ZOMBIE = psutil.STATUS_ZOMBIE

    def __init__(self):
        self.states, self.signals = {}, []
        self.denied = set()
        self.discovery_hooks, self.signal_hooks = {}, {}

    def add(self, pid, birth, parent=0, rss=10):
        self.states[pid] = {"birth": birth, "parent": parent, "rss": rss, "alive": True}

    def Process(self, pid):
        return FakeProcess(self, pid)

    def wait_procs(self, processes, timeout):
        return processes, []


class ControllerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.system = FakeSystem()
        self.system.add(100, 1., rss=10)
        self.patcher = patch("psutil.Process", side_effect=self.system.Process)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def guard(self, **kwargs):
        result = ControllerTreeGuard(SimpleNamespace(pid=100), **kwargs)
        result.psutil = self.system
        return result

    def test_sequential_child_pid_reuse_is_a_new_verified_identity(self):
        self.system.add(200, 2., parent=100, rss=30)
        guard = self.guard()
        self.assertEqual(guard.sample(), 40)
        self.system.add(200, 3., parent=100, rss=50)
        self.assertEqual(guard.sample(), 60)
        self.assertEqual(guard.known[(200, 2.)]["state"], "retired")
        self.assertEqual(guard.known[(200, 3.)]["state"], "active")
        self.assertIsNone(guard.monitor_error)
        cleanup = guard.stop(seconds=0)
        self.assertTrue(cleanup["complete"])
        self.assertEqual(self.system.signals, [(200, 3.), (100, 1.)])
        self.assertNotIn((200, 2.), self.system.signals)
        self.assertEqual(guard.diagnostics()["observed_process_count"], 3)

    def test_unrelated_replacement_is_retired_without_adoption_or_signal(self):
        self.system.add(200, 2., parent=100)
        guard = self.guard()
        guard.sample()
        self.system.add(200, 3., parent=999, rss=5000)
        self.assertEqual(guard.sample(), 10)
        self.assertTrue(guard.stop(seconds=0)["complete"])
        self.assertTrue(self.system.states[200]["alive"])
        self.assertNotIn((200, 3.), guard.known)
        self.assertEqual(self.system.signals, [(100, 1.)])

    def test_launcher_pid_replacement_is_not_the_owned_launcher(self):
        guard = self.guard()
        self.system.add(100, 4., parent=999, rss=9000)
        self.system.add(200, 5., parent=100)
        self.assertEqual(guard.sample(), 0)
        self.assertTrue(guard.stop(seconds=0)["complete"])
        self.assertEqual(self.system.signals, [])
        self.assertEqual(guard.launcher_identity, (100, 1.))
        self.assertNotIn((200, 5.), guard.known)

    def test_verified_orphan_can_admit_and_reclaim_its_new_descendant(self):
        self.system.add(200, 2., parent=100)
        guard = self.guard()
        guard.sample()
        self.system.states[100]["alive"] = False
        self.system.states[200]["parent"] = 999
        self.system.add(300, 3., parent=200)
        self.system.add(900, 1., parent=999)
        self.assertEqual(guard.sample(), 20)
        self.assertEqual(set(guard.live_descendants()), {200, 300})
        first = guard.stop(seconds=0)
        second = guard.stop(seconds=0)
        self.assertTrue(first["complete"] and second["complete"])
        self.assertEqual(self.system.signals, [(300, 3.), (200, 2.)])
        self.assertEqual(first["signalled"], second["signalled"])
        self.assertTrue(self.system.states[900]["alive"])

    def test_parent_reuse_during_discovery_invalidates_children_snapshot(self):
        self.system.add(200, 2., parent=100)
        guard = self.guard()
        self.system.discovery_hooks[100] = lambda: self.system.add(100, 3., parent=999)
        self.assertEqual(guard.sample(), 0)
        self.assertNotIn((200, 2.), guard.known)
        self.assertTrue(guard.stop(seconds=0)["complete"])
        self.assertEqual(self.system.signals, [])

    def test_child_reuse_immediately_before_signal_never_signals_replacement(self):
        self.system.add(200, 2., parent=100)
        guard = self.guard()
        guard.sample()
        self.system.signal_hooks[200] = lambda: self.system.add(200, 4., parent=999)
        self.assertTrue(guard.stop(seconds=0)["complete"])
        self.assertTrue(self.system.states[200]["alive"])
        self.assertEqual(self.system.signals, [(100, 1.)])

    def test_bounded_sample_history_preserves_counts_identity_history_and_peak(self):
        guard = self.guard(history_limit=2)
        self.system.add(200, 2., parent=100, rss=900)
        self.system.add(300, 2., parent=100, rss=100)
        self.assertEqual(guard.sample(), 1010)
        self.system.states[300]["alive"] = False
        for birth in (3., 4., 5.):
            self.system.add(200, birth, parent=100, rss=1)
            guard.sample()
        result = guard.diagnostics()
        self.assertEqual(len(result["resource_samples"]), 2)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["discarded_sample_count"], 2)
        self.assertEqual(result["peak_tree_rss_bytes"], 1010)
        self.assertEqual(result["max_children"], 2)
        self.assertEqual(result["observed_process_count"], 6)
        self.assertLessEqual(len(result["identity_events"]), 2)
        self.assertGreater(result["identity_event_count"], 2)
        self.assertTrue(guard.stop(seconds=0)["complete"])

    def test_discovery_denial_remains_incomplete_while_verified_members_are_cleaned(self):
        self.system.add(200, 2., parent=100)
        guard = self.guard()
        guard.sample()
        self.system.denied.add(100)
        with self.assertRaises(psutil.AccessDenied):
            guard.sample()
        cleanup = guard.stop(seconds=0)
        self.assertFalse(cleanup["complete"])
        self.assertTrue(cleanup["descendants_unverified"])
        self.assertEqual(cleanup["survivors"], [])
        self.assertEqual(guard.monitor_error["error_type"], "AccessDenied")
        self.assertEqual(set(self.system.signals), {(100, 1.), (200, 2.)})


class ControllerRealProcessTests(unittest.TestCase):
    def test_short_sequential_children_and_observed_orphan_are_reclaimed(self):
        root = Path(__file__).resolve().parents[2]/"reports/fuxi_fusion_20261009/controller_process_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            receipt = Path(directory)/"orphan.json"
            code = """import json,subprocess,sys,time,psutil
from pathlib import Path
for unused in range(2):
 child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(.2)'])
 child.wait()
child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'])
current=psutil.Process(child.pid)
Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid,'create_time':current.create_time()}))
time.sleep(.35)
"""
            owned = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-c", code, str(receipt)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sibling = subprocess.Popen([sys._base_executable, "-B", "-c", "import time;time.sleep(30)"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            guard = None
            try:
                guard = ControllerTreeGuard(owned, history_limit=16)
                deadline = time.monotonic()+10
                while owned.poll() is None and time.monotonic() < deadline:
                    guard.sample()
                    time.sleep(.02)
                self.assertEqual(owned.wait(timeout=2), 0)
                orphan = json.loads(receipt.read_text())
                self.assertIn((orphan["pid"], orphan["create_time"]), guard.known)
                self.assertIn(orphan["pid"], guard.live_descendants())
                cleanup = guard.stop(seconds=3)
                self.assertTrue(cleanup["complete"], cleanup)
                self.assertIsNone(sibling.poll())
                self.assertGreaterEqual(guard.diagnostics()["observed_process_count"], 4)
                self.assertGreater(guard.peak_rss, guard.peak_launcher_rss)
                self.assertLessEqual(len(guard.diagnostics()["resource_samples"]), 16)
            finally:
                if guard is not None:
                    guard.stop(seconds=3)
                for process in (owned, sibling):
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
