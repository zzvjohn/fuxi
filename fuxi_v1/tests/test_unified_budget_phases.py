"""Budget boundaries across fixed phases; fixtures create no research evidence."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fuxi_v1 import unified_runner as controller
from fuxi_v1.contracts import ContractError, digest, load_json
from fuxi_v1.governance import LEASE_ENV
from fuxi_v1.tests.test_unified_integration import workspace, CODE_IDENTITY, TEMP_BASE


MIB = 1024*1024


class FakeTreeGuard:
    """Deterministic tree RSS; its launcher is still a real owned Popen handle."""

    rss_bytes = 30*MIB

    def __init__(self, process):
        self.process = process
        self.monitor_error = self.cleanup = None

    def sample(self):
        return self.rss_bytes

    def stop(self, seconds=3.):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=3)
        self.cleanup = {"complete": True, "survivors": [], "unverified": [],
                        "descendants_unverified": False, "signalled": []}
        return self.cleanup

    def diagnostics(self):
        return {"peak_tree_rss_bytes": self.rss_bytes, "peak_launcher_rss_bytes": self.rss_bytes,
                "max_children": 0, "observed_processes": [], "resource_samples": [],
                "monitor_error": self.monitor_error, "cleanup": self.cleanup}


class Clock:
    def __init__(self):
        self.value = 10.

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@contextmanager
def phases(root, *, fail_once=None, clock=None):
    """Check actual fixed command/lease contract; replace only costly consumers."""
    observations = []
    counters = {"research": 0, "feedback": 0, "catalog": 0}
    control = {"failed": False, "all_phases_finished": False}

    def execute(command, directory, env, *, timeout, memory_mb):
        if command[:5] != [sys.executable, "-B", "-X", "utf8", "-m"]:
            raise AssertionError("phase is not launched with the established interpreter")
        if command[5] != "fuxi_v1.unified_runner" or command[6] not in ("_catalog", "_campaign", "_feedback"):
            raise AssertionError("controller exposed an unregistered internal phase")
        if command[7:9] != ["--workspace", str(root)] or command[9] != "--request" or len(command) != 11:
            raise AssertionError("fixed phase command has an unexpected argument surface")
        name = command[6]
        specification = load_json(command[10])
        output = Path(specification["output"])
        if not output.is_relative_to(root):
            raise AssertionError("phase output escapes the declared workspace")
        lease = json.loads(env[LEASE_ENV])
        expected_action = {"_catalog": "index_refresh", "_campaign": "candidate_search", "_feedback": "feedback"}[name]
        if lease["action"] != expected_action:
            raise AssertionError("phase inherited another action")
        observations.append({"phase": name, "output": str(output), "timeout": timeout,
                             "memory_mb": memory_mb, "trigger": lease["trigger"],
                             "owner_pid": lease["pid"], "owner_run_id": lease["run_id"]})
        if not timeout > 0:
            raise AssertionError("expired phase was still launched")
        resource = {"exit_code": 0, "reason": None, "elapsed_seconds": .01,
                    "memory_limit_bytes": memory_mb*MIB, "timeout_seconds": timeout,
                    "tree": {"cleanup": {"complete": True}},
                    "controller_memory": {"measurement": "outer_controller_rss_plus_child_tree_rss",
                        "peak_controller_rss_bytes": 10, "peak_combined_rss_bytes": 20,
                        "sample_count": 1, "samples": [{"controller_rss_bytes": 10,
                            "child_tree_rss_bytes": 10, "combined_rss_bytes": 20}]}}
        if name == fail_once and not control["failed"]:
            control["failed"] = True
            resource["reason"] = "workflow_timeout"
            resource["exit_code"] = -1
            controller.atomic(Path(directory)/"RESOURCE.json", resource)
            raise ContractError("fixture phase workflow_timeout")
        if name == "_campaign":
            counters["research"] += 1
            report = output/"fixture_report.json"
            controller.atomic(report, {"fixture_only": True, "campaign_id": specification["campaign_id"]})
            controller.atomic(output/"tail_heavy.bin", {"fixture_only": True})
            controller.atomic(output/"RECEIPT.json", {"report_path": str(report),
                "report_sha256": controller.file_sha(report), "status": "completed", "attempted_candidates": 2,
                "valid_descriptive_factors": 2, "engineering_errors": 0, "formal_account_valid": 0,
                "economic_reward_count": 0, "jq_candidate_count": 0})
        elif name == "_catalog":
            counters["catalog"] += 1
            controller.atomic(output/"SUMMARY.json", {"fixture_only": True, "row_count": 1})
            if output.name == "catalog_after":
                control["all_phases_finished"] = True
        else:
            counters["feedback"] += 1
            feedback = {"fixture_only": True, "economic_rewards_imported": 0,
                        "qualification_receipts_imported": 0}
            controller.atomic(output/"IMPORT.json", feedback)
            controller.atomic(output/"RECEIPT.json", feedback)
        controller.atomic(Path(directory)/"RESOURCE.json", resource)
        return resource

    with ExitStack() as stack:
        stack.enter_context(patch("fuxi_v1.unified_runner.source_identity", return_value=deepcopy(CODE_IDENTITY)))
        stack.enter_context(patch("fuxi_v1.legacy_writers.inventory", return_value={"all_declared_boundaries_present": True}))
        stack.enter_context(patch("fuxi_v1.unified_runner.bounded_process", side_effect=execute))
        if clock is not None:
            stack.enter_context(patch("fuxi_v1.unified_runner.time.monotonic", side_effect=clock.monotonic))
        yield observations, counters, control


def completed_files(root):
    return list((root/"data/fuxi_unified/runs").glob("*/COMPLETE.json"))


class CombinedMemoryTests(unittest.TestCase):
    def test_real_short_child_records_outer_and_combined_peak(self):
        from fuxi_v1.controller_processes import ControllerTreeGuard
        with workspace() as (root, _, __), \
                patch("fuxi_v1.controller_processes.ControllerTreeGuard", FakeTreeGuard), \
                patch("psutil.Process", return_value=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=40*MIB))):
            result = controller.bounded_process([sys.executable, "-B", "-c", "import time;time.sleep(.25)"],
                root/"rss_ok", dict(os.environ), timeout=3, memory_mb=100)
            memory = result["controller_memory"]
            self.assertEqual(memory["measurement"], "outer_controller_rss_plus_child_tree_rss")
            self.assertGreater(memory["sample_count"], 0)
            self.assertEqual(memory["peak_controller_rss_bytes"], 40*MIB)
            self.assertEqual(memory["peak_combined_rss_bytes"], 70*MIB)
            self.assertTrue(all(row["combined_rss_bytes"] == row["controller_rss_bytes"]+row["child_tree_rss_bytes"]
                                for row in memory["samples"]))
            self.assertEqual(load_json(root/"rss_ok/RESOURCE.json")["controller_memory"], memory)
            self.assertTrue(result["tree"]["cleanup"]["complete"])

    def test_real_child_stops_when_combined_exceeds_cap_despite_each_part_below_cap(self):
        with workspace() as (root, _, __), \
                patch("fuxi_v1.controller_processes.ControllerTreeGuard", FakeTreeGuard), \
                patch.object(FakeTreeGuard, "rss_bytes", 60*MIB), \
                patch("psutil.Process", return_value=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=40*MIB))):
            with self.assertRaises(ContractError):
                controller.bounded_process([sys.executable, "-B", "-c", "import time;time.sleep(10)"],
                    root/"rss_exceeded", dict(os.environ), timeout=3, memory_mb=80)
            resource = load_json(root/"rss_exceeded/RESOURCE.json")
            self.assertEqual(resource["reason"], "workflow_memory_exceeded")
            self.assertEqual(resource["controller_memory"]["peak_combined_rss_bytes"], 100*MIB)
            self.assertLess(resource["tree"]["peak_tree_rss_bytes"], resource["memory_limit_bytes"])
            self.assertLess(resource["controller_memory"]["peak_controller_rss_bytes"], resource["memory_limit_bytes"])
            self.assertIsNotNone(resource["exit_code"])
            self.assertTrue(resource["tree"]["cleanup"]["complete"])


class FixedPhaseBudgetTests(unittest.TestCase):
    def test_all_fixed_phases_keep_one_automatic_owner_and_frozen_commands(self):
        with workspace() as (root, request, policy), phases(root) as (calls, counts, _):
            request = dict(request, include_catalog=True)
            normalized = controller.normalize_request(root, request)
            policy["bounded_research"]["approved_request_hashes"] = [digest(normalized)]
            controller.atomic(root/"codex_automations/fuxi_runtime_policy.json", policy)
            result = controller.run(root, request, trigger="automatic")
            self.assertEqual([row["phase"] for row in calls], ["_catalog", "_campaign", "_feedback", "_catalog"])
            self.assertEqual([Path(row["output"]).name for row in calls], ["catalog", "research", "feedback", "catalog_after"])
            self.assertEqual(counts, {"research": 1, "feedback": 1, "catalog": 2})
            self.assertTrue(all(row["trigger"] == "automatic" and row["owner_pid"] == os.getpid() for row in calls))
            self.assertEqual(len({row["owner_run_id"] for row in calls}), 1)
            self.assertEqual(result["state"], "completed")
            self.assertEqual(len(completed_files(root)), 1)
            self.assertFalse((root/"data/fuxi_runtime.lock").exists())

    def test_catalog_timeout_prevents_research_and_complete(self):
        with workspace() as (root, request, _), phases(root, fail_once="_catalog") as (calls, counts, __):
            with self.assertRaisesRegex(ContractError, "workflow_timeout"):
                controller.run(root, dict(request, include_catalog=True))
            self.assertEqual([row["phase"] for row in calls], ["_catalog"])
            self.assertEqual(counts["research"], 0)
            self.assertEqual(completed_files(root), [])
            self.assertEqual(len(list((root/"data/fuxi_unified/runs").glob("*/LAST_FAILURE.json"))), 1)

    def test_feedback_timeout_resume_uses_verified_research_checkpoint(self):
        with workspace() as (root, request, _), phases(root, fail_once="_feedback") as (calls, counts, __):
            request = dict(request, include_catalog=True)
            with self.assertRaisesRegex(ContractError, "workflow_timeout"):
                controller.run(root, request)
            self.assertEqual(counts["research"], 1)
            self.assertEqual(completed_files(root), [])
            checkpoints = list((root/"data/fuxi_unified/runs").glob("*/PHASE_RESEARCH.json"))
            self.assertEqual(len(checkpoints), 1)
            result = controller.run(root, request)
            self.assertEqual(result["state"], "completed")
            self.assertEqual(counts["research"], 1, "retry reran a completed native research phase")
            self.assertEqual([row["phase"] for row in calls].count("_campaign"), 1)
            self.assertEqual(counts["feedback"], 1)
            self.assertTrue(controller.run(root, request)["cached"])
            self.assertEqual(counts["research"], 1)

    def test_normalization_time_is_deducted_before_first_phase(self):
        clock = Clock()
        with workspace() as (root, request, _), phases(root, clock=clock) as (calls, _, __):
            original = controller.normalize_request
            def normalization(*args, **kwargs):
                result = original(*args, **kwargs)
                clock.advance(1000)
                return result
            with patch("fuxi_v1.unified_runner.normalize_request", side_effect=normalization):
                result = controller.run(root, dict(request, include_catalog=True))
            self.assertGreater(len(calls), 0)
            self.assertLessEqual(calls[0]["timeout"], controller.DEFAULT_LIMITS["max_total_seconds"]-1000)
            self.assertGreaterEqual(result["workflow_elapsed_seconds"], 1000)

    def test_artifact_hash_tail_can_exhaust_deadline_without_complete(self):
        clock = Clock()
        with workspace() as (root, request, _), phases(root, clock=clock) as (calls, counts, control):
            original = controller.file_sha
            advanced = {"value": False}
            def hashing(path):
                if Path(path).name == "tail_heavy.bin" and control["all_phases_finished"] and not advanced["value"]:
                    advanced["value"] = True
                    clock.advance(controller.DEFAULT_LIMITS["max_total_seconds"]+1)
                return original(path)
            with patch("fuxi_v1.unified_runner.file_sha", side_effect=hashing), self.assertRaises(ContractError):
                controller.run(root, dict(request, include_catalog=True))
            self.assertTrue(advanced["value"], "fixture did not reach the artifact hashing tail")
            self.assertEqual(counts["research"], 1)
            self.assertEqual(len(calls), 4)
            self.assertEqual(completed_files(root), [])
            self.assertEqual(len(list((root/"data/fuxi_unified/runs").glob("*/LAST_FAILURE.json"))), 1)


if __name__ == "__main__":
    unittest.main()
