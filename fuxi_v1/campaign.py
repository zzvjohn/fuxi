"""Serial, resumable research campaign with real actions, frozen policy and bounded feedback."""

from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
from .contracts import canonical, canonical_chunks, digest, require, load_json
from .artifacts import code_manifest, publish, verify
from .data import Snapshot, Labels
from .datasets import scope_of, validate_label_calendar
from .dsl import evaluate_ast
from .evidence import EvidenceStore, now
from .evaluation import factor_statistics, DEFAULT_POLICY
from .search import node, generate, llm_candidates
from .policy import calibration_request, register_calibration
from .models import walk_forward, capabilities as model_capabilities
from .monitoring import pareto_archive, drift_report, select_parents
from .acceptance import AcceptanceService
from .jq_strategy import compile_jq_strategy
from .statistics import holm
from .learning import (feedback_card, compatibility, export_memory, valid_imports, prompt_cards,
                       behavioral_signature, checkpoint, refresh_import_dependencies, structural_family, feedback_dependencies,
                       ast_diff_motifs, motif_ledger, motif_observations, motif_vetoes, motif_gate,
                       fitness_key, fitness_triage)
from .reachability import audit_reachability


def _result_name():
    """M3 (P-20261015-008): gzip result transport opt-in; plain JSON by default."""
    return "result.json.gz" if os.environ.get("FUXI_V1_RESULT_GZIP") == "1" else "result.json"


def _vehicle_pool(results):
    """Same-batch vehicle null readings (typed random arm) for percentile positioning only."""
    pool = []
    for row in results:
        value = row.get("mean_rank_ic")
        if (row.get("actual_action") == "calibration_random_null" and isinstance(value, (int, float))
                and not isinstance(value, bool) and value == value):
            pool.append(float(value))
    return pool


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "psutil", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        if isinstance(value, bytes):
            stream.write(value)
        else:
            for chunk in canonical_chunks(value):
                stream.write(chunk)
            stream.write(b"\n")


class _WorkerTreeGuard:
    """Account for the launcher's descendants; never signal a reused PID."""

    def __init__(self, process):
        import psutil
        self.psutil, self.process = psutil, process
        self.known, self.samples = {}, []
        self.started = time.monotonic()
        self.peak_rss, self.peak_launcher_rss, self.max_children = 0, 0, 0
        self.monitor_error = None
        self.cleanup = None
        self.known[process.pid] = psutil.Process(process.pid).create_time()

    def _live(self, pid, birth):
        try:
            current = self.psutil.Process(pid)
            if current.create_time() != birth:
                raise RuntimeError("worker PID identity changed: " + str(pid))
            if current.status() == self.psutil.STATUS_ZOMBIE:
                return None
            return current
        except self.psutil.NoSuchProcess:
            return None

    def _discover(self):
        # A previously seen child may outlive the launcher. Its descendants are
        # still within this worker tree, even after the OS reparents that child.
        for pid, birth in list(self.known.items()):
            current = self._live(pid, birth)
            if current is None:
                continue
            try:
                children = current.children(recursive=True)
            except self.psutil.NoSuchProcess:
                continue
            for child in children:
                try:
                    child_birth = child.create_time()
                except self.psutil.NoSuchProcess:
                    continue
                if child.pid in self.known and self.known[child.pid] != child_birth:
                    raise RuntimeError("worker descendant PID identity changed: " + str(child.pid))
                self.known[child.pid] = child_birth

    def sample(self):
        self._discover()
        members = []
        for pid, birth in list(self.known.items()):
            current = self._live(pid, birth)
            if current is None:
                continue
            try:
                rss = current.memory_info().rss
                # Recheck identity after reading RSS, rather than attributing a
                # racing replacement process to this worker.
                if self._live(pid, birth) is None:
                    continue
            except self.psutil.NoSuchProcess:
                continue
            members.append({"pid": pid, "create_time": birth, "rss_bytes": rss})
        total = sum(member["rss_bytes"] for member in members)
        launcher = next((member["rss_bytes"] for member in members if member["pid"] == self.process.pid), 0)
        children = [member for member in members if member["pid"] != self.process.pid]
        self.peak_rss = max(self.peak_rss, total)
        self.peak_launcher_rss = max(self.peak_launcher_rss, launcher)
        self.max_children = max(self.max_children, len(children))
        self.samples.append({"elapsed_seconds": time.monotonic()-self.started,
                             "tree_rss_bytes": total, "launcher_rss_bytes": launcher,
                             "peak_tree_rss_bytes": self.peak_rss,
                             "child_count": len(children), "children": children})
        return total

    def live_descendants(self):
        return [pid for pid, birth in self.known.items()
                if pid != self.process.pid and self._live(pid, birth) is not None]

    def stop(self, seconds=3.):
        deadline = time.monotonic()+seconds
        signalled, errors = set(), []
        survivors, unverified, discovery_uncertain = [], [], False
        while True:
            try:
                self._discover()
            except Exception as exc:
                discovery_uncertain = True
                errors.append({"operation": "discover", "error_type": type(exc).__name__, "reason": str(exc)})
            waiting, survivors, unverified = [], [], []
            # Descendants first, launcher last. Every signal checks the frozen
            # (PID, creation time); no process-name search or global kill occurs.
            ordered = sorted(self.known.items(), key=lambda row: row[0] == self.process.pid)
            for pid, birth in ordered:
                try:
                    current = self._live(pid, birth)
                    if current is None:
                        continue
                    survivors.append({"pid": pid, "create_time": birth})
                    current.kill()
                    signalled.add((pid, birth))
                    waiting.append(current)
                except self.psutil.NoSuchProcess:
                    continue
                except Exception as exc:
                    unverified.append({"pid": pid, "create_time": birth})
                    errors.append({"operation": "kill", "pid": pid,
                                   "error_type": type(exc).__name__, "reason": str(exc)})
            if (not survivors and not unverified) or time.monotonic() >= deadline:
                break
            try:
                self.psutil.wait_procs(waiting, timeout=min(.25, max(0., deadline-time.monotonic())))
            except Exception as exc:
                errors.append({"operation": "wait", "error_type": type(exc).__name__, "reason": str(exc)})
            if not waiting:
                time.sleep(min(.05, max(0., deadline-time.monotonic())))
        self.cleanup = {"signalled": [{"pid": pid, "create_time": birth} for pid, birth in sorted(signalled)],
                        "survivors": survivors, "unverified": unverified, "errors": errors,
                        "descendants_unverified": discovery_uncertain,
                        "complete": not survivors and not unverified and not discovery_uncertain,
                        "budget_seconds": seconds}
        return self.cleanup

    def diagnostics(self):
        return {"memory_measurement": "sum_process_tree_rss", "peak_tree_rss_bytes": self.peak_rss,
                "peak_launcher_rss_bytes": self.peak_launcher_rss, "max_children": self.max_children,
                "observed_processes": [{"pid": pid, "create_time": birth} for pid, birth in sorted(self.known.items())],
                "resource_samples": self.samples, "monitor_error": self.monitor_error,
                "cleanup": self.cleanup}


def _worker_spawn(request, directory, *, timeout=120, memory_mb=2048):
    """Spawn-only phase of worker(): write request, launch child, start guard.

    Split from worker() for M1 candidate-level parallelism (P-20261015-002):
    every pending candidate is spawned first, then collected in proposals
    order, so wall-clock = max(child) instead of sum(child) while the facts
    write order stays bit-identical to the serial path.
    """
    import psutil
    directory.mkdir(parents=True, exist_ok=False)
    request_path, output = directory/"request.json", directory/_result_name()
    from .full_market_support import pack_grids
    write_new(request_path, pack_grids(request, directory / "request_panels"))
    env = {k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    stdout = (directory/"stdout.txt").open("wb")
    stderr = (directory/"stderr.txt").open("wb")
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "fuxi_v1.workers", "--request", str(request_path),
                                "--output", str(output)], cwd=Path(__file__).resolve().parents[1],
                               stdout=stdout, stderr=stderr, env=env)
    reason, guard, launcher_cleanup_error = None, None, None
    initialization_error = None
    try:
        guard = _WorkerTreeGuard(process)
    except Exception as exc:
        reason = "worker_monitor_failure"
        initialization_error = {"error_type": type(exc).__name__, "reason": str(exc)}
        try:
            process.kill()
        except Exception as cleanup_exc:
            launcher_cleanup_error = {"error_type": type(cleanup_exc).__name__, "reason": str(cleanup_exc)}
    return {"directory": directory, "process": process, "guard": guard, "started": started,
            "stdout": stdout, "stderr": stderr, "timeout": timeout, "memory_mb": memory_mb,
            "reason": reason, "initialization_error": initialization_error,
            "launcher_cleanup_error": launcher_cleanup_error}


def _worker_collect(handle):
    """Collect phase of worker(): poll/monitor, wait, diagnostics, read result."""
    process, guard = handle["process"], handle["guard"]
    started, timeout, memory_mb = handle["started"], handle["timeout"], handle["memory_mb"]
    directory = handle["directory"]
    reason, launcher_cleanup_error = handle["reason"], handle["launcher_cleanup_error"]
    initialization_error = handle["initialization_error"]
    try:
        while process.poll() is None:
            if time.monotonic()-started > timeout:
                reason = "worker_timeout"
            try:
                if guard is not None and guard.sample() > memory_mb*1024*1024:
                    reason = "worker_memory_budget"
            except Exception as exc:
                reason = "worker_monitor_failure"
                guard.monitor_error = {"error_type": type(exc).__name__, "reason": str(exc)}
            if reason:
                if guard is not None:
                    guard.stop()
                break
            time.sleep(.1)
        if reason is None and guard is not None:
            try:
                if guard.live_descendants():
                    reason = "worker_orphaned_process"
                    guard.stop()
            except Exception as exc:
                reason = "worker_monitor_failure"
                guard.monitor_error = {"error_type": type(exc).__name__, "reason": str(exc)}
                guard.stop()
        try:
            code = process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            reason = reason or "worker_timeout"
            code = process.poll()
    finally:
        handle["stdout"].close()
        handle["stderr"].close()
    elapsed = time.monotonic()-started
    resources = guard.diagnostics() if guard is not None else {
        "memory_measurement": "sum_process_tree_rss", "monitor_error": initialization_error,
        "cleanup": {"complete": False, "launcher_exit_known": process.poll() is not None,
                    "launcher_stopped_via_owned_handle": launcher_cleanup_error is None,
                    "launcher_cleanup_error": launcher_cleanup_error, "descendants_unverified": True,
                    "reason": "launcher identity unavailable; descendants were not signalled"}}
    resources.update(memory_limit_bytes=memory_mb*1024*1024, timeout_seconds=timeout)
    write_new(directory/"process_guard.json", resources)
    require(not reason or resources["cleanup"]["complete"],
            "worker process tree cleanup incomplete; see process_guard.json; campaign must stop")
    if code != 0 or reason:
        error = (directory/"stderr.txt").read_text(encoding="utf-8", errors="replace")[-5000:]
        return None, {"eval_status": "error", "reason": reason or "worker_failure", "exit_code": code,
                      "stderr": error, "elapsed_seconds": elapsed, "process_tree": resources}
    from .full_market_support import unpack_grids
    packed = load_json(directory/_result_name())
    result = unpack_grids(packed)
    if isinstance(packed.get("scores"), dict) and packed["scores"].get("schema_version") == "fuxi.score_grid_ref.1":
        result["_score_transport"] = packed["scores"]
    return result, {"eval_status": "valid", "exit_code": code, "elapsed_seconds": elapsed,
                    "process_tree": resources}


def worker(request, directory, *, timeout=120, memory_mb=2048):
    """Serial spawn+collect wrapper (unchanged semantics for legacy callers)."""
    handle = _worker_spawn(request, directory, timeout=timeout, memory_mb=memory_mb)
    return _worker_collect(handle)


class _PoolSlot:
    """One long-lived loop-mode worker process (M2, P-20261015-007).

    SPEEDUP 4b state discipline: the worker keeps a read-only snapshot cache
    keyed by (input_path, input_hash) and resets every per-request semantic
    before each job; it exits after ``max_requests`` jobs so this slot can
    recycle the process against memory creep. The launcher-side reader thread
    only transports status lines; facts stay deterministic (completions are
    matched by request_index, never by arrival order).
    """

    def __init__(self, index, root, timeout, memory_mb, max_requests, cwd):
        self.index = index
        self.root = root
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.max_requests = max_requests
        self.cwd = cwd
        self.process = None
        self.stdin = None
        self.lines = None
        self.pending = {}
        self.eof = False
        self.guard = None
        self.stderr_path = root / ("worker_%d.stderr.txt" % index)
        self.started = None
        self.total_requests = 0
        self.generation_requests = 0

    def _spawn(self):
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        args = [sys.executable, "-X", "utf8", "-m", "fuxi_v1.workers", "--loop"]
        if self.max_requests is not None:
            args += ["--max-requests", str(self.max_requests)]
        stderr = open(self.stderr_path, "ab")
        self.process = subprocess.Popen(args, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, env=env, text=True, encoding="utf-8", bufsize=1)
        self.stdin = self.process.stdin
        self.guard = _WorkerTreeGuard(self.process)
        self.started = time.monotonic()
        # Capture the queue reference for THIS process generation: the previous
        # generation's reader thread must keep writing to its own queue, never
        # to the replacement's (a stale EOF sentinel would otherwise be
        # mistaken for the new process dying).
        lines = queue.Queue()
        self.lines = lines
        self.pending = {}
        self.eof = False
        reader = threading.Thread(target=self._drain, args=(self.process.stdout, lines), daemon=True)
        reader.start()

    def _drain(self, stream, lines):
        try:
            for raw in stream:
                try:
                    lines.put(json.loads(raw.strip()))
                except Exception as exc:
                    lines.put({"ok": False, "request_index": None,
                               "error_type": type(exc).__name__, "reason": str(exc)})
        finally:
            lines.put({"eof": True})

    def _pump(self):
        """Move available lines into self.pending; return True if EOF was seen."""
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line.get("eof"):
                self.eof = True
            elif line.get("request_index") is not None:
                self.pending[line["request_index"]] = line

    def _teardown(self):
        if self.process is not None:
            try:
                self.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=3.)
            except Exception:
                pass
        self.process = None
        self.stdin = None
        self.pending = {}
        self.eof = False
        self.generation_requests = 0

    def ensure(self):
        if self.process is None:
            self._spawn()
            return
        # Recycle the process when the current generation reached its request
        # budget BEFORE submitting the next job: relying on the worker to exit
        # on its own leaves a window where the job line is written into the
        # dying process's stdin pipe and lost when the pipe closes.
        if self.max_requests is not None and self.generation_requests >= self.max_requests:
            self._teardown()
            self._spawn()
            return
        self._pump()
        if self.eof or self.process.poll() is not None:
            self._teardown()
            self._spawn()

    def submit(self, request, directory, request_index):
        self.ensure()
        request_path, output = directory / "request.json", directory / _result_name()
        from .full_market_support import pack_grids
        write_new(request_path, pack_grids(request, directory / "request_panels"))
        line = dict(request, output=str(output), request_index=request_index)
        self.stdin.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stdin.flush()
        self.total_requests += 1
        self.generation_requests += 1
        return request_index

    def _kill(self, reason):
        cleanup = None
        if self.guard is not None:
            cleanup = self.guard.stop()
            try:
                self.process.wait(timeout=3.)
            except Exception:
                pass
        return cleanup

    def wait_line(self, request_index, started):
        """Block until this request's status line arrives (order-matched)."""
        deadline = started + self.timeout
        reason = None
        cleanup = None
        while True:
            self._pump()
            line = self.pending.pop(request_index, None)
            if line is not None:
                return line, reason, cleanup
            if reason is None and self.guard is not None:
                try:
                    if self.guard.sample() > self.memory_mb * 1024 * 1024:
                        reason = "worker_memory_budget"
                except Exception as exc:
                    reason = "worker_monitor_failure"
                    self.guard.monitor_error = {"error_type": type(exc).__name__, "reason": str(exc)}
            if self.eof and not self.pending:
                return None, "worker_unexpected_exit", cleanup
            if time.monotonic() > deadline and reason is None:
                reason = "worker_timeout"
            if reason is not None:
                cleanup = self._kill(reason)
                self._pump()
                line = self.pending.pop(request_index, None)
                if line is not None:
                    return line, reason, cleanup
                return None, reason, cleanup
            time.sleep(.1)


class _PersistentPool:
    """M2 candidate-evaluation pool: K loop-mode workers, order-matched collection."""

    def __init__(self, size, root, timeout, memory_mb, max_requests=None, cwd=None):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.slots = [_PoolSlot(i, root, timeout, memory_mb, max_requests, cwd) for i in range(max(1, size))]
        self._next = 0
        self._counter = 0

    def spawn(self, request, directory):
        slot = self.slots[self._next % len(self.slots)]
        self._next += 1
        self._counter += 1
        index = self._counter
        slot.submit(request, directory, index)
        return {"directory": directory, "slot": slot, "request_index": index,
                "started": time.monotonic(), "timeout": self.timeout, "memory_mb": self.memory_mb}

    def collect(self, handle):
        directory, slot = handle["directory"], handle["slot"]
        started, timeout, memory_mb = handle["started"], handle["timeout"], handle["memory_mb"]
        line, reason, cleanup = slot.wait_line(handle["request_index"], started)
        elapsed = time.monotonic() - started
        if reason is None and cleanup is None and slot.guard is not None:
            try:
                if slot.guard.live_descendants():
                    reason = "worker_orphaned_process"
                    cleanup = slot._kill(reason)
            except Exception as exc:
                reason = "worker_monitor_failure"
                slot.guard.monitor_error = {"error_type": type(exc).__name__, "reason": str(exc)}
                cleanup = slot._kill(reason)
        resources = slot.guard.diagnostics() if slot.guard is not None else {
            "memory_measurement": "sum_process_tree_rss", "monitor_error": None,
            "cleanup": {"complete": False, "reason": "guard unavailable"}}
        resources.update(memory_limit_bytes=memory_mb * 1024 * 1024, timeout_seconds=timeout)
        guard_audit = {"eval_status": "valid", "worker_mode": "persistent_pool", "slot": slot.index,
                       "request_index": handle["request_index"], "worker_total_requests": slot.total_requests,
                       "cleanup": cleanup, "guard": resources}
        write_new(directory / "process_guard.json", guard_audit)
        if line is None or line.get("ok") is not True:
            stderr = ""
            if slot.stderr_path.exists():
                stderr = slot.stderr_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            if line is None:
                failure_reason = reason or "worker_failure"
            else:
                failure_reason = line.get("reason") or "worker_error"
            slot._teardown()
            return None, {"eval_status": "error", "reason": failure_reason,
                          "exit_code": None, "stderr": stderr, "elapsed_seconds": elapsed,
                          "process_tree": resources, "worker_mode": "persistent_pool", "slot": slot.index}
        from .full_market_support import unpack_grids
        packed = load_json(directory / _result_name())
        result = unpack_grids(packed)
        if isinstance(packed.get("scores"), dict) and packed["scores"].get("schema_version") == "fuxi.score_grid_ref.1":
            result["_score_transport"] = packed["scores"]
        return result, {"eval_status": "valid", "exit_code": 0, "elapsed_seconds": elapsed,
                        "process_tree": resources, "worker_mode": "persistent_pool", "slot": slot.index,
                        "worker_requests_in_slot": slot.total_requests}

    def close(self):
        for slot in self.slots:
            if slot.process is None:
                continue
            try:
                slot.stdin.close()
            except Exception:
                pass
            try:
                slot.process.wait(timeout=5.)
            except Exception:
                try:
                    slot.guard.stop() if slot.guard is not None else slot.process.kill()
                except Exception:
                    pass


def _minimum_exploration_arm(arms, history, window, min_visits=1):
    """确定性最小探索保证（Deterministic Minimum Exploration Guarantee, MEG）。

    返回最近 window 轮内选中次数 < min_visits 的臂中最久未被选中者；无饥饿臂返回 None。
    目标：capability_cost_UCB 的探索项 sqrt(2*log(t)/tries) 随 tries 单调衰减，无经济奖励时
    退化为纯 valid/cost 效率竞争，收敛到最快臂（单臂退化，BE.25 实证）。MEG 以滑动窗口
    强制每臂至少周期性获得一次评估机会，且完全确定性（按 arms 序 + 最近出现次序），可复现。
    """
    recent = history[-window:] if history else []
    recent_counts = {}
    last_seen = {}
    for i, r in enumerate(recent):
        recent_counts[r.get("arm")] = recent_counts.get(r.get("arm"), 0) + 1
        last_seen[r.get("arm")] = i
    starved = [a for a in arms if recent_counts.get(a, 0) < min_visits]
    if not starved:
        return None
    # 最久未被选中者优先（未在窗口内出现 = last_seen 缺失 → 记 -1 最久）
    return min(starved, key=lambda a: last_seen.get(a, -1))


def candidate_archetype(candidate):
    """campaign 层候选 → archetype（第 2 层 diversity budget 记账）。"""
    from .archetypes import mechanism_archetype
    return mechanism_archetype((candidate.get("hypothesis") or {}).get("mechanism"))


def _proposal_archetypes(proposals):
    """把一轮 proposal 列表归并为 archetype 计数分布（确定性）。"""
    counts = {}
    for p in proposals:
        a = candidate_archetype(p)
        counts[a] = counts.get(a, 0) + 1
    return counts


def _arm_archetype_hhi(history, arm):
    """某臂历史产出的 archetype 集中度（Herfindahl）。1.0=全挤一个 archetype，
    →1/k=均匀铺开；无历史产出返回 None。确定性（只读 history 的 archetype_distribution）。
    """
    counts = {}
    for r in history:
        if r.get("arm") != arm:
            continue
        dist = r.get("archetype_distribution") or {}
        for a, c in dist.items():
            counts[a] = counts.get(a, 0) + c
    total = sum(counts.values())
    if total <= 0:
        return None
    return sum((c / total) ** 2 for c in counts.values())


def _covered_archetypes(history):
    """历史各轮已覆盖（产出过）的 archetype 集合（确定性，用于冷启动/多样性诊断）。"""
    covered = set()
    for r in history:
        for a, c in (r.get("archetype_distribution") or {}).items():
            if c:
                covered.add(a)
    return covered


def choose_arm(round_index, mode, history, llm_enabled, min_exploration="enforce", gen_expand="legacy"):
    # P-20261017: `random`（shuffle 模板）降级为 fixed 的 shuffle_order flag，从臂列表移除；
    # `random_ast` 升入臂列表作真·随机探索臂（完整合法空间 + 零假设校准），提前到第 2 轮，
    # 修「冷启动阶段产出几乎全在模板空间」的多样性天花板。llm 保持末尾（外部依赖 + 预算）。
    arms = ["fixed", "random_ast", "refine", "combine"] + (["regime_combine"] if gen_expand == "expanded" else []) + (["llm"] if llm_enabled else [])
    if mode == "fixed":
        return "fixed", {"rule": "fixed_baseline", "scores": {}}
    if mode == "no_memory":
        return "random_ast", {"rule": "seeded_random_null_no_memory", "scores": {}}
    if round_index < len(arms):
        # Explore each admitted emitter before using any empirical preference.
        # P-20261017-004: 冷启动按 arm 轮扫（random_ast 已升到第 2 轮），并记录已覆盖
        # archetype 供后续 diversity budget 消费——非模板臂（random_ast）提前入场即
        # 覆盖模板空间外的 archetype，实现「每 archetype 至少 1 轮」的冷启动多样性。
        return arms[round_index], {"rule": "initial_service_exploration", "scores": {},
                                   "covered_archetypes": sorted(_covered_archetypes(history))}
    scores = {}
    economic = {}
    diversity = {}
    for arm in arms:
        runs = [r for r in history if r["arm"] == arm]
        valid = sum(r["valid_e1_count"] for r in runs)
        tries = sum(r["evaluated_count"] for r in runs)
        # Capability efficiency, never confused with alpha reward.
        # P-20261015-003 (2026-10-15): cost was sum(elapsed_seconds) -- wall-clock
        # dependent, so two runs of the same seed produced different UCB scores and
        # could fork the arm sequence (learning trajectory). Replaced with the
        # deterministic evaluated_count so serial vs M1-parallel runs replay the
        # identical scheduler path (SPEEDUP_PROPOSAL.md section 4.0 determinism
        # constraint). evaluated_count == candidates_per_round * tries, so ordering
        # semantics are preserved (valid per evaluated candidate).
        cost = sum(r["evaluated_count"] for r in runs)
        # P-20261017: random_ast 是零假设（无 alpha 先验），其 valid 读数率天然接近 1，
        # 会让 UCB 能力效率 valid/cost 长期霸榜、挤占模板/变异的先验利用。cost 乘 2 反映
        # 「零假设因子探索价值打折」，平衡完整空间探索与有先验利用。零假设校准读数仍走
        # _vehicle_pool 旁路（不参与 economic 评分），此处只调 UCB 的能力效率项。
        if arm == "random_ast":
            cost *= 2.0
        base = valid/max(1.,cost) + math.sqrt(2*math.log(round_index+1)/max(1,tries))
        # P-20261017-004 diversity budget：对「重复产出同一 archetype」的臂降权。
        # HHI=1（全挤一个 archetype）→ 因子 0.5；HHI→1/k（均匀铺开）→ 因子→1。
        # random_ast 豁免：它是类型化随机多样性注入源（typed_random_null 无单一语义），
        # 且已 cost×2，不应再受集中度惩罚。确定性（只读 history 的 archetype_distribution）。
        if arm != "random_ast":
            hhi = _arm_archetype_hhi(history, arm)
            if hhi is not None:
                factor = 1.0 - 0.5 * hhi
                base *= factor
                diversity[arm] = {"hhi": round(hhi, 4), "factor": round(factor, 4)}
        scores[arm] = base
        identified = sum(r.get("economic_reward_count",0) for r in runs)
        economic[arm] = sum(r.get("economic_reward_sum",0) for r in runs)/identified if identified else None
    # Keep explicit capability exploration; missing account evidence has no numeric economic score.
    admitted = [arm for arm in arms if economic[arm] is not None]
    if admitted and round_index % 3 != 0:
        selected = max(admitted,key=lambda a:(economic[a],scores[a],-arms.index(a)))
        return selected,{"rule":"identified_economic_feedback_with_reserved_exploration",
                         "scores":scores,"economic_means":economic,
                         "diversity":diversity if diversity else None}
    # capability_cost_UCB 分支：无经济反馈（或 %3 保留探索轮）时，套最小探索保证防单臂退化。
    candidate = max(arms, key=lambda a: (scores[a], -arms.index(a)))
    acquisition = {"rule": "capability_cost_UCB", "scores": scores}
    if diversity:
        acquisition["diversity"] = diversity
    meg = _minimum_exploration_arm(arms, history, window=len(arms), min_visits=1)
    if meg is not None:
        if min_exploration == "enforce":
            return meg, dict(acquisition, rule="capability_cost_UCB_min_exploration",
                             min_exploration_arm=meg, ucb_candidate=candidate)
        if min_exploration == "shadow":
            acquisition.update(rule="capability_cost_UCB_min_exploration_shadow",
                               min_exploration_arm=meg, ucb_candidate=candidate)
    return candidate, acquisition


def replay_feedback(store, campaign_id, history):
    """Rebuild the scheduler's sufficient statistics from effective revisions, not stale totals."""
    totals = {}
    effective = store.effective_rewards("capability")
    dependencies = []
    for reward in effective:
        fact = store.get(reward["event_id"])
        decision = fact["slot"][0]
        if not decision.startswith(campaign_id+":decision:"):
            continue
        dependencies.append(reward["event_id"])
        if reward["value"] is not None:
            totals[decision] = totals.get(decision,0)+reward["value"]
    econ = {}
    for reward in store.effective_rewards("economic"):
        decision = store.get(reward["event_id"])["slot"][0]
        if decision.startswith(campaign_id+":decision:"):
            dependencies.append(reward["event_id"])
            if reward["value"] is not None:
                bucket = econ.setdefault(decision,[])
                bucket.append(reward["value"])
    rebuilt = []
    for row in history:
        decision = campaign_id+":decision:"+str(row["round"])
        values = econ.get(decision,[])
        rebuilt.append(dict(row,valid_e1_count=totals.get(decision,0),
                            economic_reward_count=len(values),economic_reward_sum=sum(values)))
    identity = "scheduler_replay_"+digest([campaign_id,dependencies])
    if not store.db.execute("SELECT 1 FROM consumers WHERE id=?",(identity,)).fetchone():
        store.register_consumer(identity,dependencies,"full_effective_reward_replay")
        store.append("state_rebuilt",{"state_id":identity,"dependencies":dependencies,"method":"effective_reward_replay"})
    require(store.state_usable(identity),"scheduler state invalidated before use")
    return rebuilt,identity


def _yearly_profiles(store):
    """Aggregate stored evaluation facts into per-year signed mean rank-IC profiles.

    Reads the full `evaluation` facts (not the candidate_result summaries, which do
    not carry stage diagnostics). Prefers the history_research_v2 yearly reading when
    present; otherwise aggregates E1 per-session sections by calendar year.
    Candidates without evaluable yearly readings are omitted (regime pairing skips
    them explicitly)."""
    profiles = {}
    for _, payload in store.facts("evaluation"):
        cid = payload.get("candidate_id")
        evals = payload.get("evaluations")
        if not cid or not isinstance(evals, list):
            continue
        e1 = next((e for e in evals if isinstance(e, dict) and e.get("stage") == "E1"), None)
        diag = (e1 or {}).get("diagnostics", {})
        yearly = diag.get("yearly")
        if isinstance(yearly, dict):
            profile = {y: v.get("mean_rank_ic") for y, v in yearly.items()
                       if isinstance(v, dict) and v.get("mean_rank_ic") is not None}
            if profile:
                profiles[cid] = profile
                continue
        by_year = {}
        for section in diag.get("sections", []):
            session = section.get("session")
            ic = section.get("rank_ic")
            if isinstance(session, str) and len(session) >= 4 and ic is not None:
                by_year.setdefault(session[:4], []).append(ic)
        profile = {y: sum(values)/len(values) for y, values in by_year.items() if values}
        if profile:
            profiles[cid] = profile
    return profiles


def record_candidate_feedback(database, decision_id, candidate, result, evaluation_id,
                              policy, baseline_ast, config, campaign_id, *, scores=None, behavior_hash=None,
                              vehicle_ic_values=None):
    """Idempotent derived state: settlement may precede a restart, reward correction may not be undone."""
    e1, e3 = result["evaluations"][1], result["evaluations"][3]
    slot = [decision_id,candidate["candidate_id"],"capability",policy["policy_hash"],digest(baseline_ast)]
    effective = next((r for r in database.effective_rewards("capability") if r["slot"] == digest(slot)), None)
    if effective is not None:
        capability_event = effective["event_id"]
        # Never revive the original reading after a later adjudication.
        if effective["revision"] > 1:
            return
    else:
        result_id = database.append("evaluation_stage", e1)
        capability_event = database.reward(slot,"capability",1. if e1["eval_status"] == "valid" else 0.,result_id)
    if e3["economic_reward_eligible"]:
        economic_id = database.append("evaluation_stage",e3)
        economic_slot = [decision_id,candidate["candidate_id"],"economic",policy["policy_hash"],digest(baseline_ast)]
        if not any(r["slot"] == digest(economic_slot) for r in database.effective_rewards("economic")):
            value = max(-1.,min(1.,e3["metrics"]["paired_mean_increment"]/.001))
            database.reward(economic_slot,"economic",value,economic_id)
    learned_state = "archive_"+digest([candidate["candidate_id"],capability_event])
    if not database.db.execute("SELECT 1 FROM consumers WHERE id=?",(learned_state,)).fetchone():
        dependencies=[capability_event,evaluation_id]
        dependencies += feedback_dependencies(candidate,[c for _,c in database.facts('feedback_v2')],
                                               config.get('memory_imports',[]))
        database.register_consumer(learned_state,dependencies,"rebuild_from_candidate_evaluation")
    if behavior_hash is None and scores is not None:
        research_window = policy["request"]
        behavior_hash = behavioral_signature(scores, research_window["research_start"], research_window["research_end"])
    database.append("archive_member",{"candidate_id":candidate["candidate_id"],"state_id":learned_state,
        "mechanism":candidate["hypothesis"]["mechanism"],"quality":e1["diagnostics"]["spearman"]["mean"],
        "complexity":candidate["node_count"],"valid_reading":e1["eval_status"] == "valid",
        "structural_family":structural_family(candidate["canonical_ast"]),"behavior_hash":behavior_hash,
        "investable":False,"scope":"research_descriptive"})
    database.append("experience",{"candidate_id":candidate["candidate_id"],"evaluation_id":evaluation_id,
        "observation":"valid descriptive factor reading" if e1["eval_status"] == "valid" else "insufficient factor observations",
        "counterexample":e3["reason_code"],"level":"observed","scope":config["data_scope"],
        "economic_claim":e3["economic_reward_eligible"],"cohort":campaign_id,"state_id":learned_state})
    database.append("feedback_v2", feedback_card(candidate,result,policy,config['data_scope'],evaluation_id,learned_state,
                                                 vehicle_ic_values=vehicle_ic_values))


def execute_portfolio_action(database,campaign_id,payload,policy,baseline_scores,top_n,directory):
    experiment = digest([campaign_id,"portfolio_axis",policy["policy_hash"],digest(baseline_scores),top_n])
    cached = database.cached(experiment)
    if cached:
        database.get(cached["evidence_id"])
        return cached
    from .full_market_support import execution_options
    options = execution_options(payload)
    timeout = options["portfolio_timeout"] if options else 120
    memory_mb = options["worker_memory_mb"] if options else 2048
    attempt = database.reserve(campaign_id,experiment,timeout)
    job = {"job_kind":"portfolio","input_path":str(directory/"input.json"),"input_hash":digest(payload),
           "policy":policy,"baseline_scores":baseline_scores,"top_n":top_n}
    output,process = worker(job,directory/"attempts"/attempt,timeout=timeout,memory_mb=memory_mb)
    if output is None:
        summary = {"status":"error","action":"change_rebalance_only","process":process}
        database.finish(attempt,"failed",process["elapsed_seconds"],summary)
        return summary
    artifact = output["portfolio_experiment"]
    evidence_id = database.append("portfolio_axis_experiment",artifact)
    summary = {"status":"valid","action":"change_rebalance_only","evidence_id":evidence_id,
               "parent_ir_hash":artifact["parent_ir"]["ir_hash"],"child_ir_hash":artifact["child_ir"]["ir_hash"],
               "comparison_status":artifact["comparison"]["eval_status"],
               "decision":artifact["comparison"]["decision"],"elapsed_seconds":process["elapsed_seconds"]}
    database.finish(attempt,"completed",process["elapsed_seconds"],summary)
    return summary


def execute_model_action(database, campaign_id, algorithm, features, snapshot, labels, payload, policy,
                         baseline_scores, top_n, seed, directory):
    request = policy["request"]
    experiment = digest([campaign_id,"model",algorithm,sorted(features),digest(features),policy["policy_hash"]])
    cached = database.cached(experiment)
    if cached:
        path = directory/cached["artifact_relative_path"]
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == cached["artifact_sha256"],
                "cached model artifact missing or changed")
        return cached
    started = time.monotonic()
    from .full_market_support import execution_options
    options = execution_options(payload)
    timeout = options["model_timeout"] if options else 60
    memory_mb = options["worker_memory_mb"] if options else 2048
    attempt = database.reserve(campaign_id,experiment,timeout)
    try:
        job={"job_kind":"model","input_path":str(directory/"input.json"),"input_hash":digest(payload),
             "features":features,"baseline_scores":baseline_scores,"policy":policy,"top_n":top_n,"algorithm":algorithm,"seed":seed}
        output,process=worker(job,directory/"attempts"/attempt,timeout=timeout,memory_mb=memory_mb)
        if output is None:
            raise RuntimeError(json.dumps(process,ensure_ascii=False))
        artifact=output["artifact"]
        model_id=database.append("model_bundle",output["model_bundle"])
        relative="models/"+algorithm+"_"+attempt+".json"
        if options is not None:
            from .full_market_support import pack_grids
            artifact_storage = pack_grids(artifact, directory/"models"/(attempt+"_panels"))
        else:
            artifact_storage = artifact
        write_new(directory/relative,artifact_storage)
        summary={**output["model_summary"],"bundle_id":model_id,"elapsed_seconds":time.monotonic()-started,"artifact_relative_path":relative,
                 "artifact_sha256":hashlib.sha256((directory/relative).read_bytes()).hexdigest()}
        database.finish(attempt,"completed",summary["elapsed_seconds"],summary)
        return summary
    except Exception as exc:
        summary={"status":"error","algorithm":algorithm,"error_type":type(exc).__name__,"reason":str(exc),
                 "elapsed_seconds":time.monotonic()-started}
        database.finish(attempt,"failed",summary["elapsed_seconds"],summary)
        return summary


def _memory_source_contract(sources):
    """Bounded explicit library input; frozen descriptors prevent source growth races."""
    require(isinstance(sources, (tuple, list)) and len(sources) <= 8, "memory source bound invalid")
    result, seen = [], set()
    for source in sources:
        expected = None
        if isinstance(source, dict):
            require(set(source) == {"path", "manifest_hash"}, "memory source descriptor fields differ")
            expected = source["manifest_hash"]
            require(isinstance(expected, str) and len(expected) == 64
                    and all(c in "0123456789abcdef" for c in expected), "memory source manifest hash invalid")
            source = source["path"]
        require(isinstance(source, (str, Path)) and bool(str(source).strip()), "memory source path invalid")
        path = Path(source).resolve()
        require(path.is_file() and path.suffix.lower() == ".sqlite", "native memory source file missing")
        require(path not in seen, "duplicate native memory source")
        seen.add(path)
        result.append({"path": str(path), "manifest_hash": expected})
    return result


def run_campaign(input_path, output_root, *, rounds=5, candidates_per_round=3, mode="adaptive", seed=17,
                 llm_enabled=False, model_algorithms=("ridge", "lightgbm", "xgboost"), campaign_id=None,
                 authority_path=None, study_protocol=None, pause_after_round=None, data_receipt_id=None, scheduler_policy="legacy",
                 memory_sources=(), external_hypotheses=(), minimum_detectable_ic=None):
    require(mode in {"adaptive", "fixed", "no_memory"} and 1 <= rounds <= 200 and 1 <= candidates_per_round <= 10,
            "campaign bounds invalid")
    require(minimum_detectable_ic is None or 0 < minimum_detectable_ic < 1,
            "minimum_detectable_ic must be in (0,1) when declared")
    require(scheduler_policy in {"legacy", "selection_v2"}, "unknown scheduler policy")
    requested_memory = _memory_source_contract(memory_sources)
    require(mode != "no_memory" or not requested_memory, "no-memory arm cannot inherit memory")
    require(not requested_memory or not (study_protocol or {}).get("memory_sources"),
            "independent and scheduled memory sources cannot both be supplied")
    if study_protocol is not None:
        require(set(study_protocol) == {'schema_version','arm_schedule','max_trials','checkpoint_every','memory_sources'},
                'study protocol fields differ')
        require(study_protocol['schema_version']=='fuxi.controlled_study.1', 'unknown study protocol')
        schedule = study_protocol['arm_schedule']
        require(len(schedule)==rounds and all(a in {'fixed','random','refine','combine','llm','random_ast','refine_local','llm_patch'} for a in schedule),
                'frozen emitter schedule required')
        require(study_protocol['max_trials'] >= rounds*candidates_per_round and
                type(study_protocol['checkpoint_every']) is int and 1 <= study_protocol['checkpoint_every'] <= rounds,
                'study trial budget/checkpoint invalid')
        require(not {'llm','llm_patch'} & set(schedule) or llm_enabled, 'scheduled LLM requires enabled budget')
        require(mode != 'no_memory' or not study_protocol['memory_sources'], 'no-memory arm cannot inherit memory')
    require(pause_after_round is None or 1 <= pause_after_round <= rounds, 'invalid checkpoint pause')
    parent_policy = os.environ.get("FUXI_V1_PARENT_POLICY", "pareto")
    require(parent_policy in {"pareto", "niche_mix_shadow", "niche_mix"}, "unknown parent policy")
    motif_mode = os.environ.get("FUXI_V1_MOTIF_MEMORY", "off")
    require(motif_mode in {"off", "shadow", "enforce"}, "unknown motif memory mode")
    fitness_mode = os.environ.get("FUXI_V1_FITNESS_TRIAGE", "enforce")
    require(fitness_mode in {"off", "shadow", "enforce"}, "unknown fitness triage mode")
    min_exploration = os.environ.get("FUXI_V1_MIN_EXPLORATION", "enforce")
    require(min_exploration in {"off", "shadow", "enforce"}, "unknown min exploration mode")
    gen_expand = os.environ.get("FUXI_V1_GEN_EXPAND", "legacy")
    require(gen_expand in {"legacy", "expanded"}, "unknown generator expansion mode")
    # P5 (v2.0 landing): shadow wiring and score-ref cache key default on since
    # v0.7.186 (four-gate verified); explicit off restores the historical path.
    p5_shadow = os.environ.get("FUXI_V1_P5_SHADOW", "on")
    require(p5_shadow in {"off", "on"}, "unknown p5 shadow mode")
    score_ref_key = os.environ.get("FUXI_V1_SCORE_REF_KEY", "on")
    require(score_ref_key in {"off", "on"}, "unknown score ref key mode")
    payload = load_json(input_path)
    require(payload["schema_version"] == "fuxi.campaign_input.1", "campaign input schema required")
    require(payload["snapshot"]["exposure"] != "forward_sealed", "protected data cannot enter research campaign")
    snapshot, labels = Snapshot(payload["snapshot"]), None
    labels = Labels(payload["labels"], snapshot)
    validate_label_calendar(labels.payload, snapshot.sessions)
    require(digest({k:v for k,v in payload["market"].items() if k != "market_hash"}) == payload["market"]["market_hash"],
            "market manifest mismatch")
    from .full_market_support import hydrate_payload, execution_options
    full_options = execution_options(payload)
    payload = hydrate_payload(payload)
    require(len(snapshot.sessions) >= 120, "campaign requires at least 120 sessions for train/calibration/research workflow")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    campaign_id = campaign_id or "campaign_"+uuid.uuid4().hex
    require(campaign_id.startswith("campaign_") and campaign_id.replace("_", "").isalnum(), "invalid campaign identity")
    authority_path = Path(authority_path or root/"authority.sqlite").resolve()
    authority = EvidenceStore(authority_path)
    try:
        with authority.transaction():
            authority.expose(campaign_id, scope_of(snapshot), "research")
            authority.append("campaign_scope", {"campaign_id": campaign_id, "input_hash": digest(payload),
                             "scope": scope_of(snapshot), "output_root": str(root)})
    finally:
        authority.close()
    directory = root/campaign_id
    directory.mkdir(exist_ok=True)
    database = EvidenceStore(directory/"evidence.sqlite")
    pool = None
    try:
        existing = database.db.execute("SELECT config FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        registered_at = json.loads(existing[0])["registered_at"] if existing else now()
        code = code_manifest()
        horizon = labels.payload["label_spec"]["horizon_sessions"]
        cal_index = max(snapshot.start+25, min(80, len(snapshot.sessions)//2))
        research_index = cal_index+horizon+2
        require(research_index+20 < len(snapshot.sessions), "research window too short")
        # P-20261015-011: a confirmatory-style formal admission may declare its
        # minimum detectable IC explicitly (power planning against the physical
        # section count of the research window). None keeps the shared default.
        request = calibration_request(snapshot, calibration_end=snapshot.sessions[cal_index],
                                      research_start=snapshot.sessions[research_index],
                                      max_trials=study_protocol['max_trials'] if study_protocol else rounds*candidates_per_round,
                                      horizon=horizon,
                                      **({'minimum_detectable_ic': minimum_detectable_ic} if minimum_detectable_ic is not None else {}))
        learn_key = compatibility(payload,request)
        memory_paths = [r["path"] for r in requested_memory]
        if existing:
            saved_config = json.loads(existing[0])
            require(saved_config.get("memory_sources", []) == memory_paths, "memory sources changed across resume")
            imports = saved_config.get("memory_imports", [])
            memory_bindings = saved_config.get("memory_source_bindings", [])
            require(not requested_memory or len(memory_bindings) == len(requested_memory),
                    "frozen independent memory binding missing")
            for requested, saved in zip(requested_memory, memory_bindings):
                require(requested["manifest_hash"] is None or requested["manifest_hash"] == saved["manifest_hash"],
                        "memory source manifest changed across resume")
            require(not requested_memory or [digest(m) for m in imports] == [r["manifest_hash"] for r in memory_bindings],
                    "frozen memory manifest corrupted")
        else:
            imports = [export_memory(p, learn_key) for p in
                       (memory_paths or (study_protocol or {}).get('memory_sources', []))]
            memory_bindings = []
            for requested, manifest in zip(requested_memory, imports):
                actual = digest(manifest)
                require(requested["manifest_hash"] is None or requested["manifest_hash"] == actual,
                        "memory source manifest drift before native freeze")
                memory_bindings.append({"path": requested["path"], "manifest_hash": actual,
                                        "compatibility_hash": learn_key})
        config = {"schema_version": "fuxi.campaign.1", "registered_at": registered_at, "campaign_id": campaign_id,
                  "input_hash": digest(payload), "code_hashes": code, "environment": environment(),
                  "rounds": rounds, "candidates_per_round": candidates_per_round, "mode": mode, "seed": seed,
                  "llm_enabled": llm_enabled,
                  "llm_max_calls": sum(a in {'llm','llm_patch'} for a in study_protocol['arm_schedule']) if study_protocol else (max(1, rounds // 5) if llm_enabled else 1),
                  "llm_output_token_cap_per_call": 1800,
                  "model_algorithms": list(model_algorithms), "budget_units": rounds*candidates_per_round*300+420,
                  "budget_unit": "reserved_worker_wall_seconds", "worker_memory_mb": 2048, "worker_timeout": 300,
                  "data_scope": scope_of(snapshot), "history_exposure_inherited": snapshot.payload["exposure"]}
        if full_options is not None:
            config.update(full_market_execution=full_options,
                          budget_units=full_options["budget_units"],
                          worker_memory_mb=full_options["worker_memory_mb"],
                          worker_timeout=full_options["worker_timeout"])
            require(config["budget_units"] >= rounds*candidates_per_round*config["worker_timeout"]
                    + (len(model_algorithms)+2 if model_algorithms else 0)*full_options["model_timeout"]
                    + full_options["portfolio_timeout"], "full-market reservation budget insufficient")
        config["authority_path"] = str(authority_path)
        config.update(study_protocol=study_protocol, learning_compatibility=learn_key,
                      memory_imports=imports, data_receipt_id=data_receipt_id)
        if requested_memory:
            config.update(memory_sources=memory_paths, memory_source_bindings=memory_bindings)
        if scheduler_policy != "legacy":
            config["scheduler_policy"] = scheduler_policy
        config["parent_policy"] = parent_policy
        config["motif_memory"] = motif_mode
        config["fitness_triage"] = fitness_mode
        config["min_exploration"] = min_exploration
        config["gen_expand"] = gen_expand
        config["p5_shadow"] = p5_shadow
        config["score_ref_key"] = score_ref_key
        if existing:
            existing_config = json.loads(existing[0])
            require(existing_config.get("scheduler_policy", "legacy") == scheduler_policy,
                    "scheduler policy changed across resume")
            require(existing_config.get("parent_policy", "pareto") == parent_policy,
                    "parent policy changed across resume; evidence needs a new code identity")
            require(existing_config.get("motif_memory", "off") == motif_mode,
                    "motif memory mode changed across resume; evidence needs a new code identity")
            require(existing_config.get("fitness_triage", "off") == fitness_mode,
                    "fitness triage mode changed across resume; evidence needs a new code identity")
            require(existing_config.get("min_exploration", "off") == min_exploration,
                    "min exploration mode changed across resume; evidence needs a new code identity")
            require(existing_config.get("gen_expand", "legacy") == gen_expand,
                    "generator expansion mode changed across resume; evidence needs a new code identity")
            # Fallback "off" is the historical implicit default of pre-P5 evidence dirs;
            # resuming them under the new default (on) is therefore correctly rejected.
            require(existing_config.get("p5_shadow", "off") == p5_shadow,
                    "p5 shadow mode changed across resume; evidence needs a new code identity")
            require(existing_config.get("score_ref_key", "off") == score_ref_key,
                    "score ref key mode changed across resume; evidence needs a new code identity")
        database.campaign(campaign_id, config)
        recovered = database.recover(campaign_id)
        if not (directory/"input.json").exists():
            write_new(directory/"input.json", payload)
        else:
            require(digest(load_json(directory/"input.json")) == config["input_hash"], "frozen campaign input changed")
        for name, expected_hash in code.items():
            destination = directory/"code"/name
            if not destination.exists():
                write_new(destination, (Path(__file__).parent/name).read_bytes())
            require(hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash, "frozen code snapshot mismatch")
        database.expose(campaign_id, config["data_scope"], "research")
        baseline_ast = node("rank", node("return", {"field": "close"}, window=20))
        baseline_scores = evaluate_ast(baseline_ast, snapshot)
        policy = register_calibration(database, campaign_id, request, baseline_scores, snapshot, labels, payload["research_clock"])
        reachability = audit_reachability(payload,policy,data_receipt_id is not None)
        database.append('reachability_audit',reachability)
        if not (directory/"config.json").exists():
            write_new(directory/"config.json", config)
            write_new(directory/"policy.json", policy)
        history = [p for _,p in database.facts("round_completed")]
        archive = [p for _,p in database.facts("archive_member")]
        seen = {p["formula_hash"] for _,p in database.facts("candidate")}
        candidates_all = {p["candidate_id"]:p for _,p in database.facts("candidate")}
        results = [p for _,p in database.facts("candidate_result")]
        # P-20261018-014: the qualified pool (E0-E4 all-pass factors) feeds the E2
        # pool-redundancy gate. Rebuild on resume from settled candidate_results in
        # settle order; empty on a fresh run. Members never retroactively re-evaluated.
        qualified_pool = [{"candidate_id": r["candidate_id"],
                           "canonical_ast": candidates_all[r["candidate_id"]]["canonical_ast"]}
                          for r in results
                          if r.get("jq_candidate_allowed") and r.get("candidate_id") in candidates_all]
        # P5 (v2.0 landing): shadow ledger for the P1-P4 contract hooks. Off by
        # default; on writes only landing_shadow facts and never touches the
        # production decision path. Resume replays prior shadow state.
        from .landing_shadow import ShadowState, _safe, hook_arm_selection, \
            hook_dedup_cooldown, hook_export, hook_g0_precheck, hook_g1_complement
        shadow_state = ShadowState(p5_shadow == "on")
        if shadow_state.enabled:
            shadow_state.replay(database)
            shadow_contract_ref = digest({"policy": policy["policy_hash"],
                                          "code": code, "env": config["environment"]})
        llm_calls = sum(p.get("attempted_calls",0) for _,p in database.facts("llm_call"))
        llm_reserved = len(database.facts("llm_reservation"))
        top_n = min(full_options["top_n"] if full_options else 6,
                    len(snapshot.universe[snapshot.sessions[snapshot.start]]))
        checkpoints = [p for _,p in database.facts('learning_checkpoint')]
        # M2 (P-20261015-007): persistent loop-mode worker pool. Off by default;
        # workers spawn lazily on first submission and recycle every
        # FUXI_V1_POOL_MAX_REQUESTS jobs (SPEEDUP 4b).
        pool = None
        if os.environ.get("FUXI_V1_PERSISTENT_POOL") == "1":
            pool = _PersistentPool(
                size=int(os.environ.get("FUXI_V1_POOL_SIZE", "2")),
                root=directory/"attempts"/".pool",
                timeout=config["worker_timeout"], memory_mb=config["worker_memory_mb"],
                max_requests=int(os.environ.get("FUXI_V1_POOL_MAX_REQUESTS", "8")),
                cwd=Path(__file__).resolve().parents[1])
        for round_index in range(rounds):
            if any(r["round"] == round_index for r in history):
                continue
            if (pause_after_round is not None and round_index >= pause_after_round) or any(p['stop'] for p in checkpoints):
                break
            round_started = time.monotonic()
            imported = refresh_import_dependencies(database,imports) if mode != 'no_memory' else []
            feedback_history,scheduler_state=replay_feedback(database,campaign_id,history)
            generation_context = digest([sorted(seen), sorted(p["candidate_id"] for p in archive), mode])
            if scheduler_policy == "selection_v2" and not study_protocol:
                from .scheduler import choose
                arm, acquisition = choose(round_index, mode, feedback_history,
                    llm_enabled and llm_reserved < config["llm_max_calls"], context_hash=generation_context)
                if arm is None:
                    database.append("scheduler_stop", dict(acquisition, round=round_index), campaign_id+":scheduler_stop")
                    break
            else:
                arm, acquisition = choose_arm(round_index, mode, feedback_history, llm_enabled and llm_reserved < config["llm_max_calls"], min_exploration, gen_expand)
            if study_protocol:
                arm = study_protocol['arm_schedule'][round_index]
                acquisition = {'rule':'preregistered_matched_emitter_schedule'}
            acquisition["scheduler_state_id"]=scheduler_state
            # P5 shadow hook 1: five-arm eligibility + round_robin/ucb shadow reads.
            selection_policy = "niche_mix" if parent_policy == "niche_mix" else "pareto"
            parent_pool = select_parents(archive, selection_policy)
            parents = [candidates_all[p["candidate_id"]] for p in parent_pool if p["candidate_id"] in candidates_all
                       and (not p.get("state_id") or database.state_usable(p["state_id"]))]
            local_cards = [c for _,c in database.facts('feedback_v2') if database.state_usable(c['state_id'])]
            if mode != 'no_memory':
                # Fitness triage enforce switches imported-memory ordering from raw IC to
                # economic distance; shadow keeps the historical ordering and records only.
                if fitness_mode == "enforce":
                    imported_ordered = sorted(imported, key=fitness_key)
                else:
                    imported_ordered = sorted(imported,key=lambda c:-(c['mean_rank_ic'] or 0))
                imported_candidates = [c['candidate'] for c in imported_ordered]
                parents = list({p['candidate_id']:p for p in parents+imported_candidates}.values())[:8]
            if mode == "no_memory":
                # Same generator services, but anchors are chosen without performance/feedback.
                parents = generate(snapshot.payload['fields'],arm='random',count=4,seed=seed+round_index) if study_protocol else []
            # P5 shadow hook 1: five-arm eligibility + round_robin/ucb shadow reads,
            # recorded after the arm AND the final parent set are both settled.
            if shadow_state.enabled:
                _safe(shadow_state, database, campaign_id, "arm_selection", round_index,
                      lambda a=arm, p=parents, h=feedback_history: hook_arm_selection(
                          shadow_state, database, campaign_id, round_index, a, h,
                          budget_remaining=(rounds - round_index) * candidates_per_round,
                          llm_authorized=bool(llm_enabled and llm_reserved < config["llm_max_calls"]),
                          valid_parents=bool(p),
                          computable_parents=[x["candidate_id"] for x in p],
                          template_slots_remaining=True))
            if parent_policy == "niche_mix_shadow":
                # Shadow period: actual selection stays pareto; record whom niche-mix would have chosen.
                shadow_id = campaign_id+":parent_shadow:"+str(round_index)
                if not database.db.execute("SELECT 1 FROM facts WHERE id=?", (shadow_id,)).fetchone():
                    frontier_ids = {p["candidate_id"] for p in pareto_archive(archive)}
                    niche_pool = select_parents(archive, "niche_mix")
                    database.append("parent_policy_shadow", {
                        "round": round_index, "declared_policy": parent_policy, "actual_policy": "pareto",
                        "pareto_frontier_count": len(frontier_ids),
                        "niche_mix_pool_count": len(niche_pool),
                        "niche_mix_adds_over_pareto": sorted({p["candidate_id"] for p in niche_pool} - frontier_ids),
                        "actual_parent_ids": [p["candidate_id"] for p in parents]}, shadow_id)
            decision_id = campaign_id+":decision:"+str(round_index)
            prior_decisions = database.db.execute("SELECT payload FROM facts WHERE id=?", (decision_id,)).fetchone()
            if prior_decisions:
                decision = json.loads(prior_decisions[0])
                proposals = [database.get(cid) for cid in decision["candidate_event_ids"]]
                arm = decision["arm"]
            else:
                generation_counts = {}
                generator_status = "completed"
                if arm in {"llm", "llm_patch"}:
                    database.append("llm_reservation", {"decision_id": decision_id, "max_output_tokens": 1800,
                                    "max_calls": 1, "unknown_result_policy": "no_automatic_resubmit"}, decision_id+":llm_reservation")
                    llm_reserved += 1
                    require(llm_reserved <= config['llm_max_calls'], 'LLM reservation cap exceeded')
                    cards = prompt_cards(local_cards+imported) if mode != 'no_memory' else []
                    database.append('prompt_feedback',{'decision_id':decision_id,'cards':cards})
                    if arm == "llm_patch":
                        from .edits import llm_patches
                        if parents:
                            proposals, receipt = llm_patches(snapshot.payload["fields"], parents[:8], cards, count=candidates_per_round)
                        else:
                            proposals, receipt = [], {"status":"temporarily_unavailable", "reason":"parent_set_empty", "attempted_calls":0}
                    else:
                        proposals, receipt = llm_candidates(snapshot.payload["fields"], cards, count=candidates_per_round,
                                                            external_hypotheses=external_hypotheses)
                    generator_status = receipt["status"]
                    generation_counts["generation_attempts"] = receipt.get("raw_candidate_count", receipt.get("attempted_calls", 0))
                    database.append("llm_call", receipt)
                    llm_calls += receipt.get("attempted_calls",0)
                    local_seen = set(seen)
                    unique = []
                    for proposal in proposals:
                        if proposal["formula_hash"] not in local_seen:
                            unique.append(proposal)
                            local_seen.add(proposal["formula_hash"])
                    proposals = unique
                    if not proposals:
                        acquisition["fallback_reason"] = receipt.get("reason", "no_supported_novel_LLM_AST")
                        fallback_counts = {}
                        proposals = generate(snapshot.payload["fields"], arm="random", count=candidates_per_round,
                                             seed=seed+round_index, seen=seen, counters=fallback_counts)
                        generation_counts["generation_attempts"] += fallback_counts["generation_attempts"]
                        acquisition["actual_fallback_generator"] = "random"
                elif arm == "regime_combine":
                    profiles = _yearly_profiles(database)
                    pair_proposals = generate(snapshot.payload["fields"], arm=arm, count=candidates_per_round,
                                              seed=seed+round_index, parents=parents, seen=seen,
                                              counters=generation_counts, regime_profiles=profiles)
                    proposals = [p for p in pair_proposals
                                 if p["hypothesis"]["mechanism"] == "cross_year_regime_complement"]
                    acquisition["regime_pair_count"] = len(proposals)
                    acquisition["profiled_parent_count"] = len(profiles)
                    if not proposals:
                        acquisition["fallback_reason"] = "no_novel_complementary_regime_pairs"
                        fallback_counts = {}
                        proposals = generate(snapshot.payload["fields"], arm="random", count=candidates_per_round,
                                             seed=seed+round_index, seen=seen, counters=fallback_counts)
                        generation_counts["generation_attempts"] += fallback_counts["generation_attempts"]
                        acquisition["actual_fallback_generator"] = "random"
                elif arm == "refine_local":
                    from .edits import refine_local
                    proposals, local_receipt = refine_local(snapshot.payload["fields"], parents, candidates_per_round, seen)
                    generation_counts["generation_attempts"] = local_receipt["attempts"]
                    generation_counts["parameter_attempts"] = local_receipt["attempts"]
                    acquisition["local_receipt"] = local_receipt
                else:
                    proposals = generate(snapshot.payload["fields"], arm=arm, count=candidates_per_round,
                                         seed=seed+round_index, parents=parents, seen=seen, counters=generation_counts,
                                         shuffle_source=(arm == "fixed"))
                if motif_mode in ("shadow", "enforce") and proposals and arm in ("refine", "combine"):
                    # Edit-level memory: shadow counts veto hits, enforce withholds matching proposals.
                    known_asts = {cid: (c or {}).get("canonical_ast") for cid, c in candidates_all.items()}
                    vetoes = motif_vetoes(motif_ledger(motif_observations(local_cards, known_asts.get)))
                    gate = motif_gate(proposals, known_asts.get, vetoes, mode=motif_mode)
                    gate_id = campaign_id+":motif_gate:"+str(round_index)
                    if not database.db.execute("SELECT 1 FROM facts WHERE id=?", (gate_id,)).fetchone():
                        database.append("motif_gate", {"round": round_index, "mode": motif_mode, "arm": arm,
                            "veto_count": len(vetoes), "vetoes": vetoes, "checked": gate["checked"],
                            "hit_count": gate["hit_count"], "hits": gate["hits"]}, gate_id)
                    if motif_mode == "enforce":
                        proposals = gate["kept"]
                if full_options is not None:
                    from .full_market_support import complete_generation
                    rule=payload.get("generation_completeness_rule")
                    require(rule == {"schema_version":"fuxi.generation_completeness.1",
                        "rule":"seeded_random_ast_fill_before_any_round_outcome",
                        "supplement_seed_offset":100000,"requested_per_round":candidates_per_round},
                        "all-market generation repair must be preregistered")
                    proposals, repair = complete_generation(proposals, snapshot.payload["fields"],
                        candidates_per_round, seen, seed+rule["supplement_seed_offset"]+round_index)
                    acquisition["generation_quota_repair"] = repair
                    generation_counts["generation_attempts"] = generation_counts.get("generation_attempts",0)+repair["generation_attempts"]
                # P-20261017-004 diversity budget 记账：本轮产出的 archetype 分布写入
                # acquisition（随 research_decision 落盘）+ row（随 round_completed 落盘），
                # 供 choose_arm 的集中度降权与 reflect 的外部检索指引消费。确定性只读。
                acquisition["archetype_distribution"] = dict(_proposal_archetypes(proposals))
                # P5 shadow hooks 2+3: G0 precheck and dedup/cooldown readings before
                # registration. Production ``seen`` still rules registration.
                if shadow_state.enabled:
                    from .dsl import CATALOG as _DSL_CATALOG
                    _safe(shadow_state, database, campaign_id, "g0_precheck", round_index,
                          lambda ps=proposals: hook_g0_precheck(
                              shadow_state, database, campaign_id, round_index, ps,
                              fields=list(snapshot.payload["fields"]),
                              operators=sorted(_DSL_CATALOG["operators"]),
                              session_count=len(snapshot.sessions)))
                    _safe(shadow_state, database, campaign_id, "dedup_cooldown", round_index,
                          lambda ps=proposals: hook_dedup_cooldown(
                              shadow_state, database, campaign_id, round_index, arm, ps,
                              shadow_contract_ref))
                events = []
                for proposal in proposals:
                    events.append(database.append("candidate", proposal))
                    candidates_all[proposal["candidate_id"]] = proposal
                    seen.add(proposal["formula_hash"])
                acquisition["generation_counts"] = generation_counts
                acquisition["generator_status"] = generator_status
                decision = {"decision_id": decision_id, "round": round_index, "arm": arm,
                            "candidate_event_ids": events, "acquisition": acquisition, "parent_ids": [p["candidate_id"] for p in parents],
                            "frozen_baseline_hash": digest(baseline_ast), "policy_hash": policy["policy_hash"],
                            "memory_enabled": mode != "no_memory", "at": now()}
                decision['imported_feedback_count'] = len(imported)
                database.append("research_decision", decision, decision_id)
            round_results = []
            round_g1 = []
            round_exports = []

            def settle_candidate(candidate, attempt, experiment, output, process):
                """Write one evaluated candidate's evidence in proposals order.

                Shared by the serial and parallel paths so both produce the
                identical facts sequence (P-20261015-002)."""
                nonlocal archive
                if output is None:
                    summary = {"candidate_id": candidate["candidate_id"], "round": round_index, "arm": arm,
                               "eval_status": "error", "process": process, "valid_e1": False, "jq_candidate_allowed": False,
                               "formal_account_valid": False, "mean_rank_ic": None, "result_path": None}
                    database.finish(attempt, "failed", process["elapsed_seconds"], summary)
                    database.append("candidate_result", summary)
                    results.append(summary)
                    round_results.append(summary)
                    return
                result = output["result"]
                evaluation_id = database.append("evaluation", result)
                if data_receipt_id is None:
                    admission = AcceptanceService(database).research(evaluation_id)
                    from .full_market_support import diagnostic_jq
                    script, compiler = diagnostic_jq(result['strategy_ir'],payload['market'],admission,database)
                else:
                    # Use current authority and revocation state; never copy active receipts into a stale private ledger.
                    formal = EvidenceStore(authority_path)
                    try:
                        formal_id = formal.append('evaluation',result)
                        admission = AcceptanceService(formal).research(formal_id,data_receipt_id)
                        from .full_market_support import diagnostic_jq
                        script, compiler = diagnostic_jq(result['strategy_ir'],payload['market'],admission,formal)
                    finally:
                        formal.close()
                    database.append('authority_admission',admission)
                e1, e3 = result["evaluations"][1], result["evaluations"][3]
                mean_ic = e1["diagnostics"]["spearman"]["mean"]
                summaries = {"candidate_id": candidate["candidate_id"], "round": round_index, "arm": arm,
                            "actual_action": candidate["action"], "formula_hash": candidate["formula_hash"],
                            "eval_status": "valid", "valid_e1": e1["eval_status"] == "valid", "mean_rank_ic": mean_ic,
                            "e1_decision": e1["decision"], "e3_status": e3["eval_status"], "e3_reason": e3["reason_code"],
                            "formal_account_valid": e3["eval_status"] == "valid", "jq_candidate_allowed": admission["jq_candidate_allowed"],
                            "process": process, "evaluation_id": evaluation_id,
                            "result_path": str((directory/"attempts"/attempt/_result_name()).relative_to(root)),
                            "n_signal_pairs": e1["n_pairs"], "blockers": admission["blockers"]}
                summaries["research_p_one_sided"] = (e1["diagnostics"].get("time_uncertainty") or {}).get("p_one_sided")
                summaries['behavior_hash'] = behavioral_signature(output['scores'],request['research_start'],request['research_end'])
                summaries['structural_family'] = structural_family(candidate['canonical_ast'])
                summaries['parent_comparisons'] = result.get('learning_parent_comparisons',[])
                files = {"candidate.json": candidate, "evaluation.json": result, "admission.json": admission,
                         "strategy_ir.json": result["strategy_ir"], "jq_strategy.py": script, "compiler.json": compiler,
                         "scores.json": output["scores"], "policy.json": policy}
                if "_score_transport" in output:
                    from .full_market_support import grid_bundle_files
                    files.update(grid_bundle_files(output["scores"], output["_score_transport"]))
                bundle_path = publish(directory/"bundles", files, {"campaign_id": campaign_id, "experiment_id": experiment,
                                     "input_hash": config["input_hash"], "code_hashes": code, "purpose": compiler["purpose"]})
                summaries["bundle_path"] = str(bundle_path.relative_to(root))
                database.finish(attempt, "completed", process["elapsed_seconds"], summaries)
                database.append("candidate_result", summaries)
                record_candidate_feedback(database, decision_id, candidate, result, evaluation_id,
                                          policy, baseline_ast, config, campaign_id, scores=output["scores"],
                                          vehicle_ic_values=_vehicle_pool(results))
                # P5 shadow collection for hooks 4+5 (one fact per round).
                if shadow_state.enabled:
                    sections = e1["diagnostics"].get("sections", [])
                    round_g1.append({"candidate_id": candidate["candidate_id"],
                                     "hypothesis_id": digest(candidate["hypothesis"]),
                                     "parent_ids": candidate["parent_ids"],
                                     "source_refs": candidate.get("source_evidence_ids", []),
                                     "generation_action_id": decision_id,
                                     "daily_ic": [s["rank_ic"] for s in sections],
                                     "daily_support": [s["pair_count"] for s in sections],
                                     "evaluation_bundle": {"e1_status": e1["eval_status"],
                                                           "e3_status": e3["eval_status"]}})
                    round_exports.append((evaluation_id, candidate["candidate_id"],
                                          result.get("strategy_ir")))
                archive = [p for _,p in database.facts("archive_member")]
                round_results.append(summaries)
                results.append(summaries)

            # M1 candidate-level parallelism (P-20261015-002): spawn every pending
            # candidate, then collect/settle in proposals order. Wall-clock becomes
            # max(child) instead of sum(child); the facts write sequence is identical
            # to the serial path (bit-exact A/B verified).
            # M2 (P-20261015-007): with FUXI_V1_PERSISTENT_POOL=1 the same
            # spawn-all/collect-in-order structure dispatches into the persistent
            # pool instead of per-candidate processes.
            parallel_candidates = os.environ.get("FUXI_V1_PARALLEL_CANDIDATES") == "1"
            pool_mode = pool is not None
            spawned = []
            for candidate in proposals:
                experiment = digest({"candidate": candidate, "input": config["input_hash"], "policy": policy["policy_hash"],
                                     "code": code, "environment": config["environment"], "baseline": baseline_ast, "seed": seed})
                cached = database.cached(experiment)
                if cached:
                    # A crash after settlement must not lose evidence or credit on restart.
                    verify(root/cached["bundle_path"])
                    database.append("candidate_result", cached)
                    restored = database.get(cached["evaluation_id"])
                    record_candidate_feedback(database, decision_id, candidate, restored, cached["evaluation_id"],
                                              policy, baseline_ast, config, campaign_id,
                                              behavior_hash=cached.get("behavior_hash"),
                                              vehicle_ic_values=_vehicle_pool(results))
                    if not any(r.get("evaluation_id") == cached["evaluation_id"] for r in results):
                        results.append(cached)
                    round_results.append(cached)
                    continue
                attempt = database.reserve(campaign_id, experiment, config["worker_timeout"])
                job = {"input_path": str(directory/"input.json"), "input_hash": config["input_hash"], "candidate": candidate,
                       "baseline_ast": baseline_ast, "policy": policy, "top_n": top_n}
                job['parent_candidates'] = [p for p in parents if p['candidate_id'] in candidate['parent_ids']][:2]
                job['pool_candidates'] = list(qualified_pool)
                if parallel_candidates or pool_mode:
                    if pool_mode:
                        handle = pool.spawn(job, directory/"attempts"/attempt)
                    else:
                        handle = _worker_spawn(job, directory/"attempts"/attempt,
                                               timeout=config["worker_timeout"],
                                               memory_mb=config["worker_memory_mb"])
                    spawned.append((candidate, attempt, experiment, handle))
                    continue
                output, process = worker(job, directory/"attempts"/attempt, timeout=config["worker_timeout"], memory_mb=config["worker_memory_mb"])
                settle_candidate(candidate, attempt, experiment, output, process)
            for candidate, attempt, experiment, handle in spawned:
                if pool_mode:
                    output, process = pool.collect(handle)
                else:
                    output, process = _worker_collect(handle)
                settle_candidate(candidate, attempt, experiment, output, process)
            # P5 shadow hooks 4+5: G1/complement and export readings, one fact per round.
            if shadow_state.enabled:
                if round_g1:
                    _safe(shadow_state, database, campaign_id, "g1_complement", round_index,
                          lambda g=round_g1: hook_g1_complement(
                              shadow_state, database, campaign_id, round_index,
                              fold_segment=[request['research_start'], request['research_end']],
                              per_candidate=g))
                if round_exports:
                    _safe(shadow_state, database, campaign_id, "export", round_index,
                          lambda x=round_exports: hook_export(
                              shadow_state, database, campaign_id, round_index, x,
                              code_ref=digest(code), config_ref=digest(config),
                              input_contract_ref=config["input_hash"],
                              market=payload["market"],
                              research_window=[request['research_start'],
                                               request['research_end']]))
            # Execute a genuine model action, with features held fixed across algorithm comparisons.
            model_summary = {"status": "not_requested"}
            if model_algorithms and round_index < len(model_algorithms):
                algorithm = model_algorithms[round_index]
                if model_capabilities().get(algorithm) == "supported":
                    model_summary=execute_model_action(database,campaign_id,algorithm,{"baseline_return20":baseline_scores},
                          snapshot,labels,payload,policy,baseline_scores,top_n,seed,directory)
                else:
                    model_summary = {"status": "unsupported", "algorithm": algorithm}
            elif model_algorithms and round_index == len(model_algorithms) and proposals:
                from .models import validate_axis
                added=evaluate_ast(proposals[0]["canonical_ast"],snapshot)
                name="added_"+proposals[0]["formula_hash"]
                # Both arms use identical complete-case cells; missingness cannot change training support silently.
                if full_options is not None:
                    from .columnar import common_support_grid
                    common = common_support_grid(baseline_scores, added)
                else:
                    common={d:{s:baseline_scores[d][s] if added[d][s] is not None else None for s in baseline_scores[d]}
                            for d in baseline_scores}
                parent={"features":["baseline_return20"],"model":"ridge","protocol":policy["policy_hash"]}
                child=dict(parent,features=["baseline_return20",name])
                validate_axis(parent,child,"features")
                a=execute_model_action(database,campaign_id,"ridge",{"baseline_return20":common},
                         snapshot,labels,payload,policy,baseline_scores,top_n,seed,directory)
                b=execute_model_action(database,campaign_id,"ridge",{"baseline_return20":common,name:added},
                         snapshot,labels,payload,policy,baseline_scores,top_n,seed,directory)
                model_summary={"status":"valid" if a["status"] == b["status"] == "valid" else "data_insufficient",
                               "action":"add_one_feature","candidate_id":proposals[0]["candidate_id"],
                               "frozen_parent":parent,"child":child,"parent_model":a,"child_model":b,
                               "same_complete_case_support":True}
                database.append("feature_axis_experiment",model_summary)
            portfolio_summary = {"status": "not_requested"}
            if round_index == 4:
                portfolio_summary = execute_portfolio_action(database, campaign_id, payload, policy,
                                      baseline_scores, top_n, directory)
            row = {"round": round_index, "arm": arm, "mode": mode, "generated_count": len(proposals),
                   "selection_count": 1, "generation_context_hash": generation_context,
                   "generation_attempts": decision["acquisition"].get("generation_counts", {}).get("generation_attempts", len(proposals)),
                   "unique_generated_count": len(proposals),
                   "parameter_attempts": decision["acquisition"].get("generation_counts", {}).get("parameter_attempts", 0),
                   "generator_status": decision["acquisition"].get("generator_status", "completed"),
                   "archetype_distribution": decision["acquisition"].get("archetype_distribution", {}),
                   "evaluated_count": len(round_results), "valid_e1_count": sum(x["valid_e1"] for x in round_results),
                   "error_count": sum(x["eval_status"] == "error" for x in round_results),
                   "jq_eligible_count": sum(x["jq_candidate_allowed"] for x in round_results),
                   "elapsed_seconds": time.monotonic()-round_started, "model_action": model_summary,
                   "portfolio_action": portfolio_summary,
                   "archive_size": len(pareto_archive(archive)), "stop_reason": "candidate_budget" if proposals else "novelty_exhausted"}
            # P-20261018-014: grow the qualified pool AFTER this round settles, so the
            # next round's E2 gate sees only prior-round all-pass factors. Same-round
            # candidates never de-duplicate against each other (deterministic, parallel-safe).
            for summary in round_results:
                if summary.get("jq_candidate_allowed") and summary.get("candidate_id") in candidates_all:
                    qualified_pool.append({"candidate_id": summary["candidate_id"],
                                           "canonical_ast": candidates_all[summary["candidate_id"]]["canonical_ast"]})
            database.append("round_completed", row, campaign_id+":round:"+str(round_index))
            history.append(row)
            if fitness_mode != "off":
                triage_cards = [c for _,c in database.facts('feedback_v2') if database.state_usable(c['state_id'])]
                if triage_cards:
                    triage_id = campaign_id+":fitness_triage:"+str(round_index)
                    if not database.db.execute("SELECT 1 FROM facts WHERE id=?", (triage_id,)).fetchone():
                        vehicle_pool = _vehicle_pool(results)
                        triage_payload = dict(fitness_triage(triage_cards),
                                              round=round_index, mode=fitness_mode,
                                              vehicle_n=len(vehicle_pool),
                                              enforce_consumer="imported_memory_ordering",
                                              usage="descriptive_display_only_never_acceptance")
                        database.append("fitness_triage", triage_payload, triage_id)
            if study_protocol and len(history) % study_protocol['checkpoint_every'] == 0:
                cards = [c for _,c in database.facts('feedback_v2') if database.state_usable(c['state_id'])]
                check = checkpoint(history,results,cards,study_protocol['checkpoint_every'],checkpoints[-1] if checkpoints else None)
                database.append('learning_checkpoint',check,campaign_id+':checkpoint:'+str(len(history)))
                checkpoints.append(check)
            print(json.dumps({"campaign_id": campaign_id, **row}, ensure_ascii=False), flush=True)
        require(code_manifest() == code, "code changed mid-campaign; evidence needs a new code identity")
        if full_options is not None:
            from .columnar import selected_by_dates
            selected_before = selected_by_dates(baseline_scores, request["calibration_start"], request["calibration_end"])
            selected_after = selected_by_dates(baseline_scores, request["research_start"], request["research_end"])
        else:
            selected_before = {d:v for d,v in baseline_scores.items() if d <= request["calibration_end"]}
            selected_after = {d:v for d,v in baseline_scores.items() if d >= request["research_start"]}
        report = {"campaign_id": campaign_id, "mode": mode, "rounds": history, "candidate_results": results,
                  "unique_formula_count": len({r.get("formula_hash") for r in results if r.get("formula_hash")}),
                  "attempted_candidates": len(results), "engineering_errors": sum(r["eval_status"] == "error" for r in results),
                  "valid_descriptive_factors": sum(r["valid_e1"] for r in results),
                  "formal_account_valid": sum(r["formal_account_valid"] for r in results),
                  "jq_candidate_count": sum(r["jq_candidate_allowed"] for r in results), "external_jq_runs": 0,
                  "economic_reward_count": len(database.effective_rewards("economic")),
                  "capability_reward_count": len(database.effective_rewards("capability")),
                  "pareto_archive": pareto_archive(archive), "policy": policy, "input_hash": config["input_hash"],
                  "code_hashes": code, "environment": config["environment"], "recovered_attempts": recovered,
                  "llm_calls": llm_calls, "drift": drift_report(selected_before, selected_after),
                  "raw_trials_and_viewings": {"candidate_facts": len(database.facts("candidate")), "decision_facts": len(database.facts("research_decision"))},
                  "multiple_search_diagnostics": holm({str(i)+":"+r["candidate_id"]:r.get("research_p_one_sided") for i,r in enumerate(results)},request["alpha"]) if results else {},
                  "multiple_search_interpretation": "adaptive historical search; diagnostic Holm table is not confirmatory error-control proof",
                  "scope": snapshot.payload["exposure"], "data_quality": snapshot.payload.get("quality", {}),
                  'reachability':reachability,'checkpoints':checkpoints,
                  "scheduler_policy": scheduler_policy, "scheduler_stops": [p for _,p in database.facts("scheduler_stop")],
                  "archetype_coverage": sorted(_covered_archetypes(history)),
                  "archetype_distribution": {a: sum(r.get("archetype_distribution", {}).get(a, 0) for r in history)
                                             for a in _covered_archetypes(history)},
                  "live_permission": False, "status": "completed" if len(history)==rounds else
                      "stopped_all_arms_exhausted" if database.facts("scheduler_stop") else
                      'stopped_by_checkpoint' if any(p['stop'] for p in checkpoints) else 'checkpoint_paused'}
        if requested_memory:
            report["memory_source_bindings"] = memory_bindings
            report["imported_feedback_counts"] = [p.get("imported_feedback_count", 0)
                                                for _, p in database.facts("research_decision")]
        final_id = database.append("campaign_report", report)
        database.db.execute("UPDATE campaigns SET status=? WHERE id=?", (report['status'],campaign_id))
        state_id = "scheduler_"+digest([campaign_id, final_id])
        if not database.db.execute("SELECT 1 FROM consumers WHERE id=?", (state_id,)).fetchone():
            database.register_consumer(state_id, [r["event_id"] for r in database.effective_rewards("capability")], final_id)
        result_path = directory/("report_"+digest(report)[:16]+".json")
        if not result_path.exists():
            write_new(result_path, report)
        return result_path, report
    finally:
        if pool is not None:
            pool.close()
        database.close()
