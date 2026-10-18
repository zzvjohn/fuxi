"""Process identities and bounded telemetry for a long controller workflow.

Only descendants observed below a verified live member are admitted. An exited
identity stays in the audit history; a reused PID is not that identity. Popen's
owned launcher handle remains the caller's independent fallback for reaping.
"""

from collections import deque
from copy import deepcopy
import time


class ControllerTreeGuard:
    """Monitor and reclaim this workflow's verified (PID, creation time) tree."""

    def __init__(self, process, *, history_limit=512):
        import psutil
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        self.psutil, self.process = psutil, process
        self.started = time.monotonic()
        self.history_limit = history_limit
        self.samples = deque(maxlen=history_limit)
        self.events = deque(maxlen=history_limit)
        self.errors = deque(maxlen=history_limit)
        self.sample_count = self.event_count = self.error_count = 0
        self.peak_rss = self.peak_launcher_rss = self.max_children = 0
        self.monitor_error = self.cleanup = None
        self.known = {}
        self._active = {}
        self._signalled = set()
        self._discovery_uncertain = False
        launcher = psutil.Process(process.pid)
        self.launcher_identity = (process.pid, launcher.create_time())
        self._admit(self.launcher_identity, parent=None)

    @staticmethod
    def _identity_row(identity):
        return {"pid": identity[0], "create_time": identity[1]}

    def _event(self, event, identity, **details):
        self.event_count += 1
        self.events.append({"event": event, **self._identity_row(identity),
                            "elapsed_seconds": time.monotonic()-self.started, **details})

    def _error(self, operation, exc, identity=None):
        self.error_count += 1
        row = {"operation": operation, "error_type": type(exc).__name__, "reason": str(exc)}
        if identity is not None:
            row.update(self._identity_row(identity))
        self.errors.append(row)

    def _retire(self, identity, reason):
        record = self.known[identity]
        if record["state"] == "retired":
            return
        record.update(state="retired", retirement_reason=reason,
                      retired_after_seconds=time.monotonic()-self.started)
        if self._active.get(identity[0]) == identity:
            del self._active[identity[0]]
        self._event("identity_retired", identity, reason=reason)

    def _admit(self, identity, parent):
        previous = self._active.get(identity[0])
        if previous is not None and previous != identity:
            self._retire(previous, "pid_reused")
        if identity in self.known:
            # A retired process identity cannot return to life. Creation-time
            # equality is not permission to resurrect a dead tracked member.
            return self.known[identity]["state"] == "active"
        depth = self.known[parent]["depth"]+1 if parent is not None else 0
        self.known[identity] = {**self._identity_row(identity), "state": "active", "depth": depth,
                                "discovery_parent": self._identity_row(parent) if parent else None,
                                "observed_after_seconds": time.monotonic()-self.started}
        self._active[identity[0]] = identity
        self._event("identity_observed", identity)
        return True

    def _live(self, identity):
        if self.known[identity]["state"] != "active":
            return None
        try:
            current = self.psutil.Process(identity[0])
            if current.create_time() != identity[1]:
                self._retire(identity, "pid_reused")
                return None
            if current.status() == self.psutil.STATUS_ZOMBIE:
                self._retire(identity, "zombie")
                return None
            return current
        except self.psutil.NoSuchProcess:
            self._retire(identity, "exited")
            return None

    def _discover(self):
        # Include previously proved children after reparenting. Use immediate
        # children, verifying each ancestry edge, rather than trusting a stale
        # recursive list returned while a parent PID changes identity.
        pending = deque(self._active.values())
        visited = set()
        while pending:
            parent_identity = pending.popleft()
            if parent_identity in visited:
                continue
            visited.add(parent_identity)
            parent = self._live(parent_identity)
            if parent is None:
                continue
            try:
                children = parent.children(recursive=False)
            except self.psutil.NoSuchProcess:
                self._live(parent_identity)
                continue
            if self._live(parent_identity) is None:
                continue
            for observed in children:
                try:
                    identity = (observed.pid, observed.create_time())
                    child = self.psutil.Process(identity[0])
                    if child.create_time() != identity[1] or child.status() == self.psutil.STATUS_ZOMBIE:
                        continue
                    if child.ppid() != parent_identity[0] or identity[1] < parent_identity[1]:
                        continue
                    if self._live(parent_identity) is None:
                        break
                    # Check the candidate again after validating its live parent.
                    verified = self.psutil.Process(identity[0])
                    if verified.create_time() != identity[1] or verified.ppid() != parent_identity[0]:
                        continue
                except self.psutil.NoSuchProcess:
                    continue
                if self._admit(identity, parent_identity):
                    pending.append(identity)

    def sample(self):
        try:
            self._discover()
            members = []
            for identity in list(self._active.values()):
                current = self._live(identity)
                if current is None:
                    continue
                try:
                    rss = current.memory_info().rss
                    if self._live(identity) is None:
                        continue
                except self.psutil.NoSuchProcess:
                    continue
                members.append({**self._identity_row(identity), "rss_bytes": rss})
        except Exception as exc:
            self.monitor_error = {"error_type": type(exc).__name__, "reason": str(exc)}
            self._error("sample", exc)
            raise
        launcher = next((row["rss_bytes"] for row in members
                         if (row["pid"], row["create_time"]) == self.launcher_identity), 0)
        children = [row for row in members if (row["pid"], row["create_time"]) != self.launcher_identity]
        total = sum(row["rss_bytes"] for row in members)
        self.sample_count += 1
        self.peak_rss = max(self.peak_rss, total)
        self.peak_launcher_rss = max(self.peak_launcher_rss, launcher)
        self.max_children = max(self.max_children, len(children))
        self.samples.append({"elapsed_seconds": time.monotonic()-self.started, "tree_rss_bytes": total,
                             "launcher_rss_bytes": launcher, "peak_tree_rss_bytes": self.peak_rss,
                             "child_count": len(children), "children": children})
        return total

    def live_descendants(self):
        return [identity[0] for identity in list(self._active.values())
                if identity != self.launcher_identity and self._live(identity) is not None]

    def _remaining(self):
        survivors, unverified = [], []
        for identity in list(self._active.values()):
            try:
                if self._live(identity) is not None:
                    survivors.append(self._identity_row(identity))
            except Exception as exc:
                unverified.append(self._identity_row(identity))
                self._error("verify_remaining", exc, identity)
        return survivors, unverified

    def stop(self, seconds=3.):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 <= seconds <= 60:
            raise ValueError("cleanup seconds must be between zero and sixty")
        deadline = time.monotonic()+seconds
        while True:
            try:
                self._discover()
            except Exception as exc:
                self._discovery_uncertain = True
                self._error("discover", exc)
            waiting = []
            # Leaf identities before ancestors. The Popen handle is never used
            # as a substitute for any descendant's verified creation time.
            ordered = sorted(list(self._active.values()),
                             key=lambda identity: self.known[identity]["depth"], reverse=True)
            for identity in ordered:
                try:
                    current = self._live(identity)
                    if current is None:
                        continue
                    # psutil.kill additionally checks the Process object's
                    # creation identity immediately before sending a signal.
                    current.kill()
                    self._signalled.add(identity)
                    waiting.append(current)
                except self.psutil.NoSuchProcess:
                    # Final verification below handles exit/reuse, and records
                    # an access failure without abandoning other owned members.
                    continue
                except Exception as exc:
                    self._error("kill", exc, identity)
            if waiting:
                try:
                    self.psutil.wait_procs(waiting, timeout=min(.25, max(0., deadline-time.monotonic())))
                except Exception as exc:
                    self._error("wait", exc)
            survivors, unverified = self._remaining()
            if (not survivors and not unverified) or time.monotonic() >= deadline:
                break
            if not waiting:
                time.sleep(min(.05, max(0., deadline-time.monotonic())))
        self.cleanup = {"signalled": [self._identity_row(identity) for identity in sorted(self._signalled)],
                        "survivors": survivors, "unverified": unverified, "errors": list(self.errors),
                        "error_count": self.error_count, "descendants_unverified": self._discovery_uncertain,
                        "complete": not survivors and not unverified and not self._discovery_uncertain,
                        "budget_seconds": seconds}
        return deepcopy(self.cleanup)

    def diagnostics(self):
        return deepcopy({"memory_measurement": "sum_process_tree_rss", "peak_tree_rss_bytes": self.peak_rss,
                         "peak_launcher_rss_bytes": self.peak_launcher_rss, "max_children": self.max_children,
                         "launcher_identity": self._identity_row(self.launcher_identity),
                         "observed_processes": [self.known[identity] for identity in sorted(self.known)],
                         "observed_process_count": len(self.known), "resource_samples": list(self.samples),
                         "sample_count": self.sample_count, "sample_history_limit": self.history_limit,
                         "discarded_sample_count": self.sample_count-len(self.samples),
                         "identity_events": list(self.events), "identity_event_count": self.event_count,
                         "monitor_error": self.monitor_error, "cleanup": self.cleanup})


TreeGuard = ControllerTreeGuard
_ControllerTreeGuard = ControllerTreeGuard
