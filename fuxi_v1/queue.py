"""Dedup, cooldown and dependency queues for the generation loop (P1, v2.0 landing).

R2 blueprint section 5.3 / section 7:

* Identical AST + identical evaluation contract reads the cache, records the
  duplicate attempt and adds no new formula/behaviour output.
* Two consecutive microbatches without new valid behaviour cool the arm down
  (a configurable routing device, not a significance judgement).
* Missing fields / immature labels queue a concrete dependency task; the same
  failed candidate is not regenerated every round while the dependency stands.
"""

from .contracts import ContractError, digest, keys, require

DEDUP_SCHEMA = "fuxi.dedup_ledger.1"


class DedupLedger:
    """Records attempts by (ast_identity, eval_contract_identity)."""

    def __init__(self):
        self._seen = {}
        self.new_attempts = 0
        self.duplicate_attempts = 0

    def record(self, ast_identity, eval_contract_identity, candidate_id):
        require(isinstance(ast_identity, str) and ast_identity
                and isinstance(eval_contract_identity, str) and eval_contract_identity,
                "identities required")
        key = digest({"ast": ast_identity, "contract": eval_contract_identity})
        if key in self._seen:
            self.duplicate_attempts += 1
            return {"verdict": "duplicate", "first_candidate_id": self._seen[key],
                    "new_output": False}
        self._seen[key] = candidate_id
        self.new_attempts += 1
        return {"verdict": "new", "new_output": True}


class CooldownTracker:
    """Arm cooldown after ``threshold`` consecutive zero-behaviour microbatches."""

    def __init__(self, threshold=2):
        require(type(threshold) is int and threshold > 0, "threshold must be positive")
        self.threshold = threshold
        self._streaks = {}

    def record(self, arm, new_behavior_count):
        require(type(new_behavior_count) is int and new_behavior_count >= 0, "invalid count")
        streak = 0 if new_behavior_count > 0 else self._streaks.get(arm, 0) + 1
        self._streaks[arm] = streak
        return {"arm": arm, "zero_streak": streak,
                "cooldown": streak >= self.threshold}

    def reset(self, arm):
        self._streaks.pop(arm, None)


class DependencyQueue:
    """Concrete dependency tasks; identical needs coalesce, no regeneration."""

    def __init__(self):
        self.tasks = {}          # need_id -> {"need": need, "waiters": [candidate_id]}
        self.resolved = {}       # need_id -> resolution receipt
        self.task_order = []

    def request(self, need_id, need, candidate_id):
        require(isinstance(need_id, str) and need_id, "need_id required")
        require(isinstance(need, dict) and need, "need must describe the dependency")
        if need_id in self.resolved:
            return {"verdict": "resolved", "candidate_id": candidate_id,
                    "receipt": self.resolved[need_id]}
        if need_id in self.tasks:
            if candidate_id in self.tasks[need_id]["waiters"]:
                return {"verdict": "already_waiting", "candidate_id": candidate_id}
            self.tasks[need_id]["waiters"].append(candidate_id)
            return {"verdict": "waiting", "task_exists": True,
                    "waiters": len(self.tasks[need_id]["waiters"])}
        self.tasks[need_id] = {"need": need, "waiters": [candidate_id]}
        self.task_order.append(need_id)
        return {"verdict": "task_created", "task_exists": False,
                "waiters": 1}

    def resolve(self, need_id, receipt):
        require(need_id in self.tasks, "unknown dependency task")
        require(isinstance(receipt, dict) and receipt, "receipt required")
        waiters = self.tasks.pop(need_id)["waiters"]
        self.resolved[need_id] = {"receipt": receipt, "unblocked": waiters}
        return {"unblocked": waiters}

    def pending_tasks(self):
        return [(need_id, self.tasks[need_id]["need"], self.tasks[need_id]["waiters"])
                for need_id in self.task_order if need_id in self.tasks]
