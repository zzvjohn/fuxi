"""Transactional research facts, budgets, corrected feedback and exposure isolation."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid
from .contracts import canonical, digest, require, timestamp, keys, encode_fact_payload, load_fact


def now():
    return datetime.now(timezone.utc).isoformat()


def overlaps(a, b):
    """Compare resolved dependencies, never just window names or snapshot hashes."""
    for scope in (a, b):
        keys(scope, ("sources", "instruments", "intervals", "label_intervals", "resolved"))
        require(scope["resolved"] is True and scope["sources"] and scope["instruments"], "unresolved data scope")
        for start, end in scope["intervals"] + scope["label_intervals"]:
            require(timestamp(start) <= timestamp(end), "invalid scope interval")
    def common(x, y):
        return "*" in x or "*" in y or bool(set(x) & set(y))
    if not common(a["sources"], b["sources"]) or not common(a["instruments"], b["instruments"]):
        return False
    return any(max(timestamp(x), timestamp(u)) <= min(timestamp(y), timestamp(v))
               for x, y in a["intervals"] + a["label_intervals"]
               for u, v in b["intervals"] + b["label_intervals"])


_OPEN_ATTEMPTS = 3
_OPEN_RETRY_DELAY = 0.05


def _transient_write_error(exc):
    """Signature of the WAL shared-memory initialization race, nothing else.

    Two connections opening a fresh WAL database concurrently race to
    initialize the shared-memory index; the loser is left permanently
    flagged read-only: every later write fails with this message, the busy
    handler is never consulted, and the connection never recovers in place
    (pure sqlite3 reproduction on Windows: same-connection retries failed
    7/7 up to 1s; close+reopen succeeded 5/5).  Only a fresh connection may
    write again.  Real lock contention surfaces as "database is locked"
    after the busy timeout and must keep its current behaviour.
    """
    return (isinstance(exc, sqlite3.OperationalError)
            and "attempt to write a readonly database" in str(exc).lower())


class EvidenceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self.self_heals = 0
        self._open()

    def _open(self):
        """Open a write-ready connection; the connection's first write is here.

        The empty immediate transaction makes the WAL index initialization
        happen during setup, so a lost race is detected and repaired (the
        connection is discarded and reopened) before any caller observes it.
        """
        old = self.__dict__.get("db")
        if old is not None:
            try:
                old.close()
            except sqlite3.Error:
                pass
        for attempt in range(_OPEN_ATTEMPTS):
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript('''
        CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY,kind TEXT NOT NULL,
          payload TEXT NOT NULL,hash TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS facts_no_update BEFORE UPDATE ON facts BEGIN SELECT RAISE(ABORT,'immutable facts'); END;
        CREATE TRIGGER IF NOT EXISTS facts_no_delete BEFORE DELETE ON facts BEGIN SELECT RAISE(ABORT,'immutable facts'); END;
        CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY,config_hash TEXT NOT NULL,
          config TEXT NOT NULL,status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,campaign TEXT NOT NULL,experiment TEXT NOT NULL,
          state TEXT NOT NULL,reserved REAL NOT NULL,actual REAL,started TEXT NOT NULL,ended TEXT,result TEXT);
        CREATE TABLE IF NOT EXISTS rewards(slot TEXT NOT NULL,revision INTEGER NOT NULL,event_id TEXT UNIQUE NOT NULL,
          target TEXT NOT NULL,value REAL,source_id TEXT NOT NULL,supersedes TEXT,PRIMARY KEY(slot,revision));
        CREATE TABLE IF NOT EXISTS consumers(id TEXT PRIMARY KEY,dependencies TEXT NOT NULL,
          state TEXT NOT NULL,checkpoint TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS exposure(id TEXT PRIMARY KEY,cohort TEXT NOT NULL,scope TEXT NOT NULL,
          use TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS protection(cohort TEXT PRIMARY KEY,scope TEXT NOT NULL,frozen TEXT NOT NULL,
          state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS receipts(id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,
          state TEXT NOT NULL);
        ''')
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("COMMIT")
            except sqlite3.OperationalError as exc:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                if not _transient_write_error(exc) or attempt + 1 == _OPEN_ATTEMPTS:
                    raise
                self.self_heals += 1
                time.sleep(_OPEN_RETRY_DELAY * (attempt + 1))
                continue
            self.db = conn
            return

    def close(self):
        self._closed = True
        self.db.close()

    @contextmanager
    def transaction(self):
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if not _transient_write_error(exc) or self._closed:
                raise
            self.self_heals += 1
            self._open()
            self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def append(self, kind, payload, event_id=None):
        h = digest(payload)
        event_id = event_id or kind + "_" + h
        old = self.db.execute("SELECT hash,kind FROM facts WHERE id=?", (event_id,)).fetchone()
        if old:
            require(old["hash"] == h and old["kind"] == kind, "conflicting immutable event")
            return event_id
        self.db.execute("INSERT INTO facts VALUES(?,?,?,?,?)", (event_id, kind, encode_fact_payload(canonical(payload).decode()), h, now()))
        return event_id

    def get(self, event_id):
        row = self.db.execute("SELECT payload,hash FROM facts WHERE id=?", (event_id,)).fetchone()
        require(row is not None, "unknown evidence id")
        payload = load_fact(row["payload"])
        require(digest(payload) == row["hash"], "corrupted evidence")
        return payload

    def facts(self, kind):
        return [(r["id"], load_fact(r["payload"])) for r in self.db.execute(
            "SELECT id,payload FROM facts WHERE kind=? ORDER BY rowid", (kind,))]

    def campaign(self, campaign_id, config):
        with self.transaction():
            row = self.db.execute("SELECT config_hash FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if row:
                require(row[0] == digest(config), "campaign config is frozen; create a new campaign")
            else:
                self.db.execute("INSERT INTO campaigns VALUES(?,?,?,?)",
                                (campaign_id, digest(config), canonical(config).decode(), "queued"))
                self.append("campaign_config", config, campaign_id + ":config")

    def reserve(self, campaign_id, experiment_id, units):
        require(units > 0, "positive budget reservation required")
        with self.transaction():
            row = self.db.execute("SELECT config,status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            require(row is not None, "unregistered campaign")
            require(row["status"] != "budget_overrun", "campaign halted after budget overrun")
            cap = json.loads(row[0])["budget_units"]
            spent = self.db.execute("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM attempts WHERE campaign=?",
                                    (campaign_id,)).fetchone()[0]
            require(spent + units <= cap, "budget exhausted")
            attempt = "attempt_" + uuid.uuid4().hex
            self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)",
                            (attempt, campaign_id, experiment_id, "running", units, None, now(), None, None))
            self.db.execute("UPDATE campaigns SET status='running' WHERE id=?", (campaign_id,))
            self.append("attempt_reserved", {"attempt_id": attempt, "campaign_id": campaign_id,
                                           "experiment_id": experiment_id, "units": units})
            return attempt

    def finish(self, attempt, state, actual_units, result):
        require(state in {"completed", "failed", "waiting_data", "cancelled"} and actual_units >= 0,
                "invalid attempt settlement")
        with self.transaction():
            row = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
            require(row is not None and row["state"] == "running", "attempt already settled/unknown")
            self.db.execute("UPDATE attempts SET state=?,actual=?,ended=?,result=? WHERE id=?",
                            (state, actual_units, now(), canonical(result).decode(), attempt))
            self.append("attempt_settled", {"attempt_id": attempt, "state": state,
                                           "actual_units": actual_units, "result": result})
            if actual_units > row["reserved"]:
                self.db.execute("UPDATE campaigns SET status='budget_overrun' WHERE id=?", (row["campaign"],))

    def recover(self, campaign_id):
        """Caller holds the workspace lock. Interrupted attempt cost is conservatively reserved."""
        rows = list(self.db.execute("SELECT * FROM attempts WHERE campaign=? AND state='running'", (campaign_id,)))
        for row in rows:
            self.finish(row["id"], "failed", row["reserved"], {"reason": "interrupted_unknown_actual_cost"})
        return len(rows)

    def cached(self, experiment_id):
        row = self.db.execute("SELECT result FROM attempts WHERE experiment=? AND state='completed' ORDER BY rowid DESC LIMIT 1",
                              (experiment_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def reward(self, slot, target, value, source_id, *, supersedes=None):
        require(target in {"capability", "economic"}, "unknown feedback ledger")
        require(len(slot) == 5, "reward slot must bind decision,candidate,target,window,baseline")
        require(slot[2] == target, "reward target mismatch")
        source = self.get(source_id)
        if target == "economic" and value is not None:
            require(source.get("eval_status") == "valid" and source.get("evidence_scope") == "account"
                    and source.get("n_pairs", 0) > 0 and source.get("metrics") is not None
                    and source.get("economic_reward_eligible") is True,
                    "non-economic evidence cannot produce economic reward")
        require(value is None or isinstance(value, (float, int)) and -1 <= value <= 1, "reward outside frozen scale")
        key = digest(slot)
        with self.transaction():
            rows = list(self.db.execute("SELECT * FROM rewards WHERE slot=? ORDER BY revision", (key,)))
            for old in rows:
                if old["source_id"] == source_id and old["value"] == value:
                    return old["event_id"]
            require((not rows and supersedes is None) or (rows and supersedes == rows[-1]["event_id"]),
                    "reward correction must supersede the latest event")
            revision = len(rows) + 1
            event_id = digest([slot, revision, source_id, value])
            self.db.execute("INSERT INTO rewards VALUES(?,?,?,?,?,?,?)",
                            (key, revision, event_id, target, value, source_id, supersedes))
            self.append("reward", {"slot": slot, "revision": revision, "value": value,
                                   "source_id": source_id, "supersedes": supersedes}, event_id)
            if supersedes:
                self.invalidate(supersedes)
            return event_id

    def effective_rewards(self, target):
        return [dict(r) for r in self.db.execute('''SELECT r.* FROM rewards r JOIN
            (SELECT slot,MAX(revision) revision FROM rewards GROUP BY slot) x
            ON r.slot=x.slot AND r.revision=x.revision WHERE r.target=?''', (target,))]

    def register_consumer(self, state_id, dependencies, checkpoint):
        require(self.db.execute("SELECT 1 FROM consumers WHERE id=?", (state_id,)).fetchone() is None,
                "learned state id already exists")
        self.db.execute("INSERT INTO consumers VALUES(?,?,?,?)",
                        (state_id, canonical(dependencies).decode(), "valid", checkpoint))

    def invalidate(self, dependency):
        affected, frontier = [], {dependency}
        while frontier:
            current, frontier = frontier, set()
            for row in self.db.execute("SELECT * FROM consumers WHERE state='valid'"):
                if set(json.loads(row["dependencies"])) & current:
                    self.db.execute("UPDATE consumers SET state='rebuild_required' WHERE id=?", (row["id"],))
                    affected.append(row["id"])
                    frontier.add(row["id"])
        self.append("invalidation", {"dependency": dependency, "affected_state_ids": affected,
                                     "rebuild_status": "required", "at": now()})
        return affected

    def state_usable(self, state_id):
        row = self.db.execute("SELECT state FROM consumers WHERE id=?", (state_id,)).fetchone()
        return row is not None and row[0] == "valid"

    def protect(self, cohort, scope, frozen_at):
        overlaps(scope, scope)
        require(all(timestamp(a) > timestamp(frozen_at) for a, _ in scope["intervals"]),
                "protected forward window must start after freeze")
        for row in self.db.execute("SELECT scope FROM exposure"):
            require(not overlaps(scope, json.loads(row[0])), "protected scope already exposed")
        self.db.execute("INSERT INTO protection VALUES(?,?,?,?)", (cohort, canonical(scope).decode(), frozen_at, "sealed"))

    def expose(self, cohort, scope, use, *, acceptance=False):
        overlaps(scope, scope)
        require(use in {"research", "acceptance", "historical_inheritance", "derived"}, "unknown exposure use")
        for row in self.db.execute("SELECT * FROM protection WHERE state='sealed'"):
            if overlaps(scope, json.loads(row["scope"])):
                require(acceptance and cohort == row["cohort"] and use == "acceptance",
                        "protected data dependency intersection")
        event = "exposure_" + uuid.uuid4().hex
        self.db.execute("INSERT INTO exposure VALUES(?,?,?,?,?)", (event, cohort, canonical(scope).decode(), use, now()))
        return event

    def retrieve(self, query, scope, cohort, limit=8):
        visible = []
        for event_id, card in self.facts("experience"):
            if card.get("level") == "invalidated":
                continue
            if card.get("state_id") and not self.state_usable(card["state_id"]):
                continue
            blocked = any(overlaps(card["scope"], json.loads(row["scope"])) for row in self.db.execute(
                "SELECT scope FROM protection WHERE state='sealed'"))
            if blocked:
                continue
            if overlaps(card["scope"], scope):
                score = sum(word.lower() in json.dumps(card).lower() for word in query.split())
                visible.append((score, event_id, card))
        return [(i, c) for _, i, c in sorted(visible, key=lambda x: (-x[0], x[1]))[:limit]]

    def receipt(self, kind, payload):
        receipt_id = kind + "_" + digest(payload)
        self.db.execute("INSERT OR IGNORE INTO receipts VALUES(?,?,?,?)",
                        (receipt_id, kind, canonical(payload).decode(), "active"))
        return receipt_id

    def consume_receipt(self, receipt_id, kind):
        row = self.db.execute("SELECT * FROM receipts WHERE id=?", (receipt_id,)).fetchone()
        require(row is not None and row["kind"] == kind and row["state"] == "active", "unknown/inactive authority receipt")
        payload = json.loads(row["payload"])
        require(receipt_id == kind+"_"+digest(payload), "authority receipt content changed")
        dependencies = {"bundle_receipt": "release_bundle", "forward_receipt": "forward_acceptance",
                        "pending_receipt": "forward_pending", "research_receipt": "research_acceptance",
                        "protocol_receipt": "confirmatory_protocol", "parity_receipt": "jq_parity",
                        "data_receipt_id": "data_contract"}
        for key, parent_kind in dependencies.items():
            if payload.get(key) is not None:
                self.consume_receipt(payload[key], parent_kind)
        return payload

    def revoke(self, receipt_id, reason):
        require(self.db.execute("SELECT 1 FROM receipts WHERE id=?", (receipt_id,)).fetchone() is not None,
                "unknown receipt")
        self.db.execute("UPDATE receipts SET state='revoked' WHERE id=?", (receipt_id,))
        self.append("revocation", {"receipt_id": receipt_id, "reason": reason, "at": now()})
