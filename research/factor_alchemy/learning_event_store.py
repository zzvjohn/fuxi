"""Transactional shadow feedback chain for mechanism research decisions.

The store never changes the production MAB, ExperienceMemory or S5. Only
``load_evidence``-verified trading observations can carry an economic score.
SQLite uniqueness and an immediate transaction make replay idempotent across
processes; callers still obey the project's single-writer runtime lock.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from mechanism_search import load_evidence, select_plan


def _catalog_hash(plans: list[dict]) -> str:
    raw = json.dumps(plans, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class LearningEventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                status TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                matured_on TEXT,
                economic_lcb_bps REAL,
                payload_json TEXT NOT NULL,
                UNIQUE(plan_id, source_sha256)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                plan_catalog_sha256 TEXT NOT NULL,
                as_of TEXT NOT NULL,
                seed INTEGER NOT NULL,
                round_index INTEGER NOT NULL UNIQUE,
                evidence_count INTEGER NOT NULL,
                selected_plan_id TEXT NOT NULL,
                audit_json TEXT NOT NULL
            );
        """)
        return db

    def ingest_evidence(self, evidence_path: str | Path, root: str | Path,
                        plans: list[dict], *, observed_on: str) -> dict:
        """Verify external artifacts first, then append each observation once."""
        date.fromisoformat(observed_on)
        observations = load_evidence(evidence_path, root, plans)
        prepared = []
        for row in observations:
            # An incomplete result becomes known on ingestion even though no
            # economic label matures. This lets exploration count attempts.
            maturity = row.get("matured_on") or (observed_on if row["status"] != "valid" else None)
            if row["status"] == "valid":
                if not maturity:
                    raise ValueError("valid economic reward lacks maturity date")
                date.fromisoformat(str(maturity))
            elif maturity:
                date.fromisoformat(str(maturity))
            prepared.append((row["observation_id"], row["plan_id"], row["status"],
                             row["source_sha256"], maturity,
                             row.get("economic_lcb_bps"),
                             json.dumps(row, ensure_ascii=False, sort_keys=True)))
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            inserted = 0
            for values in prepared:
                prior = db.execute(
                    "SELECT plan_id,status,source_sha256,matured_on,economic_lcb_bps,payload_json "
                    "FROM observations WHERE observation_id=?", (values[0],)
                ).fetchone()
                if prior is not None:
                    if tuple(prior) != values[1:]:
                        raise ValueError("conflicting observation replay")
                    continue
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)", values)
                inserted += 1
        return {"verified": len(prepared), "inserted": inserted,
                "duplicates": len(prepared) - inserted}

    def decide(self, decision_id: str, plans: list[dict], *, as_of: str,
               seed: int) -> tuple[dict, dict]:
        """Choose from matured observations and persist the choice atomically."""
        date.fromisoformat(as_of)
        if not decision_id or not plans:
            raise ValueError("decision_id and plans are required")
        by_id = {plan["plan_id"]: plan for plan in plans}
        if len(by_id) != len(plans):
            raise ValueError("duplicate plan in catalog")
        catalog_sha = _catalog_hash(plans)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM decisions WHERE decision_id=?",
                               (decision_id,)).fetchone()
            if prior is not None:
                if (prior["plan_catalog_sha256"] != catalog_sha or
                        prior["as_of"] != as_of or prior["seed"] != seed):
                    raise ValueError("conflicting decision replay")
                return by_id[prior["selected_plan_id"]], json.loads(prior["audit_json"])
            round_index = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            rows = db.execute(
                "SELECT payload_json FROM observations "
                "WHERE matured_on IS NOT NULL AND matured_on <= ? "
                "ORDER BY matured_on, observation_id", (as_of,)
            ).fetchall()
            observations = [json.loads(row[0]) for row in rows]
            selected, audit = select_plan(plans, observations, seed=seed,
                                          round_index=round_index)
            audit = dict(audit, decision_id=decision_id, as_of=as_of,
                         plan_catalog_sha256=catalog_sha,
                         evidence_count=len(observations))
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                       (decision_id, catalog_sha, as_of, seed, round_index,
                        len(observations), selected["plan_id"],
                        json.dumps(audit, ensure_ascii=False, sort_keys=True)))
            return selected, audit

    def snapshot(self) -> dict:
        """Read-only status view; missing evidence is never converted to zero."""
        with closing(self._connect()) as db, db:
            statuses = dict(db.execute(
                "SELECT status,COUNT(*) FROM observations GROUP BY status"
            ).fetchall())
            decisions = [dict(row) for row in db.execute(
                "SELECT decision_id,as_of,round_index,evidence_count,selected_plan_id "
                "FROM decisions ORDER BY round_index"
            )]
        return {"observation_statuses": statuses, "decisions": decisions,
                "valid_economic_observations": statuses.get("valid", 0)}

    def trusted_memory_view(self, *, as_of: str) -> list[dict]:
        """Append-only economic Memory projection, separate from legacy WEAK rows."""
        date.fromisoformat(as_of)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT observation_id,plan_id,source_sha256,matured_on,economic_lcb_bps "
                "FROM observations WHERE status='valid' AND matured_on<=? "
                "ORDER BY matured_on,observation_id", (as_of,)
            ).fetchall()
            return [dict(row) for row in rows]
