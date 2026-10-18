"""Status-aware, evidence-backed selection of Fuxi semantic research plans.

This selector only allocates *research* generation budget.  It never changes
S5 thresholds or grants JQ eligibility.  Economic observations are read from
immutable, hash-checked result files; unavailable economics cannot be encoded
as a zero reward.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from datetime import date
from pathlib import Path
from typing import Any

from semantic_plan import SCHEMA, validate_plan


EVIDENCE_SCHEMA = "fuxi_mechanism_evidence/v1"
AXES = ("event", "context", "quality", "direction", "output")
STATUSES = {"valid", "not_evaluated", "data_insufficient",
            "precheck_rejected", "error"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_catalog(path: str | Path, frequency: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("unknown semantic plan catalog schema")
    plans = []
    ids = set()
    for raw in data.get("plans", []):
        plan = validate_plan(raw)
        axes = plan.get("search_axes")
        if not isinstance(axes, dict) or any(
            not isinstance(axes.get(axis), str) or not axes[axis].strip()
            for axis in AXES
        ):
            raise ValueError("searchable plan requires five named search_axes")
        if plan["plan_id"] in ids:
            raise ValueError("duplicate plan_id in catalog")
        ids.add(plan["plan_id"])
        if plan["frequency"] == frequency:
            plans.append(plan)
    if not plans:
        raise ValueError("no plans for requested frequency")
    return plans


def _source_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError("economic source path leaves the Fuxi workspace")
    return path


def load_evidence(path: str | Path, root: str | Path, plans: list[dict]) -> list[dict]:
    """Verify source identity and refuse invented or partial economic scores."""
    evidence_file = Path(path)
    if not evidence_file.exists():
        raise FileNotFoundError(evidence_file)
    root = Path(root).resolve()
    by_id = {p["plan_id"]: p for p in plans}
    observations = []
    ids = set()
    source_keys = set()
    for number, line in enumerate(evidence_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != EVIDENCE_SCHEMA or row.get("status") not in STATUSES:
            raise ValueError(f"invalid evidence schema/status at line {number}")
        oid = row.get("observation_id")
        pid = row.get("plan_id")
        if not oid or oid in ids or pid not in by_id:
            raise ValueError(f"unknown plan or duplicate observation at line {number}")
        ids.add(oid)
        source = _source_path(root, str(row.get("source_file") or ""))
        raw = source.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != row.get("source_sha256"):
            raise ValueError(f"economic source hash differs at line {number}")
        source_key = (pid, actual_sha)
        if source_key in source_keys:
            raise ValueError("same economic source counted twice for one plan")
        source_keys.add(source_key)
        result = json.loads(raw.decode("utf-8-sig"))
        if row["status"] != result.get("status"):
            raise ValueError("evidence status differs from source result")
        protocol_sha = result.get("source_protocol_sha256")
        if not isinstance(protocol_sha, str) or not _SHA256.fullmatch(protocol_sha):
            raise ValueError("source economic result lacks frozen protocol identity")
        if row["status"] == "valid":
            if (result.get("candidate", {}).get("semantic_plan_id") != pid or
                    result.get("book_kind") != "fixed_shares_cash" or
                    result.get("return_status") != "valid" or
                    result.get("economic_verdict") != "valid" or
                    result.get("cost_bps") != 15 or
                    result.get("full_window_covered") is not True or
                    result.get("parity", {}).get(
                        "independent_execution_or_data_validation") is not True):
                raise ValueError("valid reward lacks complete verified trading book")
            paired_dates = result.get("paired_dates")
            window = result.get("window") or {}
            try:
                parsed_dates = [date.fromisoformat(x) for x in paired_dates]
                start = date.fromisoformat(window["start"])
                end = date.fromisoformat(window["end"])
                dated = (start <= parsed_dates[0] <= parsed_dates[-1] <= end
                         and parsed_dates == sorted(set(parsed_dates)))
            except (KeyError, ValueError, TypeError, IndexError):
                dated = False
            if not dated:
                raise ValueError("valid reward lacks ordered paired dates and window")
            parity = result["parity"]
            parity_file = parity.get("independent_artifact_file")
            parity_sha = parity.get("independent_artifact_sha256")
            if not isinstance(parity_sha, str) or not _SHA256.fullmatch(parity_sha) or not parity_file:
                raise ValueError("valid reward lacks independent parity artifact")
            parity_raw = _source_path(root, parity_file).read_bytes()
            if hashlib.sha256(parity_raw).hexdigest() != parity_sha:
                raise ValueError("independent parity artifact hash differs")
            parity_evidence = json.loads(parity_raw.decode("utf-8-sig"))
            nav_error = parity_evidence.get("max_abs_nav_error")
            if (parity_evidence.get("schema") != "fuxi_independent_book_parity/v1"
                    or parity_evidence.get("source_protocol_sha256") != protocol_sha
                    or not result.get("candidate", {}).get("candidate_id")
                    or parity_evidence.get("candidate_id") != result.get("candidate", {}).get("candidate_id")
                    or parity_evidence.get("compared_dates") != paired_dates
                    or parity_evidence.get("complete_event_coverage") is not True
                    or parity_evidence.get("independent_replay") is not True
                    or not isinstance(nav_error, (int, float)) or isinstance(nav_error, bool)
                    or not math.isfinite(nav_error) or nav_error < 0 or nav_error > 1e-8):
                raise ValueError("independent parity artifact does not verify result")
            expected_metric = ("incremental_net_bps" if by_id[pid]["target_use"] ==
                               "research_feature" else "net_excess_bps")
            if result.get("metric_kind") != expected_metric:
                raise ValueError("economic metric mismatches plan target_use")
            paired = result.get("paired_net_bps")
            if (not isinstance(paired, list) or len(paired) < 26 or
                    len(paired_dates) != len(paired) or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                        not math.isfinite(x) for x in paired) or
                    result["parity"].get("compared_days", 0) < len(paired)):
                raise ValueError("valid reward lacks finite paired economic dates")
            mean = sum(paired) / len(paired)
            centered = [value - mean for value in paired]
            n = len(paired)
            # Bartlett/Newey-West lag 4 for serially dependent paired dates.
            long_run_variance = sum(value * value for value in centered) / n
            for lag in range(1, min(4, n - 1) + 1):
                covariance = sum(centered[i] * centered[i - lag]
                                 for i in range(lag, n)) / n
                long_run_variance += 2 * (1 - lag / 5) * covariance
            row["economic_lcb_bps"] = mean - 1.96 * math.sqrt(
                max(0.0, long_run_variance) / n)
            row["paired_dates"] = len(paired)
            row["verified_trading_label"] = True
        observations.append(row)
    return observations


def select_plan(plans: list[dict], observations: list[dict], *, seed: int,
                round_index: int) -> tuple[dict, dict]:
    """Exploration floor plus lower-confidence economic exploitation.

    Nonvalid attempts inform coverage only.  With no valid books the selector
    explicitly stays in exploration; it cannot infer an economic winner.
    """
    if round_index < 0:
        raise ValueError("round_index must be nonnegative")
    by_id = {p["plan_id"]: p for p in plans}
    if len(by_id) != len(plans) or not plans:
        raise ValueError("plans must have unique IDs and be nonempty")
    attempts = {pid: 0 for pid in by_id}
    valid_scores = {pid: [] for pid in by_id}
    status_counts = {status: 0 for status in STATUSES}
    axis_counts = {axis: {} for axis in AXES}
    for row in observations:
        pid = row["plan_id"]
        if pid not in by_id or row["status"] not in STATUSES:
            raise ValueError("observation does not match plan catalog")
        attempts[pid] += 1
        status_counts[row["status"]] += 1
        for axis in AXES:
            value = by_id[pid]["search_axes"][axis]
            axis_counts[axis][value] = axis_counts[axis].get(value, 0) + 1
        if row["status"] == "valid":
            if row.get("verified_trading_label") is not True:
                raise ValueError("valid observation lacks verified trading label")
            score = row.get("economic_lcb_bps")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("valid observation lacks verified lower bound")
            valid_scores[pid].append(float(score))
    valid_total = sum(map(len, valid_scores.values()))
    # Overlapping lower bounds cannot be averaged into a new confidence
    # bound. For research allocation use the weakest verified observation;
    # this remains a heuristic, never a promotion or significance claim.
    positive = {
        pid: min(values)
        for pid, values in valid_scores.items()
        if values and min(values) > 0
    }
    mode = "exploit" if positive and round_index % 4 >= 2 else "explore"
    rng = random.Random(seed + round_index)
    candidates = list(plans)
    rng.shuffle(candidates)
    if mode == "exploit":
        candidates = [p for p in candidates if p["plan_id"] in positive]
        selected = max(candidates, key=lambda p: (
            positive[p["plan_id"]],
            -attempts[p["plan_id"]],
        ))
    else:
        def exploration_key(plan: dict) -> tuple:
            novelty = sum(1 / (1 + axis_counts[axis].get(plan["search_axes"][axis], 0))
                          for axis in AXES)
            return (novelty, -attempts[plan["plan_id"]])
        selected = max(candidates, key=exploration_key)
    audit = {
        "mode": mode,
        "selected_plan_id": selected["plan_id"],
        "seed": seed,
        "round_index": round_index,
        "valid_economic_observations": valid_total,
        "status_counts": {key: status_counts[key] for key in sorted(STATUSES)},
        "attempts_by_plan": attempts,
        "economic_lcb_bps_by_plan": {
            pid: (min(values) if values else None)
            for pid, values in valid_scores.items()
        },
        "grants_promotion": False,
    }
    return selected, audit
