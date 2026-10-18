# -*- coding: utf-8 -*-
"""A6 read-only admission checks for a candidate's own future clock and JQ parity.

This module does not read future prices, submit JQ jobs, or confer S5/JQ status.
The existing R5 skeleton has no candidates, so a new candidate needs a new freeze.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _iso_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("date must be an ISO string")
    return date.fromisoformat(value)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def assess_candidate_freeze(freeze: dict, as_of: str, evidence: list[dict]) -> dict:
    """Fail closed until one candidate has a complete, time-sealed future record.

    Evidence rows represent weekly *matured* sections.  Their status is not a
    return claim: an independent complete economic book is checked separately.
    """
    errors: list[str] = []
    try:
        frozen_at = datetime.fromisoformat(freeze["frozen_at"])
        if frozen_at.tzinfo is None:
            errors.append("frozen_at_timezone_missing")
        ceiling = _iso_date(freeze["freeze_data_ceiling"])
        first_signal = _iso_date(freeze["first_signal_date"])
        cutoff = datetime.fromisoformat(freeze["first_signal_cutoff_at"])
        now = _iso_date(as_of)
        if cutoff.tzinfo is None or cutoff.date() != first_signal or frozen_at >= cutoff:
            errors.append("freeze_not_before_first_signal_cutoff")
        if not ceiling < first_signal:
            errors.append("signal_not_after_data_ceiling")
        if not frozen_at.date() <= first_signal:
            errors.append("signal_precedes_freeze")
        if now < first_signal:
            errors.append("future_not_started")
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "precheck_rejected", "reasons": ["invalid_time_contract:" + str(exc)], "matured_sections": 0}
    for field in ("candidate_id", "formula_sha256", "plan_sha256", "code_sha256",
                  "evaluation_contract_sha256", "independent_book_sha256"):
        value = freeze.get(field)
        if not value or (field.endswith("sha256") and not _hash(value)):
            errors.append("missing_or_invalid_" + field)
    if freeze.get("economic_label_status") != "valid":
        errors.append("trusted_economic_label_missing")
    if freeze.get("research_exposed") is not False:
        errors.append("research_exposure_not_excluded")
    required = freeze.get("sections_required")
    if not isinstance(required, int) or isinstance(required, bool) or required < 1:
        errors.append("invalid_section_requirement")
        required = 0
    schedule = freeze.get("section_schedule")
    if not isinstance(schedule, list) or len(schedule) < required:
        errors.append("missing_frozen_section_schedule")
        schedule = []
    try:
        expected_signals = [_iso_date(item["signal_date"]) for item in schedule]
        expected_maturities = [_iso_date(item["label_matured_on"]) for item in schedule]
        if (len(set(expected_signals)) != len(expected_signals)
                or expected_signals != sorted(expected_signals)
                or (expected_signals and expected_signals[0] != first_signal)
                or any(m <= s for s, m in zip(expected_signals, expected_maturities))):
            errors.append("invalid_frozen_section_schedule")
    except (KeyError, TypeError, ValueError):
        errors.append("invalid_frozen_section_schedule")
        expected_signals, expected_maturities = [], []
    if freeze.get("selection_changed_after_freeze"):
        errors.append("selection_change_requires_new_freeze")
    if errors:
        return {"status": "precheck_rejected", "reasons": errors, "matured_sections": 0,
                "sections_required": required}
    seen = set()
    accepted = []
    for row in evidence:
        try:
            signal = _iso_date(row["signal_date"])
            mature = _iso_date(row["label_matured_on"])
            read_at = _iso_date(row["data_as_of"])
        except (KeyError, TypeError, ValueError):
            errors.append("invalid_section_time")
            continue
        if signal in seen:
            errors.append("duplicate_signal_date")
        seen.add(signal)
        if signal < first_signal or signal <= ceiling or mature <= signal:
            errors.append("section_outside_frozen_clock")
        if mature > now or read_at < mature or read_at > now:
            errors.append("unmatured_or_inconsistent_data_read")
        if row.get("candidate_id") != freeze["candidate_id"]:
            errors.append("candidate_identity_mismatch")
        if row.get("evaluation_contract_sha256") != freeze["evaluation_contract_sha256"]:
            errors.append("contract_identity_mismatch")
        if row.get("economic_status") != "valid" or not _hash(row.get("independent_book_sha256")):
            errors.append("section_economic_book_unverified")
        if not isinstance(row.get("access_log_sequence"), int) or row["access_log_sequence"] < 1:
            errors.append("future_access_not_logged")
        accepted.append(signal)
        schedule_pos = len(accepted) - 1
        if (schedule_pos >= len(expected_signals)
                or signal != expected_signals[schedule_pos]
                or mature != expected_maturities[schedule_pos]):
            errors.append("section_not_next_in_frozen_schedule")
    if errors:
        return {"status": "precheck_rejected", "reasons": sorted(set(errors)),
                "matured_sections": 0, "sections_required": required}
    count = len(accepted)
    return {"status": "ready_for_independent_adjudication" if count >= required else "not_yet_matured",
            "reasons": [] if count >= required else ["future_sections_below_frozen_requirement"],
            "matured_sections": count, "sections_required": required,
            "first_signal_date": first_signal.isoformat(), "as_of": now.isoformat()}


def _index(rows: list[dict], keys: tuple[str, ...]) -> tuple[dict, list[str]]:
    indexed = {}
    errors = []
    for row in rows:
        try:
            key = tuple(row[k] for k in keys)
        except KeyError:
            errors.append("missing_key")
            continue
        if key in indexed:
            errors.append("duplicate_key")
        indexed[key] = row
    return indexed, errors


def compare_execution_books(local: dict, jq: dict, contract: dict) -> dict:
    """Structural per-day/account/trade/position comparison, never an S5 verdict.

    Both exports must refer to different actual engines and contain a full
    predeclared calendar.  No daily row or position/trade can silently vanish.
    """
    reasons = []
    for field in ("candidate_id", "strategy_sha256", "calendar_sha256", "cost_bps",
                  "start_date", "end_date", "initial_cash", "price_adjustment",
                  "execution_timing"):
        if local.get(field) != contract.get(field) or jq.get(field) != contract.get(field):
            reasons.append("contract_mismatch:" + field)
    if local.get("engine") != "local_fixed_share" or jq.get("engine") != "jq_platform":
        reasons.append("engine_identity_missing")
    if not local.get("run_id") or not jq.get("run_id") or local.get("run_id") == jq.get("run_id"):
        reasons.append("independent_run_ids_missing")
    dates = contract.get("session_dates")
    if not isinstance(dates, list) or not dates or len(set(dates)) != len(dates):
        reasons.append("invalid_frozen_session_calendar")
        dates = []
    if reasons:
        return {"status": "precheck_rejected", "reasons": reasons, "compared_days": 0}
    tables = {}
    for label, book in (("local", local), ("jq", jq)):
        tables[label] = {}
        for table, keys in (("daily", ("date",)), ("trades", ("date", "code", "side", "sequence")),
                            ("positions", ("date", "code"))):
            values = book.get(table)
            if not isinstance(values, list):
                reasons.append("missing_table:" + label + ":" + table)
                values = []
            indexed, table_errors = _index(values, keys)
            reasons.extend(label + ":" + table + ":" + item for item in table_errors)
            tables[label][table] = indexed
    expected = {(d,) for d in dates}
    for label in ("local", "jq"):
        if set(tables[label]["daily"]) != expected:
            reasons.append(label + ":daily_calendar_incomplete")
    if reasons:
        return {"status": "data_insufficient", "reasons": sorted(set(reasons)), "compared_days": 0}
    for table in ("trades", "positions"):
        if set(tables["local"][table]) != set(tables["jq"][table]):
            reasons.append(table + "_keys_differ")
        for label in ("local", "jq"):
            if any(key[0] not in dates for key in tables[label][table]):
                reasons.append(label + ":" + table + "_outside_calendar")
    numeric = {"daily": ("cash", "receivables", "fees", "nav"),
               "trades": ("shares", "price", "fee", "cash_delta"),
               "positions": ("shares", "mark_price", "market_value")}
    max_abs = Counter()
    for table, fields in numeric.items():
        for key in set(tables["local"][table]) & set(tables["jq"][table]):
            a, b = tables["local"][table][key], tables["jq"][table][key]
            for field in fields:
                try:
                    x, y = float(a[field]), float(b[field])
                except (KeyError, TypeError, ValueError):
                    reasons.append(table + ":invalid_" + field)
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    reasons.append(table + ":nonfinite_" + field)
                    continue
                delta = abs(x - y)
                max_abs[table + "." + field] = max(max_abs[table + "." + field], delta)
                tolerance = (0.0 if field == "shares" else
                             1e-6 if field in ("price", "mark_price") else 0.01)
                if delta > tolerance:
                    reasons.append(table + ":mismatch_" + field)
    for label in ("local", "jq"):
        book = tables[label]
        for day in dates:
            row = book["daily"][(day,)]
            try:
                asset_value = sum(float(position["market_value"])
                                  for key, position in book["positions"].items() if key[0] == day)
                discrepancy = abs(float(row["nav"]) - float(row["cash"])
                                  - float(row["receivables"]) - asset_value)
                if not math.isfinite(discrepancy) or discrepancy > 0.01:
                    reasons.append(label + ":nav_arithmetic_mismatch")
            except (KeyError, TypeError, ValueError):
                reasons.append(label + ":nav_arithmetic_unreadable")
    return {"status": "structural_parity_pass" if not reasons else "parity_failed",
            "reasons": sorted(set(reasons)), "compared_days": len(dates),
            "max_abs_difference": dict(max_abs),
            "note": "Platform source, event provenance and independent execution still require external verification."}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    print(json.dumps(assess_candidate_freeze(freeze, args.as_of, evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
