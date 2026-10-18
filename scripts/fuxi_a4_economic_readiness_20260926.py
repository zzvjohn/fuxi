"""Fail-closed readiness check for a new equal-budget economic challenge."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/factor_alchemy"))
from run_contract import FuxiWriterLock

PROBES = {
    "v2": ROOT / "reports/fuxi_a1_source_audit_20260926/strict_probe_v2.json",
    "v3": ROOT / "reports/fuxi_a1_source_audit_20260926/strict_probe_v3.json",
    "amount_top80": ROOT / "reports/fuxi_a1_source_audit_20260926/strict_probe_amount_top80_extension1.json",
}
TRACE_DIR = ROOT / "reports/fuxi_a1_source_audit_20260926"
A4 = ROOT / "reports/fuxi_a4_20260926/generator_challenge_v2.json"
OUT = ROOT / "reports/fuxi_a4_economic_readiness_20260926.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess():
    frozen = load(A4)
    if frozen["matched_seed_count"] != 5 or len(frozen["rows"]) != 5:
        raise RuntimeError("old engineering comparison no longer matches 5-seed archive")
    arms = {}
    inputs = [A4]
    for arm, probe_path in PROBES.items():
        trace_path = TRACE_DIR / f"held_event_trace_{arm}_v5.json"
        probe, trace = load(probe_path), load(trace_path)
        inputs += [probe_path, trace_path]
        if probe["expected_days"] != 320 or trace["expected_days"] != 320:
            raise RuntimeError("frozen A1 calendar mismatch")
        arms[arm] = {
            "strict_economic_status": probe["economic_status"],
            "strict_finite_nav_days": probe["finite_nav_days"],
            "strict_expected_days": probe["expected_days"],
            "sensitivity_book_status": trace["book_assessment"]["status"],
            "sensitivity_excluded_action_days": len(trace["held_exclusions"]),
            "sensitivity_fractional_events": len(trace["fractional_retirements"]),
            "independent_execution_or_data_validation":
                probe["assessment"]["independent_execution_or_data_validation"],
        }
    valid = all(item["strict_economic_status"] == "valid" and
                item["strict_finite_nav_days"] == 320 and
                item["independent_execution_or_data_validation"]
                for item in arms.values())
    archived_full_candidate_table = all(
        isinstance(row.get(name, {}).get("candidates"), list)
        for row in frozen["rows"]
        for name in ("forge", "quantum", "same_grammar_random", "same_decoder_eda")
    )
    reasons = []
    if not valid:
        reasons.append("no_complete_independently_verified_trading_label")
    if not archived_full_candidate_table:
        reasons.append("old_a4_archive_has_champions_not_full_candidate_denominator")
    if frozen["economic_status"] != "valid":
        reasons.append("old_a4_is_legacy_fitness_engineering_only")
    reasons.append("old_search_panel_includes_research_exposed_2025_2026")
    return {
        "schema": "fuxi_a4_economic_readiness/v1",
        "status": "ready" if not reasons else "blocked_data_insufficient",
        "reasons": reasons,
        "frozen_a1_arms": arms,
        "a4_matched_seeds": frozen["matched_seed_count"],
        "a4_full_candidate_archive_available": archived_full_candidate_table,
        "valid_economic_pair_count": 0,
        "formal_generator_winner": None,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
        "next_protocol": "new, separately frozen four-arm batch after valid labels; do not amend A4 v2",
    }


def main():
    with FuxiWriterLock(run_id="a4_economic_readiness_20260926"):
        if OUT.exists():
            raise RuntimeError("readiness artifact already sealed")
        result = assess()
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "reasons": result["reasons"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
