"""Freeze measurable research objectives before candidate search; calibrate only on declared data."""

import math
from statistics import stdev
from .contracts import digest, require, timestamp, keys
from .statistics import required_sections, quantile
from .evaluation import factor_statistics, DEFAULT_POLICY


def calibration_request(snapshot, *, calibration_end, research_start, minimum_detectable_ic=.03,
                        alpha=.05, power=.8, max_trials=30, horizon=5):
    require(calibration_end < research_start and calibration_end in snapshot.sessions and research_start in snapshot.sessions,
            "calibration/research windows must be distinct and chronological")
    # Bootstrap replicates must resolve the two-sided Bonferroni-corrected tail.
    # screen() (history_research_v2) runs block_inference_fast with alpha/max_trials
    # and reports tail_draws_expected=(replicates+1)*alpha/(2*max_trials); a
    # tail-resolution diagnostic of >=20 draws requires replicates >= 40*max_trials/alpha.
    # reachability.py additionally requires tail_mass >= 1 (satisfied at a far lower
    # replicate count). Derive replicates from the trial budget, keeping 199 as the
    # historical floor so single-candidate admission stays identical.
    bootstrap_replicates = max(199, math.ceil(40 * max_trials / alpha))
    return {"schema_version": "fuxi.calibration_request.1", "snapshot_hash": snapshot.hash,
            "calibration_start": snapshot.sessions[snapshot.start], "calibration_end": calibration_end,
            "research_start": research_start, "research_end": snapshot.sessions[-1],
            "minimum_detectable_ic": minimum_detectable_ic, "alpha": alpha, "power": power,
            "max_trials": max_trials, "block_length": max(2, horizon+1), "bootstrap_replicates": bootstrap_replicates,
            "research_gate_version": "history_research_v2",
            "block_grid": [6, 12, 24],
            "screen_rules": {"screen_min_pairs": 10, "screen_min_sessions": 120,
                             "screen_pair_coverage": 0.8, "screen_positive_year_fraction": 0.6,
                             "screen_positive_exchange_count": 2},
            "minimum_net_increment_daily": 0., "cost_stress_multiplier": 2., "capacity_stress_multiplier": .5,
            "multiple_search_scope": "adaptive_research_not_confirmatory",
            "formal_use_requires": "separate_confirmatory_protocol_and_certified_account_data"}


def register_calibration(store, campaign_id, request, baseline_scores, snapshot, labels, research_clock):
    keys(request, calibration_request(snapshot, calibration_end=request["calibration_end"],
                                     research_start=request["research_start"]).keys())
    require(request["snapshot_hash"] == snapshot.hash and request["max_trials"] > 0, "calibration identity mismatch")
    existing = [p for _, p in store.facts("research_policy") if p["campaign_id"] == campaign_id]
    if existing:
        require(existing[0]["request"] == request, "cannot adjust thresholds after research feedback")
        return existing[0]
    # The request is an immutable fact before extracting calibration outcome statistics.
    request_id = store.append("calibration_request", request, campaign_id+":calibration_request")
    selected = {d: row for d, row in baseline_scores.items() if request["calibration_start"] <= d <= request["calibration_end"]}
    result = factor_statistics(selected, snapshot, labels, DEFAULT_POLICY, request["calibration_end"]+"T23:59:59+08:00")
    values = [r["rank_ic"] for r in result["diagnostics"]["sections"] if r["rank_ic"] is not None]
    require(len(values) >= max(10, 2*request["block_length"]), "calibration needs more mature sections")
    sigma = stdev(values)
    need = required_sections(sigma, request["minimum_detectable_ic"],
                             request["alpha"]/request["max_trials"], request["power"], request["block_length"])
    profile = {"schema_version": "fuxi.research_policy.1", "campaign_id": campaign_id, "request_id": request_id,
               "request": request, "calibration_observations": len(values), "calibration_sigma": sigma,
               "minimum_effective_sections": need,
               "status": "calibrated_research_profile", "confirmatory_error_control_validated": False,
               "threshold_interpretation": "power-planning approximation; adaptive-search inference is diagnostic",
               "calibration_evaluation_hash": digest(result), "frozen_at": store.get(campaign_id+":config")["registered_at"]}
    profile["policy_hash"] = digest(profile)
    store.append("research_policy", profile, campaign_id+":policy")
    return profile


def register_confirmatory_protocol(store, spec):
    """Privileged operator/acceptance API, outside the generator action vocabulary."""
    keys(spec, ("schema_version", "policy_hash", "candidate_ids", "baseline_hash", "scope", "frozen_at",
                "minimum_sections", "alpha", "block_length", "bootstrap_replicates", "look_schedule",
                "cost_profile_hash", "minimum_net_increment_daily", "calibration_evidence_id"))
    require(spec["schema_version"] == "fuxi.confirmatory_protocol.1", "protocol schema mismatch")
    require(spec["candidate_ids"] and len(set(spec["candidate_ids"])) == len(spec["candidate_ids"]), "candidate family required")
    require(spec["minimum_sections"] >= 2*spec["block_length"] and 0 < spec["alpha"] < 1,
            "invalid confirmation design")
    require(spec["look_schedule"] and len(set(spec["look_schedule"])) == len(spec["look_schedule"]), "fixed look schedule required")
    require(spec["look_schedule"] == sorted(spec["look_schedule"],key=timestamp), "look schedule must be chronological")
    require(type(spec["block_length"]) is int and spec["block_length"] > 0, "invalid confirmation block length")
    require(spec["bootstrap_replicates"] >= len(spec["candidate_ids"])*len(spec["look_schedule"])/spec["alpha"],
            "bootstrap resolution too low for registered family/looks")
    from .evidence import overlaps
    overlaps(spec["scope"],spec["scope"])
    require(spec["scope"]["intervals"] and all(timestamp(a) > timestamp(spec["frozen_at"])
                for a,b in spec["scope"]["intervals"]+spec["scope"]["label_intervals"]),"forward scope must follow freeze")
    require(all(timestamp(t) > timestamp(spec["frozen_at"]) for t in spec["look_schedule"]), "looks must follow freeze")
    evidence = store.get(spec["calibration_evidence_id"])
    require(evidence.get("kind") == "confirmatory_calibration" and evidence.get("verified") is True
            and evidence.get("policy_hash") == spec["policy_hash"], "confirmatory calibration evidence missing")
    return store.receipt("confirmatory_protocol", spec)
