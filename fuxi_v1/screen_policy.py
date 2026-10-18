"""ScreenPolicy bootstrap values for the v2.0 G1 routing layer (R2 blueprint).

STATUS: proposed_not_frozen. This module is a standalone development artefact with
NO consumers on any production path. Nothing here changes current defaults,
qualification, budgets or the v0.7/v1.x screening behaviour. Per the R2 blueprint
(BLUEPRINT.md section 6), G1 is a quota router and these numbers are ranking hints
plus calibration starting points, NOT frozen admission thresholds. A ScreenPolicy
becomes frozen only through the calibration procedure below, and only a frozen
policy may carry gate semantics.

Sources per value:
  user_contract  - existing v0.7 contracts the user committed to keep
                   (ICIR>=0.3 & +IC%>=55% factor validation bar; S5_IR=0.7;
                   COST_BPS=15; MIN_COVER=0.9)
  r2_blueprint   - R2 fusion blueprint starting suggestions (rank IC 0.01/0.02
                   attention hints, ICIR 0.10/0.20 hints, 24m-train/6m-validate
                   rolling, >=2/3 segments, top_n=80 production calibre)
  mainstream     - mainstream A-share practice / literature calibration
                   (HLZ 2016 t-hurdle ladder; McLean-Pontiff 42% decay baseline)
"""

from hashlib import sha256

SCREEN_POLICY_SCHEMA = "fuxi.screen_policy.1"

BOOTSTRAP_SCREEN_POLICY = {
    "schema_version": SCREEN_POLICY_SCHEMA,
    "status": "proposed_not_frozen",
    "target_system": "v2.0",
    "blueprint_ref": "reports/fuxi_v11_fusion_blueprint_20261013/BLUEPRINT.md",
    "g1": {
        "rank_ic_attention_hints": {"low": 0.01, "high": 0.02,
                                    "source": "r2_blueprint",
                                    "semantics": "ranking hints only, no shared hard admission"},
        "icir_unannualized_hints": {"low": 0.10, "high": 0.20,
                                    "source": "r2_blueprint",
                                    "semantics": "daily cross-section IC mean / sample SD; null on zero variance"},
        "user_contract_icir_min": {"value": 0.3, "source": "user_contract",
                                   "semantics": "calibration starting point for regular tasks"},
        "user_contract_plus_ic_pct_min": {"value": 55.0, "source": "user_contract",
                                          "semantics": "direction consistency calibration starting point"},
        "min_coverage_hint": {"value": 0.9, "source": "user_contract",
                              "semantics": "planned date x instrument support coverage hint"},
        "t_hurdle_ladder": {"brackets": [(100, 2.0), (300, 2.5), (1000, 3.0), (None, 3.5)],
                            "source": "mainstream",
                            "semantics": "HLZ 2016 multiple-testing ladder; confirmatory protocols only, not G1 routing"},
        "exploration_quota": {"fraction": 0.2, "carry_over_across_microbatches": True,
                              "source": "r2_blueprint",
                              "semantics": "target share of cumulative account seats; disclose true denominator"},
        "account_quota": {"formula": "Q_total=min(8,max(1,ceil(0.2*N)))", "source": "r2_blueprint",
                          "semantics": "cumulative quota over the declared candidate cohort, not per microbatch"},
    },
    "g2": {
        "rolling_windows": {"train_months": 24, "validate_months": 6, "min_nonoverlap_segments": 3,
                            "source": "r2_blueprint",
                            "semantics": "development route comparison; 2025/2026 are research-exposed, not fresh OOS"},
        "majority_segment_rule": {"positive_segments_needed": "2_of_3", "source": "r2_blueprint",
                                  "semantics": "candidate retention policy, not a significance proof"},
        "composite_ir_reference": {"value": 0.7, "source": "user_contract",
                                   "semantics": "S5_IR reference line for rolling-window composite quality; do not lower"},
        "cost_scenarios": {"base_bps": 15.0, "stress_multipliers": {"double_cost": 2.0, "half_capacity": 0.5},
                           "source": "user_contract",
                           "semantics": "base COST_BPS plus v1.x E4-style stress; old 15bps and native fee schedules are not assumed equivalent",
                           "proxy_map_ref": "fuxi_v1.cost_spec.PROXY_15BPS_MAP",
                           "proxy_map_note": "P0.5: base_bps=15 is the legacy one-time turnover proxy; the native component decomposition (commission/stamp/impact) lives in cost_spec.PROXY_15BPS_MAP and the two protocols are never summed"},
        "top_n": {"value": 80, "weighting": "equal", "source": "r2_blueprint",
                  "semantics": "production calibre, not varied per experiment"},
    },
    "forward": {
        "look_plan": {"count": 6, "frequency_months": 1, "source": "r2_blueprint",
                      "semantics": "pre-registered E5 looks after JQ; E5 is not a precondition for first JQ export"},
        "decay_baseline": {"value": 0.42, "source": "mainstream",
                           "semantics": "McLean-Pontiff post-publication attenuation baseline"},
    },
}


def bootstrap_digest(policy=None):
    """Content digest of the bootstrap policy; used to reference it in run configs."""
    payload = policy if policy is not None else BOOTSTRAP_SCREEN_POLICY
    raw = repr(sorted((k, repr(v)) for k, v in payload.items() if k != "schema_version"))
    return sha256(raw.encode("utf-8")).hexdigest()


def freeze_candidate_policy(calibration_evidence, bootstrap=None):
    """Placeholder for the R2 calibration procedure.

    calibration_evidence must reference: locally exposed development data,
    known baselines, repeated/random controls and measured real costs, plus the
    fixed full-evaluation representative suite (DESIGN_REVIEW section 6). Returns
    a policy marked frozen only when evidence is provided; otherwise refuses.

    This function intentionally does nothing yet; it documents the contract.
    """
    base = dict(bootstrap if bootstrap is not None else BOOTSTRAP_SCREEN_POLICY)
    if not calibration_evidence or not isinstance(calibration_evidence, dict):
        raise ValueError("screen policy freeze requires explicit calibration evidence")
    required_keys = ("representative_suite_ref", "baseline_refs", "cost_measurements",
                     "control_comparisons")
    missing = [k for k in required_keys if k not in calibration_evidence]
    if missing:
        raise ValueError("missing calibration evidence keys: %s" % ", ".join(missing))
    base["status"] = "frozen"
    base["calibration_evidence_refs"] = calibration_evidence
    return base
