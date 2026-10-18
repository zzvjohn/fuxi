"""G0 precheck, G1 diagnostics and G2 shortlist quota (P1, v2.0 landing).

R2 blueprint section 6, landed as routing-only machinery (no qualification):

* G0 decides whether the expression can be legally computed and ends invalid
  work cheaply: field/unit/operator support, lookback vs data depth, field
  availability at decision time, missing/denominator rules, label maturity and
  expression identity. States: valid / precheck_rejected (capability) /
  data_insufficient (wait for data) / error (tool failure).
* G1 computes one diagnostic pass (directional daily RankIC, unannualized ICIR
  = mean/sample-SD, null on zero variance; coverage; stability; same-support
  similarity; turnover proxy) and emits routing hints only. IC thresholds are
  ranking hints, never shared hard admission.
* G2 allocates account seats from a cumulative cohort quota
  ``Q_total=min(8,max(1,ceil(0.2N)))`` with exploration carryover. Candidates
  that never receive a seat stay ``not_evaluated`` (explicit), never rejected.
"""

from .contracts import ContractError, digest, keys, require
from .routing import FIVE_STATES, validate_screen_result

G0_RESULT_SCHEMA = "fuxi.g0_result.1"
G1_RESULT_SCHEMA = "fuxi.g1_result.1"


# ---------------------------------------------------------------- G0 precheck

def g0_precheck(candidate_spec, surface):
    """Cheap legality check. ``surface`` describes the available data surface:
    {fields: set, min_session_count, label_mature_at, supported_operators: set,
    unit_rules: dict}. Returns a G0 result dict; never a qualification."""
    require(isinstance(candidate_spec, dict) and isinstance(surface, dict), "invalid inputs")
    fields = surface.get("fields") or set()
    operators = surface.get("operators") or set()
    units = surface.get("units") or {}
    for field in candidate_spec.get("required_fields", []):
        if field not in fields:
            return _g0("data_insufficient", "field_missing_wait_for_data",
                       detail={"field": field})
        required_unit = candidate_spec.get("field_units", {}).get(field)
        if required_unit and units.get(field) not in (None, required_unit):
            return _g0("precheck_rejected", "unit_mismatch", detail={"field": field})
    for operator in candidate_spec.get("operators", []):
        if operator not in operators:
            return _g0("precheck_rejected", "operator_unsupported", detail={"operator": operator},
                       capability_status="unsupported")
    lookback = candidate_spec.get("lookback")
    depth = surface.get("min_session_count")
    if lookback is not None and depth is not None and lookback > depth:
        return _g0("data_insufficient", "lookback_exceeds_data_depth",
                   detail={"lookback": lookback, "depth": depth})
    for field, available_at in candidate_spec.get("available_at_rules", {}).items():
        decision_at = surface.get("decision_time")
        if decision_at and available_at and available_at > decision_at:
            return _g0("data_insufficient", "field_not_available_at_decision_time",
                       detail={"field": field})
    if candidate_spec.get("denominator_field") and candidate_spec["denominator_field"] not in fields:
        return _g0("data_insufficient", "denominator_field_missing",
                   detail={"field": candidate_spec["denominator_field"]})
    label_mature_at = surface.get("label_mature_at")
    horizon_end = candidate_spec.get("label_horizon_end")
    if label_mature_at and horizon_end and horizon_end > label_mature_at:
        return _g0("data_insufficient", "label_not_mature",
                   detail={"mature_at": label_mature_at, "horizon_end": horizon_end})
    return _g0("valid", None)


def _g0(status, reason_code, *, detail=None, capability_status=None):
    result = {"schema_version": G0_RESULT_SCHEMA, "eval_status": status,
              "reason_code": reason_code, "detail": detail or {},
              "capability_status": capability_status}
    require(result["eval_status"] in FIVE_STATES, "invalid G0 status")
    digest(result)
    return result


# ---------------------------------------------------------------- G1 diagnostic

def g1_diagnostics(daily_ic, *, daily_support, existing_asset_similarity=None,
                   turnover_proxy=None, stability_window=None):
    """One diagnostic pass. ``daily_ic`` is a list of daily cross-section RankIC
    values (floats, may be empty when sections were constant). ICIR is the
    sample mean / sample SD of the daily series; null on zero variance or empty
    series. Nothing here is a shared hard admission."""
    require(isinstance(daily_ic, list) and isinstance(daily_support, list)
            and len(daily_ic) == len(daily_support), "daily series must align")
    for value in daily_ic:
        require(value is None or (type(value) is float and value == value), "invalid IC value")
    clean = [v for v in daily_ic if v is not None]
    n = len(clean)
    if n == 0:
        return {"schema_version": G1_RESULT_SCHEMA, "rank_ic_mean": None,
                "icir_unannualized": None, "n_sections": len(daily_ic),
                "coverage_mean": None, "constant_sections": len(daily_ic),
                "reason_code": "constant_cross_section", "max_similarity": None,
                "turnover_proxy": turnover_proxy,
                "stability_window": stability_window, "support_daily": list(daily_support)}
    mean = sum(clean) / n
    variance = sum((v - mean) ** 2 for v in clean) / (n - 1) if n > 1 else None
    icir = mean / (variance ** 0.5) if variance else None
    coverage = sum(daily_support) / max(1, len(daily_support))
    return {"schema_version": G1_RESULT_SCHEMA, "rank_ic_mean": mean,
            "icir_unannualized": icir, "n_sections": n,
            "coverage_mean": coverage, "constant_sections": 0,
            "reason_code": None, "max_similarity": existing_asset_similarity,
            "turnover_proxy": turnover_proxy,
            "stability_window": stability_window, "support_daily": list(daily_support)}


# ---------------------------------------------------------------- G2 quota

def g2_seats(cohort_n, allocated, account_budget_remaining, *, exploration_state=None):
    """Cumulative cohort quota. ``cohort_n`` = unique valid candidates ever
    routable in this cohort; ``allocated`` = seats already handed out (across
    microbatches); ``account_budget_remaining`` = remaining account runs.
    Returns seats, exploration_seats and notes."""
    require(type(cohort_n) is int and cohort_n >= 0, "cohort_n must be >= 0")
    require(type(allocated) is int and allocated >= 0, "allocated must be >= 0")
    require(type(account_budget_remaining) is int and account_budget_remaining >= 0,
            "budget must be >= 0")
    q_total = 0 if cohort_n == 0 else min(8, max(1, (cohort_n * 2 + 9) // 10))  # ceil(0.2N)
    new_seats = max(0, q_total - allocated)
    new_seats = min(new_seats, account_budget_remaining)
    state = dict(exploration_state or {})
    cumulative = state.get("cumulative_allocated", 0)
    exploration_target = (cumulative + new_seats) // 5
    exploration_granted = min(new_seats, max(0, exploration_target - state.get("exploration_granted", 0)))
    notes = {"q_total": q_total, "budget_capped": new_seats < max(0, q_total - allocated)}
    state.update({"cumulative_allocated": cumulative + new_seats,
                  "exploration_granted": state.get("exploration_granted", 0) + exploration_granted})
    return {"seats": new_seats, "exploration_seats": exploration_granted,
            "state": state, "notes": notes}


class G2ShortlistQueue:
    """Shared shortlist -> account-seat queue with explicit not_evaluated.

    Candidates that route to G2 but never receive a seat keep
    ``eval_status="not_evaluated"``; the queue never converts that into a
    rejection. A frozen baseline's account result is cached by baseline ref.
    """

    def __init__(self):
        self.pending = []       # (candidate_id, route_record)
        self.seated = []        # candidate_ids that received seats
        self.baseline_cache = {}

    def submit(self, candidate_id, route_record):
        validate_screen_result(route_record)
        require(route_record["candidate_id"] == candidate_id, "candidate id mismatch")
        self.pending.append((candidate_id, route_record))

    def allocate(self, seats):
        require(type(seats) is int and seats >= 0, "invalid seats")
        chosen = self.pending[:seats]
        self.pending = self.pending[seats:]
        self.seated.extend(cid for cid, _ in chosen)
        return [cid for cid, _ in chosen]

    def unseated(self):
        """Explicit not_evaluated disclosure for candidates without seats."""
        return [{"candidate_id": cid, "eval_status": "not_evaluated",
                 "reason_code": "account_quota_not_allocated",
                 "route": route["decision"]} for cid, route in self.pending]

    def cache_baseline(self, baseline_ref, account_result):
        self.baseline_cache[baseline_ref] = account_result
        return account_result

    def get_baseline(self, baseline_ref):
        return self.baseline_cache.get(baseline_ref)
