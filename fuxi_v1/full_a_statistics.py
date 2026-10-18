"""Frozen full-A historical diagnostics, separate from admission and old gates.

One observation is one requested trading-day section, not one stock or one
stock/label pair. Unknown days remain calendar slots. Resampling operates only
on these small time series and never on a flattened securities panel.
"""
import math
from collections import defaultdict
from datetime import date
from statistics import NormalDist, mean, stdev

from .contracts import digest, finite, require


EXCHANGES = ("XSHG", "XSHE", "XBSE")
BOARDS = ("main_sh", "main_sz", "chinext", "star", "bj")


def bootstrap_budget(*, family_size=40, alpha=.05, replicates=159999):
    """Resolution and MC precision, not validity of adaptive multiple testing."""
    require(type(family_size) is int and family_size > 0, "invalid diagnostic family size")
    require(0 < alpha < 1 and type(replicates) is int and replicates >= 99,
            "invalid bootstrap budget")
    corrected = alpha/family_size
    tail = corrected/2
    expected = (replicates+1)*tail
    return {"family_size": family_size, "alpha": alpha, "alpha_per_comparison": corrected,
            "replicates": replicates, "minimum_nonzero_p": 1/(replicates+1),
            "expected_draws_per_corrected_ci_tail": expected,
            "one_sided_resolution_adequate": 1/(replicates+1) <= corrected,
            "two_sided_tail_resolution_adequate": expected >= 1,
            "one_sided_resolution_minimum_replicates": math.ceil(1/corrected)-1,
            "two_sided_resolution_minimum_replicates": math.ceil(1/tail)-1,
            "twenty_draw_tail_minimum_replicates": math.ceil(20/tail)-1,
            "hundred_draw_tail_minimum_replicates": math.ceil(100/tail)-1,
            "relative_tail_mc_se_approx": math.sqrt((1-tail)/expected),
            "precision_target_met": expected >= 100,
            "adaptive_family_error_control_validated": False}


def freeze_method(*, frozen_at=None, family_size=40, replicates=159999,
                  block_lengths=(6, 12, 24), seed=314159, batch_size=128):
    """Return a hash-bound method. Caller saves it before outcome evaluation.

    A None clock is an unregistered template; it must not be called a freeze.
    Smaller simulation budgets are allowed for fixtures and explicitly retain
    their inadequate-resolution flags. No result changes this method.
    """
    require(type(seed) is int and type(batch_size) is int and 1 <= batch_size <= 1024,
            "invalid reproducibility or memory budget")
    blocks = list(block_lengths)
    require(blocks and len(set(blocks)) == len(blocks) and blocks == sorted(blocks)
            and all(type(b) is int and b > 0 for b in blocks), "invalid fixed block grid")
    if frozen_at is not None:
        from .contracts import timestamp
        timestamp(frozen_at)
    spec = {"schema_version": "fuxi.full_a_statistics_method.1", "frozen_at": frozen_at,
            "registered": frozen_at is not None, "seed": seed,
            "bootstrap_batch_size": batch_size, "block_lengths": blocks,
            "bootstrap_replicates": replicates, "family_size": family_size, "alpha": .05,
            "hac_lags": [5, 11, 23], "power": .8, "minimum_detectable_ic": .03,
            "short_calibration_warning_below_valid_days": 120,
            "missing_policy": "preserve_calendar_slots; ratio_of_observed_sums_and_counts",
            "block_selection": "all_frozen_blocks_reported; no_outcome_selected_block",
            "evidence_scope": "research_exposed_adaptive_historical_diagnostic",
            "formal_qualification_allowed": False, "old_gate_changes": False,
            "budget": bootstrap_budget(family_size=family_size, replicates=replicates)}
    spec["method_hash"] = digest(spec)
    return spec


def validate_method(method):
    expected = method.get("method_hash")
    body = {k: v for k, v in method.items() if k != "method_hash"}
    require(expected == digest(body), "frozen statistics method drift")
    reproduced = freeze_method(frozen_at=method["frozen_at"], family_size=method["family_size"],
        replicates=method["bootstrap_replicates"], block_lengths=method["block_lengths"],
        seed=method["seed"], batch_size=method["bootstrap_batch_size"])
    require(method == reproduced, "unsupported statistics interpretation")
    return expected


def _values(values):
    require(isinstance(values, (list, tuple)) and values, "nonempty calendar series required")
    require(all(v is None or (type(v) in (int, float) and finite(v)) for v in values),
            "observations must be finite numbers or explicit unknowns")
    return [None if v is None else float(v) for v in values]


def _interval(draws, alpha):
    import numpy as np
    return {"alpha": alpha, "lower": float(np.quantile(draws, alpha/2)),
            "upper": float(np.quantile(draws, 1-alpha/2))}


def bootstrap_diagnostic(values, *, block_length, family_size=40, alpha=.05,
                         replicates=159999, seed=314159, batch_size=128):
    """Circular fixed blocks, same draws for nominal and corrected sensitivity.

    A None contributes zero to an internal *sum* and zero to its denominator;
    it is never an observed zero IC/return. Empty resampled support refuses
    inference instead of silently deleting the affected draw.
    """
    import numpy as np
    values = _values(values)
    require(type(block_length) is int and block_length > 0, "invalid bootstrap block length")
    require(type(seed) is int and type(batch_size) is int and 1 <= batch_size <= 1024,
            "invalid bootstrap execution budget")
    budget = bootstrap_budget(family_size=family_size, alpha=alpha, replicates=replicates)
    n, valid = len(values), sum(v is not None for v in values)
    observed = [v for v in values if v is not None]
    result = {"eval_status": "data_insufficient", "calendar_slots": n, "n_valid_days": valid,
              "unknown_days": n-valid, "mean": mean(observed) if observed else None,
              "block_length": block_length, "replicates": replicates, "seed": seed,
              "batch_size": batch_size, "budget": budget, "nominal_interval": None,
              "bonferroni_sensitivity_interval": None, "p_one_sided": None,
              "formal_qualification_allowed": False,
              "interpretation": "stationary_time_block_approximation; adaptive_search_not_confirmatory"}
    if valid < 2*block_length or block_length > n:
        result["reason_code"] = "too_few_observed_days_for_fixed_block"
        return result
    mask = np.array([v is not None for v in values], dtype=float)
    x = np.array([0. if v is None else v for v in values], dtype=float)
    avg = float(x.sum()/valid)

    def circular_sums(array, length):
        if not length:
            return np.zeros(n)
        extended = np.r_[array, array[:length-1]]
        cumulative = np.r_[0., np.cumsum(extended)]
        return cumulative[length:]-cumulative[:-length]

    full, remainder = divmod(n, block_length)
    sums, counts = circular_sums(x, block_length), circular_sums(mask, block_length)
    tail_sums, tail_counts = circular_sums(x, remainder), circular_sums(mask, remainder)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    empty_draws = 0
    for lo in range(0, replicates, batch_size):
        size = min(batch_size, replicates-lo)
        starts = rng.integers(0, n, size=(size, full+bool(remainder)))
        numerator = sums[starts[:, :full]].sum(axis=1)
        denominator = counts[starts[:, :full]].sum(axis=1)
        if remainder:
            numerator += tail_sums[starts[:, -1]]
            denominator += tail_counts[starts[:, -1]]
        empty = denominator <= 0
        empty_draws += int(empty.sum())
        draws[lo:lo+size] = np.divide(numerator, denominator,
            out=np.full(size, np.nan), where=~empty)
    result["empty_support_bootstrap_draws"] = empty_draws
    if empty_draws:
        result["reason_code"] = "bootstrap_contains_unknown_only_draws"
        return result
    p = (1+int((draws-avg >= avg).sum()))/(replicates+1)
    result.update(eval_status="valid", reason_code="historical_time_block_diagnostic",
        nominal_interval=_interval(draws, alpha),
        bonferroni_sensitivity_interval=_interval(draws, alpha/family_size),
        p_one_sided=p, p_monte_carlo_se=math.sqrt(p*(1-p)/(replicates+1)),
        mean=avg)
    return result


def calendar_values(rows, *, value_key="rank_ic", sessions=None):
    """Normalize explicit rows to the full frozen calendar; absent rows stay None."""
    require(isinstance(rows, list), "daily rows must be a list")
    by_day = {}
    for row in rows:
        day = row.get("session")
        require(isinstance(day, str) and len(day) == 10 and day[4] == day[7] == "-",
                "invalid decision session")
        try:
            date.fromisoformat(day)
        except ValueError:
            require(False, "invalid calendar date")
        require(day not in by_day and value_key in row, "duplicate session or missing value field")
        by_day[day] = row[value_key]
    dates = list(sessions) if sessions is not None else sorted(by_day)
    require(dates and dates == sorted(set(dates)), "nonempty unique chronological calendar required")
    require(set(by_day) <= set(dates), "observed day outside frozen calendar")
    return dates, _values([by_day.get(d) for d in dates])


def analyze_daily_rows(rows, *, value_key="rank_ic", sessions=None, method):
    validate_method(method)
    dates, values = calendar_values(rows, value_key=value_key, sessions=sessions)
    observed = [v for v in values if v is not None]
    blocks = [bootstrap_diagnostic(values, block_length=b, family_size=method["family_size"],
        alpha=method["alpha"], replicates=method["bootstrap_replicates"],
        seed=method["seed"]+104729*b, batch_size=method["bootstrap_batch_size"])
        for b in method["block_lengths"]]
    return {"schema_version": "fuxi.full_a_series_diagnostic.1", "method_hash": method["method_hash"],
            "method_registered": method["registered"],
            "value_key": value_key, "window": [dates[0], dates[-1]], "calendar_slots": len(dates),
            "valid_days": len(observed), "unknown_days": len(dates)-len(observed),
            "mean": mean(observed) if observed else None,
            "sample_sd": stdev(observed) if len(observed) >= 2 else None,
            "blocks": blocks,
            "nominal_positive_all_fixed_blocks": all(b["eval_status"] == "valid" and
                b["nominal_interval"]["lower"] > 0 for b in blocks),
            "corrected_sensitivity_positive_all_fixed_blocks": all(b["eval_status"] == "valid" and
                b["bonferroni_sensitivity_interval"]["lower"] > 0 for b in blocks),
            "adaptive_family_error_control_validated": False,
            "formal_qualification_allowed": False, "old_gate_decision_changed": False}


def calibration_planning(values, *, effect=.03, alpha=.05, power=.8,
                         family_size=40, lags=(5, 11, 23), short_window_days=120):
    """HAC ratio-mean planning, using calendar positions including missing days.

    The influence contribution for an unobserved day is zero, not a zero IC.
    Consistency assumes stationarity of the joint observation-mask/value process;
    informative missingness, regime shifts and short windows remain limitations.
    Required sections are calendar days under the fitted approximation, never an
    admission criterion or a promise that future evidence will be adequate.
    """
    import numpy as np
    values = _values(values)
    require(effect > 0 and 0 < alpha < 1 and 0 < power < 1 and
            type(family_size) is int and family_size > 0, "invalid power planning objective")
    require(lags and all(type(k) is int and k > 0 for k in lags), "invalid HAC sensitivity lags")
    observed = [v for v in values if v is not None]
    n, valid = len(values), len(observed)
    output = {"schema_version": "fuxi.full_a_calibration_planning.1", "calendar_slots": n,
              "calibration_observations": valid, "unknown_days": n-valid,
              "calibration_sigma": stdev(observed) if valid >= 2 else None,
              "effect": effect, "alpha": alpha, "power": power, "family_size": family_size,
              "short_calibration_window": valid < short_window_days,
              "short_window_warning_below_valid_days": short_window_days,
              "estimates": [], "is_admission_gate": False, "formal_qualification_allowed": False,
              "assumptions": "stationary joint mask/value ratio mean; calibration and regime uncertainty remain"}
    if valid < 2:
        output["planning_status"] = "data_insufficient"
        return output
    center = mean(observed)
    fraction = valid/n
    influence = np.array([0. if v is None else (v-center)/fraction for v in values])
    gamma0 = float(np.dot(influence, influence)/n)
    observed_variance = sum((v-center)**2 for v in observed)/valid
    z = NormalDist().inv_cdf(1-alpha/family_size)+NormalDist().inv_cdf(power)
    for lag in lags:
        if lag >= n:
            output["estimates"].append({"lag": lag, "status": "data_insufficient",
                "reason_code": "lag_exceeds_calendar_support", "required_calendar_sections_approx": None})
            continue
        lrv = gamma0
        pair_counts = []
        for k in range(1, lag+1):
            lrv += 2*(1-k/(lag+1))*float(np.dot(influence[k:], influence[:-k])/n)
            pair_counts.append(sum(values[i] is not None and values[i-k] is not None for i in range(k, n)))
        require(lrv >= -1e-10*max(1., gamma0), "negative Bartlett HAC variance")
        lrv = max(0., lrv)
        degenerate = gamma0 == 0 or lrv == 0
        output["estimates"].append({"lag": lag, "status": "degenerate_variance" if degenerate else "estimated",
            "long_run_variance": lrv, "observed_day_variance": observed_variance,
            "variance_of_mean_approx": lrv/n,
            "dependence_ratio_vs_observed_variance": lrv/observed_variance if observed_variance else None,
            "effective_n_approx": n*observed_variance/lrv if lrv else None,
            "required_calendar_sections_approx": None if degenerate else max(2, math.ceil(z*z*lrv/effect**2)),
            "lag_observed_pair_counts": pair_counts})
    output["planning_status"] = "short_window_uncertain" if output["short_calibration_window"] else "advisory_estimates"
    return output


def _support(sections, *, expected_days=None):
    pairs = total = 0
    valid = []
    for row in sections:
        p, u = row.get("pair_count"), row.get("universe_count")
        require(type(p) is int and type(u) is int and 0 <= p <= u, "invalid support denominator")
        v = row.get("rank_ic")
        require(v is None or (type(v) in (int, float) and finite(v)), "invalid group IC")
        require(v is None or -1 <= v <= 1, "rank IC outside correlation range")
        require(v is None or p >= 3, "IC with fewer than three real pairs")
        pairs += p
        total += u
        if v is not None:
            valid.append(float(v))
    missing = len(set(expected_days or ())-{r['session'] for r in sections})
    return {"status": "not_provided" if not sections else "incomplete_calendar_support" if missing else "observed",
            "reported_days": len(sections), "unreported_calendar_days": missing,
            "requested_calendar_slots": len(expected_days) if expected_days is not None else len(sections),
            "valid_ic_days": len(valid), "unknown_ic_days": len(sections)-len(valid),
            "security_label_pairs": pairs, "eligible_security_day_slots": total,
            "pair_coverage": pairs/total if total else None,
            "pair_coverage_scope": "reported_membership_rows_only; unreported_denominators_unknown",
            "mean_rank_ic": mean(valid) if valid else None,
            "daily_mean_weighting": "equal_observed_days; pairs_are_not_independent_time_samples"}


def summarize_support(sections, *, exchange_sections=None, board_sections=None):
    """Summarize actual per-section group statistics; never substitute global IC.

    Group rows must contain independently calculated within-group rank IC and
    actual membership/pair counts. Omitted groups are explicit not_provided.
    """
    calendar_values(sections)
    years = defaultdict(list)
    for row in sections:
        years[row["session"][:4]].append(row)
    exchange_sections, board_sections = exchange_sections or {}, board_sections or {}
    require(set(exchange_sections) <= set(EXCHANGES) and set(board_sections) <= set(BOARDS),
            "unsupported exchange or board key")
    dates = {row["session"] for row in sections}
    for grouping in (exchange_sections, board_sections):
        for rows in grouping.values():
            if rows:
                calendar_values(rows, sessions=sorted(dates))
    return {"schema_version": "fuxi.full_a_support_diagnostic.1", "overall": _support(sections),
            "years": {year: _support(rows) for year, rows in sorted(years.items())},
            "exchanges": {key: _support(exchange_sections.get(key, []), expected_days=dates) for key in EXCHANGES},
            "boards": {key: _support(board_sections.get(key, []), expected_days=dates) for key in BOARDS},
            "formal_qualification_allowed": False, "interpretation": "descriptive subgroup sensitivity, not independent tests"}


def paired_account_diagnostic(candidate_rows, baseline_rows, *, method,
                              candidate_cost_profile_hash, baseline_cost_profile_hash,
                              value_key="net_return", account_paths_identified=False):
    """Matched after-cost return deltas; unknown account days remain unknown."""
    require(isinstance(candidate_cost_profile_hash, str) and len(candidate_cost_profile_hash) == 64
            and candidate_cost_profile_hash == baseline_cost_profile_hash,
            "candidate/baseline cost profile mismatch")
    require(type(account_paths_identified) is bool, "account identification must be explicit")
    a_dates, a = calendar_values(candidate_rows, value_key=value_key)
    b_dates, b = calendar_values(baseline_rows, value_key=value_key)
    require(a_dates == b_dates, "paired account calendar mismatch")
    rows = [{"session": d, "paired_net_increment": None if x is None or y is None else x-y}
            for d, x, y in zip(a_dates, a, b)]
    require(any(r["paired_net_increment"] is not None for r in rows), "zero observed paired account days")
    output = analyze_daily_rows(rows, value_key="paired_net_increment", sessions=a_dates, method=method)
    output.update(cost_profile_hash=candidate_cost_profile_hash, account_paths_identified=account_paths_identified,
        account_scope="identified_historical_diagnostic" if account_paths_identified else "unidentified_component_or_scenario_diagnostic",
        financial_increment_claim_validated=False)
    return output
