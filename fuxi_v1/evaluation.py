"""Descriptive statistics and explicit admission blockers, without uncalibrated alpha gates."""

import math
from statistics import mean, stdev
from .contracts import evaluation, keys, require, timestamp, digest
from .dsl import ranks

DEFAULT_POLICY = {
    "schema_version": "fuxi.policy.1", "policy_id": "bootstrap_diagnostic_v1",
    "calibration_status": "uncalibrated", "purpose": "diagnostic",
    "min_cross_section_pairs": 3, "min_valid_sections": 2,
    "economic_thresholds": None,
    "metric_definition": "daily_pairwise_Pearson_and_Spearman; ICIR=mean/sample_sd; no_annualization",
}


def validate_policy(policy):
    keys(policy, DEFAULT_POLICY.keys())
    require(policy["schema_version"] == "fuxi.policy.1", "unsupported policy version")
    require(policy["policy_id"] and policy["calibration_status"] == "uncalibrated"
            and policy["purpose"] == "diagnostic" and policy["economic_thresholds"] is None,
            "calibrated/formal policy registration is not implemented in kernel 0.1")
    for name, lower in (("min_cross_section_pairs", 3), ("min_valid_sections", 2)):
        require(type(policy[name]) is int and policy[name] >= lower, "invalid sufficiency setting")
    require(policy["metric_definition"] == DEFAULT_POLICY["metric_definition"], "unknown metric semantics")
    return digest(policy)


def correlation(a, b):
    require(len(a) == len(b) and len(a) > 0, "comparison requires matched nonzero pairs")
    # Normalize first to keep finite extreme inputs from overflowing sample moments.
    scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
    if scale_a == 0 or scale_b == 0:
        return None
    aa, bb = [x/scale_a for x in a], [y/scale_b for y in b]
    ma, mb = mean(aa), mean(bb)
    da, db = [x-ma for x in aa], [y-mb for y in bb]
    va, vb = math.fsum(x*x for x in da), math.fsum(y*y for y in db)
    if va == 0 or vb == 0:
        return None
    return max(-1., min(1., math.fsum(x*y for x, y in zip(da, db)) / math.sqrt(va*vb)))


def describe(values):
    if not values:
        return {"mean": None, "sample_sd": None, "icir_nonannualized": None}
    sd = stdev(values) if len(values) >= 2 else None
    return {"mean": mean(values), "sample_sd": sd,
            "icir_nonannualized": mean(values)/sd if sd not in (None, 0) else None}


def factor_statistics(scores, snapshot, labels, policy, research_clock):
    validate_policy(policy)
    if getattr(snapshot, "payload", {}).get("schema_version") == "fuxi.columnar_snapshot.1":
        from .columnar import factor_statistics_columnar
        return factor_statistics_columnar(scores, snapshot, labels, policy, research_clock)
    clock = timestamp(research_clock)
    require(scores and clock >= max(snapshot.clocks[d] for d in scores), "evaluated decisions extend beyond research clock")
    sections, ic, rank_ic = [], [], []
    n_pairs = 0
    for session, values in scores.items():
        members = snapshot.universe[session]
        require(set(values) == set(members), "score/universe mismatch")
        pairs = {s: (values[s], labels.value(session, s, clock)) for s in members}
        pairs = {s: pair for s, pair in pairs.items() if None not in pair}
        n = len(pairs)
        n_pairs += n
        pearson = spearman = None
        if n >= policy["min_cross_section_pairs"]:
            x, y = zip(*pairs.values())
            pearson = correlation(x, y)
            rx = ranks({s: v[0] for s, v in pairs.items()})
            ry = ranks({s: v[1] for s, v in pairs.items()})
            spearman = correlation(list(rx.values()), list(ry.values()))
        if pearson is not None and spearman is not None:
            ic.append(pearson)
            rank_ic.append(spearman)
        sections.append(dict(session=session, universe_count=len(members), pair_count=n,
                             signal_count=sum(v is not None for v in values.values()),
                             mature_label_count=sum(labels.value(session, s, clock) is not None for s in members),
                             pair_coverage=n/len(members), pearson_ic=pearson, rank_ic=spearman))
    diagnostics = dict(sections=sections, valid_section_count=len(ic),
                       requested_section_count=len(sections),
                       pearson=describe(ic), spearman=describe(rank_ic),
                       label_kind="prediction_proxy", uncertainty_tested=False,
                       multiple_testing_corrected=False)
    if len(ic) < policy["min_valid_sections"] or n_pairs == 0:
        return evaluation("E1", "data_insufficient", "too_few_nonconstant_mature_sections",
                          diagnostics=diagnostics, n_pairs=n_pairs)
    return evaluation("E1", "valid", "descriptive_only_no_calibrated_decision",
                      metrics={"pearson": describe(ic), "spearman": describe(rank_ic)},
                      diagnostics=diagnostics, n_pairs=n_pairs)


def admission(evaluations, policy):
    validate_policy(policy)
    require(len({x["stage"] for x in evaluations}) == len(evaluations), "duplicate gate stage")
    from .contracts import validate_evaluation
    for item in evaluations:
        validate_evaluation(item)
    by_stage = {x["stage"]: x for x in evaluations}
    e0 = by_stage.get("E0", {})
    blockers = ["formal_admission_not_implemented_in_kernel_0.1", "policy_uncalibrated",
                "strategy_and_cost_contract_not_validated", "account_evidence_not_validated"]
    for stage in ("E0", "E1", "E2", "E3", "E4"):
        item = by_stage.get(stage, {})
        if item.get("eval_status") != "valid" or item.get("decision") != "pass":
            blockers.append(stage + "_not_accepted")
    return dict(schema_version="fuxi.admission.1",
                jq_diagnostic_allowed=e0.get("eval_status") == "valid" and e0.get("decision") == "pass",
                jq_candidate_allowed=False, forward_acceptance_allowed=False,
                live_execution_allowed=False, blockers=blockers,
                note="E5 is future acceptance, not a prerequisite for the first formal JQ backtest")
