"""Quality, distribution and account drift are different operational signals."""

from statistics import mean
from .contracts import require
from .statistics import quantile


def drift_report(reference, current, account_results=()):
    from .columnar import ScoreGrid
    if isinstance(reference, ScoreGrid):
        from .full_market_support import full_market_drift
        return full_market_drift(reference, current, account_results)
    a = [v for row in reference.values() for v in row.values() if v is not None]
    b = [v for row in current.values() for v in row.values() if v is not None]
    total_a, total_b = sum(map(len, reference.values())), sum(map(len, current.values()))
    require(total_a > 0 and total_b > 0, "empty drift comparison")
    report = {"reference_coverage": len(a)/total_a, "current_coverage": len(b)/total_b,
              "mean_shift": mean(b)-mean(a) if a and b else None,
              "reference_quantiles": [quantile(a, q) for q in (.1,.5,.9)] if a else None,
              "current_quantiles": [quantile(b, q) for q in (.1,.5,.9)] if b else None,
              "account_unidentifiable": sum(r["eval_status"] != "valid" for r in account_results),
              "automatic_historical_champion_switch": False,
              "recommended_action": "research_review"}
    if report["account_unidentifiable"]:
        report["recommended_action"] = "halt_new_orders_and_reconcile_if_active"
    return report


def pareto_archive(items):
    """Preserve research tradeoffs; archive membership conveys no eligibility."""
    eligible = [i for i in items if i.get("valid_reading") and i.get("quality") is not None]
    result = []
    for item in eligible:
        dominated = any(other["quality"] >= item["quality"] and other["complexity"] <= item["complexity"]
                        and (other["quality"] > item["quality"] or other["complexity"] < item["complexity"])
                        for other in eligible if other is not item)
        if not dominated:
            result.append(item)
    return result


def niche_descriptors(item):
    """Deterministic ecological descriptor: (complexity bucket, structural family, behavior cluster).

    Window variants of one structure share a family and renamings of one behavior
    share a cluster, so neither can flood the diversity pool; membership conveys
    no eligibility and no reward.
    """
    complexity = item.get("complexity")
    bucket = min(int(complexity), 30) // 3 if isinstance(complexity, int) and complexity >= 0 else -1
    return (bucket, str(item.get("structural_family") or ""), str(item.get("behavior_hash") or ""))


def diversity_representatives(items, k):
    """One best member per non-empty niche, spread evenly when niches outnumber k.

    Selection helper only: complements the Pareto archive, never replaces it and
    never changes thresholds or grants eligibility.
    """
    require(type(k) is int and k >= 1, "positive representative count required")
    eligible = [i for i in items if i.get("valid_reading") and i.get("quality") is not None]
    groups = {}
    for item in eligible:
        groups.setdefault(niche_descriptors(item), []).append(item)
    if not groups:
        return []
    niches = sorted(groups)
    reps = [max(groups[n], key=lambda i: (i["quality"], -i["complexity"], str(i.get("candidate_id"))))
            for n in niches]
    if len(reps) > k:
        if k == 1:
            reps = [reps[0]]
        else:
            last = len(reps) - 1
            reps = [reps[(i * last) // (k - 1)] for i in range(k)]
    return reps


def select_parents(archive, policy):
    """Parent pool by declared policy; 'pareto' reproduces the historical behavior exactly."""
    require(policy in {"pareto", "niche_mix"}, "unknown parent policy")
    frontier = pareto_archive(archive)
    if policy == "pareto":
        return frontier
    merged = {i["candidate_id"]: i for i in frontier + diversity_representatives(archive, k=8)}
    return list(merged.values())
