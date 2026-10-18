"""Evidence gates for what the feedback layer is allowed to steer generation with.

Two defects this module exists to isolate, both found by auditing what the generator
actually consumes rather than what the code intends.

1. Motif rules. `experience_memory` mints a rule once `total_jq >= 3`, with the code
   saying so in its own comment ("v3.1: 降低门槛 5→3, 让负收益更快触发 forbid"). The
   generator then injects those rules as a HARD constraint. A 0-of-3 record is
   consistent with a true success rate as high as 63%. The banned keys are syntax
   patterns ("rolling|mean"), not economic statements, so the ban lands on whole families
   of ordinary factors at once.

2. The FRI evaluator. `compute_fri` returns early with every component left at its
   default when a factor has fewer than 20 IC points. The stored record is then
   `score: 0.0, grade: ''` — indistinguishable from a genuinely bad factor. 614 of 623
   stored attempts have exactly this signature, and downstream code filters at
   `min_fri=0.4`, sorts by the score, renders it into the prompt and looks for
   "three consecutive PASS with FRI > 0.5". So an absence of data silently removes real
   successes from the exemplar pool and permanently prevents the success-pattern rule
   from ever firing.

Nothing here edits history. The stored rules and attempts stay exactly as they are; this
module decides which of them a consumer may treat as evidence, and how strongly.

The sample thresholds are derived, not chosen:

    To rule out a true rate above 25% after seeing 0 successes you need n >= 11,
    because 0.05 ** (1/11) = 0.762 and 1 - 0.762 = 0.238 <= 0.25.
    To rule out a true rate above 10% you need n >= 29.

`min_samples_for_bound` computes that number rather than hard-coding it. For
nonzero observed successes, `binomial_upper_bound` uses the actual success count;
using the zero-success shortcut there silently overstates evidence for a ban.
"""
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

ALPHA = 0.05
GATE_ENV = "FUXI_MOTIF_EVIDENCE_GATE"
DEFAULT_MIN_RULE_SAMPLES = 11
STRONG_MIN_RULE_SAMPLES = 29
BOUND_FOR_DEFAULT_BAR = 0.25

_RATE_RE = re.compile(r'成功率\s*(\d+(?:\.\d+)?)\s*%')
_N_RE = re.compile(r'n\s*=\s*(\d+)')

HARD = "hard"
SOFT = "soft"
CONTRADICTORY = "contradictory"
UNREADABLE = "unreadable"


# ------------------------------------------------------------------ exact bounds

def zero_failure_upper_bound(n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson upper bound on p given 0 successes in n trials."""
    if n <= 0:
        return 1.0
    return 1.0 - alpha ** (1.0 / n)


def binomial_upper_bound(successes: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided exact binomial upper bound for the *observed* success count.

    The zero-success shortcut is valid only when successes is actually zero.  Solve
    P[X <= successes | n, p] = alpha by bisection without a SciPy dependency.
    """
    if n <= 0 or not 0 <= successes <= n or not 0 < alpha < 1:
        raise ValueError("invalid binomial sample or alpha")
    if successes == 0:
        return zero_failure_upper_bound(n, alpha)
    if successes == n:
        return 1.0

    def cdf(p: float) -> float:
        terms = [math.lgamma(n + 1) - math.lgamma(j + 1)
                 - math.lgamma(n - j + 1) + j * math.log(p)
                 + (n - j) * math.log1p(-p) for j in range(successes + 1)]
        scale = max(terms)
        return math.exp(scale) * sum(math.exp(t - scale) for t in terms)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def min_samples_for_bound(rate: float, alpha: float = ALPHA) -> int:
    """Smallest n for which 0 successes rules out a true rate strictly above `rate`."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    return int(math.ceil(math.log(alpha) / math.log(1.0 - rate)))


def all_success_lower_bound(n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson lower bound on p given n successes in n trials."""
    if n <= 0:
        return 0.0
    return alpha ** (1.0 / n)


# ------------------------------------------------------------------ rule reading

def parse_rule_evidence(rule: Dict[str, Any]) -> Optional[Tuple[int, float]]:
    """Recover (n, observed rate) from a stored rule's reason string.

    Returns None when the reason carries no sample size, because a claim with no stated
    n cannot be checked and must not be treated as if it had been.
    """
    reason = str(rule.get("reason", ""))
    n_match = _N_RE.search(reason)
    if not n_match:
        return None
    n = int(n_match.group(1))
    if n <= 0:
        return None
    rate_match = _RATE_RE.search(reason)
    if not rate_match:
        return None
    rate = float(rate_match.group(1)) / 100.0
    if not 0.0 <= rate <= 1.0:
        return None
    return n, rate


def classify_motif_rules(rules: List[Dict[str, Any]],
                         min_samples: int = DEFAULT_MIN_RULE_SAMPLES,
                         alpha: float = ALPHA) -> Dict[str, Any]:
    """Split stored rules into those that can carry their claim and those that cannot.

    A key that is both forbidden and preferred is removed from both sides: the memory
    layer is asserting two opposite things about one pattern, which is a statement about
    the rule generator, not about the pattern.

    Ordering inside each bucket is by sample size, strongest first, so a consumer that
    truncates the list drops the weakest claim rather than whichever was written last.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {"forbid": [], "prefer": []}
    for rule in rules:
        key = rule.get("rule_type")
        if key in by_type:
            by_type[key].append(rule)

    forbid_keys = {r.get("motif_key") for r in by_type["forbid"]}
    prefer_keys = {r.get("motif_key") for r in by_type["prefer"]}
    conflicted = forbid_keys & prefer_keys

    hard, soft, unreadable, contradictory = [], [], [], []
    for kind, group in by_type.items():
        for rule in group:
            key = rule.get("motif_key")
            if key in conflicted:
                contradictory.append({**rule, "conflict": kind})
                continue
            evidence = parse_rule_evidence(rule)
            if evidence is None:
                unreadable.append(rule)
                continue
            n, rate = evidence
            successes = round(n * rate)
            # Stored percentages are rounded. If they cannot represent an integer
            # count at their stated precision, they are unsuitable for a hard ban.
            percentage = _RATE_RE.search(str(rule.get("reason", ""))).group(1)
            precision = len(percentage.partition(".")[2]) if "." in percentage else 0
            rounding_tolerance = 0.5 * 10 ** (-precision) / 100.0
            count_consistent = abs(successes / n - rate) <= rounding_tolerance + 1e-12
            upper = binomial_upper_bound(successes, n, alpha)
            item = {**rule, "samples": n, "observed_rate": rate,
                    "observed_successes": successes,
                    "success_count_consistent": count_consistent,
                    "binomial_upper_bound": upper}
            sufficient = n >= min_samples
            if kind == "forbid":
                sufficient = (sufficient and count_consistent
                              and upper <= BOUND_FOR_DEFAULT_BAR)
            (hard if sufficient else soft).append(item)
    hard.sort(key=lambda r: (-r["samples"], str(r.get("motif_key"))))
    soft.sort(key=lambda r: (-r["samples"], str(r.get("motif_key"))))
    return {
        HARD: hard,
        SOFT: soft,
        CONTRADICTORY: contradictory,
        UNREADABLE: unreadable,
        "stats": {
            "rules_total": len(rules),
            "hard": len(hard),
            "soft": len(soft),
            "contradictory": len(contradictory),
            "unreadable": len(unreadable),
            "contradictory_keys": sorted(k for k in conflicted if k),
            "min_samples": min_samples,
            "alpha": alpha,
            "bound_for_default_bar": BOUND_FOR_DEFAULT_BAR,
        },
    }


def gate_enabled() -> bool:
    """FUXI_MOTIF_EVIDENCE_GATE: enforce (default) | shadow (legacy behaviour, full weight)."""
    return os.environ.get(GATE_ENV, "enforce").strip().lower() not in ("shadow", "off")


# ------------------------------------------------------------------- FRI reading

FRI_EVALUATED = "evaluated"
FRI_UNEVALUATED = "unevaluated"


def fri_evidence_status(fri: Optional[Dict[str, Any]]) -> str:
    """Was this FRI actually computed, or is the stored zero an absence of data?

    New records carry an explicit `status` written by `compute_fri`. Older records do not,
    so the fallback reads the signature a computed FRI can never produce: an early return
    leaves the grade empty while a computed FRI always lands in A/B/C/D. A record with a
    score of zero and no grade is therefore an absence of data, not a verdict of zero.
    """
    if not isinstance(fri, dict):
        return FRI_UNEVALUATED
    explicit = str(fri.get("status", "")).strip()
    if explicit == "computed":
        return FRI_EVALUATED
    if explicit == "insufficient_ic_points":
        return FRI_UNEVALUATED
    if str(fri.get("grade", "")).strip():
        return FRI_EVALUATED
    components = ("precision", "persistence", "consistency", "novelty")
    if any(float(fri.get(k) or 0.0) != 0.0 for k in components):
        return FRI_EVALUATED
    return FRI_UNEVALUATED


def fri_score_for_consumption(fri: Optional[Dict[str, Any]]) -> Optional[float]:
    """The score a consumer may rank or threshold on, or None when there is none."""
    if fri_evidence_status(fri) != FRI_EVALUATED:
        return None
    return float(fri.get("score") or 0.0)


def summarise_fri_evidence(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [a for a in attempts if fri_evidence_status(a.get("fri")) == FRI_EVALUATED]
    unevaluated = [a for a in attempts
                   if fri_evidence_status(a.get("fri")) == FRI_UNEVALUATED]
    by_paradigm_evaluated: Dict[str, int] = {}
    for a in evaluated:
        key = str(a.get("paradigm") or "")
        by_paradigm_evaluated[key] = by_paradigm_evaluated.get(key, 0) + 1
    return {
        "attempts_total": len(attempts),
        "fri_evaluated": len(evaluated),
        "fri_unevaluated": len(unevaluated),
        "evaluated_by_paradigm": by_paradigm_evaluated,
    }
