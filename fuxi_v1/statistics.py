"""Time-level paired inference. Fixed schedules and all attempts remain in scope."""

import math
import random
from statistics import mean, stdev, NormalDist
from .contracts import require, finite


def quantile(values, q):
    require(values and 0 <= q <= 1, "empty quantile/invalid probability")
    values = sorted(values)
    p = (len(values)-1)*q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] + (values[hi]-values[lo])*(p-lo)


def block_inference(values, *, block_length, alpha, replicates=999, seed=0):
    """Circular moving-block percentile CI and centered-null one-sided mean test.

    A research diagnostic under stationarity; no claim of exact finite-sample size.
    Caller controls preregistration and adaptive multiple-search interpretation.
    """
    require(values and all(finite(x) for x in values), "matched finite time series required")
    require(type(block_length) is int and block_length > 0 and 0 < alpha < 1,
            "invalid bootstrap protocol")
    require(type(replicates) is int and replicates >= 99, "insufficient bootstrap replicates")
    n, avg = len(values), mean(values)
    if n < 2*block_length:
        return {"status": "data_insufficient", "n_pairs": n, "mean": avg,
                "lower": None, "upper": None, "p_one_sided": None}
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[(start+j) % n] for j in range(block_length))
        draws.append(mean(sample[:n]))
    p = (1 + sum(value-avg >= avg for value in draws))/(replicates+1)
    return {"status": "valid", "n_pairs": n, "mean": avg,
            "lower": quantile(draws, alpha/2), "upper": quantile(draws, 1-alpha/2),
            "p_one_sided": p, "block_length": block_length, "replicates": replicates,
            "uncertainty_scope": "stationary_time_block_research_diagnostic"}


def holm(pvalues, alpha):
    require(0 < alpha < 1 and len(pvalues) > 0, "invalid multiple-test family")
    ordered = sorted((p if p is not None else 1., key) for key, p in pvalues.items())
    result, stopped, m = {}, False, len(ordered)
    for i, (p, key) in enumerate(ordered):
        require(finite(p) and 0 <= p <= 1, "invalid p value")
        reject = not stopped and p <= alpha/(m-i)
        stopped = stopped or not reject
        result[key] = {"reject_null": reject, "p": p, "threshold": alpha/(m-i), "family_size": m}
    return result


def required_sections(sigma, effect, alpha=.05, power=.8, dependence_multiplier=1.):
    require(sigma >= 0 and effect > 0 and 0 < alpha < 1 and 0 < power < 1
            and dependence_multiplier >= 1, "invalid power design")
    z = NormalDist().inv_cdf(1-alpha) + NormalDist().inv_cdf(power)
    return max(2, math.ceil((z*sigma/effect)**2*dependence_multiplier))


def performance(nav_rows):
    require(len(nav_rows) >= 2, "NAV comparison requires at least two observations")
    if any(row["nav"] is None for row in nav_rows):
        return None
    nav = [float(row["nav"]) for row in nav_rows]
    require(all(x > 0 for x in nav), "nonpositive NAV")
    returns = [b/a-1 for a, b in zip(nav, nav[1:])]
    peak, drawdown = nav[0], 0.
    for v in nav:
        peak = max(peak, v)
        drawdown = min(drawdown, v/peak-1)
    sd = stdev(returns) if len(returns) > 1 else 0
    return {"total_return": nav[-1]/nav[0]-1, "max_drawdown": drawdown,
            "daily_mean": mean(returns), "daily_sd": sd,
            "sharpe_252": mean(returns)/sd*math.sqrt(252) if sd else None,
            "returns": returns, "n_pairs": len(returns)}
