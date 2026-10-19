"""Single source of truth for worker wall-clock budgets.

2026-10-19 (user-authorized): the native worker window was raised from 300s to
420s because the certified ledger_v1 snapshot needs ~110-255s per candidate and
the parent-measured settlement could exceed the reservation by <1s, which used
to poison the campaign permanently (``campaigns.status='budget_overrun'``).

Two numbers must stay consistent wherever they are consumed:
  * ``worker_timeout_seconds()`` — the actual per-attempt wall clock;
  * ``budget_units(rounds, candidates)`` — the reservation the campaign freezes.

``OVERRUN_TOLERANCE`` keeps measurement jitter (parent spawn/teardown) from
being reported as a real budget breach. It does not relax the total cap: the
campaign still stops when cumulative spend exceeds ``budget_units``.
"""
import os

DEFAULT_WORKER_TIMEOUT_SECONDS = 420
MIN_WORKER_TIMEOUT_SECONDS = 120
FIXED_OVERHEAD_SECONDS = 420
OVERRUN_TOLERANCE = 0.02


def worker_timeout_seconds(environ=None):
    source = os.environ if environ is None else environ
    raw = source.get("FUXI_V1_WORKER_TIMEOUT")
    try:
        value = int(raw) if raw not in (None, "") else DEFAULT_WORKER_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        value = DEFAULT_WORKER_TIMEOUT_SECONDS
    return max(MIN_WORKER_TIMEOUT_SECONDS, value)


def budget_units(rounds, candidates, worker_timeout=None):
    timeout = worker_timeout if worker_timeout is not None else worker_timeout_seconds()
    return int(rounds) * int(candidates) * int(timeout) + FIXED_OVERHEAD_SECONDS


def is_overrun(actual, reserved):
    """True only when the attempt genuinely exceeded its reservation.

    A 2% (and never less than the float noise of the measurement) tolerance is
    applied so that a 300.3s measurement against a 300s reservation no longer
    halts the whole campaign.
    """
    if actual is None or reserved is None:
        return False
    allowed = max(float(reserved) * (1.0 + OVERRUN_TOLERANCE), float(reserved) + 1e-6)
    return float(actual) > allowed
