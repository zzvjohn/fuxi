# -*- coding: utf-8 -*-
"""Pair tests for independent_halt_no_event."""
import sys
from pathlib import Path

ROOT = Path(r"e:/quant")
sys.path.insert(0, str(ROOT))
from fuxi_v1.action_evidence import independent_halt_no_event

FROZEN = {"close": "2.4100", "preclose": "2.4100", "tradestatus": "0", "volume": ""}
NO_EVENTS = set()


def test_frozen_halt_without_event_accepted():
    assert independent_halt_no_event(None, dict(FROZEN), NO_EVENTS, "2024-06-25", 2.41) is True


def test_rejects_calendar_event_on_halt_day():
    assert independent_halt_no_event(None, dict(FROZEN), {"2024-06-25"}, "2024-06-25", 2.41) is False


def test_rejects_trading_status_one():
    trading = dict(FROZEN, tradestatus="1")
    assert independent_halt_no_event(None, trading, NO_EVENTS, "2024-06-25", 2.41) is False


def test_rejects_non_frozen_price_pair():
    moving = dict(FROZEN, preclose="2.4000")
    assert independent_halt_no_event(None, moving, NO_EVENTS, "2024-06-25", 2.41) is False


def test_rejects_mark_mismatch():
    assert independent_halt_no_event(None, dict(FROZEN), NO_EVENTS, "2024-06-25", 2.40) is False


def test_rejects_missing_mark():
    assert independent_halt_no_event(None, dict(FROZEN), NO_EVENTS, "2024-06-25", None) is False


def test_rejects_missing_after_quote():
    assert independent_halt_no_event(None, None, NO_EVENTS, "2024-06-25", 2.41) is False


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception:
            failed += 1
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print("total:", len(tests), "failed:", failed)
    sys.exit(1 if failed else 0)
