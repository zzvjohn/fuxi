import unittest

from fuxi_a1_three_arm_strict_probe_20260926 import quoted_session_state


class QuotedSessionStateTests(unittest.TestCase):
    def test_intraday_halt_with_quote_is_valued_but_not_open_filled(self):
        key = ("2025-06-25", "301678.SZ")
        self.assertEqual(quoted_session_state(key, set(), {key}), "partial_halt")
        self.assertEqual(quoted_session_state(key, {key}, {key}), "unknown")
        self.assertEqual(quoted_session_state(key, set(), set()), "trading")


if __name__ == "__main__":
    unittest.main()
