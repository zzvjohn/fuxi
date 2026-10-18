import unittest
from fuxi_v1.contracts import ContractError, digest
from fuxi_v1.cost_spec import (A_SHARE_TAX_SCHEDULE, NATIVE_ASHARE_REFERENCE,
                               PROXY_15BPS_MAP, cost_spec, replay_cost_comparison,
                               validate_cost_spec)


class CostSpecTests(unittest.TestCase):
    def test_cost_spec_validation(self):
        spec = cost_spec(commission_buy=0.00015, commission_sell=0.00015,
                         minimum_fee_currency=5.0, tax_schedule_ref="stamp_duty_sell_0.0005",
                         slippage_bps_per_side=3.5, participation_limit=1.0,
                         initial_cash=1_000_000.0, lot_and_settlement_rules="board_100",
                         proxy_cost_formula="one_time_bps_on_turnover_at_period_start")
        self.assertEqual(digest(spec), digest(cost_spec(
            commission_buy=0.00015, commission_sell=0.00015, minimum_fee_currency=5.0,
            tax_schedule_ref="stamp_duty_sell_0.0005", slippage_bps_per_side=3.5,
            participation_limit=1.0, initial_cash=1_000_000.0,
            lot_and_settlement_rules="board_100",
            proxy_cost_formula="one_time_bps_on_turnover_at_period_start")))
        with self.assertRaises(ContractError):
            validate_cost_spec({**spec, "participation_limit": 0.0})
        with self.assertRaises(ContractError):
            validate_cost_spec({**spec, "commission_buy": -0.1})
        with self.assertRaises(ContractError):
            validate_cost_spec({**spec, "extra": 1})

    def test_proxy_decomposition_sums_to_15(self):
        d = PROXY_15BPS_MAP["decomposition_bps"]
        self.assertAlmostEqual(d["commission_round_trip"] + d["stamp_duty_sell"]
                               + d["impact_slippage_proxy"], d["total"], places=6)
        self.assertEqual(d["total"], PROXY_15BPS_MAP["proxy_protocol"]["default_value_bps"])

    def test_replay_discloses_difference(self):
        trades = [{"side": "buy", "notional": 1_000_000.0},
                  {"side": "sell", "notional": 1_000_000.0}]
        out = replay_cost_comparison(trades, proxy_bps=15.0)
        self.assertAlmostEqual(out["proxy_total"], 2_000_000 * 15 / 1e4, places=6)
        # native: commission 1.5bp*2 + slippage 3.5bp*2 + stamp 5bp on sell
        expected = (1_000_000 * 0.00015 * 2 + 1_000_000 * 3.5 / 1e4 * 2
                    + 1_000_000 * 0.0005)
        self.assertAlmostEqual(out["native_total"], expected, places=6)
        self.assertFalse(out["double_counted"])
        self.assertNotEqual(out["proxy_total"], out["native_total"])

    def test_no_double_counting_rule_present(self):
        self.assertIn("no_double_count_rule", PROXY_15BPS_MAP)
        self.assertIn("never add both", PROXY_15BPS_MAP["relationship"])

    def test_native_reference_valid_and_identifiable(self):
        validate_cost_spec(NATIVE_ASHARE_REFERENCE)
        self.assertIsNotNone(digest(NATIVE_ASHARE_REFERENCE))
        self.assertEqual(A_SHARE_TAX_SCHEDULE["ref"],
                         NATIVE_ASHARE_REFERENCE["tax_schedule_ref"])


if __name__ == "__main__":
    unittest.main()
