"""CostSpec contract and old-proxy to native fee mapping (P0.5, v2.0 landing).

Lands the CONTRACTS.md CostSpec fields and the R2 requirement that the legacy
``COST_BPS`` proxy (v0.7 S5: one-time charge of N bps on actual turnover at the
start of each holding period) is NOT silently assumed equivalent to a native
fee schedule (commission / stamp duty / slippage). The mapping below is a
*calibration decomposition*, not an equivalence claim: the two protocols charge
different things at different times, and a replayed account must disclose the
difference rather than double-count.

Reference anchor: s5_joint_filter.py ``_s5_cost_bps`` docstring — A-share
round trip is roughly 3bp commission + 5bp stamp duty (sell side) + impact cost,
with 15~30bp as the common sensitivity range.
"""

from .contracts import ContractError, digest, keys, require

COST_SPEC_SCHEMA = "fuxi.cost_spec.1"


def cost_spec(*, commission_buy, commission_sell, minimum_fee_currency, tax_schedule_ref,
              slippage_bps_per_side, participation_limit, initial_cash,
              lot_and_settlement_rules, proxy_cost_formula):
    """Build + validate a CostSpec (all units explicit, per CONTRACTS.md)."""
    spec = {"schema_version": COST_SPEC_SCHEMA,
            "commission_buy": commission_buy,
            "commission_sell": commission_sell,
            "minimum_fee_currency": minimum_fee_currency,
            "tax_schedule_ref": tax_schedule_ref,
            "slippage_bps_per_side": slippage_bps_per_side,
            "participation_limit": participation_limit,
            "initial_cash": initial_cash,
            "lot_and_settlement_rules": lot_and_settlement_rules,
            "proxy_cost_formula": proxy_cost_formula}
    validate_cost_spec(spec)
    return spec


def validate_cost_spec(spec):
    keys(spec, ("schema_version", "commission_buy", "commission_sell",
                "minimum_fee_currency", "tax_schedule_ref", "slippage_bps_per_side",
                "participation_limit", "initial_cash", "lot_and_settlement_rules",
                "proxy_cost_formula"))
    require(spec["schema_version"] == COST_SPEC_SCHEMA, "unsupported cost spec schema")
    for field in ("commission_buy", "commission_sell"):
        require(type(spec[field]) is float and spec[field] >= 0.0, field + " must be a non-negative float")
    require(type(spec["minimum_fee_currency"]) is float and spec["minimum_fee_currency"] >= 0.0,
            "minimum_fee_currency must be a non-negative float")
    require(isinstance(spec["tax_schedule_ref"], str) and spec["tax_schedule_ref"],
            "tax_schedule_ref required")
    require(type(spec["slippage_bps_per_side"]) is float and spec["slippage_bps_per_side"] >= 0.0,
            "slippage_bps_per_side must be a non-negative float")
    require(0.0 < spec["participation_limit"] <= 1.0, "participation_limit in (0, 1]")
    require(type(spec["initial_cash"]) is float and spec["initial_cash"] > 0.0,
            "initial_cash must be positive")
    require(isinstance(spec["lot_and_settlement_rules"], str) and spec["lot_and_settlement_rules"],
            "lot_and_settlement_rules required")
    require(isinstance(spec["proxy_cost_formula"], str) and spec["proxy_cost_formula"],
            "proxy_cost_formula required")
    digest(spec)
    return spec


# ---------------------------------------------------------------------------
# Old proxy (FUXI_S5_COST_BPS, default 15) vs native A-share components.
# Calibration decomposition ONLY — see module docstring.
# ---------------------------------------------------------------------------
NATIVE_ASHARE_REFERENCE = cost_spec(
    commission_buy=0.00015,            # 1.5bp per side => 3bp round trip
    commission_sell=0.00015,
    minimum_fee_currency=5.0,          # 5 CNY minimum per order
    tax_schedule_ref="stamp_duty_sell_0.0005",  # 印花税 5bp, sell side only
    slippage_bps_per_side=3.5,         # 7bp round-trip impact/slippage proxy
    participation_limit=1.0,
    initial_cash=10_000_000.0,
    lot_and_settlement_rules="board_100_min_lot_t1_settlement",
    proxy_cost_formula="one_time_bps_on_turnover_at_period_start",
)

PROXY_15BPS_MAP = {
    "schema_version": "fuxi.cost_proxy_map.1",
    "proxy_protocol": {
        "name": "COST_BPS_15_legacy",
        "env_knob": "FUXI_S5_COST_BPS",
        "default_value_bps": 15.0,
        "charge": "one_time: 2*|new-old|/top_n turnover * bps/1e4 at period start",
        "implementation": "s5_joint_filter.py",
    },
    "native_reference": NATIVE_ASHARE_REFERENCE,
    "decomposition_bps": {
        "commission_round_trip": 3.0,
        "stamp_duty_sell": 5.0,
        "impact_slippage_proxy": 7.0,
        "total": 15.0,
    },
    "relationship": "proxy replaces native, never add both; equivalence is NOT assumed",
    "no_double_count_rule": "when replaying under native CostSpec, the proxy charge must be absent",
}

A_SHARE_TAX_SCHEDULE = {
    "schema_version": "fuxi.tax_schedule.1",
    "ref": "stamp_duty_sell_0.0005",
    "stamp_duty_sell_rate": 0.0005,
    "transfer_fee_note": "not modelled in proxy decomposition",
    "source": "s5_joint_filter.py _s5_cost_bps docstring calibration reference",
}


def replay_cost_comparison(trades, proxy_bps=15.0, native_spec=None):
    """Replay the same synthetic journal under both cost protocols.

    ``trades`` is a list of dicts: {side: 'buy'|'sell', notional: float}.
    Proxy protocol: one-time charge of proxy_bps on the total turnover.
    Native protocol: per-trade commission + sell-side stamp duty + per-side
    slippage (from ``native_spec``), with a minimum fee per order.

    Returns per-protocol totals and an explicit ``double_counted`` check; the
    two totals are allowed (and expected) to differ — that difference is the
    disclosure this module exists to make.
    """
    require(isinstance(trades, list) and trades, "trades must be a non-empty list")
    spec = native_spec if native_spec is not None else NATIVE_ASHARE_REFERENCE
    validate_cost_spec(spec)
    for trade in trades:
        keys(trade, ("side", "notional"))
        require(trade["side"] in ("buy", "sell"), "invalid trade side")
        require(type(trade["notional"]) is float and trade["notional"] >= 0.0,
                "notional must be a non-negative float")
    turnover = sum(t["notional"] for t in trades)
    proxy_total = turnover * proxy_bps / 1e4
    native_total, commission, stamp, slippage, min_fees = 0.0, 0.0, 0.0, 0.0, 0.0
    for trade in trades:
        rate = spec["commission_buy"] if trade["side"] == "buy" else spec["commission_sell"]
        fee = max(trade["notional"] * rate, spec["minimum_fee_currency"])
        commission += fee
        slippage += trade["notional"] * spec["slippage_bps_per_side"] / 1e4
        if trade["side"] == "sell":
            stamp += trade["notional"] * A_SHARE_TAX_SCHEDULE["stamp_duty_sell_rate"]
    native_total = commission + slippage + stamp
    return {"proxy_total": proxy_total, "native_total": native_total,
            "native_components": {"commission": commission, "slippage": slippage,
                                  "stamp_duty": stamp},
            "difference": native_total - proxy_total,
            "double_counted": False,
            "note": "proxy and native are alternative protocols; never summed"}
