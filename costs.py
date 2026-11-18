"""Round-trip cost of an equity delivery trade on Groww.

Rates follow Groww's published delivery rate card as of mid-2026. Broker
schedules change: re-check https://groww.in/charges before relying on them.

Per leg: brokerage min(Rs 20, 0.1%) floored at Rs 5; STT 0.1%; exchange
~0.00297%; SEBI 0.0001%; GST 18% on brokerage + exchange + SEBI.
Buy leg only: stamp duty 0.015%. Sell leg only: DP charge Rs 20 + GST.
"""

BROKERAGE_CAP, BROKERAGE_PCT, BROKERAGE_MIN = 20.0, 0.001, 5.0
STT_PCT = 0.001
EXCHANGE_PCT = 0.0000297
SEBI_PCT = 0.000001
STAMP_PCT = 0.00015
DP_CHARGE = 20.0 * 1.18
GST = 0.18


def leg_cost(value: float, is_buy: bool) -> float:
    brokerage = min(max(min(BROKERAGE_CAP, BROKERAGE_PCT * value), BROKERAGE_MIN), BROKERAGE_CAP)
    exch, sebi = EXCHANGE_PCT * value, SEBI_PCT * value
    cost = brokerage + STT_PCT * value + exch + sebi + GST * (brokerage + exch + sebi)
    cost += STAMP_PCT * value if is_buy else DP_CHARGE
    return cost


def round_trip_cost_pct(position_value: float, gross_return: float) -> float:
    """Total buy + sell cost as a fraction of the position size."""
    sell_value = position_value * (1 + gross_return)
    return (leg_cost(position_value, True) + leg_cost(sell_value, False)) / position_value
