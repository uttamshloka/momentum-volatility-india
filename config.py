"""Strategy parameters in one place.

Defaults reproduce the configuration selected during research:
9-month momentum, top decile, calmest 80% of stocks by volatility,
fixed +4% / -2% barriers with a 10-trading-day time limit.
"""

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # Signal
    mom_col: str = "mom_9m"            # momentum lookback used for ranking
    vol_window: int = 63               # trading days for realised volatility
    exclude_top_vol_frac: float = 0.20 # drop the most volatile 20% (historical cutoff)
    mom_percentile: float = 0.90       # momentum bar: 0.9 = top decile
    min_history_rows: int = 2000       # rows of prior-year data needed before trading

    # Exit rules (triple barrier)
    profit_target: float = 0.04
    stop_loss: float = 0.02
    vertical_days: int = 10
    tie_policy: str = "optimistic"     # same-day target+stop: "optimistic" or "pessimistic"

    # Portfolio construction
    risk_per_trade: float = 0.005      # capital lost if a position hits its stop
    max_position_frac: float = 0.15    # cap on any single position
    max_sector_frac: float = 0.35      # cap on any one sector
    max_new_positions_per_day: int = 4
    top_n_candidates: int = 20

    # Liquidity
    adtv_window: int = 20
    max_frac_of_adtv: float = 0.001    # position <= 0.1% of average daily traded value
    min_adtv: float = 5e7              # Rs 5 crore per day minimum


LOOKBACKS = {"1w": 5, "2w": 10, "1m": 21, "2m": 42, "3m": 63,
             "6m": 126, "9m": 189, "12m": 252}
