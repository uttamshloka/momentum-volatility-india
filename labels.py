"""Triple-barrier labels.

For each stock-day t, with entry at Close_t, look at the next
`vertical_days` daily bars:
    +1  High reaches Close_t * (1 + profit_target) first
    -1  Low reaches Close_t * (1 - stop_loss) first
     0  neither within the window (timeout); fwd_ret = Close_{t+V}/Close_t - 1

If both barriers are touched in the same daily bar the order is unknown;
`tie_policy` decides the label and the row is flagged `tie=True`.
Rows whose full window is not yet observable (end of sample) and have not
hit a barrier are left unlabelled rather than given a truncated timeout.
"""

import numpy as np
import pandas as pd


def _label_one(close, high, low, pt, sl, V, tie_policy):
    n = len(close)
    upper, lower = close * (1 + pt), close * (1 - sl)
    first_up = np.full(n, np.inf)
    first_dn = np.full(n, np.inf)
    for k in range(1, V + 1):
        hi = np.r_[high[k:], np.full(k, np.nan)]
        lo = np.r_[low[k:], np.full(k, np.nan)]
        up_hit = (hi >= upper) & np.isinf(first_up)
        dn_hit = (lo <= lower) & np.isinf(first_dn)
        first_up[up_hit] = k
        first_dn[dn_hit] = k

    label = np.full(n, np.nan)
    days = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    tie = np.zeros(n, dtype=bool)

    up_first = first_up < first_dn
    dn_first = first_dn < first_up
    same = np.isfinite(first_up) & (first_up == first_dn)
    label[up_first], days[up_first] = 1, first_up[up_first]
    label[dn_first], days[dn_first] = -1, first_dn[dn_first]
    label[same] = 1 if tie_policy == "optimistic" else -1
    days[same], tie[same] = first_up[same], True

    idx = np.arange(n)
    timeout = np.isinf(first_up) & np.isinf(first_dn) & (idx + V <= n - 1)
    label[timeout], days[timeout] = 0, V
    fwd_ret[timeout] = close[idx[timeout] + V] / close[timeout] - 1
    return label, days, fwd_ret, tie


def triple_barrier(df: pd.DataFrame, profit_target: float = 0.04, stop_loss: float = 0.02,
                   vertical_days: int = 10, tie_policy: str = "optimistic") -> pd.DataFrame:
    if tie_policy not in ("optimistic", "pessimistic"):
        raise ValueError("tie_policy must be 'optimistic' or 'pessimistic'")
    out = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        label, days, fwd, tie = _label_one(
            g["Close"].to_numpy(float), g["High"].to_numpy(float), g["Low"].to_numpy(float),
            profit_target, stop_loss, vertical_days, tie_policy)
        g["label"], g["days_to_hit"], g["fwd_ret_timeout"], g["tie"] = label, days, fwd, tie
        dates = g["Date"].to_numpy()
        res = np.full(len(g), np.datetime64("NaT"), dtype=dates.dtype)
        ok = ~np.isnan(days)
        res[ok] = dates[np.flatnonzero(ok) + days[ok].astype(int)]
        g["resolution_date"] = res
        out.append(g)
    return pd.concat(out, ignore_index=True)
