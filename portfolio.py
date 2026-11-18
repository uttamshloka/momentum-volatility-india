"""Candidate selection, sizing and diversification.

Selection uses thresholds fixed for each calendar year from PRIOR years'
data only. An earlier version recomputed 'calm' and 'top momentum' from
each day's cross-section; on stressed days that bar drifts (a stock can
look calm only because everything else is volatile) and the backtest
lost money. The stable historical bar is what the research validated.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import StrategyConfig


def annual_thresholds(df: pd.DataFrame, cfg: StrategyConfig) -> Dict[int, dict]:
    """{year: {vol_cutoff, mom_bar}} using only data from earlier years."""
    d = df.dropna(subset=["vol", cfg.mom_col])
    years = d["Date"].dt.year
    out = {}
    for y in sorted(years.unique()):
        hist = d[years < y]
        if len(hist) < cfg.min_history_rows:
            continue
        vol_cut = hist["vol"].quantile(1 - cfg.exclude_top_vol_frac)
        calm = hist[hist["vol"] <= vol_cut]
        out[int(y)] = {"vol_cutoff": vol_cut,
                       "mom_bar": calm[cfg.mom_col].quantile(cfg.mom_percentile)}
    return out


def qualify(day: pd.DataFrame, thresholds: Dict[int, dict], cfg: StrategyConfig) -> pd.DataFrame:
    """Stocks on one date that are calm enough and in the momentum top bucket."""
    d = day.dropna(subset=["vol", cfg.mom_col])
    if d.empty:
        return d
    years = [y for y in thresholds if y <= d["Date"].iloc[0].year]
    if not years:
        return d.iloc[0:0]
    th = thresholds[max(years)]
    d = d[(d["vol"] <= th["vol_cutoff"]) & (d[cfg.mom_col] >= th["mom_bar"])]
    return d.sort_values(cfg.mom_col, ascending=False).head(cfg.top_n_candidates)


def position_size(capital: float, cfg: StrategyConfig) -> float:
    """Size so that hitting the stop costs `risk_per_trade` of capital, capped."""
    return min(cfg.risk_per_trade * capital / cfg.stop_loss, cfg.max_position_frac * capital)


def liquid(cands: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Each position must be small relative to that stock's traded value,
    and the stock must clear an absolute liquidity floor."""
    ok = (cands["position_value"] <= cfg.max_frac_of_adtv * cands["adtv"]) & (cands["adtv"] >= cfg.min_adtv)
    return cands[ok]


def diversify(cands: pd.DataFrame, sector_map: Dict[str, str], capital: float, cfg: StrategyConfig,
              existing: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Take candidates best-first, skipping any that would push their sector
    above `max_sector_frac` of capital (counting open positions)."""
    exposure = dict(existing or {})
    picked = []
    for _, row in cands.iterrows():
        if len(picked) >= cfg.max_new_positions_per_day:
            break
        sector = sector_map.get(row["symbol"], "Unknown")
        if exposure.get(sector, 0.0) + row["position_value"] > cfg.max_sector_frac * capital:
            continue
        exposure[sector] = exposure.get(sector, 0.0) + row["position_value"]
        picked.append(row.to_dict() | {"sector": sector})
    return pd.DataFrame(picked)


def select(day: pd.DataFrame, thresholds, sector_map, capital, cfg: StrategyConfig,
           existing=None) -> pd.DataFrame:
    cands = qualify(day, thresholds, cfg)
    if cands.empty:
        return cands
    cands = cands.assign(position_value=position_size(capital, cfg))
    cands = liquid(cands, cfg)
    if cands.empty:
        return cands
    return diversify(cands, sector_map, capital, cfg, existing)
