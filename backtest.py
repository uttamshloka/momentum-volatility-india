"""Day-by-day portfolio simulation.

Each day: close positions whose barrier resolved, book P&L net of costs,
then open new positions sized on current capital. Entry is assumed at the
signal day's close. Open positions are carried at cost until they resolve
(no daily mark-to-market), so the equity curve moves in steps.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .costs import round_trip_cost_pct
from .portfolio import annual_thresholds, select


def run_backtest(labeled: pd.DataFrame, start: str, end: str, capital: float,
                 sector_map: Dict[str, str], cfg: StrategyConfig = StrategyConfig()):
    thresholds = annual_thresholds(labeled, cfg)
    window = labeled[(labeled["Date"] >= start) & (labeled["Date"] <= end)]
    by_date = {d: g.dropna(subset=["label"]) for d, g in window.groupby("Date")}

    cash, positions, equity, trades = float(capital), [], [], []

    for date in sorted(by_date):
        # 1. resolve maturing positions
        still_open = []
        for p in positions:
            if p["resolution_date"] <= date:
                if p["label"] == 1:
                    gross = cfg.profit_target
                elif p["label"] == -1:
                    gross = -cfg.stop_loss
                else:
                    gross = p["fwd_ret_timeout"]
                net = gross - round_trip_cost_pct(p["position_value"], gross)
                cash += p["position_value"] * (1 + net)
                trades.append({**{k: p[k] for k in ("symbol", "sector", "entry_date",
                                                    "resolution_date", "position_value", "label")},
                               "gross_ret": gross, "net_ret": net,
                               "pnl": p["position_value"] * net})
            else:
                still_open.append(p)
        positions = still_open

        # 2. open new positions
        deployed = sum(p["position_value"] for p in positions)
        total = cash + deployed
        exposure = {}
        for p in positions:
            exposure[p["sector"]] = exposure.get(p["sector"], 0.0) + p["position_value"]
        picks = select(by_date[date], thresholds, sector_map, total, cfg, exposure)
        for _, r in picks.iterrows():
            if r["position_value"] > cash:
                continue
            cash -= r["position_value"]
            positions.append({"symbol": r["symbol"], "sector": r["sector"], "entry_date": date,
                              "position_value": r["position_value"], "label": r["label"],
                              "resolution_date": r["resolution_date"],
                              "fwd_ret_timeout": r["fwd_ret_timeout"]})

        deployed = sum(p["position_value"] for p in positions)
        equity.append({"Date": date, "capital": cash + deployed, "cash": cash,
                       "deployed": deployed, "n_open": len(positions)})

    return pd.DataFrame(equity), pd.DataFrame(trades)


def summarize(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    cap = equity["capital"]
    years = (equity["Date"].iloc[-1] - equity["Date"].iloc[0]).days / 365.25
    s = {
        "start": equity["Date"].iloc[0].date(), "end": equity["Date"].iloc[-1].date(),
        "years": round(years, 1),
        "total_return": cap.iloc[-1] / cap.iloc[0] - 1,
        "cagr": (cap.iloc[-1] / cap.iloc[0]) ** (1 / max(years, 0.01)) - 1,
        "max_drawdown": (cap / cap.cummax() - 1).min(),
        "trades": len(trades),
        "win_rate": (trades["label"] == 1).mean() if len(trades) else np.nan,
        "avg_net_ret": trades["net_ret"].mean() if len(trades) else np.nan,
        "avg_open_positions": equity["n_open"].mean(),
        "avg_deployed": (equity["deployed"] / equity["capital"]).mean(),
    }
    return s


def print_summary(s: dict) -> None:
    print(f"Period            {s['start']} to {s['end']} ({s['years']} years)")
    print(f"Total return      {s['total_return']:+.1%}")
    print(f"CAGR              {s['cagr']:+.1%}")
    print(f"Max drawdown      {s['max_drawdown']:.1%}")
    print(f"Trades            {s['trades']}")
    print(f"Win rate          {s['win_rate']:.1%}")
    print(f"Avg net ret/trade {s['avg_net_ret']:+.3%}  (after costs)")
    print(f"Avg open / deployed  {s['avg_open_positions']:.1f} / {s['avg_deployed']:.0%}")
