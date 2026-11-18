"""Price data: download, load and clean.

Source is Yahoo Finance via yfinance (split/dividend adjusted). This is
research-grade data; cross-check anything you intend to trade against NSE
bhavcopy or a paid vendor.
"""

import io
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NSE_NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
PRICE_COLS = ["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------- download

def fetch_nifty500_symbols() -> List[str]:
    """Current NIFTY 500 constituents from NSE archives.

    Note: using today's constituents for a historical backtest introduces
    survivorship bias (see README). Pass your own point-in-time universe
    with --universe-csv where possible.
    """
    import requests
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
    r = requests.get(NSE_NIFTY500_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str).str.strip().tolist()


def download_symbol(symbol: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    import yfinance as yf
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(f"{symbol}.NS", start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["symbol"] = symbol
            return df[PRICE_COLS]
        except Exception as exc:  # network / rate-limit errors
            log.warning("%s attempt %d failed: %s", symbol, attempt, exc)
            time.sleep(1.5 * attempt)
    return None


def download_universe(symbols: List[str], start: str, end: str, outdir: str,
                      min_history_years: float = 10.0, sleep: float = 0.3) -> pd.DataFrame:
    """Save one Parquet file per symbol; return a manifest."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sym in symbols:
        df = download_symbol(sym, start, end)
        time.sleep(sleep)
        if df is None:
            continue
        years = (df["Date"].max() - df["Date"].min()).days / 365.25
        if years < min_history_years:
            continue
        df.to_parquet(out / f"{sym}.parquet", index=False)
        manifest.append({"symbol": sym, "rows": len(df),
                         "start": df["Date"].min(), "end": df["Date"].max()})
    man = pd.DataFrame(manifest)
    man.to_csv(out / "_manifest.csv", index=False)
    return man


# ---------------------------------------------------------------- load

def load_prices(price_dir: str) -> pd.DataFrame:
    files = [f for f in sorted(Path(price_dir).glob("*.parquet")) if not f.name.startswith("_")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {price_dir}")
    df = pd.concat([pd.read_parquet(f)[PRICE_COLS] for f in files], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["symbol", "Date"]).reset_index(drop=True)


def save_prices(df: pd.DataFrame, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for sym, g in df.groupby("symbol"):
        g.to_parquet(out / f"{sym}.parquet", index=False)


# ---------------------------------------------------------------- clean

def flag_jump_reversions(df: pd.DataFrame, jump: float = 0.20, window: int = 10,
                         tolerance: float = 0.15) -> pd.DataFrame:
    """Daily moves beyond `jump` that revert to within `tolerance` of the
    prior close inside `window` days: the signature of an unadjusted
    split/bonus or a bad tick, not a real move."""
    flags = []
    for sym, g in df.groupby("symbol", sort=False):
        close = g["Close"].to_numpy()
        dates = g["Date"].to_numpy()
        ret = np.r_[np.nan, close[1:] / close[:-1] - 1]
        for t in np.flatnonzero(np.abs(ret) >= jump):
            seg = close[t:min(t + window, len(close) - 1) + 1]
            if np.any(np.abs(seg / close[t - 1] - 1) <= tolerance):
                flags.append({"symbol": sym, "Date": dates[t], "reason": "jump_then_revert"})
    return pd.DataFrame(flags, columns=["symbol", "Date", "reason"])


def flag_momentum_outliers(df: pd.DataFrame, lookback: int = 63, z: float = 10.0) -> pd.DataFrame:
    """Robust (median/MAD) cross-sectional outliers in trailing return."""
    d = df[["symbol", "Date", "Close"]].copy()
    d["mom"] = d.groupby("symbol")["Close"].pct_change(lookback, fill_method=None)
    med = d.groupby("Date")["mom"].transform("median")
    mad = (d["mom"] - med).abs().groupby(d["Date"]).transform("median")
    counts = d.groupby("Date")["mom"].transform("count")
    rz = 0.6745 * (d["mom"] - med) / mad.replace(0, np.nan)
    bad = d[(counts >= 30) & (rz.abs() > z)]
    return bad.assign(reason="cross_sectional_outlier")[["symbol", "Date", "reason"]]


def clean_prices(df: pd.DataFrame, drop: bool = True):
    """Return (cleaned_prices, flag_report)."""
    flags = pd.concat([flag_jump_reversions(df), flag_momentum_outliers(df)], ignore_index=True)
    if drop and not flags.empty:
        key = pd.MultiIndex.from_frame(flags[["symbol", "Date"]].drop_duplicates())
        mask = pd.MultiIndex.from_frame(df[["symbol", "Date"]]).isin(key)
        df = df[~mask].reset_index(drop=True)
    return df, flags


# ---------------------------------------------------------------- sectors

def fetch_sector_map(symbols: List[str], out_path: str, sleep: float = 0.5) -> pd.DataFrame:
    """Sector per symbol from yfinance, saved incrementally (safe to resume)."""
    import yfinance as yf
    path = Path(out_path)
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["symbol"] for r in rows}
    for sym in symbols:
        if sym in done:
            continue
        try:
            info = yf.Ticker(f"{sym}.NS").info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        rows.append({"symbol": sym, "sector": sector})
        pd.DataFrame(rows).to_csv(path, index=False)
        time.sleep(sleep)
    return pd.DataFrame(rows)
