"""Features computed per symbol from past data only."""

import numpy as np
import pandas as pd

from .config import LOOKBACKS


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("symbol")["Close"]
    for name, window in LOOKBACKS.items():
        df[f"mom_{name}"] = g.pct_change(window, fill_method=None)
    return df


def add_volatility(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Annualised realised volatility of daily returns."""
    df = df.copy()
    ret = df.groupby("symbol")["Close"].pct_change(fill_method=None)
    df["vol"] = (ret.groupby(df["symbol"]).rolling(window).std()
                 .reset_index(level=0, drop=True) * np.sqrt(252))
    return df


def add_adtv(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Average daily traded value in rupees (Close x Volume)."""
    df = df.copy()
    value = df["Close"] * df["Volume"]
    df["adtv"] = (value.groupby(df["symbol"]).rolling(window).mean()
                  .reset_index(level=0, drop=True))
    return df


def build_features(df: pd.DataFrame, vol_window: int = 63, adtv_window: int = 20) -> pd.DataFrame:
    return add_adtv(add_volatility(add_momentum(df), vol_window), adtv_window)
