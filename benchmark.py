"""Compare the strategy with buying and holding an index.

Note: ^NSEI is the Nifty 50 PRICE index and excludes dividends, so it
understates a real buy-and-hold return by roughly the dividend yield.
Use a total-return series if you have one.
"""

import pandas as pd


def stats(values: pd.Series, dates: pd.Series) -> dict:
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return {"total_return": values.iloc[-1] / values.iloc[0] - 1,
            "cagr": (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1,
            "max_drawdown": (values / values.cummax() - 1).min()}


def load_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    idx = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    return idx.reset_index()[["Date", "Close"]]


def compare(equity: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    s = stats(equity["capital"], equity["Date"])
    b = stats(index["Close"], index["Date"])
    table = pd.DataFrame({"strategy": s, "index": b})
    table.loc["calmar"] = table.loc["cagr"] / table.loc["max_drawdown"].abs()
    return table
