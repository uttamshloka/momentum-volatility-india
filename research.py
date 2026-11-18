"""Does momentum add anything once volatility is controlled for?

Raw momentum deciles showed a U-shape in target hit-rate: both extreme
winners and extreme losers hit +4% more often. That is the signature of
volatility clustering, not momentum. Two checks, run on a training window:

1. Two-way grid: volatility quintiles, then momentum deciles within each.
2. Logistic regression of P(target before stop) on momentum and
   volatility percentile ranks (timeouts excluded).
"""

import numpy as np
import pandas as pd


def two_way_grid(df: pd.DataFrame, mom_col: str, profit_target: float, stop_loss: float,
                 n_vol: int = 5, n_mom: int = 10) -> pd.DataFrame:
    d = df.dropna(subset=[mom_col, "vol", "label"]).copy()
    d["vol_bucket"] = pd.qcut(d["vol"], n_vol, labels=False, duplicates="drop")
    d["mom_bucket"] = d.groupby("vol_bucket")[mom_col].transform(
        lambda x: pd.qcut(x, n_mom, labels=False, duplicates="drop"))
    g = d.groupby(["vol_bucket", "mom_bucket"])
    grid = pd.DataFrame({
        "n": g.size(),
        "p_target": g["label"].apply(lambda x: (x == 1).mean()),
        "p_stop": g["label"].apply(lambda x: (x == -1).mean()),
        "p_timeout": g["label"].apply(lambda x: (x == 0).mean()),
        "timeout_ret": g["fwd_ret_timeout"].mean(),
    }).reset_index()
    grid["naive_ev"] = (grid["p_target"] * profit_target - grid["p_stop"] * stop_loss
                        + grid["p_timeout"] * grid["timeout_ret"].fillna(0))
    return grid


def logistic_check(df: pd.DataFrame, mom_col: str):
    import statsmodels.api as sm
    d = df.dropna(subset=[mom_col, "vol", "label"])
    d = d[d["label"].isin([1, -1])]
    X = pd.DataFrame({"mom_rank": d[mom_col].rank(pct=True), "vol_rank": d["vol"].rank(pct=True)})
    return sm.Logit((d["label"] == 1).astype(int), sm.add_constant(X)).fit(disp=0)
