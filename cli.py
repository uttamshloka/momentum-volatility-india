"""Command-line entry point.

    python cli.py download  --start 2005-01-01 --outdir data/prices
    python cli.py clean     --price-dir data/prices --outdir data/prices_clean
    python cli.py sectors   --price-dir data/prices_clean
    python cli.py research  --price-dir data/prices_clean --train-end 2019-12-31
    python cli.py backtest  --price-dir data/prices_clean --start 2020-01-01 --end 2026-08-14
    python cli.py benchmark --equity results/equity.csv --start 2020-01-01 --end 2026-08-14
    python cli.py signals   --price-dir data/prices_clean --capital 1000000
"""

import argparse
from pathlib import Path

import pandas as pd

from momvol import StrategyConfig
from momvol import backtest, benchmark, data, features, labels, portfolio, research


def _sector_map(path: str) -> dict:
    if not Path(path).exists():
        print(f"WARNING: {path} not found; every stock treated as one sector.")
        return {}
    s = pd.read_csv(path)
    return dict(zip(s["symbol"], s["sector"]))


def _prepared(price_dir: str, cfg: StrategyConfig, end: str = None) -> pd.DataFrame:
    df = data.load_prices(price_dir)
    if end:
        df = df[df["Date"] <= end]
    df = features.build_features(df, cfg.vol_window, cfg.adtv_window)
    return labels.triple_barrier(df, cfg.profit_target, cfg.stop_loss,
                                 cfg.vertical_days, cfg.tie_policy)


def cmd_download(a):
    symbols = (pd.read_csv(a.universe_csv)["Symbol"].astype(str).tolist()
               if a.universe_csv else data.fetch_nifty500_symbols())
    end = a.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    man = data.download_universe(symbols, a.start, end, a.outdir, a.min_history_years)
    print(f"Saved {len(man)}/{len(symbols)} symbols to {a.outdir}")


def cmd_clean(a):
    cleaned, flags = data.clean_prices(data.load_prices(a.price_dir), drop=not a.report_only)
    flags.to_csv(a.report, index=False)
    data.save_prices(cleaned, a.outdir)
    print(f"{len(flags)} rows flagged (report: {a.report}); cleaned data in {a.outdir}")


def cmd_sectors(a):
    syms = sorted(data.load_prices(a.price_dir)["symbol"].unique())
    m = data.fetch_sector_map(syms, a.out)
    print(m["sector"].value_counts().to_string())


def cmd_research(a):
    cfg = StrategyConfig()
    df = _prepared(a.price_dir, cfg, a.train_end)
    if a.train_start:
        df = df[df["Date"] >= a.train_start]
    print(f"Same-day barrier ties: {df['tie'].mean():.2%} of labelled rows")
    for col in a.lookbacks.split(","):
        grid = research.two_way_grid(df, col, cfg.profit_target, cfg.stop_loss)
        model = research.logistic_check(df, col)
        best = grid.loc[grid["naive_ev"].idxmax()]
        print(f"\n{col}: best cell vol={int(best.vol_bucket)} mom={int(best.mom_bucket)} "
              f"naive_ev={best.naive_ev:+.4f} (n={int(best.n)})")
        print(f"  logit mom_rank {model.params['mom_rank']:+.3f} (p={model.pvalues['mom_rank']:.4f}), "
              f"vol_rank {model.params['vol_rank']:+.3f} (p={model.pvalues['vol_rank']:.4f})")
        grid.to_csv(Path(a.outdir) / f"grid_{col}.csv", index=False)


def cmd_backtest(a):
    cfg = StrategyConfig(mom_percentile=a.mom_percentile)
    df = _prepared(a.price_dir, cfg)
    equity, trades = backtest.run_backtest(df, a.start, a.end, a.capital,
                                           _sector_map(a.sector_map), cfg)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out / "equity.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    backtest.print_summary(backtest.summarize(equity, trades))


def cmd_benchmark(a):
    eq = pd.read_csv(a.equity, parse_dates=["Date"])
    idx = benchmark.load_index(a.index, a.start, a.end)
    print(benchmark.compare(eq, idx).round(3).to_string())


def cmd_signals(a):
    cfg = StrategyConfig()
    df = features.build_features(data.load_prices(a.price_dir), cfg.vol_window, cfg.adtv_window)
    latest = df["Date"].max()
    th = portfolio.annual_thresholds(df, cfg)
    cands = portfolio.qualify(df[df["Date"] == latest], th, cfg)
    if cands.empty:
        print(f"{latest.date()}: no qualifying stocks.")
        return
    cands = portfolio.liquid(cands.assign(position_value=portfolio.position_size(a.capital, cfg)), cfg)
    picks = portfolio.diversify(cands, _sector_map(a.sector_map), a.capital, cfg)
    print(f"Signals from {latest.date()} close, for the next session:")
    print(picks[["symbol", "sector", cfg.mom_col, "position_value"]].to_string(index=False)
          if not picks.empty else "none after liquidity and sector filters")


def main():
    p = argparse.ArgumentParser(description="Volatility-controlled momentum strategy")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download"); s.set_defaults(fn=cmd_download)
    s.add_argument("--start", default="2005-01-01"); s.add_argument("--end")
    s.add_argument("--outdir", default="data/prices"); s.add_argument("--universe-csv")
    s.add_argument("--min-history-years", type=float, default=10.0)

    s = sub.add_parser("clean"); s.set_defaults(fn=cmd_clean)
    s.add_argument("--price-dir", default="data/prices")
    s.add_argument("--outdir", default="data/prices_clean")
    s.add_argument("--report", default="data_quality_flags.csv")
    s.add_argument("--report-only", action="store_true")

    s = sub.add_parser("sectors"); s.set_defaults(fn=cmd_sectors)
    s.add_argument("--price-dir", default="data/prices_clean"); s.add_argument("--out", default="sector_map.csv")

    s = sub.add_parser("research"); s.set_defaults(fn=cmd_research)
    s.add_argument("--price-dir", default="data/prices_clean")
    s.add_argument("--train-start"); s.add_argument("--train-end", default="2019-12-31")
    s.add_argument("--lookbacks", default="mom_3m,mom_6m,mom_9m,mom_12m")
    s.add_argument("--outdir", default=".")

    s = sub.add_parser("backtest"); s.set_defaults(fn=cmd_backtest)
    s.add_argument("--price-dir", default="data/prices_clean")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--capital", type=float, default=1_000_000)
    s.add_argument("--sector-map", default="sector_map.csv")
    s.add_argument("--mom-percentile", type=float, default=0.9)
    s.add_argument("--outdir", default="results")

    s = sub.add_parser("benchmark"); s.set_defaults(fn=cmd_benchmark)
    s.add_argument("--equity", default="results/equity.csv")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--index", default="^NSEI")

    s = sub.add_parser("signals"); s.set_defaults(fn=cmd_signals)
    s.add_argument("--price-dir", default="data/prices_clean")
    s.add_argument("--sector-map", default="sector_map.csv")
    s.add_argument("--capital", type=float, default=1_000_000)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
