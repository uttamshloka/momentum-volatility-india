# Volatility-Controlled Momentum: Indian Equities

A research pipeline for a short-horizon momentum strategy on NSE stocks: data
cleaning, triple-barrier labelling, a check that momentum survives a
volatility control, a day-by-day portfolio backtest with realistic costs,
and a benchmark comparison.

## The idea

Recent winners tend to keep winning over weeks to months. But a simple test
of that idea on Indian stocks gave a U-shaped result: both the strongest and
the weakest past performers hit a +4% target more often than stocks in the
middle. That pattern is a sign of **volatility clustering**: volatile stocks
move a lot in both directions. So the real question was whether momentum
predicts anything once volatility is held constant.

It does. Within volatility quintiles, target hit-rate still rises with
momentum (average within-quintile rank correlation +0.55 on 2015–2019).
In a logistic regression of target-before-stop on momentum and volatility
ranks, momentum stays positive and significant (coefficient +0.11,
p < 0.001) after controlling for volatility.

## The strategy

| Element | Rule |
|---|---|
| Universe | NSE stocks with at least 10 years of history (see limitations) |
| Signal | 9-month return in the top decile, among the calmest 80% of stocks by 63-day volatility |
| Thresholds | Fixed per calendar year from **prior years only**, not from the same day's cross-section |
| Exit | +4% target, -2% stop, or 10 trading days, whichever comes first |
| Sizing | Hitting the stop loses 0.5% of capital; no position above 15% of capital |
| Diversification | No sector above 35% of capital, counting open positions |
| Liquidity | Position at most 0.1% of 20-day average traded value; stock must trade at least Rs 5 crore a day |
| Costs | Groww delivery charges: brokerage, STT, exchange, SEBI, stamp duty, DP charge, GST |

Using thresholds from prior years matters. An earlier version recalculated
"calm" and "top decile" from each day's cross-section. On stressed days that
bar drifts: a stock looks calm only because everything else is volatile, and
the top decile can be stocks that are merely falling less. That version lost
money (CAGR -14.6%, 2015–2019). The stable bar is what the research supports.

## Results

Re-run with this repository, Rs 10 lakh starting capital, net of costs:

| Period | Strategy CAGR | Nifty 50 CAGR | Strategy max DD | Nifty max DD | Calmar | Trades | Win rate | Avg net/trade | Capital deployed |
|---|---|---|---|---|---|---|---|---|---|
| 2008–2019 | 4.5% | 5.9%* | -27.7% | -59.9%* | n/a | 3,628 | 39.2% | +0.11% | 38% |
| 2020–14 Aug 2026 | 17.9% | 11.1% | -23.7% | -38.4% | 0.75 vs 0.29 | 3,995 | 40.6% | +0.20% | 68% |

\* Benchmark figure from the original research notebook; not re-run here.

Read these honestly:

- Over 2008–2019 the strategy **underperformed** buy-and-hold on return, but
  with under half the drawdown. Over 2020–2026 it beat the index on both.
- Capital sat largely idle: 38% deployed on average in 2008–2019 and 68% in
  2020–2026. Idle cash in a liquid fund would add to these figures; the
  backtest assumes it earns nothing.
- The Nifty figures are the price index (`^NSEI`), which excludes dividends,
  so the true benchmark return is somewhat higher than shown.
- In the research grid, the single highest-expectancy cell is in the most
  volatile quintile. The strategy excludes the top 20% by volatility anyway:
  the aim is momentum with controlled drawdowns, not the highest raw hit-rate.

The original notebook gave 4.8% (2008–2019) and 15.9% (2020–2026). This
rewrite checks each position's liquidity individually and no longer gives
rows at the end of the sample a shortened timeout. These changes are the likely source of the
differences.

## Limitations

- **Survivorship bias.** The default universe is today's NIFTY 500, filtered
  to stocks with 10+ years of history. Companies that were delisted or fell
  out of the index are missing, which flatters historical results. Supply a
  point-in-time universe with `--universe-csv` to remove it.
- **Parameter selection.** The signal, barriers and filters were chosen after
  testing variants on 2015–2019 data. The 2020–2026 period is closer to
  out-of-sample, but variants were compared on it too.
- **Same-day ties.** When a daily bar touches both the target and the stop,
  the order is unknown. The default counts these as wins (`tie_policy` in
  `config.py`); about 2.3% of labels are ties. Set it to `pessimistic` to see
  how much this matters.
- **Execution.** Entry is assumed at the signal day's close, with no
  slippage beyond the cost model. Yahoo Finance data has gaps and bad ticks
  for some smaller companies.
- **No live track record.** This is a backtest.

## Usage

```bash
pip install -r requirements.txt

python cli.py download  --start 2005-01-01 --outdir data/prices
python cli.py clean     --price-dir data/prices --outdir data/prices_clean
python cli.py sectors   --price-dir data/prices_clean
python cli.py research  --price-dir data/prices_clean --train-start 2015-01-01 --train-end 2019-12-31
python cli.py backtest  --price-dir data/prices_clean --start 2020-01-01 --end 2026-08-14
python cli.py benchmark --equity results/equity.csv --start 2020-01-01 --end 2026-08-14
python cli.py signals   --price-dir data/prices_clean --capital 1000000
```

## Structure

```
momvol/
  config.py      all strategy parameters
  data.py        download, load, clean (split/bad-tick and outlier checks), sector map
  features.py    momentum, realised volatility, average traded value
  labels.py      triple-barrier labels
  research.py    volatility-controlled grid and logistic regression
  portfolio.py   annual thresholds, selection, sizing, liquidity, sector caps
  costs.py       Groww delivery cost model
  backtest.py    day-by-day simulation and summary
  benchmark.py   comparison with an index
cli.py           command-line entry point
```

## Author

Abhishek Gaurav, CFA. For research and education only; not investment advice.
