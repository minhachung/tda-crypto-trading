# V8: Continuous Walk-Forward Profitability Test

**Hypothesis:** Continuous trading (not k-fold) with realistic position sizing
will reveal whether the v6/v7 direction-accuracy edge converts to profit.

**Setup:** 365 days hourly data, walk-forward from day 60, 7d holds, p>0.65 threshold, up to 50% position size.
**Date:** 2026-05-06


## Results

| Scenario | Asset | Trades | Strategy | Buy-Hold | Diff | Sharpe | Max DD |
|----------|-------|-------:|---------:|---------:|-----:|-------:|-------:|
| binance_maker | ADA | 38 | +11.99% | -55.25% | +67.24% | 0.95 | -9.31% |
| binance_maker | SOL | 37 | -15.96% | -41.82% | +25.86% | -1.44 | -24.57% |
| coinbase_taker | ADA | 38 | +10.95% | -55.25% | +66.20% | 0.88 | -9.31% |
| coinbase_taker | SOL | 37 | -17.83% | -41.82% | +23.99% | -1.63 | -25.40% |

## Trade Counts (vs k-fold v7)

v7 had ~30 actual trades total across all folds. v8 has:

- binance_maker: 75 trades
- coinbase_taker: 75 trades

This is the trade count we need for proper Sharpe estimation.

## Verdict

### Profitable on Binance.US:
- **ADA**: +11.99% over 38 trades, Sharpe 0.95

### Lost money on Binance.US:
- **SOL**: -15.96% over 37 trades, Sharpe -1.44

### Vs Buy-Hold:
- **ADA**: TDA +11.99% vs B&H -55.25% — **TDA strategy wins by 67.24%**
- **SOL**: TDA -15.96% vs B&H -41.82% — **TDA strategy wins by 25.86%**