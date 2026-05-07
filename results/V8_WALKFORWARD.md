# V8: Continuous Walk-Forward Profitability Test

**Hypothesis:** Continuous trading (not k-fold) with realistic position sizing
will reveal whether the v6/v7 direction-accuracy edge converts to profit.

**Setup:** 365 days hourly data, walk-forward from day 60, 7d holds, p>0.65 threshold, up to 50% position size.
**Date:** 2026-05-07


## Results

| Scenario | Asset | Trades | Strategy | Buy-Hold | Diff | Sharpe | Max DD |
|----------|-------|-------:|---------:|---------:|-----:|-------:|-------:|
| binance_maker | ADA | 37 | -5.20% | -60.38% | +55.17% | -0.30 | -17.57% |
| binance_maker | SOL | 35 | -15.01% | -47.18% | +32.17% | -1.45 | -19.41% |
| coinbase_taker | ADA | 37 | -9.49% | -60.38% | +50.89% | -0.61 | -18.08% |
| coinbase_taker | SOL | 35 | -18.34% | -47.18% | +28.85% | -1.82 | -21.63% |

## Trade Counts (vs k-fold v7)

v7 had ~30 actual trades total across all folds. v8 has:

- binance_maker: 72 trades
- coinbase_taker: 72 trades

This is the trade count we need for proper Sharpe estimation.

## Verdict


### Lost money on Binance.US:
- **ADA**: -5.20% over 37 trades, Sharpe -0.30
- **SOL**: -15.01% over 35 trades, Sharpe -1.45

### Vs Buy-Hold:
- **ADA**: TDA -5.20% vs B&H -60.38% — **TDA strategy wins by 55.17%**
- **SOL**: TDA -15.01% vs B&H -47.18% — **TDA strategy wins by 32.17%**