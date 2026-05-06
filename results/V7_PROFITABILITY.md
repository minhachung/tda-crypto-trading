# V7 Profitability Test

**Hypothesis:** Profitability is achievable on ADA+SOL at 7d horizon with p>0.7 threshold + low-fee venue.

**Sample:** 365 days of hourly data, split into 3 equal-time regimes to test cross-regime robustness.

**Date:** 2026-05-05


## 1. Summary

| Scenario | n Signals | Direction Acc. | Mean Sharpe | Mean Return | Beat B&H | Max DD |
|----------|----------:|---------------:|------------:|------------:|---------:|-------:|
| coinbase_taker | 2911 | 68.46% | -2.93 | 0.04% | 63% | -0.11% |
| binance_maker | 2911 | 68.46% | -2.90 | 0.06% | 63% | -0.10% |

## 2. Per-Regime Stability

This is the critical 90-day-sample fix. We split 365 days into 3 sub-windows and check if the strategy works in EACH, not just on average.

### coinbase_taker

| Regime | Asset | Direction Acc. | Sharpe | Return | B&H | Beat B&H | Trades |
|-------:|-------|---------------:|-------:|-------:|----:|---------:|-------:|
| 0 | ADA | 71.62% | -1.74 | 0.03% | 3.60% | 60% | 1 |
| 0 | SOL | 63.04% | -3.24 | 0.09% | 3.61% | 20% | 2 |
| 1 | ADA | 62.55% | -5.03 | 0.15% | -11.89% | 80% | 7 |
| 1 | SOL | 71.39% | -4.84 | -0.07% | -7.69% | 80% | 5 |
| 2 | ADA | 71.45% | -2.71 | 0.05% | -4.82% | 80% | 1 |
| 2 | SOL | 70.69% | 0.00 | 0.00% | -5.77% | 60% | 0 |

### binance_maker

| Regime | Asset | Direction Acc. | Sharpe | Return | B&H | Beat B&H | Trades |
|-------:|-------|---------------:|-------:|-------:|----:|---------:|-------:|
| 0 | ADA | 71.62% | -1.78 | 0.04% | 3.60% | 60% | 1 |
| 0 | SOL | 63.04% | -3.26 | 0.10% | 3.61% | 20% | 2 |
| 1 | ADA | 62.55% | -4.96 | 0.19% | -11.89% | 80% | 7 |
| 1 | SOL | 71.39% | -4.84 | -0.04% | -7.69% | 80% | 5 |
| 2 | ADA | 71.45% | -2.56 | 0.05% | -4.82% | 80% | 1 |
| 2 | SOL | 70.69% | 0.00 | 0.00% | -5.77% | 60% | 0 |

## 3. Verdict

### Direction accuracy persists?

v6 paper: 76.75% (ADA, n=317) and 73.12% (SOL, n=345). v7 365-day: see Scenario summary above. If similar, signal is robust across the longer sample.

### Sharpe profitable on Binance.US?

- **NO**: Mean Sharpe -2.90, mean return 0.06%/fold. Even with Binance.US fees, the model loses money. The ADA+SOL+7d hypothesis from the v6 paper is NOT confirmed at 1-year scale.

### Cross-regime stability?

- **coinbase_taker**: per-regime accuracies = {0: np.float64(0.673), 1: np.float64(0.67), 2: np.float64(0.711)}. Stable across regimes ✓
- **binance_maker**: per-regime accuracies = {0: np.float64(0.673), 1: np.float64(0.67), 2: np.float64(0.711)}. Stable across regimes ✓

## 4. Recommendation

See verdict above for the data. If Sharpe is positive on Binance.US AND per-regime accuracies are all >55%, paper trade for 60 days before committing capital. Otherwise the v6 paper finding does not generalize to a profitable strategy.
