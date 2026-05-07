# V11: Leave-One-Asset-Out + Stronger Baselines

**Goal:** Resolve two methodological items:
1. Cross-asset transferability (LOAO)
2. Comparison against stronger baselines (XGB / LightGBM / CatBoost / momentum / vol-breakout / B&H / cash / returns-only)

**Date:** 2026-05-07
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Sample:** 1095 days hourly | Horizon: 72h (3.0d)

## 1. Leave-One-Asset-Out

For each held-out asset, the model is trained on the union of the other 6 assets and evaluated only on the held-out asset's signals.

### 1.1 Per-(model, held-out asset)

| Model | Held-Out | n Signals | Direction Acc | Wilson 95% | AUC | Sharpe | Return % |
|-------|----------|----------:|--------------:|-----------|----:|-------:|---------:|
| logistic | ADA | 287 | 82.23% | [77.39%, 86.22%] | 0.541 | -4.02 | 3.57% |
| logistic | AVAX | 273 | 74.36% | [68.87%, 79.18%] | 0.539 | -2.90 | 7.88% |
| logistic | BTC | 216 | 83.80% | [78.30%, 88.11%] | 0.536 | -7.55 | 2.75% |
| logistic | DOT | 394 | 70.56% | [65.88%, 74.84%] | 0.541 | -2.44 | 4.12% |
| logistic | ETH | 314 | 68.79% | [63.46%, 73.66%] | 0.537 | -4.82 | 3.98% |
| logistic | LINK | 321 | 71.03% | [65.84%, 75.72%] | 0.535 | -4.02 | 5.92% |
| logistic | SOL | 329 | 70.52% | [65.38%, 75.18%] | 0.539 | -7.11 | 1.54% |
| xgboost | ADA | 1281 | 74.55% | [72.09%, 76.86%] | 0.591 | -3.56 | 3.63% |
| xgboost | AVAX | 913 | 74.81% | [71.89%, 77.52%] | 0.589 | -3.77 | 5.63% |
| xgboost | BTC | 2706 | 46.19% | [44.32%, 48.08%] | 0.522 | -13.83 | 1.29% |
| xgboost | DOT | 1104 | 70.02% | [67.25%, 72.65%] | 0.580 | -3.89 | 2.33% |
| xgboost | ETH | 679 | 67.89% | [64.29%, 71.30%] | 0.542 | -4.96 | 1.82% |
| xgboost | LINK | 931 | 77.55% | [74.76%, 80.11%] | 0.587 | -3.46 | 9.31% |
| xgboost | SOL | 1244 | 67.20% | [64.54%, 69.75%] | 0.567 | -5.07 | 5.45% |
| lightgbm | ADA | 1282 | 72.54% | [70.04%, 74.92%] | 0.585 | -3.62 | 5.37% |
| lightgbm | AVAX | 905 | 73.26% | [70.28%, 76.04%] | 0.589 | -4.28 | 5.16% |
| lightgbm | BTC | 2858 | 47.59% | [45.76%, 49.42%] | 0.517 | -16.19 | 1.27% |
| lightgbm | DOT | 1070 | 68.41% | [65.56%, 71.13%] | 0.577 | -3.36 | 3.34% |
| lightgbm | ETH | 654 | 67.28% | [63.59%, 70.76%] | 0.537 | -5.28 | 0.93% |
| lightgbm | LINK | 873 | 77.89% | [75.02%, 80.52%] | 0.591 | -3.51 | 8.72% |
| lightgbm | SOL | 1241 | 66.56% | [63.89%, 69.13%] | 0.568 | -5.09 | 6.97% |
| catboost | ADA | 1040 | 73.85% | [71.09%, 76.43%] | 0.587 | -3.72 | 1.45% |
| catboost | AVAX | 828 | 73.79% | [70.69%, 76.67%] | 0.589 | -4.23 | 5.88% |
| catboost | BTC | 2522 | 45.92% | [43.98%, 47.87%] | 0.515 | -14.63 | 1.25% |
| catboost | DOT | 888 | 69.82% | [66.72%, 72.75%] | 0.577 | -4.31 | 1.60% |
| catboost | ETH | 508 | 67.52% | [63.33%, 71.45%] | 0.539 | -4.19 | 3.91% |
| catboost | LINK | 711 | 75.53% | [72.24%, 78.54%] | 0.586 | -3.39 | 9.25% |
| catboost | SOL | 1045 | 65.65% | [62.71%, 68.46%] | 0.562 | -4.94 | 4.69% |

### 1.2 Per-model pooled summary

| Model | Pooled Accuracy | Pooled Signals | Mean Sharpe |
|-------|----------------:|---------------:|------------:|
| catboost | 62.62% | 7542 | -5.63 |
| lightgbm | 63.39% | 8883 | -5.90 |
| logistic | 73.76% | 2134 | -4.69 |
| xgboost | 64.12% | 8858 | -5.50 |

## 2. Baseline Ladder

Each baseline is evaluated on the same pooled data with the same backtest cost model (0.001 fee per side, 0.0005 slippage per side).

### 2.1 Per-(baseline, asset)

| Baseline | Symbol | n Signals | Direction Acc | Wilson 95% | Sharpe | Return % | B&H Return |
|----------|--------|----------:|--------------:|-----------|-------:|---------:|-----------:|
| momentum_24h_20bps | ADA | 24675 | 48.58% | [47.95%, 49.20%] | -1.67 | -23.83% | -30.14% |
| momentum_24h_20bps | AVAX | 24938 | 49.69% | [49.07%, 50.31%] | -1.81 | -26.95% | -38.80% |
| momentum_24h_20bps | BTC | 23152 | 47.87% | [47.23%, 48.52%] | -3.31 | -16.42% | 197.32% |
| momentum_24h_20bps | DOT | 24688 | 49.40% | [48.77%, 50.02%] | -2.07 | -31.18% | -76.75% |
| momentum_24h_20bps | ETH | 23970 | 48.68% | [48.05%, 49.31%] | -2.67 | -25.49% | 31.25% |
| momentum_24h_20bps | LINK | 24893 | 48.52% | [47.90%, 49.14%] | -1.82 | -23.07% | 48.41% |
| momentum_24h_20bps | SOL | 24856 | 48.02% | [47.40%, 48.64%] | -1.71 | -21.04% | 318.58% |
| vol_breakout_z15 | ADA | 3731 | 49.88% | [48.28%, 51.48%] | -1.81 | -27.91% | -30.14% |
| vol_breakout_z15 | AVAX | 3387 | 50.75% | [49.07%, 52.43%] | -1.68 | -17.96% | -38.80% |
| vol_breakout_z15 | BTC | 3599 | 49.82% | [48.19%, 51.45%] | -3.59 | -25.26% | 197.32% |
| vol_breakout_z15 | DOT | 3752 | 51.55% | [49.95%, 53.14%] | -1.89 | -25.26% | -76.75% |
| vol_breakout_z15 | ETH | 3600 | 49.94% | [48.31%, 51.58%] | -2.41 | -18.64% | 31.25% |
| vol_breakout_z15 | LINK | 3508 | 51.20% | [49.54%, 52.85%] | -1.77 | -18.85% | 48.41% |
| vol_breakout_z15 | SOL | 3820 | 50.13% | [48.55%, 51.72%] | -1.80 | -25.96% | 318.58% |
| buy_and_hold | ADA | 1 | 100.00% | [20.65%, 100.00%] | -0.64 | -3.03% | -30.14% |
| buy_and_hold | AVAX | 1 | 100.00% | [20.65%, 100.00%] | -0.60 | -3.90% | -38.80% |
| buy_and_hold | BTC | 1 | 0.00% | [0.00%, 79.35%] | -0.78 | 19.64% | 197.32% |
| buy_and_hold | DOT | 1 | 100.00% | [20.65%, 100.00%] | -1.15 | -7.68% | -76.75% |
| buy_and_hold | ETH | 1 | 100.00% | [20.65%, 100.00%] | -1.01 | 3.09% | 31.25% |
| buy_and_hold | LINK | 1 | 100.00% | [20.65%, 100.00%] | -0.48 | 4.80% | 48.41% |
| buy_and_hold | SOL | 1 | 100.00% | [20.65%, 100.00%] | -0.18 | 31.73% | 318.58% |
| cash | ADA | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | -30.14% |
| cash | AVAX | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | -38.80% |
| cash | BTC | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | 197.32% |
| cash | DOT | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | -76.75% |
| cash | ETH | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | 31.25% |
| cash | LINK | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | 48.41% |
| cash | SOL | 0 | 50.00% | [0.00%, 100.00%] | 0.00 | 0.00% | 318.58% |
| returns_only_logistic | ADA | 881 | 64.13% | [60.91%, 67.23%] | -14.88 | 0.37% | -30.14% |
| returns_only_logistic | AVAX | 1824 | 53.07% | [50.78%, 55.35%] | -14.21 | 0.31% | -38.80% |
| returns_only_logistic | BTC | 127 | 57.48% | [48.79%, 65.73%] | -136.43 | 0.10% | 197.32% |
| returns_only_logistic | DOT | 822 | 65.69% | [62.38%, 68.86%] | -14.95 | 0.30% | -76.75% |
| returns_only_logistic | ETH | 378 | 67.72% | [62.85%, 72.24%] | -34.66 | 0.08% | 31.25% |
| returns_only_logistic | LINK | 1217 | 55.38% | [52.58%, 58.15%] | -17.23 | 1.04% | 48.41% |
| returns_only_logistic | SOL | 1104 | 56.97% | [54.03%, 59.87%] | -22.65 | -0.34% | 318.58% |

### 2.2 Per-baseline pooled summary

| Baseline | Pooled Accuracy | Total Signals | Mean Sharpe | Mean Return % |
|----------|----------------:|--------------:|------------:|--------------:|
| buy_and_hold | 85.71% | 7 | -0.69 | 6.38% |
| cash | 0.00% | 0 | 0.00 | 0.00% |
| momentum_24h_20bps | 48.69% | 171172 | -2.15 | -23.99% |
| returns_only_logistic | 58.32% | 6353 | -36.43 | 0.27% |
| vol_breakout_z15 | 50.46% | 25397 | -2.14 | -22.83% |

## 3. Verdict

Read this section against the v9/v10 headline numbers.

- **Cross-asset transferability:** if LOAO accuracy stays > 50% (Wilson lower bound) on most held-out assets, the multi-asset story holds. If it drops to chance only on specific assets, those assets are likely riding asset-specific quirks rather than transferable structure.
- **Vs. stronger baselines:** XGBoost/LightGBM/CatBoost results bracket the realistic upper bound for a Base+TDA gradient-boosted classifier. If the v9 logistic configuration is within 1-2pp of the boosted ceiling, the v9 model selection was reasonable; a much larger gap would suggest a boosted model is preferable.
- **Vs. momentum / vol-breakout / returns-only logistic:** if the TDA pipeline beats these by more than the Wilson-CI margin, the case for TDA-feature contribution is strengthened. If a returns-only logistic matches it, TDA's marginal value is below the noise floor of the experiment.