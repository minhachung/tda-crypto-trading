# TDA-Crypto-Trading

A research codebase for testing whether **persistent homology** features predict short-horizon directional moves in cryptocurrency markets.

## What this is

We compute Vietoris-Rips persistent homology in dimensions 0 and 1 over 20-hour sliding windows of a 15-feature representation (returns, volatility, volume z-score, trend indicators) for seven liquid cryptocurrencies. We summarise each persistence diagram with eight scalar statistics per dimension (16 TDA features), concatenate them with the base features, and use a random forest classifier to predict binary direction over six horizons (1h, 4h, 12h, 24h, 3d, 7d).

## What we found

| Horizon | Best Model | Direction Accuracy | 95% Wilson CI | n Signals | Significant |
|---------|------------|-------------------:|---------------|----------:|------------|
| 1h | RF, θ=0.62 | 64.87% | [55.96%, 73.00%] | 117 | ✅ |
| 4h | RF, θ=0.70 | 61.59% | [53.36%, 69.11%] | 143 | ✅ |
| 12h | RF, θ=0.70 | 56.77% | [49.52%, 63.58%] | 187 | ❌ |
| 24h | Logistic, θ=0.70 | 61.09% | [58.69%, 63.39%] | 1,649 | ✅ |
| **3d** | **RF, θ=0.70** | **61.77%** | **[59.75%, 63.75%]** | **2,260** | ✅ |
| 7d | Logistic, θ=0.70 | 55.23% | [53.13%, 57.29%] | 2,193 | ✅ |

Five of six horizons achieve direction accuracy with Wilson CI lower bound > 50%. The **3-day horizon** is the most robust: tightest confidence interval, largest sample size, deeply significant.

In a continuous walk-forward backtest on ADA over 365 days, the strategy turned $10,000 into **$11,199** while the spot price collapsed -55% (buy-and-hold ended at $4,475). Sharpe 0.95.

!!! note "Limitation"
    Across the 7-asset pool with shared model parameters, the per-trade gross edge is small enough that round-trip transaction costs (typically 15 bps on a low-fee venue) consume most of it. The model demonstrates **predictability**, but profitability requires either lower fees, longer horizons, asset-specific tuning, or higher confidence thresholds.

## Where to go next

- **[Tutorial](tutorial.md)**: build intuition for what persistent homology computes on a simple toy example
- **[Validation Methodology](validation.md)**: the full v1→v9 progression that controls for cherry-picking, lookahead, sample size, and feature attribution biases
- **[API Reference](api.md)**: function-level docs for the codebase
- **[Paper](https://github.com/minhachung/tda-crypto-trading/blob/main/results/PAPER.pdf)**: 12-page formatted paper

## Quick start

```bash
git clone https://github.com/minhachung/tda-crypto-trading.git
cd tda-crypto-trading
pip install -r requirements.txt
python examples/run_validation_v6.py 90
```

This reproduces the headline figures of the paper in ~30 minutes on a Mac M1.

## Author

Minha Chung — [GitHub](https://github.com/minhachung)
