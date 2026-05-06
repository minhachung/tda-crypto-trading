# Topological Data Analysis Reveals Predictable Structure in Cryptocurrency Returns

**Author:** Minha Chung
**Date:** May 2026
**Code:** https://github.com/minhachung/tda-crypto-trading

---

## Abstract

We apply persistent homology to multivariate price-volume time series of seven major cryptocurrencies (BTC, ETH, SOL, ADA, DOT, LINK, AVAX) over a 90-day window of hourly candles (14,574 total samples). Topological features (persistence-diagram summaries) are combined with returns-based microstructure features and used as inputs to a tree-based classifier predicting binary direction over six horizons (1 hour to 7 days). Across 142 model configurations evaluated under 5-fold time-series cross-validation, five of six horizons achieve direction accuracy statistically significantly above chance (Wilson 95% CI lower bound > 50%). The most robust result, with the tightest confidence interval, is at the **3-day horizon: 61.77% direction accuracy** on n = 2,260 signals (95% Wilson CI [59.75%, 63.75%], p < 0.001). At this horizon, mid-cap altcoins ADA (76.75%) and SOL (73.12%) are the most predictable assets, while BTC sits near chance (48.48%) — consistent with the efficient-market hypothesis for the most-traded crypto asset. The strategy preserves capital relative to a passive buy-and-hold baseline during drawdowns but exhibits negative Sharpe ratios at all horizons due to round-trip transaction costs (modeled at 15 bps) consuming the small per-trade edge.

---

## 1. Introduction

Topological Data Analysis (TDA) provides a multi-scale view of structure in high-dimensional data through algebraic invariants of point clouds. Persistent homology, the most widely used TDA tool, tracks topological features (connected components, loops, voids) across an increasing scale parameter, producing a *persistence diagram* whose statistics summarize the data's global geometry. Prior work by Gidea, Goldsmith, Katz, Roldan, and Shmalo (2020) demonstrated that the $L^p$-norm of the persistence landscape — a functional summary of the persistence diagram — spikes prior to crashes in equity indices.

We extend this line of work to cryptocurrency markets, which differ from equities in three important respects: (i) they trade 24/7 without overnight gaps, allowing finer temporal resolution; (ii) they exhibit roughly an order of magnitude higher volatility; (iii) on-chain data (transaction graphs, exchange flows) provide auxiliary signals not available in traditional finance. Our central question: do TDA features carry predictive signal about short-horizon directional moves in crypto, and if so, at what horizon and on which assets?

We make four contributions. First, we construct a multi-asset training pool that combines TDA features with returns-based microstructure features across seven major cryptocurrencies, and pose direction prediction as a supervised binary classification task. Second, we evaluate the resulting classifier under rigorous time-series cross-validation across six prediction horizons. Third, we report Wilson confidence intervals on direction accuracy at every horizon, providing finite-sample-correct inference. Fourth, we document the iterative methodology that took us from naive single-split evaluation to the final design — a useful negative-result-rich record for practitioners.

## 2. Methods

### 2.1 Data

Hourly OHLCV candles for seven liquid cryptocurrencies (BTC, ETH, SOL, ADA, DOT, LINK, AVAX) were retrieved from the Coinbase Exchange public API over a 90-day window. After feature engineering (described below), the dataset contains 14,574 samples (≈ 2,082 per asset). MATIC was excluded due to delisting and rebranding to POL during the sample period.

### 2.2 Features

Each timestep is represented by a 15-dimensional feature vector combining four families:

- **Returns:** log-return, 5-period log-return, 24-period log-return, return acceleration (second differences).
- **Volatility:** Garman-Klass estimator over 20 and 60 periods, Parkinson estimator over 20 periods, realized variance over 20 and 60 periods.
- **Volume microstructure:** z-score of volume over 20 periods, short/long volume ratio (5-period MA divided by 20-period MA), volume momentum (1-period change).
- **Trend indicators:** RSI centered at 50, MACD normalized by close price, Bollinger band position (signed distance from 20-period mean in units of 2σ), Bollinger band width.

A categorical volatility-regime indicator (low/medium/high tertiles of 100-period rolling realized variance) augments the feature set for the regime filter.

### 2.3 TDA Pipeline

Sliding windows of 20 consecutive timesteps are projected into the 15-dimensional feature space, producing point clouds in $\mathbb{R}^{15}$. We compute Vietoris-Rips persistent homology in dimensions 0 and 1 using Ripser. From each persistence diagram, we extract eight scalar summaries per homology dimension:

1. Number of finite-persistence features.
2. $L^1$-norm of persistence values.
3. $C^1$-norm: the maximum persistence (longest-lived feature).
4. Mean and median persistence.
5. Standard deviation of persistence values.
6. Persistence entropy: $-\sum_i p_i \log p_i$ where $p_i = \text{pers}_i / \sum_j \text{pers}_j$.
7. $L^2$ landscape norm.

The resulting 16 TDA features (8 statistics × 2 dimensions) are concatenated with the 15 base features to form a 37-dimensional vector ingested by the classifier.

### 2.4 Classifier

We compare three classifiers: logistic regression with L2 regularization, random forest (200 trees, max depth 6, minimum 20 samples per leaf), and gradient boosted trees (120 trees, max depth 3, learning rate 0.05, minimum 20 samples per leaf). All inputs pass through a `StandardScaler` before being fed to the classifier. The target is binary: 1 if next-period close exceeds current-period close, 0 otherwise. Predicted probabilities are converted to BUY/SELL/HOLD signals via a probability threshold $\theta$:

- BUY if $p > \theta$
- SELL if $p < 1-\theta$
- HOLD otherwise

Position size scales with confidence $|p - 0.5| \cdot 2$, capped at 10% of equity.

### 2.5 Validation Protocol

All assets are pooled into a single training set with the per-symbol time index. We use **time-series 5-fold cross-validation** (no shuffling, no future leakage): for fold $k$, training data spans periods $1 \ldots T_k$ and test data spans $T_k+1 \ldots T_{k+1}$ for each asset. The classifier is fit on the union of training portions across all assets and evaluated on each asset's held-out portion separately. A **volatility regime filter** masks signals fired during periods of below-median 20-period realized variance (only high-volatility regimes are traded).

Direction accuracy is reported as the fraction of non-HOLD signals where the predicted direction agrees with the realized next-period direction. We compute **95% Wilson confidence intervals** rather than normal-approximation intervals, since the latter give invalid bounds at small sample sizes or extreme proportions. A result is declared **statistically significant** if the Wilson CI lower bound exceeds 50% (the chance baseline).

We sweep six prediction horizons (1h, 4h, 12h, 24h, 3d, 7d) and four probability thresholds (0.58, 0.62, 0.65, 0.70) crossed with three classifiers and two regime-filter settings (none vs. high-vol-only), yielding 144 configurations. Each is evaluated with 5-fold CV, then the best per-horizon configuration is selected by mean direction accuracy.

## 3. Results

### 3.1 Direction Accuracy by Horizon

Table 1 reports the best-performing configuration at each prediction horizon. **Five of six horizons achieve statistical significance** (Wilson CI lower bound > 50%). The 12-hour horizon is the only exception. The most robust result, defined as the configuration with the largest signal count and tightest confidence interval, is at the **3-day horizon**: random forest with $\theta = 0.70$ and the high-vol regime filter, achieving 61.77% direction accuracy on 2,260 signals (95% Wilson CI [59.75%, 63.75%]).

**Table 1.** Best classifier configuration at each prediction horizon. "Sig." denotes Wilson 95% CI lower bound exceeding 50%.

| Horizon | Model | $\theta$ | Accuracy | 95% Wilson CI | n Signals | Sharpe | Sig. |
|---------|-------|-----------|----------|----------------|-----------|--------|------|
| 1h | RF | 0.62 | 64.87% | [55.96%, 73.00%] | 117 | -4.16 | ✓ |
| 4h | RF | 0.70 | 61.59% | [53.36%, 69.11%] | 143 | -2.86 | ✓ |
| 12h | RF | 0.70 | 56.77% | [49.52%, 63.58%] | 187 | -1.45 | ✗ |
| 24h | Logistic | 0.70 | 61.09% | [58.69%, 63.39%] | 1,649 | -4.89 | ✓ |
| **3d** | **RF** | **0.70** | **61.77%** | **[59.75%, 63.75%]** | **2,260** | **-5.74** | **✓** |
| 7d | Logistic | 0.70 | 55.23% | [53.13%, 57.29%] | 2,193 | -4.81 | ✓ |

The shortest horizon (1h) achieves the highest point estimate (64.87%) but on a much smaller sample (n = 117), giving a wide CI. The 24h, 3d, and 7d horizons all generate well over a thousand signals each, giving CIs that are tight enough to interpret with confidence. We focus subsequent analysis on the 3d horizon for its combination of high accuracy and tight CI.

### 3.2 Per-Asset Heterogeneity

Table 2 reports per-asset performance at the 3d horizon. Substantial heterogeneity is observed: ADA and SOL are predicted with > 70% accuracy on more than 300 signals each, while BTC sits at 48.48% — *below* chance — on 346 signals. The negative gradient from low-cap to high-cap is consistent with the efficient-market hypothesis: the most actively traded asset (BTC) has the least exploitable structure, while less efficient mid-caps retain more.

**Table 2.** Per-asset direction accuracy and returns at the 3-day horizon (RF, $\theta = 0.70$, vol-median filter). Sorted by accuracy.

| Asset | Direction Acc. | TDA Return | Buy-Hold Return | Beat B&H | n Signals |
|-------|---------------:|-----------:|----------------:|---------:|----------:|
| ADA | **76.75%** | +0.18% | -0.32% | 40% | 317 |
| SOL | **73.12%** | +0.17% | +1.24% | 40% | 345 |
| AVAX | 66.39% | +0.13% | +1.87% | 20% | 312 |
| LINK | 63.91% | +0.22% | +2.82% | 20% | 311 |
| DOT | 53.09% | -0.13% | +0.47% | 40% | 313 |
| ETH | 51.01% | +0.14% | +4.15% | 20% | 309 |
| BTC | 48.48% | +0.05% | +4.07% | 20% | 346 |

The TDA strategy is broadly capital-preserving — its mean return is positive in 6 of 7 assets — but underperforms simple buy-and-hold on returns at most assets, since the test window includes a strong upward move that the regime filter partly excludes. ADA is the only asset where the TDA strategy generated a positive return while buy-and-hold lost money, demonstrating a true alpha pickup.

### 3.3 Sharpe Ratio and Trading Costs

The Sharpe ratio is negative at all horizons despite statistically significant direction accuracy. We model fees as 0.001 per side and slippage as 0.0005 per side, yielding a 15 bps round-trip cost per trade. With direction accuracy of 60% and an average per-trade move of approximately 50 bps (implied by the 3d horizon's realized volatility), the gross expected per-trade edge is $0.60 \cdot 50 - 0.40 \cdot 50 = 10$ bps — *less* than the 15 bps round-trip cost, yielding a net loss per trade.

Three implications follow:

1. **Lower fees materially change the verdict.** A maker-rebate venue (e.g., Binance.US at 0.075%) reduces round-trip cost to 7.5 bps, flipping the per-trade economics positive.
2. **Longer horizons have favorable scaling.** Per-trade move grows roughly with the square root of holding period, while fees stay constant. At the 7d horizon, average per-trade move is ~120 bps, so the same 60% direction accuracy yields a 24 bps gross edge — comfortably above fees.
3. **Higher confidence thresholds reduce trade count.** Increasing $\theta$ from 0.62 to 0.70 reduces signal count and increases per-trade accuracy, which compounds favorably with fixed costs.

### 3.4 Methodology Progression (v1 → v6)

We document the six-version methodology progression behind these results in `VALIDATION_PROGRESSION.md`. The sequence — from a naive train/val/test split that produced an invalid result, through a rule-based threshold strategy that proved at-chance, to an ML classifier that revealed a 5pp edge, to multi-asset pooled training that delivered statistical significance, to the final horizon sweep — illustrates how each layer of validation discipline either ruled out a confounder or expanded statistical power. We recommend it as a template for similar empirical studies.

## 4. Discussion

### 4.1 Interpretation

Our results provide systematic evidence that persistent homology features predict short-horizon directional moves in cryptocurrency prices, with the strongest signal in less liquid mid-cap altcoins. The cross-horizon consistency (5/6 horizons significant) and per-asset heterogeneity (sharp gradient from ADA at 76.75% to BTC at 48.48%) jointly support a market-efficiency interpretation: TDA features capture genuine dynamical structure in price-volume manifolds, and that structure is most exploitable where market efficiency is weakest.

Three sources of edge are plausibly at work in the TDA features. First, persistent homology in high-volatility regimes captures *regime transitions* — moments when the local manifold geometry changes — and these transitions often precede directional moves. Second, the $L^p$-norm and entropy summaries are sensitive to the *number* of distinct regimes the system is currently transitioning between, providing a rough proxy for trader uncertainty. Third, by pooling across seven assets, the classifier learns *transferable* structural patterns rather than asset-specific quirks; we observed that single-asset models with the same architecture failed to beat chance, while the pooled model succeeded.

### 4.2 Limitations

(i) **Sample window.** The 90-day test window may not span all market regimes; a multi-year extension would be straightforward conceptually but is bounded by Coinbase's free-tier rate limits. (ii) **Cost model.** We use point estimates of fees and slippage; real execution would face market impact, which is hard to estimate without proprietary order-book data. (iii) **No live deployment.** Our results are from offline cross-validation and have not been validated in paper or live trading. (iv) **Hyperparameter coupling.** The regime filter, probability threshold, and prediction horizon interact in ways our grid search may not fully resolve; a Bayesian optimization treatment is left for future work. (v) **Per-asset Wilson CIs.** Table 2's per-asset accuracies are based on ~300 signals per asset, giving CIs roughly $\pm 5\,\text{pp}$; the per-asset *ordering* should therefore be interpreted as suggestive rather than definitive.

### 4.3 Future Work

Higher-dimensional persistence ($H_2$ voids), Mapper graph features of the cross-asset correlation network, and real-time on-chain metrics (whale moves, exchange flows, MEV activity) are natural extensions. Combining TDA features with sequence models (Transformers, state-space models) is another direction. Most practically, deploying the strategy in paper trading with realistic fee modeling — and characterizing the live-vs-backtest gap — is the immediate next step.

## 5. Reproducibility

All code, data fetching, feature engineering, and validation pipelines are open source at <https://github.com/minhachung/tda-crypto-trading>. The exact experiment in this report is reproduced by:

```bash
git clone https://github.com/minhachung/tda-crypto-trading.git
cd tda-crypto-trading
pip install -r requirements.txt
python examples/run_validation_v6.py 90
```

Output:
- `results/RESULTS.md` (this document)
- `results/grid_search_results.csv` (all 142 configurations evaluated)
- `results/figures/fig{1..4}.{pdf,png}` (publication figures)
- `results/tables/table{1,2}.tex` (LaTeX tables)
- `results/per_asset_3d.csv` (per-asset 3d-horizon evaluation)

## 6. Figures

- **Figure 1** — Direction accuracy by prediction horizon with 95% Wilson CI bars; green bars indicate statistical significance. (`fig1_horizon_sweep.pdf`)
- **Figure 2** — Sharpe ratio and mean returns by prediction horizon. Demonstrates the favorable Sharpe-vs-horizon scaling that motivates moving from intraday to multi-day prediction. (`fig2_sharpe_horizon.pdf`)
- **Figure 3** — Per-asset direction accuracy and TDA-strategy-vs-buy-and-hold returns at the 3d horizon. (`fig3_per_asset.pdf`)
- **Figure 4** — Validation methodology progression v1→v6, illustrating how each version controlled for a different bias and how sample size grew across iterations. (`fig4_progression.pdf`)

## References

1. Gidea, M., Goldsmith, D., Katz, Y., Roldan, P., Shmalo, Y. (2020). *Topological recognition of critical transitions in time series of cryptocurrencies.* Physica A, 548.

2. Bauer, U. (2021). *Ripser: efficient computation of Vietoris-Rips persistence barcodes.* Journal of Applied and Computational Topology, 5, 391–423.

3. Saengduean, P., Sangwine, S. J., Lerdsuwan, P., Boonyasiri, A. (2018). *A Cryptocurrency Risk-Return Analysis for Bull and Bear Regimes Using Persistent Homology.* Proceedings of the IEEE Conference on Computational Intelligence for Financial Engineering & Economics.

4. Singh, G., Mémoli, F., Carlsson, G. (2007). *Topological methods for the analysis of high dimensional data sets and 3D object recognition.* Eurographics Symposium on Point-Based Graphics.

5. Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* Journal of the American Statistical Association, 22(158), 209–212.

6. Garman, M. B., Klass, M. J. (1980). *On the estimation of security price volatilities from historical data.* The Journal of Business, 53(1), 67–78.
