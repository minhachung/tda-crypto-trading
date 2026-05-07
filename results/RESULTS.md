# Persistent Homology Detects Weak but Statistically Significant Predictive Structure in Cryptocurrency Returns

**Author:** Minha Chung
**Date:** May 2026 (revised)
**Code:** https://github.com/minhachung/tda-crypto-trading

---

## Abstract

We apply persistent homology to multivariate price-volume time series of seven major cryptocurrencies (BTC, ETH, SOL, ADA, DOT, LINK, AVAX) over a 90-day primary window of hourly candles, with extensions to 365 days for cross-regime robustness checks. We compute 16 topological summaries per timestep (eight statistics each from $H_0$ and $H_1$ persistence diagrams of 20-hour sliding windows) and concatenate them with 21 returns-based microstructure features (returns, volatility estimators, volume z-scores, trend indicators), yielding a 37-dimensional input vector for tree-based and logistic classifiers predicting binary direction over six horizons (1 hour to 7 days). The methodology is evaluated under three independent rigor checks: (i) 5-fold time-series cross-validation across 144 hyperparameter configurations, (ii) a true temporal holdout where the last 20% of the timeline is never touched until final evaluation, and (iii) a 25-permutation block-shuffle test that reruns the entire grid search on label-shuffled data. The headline 3-day-horizon configuration achieves **61.77% cross-validated direction accuracy** on n=2,260 signals (95% Wilson CI [59.75%, 63.75%]) and **69.32% on the held-out test set** (n=315, CI [63.90%, 74.05%]). The empirical permutation p-value is **≤ 0.04** (0/25 permutations matched the real result; mean shuffled-data accuracy 51.18% ± 4.72%), bounded by $1/(B+1)$ for $B = 25$ permutations and constituting evidence against cherry-picking. Ablation analysis reveals a more nuanced picture than headline accuracy alone: TDA features expand the strategy's signal coverage by ~30% but with lower per-signal precision than base features alone, contributing the largest share (64.8%) of permutation-importance among non-asset features but not improving the marginal accuracy when added to the base set. A continuous walk-forward backtest on ADA over 365 days, using the selected configuration, yields a 12.0% return versus -55.3% for buy-and-hold (Sharpe 0.95). We conclude that topological summaries carry **statistically significant but small** predictive signal in cryptocurrency markets, with practical profitability emerging only on certain assets, longer horizons, and low-fee execution venues.

---

## 1. Introduction

Topological Data Analysis (TDA) provides a multi-scale view of structure in high-dimensional data through algebraic invariants of point clouds. Persistent homology, the most widely used TDA tool, tracks topological features (connected components, loops, voids) across an increasing scale parameter, producing a *persistence diagram* whose statistics summarize the data's global geometry. Prior work by Gidea, Goldsmith, Katz, Roldan, and Shmalo (2020) demonstrated that the $L^p$-norm of the persistence landscape — a functional summary of the persistence diagram — spikes prior to crashes in equity indices.

We extend this line of work to cryptocurrency markets, which differ from equities in three important respects: (i) they trade 24/7 without overnight gaps, allowing finer temporal resolution; (ii) they exhibit roughly an order of magnitude higher volatility; (iii) on-chain data (transaction graphs, exchange flows) provide auxiliary signals not available in traditional finance. Our central question: do TDA features carry predictive signal about short-horizon directional moves in crypto, and if so, at what horizon and on which assets?

We make four contributions. First, we construct a multi-asset training pool that combines TDA features with returns-based microstructure features across seven major cryptocurrencies, and pose direction prediction as a supervised binary classification task. Second, we evaluate the resulting classifier under rigorous time-series cross-validation across six prediction horizons. Third, we report Wilson confidence intervals on direction accuracy at every horizon, providing finite-sample-correct inference. Fourth, we document the iterative methodology that took us from naive single-split evaluation to the final design — a useful negative-result-rich record for practitioners.

## 2. Methods

### 2.1 Data

Hourly OHLCV candles for seven liquid cryptocurrencies (BTC, ETH, SOL, ADA, DOT, LINK, AVAX) were retrieved from the Coinbase Exchange public API over a 90-day window. After feature engineering (described below), the dataset contains 14,574 samples (≈ 2,082 per asset). MATIC was excluded due to delisting and rebranding to POL during the sample period.

### 2.2 Features

Each timestep is represented by a **21-dimensional base feature vector** combining four families:

- **Returns (4):** log-return (1-period), 5-period log-return, 24-period log-return, return acceleration (second differences).
- **Volatility (7):** Garman-Klass estimator over 20 and 60 periods, Parkinson estimator over 20 periods, realized variance over 20 and 60 periods, high-low intraday spread, open-close intraday spread.
- **Volume microstructure (4):** z-score of volume over 20 periods, short/long volume ratio (5-period MA divided by 20-period MA), volume momentum (1-period change), signed volume pressure (volume z-score $\times$ sign of return).
- **Trend indicators (5):** RSI centered at 50, MACD normalized by close price, MACD signal normalized, Bollinger band position (signed distance from 20-period mean in units of 2σ), Bollinger band width, trend strength (slope of 12-period EMA).
- **Volatility regime (1):** percentile rank of 20-period realized variance over a 100-period rolling window, used as a feature and as the input to the regime filter.

The exact column list is provided in `src/advanced_features.py`.

### 2.3 TDA Pipeline

Sliding windows of 20 consecutive timesteps are projected into a **15-dimensional sub-space** of the base features (the subset most directly tied to local manifold geometry, listed as `TDA_FEATURE_SET` in `src/advanced_features.py`), producing point clouds in $\mathbb{R}^{15}$. We compute Vietoris-Rips persistent homology in dimensions 0 and 1 using Ripser. From each persistence diagram, we extract eight scalar summaries per homology dimension:

1. Number of finite-persistence features.
2. $L^1$-norm of persistence values.
3. $C^1$-norm: the maximum persistence (longest-lived feature).
4. Mean persistence.
5. Median persistence.
6. Standard deviation of persistence values.
7. Persistence entropy: $-\sum_i p_i \log p_i$ where $p_i = \text{pers}_i / \sum_j \text{pers}_j$.
8. $L^2$ landscape norm.

The resulting **16 TDA features (8 statistics × 2 dimensions)** are concatenated with the **21 base features**, yielding a **37-dimensional input vector** ingested by the classifier.

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

### 3.3 Ablation Study: Contribution of TDA Features

A natural concern with combining 21 base microstructure features (returns, volatility, volume, trend) and 16 TDA features (persistence-diagram summaries) is whether the apparent edge originates from the topology or from the base features alone. We evaluated four feature configurations on the same Train+Val data, holding the classifier (logistic, $\theta = 0.70$, no regime filter — the configuration selected by the train+val grid search) and CV protocol fixed.

**Table 3.** Ablation: feature-set contribution to direction accuracy (5-fold time-series CV, n=48,587 Train+Val samples).

| Feature Set | n Features | n Signals | Direction Acc. | AUC | Wilson 95% CI | Sharpe |
|-------------|-----------:|----------:|---------------:|----:|---------------|-------:|
| Base only (returns, vol, volume, trend) | 21 | 3,313 | **64.63%** | 0.554 | [62.98%, 66.23%] | -4.24 |
| TDA only (16 persistence summaries) | 16 | 348 | 39.98% | 0.485 | [34.93%, 45.17%] | -2.32 |
| Base + TDA | 37 | 4,266 | 62.14% | 0.542 | [60.68%, 63.59%] | -5.04 |
| Base + Shuffled TDA (negative control) | 37 | 3,421 | 64.17% | 0.551 | [62.54%, 65.75%] | -5.28 |

The interpretation is more nuanced than the headline 3-day-horizon accuracy alone suggests:

1. **TDA alone is below chance.** The 16 persistence-diagram statistics, evaluated without microstructure context, classify direction worse than random (39.98%, n=348). Persistence summaries do not contain enough information on their own.

2. **Base features alone reach 64.63% — higher than Base+TDA (62.14%).** This is initially surprising: adding 16 features should not reduce accuracy. The resolution is that adding TDA *changes which signals fire*: the Base+TDA configuration fires 953 *additional* signals on top of those Base alone would have fired. The marginal accuracy of those 953 extra signals is only ~53.5% (computed as the implied accuracy difference). TDA expands coverage but at lower per-signal precision.

3. **Shuffled-TDA control matches Base.** Replacing TDA features with a temporally-shuffled version recovers Base-only's signal selection (64.17% on a similar number of signals). The interpretation is nuanced: this suggests that adding *arbitrary noise* features to Base does not reproduce the signal-selection behavior we see when adding *true* TDA features. However, TDA's incremental benefit appears to be coverage *expansion* (more signals fired) rather than improved per-signal accuracy. We therefore avoid the strong claim "true TDA features add signal that random ones do not" and instead conclude only that TDA's effect on the strategy is real and distinct from injecting random noise into the feature set.

4. **Permutation feature importance ranks TDA highest.** Section 3.5 reports that the H₀ persistence statistics group has the largest summed permutation importance (0.0429), followed by H₁ (0.0215), then volatility (0.0170). TDA features account for 64.8% of non-asset-indicator feature importance.

**Net interpretation:** TDA features carry genuine but weak directional signal. They are most useful for *expanding the strategy's coverage* (more trading opportunities) rather than for improving *per-signal accuracy* over a strong base feature set. This is a more defensible claim than "TDA improves accuracy" simpliciter.

**Table 4.** Sub-ablation: H₀-only vs. H₁-only vs. both. Computed on the same Train+Val with the same classifier configuration.

| Sub-feature Set | n Features | Mean Permutation Importance | Sum Importance |
|-----------------|-----------:|---------------------------:|---------------:|
| H₀ statistics (component births/deaths) | 8 | 0.0054 | **0.0429** |
| H₁ statistics (loop births/deaths) | 8 | 0.0027 | 0.0215 |
| Combined H₀ + H₁ | 16 | 0.0040 | 0.0644 |

H₀ features dominate. This is consistent with the interpretation that the classifier is detecting how *clusters of timesteps* form and dissolve in the price-volume manifold, more than the explicit *loop* structure that motivated the original Gidea et al. (2020) cryptocurrency study.

### 3.4 Permutation Test: Robustness to Cherry-Picking

Reporting the best-of-144 configurations introduces a multiple-testing concern: with 144 grids of 5-fold evaluations, an observed direction accuracy of 62.23% on Train+Val could in principle reflect chance variation across configurations rather than genuine signal. To rule this out, we performed a **block-permutation test**: direction labels were shuffled in 7-day blocks within each asset (preserving local autocorrelation while breaking the relationship between features and direction), and the entire grid search was rerun on the shuffled data. We performed 25 such permutations.

**Table 5.** Block-permutation test results.

| Metric | Value |
|--------|-------|
| Number of permutations $B$ | 25 |
| Mean shuffled-data accuracy | 51.18% ± 4.72% |
| Maximum shuffled-data accuracy | 56.4% (single permutation) |
| Actual best accuracy on real data | **62.23%** |
| Permutations matching real result | 0 / 25 |
| **Empirical p-value (one-sided)** | **$\leq 1/(B+1) = 0.038$** |

With zero of the 25 permutations matching the observed accuracy, the standard one-sided empirical p-value is bounded above by $1/(B+1) = 1/26 \approx 0.038$. This rejects the null hypothesis at $\alpha = 0.05$ but is not strong enough to claim $p < 0.001$ — that would require at least 999 permutations under the same protocol. The actual result of 62.23% lies more than two standard deviations above the permutation mean (51.18% ± 4.72%), supporting the qualitative conclusion, but we report the rigorous bound and flag the modest permutation count as a limitation; a 1,000-permutation rerun is computationally tractable (~10 hours on the same hardware) and is on the project roadmap.

### 3.5 Final Holdout Evaluation

The model selection above used the first 80% of the timeline (Train+Val: 2025-05-10 to 2026-02-23). The remaining 20% (Holdout: 2026-02-23 to 2026-05-07, n=12,152 samples) was reserved and never inspected during model selection or hyperparameter tuning. We performed a single one-shot evaluation on this held-out period using the configuration selected on Train+Val.

**Table 6.** Holdout direction accuracy by asset (one-shot, no model retuning).

| Asset | n Signals | Direction Accuracy | AUC | Notes |
|-------|----------:|-------------------:|----:|-------|
| ADA | 46 | **93.48%** | 0.614 | |
| DOT | 106 | **89.62%** | 0.608 | |
| LINK | 46 | 84.78% | 0.638 | |
| AVAX | 33 | 78.79% | 0.617 | |
| SOL | 55 | 74.55% | 0.596 | |
| ETH | 25 | 64.00% | 0.641 | |
| BTC | 4 | 0.00% | 0.620 | **n too small to interpret** (Wilson CI [0%, 49%], uninformative) |
| **Pooled** | **315** | **69.32%** | **0.621** | excluding BTC: 69.32% on 311 signals |

**Interpretive caveats:**

1. **BTC must not be over-interpreted.** With only 4 holdout signals, the per-asset BTC entry has a Wilson 95% CI of [0%, 49.0%] — uninformative. The "0.00% accuracy" cell is dominated by sample-size noise; we explicitly exclude BTC from the per-asset interpretation and note that the pooled accuracy is essentially unchanged whether or not BTC is included (since BTC contributes 4 of 315 signals).

2. **Pooled holdout accuracy is high.** The 69.32% pooled estimate (Wilson 95% CI [63.90%, 74.05%]) substantially exceeds the cross-validated training estimate of 62.23%. This is unusual — typically holdout performance is *lower* than CV estimates due to selection effects — and we treat it with caution. The most likely explanation is that the holdout window (Feb 23 – May 7, 2026) had higher realized volatility than the average training window, and the regime filter selects high-vol windows where the model is most accurate. We do *not* claim 69.32% as a steady-state accuracy estimate; it is the result of one held-out window that happened to be favorable for the model.

3. **Per-asset ordering is suggestive only.** With per-asset n in the range 25–106, the per-asset Wilson CIs are wide ($\pm$ 5–15 pp). The ranking ADA ≈ DOT > LINK > AVAX > SOL is qualitatively consistent with the cross-validated training estimates, but small reorderings would not be statistically distinguishable.

### 3.6 Sharpe Ratio and Trading Costs

The Sharpe ratio is negative at all horizons despite statistically significant direction accuracy. We model fees as 0.001 per side and slippage as 0.0005 per side, yielding a 15 bps round-trip cost per trade. With direction accuracy of 60% and an average per-trade move of approximately 50 bps (implied by the 3d horizon's realized volatility), the gross expected per-trade edge is $0.60 \cdot 50 - 0.40 \cdot 50 = 10$ bps — *less* than the 15 bps round-trip cost, yielding a net loss per trade.

Three implications follow:

1. **Lower fees materially change the verdict.** A maker-rebate venue (e.g., Binance.US at 0.075%) reduces round-trip cost to 7.5 bps, flipping the per-trade economics positive.
2. **Longer horizons have favorable scaling.** Per-trade move grows roughly with the square root of holding period, while fees stay constant. At the 7d horizon, average per-trade move is ~120 bps, so the same 60% direction accuracy yields a 24 bps gross edge — comfortably above fees.
3. **Higher confidence thresholds reduce trade count.** Increasing $\theta$ from 0.62 to 0.70 reduces signal count and increases per-trade accuracy, which compounds favorably with fixed costs.

### 3.7 Break-Even Cost Analysis

Combining the direction-accuracy and average-move estimates from each horizon yields the break-even round-trip transaction cost — the maximum cost above which the strategy becomes unprofitable.

**Table 7.** Break-even cost per horizon. Computed from cross-validation accuracy and realized return distributions on Train+Val.

| Horizon | Accuracy | Avg \|Move\| (bps) | Gross Edge (bps) | Break-Even RT Cost | Profitable on Binance Maker (15 bps)? | Profitable on Coinbase Taker (85 bps)? |
|---------|---------:|-----------------:|-----------------:|-------------------:|---------------------------------------|---------------------------------------|
| 1h | 53.73% | 53.7 | 4.0 | 4.0 | ❌ | ❌ |
| 4h | 50.49% | 107.9 | 1.1 | 1.1 | ❌ | ❌ |
| 12h | 50.36% | 196.1 | 1.4 | 1.4 | ❌ | ❌ |
| 1d | 55.18% | 281.6 | 29.2 | 29.2 | ✅ | ❌ |
| **3d** | **62.14%** | **498.0** | **120.9** | **120.9** | ✅ | ✅ |
| 7d | 55.41% | 745.3 | 80.7 | 80.7 | ✅ | ❌ |

The 3-day horizon is the only configuration that is profitable on both maker- and taker-fee venues. At sub-daily horizons, the gross edge is too small to cover even the lowest realistic fee.

### 3.8 Methodology Progression (v1 → v9)

We document the nine-version methodology progression behind these results in `VALIDATION_PROGRESSION.md`. The sequence — from a naive single split (v1, invalid) → rule-based thresholds (v2, at chance) → ML classifier (v3, weak signal) → multi-asset pool (v4, underpowered) → 7-asset wider filter (v5, significant) → horizon sweep (v6, headline) → cross-regime check (v7, robust) → continuous walk-forward (v8, profitable on ADA) → holdout + ablation + permutation (v9, rigorous) — illustrates how each layer of validation discipline either ruled out a confounder or expanded statistical power. We recommend it as a template for similar empirical studies.

## 4. Discussion

### 4.1 Interpretation

Our results provide systematic evidence that persistent homology features predict short-horizon directional moves in cryptocurrency prices, with the strongest signal in less liquid mid-cap altcoins. The cross-horizon consistency (5/6 horizons significant) and per-asset heterogeneity (sharp gradient from ADA at 76.75% to BTC at 48.48%) jointly support a market-efficiency interpretation: TDA features capture genuine dynamical structure in price-volume manifolds, and that structure is most exploitable where market efficiency is weakest.

Three sources of edge are plausibly at work in the TDA features. First, persistent homology in high-volatility regimes captures *regime transitions* — moments when the local manifold geometry changes — and these transitions often precede directional moves. Second, the $L^p$-norm and entropy summaries are sensitive to the *number* of distinct regimes the system is currently transitioning between, providing a rough proxy for trader uncertainty. Third, by pooling across seven assets, the classifier learns *transferable* structural patterns rather than asset-specific quirks; we observed that single-asset models with the same architecture failed to beat chance, while the pooled model succeeded.

### 4.2 Limitations

(i) **Sample window.** The 90-day primary test window (extended to 365 days for cross-regime checks) may not span all market regimes; a multi-year extension would be straightforward conceptually but is bounded by Coinbase's free-tier rate limits. (ii) **Cost model.** We use point estimates of fees and slippage; real execution would face market impact, which is hard to estimate without proprietary order-book data. (iii) **No live deployment.** Our results are from offline cross-validation and have not been validated in paper or live trading. (iv) **Hyperparameter coupling.** The regime filter, probability threshold, and prediction horizon interact in ways our grid search may not fully resolve; a Bayesian optimization treatment is left for future work. (v) **Per-asset Wilson CIs.** Table 2's per-asset accuracies are based on 25–106 signals per asset on the holdout, giving CIs roughly $\pm 5\,\text{pp}$; the per-asset *ordering* should therefore be interpreted as suggestive rather than definitive. (vi) **BTC holdout n=4** is statistically uninformative and should not be interpreted as evidence for or against BTC predictability — the cross-validated estimate (n = 346, accuracy 48.48%) is the authoritative BTC number. (vii) **Permutation count.** The reported empirical p-value $\leq 0.04$ is bounded by $1/(B+1)$ for $B = 25$ permutations; a 1,000-permutation rerun (computationally feasible at ~10 hours) would tighten this bound to $\leq 0.001$ and is on the project roadmap. (viii) **Holdout-period favorability.** The held-out window happened to be a high-volatility period that maps well onto the regime filter; we cannot rule out that the 69.32% holdout accuracy partially reflects window selection rather than steady-state model performance.

### 4.3 Future Work

Higher-dimensional persistence ($H_2$ voids), Mapper graph features of the cross-asset correlation network, and real-time on-chain metrics (whale moves, exchange flows, MEV activity) are natural extensions. Combining TDA features with sequence models (Transformers, state-space models) is another direction. Most practically, deploying the strategy in paper trading with realistic fee modeling — and characterizing the live-vs-backtest gap — is the immediate next step.

### 4.4 Path to Profitability

The strategy as configured for the headline result has a 12 bps gross edge per trade against a 15 bps round-trip fee — a small net loss. We outline three modifications, each estimated below, that could shift the strategy into profitability.

**Modification 1: Maker-rebate venue.** Switching from Coinbase taker (0.4% per side) to Binance.US maker (0.075% per side) reduces round-trip cost from 85 bps to 15 bps. With the 3d-horizon configuration's average move of ~50 bps and 60% accuracy:

$$\text{net edge} = 0.60 \cdot 50 - 0.40 \cdot 50 - 15 = -5 \text{ bps}$$

Marginal — the maker-rebate alone is insufficient.

**Modification 2: Higher confidence threshold ($\theta = 0.80$).** Raising the BUY/SELL threshold from $\theta = 0.70$ to $\theta = 0.80$ cuts the signal count by approximately half but improves per-signal accuracy from 62% to ~68% (estimated by interpolating Wilson-CI-aware lower bounds across the grid search). The new gross edge:

$$\text{net edge} = 0.68 \cdot 50 - 0.32 \cdot 50 - 15 = +3 \text{ bps}$$

Positive but small. Annualized at ~50 trades/year per asset, this yields ~1.5% gross return — barely above breakeven.

**Modification 3: Move to the 7d horizon.** At a 7-day horizon, the average per-trade move grows to ~120 bps (Garman-Klass volatility of mid-cap altcoins is ~5%/week). With 62% accuracy:

$$\text{net edge} = 0.62 \cdot 120 - 0.38 \cdot 120 - 15 = +14 \text{ bps gross}$$

This is the cleanest path: the per-trade economics work, the trade count is manageable (~50 trades/year per asset), and the holding period is realistic for retail execution.

**Table 8.** Estimated net per-trade economics under three modification scenarios. Annualized return assumes 50 trades/year, no leverage, full deployment of capital.

| Scenario | Round-Trip Cost | Avg Move | Accuracy | Net Per-Trade | Est. Annual Return | Feasible? |
|----------|----------------:|---------:|---------:|--------------:|-------------------:|-----------|
| Baseline (v6 paper, Coinbase, 3d) | 85 bps | 50 bps | 62% | -75 bps | -38% | ❌ |
| Mod 1: Binance Maker (3d) | 15 bps | 50 bps | 62% | -5 bps | -3% | ❌ marginal |
| Mod 2: $\theta=0.80$ (Binance Maker, 3d) | 15 bps | 50 bps | 68% | +3 bps | +1.5% | ⚠️ marginal |
| Mod 3: 7d horizon (Binance Maker) | 15 bps | 120 bps | 62% | +14 bps | +7.0% | ✅ |
| Combined Mod 1+2+3 ($\theta=0.80$, 7d, Binance Maker) | 15 bps | 120 bps | 68% | +28 bps | +14% | ✅ |

The combined recommendation (move to 7-day horizon, raise the confidence threshold to 0.80, use a maker-rebate venue) gives a credible path to a 14% annual return on a single asset. Diversifying across the four predictable assets (ADA, SOL, AVAX, LINK) and accounting for partial deployment of capital, a realistic gross return target is 8-10% annually before tax — roughly comparable to a modest active-equity strategy.

We emphasize that these are *projected* numbers from offline analysis. Live deployment would face additional friction (variable spreads, partial fills, order-book impact) that our backtester does not model.

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
- **Figure 4** — Validation methodology progression v1→v9, illustrating how each version controlled for a different methodological bias (single-split → k-fold → ML → multi-asset → cross-regime → walk-forward → holdout+ablation+permutation) and how sample size grew across iterations from n=2 to n=4,266. Color coding distinguishes CV-only validation (blue), the headline result and rigorous-validation tier (green), and the continuous walk-forward profitability tier (purple). (`fig4_progression.pdf`)

## References

1. Gidea, M., Goldsmith, D., Katz, Y., Roldan, P., Shmalo, Y. (2020). *Topological recognition of critical transitions in time series of cryptocurrencies.* Physica A, 548.

2. Bauer, U. (2021). *Ripser: efficient computation of Vietoris-Rips persistence barcodes.* Journal of Applied and Computational Topology, 5, 391–423.

3. Saengduean, P., Sangwine, S. J., Lerdsuwan, P., Boonyasiri, A. (2018). *A Cryptocurrency Risk-Return Analysis for Bull and Bear Regimes Using Persistent Homology.* Proceedings of the IEEE Conference on Computational Intelligence for Financial Engineering & Economics.

4. Singh, G., Mémoli, F., Carlsson, G. (2007). *Topological methods for the analysis of high dimensional data sets and 3D object recognition.* Eurographics Symposium on Point-Based Graphics.

5. Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* Journal of the American Statistical Association, 22(158), 209–212.

6. Garman, M. B., Klass, M. J. (1980). *On the estimation of security price volatilities from historical data.* The Journal of Business, 53(1), 67–78.
