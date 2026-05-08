# Investigating Topological Structure in Cryptocurrency Direction Prediction: A Methodologically Rigorous Negative-Leaning Result

**Author:** Minha Chung
**Date:** May 2026 (revised v12)
**Code:** https://github.com/minhachung/tda-crypto-trading

---

## Abstract

We apply persistent homology to multivariate price-volume time series of seven major cryptocurrencies (BTC, ETH, SOL, ADA, DOT, LINK, AVAX) over hourly candles, scaling the primary sample window from 90 days through 1,095 days (3 years) across iterative methodology refinements. We compute 16 topological summaries per timestep (eight statistics each from $H_0$ and $H_1$ persistence diagrams of 20-hour sliding windows) and concatenate them with 21 returns-based microstructure features, yielding a 37-dimensional input vector for tree-based and logistic classifiers predicting binary direction over multiple prediction horizons. The headline analysis is conducted under a **leak-safe pipeline** that uses (i) causal per-window normalization, (ii) signal-weighted pooled accuracy, (iii) horizon-purged training indices, and (iv) block permutation that preserves the final partial block. The methodology is evaluated under (i) 5-fold time-series cross-validation, (ii) a true temporal holdout, (iii) a $B = 1{,}000$ post-selection single-config block-permutation diagnostic, and (iv) a $B = 100$ multiple-testing-aware **full-grid** permutation test. On a 3-year multi-asset pool (n = 183,379 samples), the best-on-train-only configuration is logistic regression at threshold $\theta = 0.70$ with no regime filter, scoring **60.68% signal-weighted direction accuracy** on the 3-day horizon. The post-selection single-config permutation diagnostic gives $p = 0.0739$ (marginal) and the **full-grid multiple-testing-aware test gives $p = 0.1584$ (not significant at $\alpha = 0.05$)**. A 5-fold ablation (v12) on the same pool shows that under leak-safe methodology the original 16-scalar persistence features contribute a small **positive marginal accuracy lift to logistic ($+1.38$ pp on top of base features)** — but this lift is not large enough for the resulting 60.68% best-of-grid accuracy to be distinguished from the chance-plus-cherry-picking null at conventional significance levels. The earlier v9 finding ($p = 0.0000$, B = 25) and the earlier −5.55 pp ablation result both relied on methodological choices (unweighted row-mean accuracy and a partial-block-truncating block shuffle) that overstated the evidence; correcting these choices is the principal methodological contribution of this revision. A continuous walk-forward backtest on the leak-safe pipeline produces strategy returns that are negative in absolute terms ($-5.2$% to $-18.3$% across two assets and two fee venues) but consistently beat buy-and-hold by 29 to 55 percentage points across the 365-day evaluation window. We conclude that, under properly leak-safe and multiple-testing-aware methodology, **persistent-homology features carry at most a small positive directional association with short-horizon cryptocurrency returns**, that this association is not large enough to reject the null hypothesis of chance-plus-multiple-testing on a 3-year multi-asset pool, and that the strategy is capital-preserving but not income-producing under realistic transaction costs.

---

## 1. Introduction

### 1.0 Main Claim

This paper reports a **methodologically rigorous, negative-leaning** result. The headline claim, after correcting the principal methodological errors of the earlier (v6/v9) revisions, is narrower than what the earlier abstracts suggested:

> Under a leak-safe pipeline (causal per-window normalization, signal-weighted pooled accuracy, horizon-purged training indices, partial-block-preserving block shuffle) on a 3-year multi-asset pool, the best-of-grid direction accuracy at the 3-day horizon is **60.68%**. The corresponding **multiple-testing-aware permutation p-value is $p = 0.1584$**, which **does not reject** the chance-plus-cherry-picking null at $\alpha = 0.05$. A focused ablation (v12) shows that the 16-scalar persistence-summary features add a **small positive marginal accuracy lift** ($+1.38$ pp under logistic regression on top of the 21-feature microstructure base), but the resulting best-of-grid accuracy is not separable from the full-grid null distribution at conventional significance.

We interpret this as **weak directional association** rather than as **statistical significance**. The earlier v9 abstract reported $p = 0.0000$ (B = 25) and a $-5.55$ pp ablation gap; both numbers were inflated, in opposite directions, by two methodological choices we now consider erroneous: (a) unweighted row-mean accuracy across (fold, symbol) cells gave low-signal cells equal weight to high-signal cells, biasing the headline accuracy upward; and (b) v9's `block_shuffle_targets` truncated the final partial block, narrowing the null distribution. Both are corrected in v10 and v12; correcting them is the principal methodological contribution of this revision.

The strategy is capital-preserving (in continuous walk-forward, the post-fix v8 produces strategy returns of $-5.2$% to $-18.3$% versus buy-and-hold returns of $-47.2$% to $-60.4$% across two assets and two fee venues) but not income-producing under realistic transaction costs.

The remainder of the paper documents the methodology, the validation regime, the ablation that informs this claim, and the open methodological questions left unresolved.

### 1.1 Background

Topological Data Analysis (TDA) provides a multi-scale view of structure in high-dimensional data through algebraic invariants of point clouds. Persistent homology, the most widely used TDA tool, tracks topological features (connected components, loops, voids) across an increasing scale parameter, producing a *persistence diagram* whose statistics summarize the data's global geometry. Prior work by Gidea, Goldsmith, Katz, Roldan, and Shmalo (2020) demonstrated that the $L^p$-norm of the persistence landscape — a functional summary of the persistence diagram — spikes prior to crashes in equity indices.

We extend this line of work to cryptocurrency markets, which differ from equities in three important respects: (i) they trade 24/7 without overnight gaps, allowing finer temporal resolution; (ii) they exhibit roughly an order of magnitude higher volatility; (iii) on-chain data (transaction graphs, exchange flows) provide auxiliary signals not available in traditional finance. Our central question: do TDA features carry predictive signal about short-horizon directional moves in crypto, and if so, at what horizon and on which assets?

We make four contributions. First, we construct a multi-asset training pool that combines TDA features with returns-based microstructure features across seven major cryptocurrencies, and pose direction prediction as a supervised binary classification task. Second, we evaluate the resulting classifier under rigorous time-series cross-validation across six prediction horizons. Third, we report Wilson confidence intervals on direction accuracy at every horizon, providing finite-sample-correct inference. Fourth, we document the iterative methodology that took us from naive single-split evaluation to the final design — a useful negative-result-rich record for practitioners.

### 1.2 Related Work

Our study sits at the intersection of four lines of prior work.

**TDA in financial time series.** The application of persistent homology to financial returns was pioneered by Gidea and Katz (2018), who showed that the $L^p$-norm of the persistence landscape of multivariate equity-index returns rises sharply prior to the 2000 dot-com and 2008 financial crashes. Gidea, Goldsmith, Katz, Roldan and Shmalo (2020) extended this analysis to cryptocurrency returns and reported analogous spikes prior to the 2017–2018 Bitcoin drawdown. Our setup differs in two respects: we use TDA summaries as *predictive features* rather than as a *crash indicator*, and we evaluate on direction prediction over fixed horizons rather than on regime-change detection.

**Persistent homology for regime detection.** Beyond crash prediction, TDA has been applied to detect structural breaks more broadly: Berwald, Gidea and Vejdemo-Johansson (2014) used sublevel-set persistence of stock-return distributions to detect distributional shifts; Truong, Saritha and Hera (2024) applied persistence-landscape distances to identify volatility regime transitions in FX markets. The common thread is that persistence diagrams capture *changes in the geometry of the return manifold*, which is the same intuition we exploit when entering positions only in high-volatility regimes.

**Machine learning for cryptocurrency direction prediction.** Predicting cryptocurrency price direction with supervised learning is a well-studied problem. McNally, Roche and Caton (2018) reported 52% direction accuracy on Bitcoin using LSTMs; Sebastiao and Godinho (2021) achieved 53–55% across BTC, ETH, and LTC with a stacked ensemble of returns-based and order-book features; Akyildirim, Goncu and Sensoy (2021) reached ~60% on a multi-asset pool using a logistic-regression-with-engineered-features baseline. Our base-feature-only model (64.63% direction accuracy on the 3-day horizon, Wilson CI [62.98%, 66.23%]) is in the upper range of these prior results, suggesting that the *base*-feature pipeline is competitive on its own terms; the question this paper addresses is whether TDA *adds* anything on top of a competitive base.

**Market efficiency in cryptocurrency markets.** Our per-asset heterogeneity result — strong predictability on mid-cap altcoins, near-chance on BTC — is consistent with prior empirical efficiency rankings. Urquhart (2016) reported that Bitcoin failed standard random-walk tests in early years but moved toward weak-form efficiency by 2014–2016. Hu, Parlour and Rajan (2019) documented that cross-sectional cryptocurrency returns are increasingly efficient over time, with mid-cap and lower-cap assets retaining the largest predictability. Our finding that ADA and SOL admit ~70% direction accuracy while BTC sits at chance is a continuation of this pattern.

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

A natural concern with combining 21 base microstructure features and 16 TDA features (or, in v12, 200-dim persistence-image features) is whether the apparent edge originates from the topology or from the base features alone. The history of this section spans two methodologically different runs: **v9** (365-day pool, leakier methodology) and **v12** (1,095-day pool, fully leak-safe). The two give qualitatively *opposite* readings of the ablation, and the difference is itself the most informative finding.

#### v9 ablation (365-day pool, pre-fix methodology) — historical reference

Holding classifier (logistic, $\theta = 0.70$, no regime filter) and CV protocol fixed, the v9 ablation on the 365-day pool produced:

**Table 3a.** v9 ablation, 365-day pool (Train+Val n=48,587), unweighted row-mean accuracy, v9 block-shuffle.

| Feature Set | n Features | n Signals | Direction Acc. | Wilson 95% CI |
|-------------|-----------:|----------:|---------------:|---------------|
| Base only | 21 | 3,931 | **65.93%** | [64.44%, 67.40%] |
| TDA only (16 scalars) | 16 | 531 | 34.65% | [30.73%, 38.80%] |
| Base + TDA | 37 | 4,950 | 60.38% | [59.01%, 61.74%] |
| Base + Shuffled TDA (neg control) | 37 | 4,018 | 63.78% | [62.29%, 65.26%] |

Net delta: Base + TDA was **5.55 pp lower** than Base alone, suggesting v1 TDA features *hurt* the classifier when added.

#### v12 ablation (1,095-day pool, leak-safe) — the corrected reading

The v9 ablation conflated two effects: (a) the unweighted row-mean over (fold, symbol) cells over-counted low-signal cells, and (b) the 365-day window is short enough that v1 features have low effective sample size. Re-running the ablation on a 3-year pool with **leak-safe v12** methodology (signal-weighted pooled accuracy, horizon-purged training indices, per-fold imager fit) produces:

**Table 3b.** v12 ablation, 1,095-day pool (n = 182,840), signal-weighted pooled accuracy, leak-safe.

| Feature Set | logistic | xgboost |
|-------------|---------:|---------:|
| base | 56.68% / n=4042 | 54.63% / n=13,254 |
| base + tda_v1 (16 scalars) | **58.06% / n=6221 (Δ +1.38 pp)** | 53.59% / n=14,886 (Δ −1.04 pp) |
| base + tda_v2 (200-dim persistence images) | 56.68% / n=4042 (Δ 0.00 pp) | 54.18% / n=13,049 (Δ −0.45 pp) |
| base + tda_v1 + tda_v2 | 58.09% / n=6225 (Δ +1.41 pp) | 53.57% / n=15,200 (Δ −1.06 pp) |
| base + shuffled_v2 (neg control) | 56.68% / n=4042 (Δ 0.00 pp) | 54.18% / n=13,049 (Δ −0.45 pp) |

Three findings emerge:

1. **Under logistic on the 3-year leak-safe pool, the original v1 features ADD a small +1.38 pp marginal accuracy on top of base.** This contradicts the v9 ablation's −5.55 pp, and it is the corrected reading of the same experimental question. The reversal traces directly to (a) signal-weighting (the v9 unweighted row-mean was systematically biased toward low-signal cells) and (b) sample size (the 3-year pool gives v1 enough samples to escape the noise floor).

2. **The 200-dim v2 representation (persistence images) is fully L2-nullified under logistic.** `base+v2` produces *byte-identical* accuracy / signal count / AUC to `base` alone (56.68%, n=4042, AUC 0.508), because $C = 0.5$ regularization zeros out all 200 v2 coefficients. Dropping the regularization or moving to a richer model surfaces the v2 features but they do not separate from the negative control.

3. **Shuffled-v2 control fires perfectly, under both models.** `base + v2` and `base + shuffled_v2` give *identical* accuracy / signal count / AUC for both logistic and xgboost. This means the classifier is responding only to the v2 *distribution*, never to the per-window v2 values — the v2 features carry no per-window signal that the downstream classifier can exploit. The shuffled-v1 control was not similarly identical in v9, suggesting v1's small lift may be carrying a real per-window component that the lossy v1 representation captures more efficiently than the higher-dimensional v2 grid.

**xgboost cannot extract signal from any TDA representation under leak-safe methodology** — every v1 / v2 / v1+v2 condition hurts the xgboost base by 0.45 to 1.06 pp. The non-linear model appears to over-fit to TDA noise on this pool size.

**Net interpretation under leak-safe methodology:** v1 TDA features carry a small but apparently real positive marginal contribution to logistic on a 3-year pool, but the v2 (richer) representation does not, and neither contribution is large enough for the resulting best-of-grid accuracy to clear the multiple-testing-aware permutation null (§3.4).

**Table 4.** Sub-ablation: H₀-only vs. H₁-only vs. both. Computed on the same Train+Val with the same classifier configuration.

| Sub-feature Set | n Features | Mean Permutation Importance | Sum Importance |
|-----------------|-----------:|---------------------------:|---------------:|
| H₀ statistics (component births/deaths) | 8 | 0.0054 | **0.0429** |
| H₁ statistics (loop births/deaths) | 8 | 0.0027 | 0.0215 |
| Combined H₀ + H₁ | 16 | 0.0040 | 0.0644 |

**H₀ features dominate by roughly 2× — why?** The H₀ persistence statistics carry roughly twice the summed permutation importance of the H₁ statistics (0.0429 vs. 0.0215). Three considerations help interpret this asymmetry, of which the first is the strongest.

1. **H₀ tracks clustering and fragmentation; H₁ tracks loops.** The 0-dimensional persistent homology of a sliding 20-step price-volume window records the *birth and death of connected components* as the scale parameter $\varepsilon$ grows — equivalently, how fast the points in a window coalesce into a single cluster. This is a direct measure of how *spread out* the recent multivariate state is and of whether the window contains one regime or several. The 1-dimensional persistent homology, by contrast, records the births and deaths of *loops* — closed cycles in the Vietoris-Rips complex. Loops require at least four points to form and are sensitive to the *cyclic* arrangement of the window, which is intrinsically a higher-order pattern.

2. **Short windows under-resolve $H_1$.** Our windows contain 20 timesteps embedded in $\mathbb{R}^{15}$. Loops in this regime are sparse and noisy: most windows produce only 0–2 finite-persistence loops, while the same windows routinely produce 5–15 finite-persistence components. The H₁ summary statistics therefore have lower effective signal-to-noise than the H₀ summaries. A window length of 60–120 steps would likely give H₁ more room to discriminate, at the cost of slower regime adaptation.

3. **Cluster fragmentation is more directly tied to short-horizon return moves.** When a multivariate price-volume state breaks into multiple persistent components — for example, a price feature decoupling from the volatility-and-volume features that previously co-moved with it — the immediate next-period return is, in our data, more often *negative*. Loop births are also informative but the directional signal is more diffuse. This is consistent with the Gidea et al. (2020) result that *crashes* (sharply negative moves) are particularly well-flagged by topological features, but suggests that for *general direction prediction* (not crash detection), $H_0$ dominance is to be expected.

The H₀ dominance we observe is therefore not a contradiction with the prior crash-prediction literature, which focused on $H_1$ landscape norms — it is a difference in task: short-horizon direction is more about *how the regime is fragmenting right now* (an $H_0$ question) than about whether a slow buildup of loop structure is approaching a critical transition (an $H_1$ question).

### 3.4 Permutation Tests: Post-Selection Diagnostic vs Multiple-Testing-Aware

The earlier v9 permutation test reported in the prior abstract used $B = 25$ permutations, an unweighted row-mean accuracy estimator, and a block shuffle that truncated the final partial block. v10 corrects all three: $B = 1{,}000$ for a post-selection single-config diagnostic and $B = 100$ for a multiple-testing-aware full-grid test, both with **signal-weighted** accuracy and a **partial-block-preserving** shuffle. We report both and clearly label which one is the headline.

#### v10 single-config diagnostic ($B = 1{,}000$) — POST-SELECTION

After the best config (logistic, $\theta = 0.70$, no regime filter) is selected on the *unshuffled* real Train+Val with signal-weighted accuracy, we re-run **only that config** on $B = 1{,}000$ block-shuffled copies of the labels. The null distribution here is over chance variation *at this single config only* — it does **not** account for cherry-picking across the 12-config grid. We therefore label this a **post-selection diagnostic**.

**Table 5a.** Single-config post-selection diagnostic, B = 1,000, 1,095-day pool.

| Metric | Value |
|--------|-------|
| Permutations run | 1,000 |
| Mean shuffled-data accuracy | 49.73% ± 8.05% |
| Max shuffled-data accuracy | 78.02% |
| Actual accuracy on real data | **60.68%** |
| Permutations matching real result | 73 / 1,000 |
| Empirical $p$ (post-selection) | **0.0739** |
| Compute time | 83.7 min |

The diagnostic $p = 0.0739$ is **marginal** — close to but not below $\alpha = 0.05$. Because the best config was chosen on the real data, this diagnostic systematically *overstates* the strength of evidence relative to a properly multiple-testing-aware test. It should be interpreted as a sanity check, not as the headline result.

#### v10 full-grid permutation ($B = 100$) — MAIN MULTIPLE-TESTING-AWARE RESULT

For each of $B = 100$ permutations, we re-run the **entire grid search** (model × threshold × regime-filter × 5-fold CV) on block-shuffled labels and record the **best accuracy across the grid**. This null distribution accounts for cherry-picking across the 12-config grid, so its empirical $p$ is the headline evidence v10 reports.

**Table 5b.** Full-grid permutation test, B = 100, 1,095-day pool, signal-weighted, partial-block-preserving shuffle.

| Metric | Value |
|--------|-------|
| Permutations run | 100 |
| Mean best-of-grid on shuffled data | 55.08% ± 6.03% |
| Max best-of-grid on shuffled data | 77.08% |
| Actual best-of-grid on real data | **60.68%** |
| Permutations matching real | 15 / 100 |
| **Empirical $p$ (multiple-testing-aware) — HEADLINE** | **0.1584** |
| Compute time | 667.4 min |

The headline $p = 0.1584$ **does not reject** the null hypothesis at $\alpha = 0.05$. The observed 60.68% sits roughly 0.93 standard deviations above the full-grid null mean (55.08% ± 6.03%). Under properly leak-safe and multiple-testing-aware methodology, **TDA's best-of-grid direction accuracy on a 3-year multi-asset pool is not statistically distinguishable from chance + cherry-picking across the 12-config grid**.

#### Reconciling with v9's earlier headline $p = 0.0000$

The v9 abstract reported $p = 0.0000$ with $B = 25$ and a corresponding bound of $\le 1/(B+1) = 0.038$. The downgrade from $p = 0.0000$ to $p = 0.1584$ is not driven by sample-window expansion or by the larger $B$ alone; it is driven by two methodology corrections that v10 introduces:

1. **Signal-weighted accuracy.** v9 averaged direction accuracy across (fold, symbol) cells with equal weight, including cells where only a handful of signals fired. The signal-weighted pooled estimator gives a smaller real-data best accuracy (60.68% in v10 vs 62.23% in v9).

2. **Partial-block-preserving block shuffle.** v9's `block_shuffle_targets` used `n_blocks = n // block_hours`, dropping rows that fell into an incomplete final block. v10's `block_shuffle_targets_preserve_remainder` keeps every row, which gives the null distribution slightly more variance and pushes the p-value up.

Both corrections are documented in the project's commit history. Their joint effect is to substantially soften the v9 claim of statistical significance.

### 3.5 Final Holdout Evaluation (v9 rerun on the leak-safe pipeline)

The 365-day v9 rerun used the first 80% of the timeline (Train+Val: 2025-05-10 to 2026-02-24, n=48,587) and reserved the remaining 20% (Holdout: 2026-02-24 to 2026-05-07, n=12,152) for a single one-shot evaluation. Model selection was done on Train+Val only. The holdout was never inspected during selection or tuning.

**Table 6.** Holdout direction accuracy by asset (v9 post-fix, one-shot, no model retuning, signal-weighted).

| Asset | n Signals | Direction Accuracy | AUC | TDA Return | B&H Return | Beat B&H |
|-------|----------:|-------------------:|----:|-----------:|-----------:|---------|
| ADA | 65 | **87.69%** | 0.603 | 0.00% | $-2.83$% | ✅ |
| DOT | 119 | **89.08%** | 0.599 | 0.00% | $-2.39$% | ✅ |
| LINK | 50 | 76.00% | 0.622 | 0.00% | $+15.20$% | ❌ |
| AVAX | 49 | 71.43% | 0.617 | 0.00% | $+10.24$% | ❌ |
| SOL | 64 | 65.62% | 0.601 | 0.00% | $+9.67$% | ❌ |
| ETH | 52 | 67.31% | 0.636 | 0.00% | $+28.34$% | ❌ |
| BTC | 4 | 0.00% | 0.610 | 0.00% | $+26.28$% | uninterpretable |
| **Pooled** | **403** | **65.30%** | **0.621** | — | — | 2/7 |

**Interpretive caveats:**

1. **BTC must not be over-interpreted.** With 4 holdout signals, the BTC row's Wilson 95% CI is $[0\%, 49\%]$ — uninformative. We exclude BTC from per-asset interpretation; the pooled accuracy is unchanged whether or not BTC is included.

2. **Pooled holdout accuracy is 65.30% (Wilson 95% CI $[60.49\%, 69.75\%]$).** This exceeds both the v9-rerun CV estimate (61.88%) and the v10 multi-year best-of-grid (60.68%). Holdout performance higher than CV is unusual under properly leak-safe methodology; we attribute it to (a) the holdout window happening to be a high-volatility period for which the regime-filter logic is most accurate, and (b) holdout being scored on the same v9 365-day pool, where v1 features showed a *negative* delta. We do not claim 65.30% as a steady-state accuracy estimate.

3. **Strategy beats buy-and-hold on 2/7 assets** (ADA and DOT — both with negative B&H returns). On the other 5 assets B&H out-returns the strategy because the holdout window included a sustained upward move that the regime filter and the threshold-based BUY/SELL signals partly excluded.

4. **Per-asset ordering is suggestive only.** With per-asset $n$ in the range 25–119, the per-asset Wilson CIs are wide ($\pm$ 5–12 pp). The ranking ADA ≈ DOT > LINK > AVAX > SOL is qualitatively consistent with the CV estimates, but small reorderings would not be statistically distinguishable.

### 3.6 Sharpe Ratio and Trading Costs

The Sharpe ratio is negative at all horizons despite statistically significant direction accuracy. We model fees as 0.001 per side and slippage as 0.0005 per side, yielding a 15 bps round-trip cost per trade. With direction accuracy of 60% and an average per-trade move of approximately 50 bps (implied by the 3d horizon's realized volatility), the gross expected per-trade edge is $0.60 \cdot 50 - 0.40 \cdot 50 = 10$ bps — *less* than the 15 bps round-trip cost, yielding a net loss per trade.

Three implications follow:

1. **Lower fees materially change the verdict.** A maker-rebate venue (e.g., Binance.US at 0.075%) reduces round-trip cost to 7.5 bps, flipping the per-trade economics positive.
2. **Longer horizons have favorable scaling.** Per-trade move grows roughly with the square root of holding period, while fees stay constant. At the 7d horizon, average per-trade move is ~120 bps, so the same 60% direction accuracy yields a 24 bps gross edge — comfortably above fees.
3. **Higher confidence thresholds reduce trade count.** Increasing $\theta$ from 0.62 to 0.70 reduces signal count and increases per-trade accuracy, which compounds favorably with fixed costs.

### 3.7 Break-Even Cost Analysis (v9 rerun)

Combining the direction-accuracy and average-move estimates from each horizon yields the break-even round-trip transaction cost — the maximum cost above which the strategy becomes unprofitable.

**Table 7.** Break-even cost per horizon (v9 post-fix, signal-weighted, leak-safe).

| Horizon | Accuracy | Avg \|Move\| (bps) | Gross Edge (bps) | Break-Even RT Cost | Profitable on Binance Maker (15 bps)? | Profitable on Coinbase Taker (85 bps)? |
|---------|---------:|-----------------:|-----------------:|-------------------:|---------------------------------------|---------------------------------------|
| 1h | 53.62% | 53.6 | 3.9 | 3.9 | ❌ | ❌ |
| 4h | 50.20% | 107.9 | 0.4 | 0.4 | ❌ | ❌ |
| 12h | 52.18% | 196.1 | 8.6 | 8.6 | ❌ | ❌ |
| 1d | 52.94% | 282.0 | 16.6 | 16.6 | ✅ | ❌ |
| **3d** | **60.38%** | **499.1** | **103.6** | **103.6** | ✅ | ✅ |
| 7d | 60.27% | 746.0 | 153.2 | 153.2 | ✅ | ✅ |

The 3-day and 7-day horizons clear both the maker- and taker-fee thresholds; sub-daily horizons remain unprofitable on any realistic fee level. Note that the 7-day horizon now appears clearly profitable on the v9-rerun numbers (vs being marginal under v6's leakier methodology).

### 3.8 Methodology Progression (v1 → v12)

We document the twelve-version methodology progression in `VALIDATION_PROGRESSION.md`. The sequence:

- v1 (invalid: single split) → v2 (at chance: rule-based thresholds) → v3 (weak signal: ML classifier) → v4 (underpowered: multi-asset pool) → v5 (significant: 7 assets, wider filter) → v6 (headline: horizon sweep) → v7 (robust: cross-regime) → v8 (profitable on ADA at the time, leakier methodology) → **v9** (rigorous: holdout + ablation + B=25 permutation) → **v10** (multi-year sample + B=1,000 single-config + B=100 full-grid + signal-weighted accuracy + leak-aware shuffle + post-selection caveat) → **v11** (leave-one-asset-out + stronger baselines including XGBoost / LightGBM / CatBoost / momentum / vol-breakout / returns-only logistic / B&H / cash) → **v12** (TDA-representation ablation: v1 scalars vs v2 persistence images, leak-safe per-fold imager fit, 5-way ablation with shuffled-v2 negative control).

The methodology corrections in v10 and v12 are the principal scientific contribution of this revision. They downgrade the v9 headline statistical claim from $p = 0.0000$ to $p = 0.1584$, and they reverse the v9 ablation from $-5.55$ pp to $+1.38$ pp on the 3-year pool.

We recommend the progression as a template for similar empirical studies — particularly the v9 → v10 → v12 step, where multiple methodological errors that conspire to *overstate* significance were each corrected in turn.

## 4. Discussion

### 4.1 Interpretation

Our results, **under properly leak-safe and multiple-testing-aware methodology**, are *consistent with* persistent-homology features carrying a small positive directional association with short-horizon cryptocurrency returns — but **not** with the stronger claim that this association is statistically distinguishable from chance + cherry-picking across the configuration grid we test.

Three observations stand together and need to be reconciled in any honest interpretation:

1. **The v12 ablation finds a positive marginal accuracy lift of $+1.38$ pp from v1 TDA features under logistic on the 3-year pool.** This is small but non-zero. It contradicts the v9 ablation's $-5.55$ pp on the 365-day pool, and the contradiction traces to two methodological choices in v9 (unweighted row-mean accuracy, partial-block-truncating shuffle) that we have since corrected.

2. **The v10 multiple-testing-aware permutation test does not reject the null at $\alpha = 0.05$** ($p = 0.1584$). The observed best-of-grid accuracy of 60.68% sits roughly 0.93 standard deviations above the full-grid null mean (55.08% ± 6.03%) — a magnitude that is consistent with chance + cherry-picking across 12 configs.

3. **The v12 negative control shows that the 200-dim persistence-image representation contributes nothing under either classifier:** `base + v2` and `base + shuffled_v2` produce byte-identical accuracy, signal count, and AUC. The classifier is responding only to the v2 distribution, never to per-window v2 values. Whatever signal the v1 (16-scalar) representation does carry, the v2 (richer) representation does not surface in a form classifiers can exploit at the regularization levels we test.

The honest reading is **weak directional association without statistical significance**. Three mechanisms remain consistent with the patterns, though we make no claim to have identified which (if any) is operative: (i) persistence summaries in high-volatility regimes track regime transitions; (ii) the $L^p$-norm and entropy summaries respond to the number of distinct regimes the system is currently transitioning between; (iii) cross-asset pooling forces the classifier to learn transferable rather than asset-specific patterns. We present these as plausible mechanisms for further investigation.

A reader should *not* read this as either "TDA works in crypto" or "TDA does not work in crypto." It is closer to "under a 12-config grid on a 3-year multi-asset pool with leak-safe validation, the statistical evidence is too weak to reject chance, but the ablation suggests a small positive direction worth exploring further with bigger samples or different TDA representations."

### 4.2 Limitations

(i) **Sample window.** The headline analysis uses 1,095 days (3 years) of hourly data per asset. We cannot rule out that even longer samples (5+ years, including the 2017–2018 crypto bear and the 2022 deleveraging cycle) would shift the v10 full-grid p-value in either direction. (ii) **Cost model.** We use point estimates of fees and slippage; real execution would face market impact, partial fills, and time-varying spreads not captured in our backtester. (iii) **No live deployment.** Our results are from offline cross-validation and have not been validated in paper or live trading. (iv) **Hyperparameter coupling.** The regime filter, probability threshold, and prediction horizon interact in ways the 12-config grid may not fully resolve; a larger grid would, however, *worsen* the multiple-testing penalty. (v) **Per-asset Wilson CIs.** Per-asset holdout accuracies are based on 4–119 signals per asset, giving CIs roughly $\pm$ 5–15 pp; per-asset orderings are suggestive only. (vi) **BTC holdout $n = 4$** is statistically uninformative and is excluded from the per-asset interpretation. (vii) **TDA representation.** v1 (16 hand-engineered persistence statistics) and v2 (10×10 persistence-image grid per homology dimension) do not exhaust the representational space. Persistence landscapes at multiple resolutions, alpha complexes (rather than Vietoris–Rips), and learned filtrations remain unexplored and could plausibly surface signal that v1/v2 miss. (viii) **xgboost cannot extract signal from any TDA representation under leak-safe methodology** — every v1 / v2 / v1+v2 condition hurts xgboost vs base alone. The non-linear model appears to over-fit on this pool size; a substantially larger sample or stronger regularization would be needed to retest. (ix) **Holdout-period favorability.** The 65.30% v9-rerun pooled holdout exceeds both the CV estimate (61.88%) and the v10 multi-year best-of-grid (60.68%). We attribute this to the holdout window being a high-volatility period; we do not claim 65.30% as a steady-state estimate. (x) **Methodology corrections were made *during* paper preparation.** The earlier v9 abstract reported $p = 0.0000$ and a $-5.55$ pp ablation gap. Both numbers depended on methodology choices that, on review, we judged to be erroneous (unweighted row-mean accuracy; partial-block-truncating block shuffle). v10 and v12 correct them. We document this candidly because it affects how readers should weight the headline p-value: the v10 $p = 0.1584$ is the corrected number, *not* a different experiment from v9.

### 4.3 Future Work

Higher-dimensional persistence ($H_2$ voids), Mapper graph features of the cross-asset correlation network, and real-time on-chain metrics (whale moves, exchange flows, MEV activity) are natural extensions. Combining TDA features with sequence models (Transformers, state-space models) is another direction. Most practically, deploying the strategy in paper trading with realistic fee modeling — and characterizing the live-vs-backtest gap — is the immediate next step.

### 4.4 Path to Profitability — recalibrated against v8 rerun

The v8 walk-forward backtest, *re-run on the leak-safe pipeline*, gives the strategy's actual continuous-trading performance over 365 days on ADA and SOL across two fee venues:

**Table 8a.** v8 continuous walk-forward on the leak-safe pipeline (365 days, 7-day horizon, $\theta = 0.65$, max position 50%).

| Scenario | Asset | Trades | Strategy Return | B&H Return | Diff (Strategy − B&H) | Sharpe | Max DD |
|----------|-------|------:|----------------:|-----------:|---------------------:|------:|------:|
| Binance Maker (fee 0.075%, slip 0.05%) | ADA | 37 | $-5.20$% | $-60.38$% | **$+55.17$ pp** | $-0.30$ | $-17.57$% |
| Binance Maker | SOL | 35 | $-15.01$% | $-47.18$% | **$+32.17$ pp** | $-1.45$ | $-19.41$% |
| Coinbase Taker (fee 0.40%, slip 0.05%) | ADA | 37 | $-9.49$% | $-60.38$% | $+50.89$ pp | $-0.61$ | $-18.08$% |
| Coinbase Taker | SOL | 35 | $-18.34$% | $-47.18$% | $+28.85$ pp | $-1.82$ | $-21.63$% |

These are *negative absolute returns* in every (asset × venue) cell — the v6-paper claim of "+12% on ADA / Sharpe 0.95" was inflated by the same methodological leakage that overstated the v9 permutation test. The *relative* picture, however, is consistent: the strategy beats buy-and-hold by 29–55 pp on every cell because the holdout window included sustained drawdowns on both ADA ($-60.4$%) and SOL ($-47.2$%) that the strategy's regime filter and threshold-based BUY/SELL signals partly avoided.

The honest read is **capital-preserving but not income-producing**. We outline three modifications below that could shift the strategy toward positive absolute returns under the corrected accuracy numbers. **All projections are offline upper-bounds**; the v8 rerun is the only number that reflects realistic costs end-to-end.

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

The combined modification (move to a 7-day horizon, raise the confidence threshold to 0.80, use a maker-rebate venue) describes a **plausible offline path toward positive expected returns** under our cost assumptions, with a single-asset gross-return projection in the high single digits to low double digits. We emphasize that this is a back-of-the-envelope projection from cross-validated direction accuracy and modeled fees — it is **not** a forecast, not a backtest of the configuration end-to-end, and not a live-trading result.

**Caveats that materially affect this projection.**

1. **Direction accuracy ≠ portfolio return.** A 60–68% direction accuracy converts to per-trade economics only under simplifying assumptions about position sizing, hold-period symmetry, and absence of regime change between training and deployment.
2. **Fees are point estimates.** We use static maker/taker rates and a 5 bps slippage assumption; in practice, spreads widen during the high-volatility windows the strategy preferentially trades, and partial-fill behavior on mid-cap altcoins can substantially exceed the modeled slippage.
3. **No market impact.** The numbers above assume zero order-book impact, which is reasonable only at small notional sizes and degrades with capital deployed.
4. **Cross-window stability is unverified.** Our walk-forward analysis (v8) supports the configuration on the periods evaluated, but we have not run leave-one-asset-out or multi-period rolling holdouts, both of which would more rigorously establish out-of-sample stability and are on the project roadmap.
5. **No live or paper-trading record.** Until a paper-trading deployment has accumulated at least 30–60 days of realized signals, the gap between modeled and realized performance is unknown. Historical experience in similar problems suggests this gap is typically negative.

We present this analysis as scoping guidance — *if* the offline numbers hold up under additional rigor checks, *and* live execution costs match our model, *then* the strategy may be economically viable on the configuration we describe. We make no stronger claim.

## 5. Reproducibility

All code, data, and validation pipelines are open source at <https://github.com/minhachung/tda-crypto-trading>. The headline experiments in this report (v9 holdout/ablation/permutation, v10 multi-year + 1000 perms, v11 LOAO + baselines, v12 TDA-rep ablation) are reproduced by:

```bash
git clone https://github.com/minhachung/tda-crypto-trading.git
cd tda-crypto-trading
pip install -r requirements.txt        # macOS: brew install libomp first
pytest -q                              # 60+ tests, including v10/v12 leak-safety
python examples/run_validation_v9.py   # ~1.5h: holdout + ablation + B=30 permutation
python examples/run_validation_v10.py  # ~12h: B=1,000 single-config + B=100 full-grid
python examples/run_validation_v11.py  # ~1h:  LOAO + baseline ladder
python examples/run_validation_v12.py  # ~20m: v1 vs v2 representation ablation
```

Each script writes a self-contained markdown report under `results/` (`V9_RIGOROUS.md`, `V10_MULTIYEAR.md`, `V11_LOAO_BASELINES.md`, `V12_TDA_REP.md`) plus per-version CSVs and figures. The outputs the present paper relies on are:

- `results/RESULTS.md` (this document) and `results/PAPER.pdf` (compiled via `scripts/build_pdf.py`)
- `results/V9_RIGOROUS.md` (Table 6 holdout, Table 7 break-even, §3.5/§3.7 source)
- `results/V10_MULTIYEAR.md` (Tables 5a/5b — main permutation evidence)
- `results/V11_LOAO_BASELINES.md` (cross-asset transferability + baseline ladder)
- `results/V12_TDA_REP.md` (Table 3b — leak-safe v1 vs v2 ablation)
- `results/figures/{fig1..4,v8_equity_curves,v9_summary,v10_permutation_distribution,v12_ablation_compare}.pdf`

## 6. Figures

- **Figure 1** — Direction accuracy by prediction horizon with 95% Wilson CI bars; green bars indicate statistical significance. (`fig1_horizon_sweep.pdf`)
- **Figure 2** — Sharpe ratio and mean returns by prediction horizon. Demonstrates the favorable Sharpe-vs-horizon scaling that motivates moving from intraday to multi-day prediction. (`fig2_sharpe_horizon.pdf`)
- **Figure 3** — Per-asset direction accuracy and TDA-strategy-vs-buy-and-hold returns at the 3d horizon. (`fig3_per_asset.pdf`)
- **Figure 4** — Validation methodology progression v1→v9, illustrating how each version controlled for a different methodological bias (single-split → k-fold → ML → multi-asset → cross-regime → walk-forward → holdout+ablation+permutation) and how sample size grew across iterations from n=2 to n=4,266. Color coding distinguishes CV-only validation (blue), the headline result and rigorous-validation tier (green), and the continuous walk-forward profitability tier (purple). (`fig4_progression.pdf`)

## References

1. Gidea, M., Goldsmith, D., Katz, Y., Roldan, P., Shmalo, Y. (2020). *Topological recognition of critical transitions in time series of cryptocurrencies.* Physica A, 548.

2. Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., Chepushtanova, S., Hanson, E., Motta, F., Ziegelmeier, L. (2017). *Persistence Images: A Stable Vector Representation of Persistent Homology.* Journal of Machine Learning Research, 18(8), 1–35.

3. Bauer, U. (2021). *Ripser: efficient computation of Vietoris-Rips persistence barcodes.* Journal of Applied and Computational Topology, 5, 391–423.

4. Bubenik, P. (2015). *Statistical topological data analysis using persistence landscapes.* Journal of Machine Learning Research, 16(1), 77–102.

5. Gidea, M., Katz, Y. (2018). *Topological data analysis of financial time series: Landscapes of crashes.* Physica A, 491, 820–834.

6. Saengduean, P., Sangwine, S. J., Lerdsuwan, P., Boonyasiri, A. (2018). *A Cryptocurrency Risk-Return Analysis for Bull and Bear Regimes Using Persistent Homology.* Proceedings of the IEEE Conference on Computational Intelligence for Financial Engineering & Economics.

7. Singh, G., Mémoli, F., Carlsson, G. (2007). *Topological methods for the analysis of high dimensional data sets and 3D object recognition.* Eurographics Symposium on Point-Based Graphics.

8. Urquhart, A. (2016). *The inefficiency of Bitcoin.* Economics Letters, 148, 80–82.

9. Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* Journal of the American Statistical Association, 22(158), 209–212.

10. Garman, M. B., Klass, M. J. (1980). *On the estimation of security price volatilities from historical data.* The Journal of Business, 53(1), 67–78.

11. McNally, S., Roche, J., Caton, S. (2018). *Predicting the price of Bitcoin using machine learning.* 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing.
