# Validation Progression: v1 → v12

> **Headline status (post v10/v12 corrections):** under leak-safe paper-grade methodology on a 3-year multi-asset pool, the best-of-grid 3-day-horizon accuracy is **60.68%** with a multiple-testing-aware permutation $p$-value of **0.1584** — *not* significant at $\alpha = 0.05$. The earlier v9 abstract claim of $p = 0.0000$ was inflated by two methodology errors that v10 corrects: unweighted row-mean accuracy and a partial-block-truncating block shuffle. v12 separately reverses the v9 ablation finding from $-5.55$ pp to $+1.38$ pp (logistic, 3-year pool) by switching to signal-weighted accuracy on a longer sample. The historical narrative below is preserved as written; the numbers in v6 / v7 / v9's "✅ significant" entries are the values reported *at the time of those runs* and are superseded by v10's leak-safe re-evaluation of the same experimental question. See §3.4 of `results/RESULTS.md` for the full reconciliation.

---

## v1 → v5 (historical, pre-fix methodology)


Honest record of how the TDA strategy was iteratively validated and improved.

## Summary Table

| Version | Method | Sample Size | Direction Acc | Wilson CI Lower | Statistically Significant? |
|---------|--------|-------------|---------------|-----------------|---------------------------|
| v1 | Single train/test split (CoinGecko daily) | n=2 (test) | 0% | — | ❌ Invalid (sample too small) |
| v2 | Rule-based threshold + 5-fold CV (Coinbase hourly) | n=728 | 50.75% | 47.06% | ❌ At chance |
| v3 | ML classifier on TDA features + 5-fold CV | n=152 | 56.18% | 47.98% | ❌ Almost (5pt edge) |
| v4 | + Multi-asset pool (4 cryptos) + advanced features + regime filter | n=26 | **60.42%** | 42.53% | ❌ Right direction, too few signals |
| **v5** | **+ 7 assets, 180 days, vol_median filter, GBM** | **n=684** | **59.40%** | **55.63%** | **✅ YES** |

## Key Insights at Each Version

### v1: Initial Sanity Check (Failed Validation)
- **Method:** CoinGecko daily data (92 candles), single train/val/test split
- **Problem:** Test set had only 2 signals — Wilson CI undefined
- **Lesson:** Need 100+ signals minimum for any inference

### v2: Rule-Based Thresholds Don't Work
- **Method:** Hand-crafted thresholds on C1-norm of persistence
- **Result:** 50.75% direction accuracy — pure chance
- **Lesson:** Threshold rules are too brittle; the relationship between TDA features and direction isn't a simple "X > threshold" pattern

### v3: ML Classifier Extracts Real Signal
- **Method:** Train logistic / RF / GBM on full TDA feature vector inside each CV fold
- **Result:** 56.18% direction accuracy (logistic best, p=0.58, h=1)
- **Lesson:** ML reveals 5-6pt edge that rules can't capture, but Sharpe still negative due to fees

### v4: Multi-Asset + Advanced Features + Regime Filter
- **Method:** Pool 4 cryptos, replace OHLCV with returns/Garman-Klass/Parkinson, add vol regime filter
- **Result:** 60.42% direction accuracy on signals taken (up another 4pt)
- **New feature:** ADA hit 80% direction accuracy — strong asset-specific signal
- **Lesson:** Each layer adds value, but regime filter at q75 is too restrictive — only 26 signals fired total

### v5: More Assets + More Data + Looser Filter
- **Method:** 8 cryptos (DOT, LINK, MATIC, AVAX added), 180 days, vol > median filter
- **Goal:** Generate enough signals (target: 200+) to tighten Wilson CI below 50%

## What This Tells Us About TDA in Crypto

**Confirmed:**
- TDA features carry **real predictive signal** — the progressive accuracy improvement from 50.75% → 60.42% across well-controlled validation methods is not an artifact
- The signal is **strongest in low-cap altcoins** (ADA at 80%, SOL at 47% — counterintuitive, suggests some assets have more topological structure than others)
- **High-volatility regimes** are where TDA shines (q75 filter improves accuracy)

**Still uncertain:**
- Whether the edge is large enough to overcome fees + slippage (current Sharpe negative)
- Whether the per-asset variation is real signal or noise

**Remaining limitations:**
- Coinbase's 1h granularity caps total samples even with multi-asset pool
- TDA computation is the bottleneck (~5min per 2000 windows on Ripser)

## v5 Result: 🟢 VALIDATED

### Final Best Configuration
- **Model:** Gradient Boosting Machine (GBM)
- **Probability threshold:** 0.62
- **Regime filter:** vol > median (not q75 — q75 was too strict)
- **Pool:** 29,694 samples across 7 assets, 180 days hourly

### Direction Accuracy: 59.40% (n=684)
**95% Wilson CI: [55.63%, 62.98%] — entirely above 50% chance line.**

### Per-Asset Breakdown
| Asset | Direction Accuracy | Signals | Beat BH |
|-------|-------------------|---------|---------|
| ADA | **72.21%** | 115 | 60% |
| ETH | 65.62% | 88 | 40% |
| AVAX | 65.18% | 108 | 20% |
| DOT | 63.50% | 110 | **80%** |
| SOL | 57.88% | 112 | 60% |
| BTC | 49.55% | 52 | 60% |
| LINK | 41.83% | 99 | 40% |

### Confirmed Findings
1. **TDA signal is real and transferable across assets** — pooled training works.
2. **Mid-cap altcoins are most predictable** — ADA at 72%, DOT at 63.5%.
3. **BTC is hardest** — 49.55% accuracy, very efficient market.
4. **Regime filter at median is the sweet spot** — q75 was too strict (only 26 signals in v4).

### Remaining Issue: Sharpe Still Negative (-7.45)
Even with 59.40% direction accuracy, Sharpe is negative. Why?
- 0.001 trade fee + 0.0005 slippage = ~0.15% round-trip cost per trade
- Average winning trade is small (we trade hourly windows)
- Fees consume the small per-trade edge

**Next step (v6):** Move to longer prediction horizons (1d instead of 1h) and stricter prob thresholds (0.65+). Direction accuracy should hold or improve, but per-trade move size should expand to overcome fees.

---

## v6 → v12 (post-v5 evolution and the methodology corrections)

### v6: Horizon sweep, paper-quality output (pre-fix methodology)
- 90-day window, 6-horizon sweep, 144-config grid.
- Headline: 3-day RF $\theta = 0.70$ vol-median filter, 61.77% direction accuracy on n = 2,260 (Wilson CI [59.75%, 63.75%]).
- **Caveat:** these numbers used unweighted row-mean accuracy; the leak-safe v10 rerun on a 3-year pool gives a different best config (logistic $\theta = 0.70$ no filter, 60.68%) and a non-significant full-grid permutation p (see v10).

### v7: 365-day cross-regime profitability check (pre-fix methodology)
- Tested whether v6's headline survives in different regimes (bull/bear/chop).
- Reported 67–71% per regime — the appearance of robust accuracy.
- **Caveat:** also pre-fix; the same methodology errors that inflated v6 inflated v7's per-regime numbers symmetrically.

### v8: Continuous walk-forward on the leak-safe pipeline
- Re-run on the post-fix pipeline with the v9-rerun-selected config (logistic $\theta = 0.65$, max position 50%) over 365 days, ADA + SOL, two fee venues.
- Strategy returns: $-5.20$% to $-18.34$% across (asset × venue) cells.
- **Beats buy-and-hold by 29–55 percentage points** every cell (B&H lost 47–60% over the same window).
- Honest read: capital-preserving but not income-producing under realistic transaction costs.

### v9: Rigorous holdout + ablation + B = 30 permutation (post-fix v9)
- Best CV config: logistic $\theta = 0.70$ no filter, 61.88% on Train+Val (signal-weighted).
- Ablation: base 65.93%, base+TDA 60.38% (Δ $-5.55$ pp). **Reversed by v12 on a 3-year pool.**
- Permutation: B = 30, 0/30 matched real, $p = 0.0000$. **Reversed by v10 with B = 1,000 + B = 100 grid.**
- Holdout: pooled 65.30% on n = 403 (Wilson CI [60.49%, 69.75%]); 2/7 assets beat B&H (ADA, DOT).
- v9 was the cleanest result *available at the time*. Subsequent runs (v10, v12) revealed that the v9 methodology had two errors that compound: unweighted row-mean accuracy and partial-block-truncating block shuffle.

### v10: Multi-year + B = 1,000 + B = 100 grid + signal-weighted + leak-aware shuffle (paper-grade headline)
- 1,095-day pool, 7 assets, 183,379 pooled samples.
- Best CV config: logistic $\theta = 0.70$ no filter, **60.68% signal-weighted direction accuracy**.
- **Single-config post-selection diagnostic** (B = 1,000): 73 of 1,000 matched real, **$p = 0.0739$** (marginal). Mean shuffled accuracy 49.73% ± 8.05%.
- **Full-grid multiple-testing-aware MAIN result** (B = 100): 15 of 100 matched best-of-grid, **$p = 0.1584$** (NOT significant at $\alpha = 0.05$). Mean best-of-grid on shuffled 55.08% ± 6.03%.
- Methodology corrections vs v9: signal-weighted accuracy replaces unweighted row mean; block_shuffle_targets_preserve_remainder replaces v9's truncating shuffle. Both corrections widen the null distribution and reduce the real-data accuracy.
- Compute: 763.1 min total.
- **This is the headline result the paper now reports.**

### v11: Leave-one-asset-out + stronger baseline ladder
- For each held-out asset, trained on the union of the other 6 pooled, evaluated only on the held-out asset's signals.
- logistic LOAO: 73.76% on n = 2,134 (most selective; BTC LOAO holds at 83.80%).
- xgboost / lightgbm / catboost LOAO: 62–64% on ~8,000 signals each, but BTC LOAO collapses to ~46%.
- Returns-only logistic LOAO baseline: 58.32% — TDA + microstructure logistic beats returns-only logistic by ~15 pp on identical data.
- Rule baselines (momentum, vol-breakout, B&H, cash) at chance.
- Headline: TDA + microstructure logistic carries cross-asset transferability that pure returns-only does not.

### v12: TDA-representation ablation (v1 scalars vs v2 persistence images, leak-safe)
- 5-way ablation × 2 classifiers, 1,095-day pool, **leak-safe** (causal point clouds, horizon-purged train, per-fold imager fit, partial-block-preserving shuffle).
- Logistic + L2 ($C = 0.5$): base 56.68%, **base + tda_v1 = 58.06% (Δ +1.38 pp)**, base + tda_v2 = 56.68% (Δ 0.00 pp), base + v1 + v2 = 58.09%, base + shuffled_v2 = 56.68% (control fires identically to base+v2).
- xgboost: every TDA condition hurts by 0.45–1.06 pp vs base alone.
- The +1.38 pp logistic v1 lift on the 3-year pool **reverses the v9 ablation finding** ($-5.55$ pp on 365 days, leakier methodology).
- The 200-dim v2 representation is L2-nullified under logistic and produces byte-identical results to the shuffled control under both classifiers — meaning the classifier responds to the v2 distribution but not to per-window v2 values.
- Honest read: v1 carries a small per-window signal that the lossy 16-scalar representation captures more efficiently than the 200-dim v2 grid; xgboost cannot extract anything; multi-asset 3-year pool is the regime under which v1 helps logistic at all.

---

## Methodological synthesis

The v9 → v10 → v12 transition is the principal scientific story of this revision. Each correction in v10 and v12 was made independently and for a different reason:

1. **Signal-weighted accuracy (v10).** v9 averaged direction accuracy across (fold, symbol) cells with equal weight even when one cell had 5,000 signals and another had 5. Signal-weighted is the correct pooled estimator. Effect: the v9 best-on-Train+Val drops from 62.23% to 61.88%; the v10 best-on-3-year drops from 62.94% (rerun under the older method) to 60.68%.

2. **Partial-block-preserving block shuffle (v10).** v9's `block_shuffle_targets` used `n_blocks = n // block_hours` and dropped rows that fell into an incomplete final block, narrowing the null distribution. The corrected helper preserves all rows. Effect: the multiple-testing-aware p-value rises from $\le 0.038$ (B = 25) to $0.1584$ (B = 100, full grid).

3. **Causal per-window normalization in TDA (v12 + multi_asset_pipeline.fix).** Earlier point-cloud construction normalized X_tda using global mean/std; the corrected pipeline normalizes each window with history up to the window's end only. Effect: v12 ablation on 3-year pool gives $+1.38$ pp for v1 vs the v9 $-5.55$ pp.

4. **Horizon-purged training indices (v10/v12).** Earlier evaluator did not purge train rows whose target peeks `horizon` rows ahead into the test fold; the corrected evaluator does. Effect: validation accuracy is reduced slightly across all configs.

The outgoing message: **the v6/v9 abstract numbers are no longer the right numbers to cite.** The v10 full-grid $p = 0.1584$ and the v12 ablation $+1.38$ pp are the corrected numbers, and they tell a more conservative — but methodologically defensible — story.
