# V9: Rigorous Scientific Validation

**Methodology upgrades over v6:**
1. True temporal holdout (last 20% never touched until final eval)
2. Ablation study: base vs TDA vs base+TDA vs base+shuffled-TDA
3. Permutation test for cherry-picking concern
4. Break-even cost analysis
5. Grouped feature importance

**Date:** 2026-05-06
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Sample:** 365 days hourly | Horizon: 72h (3.0d)

## 1. Holdout Split

| Split | Samples | Period |
|-------|--------:|--------|
| Train+Val (used for selection) | 48,587 | 2025-05-10 → 2026-02-23 |
| **Holdout (touched ONCE)** | 12,152 | 2026-02-23 → 2026-05-07 |

## 2. Best Configuration (selected on Train+Val ONLY)

| Parameter | Value |
|-----------|-------|
| Model | logistic |
| Probability threshold | 0.70 |
| Regime filter | none |
| CV direction accuracy | 62.23% |
| CV AUC | 0.542 |
| CV Wilson CI | [61.09%, 63.37%] |

## 3. Ablation Study — Does TDA Help?

| Feature Set | n Features | n Signals | Accuracy | AUC | Wilson CI | Sharpe |
|-------------|-----------:|----------:|---------:|----:|-----------|-------:|
| base_only | 21 | 3313 | 64.63% | 0.554 | [62.98%, 66.23%] | -4.24 |
| tda_only | 16 | 348 | 39.98% | 0.485 | [34.93%, 45.17%] | -2.32 |
| base_plus_tda | 37 | 4266 | 62.14% | 0.542 | [60.68%, 63.59%] | -5.04 |
| base_plus_shuffled_tda | 37 | 3421 | 64.17% | 0.551 | [62.54%, 65.75%] | -5.28 |

**Ablation verdict:** ❌ TDA does NOT help (-2.5pp). Base features carry the signal.

## 4. Permutation Test — Is the Result Cherry-Picked?

Block-shuffled direction labels in 7-day blocks (preserves intra-block autocorrelation), then reran the entire grid search.

| Metric | Value |
|--------|-------|
| Permutations run | 25 |
| Mean shuffled-data accuracy | 51.18% ± 4.72% |
| Actual best accuracy (real data) | 62.23% |
| **Permutation p-value** | **0.0000** |
| Significant at α = 0.05? | ✅ YES |

## 5. FINAL HOLDOUT RESULTS — Headline Numbers

This is the only evaluation done on the never-touched holdout data.

| Asset | Signals | Direction Acc | AUC | TDA Return | B&H Return | Trades | Beat B&H |
|-------|--------:|---------------|----:|-----------:|-----------:|-------:|----------|
| ADA | 46 | 93.48% | 0.614 | 0.00% | 1.03% | 0 | ❌ |
| AVAX | 33 | 78.79% | 0.617 | 0.00% | 13.40% | 0 | ❌ |
| BTC | 4 | 0.00% | 0.620 | 0.00% | 25.38% | 0 | ❌ |
| DOT | 106 | 89.62% | 0.608 | 0.00% | 3.46% | 0 | ❌ |
| ETH | 25 | 64.00% | 0.641 | 0.00% | 25.93% | 0 | ❌ |
| LINK | 46 | 84.78% | 0.638 | 0.00% | 20.21% | 0 | ❌ |
| SOL | 55 | 74.55% | 0.596 | 0.00% | 13.37% | 0 | ❌ |

**Pooled holdout accuracy:** 69.32% on 315 signals
**95% Wilson CI:** [63.90%, 74.05%]
**Significant?** ✅ YES (lower bound vs 50%)
**Beat buy-hold:** 0/7 assets

## 6. Break-Even Cost Analysis

Determines maximum round-trip cost (in basis points) before strategy becomes unprofitable, per horizon.

| Horizon | Direction Acc | Avg \|Move\| (bps) | Gross Edge (bps) | Break-Even RT Cost | Binance Profitable? | Coinbase Profitable? |
|---------|--------------:|----------------:|-----------------:|-------------------:|---------------------|---------------------|
| 1h | 53.73% | 53.7 | 4.0 | 4.0 | ❌ (15) | ❌ (85) |
| 4h | 50.49% | 107.9 | 1.1 | 1.1 | ❌ (15) | ❌ (85) |
| 12h | 50.36% | 196.1 | 1.4 | 1.4 | ❌ (15) | ❌ (85) |
| 1d | 55.18% | 281.6 | 29.2 | 29.2 | ✅ (15) | ❌ (85) |
| 3d | 62.14% | 498.0 | 120.9 | 120.9 | ✅ (15) | ✅ (85) |
| 7d | 55.41% | 745.3 | 80.7 | 80.7 | ✅ (15) | ❌ (85) |

## 7. Grouped Feature Importance (Permutation, on held-out validation)

| Group | n Features | Mean Importance | Sum Importance |
|-------|-----------:|----------------:|---------------:|
| tda_h0 | 8 | 0.0054 | 0.0429 |
| tda_h1 | 8 | 0.0027 | 0.0215 |
| volatility | 7 | 0.0024 | 0.0170 |
| returns | 4 | 0.0030 | 0.0119 |
| trend | 7 | 0.0014 | 0.0100 |
| volume | 3 | -0.0013 | -0.0039 |

**TDA's share of non-asset feature importance: 64.8%**


## 8. Verdict

**Three independent rigor checks:**

1. **Holdout test:** Headline accuracy survives a never-touched test set? See section 5.
2. **Ablation:** TDA adds -2.5pp over base features alone.
3. **Permutation:** p = 0.0000 (significant at α=0.05).

If all three pass, the v6 finding is bulletproof. If one or more fail, the honest verdict shifts toward 'TDA shows promise but stronger than v6 paper claimed.'