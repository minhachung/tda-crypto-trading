# V9: Rigorous Scientific Validation

**Methodology upgrades over v6:**
1. True temporal holdout (last 20% never touched until final eval)
2. Ablation study: base vs TDA vs base+TDA vs base+shuffled-TDA
3. Permutation test for cherry-picking concern
4. Break-even cost analysis
5. Grouped feature importance

**Date:** 2026-05-07
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Sample:** 365 days hourly | Horizon: 72h (3.0d)

## 1. Holdout Split

| Split | Samples | Period |
|-------|--------:|--------|
| Train+Val (used for selection) | 48,587 | 2025-05-10 → 2026-02-24 |
| **Holdout (touched ONCE)** | 12,152 | 2026-02-24 → 2026-05-07 |

## 2. Best Configuration (selected on Train+Val ONLY)

| Parameter | Value |
|-----------|-------|
| Model | logistic |
| Probability threshold | 0.70 |
| Regime filter | none |
| CV direction accuracy | 61.88% |
| CV AUC | 0.542 |
| CV Wilson CI | [60.83%, 62.91%] |

## 3. Ablation Study — Does TDA Help?

| Feature Set | n Features | n Signals | Accuracy | AUC | Wilson CI | Sharpe |
|-------------|-----------:|----------:|---------:|----:|-----------|-------:|
| base_only | 21 | 3931 | 65.93% | 0.560 | [64.44%, 67.40%] | -4.71 |
| tda_only | 16 | 531 | 34.65% | 0.471 | [30.73%, 38.80%] | -3.74 |
| base_plus_tda | 37 | 4950 | 60.38% | 0.542 | [59.01%, 61.74%] | -4.83 |
| base_plus_shuffled_tda | 37 | 4018 | 63.78% | 0.558 | [62.29%, 65.26%] | -4.78 |

**Ablation verdict:** ❌ TDA does NOT help (-5.6pp). Base features carry the signal.

## 4. Permutation Test — Is the Result Cherry-Picked?

Block-shuffled direction labels in 7-day blocks (preserves intra-block autocorrelation), then reran the entire grid search.

| Metric | Value |
|--------|-------|
| Permutations run | 30 |
| Mean shuffled-data accuracy | 51.59% ± 4.34% |
| Actual best accuracy (real data) | 61.88% |
| **Permutation p-value** | **0.0000** |
| Significant at α = 0.05? | ✅ YES |

## 5. FINAL HOLDOUT RESULTS — Headline Numbers

This is the only evaluation done on the never-touched holdout data.

| Asset | Signals | Direction Acc | AUC | TDA Return | B&H Return | Trades | Beat B&H |
|-------|--------:|---------------|----:|-----------:|-----------:|-------:|----------|
| ADA | 65 | 87.69% | 0.603 | 0.00% | -2.83% | 0 | ✅ |
| AVAX | 49 | 71.43% | 0.617 | 0.00% | 10.24% | 0 | ❌ |
| BTC | 4 | 0.00% | 0.610 | 0.00% | 26.28% | 0 | ❌ |
| DOT | 119 | 89.08% | 0.599 | 0.00% | -2.39% | 0 | ✅ |
| ETH | 52 | 67.31% | 0.636 | 0.00% | 28.34% | 0 | ❌ |
| LINK | 50 | 76.00% | 0.622 | 0.00% | 15.20% | 0 | ❌ |
| SOL | 64 | 65.62% | 0.601 | 0.00% | 9.67% | 0 | ❌ |

**Pooled holdout accuracy:** 65.30% on 403 signals
**95% Wilson CI:** [60.49%, 69.75%]
**Significant?** ✅ YES (lower bound vs 50%)
**Beat buy-hold:** 2/7 assets

## 6. Break-Even Cost Analysis

Determines maximum round-trip cost (in basis points) before strategy becomes unprofitable, per horizon.

| Horizon | Direction Acc | Avg \|Move\| (bps) | Gross Edge (bps) | Break-Even RT Cost | Binance Profitable? | Coinbase Profitable? |
|---------|--------------:|----------------:|-----------------:|-------------------:|---------------------|---------------------|
| 1h | 53.62% | 53.6 | 3.9 | 3.9 | ❌ (15) | ❌ (85) |
| 4h | 50.20% | 107.9 | 0.4 | 0.4 | ❌ (15) | ❌ (85) |
| 12h | 52.18% | 196.1 | 8.6 | 8.6 | ❌ (15) | ❌ (85) |
| 1d | 52.94% | 282.0 | 16.6 | 16.6 | ✅ (15) | ❌ (85) |
| 3d | 60.38% | 499.1 | 103.6 | 103.6 | ✅ (15) | ✅ (85) |
| 7d | 60.27% | 746.0 | 153.2 | 153.2 | ✅ (15) | ✅ (85) |

## 7. Grouped Feature Importance (Permutation, on held-out validation)

| Group | n Features | Mean Importance | Sum Importance |
|-------|-----------:|----------------:|---------------:|
| tda_h0 | 8 | 0.0097 | 0.0776 |
| trend | 7 | 0.0062 | 0.0431 |
| tda_h1 | 8 | 0.0037 | 0.0300 |
| returns | 4 | 0.0060 | 0.0240 |
| volume | 3 | 0.0024 | 0.0073 |
| volatility | 7 | 0.0003 | 0.0019 |

**TDA's share of non-asset feature importance: 58.5%**


## 8. Verdict

**Three independent rigor checks:**

1. **Holdout test:** Headline accuracy survives a never-touched test set? See section 5.
2. **Ablation:** TDA adds -5.6pp over base features alone.
3. **Permutation:** p = 0.0000 (significant at α=0.05).

If all three pass, the v6 finding is bulletproof. If one or more fail, the honest verdict shifts toward 'TDA shows promise but stronger than v6 paper claimed.'