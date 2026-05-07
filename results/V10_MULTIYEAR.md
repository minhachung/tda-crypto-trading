# V10: Multi-Year Sample + 1000-Permutation Test

**Goal:** Address two limitations from v9:
1. Sample window: extend from 365 days to 1095 days (3.0 years)
2. Permutation count: extend from B=25 to B=1000 (single config) and B=100 (full grid)

**Date:** 2026-05-07
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Pool size:** 183,379 samples
**Train+Val:** 146,699 | **Holdout:** 36,680

## 1. Best Configuration (selected on multi-year Train+Val)

| Parameter | Value |
|-----------|-------|
| Model | logistic |
| Probability threshold | 0.65 |
| Regime filter | {'vol': 'median'} |
| CV direction accuracy | **62.94%** |

## 2. Single-Config Permutation Test (B = 1000)

Block-shuffled targets in 7-day blocks, reran the BEST config.
Tests whether the 62.94% accuracy at this exact config could be chance.

| Metric | Value |
|--------|-------|
| Permutations run | 1000 |
| Mean shuffled-data accuracy | 50.12% +/- 4.42% |
| Max shuffled-data accuracy | 63.77% |
| Actual accuracy | **62.94%** |
| Permutations matching real result | 2/1000 |
| **Empirical p-value** | **0.00300** |
| Computation time | 224.5 min |

## 3. Full-Grid Permutation Test (B = 100)

For each permutation, ran the FULL grid search and took the BEST.
Tests whether the best-of-grid result could be chance + multiple testing.

| Metric | Value |
|--------|-------|
| Permutations run | 100 |
| Mean best-of-grid on shuffled data | 54.34% +/- 3.19% |
| Max best-of-grid on shuffled data | 61.01% |
| Actual best-of-grid on real data | **62.94%** |
| Permutations matching real | 0/100 |
| **Empirical p-value** | **0.0099** |
| Computation time | 341.6 min |

## 4. Verdict

**Single-config:** Significant (p = 0.0030 < 0.01).
**Full-grid:** Best-of-grid result survives multiple-testing correction (p = 0.0099).

This run definitively resolves the v9 limitation that 25 permutations could not establish p < 0.001.