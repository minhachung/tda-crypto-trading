# V10: Multi-Year Sample + 1000-Permutation Test

**Goal:** Address two limitations from v9:
1. Sample window: extend from 365 days to 1095 days (3.0 years)
2. Permutation count: extend from B=25 to B=1000 (single config) and B=100 (full grid)

**Date:** 2026-05-08
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Pool size:** 183,379 samples
**Train+Val:** 146,699 | **Holdout:** 36,680

## 1. Best Configuration (selected on multi-year Train+Val)

| Parameter | Value |
|-----------|-------|
| Model | logistic |
| Probability threshold | 0.70 |
| Regime filter | None |
| CV direction accuracy | **60.68%** |

## 2. Post-Selection Single-Config Diagnostic (B = 1000)

Block-shuffled targets in 7-day blocks (final partial block preserved), reran the BEST config.

**Important caveat:** the best config was selected on the unshuffled real data, so this null distribution is over chance variation *at this single config only*, not over chance + multiple testing across the grid. Treat this section as a sanity check; section 3 (full-grid permutation) is the headline multiple-testing-aware result.

| Metric | Value |
|--------|-------|
| Permutations run | 1000 |
| Mean shuffled-data accuracy | 49.73% +/- 8.05% |
| Max shuffled-data accuracy | 78.02% |
| Actual accuracy | **60.68%** |
| Permutations matching real result | 73/1000 |
| **Empirical p-value** | **0.07393** |
| Computation time | 83.7 min |

## 3. Full-Grid Permutation Test (B = 100) — main multiple-testing-aware result

For each permutation, reran the FULL grid search on block-shuffled targets and recorded the BEST accuracy across the grid.
This null distribution accounts for cherry-picking across model × threshold × regime configurations, so its empirical p-value is the headline evidence v10 reports.

| Metric | Value |
|--------|-------|
| Permutations run | 100 |
| Mean best-of-grid on shuffled data | 55.08% +/- 6.03% |
| Max best-of-grid on shuffled data | 77.08% |
| Actual best-of-grid on real data | **60.68%** |
| Permutations matching real | 15/100 |
| **Empirical p-value** | **0.1584** |
| Computation time | 667.4 min |

## 4. Verdict

**Headline (full-grid, multiple-testing-aware):**
- Not significant (p = 0.1584).

**Post-selection single-config diagnostic** (sanity check, NOT a headline p-value):
- Diagnostic p = 0.0739.

This run materially strengthens the permutation evidence relative to v9, subject to the grid, sample window, and data source tested.