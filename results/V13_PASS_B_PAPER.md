# V13 Marginal TDA Contribution Test
**Date:** 2026-05-12 08:55
**Days:** 1095
**Symbols:** BTC,ETH,SOL,ADA,DOT,LINK,AVAX
**Horizon:** 72 rows | Window: 20 | Embargo: 24
**Synthetic features:** DISABLED (paper-grade)
**Permutations:** B=99 (min achievable p = 0.0100)
## Reality check: which feature columns are real vs synthetic?
- **Real feature columns (59):** `timestamp, open, high, low, close, volume, log_return, log_return_5, log_return_24, accel, gk_vol_20, gk_vol_60, parkinson_20, rv_20, rv_60, hl_spread, oc_spread, vol_zscore_20, vol_momentum, vol_ratio_5_20...`

## Real external data coverage per asset
| Asset | On-chain (blockchain.info) | Cross-exchange (Coinbase+Kraken) |
|-------|---------------------------:|---------------------------------:|
| BTC | ✓ 1096 daily rows | ✓ 721 daily rows |
| ETH | — | ✓ 721 daily rows |
| SOL | — | ✓ 721 daily rows |
| ADA | — | ✓ 721 daily rows |
| DOT | — | ✓ 721 daily rows |
| LINK | — | ✓ 721 daily rows |
| AVAX | — | ✓ 721 daily rows |

## 7-condition ablation table
| Condition | Signal-weighted Acc | n_signals | mean AUC |
|-----------|-------------------:|----------:|---------:|
| base | 0.5737 | 3188 | 0.4957 |
| v1_only | 0.5871 | 1797 | 0.5032 |
| v2_only | 0.5000 | 0 | 0.5000 |
| base_plus_v1 | 0.5391 | 6008 | 0.5009 |
| base_plus_v2 | 0.5740 | 3190 | 0.4958 |
| base_plus_v1_plus_v2 | 0.5386 | 6003 | 0.5009 |
| base_plus_shuffled_v2 | 0.5740 | 3190 | 0.4958 |

## v2 verdict (real vs shuffled)
🟡 Real v2 vs shuffled: +0.00 pp — within noise. v2 likely carries no real signal beyond the shuffled control.

## Permutation test — FULL-GRID (multiple-testing aware)
- **Real best:** `v1_only` = 0.5871 (picked among 6 real conditions)
- **Null distribution:** for each of B=99 permutations, all real conditions were re-evaluated under the SAME shuffled labels, and the MAX accuracy across conditions was taken. This is the multiple-testing-aware null.
- Null max mean ± std: 0.8845 ± 0.1050
- **p_grid = 1.0000** (B=99, lower-bounded at 0.0100)
- Verdict: ❌ NOT significant at α=0.05 after multiple-testing correction.

## H0 vs H1 contribution
| Homology | Signal-weighted Acc | n_test |
|----------|-------------------:|-------:|
| H0 only | 0.5087 | 152355 |
| H1 only | 0.5087 | 152355 |

## Methodology notes
- **Targets:** built via `multi_asset_pipeline.add_targets`; final `horizon` rows per asset are dropped (no silent target=0).
- **Splits:** per-asset time-series 5-fold; train indices purged so `max(train) + horizon + embargo < min(test)`.
- **TDA v2 imager:** fit on each fold's purged-train diagrams; transform-only on val/test.
- **Causal point-cloud normalization:** each window's mean/std uses only history up to window end.
- **Permutation:** train+val labels shuffled consistently; test labels untouched. p = (n_above + 1) / (B + 1) so the reported p is never 0.
- **Model selection:** condition chosen by signal-weighted accuracy across all folds is a *post-selection* choice; the permutation p reported is therefore a diagnostic, not a multiple-testing-aware main result. For multiple-testing-aware p, the full 7-condition × B grid would need to be permuted.

## Comparison vs prior versions
| Version | Headline | p-value | Notes |
|---------|---------:|--------:|-------|
| v10 (1095d, 7 assets) | 60.68% | 0.1584 | full-grid, NOT significant |
| v12 (logistic + tda_v1) | 58.06% | n/a | leak-safe ablation |
| **v13** | 58.71% (v1_only) | **1.0000** | post-selection single-config |
