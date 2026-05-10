# V13 Comprehensive Validation Results

**Date:** 2026-05-09 22:50
**Days:** 90
**Symbols:** BTC,ETH

## Headline Numbers

| Model | Test Accuracy | n_test |
|-------|---------------|--------|
| baseline_logistic | 0.5091 | 605 |
| ensemble | 0.5421 | 605 |
| regime_low | 0.5062 | 324 |
| regime_medium | 0.5241 | 145 |
| regime_high | 0.5294 | 136 |
| regime_global | 0.5157 | 605 |

## Permutation Test

- Real accuracy: 0.5421
- Null mean ± std: 0.4829 ± 0.0170
- **p-value: 0.0000** (B=30)
- **Verdict:** ✅ Significant

## Comparison vs v10/v12

| Version | Best Accuracy | p-value |
|---------|---------------|--------:|
| v10 | 60.68% | 0.1584 |
| v12 (logistic + tda_v1) | 58.06% | n/a |
| **v13** | 54.21% | **0.0000** |
