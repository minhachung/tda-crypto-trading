# V12: TDA-Representation Ablation

**Question:** Is v9's ablation-negative result (Base+TDA underperforms Base alone by 5.6pp) caused by the v1 representation (8 hand-engineered scalar stats per dimension) being too lossy?

**Test:** Five-way ablation under both a linear (logistic, L2) and a non-linear (xgboost) classifier, on identical point clouds, with persistence images replacing the scalar TDA features. Persistence imagers are fit *per fold* on training diagrams only — never on the full pool — so no future TDA-feature distribution information leaks into the representation.

**Date:** 2026-05-08
**Symbols:** BTC, ETH, SOL, ADA, DOT, LINK, AVAX
**Sample:** 1095 days hourly | Horizon: 72h | Imager: 10×10 per H₀/H₁

## 1. Ablation Results (5-fold time-series CV)

| Feature Set | Model | n Signals | Direction Acc | Wilson 95% | AUC |
|-------------|-------|----------:|--------------:|-----------|----:|
| base | logistic | 4042 | 56.68% | [55.15%, 58.20%] | 0.508 |
| base | xgboost | 13254 | 54.63% | [53.78%, 55.48%] | 0.513 |
| base_plus_v1 | logistic | 6221 | 58.06% | [56.83%, 59.28%] | 0.515 |
| base_plus_v1 | xgboost | 14886 | 53.59% | [52.79%, 54.39%] | 0.513 |
| base_plus_v2 | logistic | 4042 | 56.68% | [55.15%, 58.20%] | 0.508 |
| base_plus_v2 | xgboost | 13049 | 54.18% | [53.32%, 55.03%] | 0.511 |
| base_plus_v1_plus_v2 | logistic | 6225 | 58.09% | [56.86%, 59.31%] | 0.515 |
| base_plus_v1_plus_v2 | xgboost | 15200 | 53.57% | [52.78%, 54.36%] | 0.512 |
| base_plus_shuffled_v2 | logistic | 4042 | 56.68% | [55.15%, 58.20%] | 0.508 |
| base_plus_shuffled_v2 | xgboost | 13049 | 54.18% | [53.32%, 55.03%] | 0.511 |

## 2. Per-Model Diagnostic

Δ measures the absolute change in pooled direction accuracy vs the model's `base` row.

### logistic

| Component | Accuracy | Δ vs base |
|-----------|---------:|----------:|
| base                  | 56.68% |  — |
| base + tda_v1         | 58.06% | +1.38% |
| base + tda_v2         | 56.68% | +0.00% |
| base + v1 + v2        | 58.09% | +1.41% |
| base + shuffled_v2 (control) | 56.68% | +0.00% |

**Verdict:** The richer v2 representation does not add positive marginal accuracy. The v9 ablation-negative result reflects a deeper limitation than feature lossiness; future work should explore window size, filtration choice, or alternative TDA constructions.

Unshuffled-v2 vs shuffled-v2 separation: 0.00%. A small separation suggests the classifier is exploiting the *distribution* of v2 features rather than their window-specific values.

### xgboost

| Component | Accuracy | Δ vs base |
|-----------|---------:|----------:|
| base                  | 54.63% |  — |
| base + tda_v1         | 53.59% | -1.04% |
| base + tda_v2         | 54.18% | -0.45% |
| base + v1 + v2        | 53.57% | -1.06% |
| base + shuffled_v2 (control) | 54.18% | -0.45% |

**Verdict:** The richer v2 representation does not add positive marginal accuracy. The v9 ablation-negative result reflects a deeper limitation than feature lossiness; future work should explore window size, filtration choice, or alternative TDA constructions.

Unshuffled-v2 vs shuffled-v2 separation: 0.00%. A small separation suggests the classifier is exploiting the *distribution* of v2 features rather than their window-specific values.

## 3. Methodology Notes

- **Leak-safe imager fit.** A fresh `PersistenceImager` is fit on the union of training-fold diagrams across all assets at every fold, then used to transform that fold's training and test diagrams. The full-pool fit (which would leak holdout distribution into the feature representation) is never performed.

- **Fixed image grid.** The imager is configured with `birth_range = pers_range = max(birth_max, pers_max)` of the training diagrams, with `pixel_size` chosen to produce exactly a 10×10 grid per homology dimension. No silent padding or cropping.

- **Empty-diagram robustness.** If a fold's training diagrams contain no finite-persistence content for a given homology dimension, that dimension's image features are returned as zeros instead of raising at fit time.

- **Regularisation.** Logistic uses `C=0.5` (stronger L2 than v9's `C=1.0`) to control over-fit on the 200-dim v2 representation. xgboost uses depth=4, n_estimators=200, subsample/colsample 0.9.

- **Negative control.** `base_plus_shuffled_v2` shuffles the v2 feature rows within each train fold, breaking the v2-window correspondence while preserving the v2 distribution. If this control matches `base_plus_v2`, the classifier is responding to the *distribution* of v2 rather than to per-window values.

## 4. Read against v9

v9 reported, on a 365-day pool: Base 65.93%, Base+v1_TDA 60.38% (Δ = −5.55pp). v12 results above are on a 1095-day pool with identical point clouds for both representations — the only variable is the persistence-diagram → feature mapping. A non-negative v2 delta narrows the v9 gap and supports the diagnosis that v1's lossy scalar summaries, not TDA itself, were the issue.
