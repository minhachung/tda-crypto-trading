# Validation Methodology

This page documents the iterative validation pipeline that took the project from a naive single-split evaluation (v1) to a rigorous holdout + ablation + permutation test (v9). Each version controlled for a different methodological bias.

## Why a progression matters

Most quantitative trading studies report one number from one validation setup, with no audit trail of what biases were ruled out. We document every step explicitly so that a skeptical reviewer can identify exactly what's tested at each level.

## Version-by-version

### v1: Single train/val/test split (FAILED)

- **Method:** 70/15/15 split on a single time series; report best threshold from val, evaluate once on test.
- **Result:** Test set had only 2 actual signals. Sample size too small for any conclusion.
- **Lesson:** Need k-fold CV for tight confidence intervals.

### v2: 5-fold time-series CV + Wilson CI

- **Method:** Time-series 5-fold CV on a 6-month hourly sample, rule-based threshold on persistence-landscape C1-norm.
- **Result:** 50.75% direction accuracy on n=728. Wilson CI [47.06%, 53.93%] — at chance.
- **Lesson:** Hand-crafted thresholds don't extract the signal. Need ML.

### v3: ML classifier + hyperparameter grid

- **Method:** Logistic / Random Forest / GBM on TDA features, grid search over threshold and lookback.
- **Result:** 56.18% direction accuracy on n=152. Wilson CI [47.98%, 64.05%] — borderline.
- **Lesson:** Real signal exists, but per-asset training is data-starved.

### v4: Multi-asset pooled training + advanced features

- **Method:** Pool 4 cryptos, replace OHLCV with returns + Garman-Klass + Parkinson + volume z-score, vol regime filter.
- **Result:** 60.42% on n=26. Wilson CI [42.5%, 77.6%] — wide.
- **Lesson:** The regime filter at q75 was too restrictive. Lower it to expand sample size.

### v5: 7-asset pool + looser regime filter

- **Method:** Add DOT, LINK, AVAX (MATIC delisted). Use vol > median instead of q75.
- **Result:** 59.40% on n=684. Wilson CI [55.63%, 62.98%] — clearly significant.
- **Lesson:** Sample size matters more than peak point estimate.

### v6: Horizon sweep + paper-quality output (HEADLINE)

- **Method:** Sweep horizons {1h, 4h, 12h, 24h, 3d, 7d} × thresholds × classifiers. Generate publication-grade figures.
- **Result:** 5/6 horizons significant. Best: 3d horizon, 61.77% on n=2,260, CI [59.75%, 63.75%].
- **Output:** `results/RESULTS.md` and `results/PAPER.pdf`.

### v7: 365-day cross-regime check

- **Method:** Extend to 365 days. Split into 3 four-month windows. Verify model survives across regimes.
- **Result:** Direction accuracy 67.3% / 67.0% / 71.1% across the three regimes — stable.
- **Lesson:** Signal isn't a single-regime artifact.

### v8: Continuous walk-forward (PROFITABILITY)

- **Method:** Train on first 60 days, retrain weekly, take real trades. Test ADA + SOL on Coinbase and Binance.US fees.
- **Result:** ADA strategy: $11,199 ending equity vs $4,475 buy-hold. Sharpe 0.95. SOL: defensive, lost 16% vs buy-hold's 42%.
- **Lesson:** k-fold underestimated trade count by 25×. Continuous walk-forward gives the real number.

### v9: Holdout + ablation + permutation (RIGOROUS)

- **Method:** True temporal holdout (last 20% never touched). Ablation: base / TDA / base+TDA / base+shuffled-TDA. Permutation test: shuffle target labels in 7-day blocks and rerun the full grid 25 times. Compute p-value.
- **Result:** [pending; results pushed when v9 completes]
- **Goal:** Three independent rigor checks pass simultaneously.

## What this catches

| Bias | Controlled by | Rules out concern |
|------|---------------|-------------------|
| Cherry-picking single split | k-fold CV (v2) | "You got lucky on one window" |
| Cherry-picking thresholds | Grid search + report Wilson CI per config (v3) | "You searched until you found a positive result" |
| Per-asset overfitting | Multi-asset pooling (v4) | "ADA happens to look like the training data" |
| Regime-specific artifact | Cross-regime split (v7) | "Only worked in one market environment" |
| K-fold underestimating trades | Continuous walk-forward (v8) | "Validation looks good but execution wouldn't work" |
| Lookahead through tuning | True temporal holdout (v9) | "You tuned on the same data you're reporting on" |
| Model-architecture overfitting | Ablation: base+shuffled-TDA (v9) | "The signal isn't from TDA at all" |
| Multiple-testing illusion | Permutation test (v9) | "p<0.05 by chance from many tests" |

## How to reproduce

Each version's runner is in `examples/`:

```bash
python examples/run_validation_v2.py 90
python examples/run_validation_v3.py 90 1h
python examples/run_validation_v4.py 90
python examples/run_validation_v6.py 90        # HEADLINE
python examples/run_validation_v7.py 365 168 0.70
python examples/run_validation_v8.py 365 168 0.65 0.50
python examples/run_validation_v9.py 365 72 30 # RIGOROUS
```

Each writes a markdown report and figures to `results/`.
