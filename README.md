# Topological Data Analysis for Cryptocurrency Return Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/minhachung/tda-crypto-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/minhachung/tda-crypto-trading/actions/workflows/ci.yml)

> Persistent homology features extracted from sliding windows of OHLCV + microstructure data combined with a tree-based classifier produce a small positive but **statistically non-significant** directional association in cryptocurrency markets. Under leak-safe paper-grade methodology on a 3-year multi-asset pool, the best-of-grid 3-day-horizon configuration achieves **60.68% signal-weighted direction accuracy** (n = 183,379), but the multiple-testing-aware **full-grid permutation p-value is 0.1584** — *not* significant at $\alpha = 0.05$. This README documents the full v1→v12 methodology progression and the v10/v12 corrections that downgraded the earlier v9 claim of $p = 0.0000$ to the current honest negative-leaning result.

## Overview

This repository contains the full code, data pipeline, validation framework, and reproducibility tooling for an academic study on whether topological summaries of crypto market microstructure carry predictive signal about future price direction.

We compute Vietoris-Rips persistent homology in dimensions 0 and 1 over 20-hour sliding windows of a 15-feature representation (returns, volatility, volume z-score, trend indicators) for seven liquid cryptos (BTC, ETH, SOL, ADA, DOT, LINK, AVAX), summarise each persistence diagram with either 8 scalar statistics per dimension (16-dim v1) or a 10×10 persistence-image grid per dimension (200-dim v2), and combine them with the base features as inputs to logistic / random-forest / XGBoost classifiers predicting binary direction.

The paper now makes three honest claims, after the v10/v12 methodology corrections:

1. **The directional association is small and not statistically significant.** Under properly leak-safe and multiple-testing-aware methodology, the best-of-grid accuracy of 60.68% on a 3-year multi-asset pool fails to reject the chance-plus-cherry-picking null ($p = 0.1584$, B = 100 full-grid permutation; the $B = 1{,}000$ post-selection single-config diagnostic is marginal at $p = 0.0739$).
2. **TDA *does* add a small ablation lift under logistic on the 3-year pool.** v12 reports `base + tda_v1` = 58.06% vs `base` alone = 56.68% on signal-weighted accuracy — a $+1.38$pp delta. This contradicts the v9 ablation's $-5.55$pp on 365 days, which was inflated by methodology choices (unweighted row-mean accuracy, partial-block-truncating shuffle) that v10/v12 correct.
3. **The strategy is capital-preserving but not income-producing.** v8 continuous walk-forward on the leak-safe pipeline shows strategy returns of $-5.2$% to $-18.3$% across two assets and two fee venues, but consistently beats buy-and-hold by 29 to 55 percentage points across the 365-day evaluation window (B&H lost 47 to 60% over the same window).

## Repository structure

```
tda-crypto-trading/
├── src/                          # Core modules (data, features, models, backtest)
│   ├── binance_data.py           # Coinbase Exchange API fetcher (paginated hourly)
│   ├── advanced_features.py      # 15 microstructure features (Garman-Klass, etc.)
│   ├── persistent_homology.py    # Ripser-based H0/H1 computation + 8 statistics
│   ├── multi_asset_pipeline.py   # Pool features across assets
│   ├── trading_signals.py        # Probability → BUY/SELL/HOLD
│   ├── ml_signals.py             # ML classifier signal generator
│   ├── regime_filter.py          # Volatility regime filter
│   ├── backtester.py             # Vectorized cost-aware backtester
│   ├── validation.py             # v1 (basic)
│   └── validation_v2.py          # v2 (k-fold + grid search + Wilson CI + bootstrap)
├── examples/                     # Validation runners (v1 → v9)
│   ├── run_validation.py         # v1
│   ├── run_validation_v2.py      # v2: k-fold + grid search
│   ├── run_validation_v3.py      # v3: ML classifier
│   ├── run_validation_v4.py      # v4: multi-asset pool + advanced features
│   ├── run_validation_v6.py      # v6: horizon sweep + paper-quality output
│   ├── run_validation_v7.py      # v7: ADA+SOL profitability + 365 days
│   ├── run_validation_v8.py      # v8: continuous walk-forward
│   └── run_validation_v9.py      # v9: holdout + ablation + permutation test
├── results/
│   ├── RESULTS.md                # Main paper (markdown source)
│   ├── PAPER.pdf                 # Compiled 12-page paper
│   ├── V8_WALKFORWARD.md         # Profitability walk-forward results
│   ├── V9_RIGOROUS.md            # Holdout + ablation + permutation results
│   ├── figures/                  # All publication figures (.pdf and .png)
│   └── tables/                   # LaTeX-ready tables
├── docs/                         # Theory + architecture documentation
├── tests/                        # pytest suite (unit + integration)
├── scripts/build_pdf.py          # Markdown → PDF converter
├── requirements.txt              # Pinned dependencies
└── README.md                     # This file
```

## Headline Results — paper-grade (post v10/v12 corrections)

**v10 multi-year permutation tests (3-year multi-asset pool, n = 183,379):**

| Test | $B$ | Real-data accuracy | Null mean ± std | $p$-value | Verdict |
|------|----:|-------------------:|-----------------|----------:|---------|
| Single-config (post-selection diagnostic) | 1,000 | 60.68% | 49.73% ± 8.05% | **0.0739** | marginal |
| **Full-grid (multiple-testing-aware MAIN)** | 100 | 60.68% | 55.08% ± 6.03% | **0.1584** | **NOT significant** |

**v12 ablation under leak-safe methodology (1,095-day pool, signal-weighted, per-fold imager fit):**

| Feature Set | logistic | xgboost |
|-------------|---------:|---------:|
| base | 56.68% / 4,042 | 54.63% / 13,254 |
| base + tda_v1 (16 scalars) | **58.06% / 6,221 (Δ +1.38 pp)** | 53.59% / 14,886 (Δ −1.04 pp) |
| base + tda_v2 (200-dim images) | 56.68% / 4,042 (Δ 0.00 pp) | 54.18% / 13,049 (Δ −0.45 pp) |
| base + v1 + v2 | 58.09% / 6,225 (Δ +1.41 pp) | 53.57% / 15,200 (Δ −1.06 pp) |
| base + shuffled_v2 (control) | 56.68% / 4,042 (Δ 0.00 pp) | 54.18% / 13,049 (Δ −0.45 pp) |

**v9 holdout (one-shot, never-touched window, signal-weighted):**

| Metric | Value |
|--------|-------|
| Pooled holdout accuracy | **65.30%** (Wilson CI [60.49%, 69.75%]) |
| n signals | 403 |
| Beat B&H | 2 / 7 assets (ADA, DOT) |
| BTC contribution | $n = 4$ — uninterpretable, excluded |

**v8 continuous walk-forward on the leak-safe pipeline (365 days):**

| Scenario | Asset | Strategy | B&H | Δ vs B&H | Sharpe |
|----------|-------|---------:|----:|---------:|-------:|
| Binance Maker | ADA | $-5.20$% | $-60.38$% | **$+55.17$ pp** | $-0.30$ |
| Binance Maker | SOL | $-15.01$% | $-47.18$% | $+32.17$ pp | $-1.45$ |
| Coinbase Taker | ADA | $-9.49$% | $-60.38$% | $+50.89$ pp | $-0.61$ |
| Coinbase Taker | SOL | $-18.34$% | $-47.18$% | $+28.85$ pp | $-1.82$ |

The strategy is **capital-preserving** (beats buy-and-hold by 29–55 pp on every cell during a sustained drawdown) but produces **negative absolute returns** under realistic transaction costs.

## Quick start

### Option A: Local Python (3.10+)

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce the v9 rigorous result (holdout + ablation + B=30 permutation)
python examples/run_validation_v9.py 365 72 30        # ~1.5h

# Reproduce the v10 paper-grade headline (multi-year + B=1,000 single-config + B=100 full-grid)
python examples/run_validation_v10.py 1095 1000 100   # ~12h on Apple Silicon

# Reproduce the v12 leak-safe ablation (v1 vs v2 representation, both classifiers)
python examples/run_validation_v12.py 1095            # ~20m
```

### Option B: Continuous walk-forward profitability test

```bash
# v8 — answers "would this have made money on real data?" (leak-safe pipeline)
python examples/run_validation_v8.py 365 168 0.65 0.50
```

### Option C: Just read the paper

[`results/PAPER.pdf`](results/PAPER.pdf) — 12-page formatted document with abstract, methods, results, discussion, references, and embedded figures.

## Reproducing the paper

The figures and tables in `results/PAPER.pdf` are reproduced by running:

```bash
python examples/run_validation_v6.py 90    # ~30 minutes on a Mac M1
```

Output goes to `results/figures/*.pdf` (4 figures), `results/tables/*.tex` (LaTeX tables), and `results/RESULTS.md` (paper source).

For the rigorous holdout + ablation + permutation evidence:

```bash
python examples/run_validation_v9.py 365 72 30   # ~45 minutes
```

## Validation Methodology Progression — v1 through v12

See [`VALIDATION_PROGRESSION.md`](VALIDATION_PROGRESSION.md) for the full narrative. Headline numbers below reflect the **post-fix** (leak-safe, signal-weighted, partial-block-preserving) reruns where applicable.

| Version | What it added | Headline | Significance |
|--------:|---------------|----------|--------------|
| v1 | Single train/val/test split | invalid (n=2) | — |
| v2 | 5-fold time-series CV + Wilson CI | 50.75% | ❌ |
| v3 | ML classifier on TDA features | 56.18% | borderline |
| v4 | Multi-asset pool + advanced features | 60.42% (n=26) | underpowered |
| v5 | 7 assets + looser filter (n=684) | 59.40% | ✅ Wilson lower-bound > 50% |
| v6 | Horizon sweep + paper output | 61.77% (3d, pre-fix) | ✅ in v6 framing |
| v7 | 365-day cross-regime profitability | 67–71% per regime (pre-fix) | ✅ in v7 framing |
| v8 | Continuous walk-forward (leak-safe rerun) | Strategy −5.2% to −18.3%; **beats B&H by +29–55 pp** | capital-preserving |
| v9 | Rigorous holdout + ablation + B=30 perm (post-fix) | CV 61.88%, holdout 65.30%, ablation Δ=−5.6pp, p=0.0000 | ✅ at $\alpha = 0.05$ within v9 framing |
| **v10** | **Multi-year + B=1,000 + B=100 grid + signal-weighted + leak-aware shuffle** | **Best 60.68%, single-config p=0.0739, full-grid p=0.1584** | **NOT significant at $\alpha = 0.05$** |
| v11 | LOAO + stronger baselines (XGB/LGBM/CatBoost/momentum/vol-breakout) | logistic LOAO 73.76% vs returns-only 58.32% (+15 pp) | ✅ cross-asset transferability |
| **v12** | **TDA-rep ablation (v1 scalars vs v2 persistence images, leak-safe)** | **base+v1 logistic +1.38 pp; v2 L2-nullified; shuffled control fires** | reverses v9 ablation sign |

## Limitations

- **Sample window:** Headline 1,095 days (3 years) in v10 and v12. Even longer samples (5+ years) could shift the v10 full-grid p-value in either direction.
- **Cost model:** Point estimates of fees and slippage; does not model market impact, partial fills, or exchange downtime.
- **No live deployment:** All results are offline cross-validation. Live-vs-backtest gap is unknown.
- **TDA representation:** Persistence landscapes at multiple resolutions, alpha complexes, and learned filtrations remain unexplored and could plausibly surface signal that v1/v2 miss.
- **xgboost cannot extract TDA signal:** Every v1 / v2 / v1+v2 condition under xgboost hurts vs base alone on the leak-safe pipeline. Larger samples or stronger regularization would be needed to retest.
- **Methodology was corrected during paper preparation:** The earlier v9 abstract reported $p = 0.0000$. The v10 corrected p-value is $0.1584$. Both numbers refer to the *same* experimental question; only the methodology differs. We document the change in `VALIDATION_PROGRESSION.md`.

## Citation

If you build on this work, please cite:

```bibtex
@misc{chung2026tda,
  author = {Chung, Minha},
  title  = {Topological Data Analysis Reveals Predictable Structure in Cryptocurrency Returns},
  year   = {2026},
  url    = {https://github.com/minhachung/tda-crypto-trading},
  note   = {Preprint, version 6}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

This work builds on the persistence-landscape-of-crashes line of research initiated by Gidea, Goldsmith, Katz, Roldan, and Shmalo (2020). The Ripser library by Bauer (2021) is used throughout for persistent homology computation.
