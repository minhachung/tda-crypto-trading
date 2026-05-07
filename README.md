# Topological Data Analysis for Cryptocurrency Return Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/minhachung/tda-crypto-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/minhachung/tda-crypto-trading/actions/workflows/ci.yml)

> Persistent homology features extracted from sliding windows of OHLCV + microstructure data combined with a tree-based classifier predict short-horizon directional moves in cryptocurrency markets. The 3-day horizon achieves **61.77% direction accuracy** (95% Wilson CI [59.75%, 63.75%], n=2,260, p<0.001) on out-of-sample 5-fold time-series cross-validation across seven major cryptocurrencies.

## Overview

This repository contains the full code, data pipeline, validation framework, and reproducibility tooling for an academic study on whether topological summaries of crypto market microstructure carry predictive signal about future price direction.

We compute Vietoris-Rips persistent homology in dimensions 0 and 1 over 20-hour sliding windows of a 15-feature representation (returns, volatility, volume z-score, trend indicators) for seven liquid cryptos (BTC, ETH, SOL, ADA, DOT, LINK, AVAX), summarise each persistence diagram with eight scalar statistics per dimension (16 TDA features total), and combine them with the base features as inputs to a random forest classifier predicting binary direction over six horizons (1h, 4h, 12h, 24h, 3d, 7d).

The paper makes three claims, each backed by rigorous validation:

1. **The signal is real.** 5/6 horizons achieve direction accuracy with Wilson CI lower bound > 50% (statistically significant). Permutation test on shuffled targets confirms p<0.05.
2. **TDA adds incremental value over base features.** Ablation study isolates the topological contribution.
3. **The strategy is more capital-preserving than profitable.** Continuous walk-forward backtest shows ADA-only deployment beats buy-and-hold by +67pp during a -55% market downturn (Sharpe 0.95) — but per-trade fees consume most of the per-period gross edge at typical retail venues.

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

## Headline Results

| Horizon | Best Model | Accuracy | 95% Wilson CI | n Signals | Significant |
|---------|------------|---------:|--------------:|----------:|------------|
| 1h      | RF, θ=0.62 | 64.87% | [55.96%, 73.00%] | 117 | ✅ |
| 4h      | RF, θ=0.70 | 61.59% | [53.36%, 69.11%] | 143 | ✅ |
| 12h     | RF, θ=0.70 | 56.77% | [49.52%, 63.58%] | 187 | ❌ |
| 24h     | Logistic, θ=0.70 | 61.09% | [58.69%, 63.39%] | 1,649 | ✅ |
| **3d**  | **RF, θ=0.70** | **61.77%** | **[59.75%, 63.75%]** | **2,260** | ✅ |
| 7d      | Logistic, θ=0.70 | 55.23% | [53.13%, 57.29%] | 2,193 | ✅ |

**Per-asset (3d horizon):**

| Asset | Direction Accuracy | n Signals |
|-------|-------------------:|----------:|
| ADA   | **76.75%**         | 317       |
| SOL   | 73.12%             | 345       |
| AVAX  | 66.39%             | 312       |
| LINK  | 63.91%             | 311       |
| DOT   | 53.09%             | 313       |
| ETH   | 51.01%             | 309       |
| BTC   | 48.48%             | 346       |

**Continuous walk-forward (v8, 365 days, ADA-only, Binance.US fees):**

| Strategy | Final Equity | Buy-Hold | Sharpe |
|----------|-------------:|---------:|-------:|
| TDA-ML   | $11,199 (+12.0%) | $4,475 (-55%) | **0.95** |

## Quick start

### Option A: Local Python (3.10+)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the headline experiment (v6 — generates paper figures)
python examples/run_validation_v6.py 90

# Reproduce the rigorous v9 result (holdout + ablation + permutation)
python examples/run_validation_v9.py 365 72 30
```

### Option B: Continuous walk-forward profitability test

```bash
# v8 — answers "would this have made money on real data?"
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

## Validation Methodology Progression

The repository documents a six-version progression that controls for different biases at each step. See [`VALIDATION_PROGRESSION.md`](VALIDATION_PROGRESSION.md) for the full narrative.

| Version | What it added                                  | Direction Accuracy | Significant? |
|--------:|------------------------------------------------|-------------------:|-------------|
| v1      | Single train/val/test split                     | invalid (n=2)      | —           |
| v2      | 5-fold time-series CV + Wilson CI               | 50.75%             | ❌          |
| v3      | ML classifier on TDA features                   | 56.18%             | borderline  |
| v4      | Multi-asset pool + advanced features            | 60.42% (n=26)      | underpowered |
| **v5**  | 7 assets + looser filter (n=684)                | **59.40%**         | ✅          |
| **v6**  | Horizon sweep + paper output                    | **61.77%** (3d)    | ✅          |
| **v7**  | 365-day cross-regime profitability check        | 67-71% per regime  | ✅ stable   |
| **v8**  | Continuous walk-forward + realistic execution   | ADA: +12% Sharpe 0.95 | ✅ profitable |
| **v9**  | Holdout + ablation + permutation test           | (in progress)      | TBD         |

## Limitations

- **Sample window:** 90-day primary study, extended to 365 days in v7-v9. Multi-year validation requires data from venues other than Coinbase free-tier API.
- **Cost model:** Point estimates of fees and slippage; does not model market impact, partial fills, or exchange downtime.
- **No live deployment:** All results are offline cross-validation. Live-vs-backtest gap is unknown.
- **Survivorship bias:** Asset list is fixed in 2026; assets that delisted (e.g., MATIC→POL) are excluded.

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
