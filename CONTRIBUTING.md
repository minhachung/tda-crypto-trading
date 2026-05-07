# Contributing

Thanks for your interest in TDA-Crypto-Trading. This document covers how to set up a dev environment, the conventions we follow, and how to extend the project.

## Setting Up

```bash
git clone https://github.com/minhachung/tda-crypto-trading.git
cd tda-crypto-trading

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .   # editable install
pip install pytest pytest-cov ruff black
```

Run the test suite to confirm everything works:

```bash
pytest tests/ -v
```

Run a quick validation smoke test:

```bash
python examples/run_validation_v6.py 30   # 30 days, ~3 minutes
```

## Coding Standards

- **Python 3.10+** — type hints encouraged but not required for plotting/glue code
- **Formatting:** `black src/ examples/ tests/`
- **Linting:** `ruff check src/ examples/ tests/`
- **Imports:** absolute (`from src.persistent_homology import ...`), no `from src import *`
- **Docstrings:** Numpy-style for public functions; one-liners are fine for internals

CI runs `ruff check` and `pytest` on every push. Failing either blocks merge.

## Testing

- Place new tests in `tests/`
- Mirror the source structure (`tests/test_<module>.py`)
- Use `tmp_path` fixture for any test that writes files
- Mark slow tests with `@pytest.mark.slow` so quick CI runs can skip them

```python
# tests/test_features.py
import pytest

@pytest.mark.slow
def test_full_pipeline_on_real_data():
    ...
```

## Adding New Components

### A new TDA summary statistic

1. Open `src/persistent_homology.py`
2. Add a function in the `PersistentHomologyAnalyzer.extract_features` method, prefixed with `H0_` or `H1_`
3. Add a unit test in `tests/test_persistent_homology.py` that constructs a known point cloud (e.g., points on a circle) and asserts the statistic takes the expected value
4. Re-run `examples/run_validation_v9.py` and update the ablation table in `results/RESULTS.md` if the new feature changes the numbers

### A new classifier

1. Open `src/ml_signals.py`
2. Add a branch in the `MLSignalGenerator._make_pipeline()` method
3. Register the classifier name in the `model_type` parameter docstring
4. Add a fixture in `tests/test_ml_signals.py` that fits the classifier on synthetic data and asserts non-trivial accuracy
5. Run a grid search to confirm the new classifier doesn't change the headline finding

### A new asset

Adding an asset to the multi-asset pipeline:

1. Add the symbol mapping to `SYMBOL_MAP` in `src/binance_data.py` (e.g., `'NEW': 'NEW-USD'`)
2. Verify Coinbase has historical hourly data for that symbol
3. Add the symbol to the default `symbols` list in `examples/run_validation_v6.py` and `examples/run_validation_v7.py`
4. Re-run validation and check whether the new asset is predictable (per-asset table)

### A new validation experiment

1. Create a new file `examples/run_validation_vN.py`
2. Import from `src/multi_asset_pipeline.py` and `src/validation_v2.py` rather than reinventing them
3. Save outputs to `results/V<N>_*.md` and `results/figures/v<N>_*.{png,pdf}`
4. Update `VALIDATION_PROGRESSION.md` with what the new version controls for vs prior versions

## Pull Request Process

1. Branch off `main` (no `develop` branch — we use trunk-based development)
2. Use a descriptive branch name: `feature/h2-persistence`, `fix/coinbase-rate-limit`, `docs/clarify-fees`
3. Open a PR with the template filled in
4. CI must be green before review
5. One reviewer approval required for merge
6. We squash-merge to keep history clean

## What We Don't Want

- Refactors that change behaviour without being explicitly motivated
- New dependencies for things that can be done with the existing stack (e.g., adding `polars` when `pandas` works)
- Speculative or untested optimizations — benchmark first
- "Improving" the validation methodology without re-running and updating the results files

## Questions?

Open a discussion or issue. The maintainer is reachable via GitHub.
