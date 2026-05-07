"""Test that time-series cross-validation produces non-leaking, ordered splits.

Also covers the n_signals contract for v9's evaluate_kfold, which v10's
signal-weighted accuracy depends on: every (fold, symbol) row in the
returned DataFrame must have a non-negative integer n_signals value
bounded by n_test.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.validation_v2 import time_series_kfold


def test_kfold_indices_strictly_increasing():
    """Test indices in fold k must all be > training indices."""
    folds = time_series_kfold(n_samples=100, n_splits=5)
    for i, (train_idx, test_idx) in enumerate(folds):
        max_train = train_idx.max()
        min_test = test_idx.min()
        assert min_test > max_train, \
            f"Fold {i}: test index {min_test} <= max train {max_train}"


def test_kfold_no_overlap():
    """Train and test sets must be disjoint within each fold."""
    folds = time_series_kfold(n_samples=200, n_splits=5)
    for i, (train_idx, test_idx) in enumerate(folds):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Fold {i} has {len(overlap)} overlapping indices"


def test_kfold_grows_train_window():
    """Each successive fold should have a larger training window (expanding window CV)."""
    folds = time_series_kfold(n_samples=200, n_splits=5)
    train_lens = [len(train_idx) for train_idx, _ in folds]
    for i in range(1, len(train_lens)):
        assert train_lens[i] >= train_lens[i - 1], \
            f"Fold {i} training set ({train_lens[i]}) smaller than fold {i-1} ({train_lens[i-1]})"


def test_kfold_returns_n_splits():
    """We should get exactly n_splits folds (or fewer if data is too small)."""
    folds = time_series_kfold(n_samples=200, n_splits=5)
    assert len(folds) == 5


def test_kfold_too_few_samples():
    """Tiny n_samples shouldn't crash, just return fewer folds or smaller folds."""
    folds = time_series_kfold(n_samples=30, n_splits=5)
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_kfold_no_shuffle():
    """Folds must respect time order — no random shuffle."""
    folds = time_series_kfold(n_samples=100, n_splits=5)
    for train_idx, test_idx in folds:
        assert list(train_idx) == sorted(train_idx)
        assert list(test_idx) == sorted(test_idx)


# ============================================================
# v9 evaluate_kfold — n_signals column contract
#
# v10's signal-weighted accuracy aggregates fold/asset rows by:
#   weighted = (acc * n_signals).sum() / n_signals.sum()
# That formula is only correct if every row's n_signals is a
# non-negative integer and is bounded by the test-set size for that
# row. These tests pin that contract.
# ============================================================

def _make_pool(n_per_symbol=120, symbols=('AAA', 'BBB'), seed=0):
    """Synthetic multi-asset pool with the columns evaluate_kfold needs:
    timestamp, symbol, close, target, plus a handful of ML_FEATURE_SET
    feature columns (random normals)."""
    from src.advanced_features import ML_FEATURE_SET
    rng = np.random.RandomState(seed)
    pieces = []
    for s_idx, sym in enumerate(symbols):
        rows = {
            'timestamp': pd.date_range('2024-01-01',
                                         periods=n_per_symbol, freq='h'),
            'symbol': sym,
            'close': 100.0 + np.cumsum(rng.randn(n_per_symbol) * 0.5)
                       + s_idx * 50,
            'target': rng.randint(0, 2, size=n_per_symbol).astype(float),
        }
        for col in ML_FEATURE_SET:
            rows[col] = rng.randn(n_per_symbol)
        pieces.append(pd.DataFrame(rows))
    return pd.concat(pieces, ignore_index=True)


def test_evaluate_kfold_emits_n_signals_column():
    """v9's evaluate_kfold must populate an ``n_signals`` column on its
    returned DataFrame. v10's weighted_accuracy reads this column."""
    from examples.run_validation_v9 import evaluate_kfold
    from src.advanced_features import ML_FEATURE_SET

    df = _make_pool(n_per_symbol=120)
    feat_cols = ML_FEATURE_SET[:5]
    fold_df = evaluate_kfold(df, feat_cols, model_type='logistic',
                              prob_threshold=0.55, n_splits=3,
                              regime_filter=None)

    assert len(fold_df) > 0, \
        "evaluate_kfold returned no fold rows on a 240-sample pool"
    assert 'n_signals' in fold_df.columns, \
        "evaluate_kfold did not emit an n_signals column — v10's " \
        "weighted_accuracy contract is broken."


def test_evaluate_kfold_n_signals_is_nonneg_integer():
    """n_signals counts BUY+SELL signals, so every value must be a
    non-negative integer."""
    from examples.run_validation_v9 import evaluate_kfold
    from src.advanced_features import ML_FEATURE_SET

    df = _make_pool(n_per_symbol=120)
    feat_cols = ML_FEATURE_SET[:5]
    fold_df = evaluate_kfold(df, feat_cols, model_type='logistic',
                              prob_threshold=0.55, n_splits=3,
                              regime_filter=None)

    for _, row in fold_df.iterrows():
        n_sig = row['n_signals']
        assert n_sig >= 0, f"Negative n_signals value: {n_sig}"
        # n_signals is a count; either int dtype or a float that is
        # equal to its int cast.
        assert int(n_sig) == n_sig, \
            f"n_signals value {n_sig} is not an integer"


def test_evaluate_kfold_n_signals_bounded_by_n_test():
    """A signal fires per test sample, so n_signals can never exceed
    n_test for any (fold, symbol) row."""
    from examples.run_validation_v9 import evaluate_kfold
    from src.advanced_features import ML_FEATURE_SET

    df = _make_pool(n_per_symbol=120)
    feat_cols = ML_FEATURE_SET[:5]
    fold_df = evaluate_kfold(df, feat_cols, model_type='logistic',
                              prob_threshold=0.55, n_splits=3,
                              regime_filter=None)

    assert 'n_test' in fold_df.columns, \
        "evaluate_kfold must emit n_test alongside n_signals"
    for _, row in fold_df.iterrows():
        assert row['n_signals'] <= row['n_test'], (
            f"n_signals ({row['n_signals']}) exceeds n_test "
            f"({row['n_test']}) for fold={row['fold']}, "
            f"symbol={row['symbol']}"
        )


def test_evaluate_kfold_higher_threshold_yields_no_more_signals():
    """A stricter probability threshold can only fire fewer (or equal)
    signals than a looser threshold — n_signals must be monotonic in
    -threshold for the same data + model."""
    from examples.run_validation_v9 import evaluate_kfold
    from src.advanced_features import ML_FEATURE_SET

    df = _make_pool(n_per_symbol=120)
    feat_cols = ML_FEATURE_SET[:5]

    fold_loose = evaluate_kfold(df, feat_cols, model_type='logistic',
                                 prob_threshold=0.55, n_splits=3,
                                 regime_filter=None)
    fold_strict = evaluate_kfold(df, feat_cols, model_type='logistic',
                                  prob_threshold=0.75, n_splits=3,
                                  regime_filter=None)

    # Aggregate signal counts must be (loose) >= (strict).
    n_loose = int(fold_loose['n_signals'].sum())
    n_strict = int(fold_strict['n_signals'].sum())
    assert n_loose >= n_strict, (
        f"Stricter threshold (0.75) produced MORE signals "
        f"({n_strict}) than the looser threshold (0.55, {n_loose}). "
        f"This breaks the n_signals contract that v10 relies on for "
        f"signal-weighted accuracy."
    )
