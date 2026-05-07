"""Test that time-series cross-validation produces non-leaking, ordered splits."""

import numpy as np
import pytest

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
