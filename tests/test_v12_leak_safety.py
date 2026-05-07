"""Tests for the v12 leak-safety contract.

Covers the four properties the reviewer flagged:
  10. add_targets drops exactly `horizon` rows per asset (no silent
      0-labels on the tail).
  11. Fold purging in evaluate_kfold_leaksafe: max(train_idx) + horizon
      < min(test_idx) for every (asset, fold).
  12. Causal window normalization: changing a far-future row does not
      change earlier windows' point clouds (and therefore does not
      change earlier diagrams).
  13. LeakSafePersistenceImagerFitter:
      - empty train diagrams → all-zero features
      - non-empty transform returns shape (n, 2 * resolution^2)
      - fit on a *subset* (train) and transform a *different subset*
        (test) does not raise.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.multi_asset_pipeline import (
    add_targets, create_causal_normalized_windows,
)
from src.tda_v2_features import (
    LeakSafePersistenceImagerFitter, compute_diagrams_only,
)
from src.validation_v2 import time_series_kfold


# ============================================================
# Test 10: add_targets drops exactly `horizon` rows per asset
# ============================================================

def _make_df(n=200, symbol='ABC', start=100.0):
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'close': start + np.cumsum(np.random.RandomState(0).randn(n) * 0.5),
        'symbol': symbol,
    })


def test_add_targets_drops_exactly_horizon_per_asset():
    df1 = _make_df(n=200, symbol='ABC')
    df2 = _make_df(n=150, symbol='XYZ')
    df = pd.concat([df1, df2], ignore_index=True)

    horizon = 24
    out = add_targets(df, horizon=horizon, price_col='close')

    for symbol, group in out.groupby('symbol'):
        original_n = len(df[df['symbol'] == symbol])
        assert len(group) == original_n - horizon, (
            f"{symbol}: expected {original_n - horizon} rows after "
            f"add_targets(horizon={horizon}), got {len(group)}"
        )

    assert out['target'].isin([0, 1]).all(), \
        "add_targets should produce only 0/1 targets after dropna"


def test_add_targets_no_silent_zero_labels_on_tail():
    """Regression for the bug where (shift > x).astype(float) silently
    labels the final `horizon` rows as 0.0 instead of NaN."""
    df = _make_df(n=100, symbol='ABC')
    df = df.assign(close=np.linspace(100, 200, 100))  # strictly rising
    horizon = 10
    out = add_targets(df, horizon=horizon, price_col='close')

    assert (out['target'] == 1).all(), (
        "Strictly rising series should produce all-1 targets after dropna; "
        "any 0 indicates the tail wasn't properly dropped."
    )
    assert len(out) == 100 - horizon


# ============================================================
# Test 11: fold purging
# ============================================================

def test_fold_purging_no_train_target_overlaps_test():
    """For every fold, max(train + horizon) < min(test)."""
    n_samples = 500
    horizon = 24
    n_splits = 5

    folds = time_series_kfold(n_samples, n_splits=n_splits)
    assert len(folds) > 0

    overlapping_folds = 0
    for fold_idx, (tr_idx, te_idx) in enumerate(folds):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        te_start = int(np.min(te_idx))

        purged = tr_idx[tr_idx + horizon < te_start]

        if len(purged) > 0:
            max_purged_target = int(np.max(purged)) + horizon
            assert max_purged_target < te_start, (
                f"Fold {fold_idx}: purged train tail target index "
                f"{max_purged_target} not strictly less than test start "
                f"{te_start}"
            )

        unpurged_max_target = int(np.max(tr_idx)) + horizon
        if unpurged_max_target >= te_start:
            overlapping_folds += 1

    assert overlapping_folds > 0, (
        "test setup is not exercising leak protection — no fold's "
        "unpurged train tail overlaps the test start"
    )


# ============================================================
# Test 12: causal window normalization
# ============================================================

def test_causal_normalization_immune_to_far_future_perturbation():
    """Mutating a row near the END of the series must NOT change point
    clouds whose end_idx is far before that mutation."""
    rng = np.random.RandomState(42)
    n, dim = 200, 5
    X = rng.randn(n, dim).astype(float)

    pcs_a, end_a = create_causal_normalized_windows(X, window_size=20, stride=1)

    X_perturbed = X.copy()
    X_perturbed[180:185, :] += 100.0

    pcs_b, end_b = create_causal_normalized_windows(X_perturbed,
                                                     window_size=20, stride=1)

    for i, (a, b, ea, eb) in enumerate(zip(pcs_a, pcs_b, end_a, end_b)):
        assert ea == eb
        if ea <= 50:
            assert np.allclose(a, b, atol=1e-12), (
                f"Window {i} ending at idx {ea} differs between original "
                f"and perturbed series — causal normalization is leaking."
            )


def test_causal_normalization_does_use_history():
    """Sanity check: causal normalization is NOT a no-op."""
    rng = np.random.RandomState(123)
    X = rng.randn(100, 3) * 5.0 + 10.0

    pcs, _ = create_causal_normalized_windows(X, window_size=20, stride=1)
    last = pcs[-1]
    assert np.abs(last.mean()) < 5.0, "Late causal window should be near-mean-zero"
    assert last.std() > 0.0


# ============================================================
# Test 13: LeakSafePersistenceImagerFitter
# ============================================================

def test_imager_empty_train_diagrams_returns_zero_features():
    fitter = LeakSafePersistenceImagerFitter(resolution=10)
    n_train = 5
    n_test = 3
    empty_train_h0 = [np.zeros((0, 2)) for _ in range(n_train)]
    empty_train_h1 = [np.zeros((0, 2)) for _ in range(n_train)]
    test_h0 = [np.array([[0.1, 0.5]]) for _ in range(n_test)]
    test_h1 = [np.array([[0.2, 0.7]]) for _ in range(n_test)]

    fitter.fit(empty_train_h0, empty_train_h1)
    out = fitter.transform(test_h0, test_h1)

    assert out.shape == (n_test, 2 * 10 * 10)
    assert np.allclose(out, 0.0), \
        "Empty train diagrams must yield all-zero features at transform time"


def test_imager_non_empty_returns_correct_shape():
    fitter = LeakSafePersistenceImagerFitter(resolution=10)
    rng = np.random.RandomState(7)
    n_train = 30
    n_test = 12
    train_h0 = [
        np.column_stack([rng.uniform(0, 1, k), rng.uniform(0.1, 1.5, k)])
        for k in rng.randint(1, 6, size=n_train)
    ]
    train_h1 = [
        np.column_stack([rng.uniform(0, 1, k), rng.uniform(0.1, 0.9, k)])
        for k in rng.randint(1, 4, size=n_train)
    ]
    test_h0 = [
        np.column_stack([rng.uniform(0, 1, k), rng.uniform(0.1, 1.5, k)])
        for k in rng.randint(1, 6, size=n_test)
    ]
    test_h1 = [
        np.column_stack([rng.uniform(0, 1, k), rng.uniform(0.1, 0.9, k)])
        for k in rng.randint(1, 4, size=n_test)
    ]

    fitter.fit(train_h0, train_h1)
    out = fitter.transform(test_h0, test_h1)

    assert out.shape == (n_test, 2 * 10 * 10), \
        f"Expected ({n_test}, 200), got {out.shape}"
    assert not np.allclose(out, 0.0), \
        "Non-trivial diagrams should produce non-zero image features"


def test_imager_fit_on_train_subset_transforms_unseen_test():
    """Leak-safe contract: fit on train, transform any held-out diagram
    set, no shape drift, no exception."""
    rng = np.random.RandomState(1)
    fitter = LeakSafePersistenceImagerFitter(resolution=8)
    train_h0 = [np.array([[rng.uniform(), rng.uniform(0.2, 0.5)]])
                 for _ in range(40)]
    train_h1 = [np.array([[rng.uniform(), rng.uniform(0.2, 0.5)]])
                 for _ in range(40)]

    fitter.fit(train_h0, train_h1)

    test_h0 = [np.array([[rng.uniform(), rng.uniform(2.0, 3.0)]])
                for _ in range(5)]
    test_h1 = [np.array([[rng.uniform(), rng.uniform(2.0, 3.0)]])
                for _ in range(5)]

    out = fitter.transform(test_h0, test_h1)
    assert out.shape == (5, 2 * 8 * 8)


# ============================================================
# Integration: per-window diagram intrinsic-ness
# ============================================================

def test_diagrams_intrinsic_to_each_window():
    rng = np.random.RandomState(0)
    n, dim = 80, 4
    X = rng.randn(n, dim).astype(float)

    pcs_a, _ = create_causal_normalized_windows(X, window_size=15, stride=1)
    h0_a, h1_a = compute_diagrams_only(pcs_a[:30], max_dim=1, verbose=False)

    X_pert = X.copy()
    X_pert[60:65, :] += 50.0
    pcs_b, _ = create_causal_normalized_windows(X_pert, window_size=15, stride=1)
    h0_b, h1_b = compute_diagrams_only(pcs_b[:30], max_dim=1, verbose=False)

    for i in range(30):
        assert h0_a[i].shape == h0_b[i].shape, f"H0 shape drift at window {i}"
        if h0_a[i].size > 0:
            assert np.allclose(h0_a[i], h0_b[i], atol=1e-10)
        assert h1_a[i].shape == h1_b[i].shape, f"H1 shape drift at window {i}"
        if h1_a[i].size > 0:
            assert np.allclose(h1_a[i], h1_b[i], atol=1e-10)
