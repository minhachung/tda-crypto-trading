"""Methodology tests for v10's local helpers.

Covers:
  - weighted_accuracy: signal-weighted vs unweighted accuracy
  - block_shuffle_targets_preserve_remainder: row-count preservation,
    partial-final-block handling, target permutation, no NaN injection.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from examples.run_validation_v10 import (
    weighted_accuracy,
    block_shuffle_targets_preserve_remainder,
    _validate_perm_results,
)


# ============================================================
# weighted_accuracy
# ============================================================

def test_weighted_accuracy_equal_weights_matches_row_mean():
    """When every row has the same n_signals, weighted == unweighted."""
    fold_df = pd.DataFrame({
        'direction_accuracy': [0.5, 0.6, 0.7, 0.8],
        'n_signals': [10, 10, 10, 10],
    })
    weighted, n = weighted_accuracy(fold_df)
    assert n == 40
    assert weighted == sum([0.5, 0.6, 0.7, 0.8]) / 4


def test_weighted_accuracy_unequal_weights_diverges_from_row_mean():
    """When n_signals varies, weighted accuracy must reflect the
    signal-weighted contribution. This is the core fix: the previous
    v10 used the unweighted row mean, which over-counted low-signal
    cells."""
    fold_df = pd.DataFrame({
        'direction_accuracy': [0.50, 0.80],
        'n_signals': [1000, 10],
    })
    weighted, n = weighted_accuracy(fold_df)
    assert n == 1010
    expected = (0.50 * 1000 + 0.80 * 10) / 1010
    assert abs(weighted - expected) < 1e-12

    row_mean = fold_df['direction_accuracy'].mean()
    assert abs(weighted - row_mean) > 0.1, (
        "Test fixture is degenerate: weighted and unweighted produce "
        "the same value, so the test does not exercise the fix."
    )


def test_weighted_accuracy_zero_signals_returns_chance():
    """When no signals fired, weighted_accuracy returns the neutral
    chance value (0.5, 0) — matches the grid-search guard that requires
    n_signals >= 30."""
    fold_df = pd.DataFrame({
        'direction_accuracy': [0.0, 0.0],
        'n_signals': [0, 0],
    })
    weighted, n = weighted_accuracy(fold_df)
    assert n == 0
    assert weighted == 0.5


# ============================================================
# block_shuffle_targets_preserve_remainder
# ============================================================

def _make_target_df(n=200, symbol='ABC', start='2024-01-01'):
    """Build a synthetic df with the columns block_shuffle relies on."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        'timestamp': pd.date_range(start, periods=n, freq='h'),
        'symbol': symbol,
        # Use range so we can detect any reordering deterministically.
        'target': np.arange(n, dtype=float),
        'close': 100.0 + np.cumsum(rng.randn(n) * 0.5),
    })


def test_block_shuffle_preserves_total_row_count_with_partial_block():
    """200 rows / 168-hour blocks → blocks of length [168, 32]. The
    final 32-row partial block must NOT be dropped (this is the v9 bug
    v10 corrects)."""
    df = _make_target_df(n=200)
    out = block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=7)
    assert len(out) == len(df) == 200, (
        f"Row drift: input had 200 rows, output has {len(out)}. "
        "block_shuffle_targets_preserve_remainder must preserve every row."
    )


def test_block_shuffle_preserves_target_set_with_partial_block():
    """The output target column must be a permutation of the input
    target column (no values lost, no values invented)."""
    df = _make_target_df(n=200)
    out = block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=7)
    assert sorted(out['target'].tolist()) == sorted(df['target'].tolist()), (
        "Output targets are not a permutation of input targets; "
        "rows were lost or invented during the block shuffle."
    )


def test_block_shuffle_changes_target_order_for_nontrivial_input():
    """For a 200-row series with two blocks, at least one of the two
    canonical orderings is the identity. Across multiple seeds the
    target ordering must change at least once."""
    df = _make_target_df(n=200)
    seen_change = False
    for seed in range(10):
        out = block_shuffle_targets_preserve_remainder(df, block_hours=168,
                                                          seed=seed)
        if not np.array_equal(out['target'].values, df['target'].values):
            seen_change = True
            break
    assert seen_change, (
        "Across 10 seeds, block_shuffle never changed target order — "
        "the shuffle is a no-op."
    )


def test_block_shuffle_no_nan_in_output_target():
    """No row should end up with a NaN target after shuffling. (v9's
    implementation could leave un-filled tail rows as NaN if the
    final partial block fell off; v10's must not.)"""
    df = _make_target_df(n=275)  # not a multiple of 168 → partial block
    out = block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=1)
    assert out['target'].isna().sum() == 0, (
        "block_shuffle introduced NaN targets — likely the partial final "
        "block was dropped instead of preserved."
    )


def test_block_shuffle_per_symbol_independent():
    """When two symbols have different row counts, each symbol's targets
    must remain a permutation of its own original targets — the shuffle
    is per-symbol, not pooled."""
    df_a = _make_target_df(n=200, symbol='AAA',
                            start='2024-01-01').assign(
        target=lambda d: d['target'] + 1000  # AAA targets are 1000..1199
    )
    df_b = _make_target_df(n=150, symbol='BBB',
                            start='2024-01-01').assign(
        target=lambda d: d['target'] + 5000  # BBB targets are 5000..5149
    )
    df = pd.concat([df_a, df_b], ignore_index=True)

    out = block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=11)

    aaa_out = out[out['symbol'] == 'AAA']['target'].values
    bbb_out = out[out['symbol'] == 'BBB']['target'].values

    assert sorted(aaa_out.tolist()) == sorted((np.arange(200) + 1000).tolist()), (
        "AAA targets contain values from another symbol — shuffle leaked "
        "across symbol groups."
    )
    assert sorted(bbb_out.tolist()) == sorted((np.arange(150) + 5000).tolist()), (
        "BBB targets contain values from another symbol — shuffle leaked "
        "across symbol groups."
    )

    assert len(aaa_out) == 200
    assert len(bbb_out) == 150


def test_block_shuffle_only_permutes_target_column():
    """Features, prices, and timestamps must be unchanged by the shuffle."""
    df = _make_target_df(n=200)
    original_close = df['close'].copy()
    original_timestamp = df['timestamp'].copy()
    out = block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=3)
    assert (out['close'].values == original_close.values).all(), \
        "close column was modified by block_shuffle"
    assert (out['timestamp'].values == original_timestamp.values).all(), \
        "timestamp column was modified by block_shuffle"


# ============================================================
# Empty-permutation handling
# ============================================================

def test_validate_perm_results_raises_on_empty_array():
    """An empty permutation array would silently produce p-value = 1.0
    via (0+1)/(0+1) and crash on .max(); the validator must turn that
    into a clear RuntimeError before any downstream report code runs."""
    with pytest.raises(RuntimeError, match=r"produced 0 valid"):
        _validate_perm_results(np.array([]), "Test phase", 100)


def test_validate_perm_results_raises_on_empty_list():
    """A bare empty list (rather than an empty np.array) must also be
    caught — both np.sum and len() handle it identically, and we want
    callers that pass [] to fail with the same readable error."""
    with pytest.raises(RuntimeError, match=r"Test phase"):
        _validate_perm_results([], "Test phase", 50)


def test_validate_perm_results_passes_on_nonempty():
    """A non-empty permutation array must pass through silently."""
    _validate_perm_results(np.array([0.5, 0.6, 0.7]), "Test phase", 3)
    # The single-element edge case is also valid — at least one
    # permutation succeeded, so the report can compute a p-value
    # (even if a B=1 p-value is not particularly informative).
    _validate_perm_results(np.array([0.5]), "Test phase", 1)


def test_validate_perm_results_message_includes_remediation_hints():
    """The RuntimeError message must include actionable hints about
    why permutations may have produced zero samples."""
    try:
        _validate_perm_results(np.array([]), "Full-grid permutation", 100)
    except RuntimeError as exc:
        msg = str(exc)
        assert "n_signals" in msg or "thresholds" in msg, (
            f"Error message lacks remediation hints. Got: {msg}"
        )
        assert "0 valid" in msg
        assert "100 attempted" in msg
    else:
        pytest.fail("RuntimeError was not raised")


# ============================================================
# Integration test: run_grid_permutation uses weighted accuracy
# ============================================================

def test_run_grid_permutation_uses_signal_weighted_accuracy(monkeypatch):
    """Drives examples.run_validation_v10.run_grid_permutation with
    evaluate_kfold stubbed to return a fixed fold_df where
    signal-weighted accuracy != unweighted row mean. The recorded
    best-of-grid accuracy must equal the WEIGHTED value, not the row
    mean.

    Will fail if a regression replaces weighted_accuracy() with
    fold_df['direction_accuracy'].mean() in the grid permutation
    scoring path."""
    import examples.run_validation_v10 as v10mod

    # Construct a fold_df where weighted vs unweighted mean diverge.
    fixed_fold_df = pd.DataFrame({
        'fold': [0, 0],
        'symbol': ['AAA', 'BBB'],
        'n_signals': [1000, 10],
        'direction_accuracy': [0.50, 0.80],
        'auc': [0.50, 0.50],
        'tda_return_pct': [0.0, 0.0],
        'tda_sharpe': [0.0, 0.0],
        'tda_n_trades': [0, 0],
        'buy_hold_return_pct': [0.0, 0.0],
        'outperformed_bh': [False, False],
    })
    weighted_expected = (0.50 * 1000 + 0.80 * 10) / 1010  # ~0.5030
    row_mean_expected = (0.50 + 0.80) / 2                  # 0.65
    # Confound check: the fixture must distinguish the two.
    assert abs(weighted_expected - row_mean_expected) > 0.1

    # Stub evaluate_kfold to return the fixed fold_df regardless of
    # the model_type / threshold / regime_filter combination.
    def stub_evaluate_kfold(*args, **kwargs):
        return fixed_fold_df.copy()

    monkeypatch.setattr(v10mod, 'evaluate_kfold', stub_evaluate_kfold)

    # Stub the block-shuffle and add_targets helpers so the test does
    # not need a real DataFrame structure (timestamp column, target
    # shifting logic, etc.).
    def stub_shuffle(df, block_hours=168, seed=0):
        return df.copy()

    monkeypatch.setattr(v10mod, 'block_shuffle_targets_preserve_remainder',
                          stub_shuffle)

    def stub_add_targets(df, horizon=72):
        return df.copy()

    monkeypatch.setattr(v10mod, 'add_targets', stub_add_targets)

    # Minimal placeholder df; stubs ignore its content.
    fake_df = pd.DataFrame({
        'symbol': ['AAA'] * 50 + ['BBB'] * 50,
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'target': np.zeros(100),
        'feat_1': np.zeros(100),
        'close': 100.0 + np.arange(100) * 0.1,
    })

    n_iter = 2
    accs = v10mod.run_grid_permutation(
        fake_df, ['feat_1'], horizon=72, n_iter=n_iter, n_splits=2,
    )

    assert len(accs) == n_iter, (
        f"Expected {n_iter} grid-permutation results, got {len(accs)}"
    )
    for i, acc in enumerate(accs):
        assert abs(acc - weighted_expected) < 1e-9, (
            f"Permutation {i}: recorded accuracy {acc:.6f} does not match "
            f"the signal-weighted expectation {weighted_expected:.6f}. "
            f"Likely regression: run_grid_permutation switched back to "
            f"the unweighted row mean (={row_mean_expected:.6f})."
        )
        assert abs(acc - row_mean_expected) > 0.05, (
            f"Permutation {i}: recorded accuracy {acc:.6f} matches the "
            f"unweighted row mean {row_mean_expected:.6f} to within 5pp. "
            f"This indicates the test stub did not exercise the weighted "
            f"path."
        )
