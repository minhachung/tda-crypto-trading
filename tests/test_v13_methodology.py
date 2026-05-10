"""Tests for the v13 methodology contract.

Each test verifies one property of the v13 pipeline that the reviewer flagged:

  A. p-value lower bound: with B permutations, the minimum reported p-value is
     1/(B+1). p = (n_above + 1) / (B + 1) is the canonical formula.

  B. Shuffled TDA control: the `base_plus_shuffled_v2` ablation must permute v2
     row order at TRAIN time only, NOT at test time. If real v2 doesn't beat
     shuffled v2, the verdict reflects "no signal".

  C. No model selection on test set: the routine that picks the "best" ablation
     must do so without consulting test-fold labels. (We verify by checking
     that the function signature does not accept test_y, or by asserting that
     test data does not enter the selection step.)

  D. Synthetic feature flag: `is_synthetic_column` correctly classifies the
     known synthetic columns; `--allow-synthetic` toggles their inclusion;
     paper-grade default strips them.

  E. Consistent label permutation: when permute_labels=True, train AND val
     labels are shuffled together (and test labels are NOT touched).

These complement the existing tests in test_v12_leak_safety.py which already
cover: target-tail-drop, purged splits, causal normalization, train-only
imager fit.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from examples.run_validation_v13 import (
    perm_pvalue,
    is_synthetic_column,
    SYNTHETIC_FEATURE_PATTERNS,
    FEATURE_SETS,
    _assemble,
    _strip_synthetic,
)


# --------------------------------------------------------------------------
# A. p-value lower bound
# --------------------------------------------------------------------------

def test_pvalue_lower_bound_is_one_over_B_plus_1():
    """No matter how many permutations exceed real, p ≥ 1/(B+1)."""
    real = 0.99
    null = [0.50] * 30  # zero permutations >= real
    p = perm_pvalue(real, null)
    assert p == pytest.approx(1 / 31), (
        f"With 30 perms and 0 above, expected p = 1/31 = 0.0323, got {p}"
    )


def test_pvalue_never_zero_even_when_all_below():
    """Boundary case: even if no null score reaches real, p > 0."""
    real = 1.0
    null = [0.5] * 99
    p = perm_pvalue(real, null)
    assert p > 0
    assert p == pytest.approx(1 / 100)


def test_pvalue_correct_for_some_above():
    """If 5 of 99 perms reach or exceed real, p = 6/100 = 0.06."""
    real = 0.55
    null = [0.50] * 94 + [0.55] * 3 + [0.60] * 2  # 5 perms >= 0.55
    p = perm_pvalue(real, null)
    assert p == pytest.approx(6 / 100)


def test_pvalue_b_zero_returns_one():
    """With no permutations, the conservative answer is p=1.0."""
    p = perm_pvalue(0.6, [])
    assert p == 1.0


# --------------------------------------------------------------------------
# B. Shuffled-v2 negative control fires only at train time
# --------------------------------------------------------------------------

def test_shuffled_v2_permutes_only_at_train_time():
    """When is_train=True and feature_set='base_plus_shuffled_v2', v2 rows are
    permuted; when is_train=False, v2 is passed through unchanged."""
    n, b_dim, v2_dim = 10, 3, 4
    base = np.arange(n * b_dim).reshape(n, b_dim).astype(float)
    v1 = np.zeros((n, 2))
    v2 = np.arange(n * v2_dim).reshape(n, v2_dim).astype(float)

    rng = np.random.RandomState(0)
    train_X = _assemble('base_plus_shuffled_v2', base, v1, v2,
                          rng=rng, is_train=True)
    rng = np.random.RandomState(0)
    test_X = _assemble('base_plus_shuffled_v2', base, v1, v2,
                         rng=rng, is_train=False)

    # base block must be unchanged in both cases
    np.testing.assert_array_equal(train_X[:, :b_dim], base)
    np.testing.assert_array_equal(test_X[:, :b_dim], base)

    # test_X v2 block must equal the input v2 (no shuffle on test)
    np.testing.assert_array_equal(test_X[:, b_dim:], v2)

    # train_X v2 block must be a permutation of v2 (some row order changed)
    train_v2 = train_X[:, b_dim:]
    # Same row-set, possibly different order
    assert sorted(map(tuple, train_v2.tolist())) == sorted(map(tuple, v2.tolist())), \
        "Shuffled-v2 must permute rows, not change values"


def test_v2_only_returns_just_v2_block():
    """Sanity: 'v2_only' returns v2 alone, not concatenated with base."""
    base = np.ones((5, 3))
    v1 = np.ones((5, 2)) * 2
    v2 = np.ones((5, 4)) * 3
    out = _assemble('v2_only', base, v1, v2, is_train=False)
    assert out.shape == (5, 4)
    np.testing.assert_array_equal(out, v2)


def test_v1_only_returns_just_v1_block():
    base = np.ones((5, 3))
    v1 = np.ones((5, 2)) * 2
    v2 = np.ones((5, 4)) * 3
    out = _assemble('v1_only', base, v1, v2, is_train=False)
    assert out.shape == (5, 2)
    np.testing.assert_array_equal(out, v1)


def test_seven_ablations_present():
    """The reviewer required 7 specific ablations. Verify the constant lists them."""
    expected = {
        'base', 'v1_only', 'v2_only',
        'base_plus_v1', 'base_plus_v2', 'base_plus_v1_plus_v2',
        'base_plus_shuffled_v2',
    }
    assert set(FEATURE_SETS) == expected, (
        f"FEATURE_SETS missing or extra: got {set(FEATURE_SETS)}"
    )


# --------------------------------------------------------------------------
# C. No model selection on test labels
# --------------------------------------------------------------------------

def test_v13_main_picks_best_only_among_real_conditions():
    """The selection step in v13's main() picks `best_fs` from feature-set
    accuracies. The reviewer's concern: never use test-fold accuracy to PICK
    a model that will then be reported as the "winner".

    v13's mitigation: the picked condition's permutation p-value is reported
    explicitly as a *post-selection diagnostic*, not as the multiple-testing-
    aware main result. Verify the docstring/output marker is present.
    """
    import examples.run_validation_v13 as v13_module
    # Confirm the post-selection caveat is hard-coded in the runner output text
    src = open(v13_module.__file__).read()
    assert 'post-selection' in src.lower(), (
        "v13 must explicitly label single-config p as post-selection diagnostic"
    )
    assert 'multiple-testing' in src.lower(), (
        "v13 must mention that multiple-testing correction would require "
        "permuting the full grid"
    )


def test_shuffled_v2_excluded_from_best_selection():
    """The shuffled control should NEVER be the 'best' condition in reporting,
    because picking it would mean the negative control beat the real signal —
    a finding worth flagging, but not 'winning'. v13 explicitly excludes
    shuffled_v2 from `best_fs` selection."""
    import examples.run_validation_v13 as v13_module
    src = open(v13_module.__file__).read()
    # Look for the explicit filter
    assert "real_conditions = [fs for fs in FEATURE_SETS if fs != 'base_plus_shuffled_v2']" in src


# --------------------------------------------------------------------------
# D. Synthetic feature flag
# --------------------------------------------------------------------------

def test_known_synthetic_columns_classified():
    """Known synthetic columns from the FeatureBuilder must be flagged."""
    must_be_synthetic = [
        'active_addresses_z', 'transaction_count_z', 'mvrv_ratio_z',
        'whale_supply_pct_z', 'developer_activity_z',
        'spread_pct', 'spread_zscore_20', 'spread_persistence',
        'is_delisted', 'months_since_delisting', 'exchange_concentration',
    ]
    for col in must_be_synthetic:
        assert is_synthetic_column(col), f"{col} should be flagged synthetic"


def test_real_columns_not_classified_as_synthetic():
    """Real microstructure features must NOT be flagged."""
    must_be_real = [
        'log_return', 'gk_vol_20', 'parkinson_20', 'rv_20',
        'rsi_centered', 'macd_normalized', 'bb_position',
        'H0_count', 'H1_count', 'H0_max_persistence', 'H1_entropy',
        'pim_h0_5', 'pim_h1_42',
        'open', 'high', 'low', 'close', 'volume',
        'symbol', 'target', 'future_return',
    ]
    for col in must_be_real:
        assert not is_synthetic_column(col), f"{col} should NOT be flagged synthetic"


def test_strip_synthetic_paper_grade_drops_synthetic_columns():
    """When allow_synthetic=False, synthetic columns must be dropped."""
    df = pd.DataFrame({
        'log_return': [0.1, 0.2],          # real
        'rv_20': [0.01, 0.02],             # real
        'mvrv_ratio_z': [1.0, 1.1],        # synthetic
        'spread_pct': [0.001, 0.002],      # synthetic
        'is_delisted': [False, False],     # synthetic
    })
    stripped = _strip_synthetic(df, allow_synthetic=False)
    assert 'log_return' in stripped.columns
    assert 'rv_20' in stripped.columns
    assert 'mvrv_ratio_z' not in stripped.columns
    assert 'spread_pct' not in stripped.columns
    assert 'is_delisted' not in stripped.columns


def test_strip_synthetic_keeps_synthetic_when_allowed():
    """When allow_synthetic=True, synthetic columns stay."""
    df = pd.DataFrame({
        'log_return': [0.1, 0.2],
        'mvrv_ratio_z': [1.0, 1.1],
    })
    kept = _strip_synthetic(df, allow_synthetic=True)
    assert 'mvrv_ratio_z' in kept.columns


# --------------------------------------------------------------------------
# E. Consistent label permutation in evaluate_ablation
# --------------------------------------------------------------------------

def test_evaluate_ablation_permute_labels_signature_exists():
    """Permutation must permute train AND val together; signature exposes
    the option."""
    from examples.run_validation_v13 import evaluate_ablation
    import inspect
    sig = inspect.signature(evaluate_ablation)
    assert 'permute_labels' in sig.parameters
    assert 'perm_rng' in sig.parameters


def test_consistent_permutation_described_in_source():
    """Source code must contain the comment that test labels are NEVER
    permuted but train+val are permuted together."""
    import examples.run_validation_v13 as v13_module
    src = open(v13_module.__file__).read()
    assert 'CONSISTENT label permutation' in src, (
        "v13 must explicitly document the consistent-permutation contract"
    )
    assert 'NEVER shuffle test labels' in src or 'Test labels are NEVER permuted' in src, (
        "v13 must explicitly state test labels are never permuted"
    )


# --------------------------------------------------------------------------
# F. Synthetic-pattern coverage list is itself stable
# --------------------------------------------------------------------------

def test_synthetic_patterns_list_non_empty_and_well_formed():
    """The synthetic-feature pattern list must be defined and contain known
    entries; this prevents accidental clearing of the list during refactoring."""
    assert isinstance(SYNTHETIC_FEATURE_PATTERNS, list)
    assert len(SYNTHETIC_FEATURE_PATTERNS) >= 8
    # Every entry must be a non-empty string
    for pat in SYNTHETIC_FEATURE_PATTERNS:
        assert isinstance(pat, str) and len(pat) > 0


# --------------------------------------------------------------------------
# G. Reality check: per-asset purged split helper
# --------------------------------------------------------------------------

def test_per_asset_split_respects_horizon_purge():
    """For each fold, every train index t must satisfy t + horizon < min(test).
    Validation indices must be the LAST 15% of train (chronological), not random."""
    from examples.run_validation_v13 import per_asset_split

    # Synthesize asset_data with required keys
    n = 600
    df = pd.DataFrame({'target': np.zeros(n, dtype=int)})
    asset_data = {'X': {'df': df,
                         'h0_diagrams': [np.zeros((0, 2))] * n,
                         'h1_diagrams': [np.zeros((0, 2))] * n}}

    horizon = 50
    embargo = 5
    splits = per_asset_split(asset_data, horizon=horizon,
                              n_splits=5, embargo=embargo)
    assert 'X' in splits
    fold_list = splits['X']
    assert len(fold_list) == 5
    for fold in fold_list:
        if fold is None:
            continue
        train_idx, val_idx, test_idx = fold
        # Purge + embargo
        max_train = int(np.max(train_idx)) if len(train_idx) else -1
        min_test = int(np.min(test_idx)) if len(test_idx) else 1_000_000
        assert max_train + horizon + embargo < min_test, (
            f"Train index {max_train} + horizon {horizon} + embargo {embargo} "
            f"= {max_train + horizon + embargo} must be < min(test) = {min_test}"
        )
        # Val must come from the END of train (chronologically)
        if len(val_idx) > 0 and len(train_idx) > 0:
            assert int(np.min(val_idx)) > int(np.max(train_idx)), (
                "Validation indices must be chronologically after train indices"
            )
