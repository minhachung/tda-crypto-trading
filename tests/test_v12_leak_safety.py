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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.multi_asset_pipeline import (
    add_targets, create_causal_normalized_windows,
)
from src.tda_v2_features import (
    LeakSafePersistenceImagerFitter, compute_diagrams_only,
)
from src.validation_v2 import time_series_kfold

# Import v12's own add_targets so we exercise the wrapper (and not just
# the underlying src.multi_asset_pipeline.add_targets).
from examples.run_validation_v12 import add_targets as v12_add_targets


def _diagram(rng, k, birth_lo=0.0, birth_hi=1.0,
              pers_lo=0.1, pers_hi=1.5):
    """Synthesize a valid (birth, death) diagram of length k.

    Always generates death > birth via birth + persistence, so
    no point is on the wrong side of the diagonal."""
    if k <= 0:
        return np.zeros((0, 2))
    births = rng.uniform(birth_lo, birth_hi, size=k)
    perss = rng.uniform(pers_lo, pers_hi, size=k)
    return np.column_stack([births, births + perss])


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
# v12-specific: test the actual v12 wrapper, not just src
# ============================================================

def test_v12_add_targets_drops_exactly_horizon_per_asset():
    """v12.add_targets is a wrapper that hardcodes price_col='close'.
    Verify it drops exactly `horizon` rows per asset and produces only
    0/1 targets."""
    df1 = _make_df(n=200, symbol='ABC')
    df2 = _make_df(n=150, symbol='XYZ')
    df = pd.concat([df1, df2], ignore_index=True)

    horizon = 24
    out = v12_add_targets(df, horizon=horizon)

    for symbol, group in out.groupby('symbol'):
        original_n = len(df[df['symbol'] == symbol])
        assert len(group) == original_n - horizon, (
            f"{symbol}: v12_add_targets dropped {original_n - len(group)} rows, "
            f"expected exactly {horizon}"
        )
    assert out['target'].isin([0, 1]).all()


def test_v12_add_targets_strictly_rising_all_ones():
    """v12.add_targets on a strictly rising series must produce 1.0 for
    every retained row (the bug case where the tail leaked 0 labels)."""
    df = _make_df(n=100, symbol='ABC')
    df = df.assign(close=np.linspace(100, 200, 100))
    horizon = 10
    out = v12_add_targets(df, horizon=horizon)
    assert (out['target'] == 1).all(), (
        "v12.add_targets on a strictly rising close series must yield all "
        "1-targets; any 0 indicates the tail wasn't properly dropped."
    )
    assert len(out) == 100 - horizon


def test_v12_add_targets_uses_close_column():
    """v12.add_targets must hardcode price_col='close'. If a caller
    accidentally has a non-close price column, the wrapper should still
    use 'close' (and would raise if 'close' is missing — the contract
    is exactly that)."""
    df = _make_df(n=50, symbol='ABC')
    df = df.assign(close=np.linspace(10, 20, 50))
    out = v12_add_targets(df, horizon=5)
    # Up direction throughout, so target should be 1.
    assert (out['target'] == 1).all()
    # future_return should be positive throughout.
    assert (out['future_return'] > 0).all()


# ============================================================
# Test 11: fold purging
# ============================================================

def test_fold_purging_no_train_target_overlaps_test():
    """Low-level formula sanity check: applies the purge formula
    `tr_idx[tr_idx + horizon < te_start]` directly to time_series_kfold
    output and verifies (a) no purged train target overlaps the test
    fold and (b) at least one fold's *unpurged* tail would have
    overlapped — proving the formula is non-trivial.

    The integration test ``test_evaluate_kfold_leaksafe_purges_train_indices``
    below is the stronger guarantee: it verifies v12's *actual call site*
    applies this same formula. This test stays as documentation of the
    formula itself, independent of v12.

    For every fold, max(train + horizon) < min(test)."""
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
    rng = np.random.RandomState(0)
    test_h0 = [_diagram(rng, 1, birth_lo=0.1, birth_hi=0.2,
                         pers_lo=0.3, pers_hi=0.4) for _ in range(n_test)]
    test_h1 = [_diagram(rng, 1, birth_lo=0.1, birth_hi=0.2,
                         pers_lo=0.3, pers_hi=0.4) for _ in range(n_test)]

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
    train_h0 = [_diagram(rng, k, pers_lo=0.1, pers_hi=1.5)
                  for k in rng.randint(1, 6, size=n_train)]
    train_h1 = [_diagram(rng, k, pers_lo=0.1, pers_hi=0.9)
                  for k in rng.randint(1, 4, size=n_train)]
    test_h0 = [_diagram(rng, k, pers_lo=0.1, pers_hi=1.5)
                 for k in rng.randint(1, 6, size=n_test)]
    test_h1 = [_diagram(rng, k, pers_lo=0.1, pers_hi=0.9)
                 for k in rng.randint(1, 4, size=n_test)]

    fitter.fit(train_h0, train_h1)
    out = fitter.transform(test_h0, test_h1)

    assert out.shape == (n_test, 2 * 10 * 10), \
        f"Expected ({n_test}, 200), got {out.shape}"
    assert not np.allclose(out, 0.0), \
        "Non-trivial diagrams should produce non-zero image features"

    # Skip flags must be False on a successful fit — proves both H0 and
    # H1 imagers were actually constructed (not silently fallbacked to
    # the empty-train zero path).
    assert not fitter._h0_skipped, \
        "H0 imager was skipped on a non-empty train set"
    assert not fitter._h1_skipped, \
        "H1 imager was skipped on a non-empty train set"

    # Each diagram must be valid (death > birth).
    for diag in train_h0 + train_h1 + test_h0 + test_h1:
        if len(diag) > 0:
            assert (diag[:, 1] > diag[:, 0]).all(), \
                "Test fixture produced an invalid diagram (death <= birth)"


def test_imager_fit_on_train_subset_transforms_unseen_test():
    """Leak-safe contract: fit on train, transform any held-out diagram
    set, no shape drift, no exception. Uses TEST diagrams whose
    (birth, persistence) ranges fall OUTSIDE the train ranges — common
    in finance data where holdout volatility may exceed train ranges.

    This test validates **robustness** (fixed shape, no exception) on
    out-of-range inputs. It deliberately does NOT assert that the
    transformed output is non-zero: out-of-train-range diagrams may
    legitimately map to mostly-zero or fully-zero feature vectors
    depending on where the fitted grid clips them. Asserting non-zero
    here would conflate robustness with extrapolation quality."""
    rng = np.random.RandomState(1)
    fitter = LeakSafePersistenceImagerFitter(resolution=8)

    # Train diagrams with small persistence.
    train_h0 = [_diagram(rng, 1, birth_lo=0.0, birth_hi=0.5,
                           pers_lo=0.2, pers_hi=0.5) for _ in range(40)]
    train_h1 = [_diagram(rng, 1, birth_lo=0.0, birth_hi=0.5,
                           pers_lo=0.2, pers_hi=0.5) for _ in range(40)]

    fitter.fit(train_h0, train_h1)

    # Test diagrams with persistence well above the train range.
    test_h0 = [_diagram(rng, 1, birth_lo=0.0, birth_hi=0.5,
                          pers_lo=2.0, pers_hi=3.0) for _ in range(5)]
    test_h1 = [_diagram(rng, 1, birth_lo=0.0, birth_hi=0.5,
                          pers_lo=2.0, pers_hi=3.0) for _ in range(5)]

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


# ============================================================
# Integration tests: drive the actual v12 pipeline
# These prove that v12's *call sites* (not just the helpers) use
# causal normalization and horizon-purged train indices.
# Without these tests, a regression in v12 (e.g. swapping
# create_causal_normalized_windows for a global _normalize, or
# removing the fold-purge) would not be caught.
# ============================================================

# Build a synthetic asset-data dict with valid ML features, v1 columns,
# target, close, and per-row diagrams.
def _make_asset_data(n_rows, symbol, rng, n_diag=1):
    from src.advanced_features import ML_FEATURE_SET

    rows = {}
    rows['timestamp'] = pd.date_range('2024-01-01', periods=n_rows, freq='h')
    rows['close'] = 100.0 + np.cumsum(rng.randn(n_rows) * 0.5)
    rows['symbol'] = symbol
    for col in ML_FEATURE_SET:
        rows[col] = rng.randn(n_rows)
    rows['H0_count'] = rng.randn(n_rows)
    rows['H1_count'] = rng.randn(n_rows)
    rows['target'] = rng.randint(0, 2, size=n_rows).astype(int)
    df = pd.DataFrame(rows)

    h0_diagrams = [_diagram(rng, n_diag) for _ in range(n_rows)]
    h1_diagrams = [_diagram(rng, n_diag) for _ in range(n_rows)]
    return {
        'df': df,
        'h0_diagrams': h0_diagrams,
        'h1_diagrams': h1_diagrams,
    }


def test_evaluate_kfold_leaksafe_purges_train_indices(monkeypatch):
    """Integration test: drive v12.evaluate_kfold_leaksafe directly with
    synthetic asset_data, monkeypatch the imager fit() to record the
    number of training diagrams it receives at each fold, and assert
    those counts equal the expected purged-train counts.

    Will fail if v12 stops applying `tr_idx[tr_idx + horizon < te_start]`."""
    import examples.run_validation_v12 as v12mod
    from src.tda_v2_features import LeakSafePersistenceImagerFitter
    from src.validation_v2 import time_series_kfold

    rng = np.random.RandomState(0)
    n_splits = 5
    horizon = 10

    asset_data = {
        'AAA': _make_asset_data(n_rows=200, symbol='AAA', rng=rng),
        'BBB': _make_asset_data(n_rows=150, symbol='BBB', rng=rng),
    }

    # Stub the imager fit so it (a) records the diagram counts it
    # receives at each call and (b) skips the actual persim work.
    # transform() then returns zeros for both dims (because both
    # _hX_skipped flags are True), which keeps the rest of the
    # evaluate_kfold pipeline running without expensive computation.
    fit_calls = []

    def stub_fit(self, train_h0, train_h1):
        fit_calls.append({
            'h0_count': len(train_h0),
            'h1_count': len(train_h1),
        })
        self._pim_h0 = None
        self._pim_h1 = None
        self._h0_skipped = True
        self._h1_skipped = True
        return self

    monkeypatch.setattr(LeakSafePersistenceImagerFitter, 'fit', stub_fit)

    # Run the v12 evaluator on the v2 path so the imager IS instantiated
    # and fit() is called once per fold.
    v12mod.evaluate_kfold_leaksafe(
        asset_data,
        feature_set='base_plus_v2',
        model_type='logistic',
        prob_threshold=0.65,
        n_splits=n_splits,
        horizon=horizon,
        verbose=False,
    )

    # The imager must be fit exactly once per fold under
    # feature_set='base_plus_v2'. A regression that fits the imager
    # globally before the fold loop, or fits it once and reuses across
    # folds, or fails to fit it at all, would all violate this count.
    assert len(fit_calls) == n_splits, (
        f"Expected the imager to be fit exactly once per fold "
        f"({n_splits} folds → {n_splits} fit() calls) when running with "
        f"feature_set='base_plus_v2', but recorded {len(fit_calls)} "
        f"fit() calls. Likely regression: imager fit moved outside the "
        f"per-fold loop, or the v2 path stopped fitting the imager."
    )

    # Compute the expected purged train count per fold using the same
    # formula v12 applies internally. If v12 drops the purge, its actual
    # counts will exceed these expected values.
    expected_per_fold = []
    for fold_idx in range(n_splits):
        total = 0
        for symbol, data in asset_data.items():
            folds = time_series_kfold(len(data['df']), n_splits=n_splits)
            if fold_idx >= len(folds):
                continue
            tr_idx, te_idx = folds[fold_idx]
            te_start = int(np.min(te_idx))
            purged = tr_idx[tr_idx + horizon < te_start]
            total += len(purged)
        expected_per_fold.append(total)

    actual_h0 = [c['h0_count'] for c in fit_calls]
    actual_h1 = [c['h1_count'] for c in fit_calls]

    assert actual_h0 == expected_per_fold, (
        f"Imager received unpurged train diagrams.\n"
        f"  expected purged counts per fold (with horizon={horizon}): "
        f"{expected_per_fold}\n"
        f"  actual h0 counts                                       : "
        f"{actual_h0}\n"
        "  Likely regression: evaluate_kfold_leaksafe stopped applying "
        "tr_idx[tr_idx + horizon < te_start]."
    )
    assert actual_h1 == expected_per_fold, (
        f"H1 train counts disagree with H0; v12 should pass identical "
        f"purged train rows to both dimensions of the imager.\n"
        f"  expected: {expected_per_fold}\n"
        f"  actual h1: {actual_h1}"
    )

    # Sanity guard: the unpurged total per fold must be strictly larger,
    # otherwise this test would pass even with broken purging.
    unpurged_per_fold = []
    for fold_idx in range(n_splits):
        total = 0
        for symbol, data in asset_data.items():
            folds = time_series_kfold(len(data['df']), n_splits=n_splits)
            if fold_idx >= len(folds):
                continue
            tr_idx, _ = folds[fold_idx]
            total += len(tr_idx)
        unpurged_per_fold.append(total)

    assert any(u > e for u, e in zip(unpurged_per_fold, expected_per_fold)), (
        "Test setup is degenerate: no fold has any train rows that would "
        "have been purged. Increase horizon or test fold structure."
    )


def test_prepare_asset_uses_causal_normalization(monkeypatch):
    """Integration test: drive v12.prepare_asset directly. Monkeypatch
    every I/O and feature dependency to deterministic synthetics, then
    capture the point clouds passed into compute_diagrams_only and
    compare them across two runs that differ ONLY in a far-future row.

    If v12.prepare_asset goes back to a global `_normalize(X_tda)`,
    the perturbation will leak into early windows and this test will
    fail. With causal normalization, early windows are byte-identical
    across the two runs."""
    import examples.run_validation_v12 as v12mod

    n = 200
    window_size = 20
    rng = np.random.RandomState(0)
    base_close = 100.0 + np.cumsum(rng.randn(n) * 0.5)

    def _make_ohlcv(close_arr):
        m = len(close_arr)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=m, freq='h'),
            'open': close_arr,
            'high': close_arr + 0.5,
            'low': close_arr - 0.5,
            'close': close_arr,
            'volume': 1000.0 + np.arange(m) * 0.1,  # deterministic
        })

    state = {'ohlcv': _make_ohlcv(base_close)}

    class _FakeFetcher:
        def __init__(self, **kwargs):
            pass

        def fetch_history(self, days):
            return state['ohlcv']

    monkeypatch.setattr(v12mod, 'HighFreqFetcher', _FakeFetcher)

    # Pass-through advanced features: keeps row count, deterministic.
    def _fake_build_features(df):
        return df.copy()

    monkeypatch.setattr(v12mod, 'build_advanced_features', _fake_build_features)

    # Deterministic 15-d TDA feature matrix derived from close.
    def _fake_get_tda_features(df):
        close = df['close'].values
        cols = np.column_stack([close + i * 0.001 for i in range(15)])
        return cols, [f'col_{i}' for i in range(15)]

    monkeypatch.setattr(v12mod, 'get_tda_features', _fake_get_tda_features)

    # Stub v1 scalars (avoid running ripser).
    def _fake_v1(point_clouds, end_indices=None, verbose=True):
        m = len(point_clouds)
        return pd.DataFrame({
            'H0_count': np.zeros(m),
            'H1_count': np.zeros(m),
            'window_idx': np.arange(m),
            'end_idx': (np.asarray(end_indices, dtype=int)
                          if end_indices is not None
                          else np.arange(m)),
        })

    monkeypatch.setattr(v12mod, 'compute_features_for_windows', _fake_v1)

    # Capture point clouds at each call.
    captures = []

    def _fake_diags(point_clouds, max_dim=1, verbose=True):
        captures.append([np.asarray(pc, dtype=float).copy()
                          for pc in point_clouds])
        m = len(point_clouds)
        zeros = [np.zeros((0, 2)) for _ in range(m)]
        return zeros, list(zeros)

    monkeypatch.setattr(v12mod, 'compute_diagrams_only', _fake_diags)

    # --- Run 1: original ---
    data1 = v12mod.prepare_asset('FAKE', days=10,
                                    window_size=window_size, verbose=False)
    assert data1 is not None
    pcs_a = captures[-1]

    # --- Run 2: same except a far-future row mutated ---
    perturbed = base_close.copy()
    perturbed[180:185] += 1000.0
    state['ohlcv'] = _make_ohlcv(perturbed)

    data2 = v12mod.prepare_asset('FAKE', days=10,
                                    window_size=window_size, verbose=False)
    assert data2 is not None
    pcs_b = captures[-1]

    # The two runs must produce the same number of point clouds.
    assert len(pcs_a) == len(pcs_b), \
        f"len(pcs) drift: {len(pcs_a)} vs {len(pcs_b)}"

    # Early windows (those whose end_idx is well below the perturbation)
    # must be byte-identical. With window_size=20, window i has
    # end_idx = i + 19; for i < 100 the window ends below idx 119, well
    # before the perturbation at idx 180-184.
    n_early = 100
    assert len(pcs_a) >= n_early, \
        f"Need >= {n_early} early windows for the test, got {len(pcs_a)}"

    for i in range(n_early):
        assert np.allclose(pcs_a[i], pcs_b[i], atol=1e-12), (
            f"Early window {i} (end_idx ~ {i + window_size - 1}) differs "
            f"between original and far-future-perturbed runs. "
            f"prepare_asset is leaking future distribution into early "
            f"point clouds — likely regression to a global _normalize "
            f"call before windowing."
        )

    # Sanity guard: the LATE window must reflect the perturbation,
    # otherwise the perturbation didn't propagate into the TDA features
    # and this test isn't actually exercising leak protection.
    last = len(pcs_a) - 1
    assert not np.allclose(pcs_a[last], pcs_b[last], atol=1e-6), (
        "Sanity guard: the last window should reflect the far-future "
        "perturbation. If it doesn't, the test setup didn't actually "
        "perturb the TDA features and the assertion above is vacuous."
    )
