"""Integration test: full pipeline on a tiny synthetic dataset."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_features import build_advanced_features
from src.persistent_homology import compute_features_for_windows
from src.backtester import Backtester


def make_synthetic_pool(n_per_asset=400, n_assets=2, seed=42):
    """Create a tiny synthetic multi-asset OHLCV pool."""
    rng = np.random.RandomState(seed)
    rows = []
    for asset in [f'A{i}' for i in range(n_assets)]:
        base = 100 + np.cumsum(rng.randn(n_per_asset) * 0.5)
        ts = pd.date_range('2025-01-01', periods=n_per_asset, freq='1h')
        df = pd.DataFrame({
            'timestamp': ts,
            'open':  base + rng.randn(n_per_asset) * 0.1,
            'high':  base + np.abs(rng.randn(n_per_asset) * 0.3),
            'low':   base - np.abs(rng.randn(n_per_asset) * 0.3),
            'close': base,
            'volume': 1000 + rng.randn(n_per_asset) * 100,
            'symbol': asset,
        })
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_pipeline_runs_end_to_end_synthetic():
    """Tiny pipeline should not crash."""
    pool = make_synthetic_pool()

    enriched_parts = []
    for sym, group in pool.groupby('symbol'):
        feats = build_advanced_features(group)
        feats['symbol'] = sym
        enriched_parts.append(feats)
    enriched = pd.concat(enriched_parts, ignore_index=True)

    assert len(enriched) > 0
    assert 'log_return' in enriched.columns


def test_persistence_pipeline_on_synthetic():
    """Persistent homology pipeline runs without crashing on small windows."""
    pool = make_synthetic_pool(n_per_asset=200, n_assets=1)
    feats = build_advanced_features(pool)
    if len(feats) < 50:
        return
    feature_cols = ['log_return', 'gk_vol_20', 'rv_20']
    feature_cols = [c for c in feature_cols if c in feats.columns]
    X = feats[feature_cols].values

    point_clouds = []
    end_idx = []
    for i in range(0, len(X) - 20):
        point_clouds.append(X[i:i + 20])
        end_idx.append(i + 19)

    if not point_clouds:
        return
    tda_features_df = compute_features_for_windows(
        point_clouds[:50], end_indices=end_idx[:50], verbose=False,
    )
    assert len(tda_features_df) > 0
    assert 'H1_l1_norm' in tda_features_df.columns


def test_backtester_runs_on_dummy_signals():
    """Backtester completes on a minimal valid input."""
    prices = np.array([100, 101, 102, 99, 100, 105, 110])
    signals = pd.DataFrame({
        'signal': ['BUY', 'HOLD', 'SELL', 'HOLD', 'BUY', 'HOLD', 'SELL'],
        'position_size': [0.5, 0.0, -0.5, 0.0, 0.5, 0.0, -0.5],
    })
    bt = Backtester(initial_capital=10000)
    res = bt.run(prices, signals)
    assert 'metrics' in res
    assert res['metrics']['final_equity'] > 0


if __name__ == '__main__':
    print("Running integration tests as smoke test...")
    test_pipeline_runs_end_to_end_synthetic()
    print("  test_pipeline_runs_end_to_end_synthetic: PASSED")
    test_persistence_pipeline_on_synthetic()
    print("  test_persistence_pipeline_on_synthetic: PASSED")
    test_backtester_runs_on_dummy_signals()
    print("  test_backtester_runs_on_dummy_signals: PASSED")
    print("\nAll smoke tests passed.")
