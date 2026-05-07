"""Tests for leak-resistant target construction and feature windows."""

import numpy as np
import pandas as pd

from examples.run_validation_v8 import add_targets as add_targets_v8
from examples.run_validation_v9 import add_targets as add_targets_v9
from src.multi_asset_pipeline import add_targets, create_causal_normalized_windows


def make_price_frame(n=10):
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='1h'),
        'close': np.arange(100.0, 100.0 + n),
        'symbol': ['BTC'] * n,
    })


def test_multi_asset_targets_drop_unknown_future_rows():
    df = make_price_frame(n=10)
    targeted = add_targets(df, horizon=3)
    assert len(targeted) == 7
    assert targeted['target'].notna().all()
    assert targeted['future_return'].notna().all()


def test_validation_targets_drop_unknown_future_rows():
    df = make_price_frame(n=10)
    assert len(add_targets_v8(df, horizon=3)) == 7
    assert len(add_targets_v9(df, horizon=3)) == 7


def test_causal_normalized_windows_do_not_use_later_rows():
    X = np.arange(20.0).reshape(10, 2)
    X_with_future_spike = X.copy()
    X_with_future_spike[-1] = 10000.0

    windows, _ = create_causal_normalized_windows(X, window_size=3)
    windows_spiked, _ = create_causal_normalized_windows(X_with_future_spike, window_size=3)

    np.testing.assert_allclose(windows[0], windows_spiked[0])
