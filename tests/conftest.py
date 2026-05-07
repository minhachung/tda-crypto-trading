"""Shared pytest fixtures."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """Deterministic OHLCV with 500 rows for testing."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range('2025-01-01', periods=n, freq='1h')
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': base + np.random.randn(n) * 0.1,
        'high': base + np.abs(np.random.randn(n) * 0.3),
        'low': base - np.abs(np.random.randn(n) * 0.3),
        'close': base,
        'volume': 1000 + np.random.randn(n) * 100,
    })
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df


@pytest.fixture
def circle_point_cloud():
    """Points sampled from a unit circle — has one prominent H1 feature."""
    np.random.seed(0)
    n = 30
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += np.random.randn(*pts.shape) * 0.02
    return pts


@pytest.fixture
def random_point_cloud():
    """Random uniform points in [0,1]^3 — should have no strong H1 signal."""
    np.random.seed(0)
    return np.random.rand(50, 3)
