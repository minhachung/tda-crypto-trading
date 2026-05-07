"""Test feature engineering: no lookahead, correctness on known cases."""

import numpy as np
import pandas as pd
import pytest

from src.advanced_features import (
    build_advanced_features,
    garman_klass_volatility,
    parkinson_volatility,
)


def test_garman_klass_volatility_known_case():
    """GK volatility on a constant-price series should be 0."""
    df = pd.DataFrame({
        'open':  [100.0] * 30,
        'high':  [100.0] * 30,
        'low':   [100.0] * 30,
        'close': [100.0] * 30,
    })
    gk = garman_klass_volatility(df, window=20)
    assert gk.iloc[-1] == 0.0 or np.isnan(gk.iloc[-1])


def test_garman_klass_volatility_nontrivial():
    """GK on a noisy series should be positive."""
    np.random.seed(0)
    n = 50
    base = 100.0
    df = pd.DataFrame({
        'open':  base + np.random.randn(n) * 0.5,
        'high':  base + 1.0,
        'low':   base - 1.0,
        'close': base + np.random.randn(n) * 0.5,
    })
    gk = garman_klass_volatility(df, window=20)
    assert gk.iloc[-1] > 0


def test_parkinson_volatility_increases_with_range():
    """Parkinson vol should be higher when high-low range is wider."""
    n = 30
    narrow = pd.DataFrame({
        'open':  [100.0] * n, 'high':  [100.5] * n,
        'low':   [99.5] * n,  'close': [100.0] * n,
    })
    wide = pd.DataFrame({
        'open':  [100.0] * n, 'high':  [105.0] * n,
        'low':   [95.0] * n,  'close': [100.0] * n,
    })
    pk_narrow = parkinson_volatility(narrow, window=20).iloc[-1]
    pk_wide = parkinson_volatility(wide, window=20).iloc[-1]
    assert pk_wide > pk_narrow


def test_no_lookahead_in_rolling_features(synthetic_ohlcv):
    """Feature at time t must depend only on data <= t (rolling, not centered)."""
    df = synthetic_ohlcv.copy()
    feats_full = build_advanced_features(df)

    cut = len(df) // 2
    feats_truncated = build_advanced_features(df.iloc[:cut])

    common = min(len(feats_full), len(feats_truncated))
    n_check = min(50, common - 30)
    for col in ['gk_vol_20', 'rv_20', 'log_return']:
        if col in feats_full.columns and col in feats_truncated.columns:
            full_tail = feats_full[col].iloc[20:20 + n_check].values
            trunc_tail = feats_truncated[col].iloc[20:20 + n_check].values
            same = full_tail.shape == trunc_tail.shape
            if same:
                np.testing.assert_allclose(
                    full_tail, trunc_tail, rtol=1e-6, atol=1e-9,
                    err_msg=f"Lookahead detected in {col}",
                )


def test_build_advanced_features_drops_warmup(synthetic_ohlcv):
    """After feature engineering, the warmup rows should be dropped."""
    feats = build_advanced_features(synthetic_ohlcv)
    assert len(feats) < len(synthetic_ohlcv)
    assert not feats.isna().any().any(), "NaN values should be dropped"


def test_build_advanced_features_columns(synthetic_ohlcv):
    """Verify expected feature columns are present."""
    feats = build_advanced_features(synthetic_ohlcv)
    expected = {'log_return', 'gk_vol_20', 'parkinson_20', 'rsi_centered',
                'macd_normalized', 'bb_position', 'vol_zscore_20', 'vol_regime'}
    assert expected.issubset(set(feats.columns))
