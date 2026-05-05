"""
Regime Filter: Only allow trades in favorable market conditions.

Filters:
  - Volatility regime: only trade when realized vol > median (fat-tail edge)
  - Trend regime: only trade with trend (or against if reversion strategy)
  - Volume regime: only trade with above-average volume (liquidity)

Applied as post-hoc mask on ML signals to reduce trade count + improve quality.
"""

import numpy as np
import pandas as pd


def vol_regime_mask(df, vol_col='rv_20', threshold='median'):
    """Mask True where volatility is above threshold."""
    vol = df[vol_col].values
    if threshold == 'median':
        thresh = np.nanmedian(vol)
    elif threshold == 'q75':
        thresh = np.nanquantile(vol, 0.75)
    elif threshold == 'q25':
        thresh = np.nanquantile(vol, 0.25)
    elif isinstance(threshold, (int, float)):
        thresh = threshold
    else:
        return np.ones(len(df), dtype=bool)
    return vol > thresh


def trend_regime_mask(df, trend_col='trend_strength', mode='positive'):
    """Mask True where trend matches mode."""
    trend = df[trend_col].values
    if mode == 'positive':
        return trend > 0
    elif mode == 'negative':
        return trend < 0
    elif mode == 'strong':
        return np.abs(trend) > np.nanquantile(np.abs(trend), 0.5)
    else:
        return np.ones(len(df), dtype=bool)


def volume_regime_mask(df, vol_col='vol_zscore_20', threshold=0.0):
    """Mask True where volume z-score is above threshold."""
    if vol_col not in df.columns:
        return np.ones(len(df), dtype=bool)
    return df[vol_col].values > threshold


def apply_regime_filter(signals_df, features_df, filters=None):
    """
    Mask out signals that don't pass regime filter.

    Args:
        signals_df: with 'signal' column
        features_df: with regime-relevant columns
        filters: dict like {'vol': 'median', 'volume': 0.0}
    """
    if filters is None:
        filters = {'vol': 'median'}

    n = min(len(signals_df), len(features_df))
    mask = np.ones(n, dtype=bool)

    if 'vol' in filters and 'rv_20' in features_df.columns:
        mask &= vol_regime_mask(features_df.iloc[:n], 'rv_20', filters['vol'])

    if 'trend' in filters and 'trend_strength' in features_df.columns:
        mask &= trend_regime_mask(features_df.iloc[:n], 'trend_strength', filters['trend'])

    if 'volume' in filters and 'vol_zscore_20' in features_df.columns:
        mask &= volume_regime_mask(features_df.iloc[:n], 'vol_zscore_20', filters['volume'])

    out = signals_df.iloc[:n].copy().reset_index(drop=True)
    out.loc[~mask, 'signal'] = 'HOLD'
    out.loc[~mask, 'position_size'] = 0.0
    return out
