"""
Advanced Feature Engineering for TDA Point Clouds.

Replaces raw OHLCV with stationary, signal-rich features:
  - Log returns (stationarity)
  - Garman-Klass volatility (uses OHLC, lower variance than close-to-close)
  - Parkinson volatility (HL range based)
  - Volume z-score (cross-period normalization)
  - Volume momentum (Δ volume / volume)
  - RSI/MACD/BB normalized to z-scores
  - Price acceleration (second derivative)
  - Realized vol over multiple horizons
  - High-low spread (intraday range)

Plus regime detection features:
  - Volatility regime (rolling vol percentile)
  - Trend regime (slope of EMA)
"""

import numpy as np
import pandas as pd


# ============================================================
# Volatility Estimators
# ============================================================

def garman_klass_volatility(df, window=20):
    """
    Garman-Klass volatility estimator.
    σ²_GK = 0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²
    Uses OHLC, ~7x more efficient than close-to-close.
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    var_term = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rolling_var = var_term.rolling(window=window).mean()
    return np.sqrt(rolling_var.clip(lower=0))


def parkinson_volatility(df, window=20):
    """Parkinson volatility from high-low range. ~4x more efficient than C2C."""
    log_hl = np.log(df['high'] / df['low'])
    var_term = log_hl**2 / (4 * np.log(2))
    rolling_var = var_term.rolling(window=window).mean()
    return np.sqrt(rolling_var.clip(lower=0))


def realized_volatility(returns, window):
    return returns.rolling(window=window).std()


# ============================================================
# Feature Builder
# ============================================================

def build_advanced_features(df):
    """
    Replace raw OHLCV pipeline with engineered features.

    Input: DataFrame with timestamp, open, high, low, close, volume.
    Output: DataFrame with engineered features (drops warmup rows).
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_24'] = np.log(df['close'] / df['close'].shift(24))

    df['accel'] = df['log_return'].diff()

    df['gk_vol_20'] = garman_klass_volatility(df, window=20)
    df['gk_vol_60'] = garman_klass_volatility(df, window=60)
    df['parkinson_20'] = parkinson_volatility(df, window=20)
    df['rv_20'] = realized_volatility(df['log_return'], window=20)
    df['rv_60'] = realized_volatility(df['log_return'], window=60)

    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['oc_spread'] = (df['close'] - df['open']) / df['open']

    df['vol_zscore_20'] = (
        (df['volume'] - df['volume'].rolling(window=20).mean())
        / df['volume'].rolling(window=20).std()
    )
    df['vol_momentum'] = df['volume'].pct_change()
    df['vol_ratio_5_20'] = (
        df['volume'].rolling(window=5).mean()
        / df['volume'].rolling(window=20).mean()
    )

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    df['rsi_centered'] = (rsi - 50) / 50

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd_normalized'] = macd / df['close']
    df['macd_signal_normalized'] = macd.ewm(span=9, adjust=False).mean() / df['close']

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_position'] = (df['close'] - sma20) / (2 * std20).replace(0, np.nan)
    df['bb_width'] = (4 * std20) / sma20

    df['vol_regime'] = (
        df['rv_20'].rolling(window=100).rank(pct=True).fillna(0.5)
    )
    df['trend_strength'] = (
        ema12 - ema12.shift(20)
    ) / df['close']

    df['vol_pressure'] = df['vol_zscore_20'] * np.sign(df['log_return'])

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# ============================================================
# Feature Sets for Different Use Cases
# ============================================================

# For TDA point clouds - features that should be stationary and informative
TDA_FEATURE_SET = [
    'log_return',
    'log_return_5',
    'accel',
    'gk_vol_20',
    'parkinson_20',
    'rv_20',
    'hl_spread',
    'oc_spread',
    'vol_zscore_20',
    'vol_ratio_5_20',
    'rsi_centered',
    'macd_normalized',
    'bb_position',
    'bb_width',
    'trend_strength',
]

# For ML classifier - everything (model picks what matters)
ML_FEATURE_SET = TDA_FEATURE_SET + [
    'log_return_24',
    'gk_vol_60',
    'rv_60',
    'macd_signal_normalized',
    'vol_pressure',
    'vol_regime',
]


def get_tda_features(df, feature_set=None):
    """Extract TDA point-cloud features."""
    if feature_set is None:
        feature_set = TDA_FEATURE_SET
    cols = [c for c in feature_set if c in df.columns]
    return df[cols].values, cols


def get_ml_features(df, feature_set=None):
    """Extract ML classifier features (combined raw + TDA-derived)."""
    if feature_set is None:
        feature_set = ML_FEATURE_SET
    cols = [c for c in feature_set if c in df.columns]
    return df[cols].values, cols
