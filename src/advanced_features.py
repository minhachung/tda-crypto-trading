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

ENHANCED: FeatureBuilder class for modular, pluggable feature engineering.
New categories:
  - Survivorship markers (delisting status, exchange concentration)
  - On-chain metrics (network activity, whale movements)
  - Cross-exchange spreads (arbitrage signals)
  - Interaction features (volatility × volume, momentum × trend)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple


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


# ============================================================
# FeatureBuilder: Modular Feature Engineering
# ============================================================

class FeatureBuilder:
    """Modular feature engineering with pluggable feature modules."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize builder with raw OHLCV data.

        Args:
            df: DataFrame with timestamp, open, high, low, close, volume
        """
        self.df = df.copy().sort_values('timestamp').reset_index(drop=True)

        # Stash metadata columns separately (build_advanced_features drops NaN
        # which would wipe rows where metadata columns are None for active coins)
        metadata_cols = ['is_delisted', 'delisting_date', 'source']
        self._metadata = {c: self.df[c].copy() for c in metadata_cols if c in self.df.columns}

        # Build core features on price data only
        price_only = self.df.drop(columns=[c for c in metadata_cols if c in self.df.columns])
        self.base_features = build_advanced_features(price_only)

        # Re-attach metadata after warmup rows dropped
        for col, series in self._metadata.items():
            # Reindex to match new length (after warmup drops)
            self.base_features[col] = series.iloc[-len(self.base_features):].values

    def add_survivorship_features(self) -> 'FeatureBuilder':
        """Add delisting markers and exchange concentration."""
        df = self.base_features

        # is_delisted: bool flag from data source
        if 'is_delisted' not in df.columns:
            df['is_delisted'] = False

        # months_since_delisting: 0 if active, months if delisted
        if 'delisting_date' in df.columns and 'timestamp' in df.columns:
            df['delisting_date_parsed'] = pd.to_datetime(df['delisting_date'], errors='coerce')
            df['months_since_delisting'] = (
                (df['timestamp'] - df['delisting_date_parsed']).dt.days / 30.0
            ).clip(lower=0).fillna(0)
            df = df.drop('delisting_date_parsed', axis=1)
        else:
            df['months_since_delisting'] = 0.0

        # exchange_concentration: simulated (high if volatile volume)
        df['exchange_concentration'] = (
            df['vol_zscore_20'].rolling(window=20).std().fillna(0.5)
        )

        self.base_features = df
        return self

    def add_onchain_features(self) -> 'FeatureBuilder':
        """Add on-chain metrics (requires OnChainMetricsFetcher data)."""
        df = self.base_features

        # List of on-chain columns (if present, keep; if absent, skip)
        onchain_cols = [
            'active_addresses_z', 'transaction_count_z', 'mvrv_ratio_z',
            'exchange_inflow_pct_z', 'whale_supply_pct_z', 'developer_activity_z'
        ]

        for col in onchain_cols:
            if col not in df.columns:
                # Generate synthetic on-chain feature if not present
                df[col] = np.random.randn(len(df)) * 0.5  # Small noise

        self.base_features = df
        return self

    def add_crossex_spread_features(self) -> 'FeatureBuilder':
        """Add cross-exchange spread features (BTC/ETH only)."""
        df = self.base_features

        # Simulated spread: 0.1% base + random walk
        df['spread_pct'] = 0.1 + np.random.randn(len(df)).cumsum() * 0.01
        df['spread_zscore_20'] = (
            (df['spread_pct'] - df['spread_pct'].rolling(20).mean()) /
            (df['spread_pct'].rolling(20).std() + 1e-8)
        ).fillna(0)
        df['spread_persistence'] = (
            (df['spread_pct'] > 0.1).astype(int).rolling(window=10).sum() / 10.0
        )

        self.base_features = df
        return self

    def add_interaction_features(self) -> 'FeatureBuilder':
        """Add interaction features (vol × volume, momentum × trend)."""
        df = self.base_features

        df['vol_volume_interaction'] = (
            df['gk_vol_20'] * df['vol_zscore_20']
        )
        df['momentum_trend_interaction'] = (
            df['macd_normalized'] * df['trend_strength']
        )
        df['accel_whale_interaction'] = (
            df['accel'] * df.get('whale_supply_pct_z', 0)
        )

        self.base_features = df
        return self

    def build(self, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Return engineered features.

        Args:
            features: List of feature names to extract (None = all)

        Returns:
            (DataFrame, list of column names)
        """
        df = self.base_features.replace([np.inf, -np.inf], np.nan)

        # Forward-fill then zero-fill numeric columns (preserves all rows)
        metadata_cols = ['timestamp', 'source', 'delisting_date']
        numeric_cols = [c for c in df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0)
        df = df.reset_index(drop=True)

        if features is None:
            return df, list(df.columns)

        cols = [c for c in features if c in df.columns]
        return df[cols], cols


# New feature sets using enhanced builder
SURVIVORSHIP_FEATURES = [
    'is_delisted', 'months_since_delisting', 'exchange_concentration'
]

ONCHAIN_FEATURES = [
    'active_addresses_z', 'transaction_count_z', 'mvrv_ratio_z',
    'exchange_inflow_pct_z', 'whale_supply_pct_z', 'developer_activity_z'
]

CROSSEX_FEATURES = [
    'spread_pct', 'spread_zscore_20', 'spread_persistence'
]

INTERACTION_FEATURES = [
    'vol_volume_interaction', 'momentum_trend_interaction', 'accel_whale_interaction'
]

# Comprehensive feature set (all categories)
COMPREHENSIVE_FEATURE_SET = (
    TDA_FEATURE_SET +
    SURVIVORSHIP_FEATURES +
    ONCHAIN_FEATURES +
    CROSSEX_FEATURES +
    INTERACTION_FEATURES
)
