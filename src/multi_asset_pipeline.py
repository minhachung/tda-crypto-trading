"""
Multi-Asset Pooled Training Pipeline.

Fetches multiple cryptos, computes advanced features + TDA features,
pools them into one labeled dataset, then trains ML classifier on the pool.

Why pooling:
  - 4 assets x 4380 hourly candles = 17,520 samples vs 4380 single-asset
  - Cross-asset learning forces model to find transferable signals
  - Tighter Wilson CI from larger n
"""

import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.binance_data import HighFreqFetcher
from src.advanced_features import (
    build_advanced_features,
    get_tda_features,
    TDA_FEATURE_SET,
    ML_FEATURE_SET,
)
from src.persistent_homology import compute_features_for_windows


def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def create_causal_normalized_windows(X, window_size=20, stride=1):
    """Create point clouds normalized only with data available at each window end."""
    pcs, end_idx = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        end = i + window_size
        history = X[:end]
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        std[std == 0] = 1.0
        pcs.append((X[i:end] - mean) / std)
        end_idx.append(end - 1)
    return pcs, end_idx


def create_sliding_windows(X, window_size=20, stride=1):
    pcs, end_idx = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        pcs.append(X[i:i + window_size])
        end_idx.append(i + window_size - 1)
    return pcs, end_idx


def fetch_and_build_asset(symbol, days=180, interval='1h', window_size=20, stride=1):
    """Fetch one asset, compute advanced features + TDA features."""
    print(f"  [{symbol}] Fetching {days} days of {interval} from Coinbase")
    fetcher = HighFreqFetcher(symbol=symbol, interval=interval)
    raw_df = fetcher.fetch_history(days=days)

    print(f"  [{symbol}] Building advanced features")
    df = build_advanced_features(raw_df)
    print(f"  [{symbol}] After feature engineering: {len(df)} rows")

    if len(df) < window_size + 50:
        print(f"  [{symbol}] WARNING: too few rows after feature build")
        return None

    X_tda, tda_cols = get_tda_features(df)
    pcs, end_idx = create_causal_normalized_windows(
        X_tda, window_size=window_size, stride=stride
    )
    print(f"  [{symbol}] Created {len(pcs)} TDA point clouds")

    print(f"  [{symbol}] Computing persistent homology")
    tda_features_df = compute_features_for_windows(
        pcs, end_indices=end_idx, verbose=False
    )

    aligned_idx = tda_features_df['end_idx'].astype(int).values
    aligned_idx = aligned_idx[aligned_idx < len(df)]
    aligned_df = df.iloc[aligned_idx].reset_index(drop=True)
    tda_features_df = tda_features_df.iloc[:len(aligned_df)].reset_index(drop=True)

    combined = pd.concat([
        aligned_df.reset_index(drop=True),
        tda_features_df.reset_index(drop=True),
    ], axis=1)

    combined['symbol'] = symbol
    print(f"  [{symbol}] Final: {len(combined)} samples")
    return combined


def fetch_multi_asset_pool(symbols, days=180, interval='1h', window_size=20):
    """Fetch and combine multiple assets into a pooled dataset."""
    pool = []
    for symbol in symbols:
        try:
            asset_df = fetch_and_build_asset(symbol, days, interval, window_size)
            if asset_df is not None:
                pool.append(asset_df)
            time.sleep(1)
        except Exception as e:
            print(f"  [{symbol}] ERROR: {e}")

    if not pool:
        raise ValueError("No assets successfully fetched")

    pooled = pd.concat(pool, ignore_index=True)
    print(f"\n[Pool] Combined: {len(pooled)} samples across {len(symbols)} assets")
    return pooled


def add_targets(df, horizon=1, price_col='close'):
    """Add binary direction target: 1 if up after `horizon`, else 0."""
    df = df.copy()
    if 'symbol' in df.columns:
        future_price = df.groupby('symbol')[price_col].transform(lambda x: x.shift(-horizon))
    else:
        future_price = df[price_col].shift(-horizon)
    df['future_price'] = future_price
    df['future_return'] = df['future_price'] / df[price_col] - 1.0
    df['target'] = np.where(df['future_price'].notna(), df['future_price'] > df[price_col], np.nan)
    return df.dropna(subset=['target', 'future_return']).reset_index(drop=True)


# ============================================================
# Feature column extraction
# ============================================================

def get_combined_feature_cols(df):
    """All feature columns: ML features + TDA features."""
    ml_cols = [c for c in ML_FEATURE_SET if c in df.columns]
    tda_cols = [c for c in df.columns if c.startswith(('H0_', 'H1_'))]
    return ml_cols + tda_cols
