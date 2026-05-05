"""
Data Pipeline: Fetch and preprocess crypto data for TDA analysis.

Fetches OHLCV from CoinGecko (free), computes technical indicators,
normalizes features, and creates point clouds for persistent homology.
"""

import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch OHLCV and on-chain data for crypto assets."""

    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    SYMBOL_TO_ID = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'ADA': 'cardano',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
    }

    def __init__(self, symbol='BTC', days=365):
        self.symbol = symbol.upper()
        self.coin_id = self.SYMBOL_TO_ID.get(self.symbol, symbol.lower())
        self.days = days

    def fetch_ohlcv(self, retries=3):
        """Fetch OHLC + Volume from CoinGecko."""
        url = f"{self.COINGECKO_BASE}/coins/{self.coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': str(self.days)}

        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                ohlc_data = response.json()

                if not ohlc_data:
                    raise ValueError(f"No OHLC data returned for {self.symbol}")

                df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                volume_df = self._fetch_volume()
                if volume_df is not None:
                    df = pd.merge_asof(
                        df.sort_values('timestamp'),
                        volume_df.sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest',
                    )
                else:
                    df['volume'] = 0.0

                return df.reset_index(drop=True)

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"  Retry {attempt + 1}/{retries}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _fetch_volume(self):
        """Fetch volume separately from market_chart endpoint."""
        url = f"{self.COINGECKO_BASE}/coins/{self.coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(self.days), 'interval': 'daily'}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            volumes = data.get('total_volumes', [])
            if not volumes:
                return None
            df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"  Volume fetch failed: {e}")
            return None


def add_technical_indicators(df):
    """Add RSI, MACD, Bollinger Bands using pandas (no ta-lib dependency)."""
    df = df.copy()

    df['rsi'] = compute_rsi(df['close'], period=14)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_high'] = sma20 + (std20 * 2)
    df['bb_low'] = sma20 - (std20 * 2)
    df['bb_mid'] = sma20
    df['bb_width'] = df['bb_high'] - df['bb_low']

    df['return_1d'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_10d'] = df['return_1d'].rolling(window=10).std()

    return df.dropna().reset_index(drop=True)


def compute_rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def normalize_features(df, feature_cols):
    """Z-score normalize features."""
    X = df[feature_cols].values.astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_normalized = (X - mean) / std
    return X_normalized, {'mean': mean, 'std': std}


def create_sliding_windows(X, window_size=10, stride=1):
    """
    Create sliding window point clouds.

    Each window becomes a point cloud where each row is a time step.
    Returns list of (window_size, n_features) arrays.
    """
    point_clouds = []
    end_indices = []

    for i in range(0, len(X) - window_size + 1, stride):
        window = X[i:i + window_size]
        point_clouds.append(window)
        end_indices.append(i + window_size - 1)

    return point_clouds, end_indices


def run_pipeline(symbol='BTC', days=365, window_size=10, stride=1, save_dir='data'):
    """End-to-end data pipeline."""
    os.makedirs(f'{save_dir}/raw', exist_ok=True)
    os.makedirs(f'{save_dir}/processed', exist_ok=True)

    print(f"[Data Pipeline] Fetching {symbol} - {days} days")
    fetcher = DataFetcher(symbol=symbol, days=days)
    df = fetcher.fetch_ohlcv()
    print(f"  Fetched {len(df)} rows of OHLCV data")

    print(f"[Data Pipeline] Computing technical indicators")
    df = add_technical_indicators(df)
    print(f"  After indicators: {len(df)} rows, {len(df.columns)} columns")

    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'bb_mid',
                    'volatility_10d']

    print(f"[Data Pipeline] Normalizing {len(feature_cols)} features")
    X_normalized, scaler_params = normalize_features(df, feature_cols)

    print(f"[Data Pipeline] Creating sliding windows (size={window_size}, stride={stride})")
    point_clouds, end_indices = create_sliding_windows(X_normalized, window_size, stride)
    print(f"  Created {len(point_clouds)} point clouds")

    raw_path = f'{save_dir}/raw/{symbol}_ohlcv.csv'
    df.to_csv(raw_path, index=False)
    print(f"  Saved raw data: {raw_path}")

    pc_array = np.array(point_clouds)
    pc_path = f'{save_dir}/processed/{symbol}_point_clouds.npy'
    np.save(pc_path, pc_array)
    print(f"  Saved point clouds: {pc_path} (shape: {pc_array.shape})")

    idx_path = f'{save_dir}/processed/{symbol}_window_indices.npy'
    np.save(idx_path, np.array(end_indices))
    print(f"  Saved window indices: {idx_path}")

    return {
        'df': df,
        'point_clouds': point_clouds,
        'end_indices': end_indices,
        'feature_cols': feature_cols,
        'scaler_params': scaler_params,
    }


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    result = run_pipeline(symbol=symbol, days=days)
    print(f"\n[Done] Pipeline complete for {symbol}")
    print(f"  DataFrame shape: {result['df'].shape}")
    print(f"  Point clouds: {len(result['point_clouds'])}")
