"""
High-Frequency Data Fetcher: Coinbase Exchange API.

Free, globally accessible (unlike Binance which is geo-blocked in some regions).
Provides hourly OHLCV with full historical pagination - 24x more data than CoinGecko.

Endpoint: https://api.exchange.coinbase.com/products/{product}/candles
Returns: [time, low, high, open, close, volume] in descending order.
Max 300 candles per request.
"""

import os
import time
import numpy as np
import pandas as pd
import requests


class HighFreqFetcher:
    """Coinbase Exchange API fetcher with pagination."""

    BASE = "https://api.exchange.coinbase.com/products/{symbol}/candles"

    SYMBOL_MAP = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
        'ADA': 'ADA-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
        'MATIC': 'MATIC-USD', 'AVAX': 'AVAX-USD',
        'XRP': 'XRP-USD', 'DOGE': 'DOGE-USD', 'LTC': 'LTC-USD',
        'BCH': 'BCH-USD', 'ATOM': 'ATOM-USD',
    }

    GRANULARITY_SEC = {
        '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '6h': 21600, '1d': 86400,
    }

    MAX_CANDLES_PER_CALL = 300

    def __init__(self, symbol='BTC', interval='1h'):
        self.symbol = symbol.upper()
        self.product = self.SYMBOL_MAP.get(self.symbol, f'{self.symbol}-USD')
        self.interval = interval
        self.granularity = self.GRANULARITY_SEC[interval]

    def fetch_chunk(self, start_iso, end_iso):
        url = self.BASE.format(symbol=self.product)
        params = {
            'granularity': self.granularity,
            'start': start_iso,
            'end': end_iso,
        }
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt < 3:
                    time.sleep(1.5 ** attempt)
                else:
                    raise

    def fetch_history(self, days=365):
        """Paginate to fetch full history."""
        end_dt = pd.Timestamp.utcnow().floor('h')
        start_dt = end_dt - pd.Timedelta(days=days)

        chunk_seconds = self.MAX_CANDLES_PER_CALL * self.granularity
        all_candles = []

        cursor = start_dt
        n_calls = 0
        while cursor < end_dt:
            chunk_end = min(cursor + pd.Timedelta(seconds=chunk_seconds), end_dt)
            try:
                candles = self.fetch_chunk(cursor.isoformat(), chunk_end.isoformat())
                if candles:
                    all_candles.extend(candles)
                n_calls += 1
                if n_calls % 10 == 0:
                    print(f"    Fetched {len(all_candles)} candles ({n_calls} calls)...")
            except Exception as e:
                print(f"    Skipped chunk {cursor}: {e}")
            cursor = chunk_end
            time.sleep(0.35)

        if not all_candles:
            raise ValueError(f"No data for {self.product}")

        df = pd.DataFrame(all_candles, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def add_technical_indicators(df):
    """Compute RSI, MACD, Bollinger Bands + return/volatility features."""
    df = df.copy()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_high'] = sma20 + 2 * std20
    df['bb_low'] = sma20 - 2 * std20
    df['bb_mid'] = sma20

    df['return_1'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['return_1'].rolling(window=20).std()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

    return df.dropna().reset_index(drop=True)


def normalize_features(df, feature_cols):
    X = df[feature_cols].values.astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, {'mean': mean, 'std': std}


def create_sliding_windows(X, window_size=20, stride=1):
    pcs, idx = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        pcs.append(X[i:i + window_size])
        idx.append(i + window_size - 1)
    return pcs, idx


def run_hf_pipeline(symbol='BTC', days=365, interval='1h',
                    window_size=20, stride=1, save_dir='data'):
    """End-to-end pipeline using Coinbase hourly data."""
    os.makedirs(f'{save_dir}/raw', exist_ok=True)
    os.makedirs(f'{save_dir}/processed', exist_ok=True)

    print(f"[HF Pipeline] Fetching {symbol} {interval} from Coinbase ({days} days)")
    fetcher = HighFreqFetcher(symbol=symbol, interval=interval)
    df = fetcher.fetch_history(days=days)
    print(f"  Fetched {len(df)} candles ({df['timestamp'].min()} -> {df['timestamp'].max()})")

    print(f"[HF Pipeline] Computing technical indicators")
    df = add_technical_indicators(df)
    print(f"  After indicators: {len(df)} rows")

    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'bb_mid',
                    'volatility_20', 'volume_ratio']

    print(f"[HF Pipeline] Normalizing {len(feature_cols)} features")
    X_norm, scaler = normalize_features(df, feature_cols)

    print(f"[HF Pipeline] Creating sliding windows (size={window_size}, stride={stride})")
    pcs, idx = create_sliding_windows(X_norm, window_size, stride)
    print(f"  Created {len(pcs)} point clouds")

    raw_path = f'{save_dir}/raw/{symbol}_hf_ohlcv.csv'
    df.to_csv(raw_path, index=False)
    pc_path = f'{save_dir}/processed/{symbol}_hf_point_clouds.npy'
    np.save(pc_path, np.array(pcs))
    idx_path = f'{save_dir}/processed/{symbol}_hf_window_indices.npy'
    np.save(idx_path, np.array(idx))

    return {'df': df, 'point_clouds': pcs, 'end_indices': idx,
            'feature_cols': feature_cols, 'scaler_params': scaler}


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    interval = sys.argv[3] if len(sys.argv) > 3 else '1h'
    res = run_hf_pipeline(symbol=symbol, days=days, interval=interval)
    print(f"\n[Done] {len(res['df'])} candles, {len(res['point_clouds'])} windows")
