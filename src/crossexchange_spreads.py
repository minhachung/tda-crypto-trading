"""
Cross-Exchange Spread Fetcher: REAL data from Coinbase + Kraken (free).

Both exchanges expose free, no-auth historical OHLC endpoints. We fetch
matching daily candles from each, then compute:

  spread_pct       = (kraken_close - coinbase_close) / coinbase_close
  spread_zscore_60 = causal 60-day z-score of spread_pct

These are REAL arbitrage-pressure features, not synthetic. They reflect
genuine fragmentation between exchanges and have been used in academic
crypto microstructure research.

Coverage: any asset listed on both Coinbase + Kraken (BTC/ETH/SOL/ADA/etc.).
For assets only on one venue, returns an EMPTY DataFrame (caller can detect
missing data instead of falling back to synthetic).
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


COINBASE_PRODUCTS = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD', 'ADA': 'ADA-USD',
    'DOT': 'DOT-USD', 'LINK': 'LINK-USD', 'AVAX': 'AVAX-USD',
    'XRP': 'XRP-USD', 'DOGE': 'DOGE-USD', 'LTC': 'LTC-USD',
    'BCH': 'BCH-USD', 'ATOM': 'ATOM-USD', 'MATIC': 'MATIC-USD',
}

KRAKEN_PAIRS = {
    'BTC': 'XXBTZUSD', 'ETH': 'XETHZUSD', 'SOL': 'SOLUSD', 'ADA': 'ADAUSD',
    'DOT': 'DOTUSD', 'LINK': 'LINKUSD', 'AVAX': 'AVAXUSD',
    'XRP': 'XXRPZUSD', 'DOGE': 'XDGUSD', 'LTC': 'XLTCZUSD',
    'BCH': 'BCHUSD', 'ATOM': 'ATOMUSD', 'MATIC': 'MATICUSD',
}

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"
KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"


def is_cross_exchange_supported(symbol: str) -> bool:
    s = symbol.upper()
    return s in COINBASE_PRODUCTS and s in KRAKEN_PAIRS


class CrossExchangeSpreadFetcher:
    """Compute real Coinbase-vs-Kraken close-price spreads."""

    def __init__(self, symbol: str, days: int = 365, timeout: int = 30):
        self.symbol = symbol.upper()
        self.days = days
        self.timeout = timeout
        self.is_supported = is_cross_exchange_supported(self.symbol)

    def fetch_spreads(self) -> pd.DataFrame:
        """Return DataFrame with timestamp, spread_pct, spread_zscore_60."""
        if requests is None:
            warnings.warn("requests not installed")
            return self._empty_schema()
        if not self.is_supported:
            return self._empty_schema()

        try:
            cb = self._fetch_coinbase_daily()
            time.sleep(0.5)  # be polite
            kr = self._fetch_kraken_daily()
            if cb.empty or kr.empty:
                return self._empty_schema()
            return self._compute_spreads(cb, kr)
        except Exception as e:
            print(f"  [spread {self.symbol}] fetch failed: {e}")
            return self._empty_schema()

    # --- internals ---

    def _fetch_coinbase_daily(self) -> pd.DataFrame:
        product = COINBASE_PRODUCTS[self.symbol]
        end = pd.Timestamp.utcnow().floor('D')
        start = end - pd.Timedelta(days=self.days)

        # Coinbase: max 300 candles per call. Daily granularity = 86400 s.
        # 300 days per chunk.
        chunks = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + pd.Timedelta(days=300), end)
            params = {
                'granularity': 86400,
                'start': cursor.isoformat(),
                'end': chunk_end.isoformat(),
            }
            r = requests.get(COINBASE_CANDLES.format(product=product),
                             params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if data:
                chunks.extend(data)
            cursor = chunk_end
            time.sleep(0.3)

        if not chunks:
            return pd.DataFrame()

        df = pd.DataFrame(chunks, columns=['time', 'low', 'high', 'open',
                                            'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s').dt.floor('D')
        df['close'] = df['close'].astype(float)
        df = df.drop_duplicates(subset='timestamp')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df[['timestamp', 'close']].rename(columns={'close': 'cb_close'})

    def _fetch_kraken_daily(self) -> pd.DataFrame:
        pair = KRAKEN_PAIRS[self.symbol]
        # Kraken OHLC: interval=1440 (minutes per day), `since` is unix seconds.
        # Returns up to 720 candles per call. We just take the most recent
        # `days` worth, which fits in a single call for days <= 720.
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=self.days)).timestamp())
        params = {'pair': pair, 'interval': 1440, 'since': since}

        r = requests.get(KRAKEN_OHLC, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if payload.get('error'):
            raise RuntimeError(f"Kraken API error: {payload['error']}")

        result = payload.get('result', {})
        # Kraken returns the data under the resolved pair key, which may differ
        # from what we requested (e.g. "XXBTZUSD" → "XBTUSD"). Find the data key.
        data_key = next((k for k in result if k != 'last'), None)
        if data_key is None:
            return pd.DataFrame()

        rows = result[data_key]
        if not rows:
            return pd.DataFrame()

        # [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low',
                                          'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s').dt.floor('D')
        df['close'] = df['close'].astype(float)
        df = df.drop_duplicates(subset='timestamp')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df[['timestamp', 'close']].rename(columns={'close': 'kr_close'})

    def _compute_spreads(self, cb: pd.DataFrame, kr: pd.DataFrame) -> pd.DataFrame:
        """Inner-join on date, compute pct spread + causal rolling z-score."""
        merged = pd.merge(cb, kr, on='timestamp', how='inner')
        if merged.empty:
            return self._empty_schema()

        merged['spread_pct'] = (merged['kr_close'] - merged['cb_close']) / merged['cb_close']

        # Causal 60-day z-score
        roll_mean = merged['spread_pct'].rolling(window=60, min_periods=10).mean()
        roll_std = merged['spread_pct'].rolling(window=60, min_periods=10).std()
        merged['spread_zscore_60'] = (merged['spread_pct'] - roll_mean) / (roll_std + 1e-8)

        # Persistence: fraction of last 10 days where spread > 0
        merged['spread_persistence'] = (
            (merged['spread_pct'] > 0).astype(int)
            .rolling(window=10, min_periods=3).mean()
        )

        return merged[['timestamp', 'spread_pct', 'spread_zscore_60',
                       'spread_persistence']]

    def _empty_schema(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['timestamp', 'spread_pct',
                                      'spread_zscore_60', 'spread_persistence'])
