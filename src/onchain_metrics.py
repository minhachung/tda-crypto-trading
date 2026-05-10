"""
On-Chain Metrics Fetcher: REAL data from blockchain.info charts API (free).

blockchain.info exposes a no-auth, free public charts API for Bitcoin
network metrics. ETH/altcoin coverage is NOT available without a paid
provider, so this module honestly returns empty data for non-BTC assets
(callers detect via `len(df) == 0` and treat the asset as having no
on-chain features rather than fabricating synthetic ones).

Endpoint:  https://api.blockchain.info/charts/{metric}?timespan={N}days&format=json
Metrics:   n-transactions, hash-rate, mempool-size, avg-block-size,
           transaction-fees-usd, miners-revenue
Coverage:  BTC only (free)

Why this trade-off (BTC-only real > all-asset synthetic):
  - The user explicitly said "no paid services"
  - Synthetic on-chain features bias every model that touches them — they
    look like signal in-sample (random walks have structure) but cannot
    generalise. BTC-real + everything-else-empty is the honest version.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# blockchain.info charts: free, no auth, BTC only.
BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info/charts"
BLOCKCHAIN_INFO_METRICS = {
    'n-transactions':       'transaction_count',
    'hash-rate':            'hash_rate',
    'mempool-size':         'mempool_size',
    'avg-block-size':       'mean_block_size',
    'transaction-fees-usd': 'total_fees_usd',
    'miners-revenue':       'miners_revenue',
}


def is_real_onchain_supported(symbol: str) -> bool:
    """True iff a free, no-auth real on-chain source exists for this symbol.

    Currently BTC only via blockchain.info. ETH would need Etherscan with
    a free API key (sign-up required); altcoins need paid providers.
    """
    return symbol.upper() == 'BTC'


class OnChainMetricsFetcher:
    """Fetch real on-chain metrics from blockchain.info (BTC, free, no auth).

    For non-BTC assets returns an empty DataFrame so callers detect
    "no on-chain features available" rather than getting synthetic noise.
    """

    def __init__(self, symbol: str, days: int = 365, timeout: int = 30):
        self.symbol = symbol.upper()
        self.days = days
        self.timeout = timeout
        self.is_supported = is_real_onchain_supported(self.symbol)

    def fetch_metrics(self) -> pd.DataFrame:
        """Fetch real on-chain metrics or return empty DataFrame."""
        if requests is None:
            warnings.warn("requests not installed; cannot fetch blockchain.info")
            return self._empty_schema()
        if not self.is_supported:
            return self._empty_schema()
        try:
            df = self._fetch_blockchain_info()
            if df.empty:
                return self._empty_schema()
            return self._z_score(df)
        except Exception as e:
            print(f"  [onchain {self.symbol}] fetch failed: {e}")
            return self._empty_schema()

    def fetch_network_status(self) -> Dict[str, bool]:
        """Metadata flags about this asset."""
        return {
            'is_real_onchain_supported': self.is_supported,
            'has_staking': self.symbol in {'ETH', 'ADA', 'DOT', 'SOL'},
            'has_defi': self.symbol in {'ETH', 'LINK', 'MATIC'},
            'is_layer_2': self.symbol in {'MATIC', 'ARB', 'OPT'},
        }

    # --- internals ---

    def _fetch_blockchain_info(self) -> pd.DataFrame:
        """Hit blockchain.info /charts endpoints and merge into one DataFrame."""
        timespan = f"{self.days}days"
        per_metric = []
        for metric_slug, friendly_name in BLOCKCHAIN_INFO_METRICS.items():
            url = f"{BLOCKCHAIN_INFO_BASE}/{metric_slug}"
            try:
                r = requests.get(url, params={'timespan': timespan,
                                              'format': 'json'},
                                  timeout=self.timeout,
                                  headers={'User-Agent': 'tda-crypto-trading-research/1.0'})
                r.raise_for_status()
                payload = r.json()
            except Exception as e:
                print(f"    [onchain] {metric_slug} unavailable: {e}")
                continue

            values = payload.get('values', [])
            if not values:
                continue
            metric_df = pd.DataFrame(values).rename(columns={'x': 'unix',
                                                             'y': friendly_name})
            metric_df['timestamp'] = pd.to_datetime(metric_df['unix'],
                                                     unit='s').dt.floor('D')
            metric_df = metric_df[['timestamp', friendly_name]]
            metric_df = metric_df.drop_duplicates(subset='timestamp')
            per_metric.append(metric_df)

        if not per_metric:
            return pd.DataFrame()

        # Outer-join all metrics on timestamp
        merged = per_metric[0]
        for m in per_metric[1:]:
            merged = pd.merge(merged, m, on='timestamp', how='outer')
        return merged.sort_values('timestamp').reset_index(drop=True)

    def _z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `*_z` columns (rolling 60-day z-score, causal — no lookahead)."""
        out = df.copy()
        for col in df.columns:
            if col == 'timestamp':
                continue
            roll_mean = out[col].rolling(window=60, min_periods=10).mean()
            roll_std = out[col].rolling(window=60, min_periods=10).std()
            out[f'{col}_z'] = (out[col] - roll_mean) / (roll_std + 1e-8)
        return out

    def _empty_schema(self) -> pd.DataFrame:
        """Empty DataFrame with the same column schema as a successful fetch."""
        cols = ['timestamp'] + list(BLOCKCHAIN_INFO_METRICS.values())
        cols += [f'{c}_z' for c in BLOCKCHAIN_INFO_METRICS.values()]
        return pd.DataFrame(columns=cols)
