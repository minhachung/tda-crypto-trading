"""
On-Chain Metrics Fetcher: Network activity + whale movement data.

Free sources:
- Glassnode API (requires free account, but no credit card)
- Messari API (free tier, no key required)
- CryptoQuant (emerging data provider)

For MVP: Synthetic on-chain metrics (Z-scored network stats).
For production: Replace with live Glassnode/Messari feeds.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import warnings

try:
    import requests
except ImportError:
    requests = None


class OnChainMetricsFetcher:
    """Fetch on-chain metrics from free data sources."""

    GLASSNODE_API = "https://api.glassnode.com/v1"
    MESSARI_API = "https://data.messari.io/api/v1"

    def __init__(self, symbol: str, days: int = 365):
        """
        Initialize on-chain fetcher.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Historical window
        """
        self.symbol = symbol.upper()
        self.days = days
        self.asset_slug = self._get_asset_slug()

    def _get_asset_slug(self) -> str:
        """Map symbol to API slug."""
        slug_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
            'ADA': 'cardano', 'DOT': 'polkadot', 'LINK': 'chainlink',
            'MATIC': 'polygon', 'AVAX': 'avalanche', 'XRP': 'xrp',
        }
        return slug_map.get(self.symbol, self.symbol.lower())

    def fetch_metrics(self) -> pd.DataFrame:
        """
        Fetch on-chain metrics.

        Returns:
            DataFrame with columns: timestamp, active_addresses, transaction_count,
                                    mvrv_ratio, exchange_inflow_pct, whale_supply_pct,
                                    developer_activity (+ Z-scored versions)
        """
        # Try Glassnode first; fall back to synthetic
        try:
            return self._fetch_glassnode()
        except Exception as e:
            print(f"  Glassnode fetch failed for {self.symbol}: {e}")
            return self._generate_synthetic_metrics()

    def _fetch_glassnode(self) -> pd.DataFrame:
        """Fetch from Glassnode API (requires API key)."""
        # Placeholder: Glassnode requires authentication
        # For MVP, we generate synthetic metrics instead
        raise NotImplementedError("Glassnode integration requires API key setup")

    def _generate_synthetic_metrics(self) -> pd.DataFrame:
        """
        Generate synthetic on-chain metrics for MVP.

        Returns:
            DataFrame with realistic but synthetic on-chain time series.
        """
        end_date = pd.Timestamp.utcnow()
        start_date = end_date - pd.Timedelta(days=self.days)
        timestamps = pd.date_range(start_date, end_date, freq='D')

        n = len(timestamps)

        # Synthetic time series: random walk + seasonal + trend
        active_addresses = np.cumsum(np.random.randn(n) * 100) + 1000 * (1 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 365))
        active_addresses = np.maximum(active_addresses, 100)  # Ensure positive

        transaction_count = np.cumsum(np.random.randn(n) * 500) + 5000 + 1000 * np.sin(np.arange(n) * 2 * np.pi / 365)
        transaction_count = np.maximum(transaction_count, 10)

        # MVRV ratio: mean-reversion (oscillates around 1.0)
        mvrv_ratio = 1.0 + 0.3 * np.sin(np.arange(n) * 2 * np.pi / 90) + np.random.randn(n) * 0.1
        mvrv_ratio = np.maximum(mvrv_ratio, 0.5)

        # Exchange inflow: 50% of time negative (accumulation), 50% positive (distribution)
        exchange_inflow = np.random.randn(n) * 5 + 2 * np.sin(np.arange(n) * 2 * np.pi / 180)

        # Whale supply: % of coins in wallets > $1M (inverse relationship with price)
        whale_supply = 30 + 5 * np.sin(np.arange(n) * 2 * np.pi / 120) + np.random.randn(n) * 2
        whale_supply = np.clip(whale_supply, 5, 60)

        # Developer activity: commits/week (more for ETH/DOT than others)
        base_commits = {'ETH': 150, 'DOT': 120}.get(self.symbol, 50)
        developer_activity = base_commits + np.cumsum(np.random.randn(n) * 5)
        developer_activity = np.maximum(developer_activity, 1)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'active_addresses': active_addresses,
            'transaction_count': transaction_count,
            'mvrv_ratio': mvrv_ratio,
            'exchange_inflow_pct': exchange_inflow,
            'whale_supply_pct': whale_supply,
            'developer_activity': developer_activity,
        })

        # Z-score all features
        for col in df.columns[1:]:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        return df

    def fetch_network_status(self) -> Dict[str, bool]:
        """Return network status (active, delisting risk, etc.)."""
        status = {
            'is_active': True,
            'has_staking': self.symbol in ['ETH', 'ADA', 'DOT', 'SOL'],
            'has_defi': self.symbol in ['ETH', 'LINK', 'MATIC'],
            'is_layer_2': self.symbol in ['MATIC', 'ARB', 'OPT'],
        }
        return status
