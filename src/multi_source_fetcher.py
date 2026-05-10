"""
Multi-Source Data Fetcher: Unifies Coinbase (high-frequency) + CMC (completeness).

Strategy:
- Use Coinbase for recent 3 years (hourly, high-frequency ready)
- Use CMC fallback for delisted/historical breadth
- Merge on timestamp, flag source for each row
- Add on-chain metrics via OnChainMetricsFetcher
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import warnings

from coinmarketcap_data import CMCHistoryFetcher
from onchain_metrics import OnChainMetricsFetcher


class MultiSourceDataFetcher:
    """Fetch OHLCV from multiple sources with automatic fallback."""

    def __init__(self, symbol: str, days: int = 365, use_coinbase: bool = True, use_onchain: bool = True):
        """
        Initialize multi-source fetcher.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Historical window
            use_coinbase: Fetch from Coinbase (primary source)
            use_onchain: Attach on-chain metrics
        """
        self.symbol = symbol.upper()
        self.days = days
        self.use_coinbase = use_coinbase
        self.use_onchain = use_onchain

    def fetch(self, force_source: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV with automatic source selection and merging.

        Args:
            force_source: Force specific source ('coinbase', 'cmc', None for auto)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
                                    is_delisted, delisting_date, source,
                                    + on-chain metrics if use_onchain=True
        """
        print(f"\n📊 Fetching {self.symbol} ({self.days} days)")

        if force_source == 'coinbase' or (force_source is None and self.use_coinbase):
            try:
                df = self._fetch_coinbase()
                if not df.empty:
                    print(f"  ✓ Coinbase: {len(df)} candles")
                    return self._add_onchain_metrics(df)
            except Exception as e:
                print(f"  ✗ Coinbase failed: {e}")

        if force_source == 'cmc' or force_source is None:
            try:
                df = self._fetch_cmc()
                if not df.empty:
                    print(f"  ✓ CMC: {len(df)} candles")
                    return self._add_onchain_metrics(df)
            except Exception as e:
                print(f"  ✗ CMC failed: {e}")

        # All sources failed
        print(f"  ✗ All sources failed for {self.symbol}")
        return self._empty_dataframe()

    def _fetch_coinbase(self) -> pd.DataFrame:
        """Fetch from Coinbase (high-frequency, recent data)."""
        try:
            from binance_data import HighFreqFetcher
            fetcher = HighFreqFetcher(self.symbol, interval='1h')
            df = fetcher.fetch_history(days=self.days)

            # Add metadata columns
            df['is_delisted'] = False
            df['delisting_date'] = None
            df['source'] = 'coinbase'

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'is_delisted', 'delisting_date', 'source']]

        except Exception as e:
            raise RuntimeError(f"Coinbase fetch failed: {e}")

    def _fetch_cmc(self) -> pd.DataFrame:
        """Fetch from CoinMarketCap (includes delisted coins)."""
        try:
            fetcher = CMCHistoryFetcher(self.symbol, days=self.days)
            df = fetcher.fetch_history()

            # Add delisting metadata
            is_delisted, delisting_date = CMCHistoryFetcher.get_delisting_metadata(self.symbol)
            if is_delisted:
                df['is_delisted'] = True
                df['delisting_date'] = delisting_date

            return df

        except Exception as e:
            raise RuntimeError(f"CMC fetch failed: {e}")

    def _merge_sources(self, coinbase_df: pd.DataFrame, cmc_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Coinbase + CMC data."""
        # Align on timestamp
        coinbase_df = coinbase_df.set_index('timestamp')
        cmc_df = cmc_df.set_index('timestamp')

        # Coinbase takes precedence (higher quality); CMC fills gaps
        merged = coinbase_df.copy()

        # Fill missing rows from CMC
        missing_idx = cmc_df.index.difference(merged.index)
        if len(missing_idx) > 0:
            merged = pd.concat([merged, cmc_df.loc[missing_idx]])
            merged = merged.sort_index()

        merged = merged.reset_index()

        # Reconciliation check
        row_divergence = (merged['close_coinbase'] - merged['close_cmc']).abs() / merged['close_cmc']
        high_div = (row_divergence > 0.05).sum()
        if high_div > 0:
            print(f"  ⚠ Warning: {high_div} rows with >5% price divergence between sources")

        return merged

    def _add_onchain_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach on-chain metrics if enabled."""
        if not self.use_onchain:
            return df

        try:
            onchain = OnChainMetricsFetcher(self.symbol, days=self.days)
            onchain_df = onchain.fetch_metrics()

            # Normalize timezones to avoid tz-aware vs tz-naive merge issues
            df_ts = pd.to_datetime(df['timestamp']).dt.tz_localize(None) if pd.to_datetime(df['timestamp']).dt.tz is not None else pd.to_datetime(df['timestamp'])
            onchain_ts = pd.to_datetime(onchain_df['timestamp']).dt.tz_localize(None) if pd.to_datetime(onchain_df['timestamp']).dt.tz is not None else pd.to_datetime(onchain_df['timestamp'])

            # Floor to date (datetime64 type, not object) for merging
            df = df.copy()
            df['_merge_date'] = df_ts.dt.floor('D')
            onchain_df = onchain_df.copy()
            onchain_df['_merge_date'] = onchain_ts.dt.floor('D')

            onchain_cols = ['active_addresses', 'transaction_count', 'mvrv_ratio',
                            'exchange_inflow_pct', 'whale_supply_pct', 'developer_activity',
                            'active_addresses_z', 'transaction_count_z', 'mvrv_ratio_z',
                            'exchange_inflow_pct_z', 'whale_supply_pct_z', 'developer_activity_z']
            existing_oc = [c for c in onchain_cols if c in onchain_df.columns]

            merged = pd.merge_asof(
                df.sort_values('_merge_date'),
                onchain_df[['_merge_date'] + existing_oc].drop_duplicates('_merge_date').sort_values('_merge_date'),
                on='_merge_date',
                direction='backward'
            )

            merged = merged.drop('_merge_date', axis=1)
            return merged.sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            print(f"  ⚠ On-chain metrics skipped: {e}")
            return df

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'is_delisted', 'delisting_date', 'source']
        return pd.DataFrame(columns=columns)
