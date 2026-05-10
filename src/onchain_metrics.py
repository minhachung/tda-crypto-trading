"""
On-Chain Metrics Fetcher: REAL data from free APIs.

  - BTC: blockchain.info charts API (no auth, no signup, free)
         https://api.blockchain.info/charts/{metric}?timespan={N}days
  - ETH: Etherscan V2 (free tier) — requires ETHERSCAN_API_KEY env var.
         Free tier doesn't expose daily-aggregated history, so we walk
         block-by-day: getblocknobytime(midnight) → eth_getBlockByNumber
         → extract tx_count + gas_used + base_fee + gas_limit.
         2 API calls per day; 100k/day rate limit on free tier means we
         can cover any window we care about.

For altcoins (SOL, ADA, DOT, LINK, AVAX, …) no free historical on-chain
source exists. fetch_metrics returns an empty DataFrame so callers can
detect missing data instead of silently merging in synthetic noise.

Why this trade-off (BTC+ETH real > all-asset synthetic):
  Synthetic on-chain features bias every model that touches them — they
  look like signal in-sample (random walks have structure) but cannot
  generalise. Real-only-where-free is the honest version.
"""

from __future__ import annotations

import os
import time
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

# Etherscan V2: free tier. Block-by-day walk gives us daily aggregates.
ETHERSCAN_V2_BASE = "https://api.etherscan.io/v2/api"
ETHERSCAN_CHAINID = {'ETH': 1}


def is_real_onchain_supported(symbol: str) -> bool:
    """True iff a free real on-chain source exists for this symbol.

    - BTC: always supported (blockchain.info, no key needed)
    - ETH: supported iff ETHERSCAN_API_KEY is set in environment
    - everything else: not supported on free tier
    """
    s = symbol.upper()
    if s == 'BTC':
        return True
    if s == 'ETH':
        return bool(os.environ.get('ETHERSCAN_API_KEY'))
    return False


def _has_etherscan_key() -> bool:
    return bool(os.environ.get('ETHERSCAN_API_KEY'))


class OnChainMetricsFetcher:
    """Fetch real on-chain metrics from free APIs:
       BTC → blockchain.info (no auth)
       ETH → Etherscan V2 (free tier, requires ETHERSCAN_API_KEY env var)

    For unsupported assets returns an empty DataFrame so callers detect
    "no on-chain features available" rather than getting synthetic noise.
    """

    def __init__(self, symbol: str, days: int = 365, timeout: int = 30):
        self.symbol = symbol.upper()
        self.days = days
        self.timeout = timeout
        self.is_supported = is_real_onchain_supported(self.symbol)
        self._etherscan_key = os.environ.get('ETHERSCAN_API_KEY', '')

    def fetch_metrics(self) -> pd.DataFrame:
        """Fetch real on-chain metrics or return empty DataFrame."""
        if requests is None:
            warnings.warn("requests not installed")
            return self._empty_schema()
        if not self.is_supported:
            return self._empty_schema()
        try:
            if self.symbol == 'BTC':
                df = self._fetch_blockchain_info()
            elif self.symbol == 'ETH':
                df = self._fetch_etherscan_eth()
            else:
                return self._empty_schema()
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

    def _fetch_etherscan_eth(self) -> pd.DataFrame:
        """Walk Etherscan V2 block-by-day and aggregate free metrics.

        Two API calls per day (getblocknobytime + eth_getBlockByNumber).
        Etherscan free-tier rate limit: 5 calls/sec, 100k/day. We pause
        ~0.25s between paired calls to stay well under both limits.

        Per-day extracts:
          eth_transaction_count : len(block.transactions)
          eth_gas_used          : decimal of block.gasUsed (wei → gas units)
          eth_base_fee_gwei     : block.baseFeePerGas in gwei (post-EIP-1559;
                                  null pre-merge)
          eth_gas_limit         : block.gasLimit
        """
        if not self._etherscan_key:
            return pd.DataFrame()

        end_dt = pd.Timestamp.utcnow().floor('D')
        start_dt = end_dt - pd.Timedelta(days=self.days)
        # Use midnight UTC of each day as the query timestamp
        days = pd.date_range(start_dt, end_dt, freq='D', inclusive='left')

        rows = []
        chainid = ETHERSCAN_CHAINID['ETH']

        for i, day in enumerate(days):
            ts = int(day.timestamp())
            block_n = self._etherscan_block_at(chainid, ts)
            if block_n is None:
                continue
            block = self._etherscan_block(chainid, block_n)
            if block is None:
                continue

            try:
                tx_count = len(block.get('transactions', []))
                gas_used = int(block.get('gasUsed', '0x0'), 16)
                gas_limit = int(block.get('gasLimit', '0x0'), 16)
                base_fee_hex = block.get('baseFeePerGas')
                base_fee_gwei = (int(base_fee_hex, 16) / 1e9
                                  if base_fee_hex else float('nan'))
            except (ValueError, TypeError):
                continue

            rows.append({
                'timestamp': day,
                'eth_transaction_count': float(tx_count),
                'eth_gas_used': float(gas_used),
                'eth_gas_limit': float(gas_limit),
                'eth_base_fee_gwei': base_fee_gwei,
            })
            if (i + 1) % 100 == 0:
                print(f"    [etherscan ETH] {i + 1}/{len(days)} days fetched")

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

    def _etherscan_block_at(self, chainid: int, ts: int) -> Optional[int]:
        """Get block number closest to (and before) a unix timestamp."""
        params = {
            'chainid': chainid,
            'module': 'block',
            'action': 'getblocknobytime',
            'timestamp': ts,
            'closest': 'before',
            'apikey': self._etherscan_key,
        }
        try:
            r = requests.get(ETHERSCAN_V2_BASE, params=params,
                              timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
            time.sleep(0.22)  # rate-limit pad (≤5 req/sec)
            if str(payload.get('status')) != '1':
                return None
            return int(payload['result'])
        except Exception:
            return None

    def _etherscan_block(self, chainid: int, block_n: int) -> Optional[Dict]:
        """Fetch a single block via the JSON-RPC proxy. boolean=false returns
        transaction hashes only (lighter); we use len() for tx count."""
        params = {
            'chainid': chainid,
            'module': 'proxy',
            'action': 'eth_getBlockByNumber',
            'tag': hex(block_n),
            'boolean': 'false',
            'apikey': self._etherscan_key,
        }
        try:
            r = requests.get(ETHERSCAN_V2_BASE, params=params,
                              timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
            time.sleep(0.22)  # rate-limit pad
            return payload.get('result')
        except Exception:
            return None

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
        """Empty DataFrame with a unified BTC+ETH column schema."""
        btc_cols = list(BLOCKCHAIN_INFO_METRICS.values())
        eth_cols = ['eth_transaction_count', 'eth_gas_used',
                    'eth_gas_limit', 'eth_base_fee_gwei']
        cols = ['timestamp'] + btc_cols + eth_cols
        cols += [f'{c}_z' for c in btc_cols + eth_cols]
        return pd.DataFrame(columns=cols)
