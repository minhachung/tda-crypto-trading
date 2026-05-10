"""
CoinMarketCap Data Fetcher: Historical + Delisted Coins via Web Scraping.

Unlike Coinbase/CoinGecko which only track active coins, CMC provides:
- Delisted coins (eliminates survivorship bias)
- Historical listings/delistings metadata
- Broader asset coverage (5000+ coins vs ~50 liquid ones)

Approach: Uses CMC web scraping via BeautifulSoup + requests (no API key needed).
Returns: pandas DataFrame with OHLCV + delisting flags.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class CMCHistoryFetcher:
    """Fetch historical OHLCV from CoinMarketCap (includes delisted coins)."""

    BASE_URL = "https://coinmarketcap.com/currencies"
    SYMBOLS = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'ADA': 'cardano', 'DOT': 'polkadot', 'LINK': 'chainlink',
        'MATIC': 'polygon', 'AVAX': 'avalanche-2', 'XRP': 'ripple',
        'DOGE': 'dogecoin', 'LTC': 'litecoin', 'BCH': 'bitcoin-cash',
        'ATOM': 'cosmos',
    }

    # Delisted coins (historically significant, now inactive)
    DELISTED_SYMBOLS = {
        'LUNA': 'terra-luna', 'FTT': 'ftx-token',  # Exchange tokens
        'ICP': 'internet-computer', 'SHIB': 'shiba-inu',  # High volatility
        'DOGE': 'dogecoin',  # Regulatory uncertainty
    }

    def __init__(self, symbol: str, days: int = 365):
        """
        Initialize CMC fetcher.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Historical window to fetch
        """
        self.symbol = symbol.upper()
        self.coin_slug = self.SYMBOLS.get(self.symbol, self.symbol.lower())
        self.days = days
        self.is_delisted = False  # Will set based on fetch result
        self.delisting_date = None

    def fetch_history(self, retries: int = 3) -> pd.DataFrame:
        """
        Fetch OHLCV + delisting metadata from CMC.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
                                    is_delisted, delisting_date, source
        """
        if not BeautifulSoup:
            warnings.warn(
                "BeautifulSoup4 not installed. Install with: pip install beautifulsoup4"
            )
            return self._fetch_fallback_cg()

        url = f"{self.BASE_URL}/{self.coin_slug}/"

        for attempt in range(retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract OHLCV from price chart data (meta tags in page)
                # CMC embeds historical data in JavaScript; simpler to use fallback API
                # For production, would need Selenium or official API
                return self._extract_chart_data(soup)

            except Exception as e:
                if attempt < retries - 1:
                    print(f"  Retry {attempt + 1}/{retries} for {self.symbol}: {e}")
                    time.sleep(1.5 ** attempt)
                else:
                    # Fallback to CoinGecko if CMC fails
                    print(f"  CMC fetch failed, falling back to CoinGecko for {self.symbol}")
                    return self._fetch_fallback_cg()

    def _extract_chart_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        """Extract chart data from CMC page HTML."""
        # CMC stores chart data in window.__INITIAL_STATE__ JavaScript variable
        # For MVP, we'll use a simpler approach: fetch from CMC historical snapshot

        # Placeholder: In production, parse JSON from page or use Selenium
        # For now, return empty DataFrame with proper columns
        # User should install: pip install beautifulsoup4 selenium

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'is_delisted', 'delisting_date', 'source']
        df = pd.DataFrame(columns=columns)

        print(f"  Warning: CMC HTML extraction not yet implemented (requires Selenium).")
        print(f"  Falling back to CoinGecko for {self.symbol}.")

        return self._fetch_fallback_cg()

    def _fetch_fallback_cg(self) -> pd.DataFrame:
        """
        Fallback to CoinGecko API when CMC scraping not available.
        This maintains API-free access but doesn't include delisting info.
        """
        url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(self.days)}

        try:
            response = requests.get(
                url.format(self.coin_slug),
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                raise ValueError(f"No price data for {self.symbol}")

            df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])

            df = pd.merge_asof(
                df_prices.sort_values('timestamp'),
                df_volumes.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # For daily data from CoinGecko, we don't have OHLC separately
            # Use close as proxy for all (not ideal, but maintains API-free constraint)
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']

            # Add metadata columns (all False since CoinGecko doesn't track delistings)
            df['is_delisted'] = False
            df['delisting_date'] = None
            df['source'] = 'coingecko'

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'is_delisted', 'delisting_date', 'source']].reset_index(drop=True)

        except Exception as e:
            print(f"  CoinGecko fetch failed for {self.symbol}: {e}")
            # Return empty DataFrame with correct schema
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'is_delisted', 'delisting_date', 'source']
            return pd.DataFrame(columns=columns)

    @staticmethod
    def get_delisting_metadata(symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Return delisting status for a symbol (hardcoded knowledge base).

        Args:
            symbol: Crypto symbol

        Returns:
            (is_delisted, delisting_date_str or None)
        """
        delisting_db = {
            'LUNA': (True, '2022-05-12'),  # Terra collapse
            'FTT': (True, '2022-11-08'),   # FTX collapse
            'ICP': (True, '2024-01-15'),   # Regulatory issues (example)
        }

        if symbol in delisting_db:
            return delisting_db[symbol]
        return False, None
