#!/usr/bin/env python
"""
TDA Crypto Trading - Full Pipeline Runner

Runs the complete pipeline end-to-end:
  1. Fetch OHLCV data from CoinGecko
  2. Compute technical indicators
  3. Build sliding-window point clouds
  4. Compute persistent homology + extract features
  5. Generate trading signals
  6. Backtest the strategy
  7. Print performance metrics

Usage:
  python main.py                 # Default: BTC, 365 days
  python main.py ETH             # Ethereum
  python main.py BTC 730         # Bitcoin, 2 years
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import run_pipeline as run_data_pipeline
from src.persistent_homology import run_persistence_pipeline
from src.trading_signals import run_signal_pipeline
from src.backtester import run_backtest, format_metrics


def run_full_pipeline(symbol='BTC', days=365, window_size=10, stride=1,
                     c1_threshold_std=1.5, save_dir='data'):
    """Execute the full TDA trading pipeline."""

    print(f"\n{'#' * 70}")
    print(f"#   TDA-Based Crypto Trading System")
    print(f"#   Symbol: {symbol} | Days: {days} | Window: {window_size}")
    print(f"{'#' * 70}\n")

    t_start = time.time()

    print(f"\n[1/4] Data Pipeline")
    print(f"-" * 70)
    data_result = run_data_pipeline(
        symbol=symbol, days=days, window_size=window_size, stride=stride,
        save_dir=save_dir,
    )
    t1 = time.time()
    print(f"   Time: {t1 - t_start:.1f}s")

    print(f"\n[2/4] Persistent Homology")
    print(f"-" * 70)
    features_df = run_persistence_pipeline(symbol=symbol, save_dir=save_dir)
    t2 = time.time()
    print(f"   Time: {t2 - t1:.1f}s")

    print(f"\n[3/4] Trading Signals")
    print(f"-" * 70)
    signals_df = run_signal_pipeline(
        symbol=symbol, save_dir=save_dir,
        c1_threshold_std=c1_threshold_std,
    )
    t3 = time.time()
    print(f"   Time: {t3 - t2:.1f}s")

    print(f"\n[4/4] Backtest")
    print(f"-" * 70)
    results = run_backtest(symbol=symbol, save_dir=save_dir)
    t4 = time.time()
    print(f"   Time: {t4 - t3:.1f}s")

    print(f"\n{'#' * 70}")
    print(f"#   Pipeline Complete | Total time: {t4 - t_start:.1f}s")
    print(f"{'#' * 70}\n")

    return {
        'data': data_result,
        'features': features_df,
        'signals': signals_df,
        'backtest': results,
    }


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    try:
        results = run_full_pipeline(symbol=symbol, days=days)
        return 0
    except KeyboardInterrupt:
        print("\n\n[Aborted] Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
