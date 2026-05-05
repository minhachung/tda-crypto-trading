#!/usr/bin/env python
"""
Strategy 1 Example: Persistent Homology Price-Volume Manifolds.

Runs the full TDA pipeline on BTC and visualizes:
  - Persistence diagrams
  - C1-norm time series
  - Trading signals overlaid on price
  - Equity curve vs buy-hold
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_pipeline import run_pipeline
from src.persistent_homology import (
    PersistentHomologyAnalyzer,
    compute_features_for_windows,
)
from src.trading_signals import TradingSignalGenerator
from src.backtester import Backtester, format_metrics


def run_strategy_1(symbol='BTC', days=365, c1_threshold_std=1.0):
    """Run Strategy 1 with visualization."""

    print(f"\n{'=' * 70}")
    print(f"Strategy 1: Persistent Homology Price-Volume Manifolds")
    print(f"Symbol: {symbol} | Days: {days}")
    print(f"{'=' * 70}\n")

    print("Step 1: Fetching and preprocessing data...")
    data = run_pipeline(symbol=symbol, days=days, window_size=10, stride=1)
    df = data['df']
    point_clouds = data['point_clouds']
    end_indices = data['end_indices']

    print(f"\nStep 2: Computing persistent homology...")
    features_df = compute_features_for_windows(point_clouds, end_indices=end_indices,
                                                verbose=False)
    print(f"  Features computed: {features_df.shape}")

    print(f"\nStep 3: Visualizing a sample persistence diagram...")
    sample_idx = len(point_clouds) // 2
    analyzer = PersistentHomologyAnalyzer(point_clouds[sample_idx])
    analyzer.compute()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if 'H1' in analyzer.diagrams and len(analyzer.diagrams['H1']) > 0:
        h1 = analyzer.diagrams['H1']
        h1_finite = h1[~np.isinf(h1[:, 1])]
        if len(h1_finite) > 0:
            axes[0].scatter(h1_finite[:, 0], h1_finite[:, 1], alpha=0.7, s=80, c='red',
                           label=f'H1 (loops): {len(h1_finite)}')
            mx = max(h1_finite.max(), 1.0)
            axes[0].plot([0, mx], [0, mx], 'k--', alpha=0.3)
    if 'H0' in analyzer.diagrams and len(analyzer.diagrams['H0']) > 0:
        h0 = analyzer.diagrams['H0']
        h0_finite = h0[~np.isinf(h0[:, 1])]
        if len(h0_finite) > 0:
            axes[0].scatter(h0_finite[:, 0], h0_finite[:, 1], alpha=0.5, s=40, c='blue',
                           label=f'H0 (components): {len(h0_finite)}')

    axes[0].set_xlabel('Birth')
    axes[0].set_ylabel('Death')
    axes[0].set_title(f'Persistence Diagram (window {sample_idx})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(features_df['H1_c1_norm'].values, 'b-', label='H1 C1-norm', linewidth=2)
    axes[1].plot(features_df['H1_l1_norm'].values, 'g--', label='H1 L1-norm', alpha=0.6)
    axes[1].plot(features_df['H1_entropy'].values, 'r:', label='H1 Entropy', alpha=0.6)
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('TDA Features Over Time')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/{symbol}_tda_visualization.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {save_path}")

    print(f"\nStep 4: Generating trading signals (threshold={c1_threshold_std}σ)...")
    generator = TradingSignalGenerator(
        c1_threshold_std=c1_threshold_std,
        c1_drop_threshold=-0.5,
        confidence_threshold=0.3,
        lookback_window=10,
    )
    signals_df = generator.generate_signals(features_df, price_series=df['close'].values)
    summary = generator.signal_summary(signals_df)
    print(f"  Buy: {summary['buy_signals']}, Sell: {summary['sell_signals']}, "
          f"Hold: {summary['hold_signals']}")

    print(f"\nStep 5: Backtesting...")
    if 'end_idx' in signals_df.columns:
        idx = signals_df['end_idx'].astype(int).values
        idx = idx[idx < len(df)]
        prices_aligned = df['close'].values[idx]
        signals_df = signals_df.iloc[:len(prices_aligned)].reset_index(drop=True)
    else:
        prices_aligned = df['close'].values[-len(signals_df):]

    backtester = Backtester(initial_capital=10000)
    results = backtester.run(prices_aligned, signals_df)
    print(format_metrics(results['metrics']))

    print(f"\nStep 6: Visualizing trades...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(prices_aligned, 'b-', label='Price', linewidth=1.5)
    buys = signals_df[signals_df['signal'] == 'BUY']
    sells = signals_df[signals_df['signal'] == 'SELL']

    if len(buys) > 0:
        axes[0].scatter(buys.index, prices_aligned[buys.index],
                       c='green', s=80, marker='^', label=f'Buy ({len(buys)})', zorder=5)
    if len(sells) > 0:
        axes[0].scatter(sells.index, prices_aligned[sells.index],
                       c='red', s=80, marker='v', label=f'Sell ({len(sells)})', zorder=5)

    axes[0].set_ylabel('Price (USD)')
    axes[0].set_title(f'{symbol} Price + TDA Trading Signals')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    bh_equity = 10000 * (prices_aligned / prices_aligned[0])
    axes[1].plot(results['equity_curve'].values, 'b-', label='TDA Strategy', linewidth=2)
    axes[1].plot(bh_equity, 'gray', label='Buy & Hold', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel('Equity (USD)')
    axes[1].set_title('Strategy vs Buy & Hold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/{symbol}_strategy1_results.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved strategy plot: {save_path}")

    return results


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    results = run_strategy_1(symbol=symbol, days=days)
