#!/usr/bin/env python
"""
Validation v2 Runner: K-fold CV + Grid Search + Coinbase HF Data.

Uses high-frequency hourly data from Coinbase (24x more samples than CoinGecko)
and runs a 5-fold time-series cross-validation with grid search.

Usage:
  python examples/run_validation_v2.py BTC 365
  python examples/run_validation_v2.py ETH 180
  python examples/run_validation_v2.py BTC 90 1h    # 90 days hourly
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.binance_data import run_hf_pipeline
from src.persistent_homology import compute_features_for_windows
from src.validation_v2 import (
    run_validation_v2,
    format_v2_report,
)


def run_v2(symbol='BTC', days=365, interval='1h', cross_assets=None,
           n_splits=5, save_report=True):
    """Full v2 validation pipeline."""

    if cross_assets is None:
        cross_assets = ['ETH'] if symbol != 'ETH' else ['BTC']

    print(f"\n{'#' * 70}")
    print(f"#  TDA Strategy Validation V2")
    print(f"#  Symbol: {symbol} | Days: {days} | Interval: {interval}")
    print(f"#  K-Fold CV: {n_splits} folds | Cross-asset: {cross_assets}")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/4] Fetching high-frequency data ({symbol})...")
    data = run_hf_pipeline(symbol=symbol, days=days, interval=interval,
                           window_size=20, stride=1)
    df = data['df']
    point_clouds = data['point_clouds']
    end_indices = data['end_indices']

    print(f"\n[2/4] Computing TDA features ({len(point_clouds)} windows)...")
    features_df = compute_features_for_windows(
        point_clouds, end_indices=end_indices, verbose=False
    )

    if 'end_idx' in features_df.columns:
        idx = features_df['end_idx'].astype(int).values
        idx = idx[idx < len(df)]
        prices_aligned = df['close'].values[idx]
        features_df = features_df.iloc[:len(prices_aligned)].reset_index(drop=True)
    else:
        prices_aligned = df['close'].values[-len(features_df):]

    print(f"  Features: {len(features_df)} | Prices: {len(prices_aligned)}")

    if len(features_df) < 100:
        print(f"  [Warning] Only {len(features_df)} feature windows - may not be enough for k-fold")
        print(f"  [Recommendation] Increase --days or use shorter --interval")

    print(f"\n[3/4] Running V2 validation (grid search + k-fold)...")
    primary_results = run_validation_v2(
        features_df, prices_aligned, n_splits=n_splits, symbol=symbol, verbose=True
    )

    cross_results = {}
    if cross_assets and 'error' not in primary_results:
        print(f"\n[4/4] Cross-asset validation...")
        for sym in cross_assets:
            print(f"\n  --- {sym} ---")
            try:
                ca_data = run_hf_pipeline(symbol=sym, days=days, interval=interval,
                                          window_size=20, stride=1)
                ca_features = compute_features_for_windows(
                    ca_data['point_clouds'], end_indices=ca_data['end_indices'],
                    verbose=False,
                )
                if 'end_idx' in ca_features.columns:
                    idx = ca_features['end_idx'].astype(int).values
                    idx = idx[idx < len(ca_data['df'])]
                    ca_prices = ca_data['df']['close'].values[idx]
                    ca_features = ca_features.iloc[:len(ca_prices)].reset_index(drop=True)
                else:
                    ca_prices = ca_data['df']['close'].values[-len(ca_features):]

                from src.validation_v2 import kfold_validate
                ca_kfold = kfold_validate(
                    ca_features, ca_prices, primary_results['best_params'],
                    n_splits=n_splits,
                )
                cross_results[sym] = {'kfold': ca_kfold}
            except Exception as e:
                print(f"  Error on {sym}: {e}")
                cross_results[sym] = {'error': str(e)}

    elapsed = time.time() - t0
    print(f"\n[Done] Total time: {elapsed:.1f}s")

    print(f"\n[Report] Generating validation report v2...")
    report = format_v2_report(primary_results, cross_asset=cross_results)

    if save_report:
        with open('VALIDATION_REPORT_V2.md', 'w') as f:
            f.write(report)
        print(f"  Saved: VALIDATION_REPORT_V2.md")

    print(f"\n{'#' * 70}")
    print(f"#  Validation V2 Complete")
    print(f"{'#' * 70}\n")

    save_v2_plots(primary_results, symbol)

    print(report[:4000])
    print(f"\n... [see VALIDATION_REPORT_V2.md for full report]")

    return primary_results, cross_results


def save_v2_plots(results, symbol):
    """Save k-fold validation plots."""
    if 'error' in results:
        return

    os.makedirs('data/persistence', exist_ok=True)

    fold_df = results['kfold']['fold_results']
    agg = results['kfold']['aggregate']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = fold_df['fold'].values
    width = 0.35
    axes[0, 0].bar(x - width/2, fold_df['tda_return_pct'].values, width,
                   label='TDA', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, fold_df['buy_hold_return_pct'].values, width,
                   label='Buy & Hold', color='gray', alpha=0.8)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].set_title(f'Per-Fold Returns: TDA vs Buy & Hold ({symbol})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    accs = fold_df['direction_accuracy'].values
    colors = ['green' if a > 0.5 else 'red' for a in accs]
    axes[0, 1].bar(x, accs, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', label='Chance (50%)')
    axes[0, 1].axhline(y=accs.mean(), color='blue', linestyle='-',
                        label=f'Mean = {accs.mean():.2%}')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Direction Accuracy')
    axes[0, 1].set_title('Direction Accuracy by Fold')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    sharpes = fold_df['tda_sharpe'].values
    axes[1, 0].bar(x, sharpes, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axhline(y=sharpes.mean(), color='red', linestyle='--',
                       label=f'Mean = {sharpes.mean():.2f}')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('Sharpe Ratio by Fold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    bs = results['bootstrap_sharpe']
    axes[1, 1].errorbar([0], [bs['mean']],
                       yerr=[[bs['mean'] - bs['lower']], [bs['upper'] - bs['mean']]],
                       fmt='o', markersize=14, capsize=12, capthick=2,
                       color='blue',
                       label=f"95% CI: [{bs['lower']:.2f}, {bs['upper']:.2f}]")
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title('Bootstrap Sharpe Ratio (95% CI)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/{symbol}_validation_v2_plots.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots: {save_path}")


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    interval = sys.argv[3] if len(sys.argv) > 3 else '1h'

    cross = ['ETH', 'SOL'] if symbol == 'BTC' else ['BTC']
    run_v2(symbol=symbol, days=days, interval=interval, cross_assets=cross[:1])
