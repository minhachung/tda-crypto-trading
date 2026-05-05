#!/usr/bin/env python
"""
Full validation pipeline runner.

Produces VALIDATION_REPORT.md with:
  - Walk-forward out-of-sample test
  - Direction accuracy
  - Bootstrap CI for Sharpe
  - Statistical significance tests
  - Baseline comparisons
  - Cross-asset robustness check
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_pipeline import run_pipeline
from src.persistent_homology import compute_features_for_windows
from src.validation import (
    walk_forward_validate,
    cross_asset_validation,
    baseline_buy_hold,
    baseline_random,
    baseline_ma_crossover,
    direction_accuracy,
    format_validation_report,
)


def run_full_validation(symbol='BTC', days=365, cross_assets=None, save_report=True):
    """Run full validation suite and save report."""

    if cross_assets is None:
        cross_assets = ['ETH', 'SOL']

    print(f"\n{'#' * 70}")
    print(f"#  TDA Strategy Validation")
    print(f"#  Primary: {symbol} | Days: {days}")
    print(f"#  Cross-asset: {cross_assets}")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/4] Fetching primary data ({symbol})...")
    data = run_pipeline(symbol=symbol, days=days, window_size=10, stride=1)
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

    print(f"  Aligned: {len(features_df)} feature rows / {len(prices_aligned)} prices")

    print(f"\n[3/4] Walk-forward validation...")
    wf_results = walk_forward_validate(features_df, prices_aligned)

    print(f"\n   Computing baseline strategies on test set...")
    test_prices = prices_aligned[-len(wf_results['returns']):]
    baselines = {
        'Buy & Hold': baseline_buy_hold(test_prices),
        'MA Crossover': baseline_ma_crossover(test_prices),
        'Random Signals': baseline_random(test_prices),
    }

    print(f"\n[4/4] Cross-asset validation...")
    print(f"   Testing on: {cross_assets}")
    print(f"   (Sleeping 2s between requests to respect CoinGecko rate limits)")
    cross_asset_results = {}
    for sym in cross_assets:
        cross_asset_results.update(
            cross_asset_validation([sym], days=days,
                                   threshold=wf_results['best_threshold'])
        )
        time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n  Total validation time: {elapsed:.1f}s")

    print(f"\n[Report] Generating validation report...")
    report = format_validation_report(symbol, wf_results, baselines, cross_asset_results)

    if save_report:
        report_path = 'VALIDATION_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  Saved: {report_path}")

    print(f"\n{'#' * 70}")
    print(f"#  Validation Complete")
    print(f"{'#' * 70}\n")

    print(report[:3000])
    print(f"\n... [truncated — see VALIDATION_REPORT.md for full report]")

    save_validation_plots(wf_results, baselines, symbol)

    return wf_results, baselines, cross_asset_results


def save_validation_plots(wf_results, baselines, symbol):
    """Save validation plots."""
    os.makedirs('data/persistence', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    equity = np.array(wf_results['equity_curve'])
    axes[0, 0].plot(equity, 'b-', linewidth=2, label='TDA (test)')
    bh_curve = baselines['Buy & Hold']['final_equity']
    axes[0, 0].axhline(y=bh_curve, color='gray', linestyle='--',
                       label=f"Buy & Hold final = ${bh_curve:.0f}")
    axes[0, 0].set_title(f'Test Set Equity Curve ({symbol})')
    axes[0, 0].set_xlabel('Window')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    returns = np.array(wf_results['returns'])
    axes[0, 1].hist(returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Zero')
    axes[0, 1].axvline(x=returns.mean(), color='green', linestyle='-',
                       label=f'Mean = {returns.mean():.4f}')
    axes[0, 1].set_title('Return Distribution')
    axes[0, 1].set_xlabel('Return per window')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    da = wf_results['direction_accuracy']
    categories = ['BUY signals', 'SELL signals', 'Overall']
    accuracies = [da['buy_accuracy'], da['sell_accuracy'], da['overall_accuracy']]
    counts = [da['n_buy'], da['n_sell'], da['n_signals']]
    colors = ['green' if a > 0.5 else 'red' for a in accuracies]
    axes[1, 0].bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Chance (50%)')
    for i, (a, c) in enumerate(zip(accuracies, counts)):
        axes[1, 0].text(i, a + 0.02, f'n={c}', ha='center', fontsize=10)
    axes[1, 0].set_title('Direction Accuracy by Signal Type')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    bs = wf_results['bootstrap_sharpe']
    sharpe_actual = wf_results['test_metrics']['sharpe_ratio']
    axes[1, 1].errorbar([0], [bs['mean']],
                       yerr=[[bs['mean'] - bs['lower']], [bs['upper'] - bs['mean']]],
                       fmt='o', markersize=12, capsize=10, capthick=2,
                       label=f'Bootstrap CI [{bs["lower"]:.2f}, {bs["upper"]:.2f}]')
    axes[1, 1].scatter([0], [sharpe_actual], color='red', s=80, zorder=5,
                       label=f'Test Sharpe = {sharpe_actual:.2f}')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title('Bootstrap Sharpe Confidence Interval (95%)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/{symbol}_validation_plots.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots: {save_path}")


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365

    cross_assets = ['ETH', 'SOL']
    if symbol in cross_assets:
        cross_assets = [s for s in ['BTC', 'ETH', 'SOL', 'ADA'] if s != symbol][:2]

    run_full_validation(symbol=symbol, days=days, cross_assets=cross_assets)
