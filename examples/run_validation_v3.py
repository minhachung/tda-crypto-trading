#!/usr/bin/env python
"""
Validation v3: ML classifier on TDA features + k-fold CV + grid search.

The v2 results showed rule-based threshold strategies are at chance.
v3 trains a logistic/RF/GBM classifier on TDA features within each fold
to extract more information from the topological signals.

Usage:
  python examples/run_validation_v3.py BTC 90 1h
  python examples/run_validation_v3.py ETH 180 1h
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.binance_data import run_hf_pipeline
from src.persistent_homology import compute_features_for_windows
from src.ml_signals import grid_search_ml, evaluate_ml_kfold
from src.validation_v2 import wilson_interval, bootstrap_metric, sharpe_from_returns, t_test_returns


def run_v3(symbol='BTC', days=90, interval='1h', n_splits=5, save_report=True):
    print(f"\n{'#' * 70}")
    print(f"#  TDA Strategy Validation V3 (ML-based)")
    print(f"#  Symbol: {symbol} | Days: {days} | Interval: {interval}")
    print(f"#  K-Fold CV: {n_splits} folds | Models: LogReg, RF, GBM")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/3] Fetching Coinbase {interval} data ({days} days)...")
    data = run_hf_pipeline(symbol=symbol, days=days, interval=interval,
                           window_size=20, stride=1)
    df = data['df']
    point_clouds = data['point_clouds']
    end_indices = data['end_indices']

    print(f"\n[2/3] Computing TDA features ({len(point_clouds)} windows)...")
    features_df = compute_features_for_windows(
        point_clouds, end_indices=end_indices, verbose=False
    )

    if 'end_idx' in features_df.columns:
        idx = features_df['end_idx'].astype(int).values
        idx = idx[idx < len(df)]
        prices = df['close'].values[idx]
        features_df = features_df.iloc[:len(prices)].reset_index(drop=True)
    else:
        prices = df['close'].values[-len(features_df):]

    print(f"  Aligned: {len(features_df)} feature rows / {len(prices)} prices")

    print(f"\n[3/3] Grid searching ML models with {n_splits}-fold CV...")
    print(f"      (3 models x 4 thresholds x 3 horizons = 36 combos)")
    grid_df = grid_search_ml(features_df, prices, n_splits=n_splits, verbose=False)

    if len(grid_df) == 0:
        print(f"  No valid combos. Try more data or fewer folds.")
        return None

    top5 = grid_df.head(5)
    print(f"\n  Top 5 configurations by direction accuracy:")
    print(top5.to_string(index=False))

    best = grid_df.iloc[0]
    print(f"\n  Best: {best['model_type']} @ p={best['prob_threshold']:.2f} h={int(best['horizon'])}")
    print(f"  Direction accuracy: {best['mean_direction_acc']:.2%} ± {best['std_direction_acc']:.2%}")
    print(f"  Sharpe: {best['mean_sharpe']:.2f}, Return: {best['mean_return_pct']:.2f}%")
    print(f"  Beat buy-hold in {best['pct_beat_bh']:.0%} of folds")

    print(f"\n[4/4] Detailed run with best config...")
    fold_df = evaluate_ml_kfold(
        features_df, prices, n_splits=n_splits,
        model_type=best['model_type'],
        prob_threshold=best['prob_threshold'],
        horizon=int(best['horizon']),
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")

    n_buy = int(fold_df['n_buy'].sum())
    n_sell = int(fold_df['n_sell'].sum())
    total_signals = n_buy + n_sell
    avg_acc = float(fold_df['direction_accuracy'].mean())
    successes = int(round(avg_acc * total_signals))
    p, lo, hi = wilson_interval(successes, total_signals) if total_signals > 0 else (0, 0, 0)
    direction_significant = lo > 0.5

    report = format_v3_report(symbol, best, fold_df, p, lo, hi, direction_significant, grid_df)

    if save_report:
        with open('VALIDATION_REPORT_V3.md', 'w') as f:
            f.write(report)
        print(f"\n  Saved: VALIDATION_REPORT_V3.md")

    save_v3_plots(symbol, fold_df, best, grid_df)

    print(f"\n{'#' * 70}")
    print(f"#  Validation V3 Complete")
    print(f"{'#' * 70}\n")

    print(report[:3500])
    print(f"\n... [see VALIDATION_REPORT_V3.md for full report]")

    return {'best': best, 'fold_df': fold_df, 'grid_df': grid_df}


def format_v3_report(symbol, best, fold_df, dir_acc, dir_lo, dir_hi, dir_sig, grid_df):
    lines = []
    lines.append(f"# Validation Report v3: ML-Based TDA Strategy\n")
    lines.append(f"**Symbol:** {symbol}")
    lines.append(f"**Method:** ML classifier on TDA features + 5-fold time-series CV")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n---\n")

    lines.append(f"## 1. Best Model (from grid search)\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Model type | **{best['model_type']}** |")
    lines.append(f"| Probability threshold | {best['prob_threshold']:.2f} |")
    lines.append(f"| Forecast horizon | {int(best['horizon'])} period(s) |")
    lines.append(f"| Mean direction accuracy | **{best['mean_direction_acc']:.2%}** ± {best['std_direction_acc']:.2%} |")
    lines.append(f"| Mean Sharpe | {best['mean_sharpe']:.2f} |")
    lines.append(f"| Mean return per fold | {best['mean_return_pct']:.2f}% |")
    lines.append(f"| Beat buy-hold | {best['pct_beat_bh']:.0%} of folds |")
    lines.append("")

    lines.append(f"## 2. K-Fold Performance Detail\n")
    lines.append(f"| Fold | n_train | n_test | TDA Return | Sharpe | Direction | Trades | Buy&Hold | Beat BH |")
    lines.append(f"|------|---------|--------|-----------|--------|-----------|--------|----------|---------|")
    for _, row in fold_df.iterrows():
        beat = "✅" if row['outperformed_bh'] else "❌"
        lines.append(f"| {int(row['fold'])} | {int(row['n_train'])} | {int(row['n_test'])} | "
                     f"{row['tda_return_pct']:.2f}% | {row['tda_sharpe']:.2f} | "
                     f"{row['direction_accuracy']:.2%} | {int(row['tda_n_trades'])} | "
                     f"{row['buy_hold_return_pct']:.2f}% | {beat} |")
    lines.append("")

    lines.append(f"## 3. Statistical Significance\n")
    total_sig = int(fold_df['n_buy'].sum() + fold_df['n_sell'].sum())
    lines.append(f"### Wilson Confidence Interval for Direction Accuracy\n")
    lines.append(f"- **Total signals:** {total_sig}")
    lines.append(f"- **Direction accuracy:** {dir_acc:.2%}")
    lines.append(f"- **95% Wilson CI:** [{dir_lo:.2%}, {dir_hi:.2%}]")
    if dir_sig:
        lines.append(f"- ✅ **Lower bound > 50% — significantly better than chance**")
    else:
        lines.append(f"- ⚠️ Lower bound ≤ 50% — not significantly better than chance")
    lines.append("")

    lines.append(f"## 4. Top 10 Grid Search Configurations\n")
    lines.append(f"| Model | Thresh | Horizon | Acc | Sharpe | Return | Beat BH |")
    lines.append(f"|-------|--------|---------|-----|--------|--------|---------|")
    for _, row in grid_df.head(10).iterrows():
        lines.append(f"| {row['model_type']} | {row['prob_threshold']:.2f} | "
                     f"{int(row['horizon'])} | {row['mean_direction_acc']:.2%} | "
                     f"{row['mean_sharpe']:.2f} | {row['mean_return_pct']:.2f}% | "
                     f"{row['pct_beat_bh']:.0%} |")
    lines.append("")

    lines.append(f"## 5. Verdict\n")
    wins, issues = [], []

    if best['mean_direction_acc'] > 0.55:
        wins.append(f"Direction accuracy clearly above chance: {best['mean_direction_acc']:.2%}")
    elif best['mean_direction_acc'] > 0.51:
        wins.append(f"Direction accuracy slightly above chance: {best['mean_direction_acc']:.2%}")
    else:
        issues.append(f"Direction accuracy at/below chance: {best['mean_direction_acc']:.2%}")

    if dir_sig:
        wins.append(f"Statistically significant (Wilson CI lower > 50%)")
    else:
        issues.append(f"Not statistically significant (Wilson CI: [{dir_lo:.2%}, {dir_hi:.2%}])")

    if best['mean_return_pct'] > best['mean_buy_hold_pct']:
        wins.append(f"Beats buy-hold on average ({best['mean_return_pct']:.2f}% vs "
                    f"{best['mean_buy_hold_pct']:.2f}%)")
    else:
        issues.append(f"Underperforms buy-hold ({best['mean_return_pct']:.2f}% vs "
                      f"{best['mean_buy_hold_pct']:.2f}%)")

    if best['pct_beat_bh'] >= 0.6:
        wins.append(f"Beat buy-hold in {best['pct_beat_bh']:.0%} of folds")
    elif best['pct_beat_bh'] < 0.4:
        issues.append(f"Beat buy-hold only in {best['pct_beat_bh']:.0%} of folds")

    lines.append(f"### ✅ Strengths")
    for w in wins:
        lines.append(f"- {w}")
    if not wins:
        lines.append(f"- None identified.")

    lines.append(f"\n### ⚠️ Weaknesses")
    for i in issues:
        lines.append(f"- {i}")
    if not issues:
        lines.append(f"- None identified.")

    score = len(wins) - len(issues)
    lines.append(f"\n### Overall")
    if score >= 2:
        lines.append(f"**🟢 PASS** — ML approach validates: {len(wins)} wins vs {len(issues)} issues.")
    elif score >= 0:
        lines.append(f"**🟡 MARGINAL** — Better than rule-based but not strong: {len(wins)} wins vs {len(issues)} issues.")
    else:
        lines.append(f"**🔴 NEEDS WORK** — Same problem as rule-based: {len(issues)} issues vs {len(wins)} wins.")
        lines.append(f"\nNext steps to consider:")
        lines.append(f"- Engineer better features (returns, log-returns, volume profile)")
        lines.append(f"- Use longer history (multi-year)")
        lines.append(f"- Add macro/sentiment features")
        lines.append(f"- Try different point-cloud constructions (Takens embedding)")

    lines.append(f"\n---\n*Generated by `src/ml_signals.py` + `examples/run_validation_v3.py`*")
    return "\n".join(lines)


def save_v3_plots(symbol, fold_df, best, grid_df):
    os.makedirs('data/persistence', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = fold_df['fold'].values
    width = 0.35
    axes[0, 0].bar(x - width/2, fold_df['tda_return_pct'].values, width,
                   label='ML-TDA', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, fold_df['buy_hold_return_pct'].values, width,
                   label='Buy & Hold', color='gray', alpha=0.8)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].set_title(f'V3 Per-Fold Returns: {best["model_type"]} ({symbol})')
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
    axes[0, 1].set_ylim(0.3, 0.7)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    top10 = grid_df.head(10)
    labels = [f"{r['model_type'][:4]} t={r['prob_threshold']:.2f} h={int(r['horizon'])}"
              for _, r in top10.iterrows()]
    accs10 = top10['mean_direction_acc'].values
    colors10 = ['green' if a > 0.5 else 'red' for a in accs10]
    axes[1, 0].barh(range(len(labels)), accs10, color=colors10, alpha=0.7, edgecolor='black')
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_yticklabels(labels, fontsize=8)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Direction Accuracy')
    axes[1, 0].set_title('Top 10 Configurations')
    axes[1, 0].grid(alpha=0.3, axis='x')

    sharpes = fold_df['tda_sharpe'].values
    axes[1, 1].bar(x, sharpes, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].axhline(y=sharpes.mean(), color='red', linestyle='--',
                       label=f'Mean = {sharpes.mean():.2f}')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].set_title('Sharpe by Fold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/{symbol}_validation_v3_plots.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots: {save_path}")


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    interval = sys.argv[3] if len(sys.argv) > 3 else '1h'
    run_v3(symbol=symbol, days=days, interval=interval)
