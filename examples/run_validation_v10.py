#!/usr/bin/env python
"""
V10: Multi-Year Sample + 1000-Permutation Test.

Addresses two open methodological items from v9:

  1. SAMPLE WINDOW LIMITATION
     v9 used 365 days. Some market regimes (e.g., sustained bear markets)
     may not be represented. v10 fetches 3 years of hourly data
     (~26k candles per asset, ~180k pooled) to span more regimes.

  2. PERMUTATION COUNT
     v9 ran 25 permutations, giving an empirical p-value bound of
     1/(B+1) ~= 0.038. To claim p <= 0.001 we need B >= 999. v10 runs:
       - 1000 permutations on the BEST CONFIGURATION (single-config test)
       - 100 permutations on the FULL GRID (multiple-testing-aware test)

Output:
  - results/V10_MULTIYEAR.md
  - results/v10_permutations_1000.csv      (column: permutation_accuracy)
  - results/v10_permutations_grid_100.csv  (column: permutation_best_grid_accuracy)
  - results/figures/v10_permutation_distribution.{png,pdf}

Usage:
  python examples/run_validation_v10.py                     # 1095 days, 1000 single, 100 grid
  python examples/run_validation_v10.py 1095 1000 100       # explicit
  python examples/run_validation_v10.py 365 200 50          # quicker version
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.multi_asset_pipeline import (
    fetch_multi_asset_pool,
    get_combined_feature_cols,
)
from examples.run_validation_v9 import (
    add_targets, evaluate_kfold, temporal_split,
)
# NOTE: v10 deliberately does NOT use v9's `block_shuffle_targets`. v9's
# implementation truncates the final partial block (``n_blocks = n //
# block_hours`` then drops un-filled tail rows). v10 defines a local
# helper below that preserves all rows including the final partial
# block.


# ============================================================
# v10-local helpers (signal-weighted accuracy + leak-aware block shuffle)
# ============================================================

def weighted_accuracy(fold_df):
    """Signal-weighted accuracy across (fold, symbol) rows of ``evaluate_kfold``.

    Returns (weighted_accuracy, n_signals_total). When n_signals_total
    is zero, returns (0.5, 0) — the neutral chance value, matching the
    grid-search guard that requires ``n_signals >= 30``.

    The earlier v10 used ``float(fold_df['direction_accuracy'].mean())``,
    which weighs each (fold, symbol) row equally even if one row has
    many more signals. That biases the headline accuracy toward
    low-signal-count cells (e.g., a single fold with 5 signals can
    drag the mean as much as a fold with 5,000). The signal-weighted
    form is the correct pooled estimator and matches the per-asset
    Wilson CI computation v9 uses.
    """
    n_sig = int(fold_df['n_signals'].sum())
    if n_sig == 0:
        return 0.5, 0
    weighted = float(
        (fold_df['direction_accuracy'] * fold_df['n_signals']).sum() / n_sig
    )
    return weighted, n_sig


def block_shuffle_targets_preserve_remainder(df, block_hours=168, seed=0):
    """Block-shuffle ``target`` per symbol while preserving every row.

    Algorithm:
      1. For each symbol group, sort rows by timestamp.
      2. Split the per-symbol target series into contiguous blocks of
         ``block_hours`` rows, including a final shorter block when
         ``n % block_hours != 0``.
      3. Shuffle the *order* of the blocks with the provided seed.
      4. Reassemble target values in the new block order and write
         them back to the chronological row positions.

    Compared to v9's ``block_shuffle_targets``, this keeps the full
    sample size: no rows are dropped because of an unfilled tail. Only
    the ``target`` column is permuted; features, prices, and timestamps
    are unchanged. The resulting DataFrame has exactly the same number
    of rows as the input.
    """
    out = df.copy()
    rng = np.random.RandomState(seed)
    for sym, group in out.groupby('symbol', sort=False):
        order_idx = group.sort_values('timestamp').index.values
        n = len(order_idx)
        if n == 0:
            continue
        targets_in_order = out.loc[order_idx, 'target'].values
        starts = list(range(0, n, block_hours))
        blocks = [targets_in_order[s:min(s + block_hours, n)] for s in starts]
        block_order = list(range(len(blocks)))
        rng.shuffle(block_order)
        shuffled = np.concatenate([blocks[i] for i in block_order])
        # length must match exactly — this is the leak-aware contract
        assert len(shuffled) == n, (
            f"block-shuffle row drift for {sym}: expected {n}, got "
            f"{len(shuffled)}"
        )
        out.loc[order_idx, 'target'] = shuffled
    return out


def run_single_config_permutation(train_val_df, feature_cols, model_type, threshold,
                                   horizon, regime_filter, n_iter, n_splits=5):
    """B-permutation test of the SELECTED best config.

    Caveat: this is a **post-selection** diagnostic — the best config
    was chosen on the unshuffled real data, so the null distribution
    here is over chance variation *at that single config*, not over
    chance + multiple testing. Use ``run_grid_permutation`` for the
    multiple-testing-aware p-value.
    """
    df_t = add_targets(train_val_df, horizon=horizon)
    accs = []
    for it in range(n_iter):
        if it % 50 == 0:
            print(f"    Single-config permutation {it}/{n_iter}...")
        df_shuffled = block_shuffle_targets_preserve_remainder(df_t, seed=it)
        fold_df = evaluate_kfold(
            df_shuffled, feature_cols, model_type, threshold,
            n_splits=n_splits, horizon=horizon, regime_filter=regime_filter,
        )
        if len(fold_df) > 0:
            acc, _ = weighted_accuracy(fold_df)
            accs.append(acc)
    return np.array(accs)


def run_grid_permutation(train_val_df, feature_cols, horizon, n_iter, n_splits=5):
    """B-permutation test of the full grid (multiple-testing-aware).

    For each permutation, runs the entire model/threshold/regime grid
    on shuffled labels and records the best accuracy across the grid.
    The empirical p-value computed against this distribution is the
    main, multiple-testing-aware evidence v10 reports.
    """
    df_t = add_targets(train_val_df, horizon=horizon)
    best_accs = []

    for it in range(n_iter):
        if it % 10 == 0:
            print(f"    Grid permutation {it}/{n_iter}...")
        df_shuffled = block_shuffle_targets_preserve_remainder(df_t, seed=10000 + it)
        best_in_grid = -1.0
        for model_type in ['logistic', 'rf']:
            for thresh in [0.62, 0.65, 0.70]:
                for filt in [None, {'vol': 'median'}]:
                    fold_df = evaluate_kfold(
                        df_shuffled, feature_cols, model_type, thresh,
                        n_splits=n_splits, horizon=horizon, regime_filter=filt,
                    )
                    if len(fold_df) > 0 and int(fold_df['n_signals'].sum()) >= 30:
                        acc, _ = weighted_accuracy(fold_df)
                        if acc > best_in_grid:
                            best_in_grid = acc
        if best_in_grid > 0:
            best_accs.append(best_in_grid)
    return np.array(best_accs)


def run_v10(symbols=None, days=1095, n_perm_single=1000, n_perm_grid=100):
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX']

    print(f"\n{'#' * 70}")
    print(f"#  V10: Multi-Year Sample + Large Permutation Test")
    print(f"#  Symbols: {symbols} | Days: {days} ({days/365:.1f} years)")
    print(f"#  Single-config permutations: {n_perm_single}")
    print(f"#  Grid permutations: {n_perm_grid}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print(f"[1/6] Fetching {days}-day multi-asset pool...")
    pooled_df = fetch_multi_asset_pool(symbols, days=days, interval='1h')
    feature_cols = get_combined_feature_cols(pooled_df)
    print(f"  Pool size: {len(pooled_df):,} samples")
    print(f"  Period: {pooled_df.timestamp.min().date()} -> {pooled_df.timestamp.max().date()}")
    print(f"  Features: {len(feature_cols)}")

    print(f"\n[2/6] Train/Val/Holdout split (80/20)...")
    train_val_df, holdout_df = temporal_split(pooled_df, train_val_pct=0.80)
    print(f"  Train+Val: {len(train_val_df):,}")
    print(f"  Holdout:   {len(holdout_df):,}")

    horizon = 72
    print(f"\n[3/6] Establishing best config on {days}-day Train+Val (horizon=72h, 3d)...")
    train_val_targeted = add_targets(train_val_df, horizon=horizon)
    best_acc_real = -1.0
    best_config = None
    for model_type in ['logistic', 'rf']:
        for thresh in [0.62, 0.65, 0.70]:
            for filt in [None, {'vol': 'median'}]:
                fold_df = evaluate_kfold(
                    train_val_targeted, feature_cols, model_type, thresh,
                    n_splits=5, horizon=horizon, regime_filter=filt,
                )
                if len(fold_df) == 0 or int(fold_df['n_signals'].sum()) < 30:
                    continue
                acc, _ = weighted_accuracy(fold_df)
                if acc > best_acc_real:
                    best_acc_real = acc
                    best_config = (model_type, thresh, filt)

    if best_config is None:
        raise RuntimeError(
            "No valid v10 config found; try lowering thresholds, "
            "checking data availability, or reducing n_splits."
        )

    print(f"  Best on real data: {best_config}, accuracy = {best_acc_real:.2%}")

    print(f"\n[4/6] {n_perm_single}-permutation POST-SELECTION DIAGNOSTIC at the best config...")
    print(f"  This is a single-config null distribution at the config selected")
    print(f"  on real data. It tests chance variation at THIS config only —")
    print(f"  not chance + multiple testing. The full-grid test in [5/6] is")
    print(f"  the main, multiple-testing-aware evidence.")
    t_start = time.time()
    perm_accs_single = run_single_config_permutation(
        train_val_df, feature_cols,
        model_type=best_config[0], threshold=best_config[1],
        horizon=horizon, regime_filter=best_config[2],
        n_iter=n_perm_single,
    )
    elapsed_single = time.time() - t_start
    print(f"  Single-config permutation done in {elapsed_single/60:.1f} min")

    n_above = int(np.sum(perm_accs_single >= best_acc_real))
    p_val_single = (n_above + 1) / (len(perm_accs_single) + 1)
    print(f"  {n_above}/{len(perm_accs_single)} permutations matched real result")
    print(f"  Empirical p-value (single-config DIAGNOSTIC): {p_val_single:.5f}")
    print(f"  Treat this as a sanity check, not as the headline p-value.")

    print(f"\n[5/6] {n_perm_grid}-permutation FULL-GRID test (multiple-testing-aware MAIN result)...")
    print(f"  For every permutation, runs the entire grid and takes the BEST.")
    print(f"  The empirical p-value here accounts for cherry-picking across the grid.")
    t_start = time.time()
    perm_accs_grid = run_grid_permutation(
        train_val_df, feature_cols, horizon=horizon, n_iter=n_perm_grid,
    )
    elapsed_grid = time.time() - t_start
    print(f"  Grid permutation done in {elapsed_grid/60:.1f} min")

    n_above_grid = int(np.sum(perm_accs_grid >= best_acc_real))
    p_val_grid = (n_above_grid + 1) / (len(perm_accs_grid) + 1)
    print(f"  {n_above_grid}/{len(perm_accs_grid)} grid permutations matched")
    print(f"  Empirical p-value (full grid): {p_val_grid:.4f}")

    print(f"\n[6/6] Writing report and figures...")
    write_v10_report(
        symbols, days, best_config, best_acc_real,
        perm_accs_single, perm_accs_grid,
        n_perm_single, n_perm_grid,
        len(pooled_df), len(train_val_df), len(holdout_df),
        elapsed_single, elapsed_grid,
    )

    plot_permutation_distributions(perm_accs_single, perm_accs_grid, best_acc_real)

    pd.DataFrame({'permutation_accuracy': perm_accs_single}).to_csv(
        'results/v10_permutations_1000.csv', index=False
    )
    pd.DataFrame({'permutation_best_grid_accuracy': perm_accs_grid}).to_csv(
        'results/v10_permutations_grid_100.csv', index=False
    )
    print(f"  Saved: results/v10_permutations_1000.csv")
    print(f"  Saved: results/v10_permutations_grid_100.csv")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"\n{'#' * 70}")
    print(f"#  V10 Complete - see results/V10_MULTIYEAR.md")
    print(f"{'#' * 70}\n")


def write_v10_report(symbols, days, best_config, best_acc, perm_single, perm_grid,
                      n_single, n_grid, pool_size, tv_size, holdout_size,
                      elapsed_single, elapsed_grid):
    md = []
    md.append(f"# V10: Multi-Year Sample + 1000-Permutation Test\n")
    md.append(f"**Goal:** Address two limitations from v9:")
    md.append(f"1. Sample window: extend from 365 days to {days} days ({days/365:.1f} years)")
    md.append(f"2. Permutation count: extend from B=25 to B={n_single} (single config) and B={n_grid} (full grid)\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    md.append(f"**Symbols:** {', '.join(symbols)}")
    md.append(f"**Pool size:** {pool_size:,} samples")
    md.append(f"**Train+Val:** {tv_size:,} | **Holdout:** {holdout_size:,}\n")

    md.append(f"## 1. Best Configuration (selected on multi-year Train+Val)\n")
    md.append(f"| Parameter | Value |")
    md.append(f"|-----------|-------|")
    md.append(f"| Model | {best_config[0]} |")
    md.append(f"| Probability threshold | {best_config[1]:.2f} |")
    md.append(f"| Regime filter | {str(best_config[2])} |")
    md.append(f"| CV direction accuracy | **{best_acc:.2%}** |\n")

    n_above_s = int(np.sum(perm_single >= best_acc))
    p_s = (n_above_s + 1) / (len(perm_single) + 1)
    md.append(f"## 2. Post-Selection Single-Config Diagnostic (B = {n_single})\n")
    md.append(f"Block-shuffled targets in 7-day blocks (final partial block preserved), reran the BEST config.")
    md.append(f"\n**Important caveat:** the best config was selected on the unshuffled real data, "
              f"so this null distribution is over chance variation *at this single config only*, "
              f"not over chance + multiple testing across the grid. Treat this section as a "
              f"sanity check; section 3 (full-grid permutation) is the headline multiple-testing-aware result.\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Permutations run | {len(perm_single)} |")
    md.append(f"| Mean shuffled-data accuracy | {perm_single.mean():.2%} +/- {perm_single.std():.2%} |")
    md.append(f"| Max shuffled-data accuracy | {perm_single.max():.2%} |")
    md.append(f"| Actual accuracy | **{best_acc:.2%}** |")
    md.append(f"| Permutations matching real result | {n_above_s}/{len(perm_single)} |")
    md.append(f"| **Empirical p-value** | **{p_s:.5f}** |")
    md.append(f"| Computation time | {elapsed_single/60:.1f} min |\n")

    n_above_g = int(np.sum(perm_grid >= best_acc))
    p_g = (n_above_g + 1) / (len(perm_grid) + 1)
    md.append(f"## 3. Full-Grid Permutation Test (B = {n_grid}) — main multiple-testing-aware result\n")
    md.append(f"For each permutation, reran the FULL grid search on block-shuffled targets and recorded the BEST accuracy across the grid.")
    md.append(f"This null distribution accounts for cherry-picking across model × threshold × regime configurations, so its empirical p-value is the headline evidence v10 reports.\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Permutations run | {len(perm_grid)} |")
    md.append(f"| Mean best-of-grid on shuffled data | {perm_grid.mean():.2%} +/- {perm_grid.std():.2%} |")
    md.append(f"| Max best-of-grid on shuffled data | {perm_grid.max():.2%} |")
    md.append(f"| Actual best-of-grid on real data | **{best_acc:.2%}** |")
    md.append(f"| Permutations matching real | {n_above_g}/{len(perm_grid)} |")
    md.append(f"| **Empirical p-value** | **{p_g:.4f}** |")
    md.append(f"| Computation time | {elapsed_grid/60:.1f} min |\n")

    md.append(f"## 4. Verdict\n")
    md.append(f"**Headline (full-grid, multiple-testing-aware):**")
    if p_g < 0.001:
        md.append(f"- Strongly significant (p = {p_g:.4f} < 0.001).")
    elif p_g < 0.01:
        md.append(f"- Significant (p = {p_g:.4f} < 0.01).")
    elif p_g < 0.05:
        md.append(f"- Marginally significant (p = {p_g:.4f} < 0.05).")
    else:
        md.append(f"- Not significant (p = {p_g:.4f}).")

    md.append(f"\n**Post-selection single-config diagnostic** (sanity check, NOT a headline p-value):")
    if p_s < 0.001:
        md.append(f"- Diagnostic p = {p_s:.5f} (< 0.001).")
    elif p_s < 0.01:
        md.append(f"- Diagnostic p = {p_s:.4f} (< 0.01).")
    elif p_s < 0.05:
        md.append(f"- Diagnostic p = {p_s:.4f} (< 0.05).")
    else:
        md.append(f"- Diagnostic p = {p_s:.4f}.")

    md.append(f"\nThis run materially strengthens the permutation evidence relative to v9, "
              f"subject to the grid, sample window, and data source tested.")

    md_text = "\n".join(md)
    with open('results/V10_MULTIYEAR.md', 'w') as f:
        f.write(md_text)
    print(f"  Saved: results/V10_MULTIYEAR.md")


def plot_permutation_distributions(perm_single, perm_grid, best_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(perm_single * 100, bins=40, alpha=0.7, color='gray', edgecolor='black',
            label=f'Single-config (n={len(perm_single)})')
    ax.axvline(x=best_acc * 100, color='red', linewidth=2.5,
                label=f'Real data: {best_acc:.1%}')
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Chance')
    n_above = int(np.sum(perm_single >= best_acc))
    p = (n_above + 1) / (len(perm_single) + 1)
    ax.set_xlabel('Direction Accuracy (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Single-Config Permutation Test\n({n_above} / {len(perm_single)} >= real, p = {p:.5f})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(perm_grid * 100, bins=20, alpha=0.7, color='steelblue', edgecolor='black',
            label=f'Full-grid maximum (n={len(perm_grid)})')
    ax.axvline(x=best_acc * 100, color='red', linewidth=2.5,
                label=f'Real best-of-grid: {best_acc:.1%}')
    ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Chance')
    n_above_g = int(np.sum(perm_grid >= best_acc))
    p_g = (n_above_g + 1) / (len(perm_grid) + 1)
    ax.set_xlabel('Best-of-Grid Direction Accuracy (%)')
    ax.set_title(f'Full-Grid Permutation Test\n({n_above_g} / {len(perm_grid)} >= real, p = {p_g:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/v10_permutation_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/figures/v10_permutation_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: results/figures/v10_permutation_distribution.{{png,pdf}}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    n_perm_single = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_perm_grid = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    run_v10(days=days, n_perm_single=n_perm_single, n_perm_grid=n_perm_grid)
