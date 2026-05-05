#!/usr/bin/env python
"""
Validation v4: Multi-asset pooled training + advanced features +
high-confidence threshold + volatility regime filter.

Combines all four improvements identified after v3:
  1. Multi-asset pool (BTC + ETH + SOL + ADA)
  2. Better features (log-returns, Garman-Klass vol, volume z-score)
  3. High prob threshold (>0.60) to reduce false signals
  4. Volatility regime filter (only trade when vol > median)

Usage:
  python examples/run_validation_v4.py 180   # 180 days x 4 assets
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.multi_asset_pipeline import (
    fetch_multi_asset_pool,
    add_targets,
    get_combined_feature_cols,
)
from src.regime_filter import apply_regime_filter
from src.validation_v2 import wilson_interval, time_series_kfold
from src.backtester import Backtester


def make_classifier(model_type='gbm', random_state=42):
    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state)
    elif model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=random_state, n_jobs=-1,
        )
    elif model_type == 'gbm':
        clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=random_state,
        )
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def signals_from_proba(proba, prob_threshold=0.60, max_position_size=0.10):
    """Convert probabilities to BUY/SELL/HOLD."""
    signals, sizes, confs = [], [], []
    for p in proba:
        confidence = abs(p - 0.5) * 2
        if p > prob_threshold:
            signals.append('BUY')
            sizes.append(max_position_size * confidence)
        elif p < 1 - prob_threshold:
            signals.append('SELL')
            sizes.append(-max_position_size * confidence)
        else:
            signals.append('HOLD')
            sizes.append(0.0)
        confs.append(confidence)
    return pd.DataFrame({
        'signal': signals,
        'position_size': sizes,
        'confidence_final': confs,
        'p_up': proba,
    })


def evaluate_pooled_kfold(pooled_df, feature_cols, model_type='gbm',
                          prob_threshold=0.60, n_splits=5,
                          regime_filters=None, verbose=False):
    """
    Pooled K-fold:
      - For each fold, split each asset's time series into train/test
      - Train on ALL assets' train portions pooled
      - Test on each asset's test portion separately
    """
    symbols = pooled_df['symbol'].unique().tolist()

    asset_folds = {}
    for symbol in symbols:
        asset_df = pooled_df[pooled_df['symbol'] == symbol].reset_index(drop=True)
        folds = time_series_kfold(len(asset_df), n_splits=n_splits)
        asset_folds[symbol] = (asset_df, folds)

    fold_results = []

    for fold_idx in range(n_splits):
        train_X_list, train_y_list = [], []
        for symbol in symbols:
            asset_df, folds = asset_folds[symbol]
            if fold_idx >= len(folds):
                continue
            train_idx, _ = folds[fold_idx]
            train_X_list.append(asset_df.iloc[train_idx][feature_cols].values)
            train_y_list.append(asset_df.iloc[train_idx]['target'].values)

        if not train_X_list:
            continue
        X_train = np.vstack(train_X_list)
        y_train = np.concatenate(train_y_list).astype(int)

        valid = ~np.any(np.isnan(X_train), axis=1)
        X_train, y_train = X_train[valid], y_train[valid]
        if len(np.unique(y_train)) < 2 or len(X_train) < 50:
            continue

        clf = make_classifier(model_type)
        clf.fit(X_train, y_train)

        for symbol in symbols:
            asset_df, folds = asset_folds[symbol]
            if fold_idx >= len(folds):
                continue
            _, test_idx = folds[fold_idx]
            test_df = asset_df.iloc[test_idx].reset_index(drop=True)
            X_test = test_df[feature_cols].values
            valid = ~np.any(np.isnan(X_test), axis=1)

            if valid.sum() == 0:
                continue

            test_df = test_df.iloc[valid].reset_index(drop=True)
            X_test = X_test[valid]
            test_prices = test_df['close'].values
            true_targets = test_df['target'].values.astype(int)

            proba = clf.predict_proba(X_test)
            p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

            signals = signals_from_proba(p_up, prob_threshold=prob_threshold)

            if regime_filters is not None:
                signals = apply_regime_filter(signals, test_df, regime_filters)

            n_buy = int((signals['signal'] == 'BUY').sum())
            n_sell = int((signals['signal'] == 'SELL').sum())

            correct = 0; total = 0
            for i in range(min(len(signals), len(true_targets))):
                sig = signals.iloc[i]['signal']
                if sig == 'BUY':
                    total += 1
                    if true_targets[i] == 1:
                        correct += 1
                elif sig == 'SELL':
                    total += 1
                    if true_targets[i] == 0:
                        correct += 1
            acc = correct / total if total > 0 else 0.5

            bt = Backtester()
            res = bt.run(test_prices, signals)
            bh = (test_prices[-1] / test_prices[0] - 1) * 100

            fold_results.append({
                'fold': fold_idx,
                'symbol': symbol,
                'n_test': len(test_df),
                'n_buy': n_buy, 'n_sell': n_sell,
                'n_signals': total,
                'direction_accuracy': acc,
                'tda_return_pct': res['metrics']['total_return_pct'],
                'tda_sharpe': res['metrics']['sharpe_ratio'],
                'tda_n_trades': res['metrics'].get('completed_trades', 0),
                'buy_hold_return_pct': bh,
                'outperformed_bh': res['metrics']['total_return_pct'] > bh,
            })

    return pd.DataFrame(fold_results)


def grid_search_v4(pooled_df, feature_cols, n_splits=5, verbose=True):
    """Grid search over models, thresholds, and regime filters."""
    grid = []
    for model_type in ['logistic', 'rf', 'gbm']:
        for thresh in [0.55, 0.58, 0.60, 0.62]:
            for filt in [None, {'vol': 'median'}, {'vol': 'q75'}]:
                grid.append((model_type, thresh, filt))

    if verbose:
        print(f"  Grid: {len(grid)} configs x {n_splits} folds x {len(pooled_df['symbol'].unique())} assets")

    rows = []
    for i, (model_type, thresh, filt) in enumerate(grid):
        if verbose and i % 6 == 0:
            print(f"    [{i+1}/{len(grid)}] {model_type} p={thresh} filter={filt}")
        try:
            fold_df = evaluate_pooled_kfold(
                pooled_df, feature_cols,
                model_type=model_type, prob_threshold=thresh,
                n_splits=n_splits, regime_filters=filt,
            )
            if len(fold_df) == 0:
                continue
            rows.append({
                'model_type': model_type,
                'prob_threshold': thresh,
                'regime_filter': str(filt) if filt else 'none',
                'n_evaluations': len(fold_df),
                'mean_direction_acc': float(fold_df['direction_accuracy'].mean()),
                'std_direction_acc': float(fold_df['direction_accuracy'].std()),
                'mean_sharpe': float(fold_df['tda_sharpe'].mean()),
                'mean_return_pct': float(fold_df['tda_return_pct'].mean()),
                'mean_buy_hold_pct': float(fold_df['buy_hold_return_pct'].mean()),
                'pct_beat_bh': float(fold_df['outperformed_bh'].mean()),
                'mean_n_trades': float(fold_df['tda_n_trades'].mean()),
                'total_signals': int(fold_df['n_signals'].sum()),
            })
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    return pd.DataFrame(rows).sort_values('mean_direction_acc', ascending=False)


def run_v4(symbols=None, days=180, interval='1h', n_splits=5, save_report=True):
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA']

    print(f"\n{'#' * 70}")
    print(f"#  TDA Strategy Validation V4 (Multi-Asset + Advanced Features)")
    print(f"#  Symbols: {symbols} | Days: {days} | Interval: {interval}")
    print(f"#  K-Fold: {n_splits} | Regime filter: vol > median or q75")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/3] Fetching multi-asset pool...")
    pooled_df = fetch_multi_asset_pool(symbols, days=days, interval=interval)

    print(f"\n[2/3] Adding direction targets (horizon=1)...")
    pooled_df = add_targets(pooled_df, horizon=1)

    feature_cols = get_combined_feature_cols(pooled_df)
    print(f"  Using {len(feature_cols)} features: {len(feature_cols)} cols")
    print(f"  Pool: {len(pooled_df)} samples, target balance: "
          f"{pooled_df['target'].mean():.2%}")

    print(f"\n[3/3] Grid search over (model, threshold, regime filter)...")
    grid_df = grid_search_v4(pooled_df, feature_cols, n_splits=n_splits)

    if len(grid_df) == 0:
        print(f"  No valid configs.")
        return None

    top = grid_df.head(10)
    print(f"\n  Top 10 configurations by direction accuracy:")
    print(top.to_string(index=False))

    best = grid_df.iloc[0]
    print(f"\n  BEST: {best['model_type']} @ p={best['prob_threshold']:.2f} "
          f"filter={best['regime_filter']}")
    print(f"    Direction accuracy: {best['mean_direction_acc']:.2%} ± {best['std_direction_acc']:.2%}")
    print(f"    Sharpe: {best['mean_sharpe']:.2f}, Return: {best['mean_return_pct']:.2f}%")
    print(f"    Beat buy-hold: {best['pct_beat_bh']:.0%}")

    print(f"\n[Detail] Re-running best config for per-fold breakdown...")
    filt = None if best['regime_filter'] == 'none' else eval(best['regime_filter'])
    fold_df = evaluate_pooled_kfold(
        pooled_df, feature_cols,
        model_type=best['model_type'],
        prob_threshold=best['prob_threshold'],
        n_splits=n_splits,
        regime_filters=filt,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")

    total_signals = int(fold_df['n_signals'].sum())
    avg_acc = float(fold_df['direction_accuracy'].mean())
    successes = int(round(avg_acc * total_signals))
    p, lo, hi = wilson_interval(successes, total_signals) if total_signals > 0 else (0, 0, 0)
    direction_significant = lo > 0.5

    report = format_v4_report(symbols, best, fold_df, p, lo, hi,
                              direction_significant, grid_df)

    if save_report:
        with open('VALIDATION_REPORT_V4.md', 'w') as f:
            f.write(report)
        print(f"\n  Saved: VALIDATION_REPORT_V4.md")

    save_v4_plots(fold_df, best, grid_df, symbols)

    print(f"\n{'#' * 70}")
    print(f"#  Validation V4 Complete")
    print(f"{'#' * 70}\n")

    print(report[:4500])
    print(f"\n... [see VALIDATION_REPORT_V4.md for full report]")

    return {'best': best, 'fold_df': fold_df, 'grid_df': grid_df}


def format_v4_report(symbols, best, fold_df, dir_acc, dir_lo, dir_hi, dir_sig, grid_df):
    lines = []
    lines.append(f"# Validation Report v4: Multi-Asset Pooled TDA Strategy\n")
    lines.append(f"**Assets:** {', '.join(symbols)}")
    lines.append(f"**Method:** Pooled multi-asset training + advanced features + regime filter")
    lines.append(f"**CV:** 5-fold time-series cross-validation (per asset)")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n---\n")

    lines.append(f"## 1. Best Configuration\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Model | **{best['model_type']}** |")
    lines.append(f"| Probability threshold | {best['prob_threshold']:.2f} |")
    lines.append(f"| Regime filter | {best['regime_filter']} |")
    lines.append(f"| Mean direction accuracy | **{best['mean_direction_acc']:.2%}** ± {best['std_direction_acc']:.2%} |")
    lines.append(f"| Mean Sharpe | {best['mean_sharpe']:.2f} |")
    lines.append(f"| Mean return per fold | {best['mean_return_pct']:.2f}% |")
    lines.append(f"| Beat buy-hold | {best['pct_beat_bh']:.0%} |")
    lines.append(f"| Total signals across folds | {best['total_signals']} |")
    lines.append("")

    lines.append(f"## 2. Per-Fold Per-Asset Breakdown\n")
    lines.append(f"| Fold | Asset | n_test | Signals | Acc | Return | Sharpe | BH | Beat? |")
    lines.append(f"|------|-------|--------|---------|-----|--------|--------|----|----|")
    for _, row in fold_df.iterrows():
        beat = "✅" if row['outperformed_bh'] else "❌"
        lines.append(f"| {int(row['fold'])} | {row['symbol']} | {int(row['n_test'])} | "
                     f"{int(row['n_signals'])} | {row['direction_accuracy']:.2%} | "
                     f"{row['tda_return_pct']:.2f}% | {row['tda_sharpe']:.2f} | "
                     f"{row['buy_hold_return_pct']:.2f}% | {beat} |")
    lines.append("")

    lines.append(f"## 3. Aggregate Statistics\n")
    by_asset = fold_df.groupby('symbol').agg({
        'direction_accuracy': 'mean',
        'tda_return_pct': 'mean',
        'tda_sharpe': 'mean',
        'buy_hold_return_pct': 'mean',
        'outperformed_bh': 'mean',
        'n_signals': 'sum',
    }).round(4)
    lines.append(f"### Per-Asset Performance\n")
    lines.append(f"| Asset | Acc | Return | Sharpe | BH | Beat BH | Signals |")
    lines.append(f"|-------|-----|--------|--------|----|----|---------|")
    for symbol, row in by_asset.iterrows():
        lines.append(f"| {symbol} | {row['direction_accuracy']:.2%} | "
                     f"{row['tda_return_pct']:.2f}% | {row['tda_sharpe']:.2f} | "
                     f"{row['buy_hold_return_pct']:.2f}% | {row['outperformed_bh']:.0%} | "
                     f"{int(row['n_signals'])} |")
    lines.append("")

    lines.append(f"## 4. Statistical Significance\n")
    lines.append(f"### Wilson Confidence Interval for Direction Accuracy\n")
    lines.append(f"- **Total signals:** {int(fold_df['n_signals'].sum())}")
    lines.append(f"- **Direction accuracy:** {dir_acc:.2%}")
    lines.append(f"- **95% Wilson CI:** [{dir_lo:.2%}, {dir_hi:.2%}]")
    if dir_sig:
        lines.append(f"- ✅ **Lower bound > 50% — STATISTICALLY SIGNIFICANT**")
    else:
        lines.append(f"- ⚠️ Lower bound ≤ 50% — not significantly better than chance")
    lines.append("")

    lines.append(f"## 5. Top 10 Grid Search Results\n")
    lines.append(f"| Model | Thresh | Filter | Acc | Sharpe | Return | Beat BH |")
    lines.append(f"|-------|--------|--------|-----|--------|--------|---------|")
    for _, row in grid_df.head(10).iterrows():
        lines.append(f"| {row['model_type']} | {row['prob_threshold']:.2f} | "
                     f"{row['regime_filter']} | {row['mean_direction_acc']:.2%} | "
                     f"{row['mean_sharpe']:.2f} | {row['mean_return_pct']:.2f}% | "
                     f"{row['pct_beat_bh']:.0%} |")
    lines.append("")

    lines.append(f"## 6. Verdict\n")
    wins, issues = [], []
    if best['mean_direction_acc'] > 0.55:
        wins.append(f"Direction accuracy clearly above chance: {best['mean_direction_acc']:.2%}")
    elif best['mean_direction_acc'] > 0.51:
        wins.append(f"Direction accuracy slightly above chance: {best['mean_direction_acc']:.2%}")
    else:
        issues.append(f"Direction accuracy at/below chance")
    if dir_sig:
        wins.append(f"Statistically significant (Wilson CI lower > 50%)")
    else:
        issues.append(f"Not statistically significant (Wilson CI: [{dir_lo:.2%}, {dir_hi:.2%}])")
    if best['mean_return_pct'] > best['mean_buy_hold_pct']:
        wins.append(f"Beats buy-hold ({best['mean_return_pct']:.2f}% vs {best['mean_buy_hold_pct']:.2f}%)")
    else:
        issues.append(f"Underperforms buy-hold ({best['mean_return_pct']:.2f}% vs {best['mean_buy_hold_pct']:.2f}%)")
    if best['pct_beat_bh'] >= 0.5:
        wins.append(f"Beat buy-hold in {best['pct_beat_bh']:.0%} of evaluations")
    else:
        issues.append(f"Beat buy-hold only in {best['pct_beat_bh']:.0%} of evaluations")
    if best['mean_sharpe'] > 0:
        wins.append(f"Positive Sharpe: {best['mean_sharpe']:.2f}")
    else:
        issues.append(f"Negative Sharpe: {best['mean_sharpe']:.2f}")

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
    if score >= 3:
        lines.append(f"**🟢 PASS** — Strategy validated. {len(wins)} wins vs {len(issues)} issues.")
        lines.append(f"Suitable for paper trading.")
    elif score >= 0:
        lines.append(f"**🟡 MARGINAL** — Mixed: {len(wins)} wins vs {len(issues)} issues.")
    else:
        lines.append(f"**🔴 NEEDS WORK** — {len(issues)} issues vs {len(wins)} wins.")

    lines.append(f"\n---\n*Generated by `examples/run_validation_v4.py`*")
    return "\n".join(lines)


def save_v4_plots(fold_df, best, grid_df, symbols):
    os.makedirs('data/persistence', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    by_asset = fold_df.groupby('symbol')[['direction_accuracy']].mean()
    accs = by_asset['direction_accuracy'].values
    asset_names = by_asset.index.tolist()
    colors = ['green' if a > 0.5 else 'red' for a in accs]
    axes[0, 0].bar(asset_names, accs, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='Chance')
    axes[0, 0].axhline(y=accs.mean(), color='blue', linestyle='-',
                        label=f'Mean = {accs.mean():.2%}')
    axes[0, 0].set_ylabel('Direction Accuracy')
    axes[0, 0].set_title(f'Per-Asset Accuracy ({best["model_type"]})')
    axes[0, 0].set_ylim(0.3, 0.7)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    by_asset_ret = fold_df.groupby('symbol').agg({
        'tda_return_pct': 'mean',
        'buy_hold_return_pct': 'mean',
    })
    x = np.arange(len(by_asset_ret))
    width = 0.35
    axes[0, 1].bar(x - width/2, by_asset_ret['tda_return_pct'], width,
                   label='TDA-ML', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, by_asset_ret['buy_hold_return_pct'], width,
                   label='Buy & Hold', color='gray', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(by_asset_ret.index)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_ylabel('Mean Return per Fold (%)')
    axes[0, 1].set_title('Returns: TDA vs Buy & Hold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    top10 = grid_df.head(10)
    labels = [f"{r['model_type'][:4]} t={r['prob_threshold']:.2f} f={r['regime_filter'][:6]}"
              for _, r in top10.iterrows()]
    accs10 = top10['mean_direction_acc'].values
    colors10 = ['green' if a > 0.5 else 'red' for a in accs10]
    axes[1, 0].barh(range(len(labels)), accs10, color=colors10, alpha=0.7)
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_yticklabels(labels, fontsize=8)
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Direction Accuracy')
    axes[1, 0].set_title('Top 10 Configurations')
    axes[1, 0].grid(alpha=0.3, axis='x')

    fold_means = fold_df.groupby('fold')['direction_accuracy'].mean().values
    fold_idx = sorted(fold_df['fold'].unique())
    axes[1, 1].plot(fold_idx, fold_means, 'o-', color='steelblue', linewidth=2, markersize=10)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='Chance')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Mean Direction Accuracy')
    axes[1, 1].set_title('Direction Accuracy Over Time (folds)')
    axes[1, 1].set_ylim(0.3, 0.7)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = f'data/persistence/v4_validation_plots.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots: {save_path}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    symbols = ['BTC', 'ETH', 'SOL', 'ADA']
    run_v4(symbols=symbols, days=days)
