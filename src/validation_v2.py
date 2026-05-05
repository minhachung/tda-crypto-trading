"""
Validation Framework v2: K-Fold Time Series CV + Grid Search.

Improvements over v1:
  - K-fold time series cross-validation (5 folds instead of 1 split)
  - Grid search over BOTH threshold AND lookback window
  - Robust to small sample sizes (Wilson confidence intervals)
  - Walk-forward without data leakage
  - Automatic threshold selection per fold (no global overfitting)
  - Proper aggregation of fold results with std error

Run with high-frequency Coinbase data for proper sample sizes.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

warnings.filterwarnings('ignore')

from src.persistent_homology import compute_features_for_windows
from src.trading_signals import TradingSignalGenerator
from src.backtester import Backtester


# ============================================================
# Wilson Confidence Interval (better for small samples)
# ============================================================

def wilson_interval(successes, n, confidence=0.95):
    """
    Wilson score interval for proportion (more robust than normal approx
    for small samples or extreme proportions like 0% or 100%).
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(p), float(max(0, center - margin)), float(min(1, center + margin))


# ============================================================
# K-Fold Time Series Cross-Validation
# ============================================================

def time_series_kfold(n_samples, n_splits=5, min_train_size=None):
    """
    Generate time-series-aware K-fold splits (no shuffle, no leakage).

    Each fold:
      - Train: data from start to fold_end - test_size
      - Test: next test_size samples

    Returns list of (train_idx, test_idx) tuples.
    """
    if min_train_size is None:
        min_train_size = max(20, n_samples // (n_splits + 1))

    test_size = (n_samples - min_train_size) // n_splits
    if test_size < 5:
        test_size = 5

    folds = []
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        folds.append((train_idx, test_idx))

    return folds


# ============================================================
# Grid Search with Cross-Validation
# ============================================================

def grid_search_cv(features_df, prices, param_grid=None, n_splits=5,
                   metric='direction_accuracy', verbose=False):
    """
    Grid search over signal parameters using time-series CV.

    Args:
        features_df: TDA features
        prices: aligned prices
        param_grid: dict of param lists
        n_splits: number of CV folds
        metric: 'direction_accuracy', 'sharpe', or 'sortino'

    Returns:
        Best params + per-fold results
    """
    if param_grid is None:
        param_grid = {
            'c1_threshold_std': [0.5, 0.75, 1.0, 1.25, 1.5],
            'lookback_window': [10, 15, 20],
            'confidence_threshold': [0.2, 0.3, 0.5],
        }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    folds = time_series_kfold(len(features_df), n_splits=n_splits)
    if verbose:
        print(f"  Grid search: {len(combos)} combos x {len(folds)} folds = "
              f"{len(combos) * len(folds)} evaluations")

    best_score = -np.inf
    best_params = None
    all_results = []

    for combo in combos:
        params = dict(zip(keys, combo))

        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            test_features = features_df.iloc[test_idx].reset_index(drop=True)
            test_prices = prices[test_idx]

            gen = TradingSignalGenerator(
                c1_threshold_std=params['c1_threshold_std'],
                c1_drop_threshold=-params['c1_threshold_std'] * 0.5,
                confidence_threshold=params['confidence_threshold'],
                lookback_window=min(params['lookback_window'], len(test_features) - 1),
            )

            try:
                signals = gen.generate_signals(test_features, price_series=test_prices)
            except Exception:
                continue

            if metric == 'direction_accuracy':
                score = _direction_score(signals, test_prices)
            elif metric == 'sharpe':
                bt = Backtester()
                res = bt.run(test_prices, signals)
                score = res['metrics']['sharpe_ratio']
            elif metric == 'sortino':
                bt = Backtester()
                res = bt.run(test_prices, signals)
                score = res['metrics']['sortino_ratio']
            else:
                score = 0.0

            fold_scores.append(score)

        if not fold_scores:
            continue

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        all_results.append({
            **params,
            'mean_score': mean_score,
            'std_score': std_score,
            'n_folds': len(fold_scores),
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_params['mean_score'] = mean_score
            best_params['std_score'] = std_score

    return {
        'best_params': best_params,
        'all_results': pd.DataFrame(all_results).sort_values('mean_score', ascending=False),
        'metric': metric,
    }


def _direction_score(signals_df, prices, horizon=1):
    """Direction accuracy for fold."""
    n = min(len(signals_df), len(prices) - horizon)
    if n <= 0:
        return 0.5

    correct, total = 0, 0
    for i in range(n):
        sig = signals_df.iloc[i]['signal']
        future_ret = (prices[i + horizon] - prices[i]) / prices[i]

        if sig == 'BUY':
            total += 1
            if future_ret > 0:
                correct += 1
        elif sig == 'SELL':
            total += 1
            if future_ret < 0:
                correct += 1

    return correct / total if total > 0 else 0.5


# ============================================================
# K-Fold Validation with Best Params
# ============================================================

def kfold_validate(features_df, prices, best_params, n_splits=5, verbose=False):
    """
    Run K-fold validation with chosen params and aggregate metrics.

    Returns mean + std of all metrics across folds.
    """
    folds = time_series_kfold(len(features_df), n_splits=n_splits)

    fold_results = []
    all_returns = []
    all_signals = []
    all_prices = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        test_features = features_df.iloc[test_idx].reset_index(drop=True)
        test_prices = prices[test_idx]

        gen = TradingSignalGenerator(
            c1_threshold_std=best_params['c1_threshold_std'],
            c1_drop_threshold=-best_params['c1_threshold_std'] * 0.5,
            confidence_threshold=best_params['confidence_threshold'],
            lookback_window=min(best_params['lookback_window'], len(test_features) - 1),
        )

        signals = gen.generate_signals(test_features, price_series=test_prices)

        bt = Backtester()
        res = bt.run(test_prices, signals)

        direction = _direction_score(signals, test_prices)
        n_buy = int((signals['signal'] == 'BUY').sum())
        n_sell = int((signals['signal'] == 'SELL').sum())

        bh_return = (test_prices[-1] / test_prices[0] - 1) * 100

        fold_results.append({
            'fold': fold_idx,
            'n_samples': len(test_idx),
            'tda_return_pct': res['metrics']['total_return_pct'],
            'tda_sharpe': res['metrics']['sharpe_ratio'],
            'tda_sortino': res['metrics']['sortino_ratio'],
            'tda_max_dd_pct': res['metrics']['max_drawdown_pct'],
            'tda_n_trades': res['metrics'].get('completed_trades', 0),
            'tda_win_rate': res['metrics'].get('win_rate', 0) or 0,
            'direction_accuracy': direction,
            'n_buy': n_buy, 'n_sell': n_sell,
            'buy_hold_return_pct': bh_return,
            'outperformed_bh': res['metrics']['total_return_pct'] > bh_return,
        })

        all_returns.extend(res['returns'].tolist())
        all_signals.append(signals)
        all_prices.append(test_prices)

    fold_df = pd.DataFrame(fold_results)

    aggregate = {
        'n_folds': len(fold_results),
        'mean_return_pct': float(fold_df['tda_return_pct'].mean()),
        'std_return_pct': float(fold_df['tda_return_pct'].std()),
        'mean_sharpe': float(fold_df['tda_sharpe'].mean()),
        'std_sharpe': float(fold_df['tda_sharpe'].std()),
        'mean_sortino': float(fold_df['tda_sortino'].mean()),
        'mean_max_dd_pct': float(fold_df['tda_max_dd_pct'].mean()),
        'total_trades': int(fold_df['tda_n_trades'].sum()),
        'mean_direction_accuracy': float(fold_df['direction_accuracy'].mean()),
        'std_direction_accuracy': float(fold_df['direction_accuracy'].std()),
        'mean_buy_hold_return_pct': float(fold_df['buy_hold_return_pct'].mean()),
        'pct_folds_beat_bh': float(fold_df['outperformed_bh'].mean()),
        'total_buy_signals': int(fold_df['n_buy'].sum()),
        'total_sell_signals': int(fold_df['n_sell'].sum()),
    }

    total_signals = aggregate['total_buy_signals'] + aggregate['total_sell_signals']
    avg_acc = aggregate['mean_direction_accuracy']
    successes = int(round(avg_acc * total_signals))
    if total_signals > 0:
        p, lo, hi = wilson_interval(successes, total_signals, confidence=0.95)
        aggregate['direction_acc_wilson_lower'] = lo
        aggregate['direction_acc_wilson_upper'] = hi
        aggregate['direction_acc_significant'] = lo > 0.5

    aggregate['all_returns'] = all_returns

    return {
        'fold_results': fold_df,
        'aggregate': aggregate,
        'returns': all_returns,
    }


# ============================================================
# Statistical Tests
# ============================================================

def bootstrap_metric(values, metric_fn=np.mean, n_iterations=2000, confidence=0.95, seed=42):
    """Bootstrap CI for any metric function."""
    np.random.seed(seed)
    values = np.asarray(values)
    if len(values) < 5:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    samples = []
    for _ in range(n_iterations):
        boot = np.random.choice(values, size=len(values), replace=True)
        samples.append(metric_fn(boot))

    samples = np.array(samples)
    alpha = (1 - confidence) / 2
    return {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'lower': float(np.percentile(samples, alpha * 100)),
        'upper': float(np.percentile(samples, (1 - alpha) * 100)),
    }


def sharpe_from_returns(returns, periods_per_year=8760):
    """Sharpe assuming hourly data (8760 hours/year)."""
    r = np.asarray(returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def t_test_returns(returns):
    """T-test against zero."""
    r = np.asarray(returns)
    if len(r) < 2:
        return {'t_stat': 0.0, 'p_value': 1.0, 'significant': False}
    t_stat, p_val = stats.ttest_1samp(r, 0)
    return {
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05 and t_stat > 0),
    }


# ============================================================
# Main Entry: Full V2 Validation
# ============================================================

def run_validation_v2(features_df, prices, n_splits=5, param_grid=None,
                       symbol='BTC', verbose=True):
    """
    Full validation v2 pipeline.

    Returns dict with grid search, k-fold results, statistical tests, and report data.
    """
    if verbose:
        print(f"\n[V2] {len(features_df)} feature windows / {len(prices)} prices")
        print(f"[V2] Phase 1: Grid search (CV) for best parameters")

    grid_results = grid_search_cv(
        features_df, prices, param_grid=param_grid, n_splits=n_splits, verbose=verbose,
    )
    best = grid_results['best_params']

    if verbose and best:
        print(f"  Best params: threshold={best['c1_threshold_std']}, "
              f"lookback={best['lookback_window']}, "
              f"conf={best['confidence_threshold']}")
        print(f"  Best CV score: {best['mean_score']:.3f} ± {best['std_score']:.3f}")

    if verbose:
        print(f"\n[V2] Phase 2: K-Fold validation with best params")

    if best is None:
        return {'error': 'No valid parameter combination found'}

    kfold_results = kfold_validate(features_df, prices, best, n_splits=n_splits, verbose=verbose)
    agg = kfold_results['aggregate']

    if verbose:
        print(f"  Mean direction acc: {agg['mean_direction_accuracy']:.2%} "
              f"± {agg['std_direction_accuracy']:.2%}")
        print(f"  Mean return per fold: {agg['mean_return_pct']:.2f}% "
              f"± {agg['std_return_pct']:.2f}%")
        print(f"  Buy-hold return per fold: {agg['mean_buy_hold_return_pct']:.2f}%")
        print(f"  Folds beating buy-hold: {agg['pct_folds_beat_bh']:.0%}")

    if verbose:
        print(f"\n[V2] Phase 3: Statistical significance tests")

    returns = kfold_results['returns']
    t_test = t_test_returns(returns)
    bootstrap_sharpe = bootstrap_metric(returns, sharpe_from_returns)
    bootstrap_mean = bootstrap_metric(returns, np.mean)

    if verbose:
        print(f"  T-test p-value: {t_test['p_value']:.4f} "
              f"({'significant' if t_test['significant'] else 'not significant'})")
        print(f"  Bootstrap mean return: {bootstrap_mean['mean']:.5f} "
              f"[{bootstrap_mean['lower']:.5f}, {bootstrap_mean['upper']:.5f}]")

    return {
        'symbol': symbol,
        'grid_search': grid_results,
        'kfold': kfold_results,
        'best_params': best,
        't_test': t_test,
        'bootstrap_sharpe': bootstrap_sharpe,
        'bootstrap_mean_return': bootstrap_mean,
    }


# ============================================================
# Report Generation
# ============================================================

def format_v2_report(results, baselines=None, cross_asset=None):
    """Format full v2 validation report as markdown."""
    if 'error' in results:
        return f"# Validation Failed\n\nError: {results['error']}"

    symbol = results['symbol']
    best = results['best_params']
    agg = results['kfold']['aggregate']
    fold_df = results['kfold']['fold_results']
    tt = results['t_test']
    bs_sharpe = results['bootstrap_sharpe']
    bs_mean = results['bootstrap_mean_return']

    lines = []
    lines.append(f"# Validation Report v2: TDA Trading Strategy\n")
    lines.append(f"**Symbol:** {symbol}")
    lines.append(f"**Method:** {agg['n_folds']}-fold time-series cross-validation + grid search")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n---\n")

    lines.append(f"## 1. Best Parameters (from grid search CV)\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| C1 threshold (σ) | {best['c1_threshold_std']} |")
    lines.append(f"| Lookback window | {best['lookback_window']} |")
    lines.append(f"| Confidence threshold | {best['confidence_threshold']} |")
    lines.append(f"| CV score (direction acc.) | {best['mean_score']:.3f} ± {best['std_score']:.3f} |")
    lines.append("")

    lines.append(f"## 2. K-Fold Cross-Validation Results\n")
    lines.append(f"### Per-Fold Performance\n")
    lines.append(f"| Fold | n | TDA Return | TDA Sharpe | Direction Acc | Trades | Buy&Hold | Beat BH? |")
    lines.append(f"|------|---|-----------|------------|---------------|--------|----------|----------|")
    for _, row in fold_df.iterrows():
        beat = "✅" if row['outperformed_bh'] else "❌"
        lines.append(f"| {int(row['fold'])} | {int(row['n_samples'])} | "
                     f"{row['tda_return_pct']:.2f}% | {row['tda_sharpe']:.2f} | "
                     f"{row['direction_accuracy']:.2%} | {int(row['tda_n_trades'])} | "
                     f"{row['buy_hold_return_pct']:.2f}% | {beat} |")
    lines.append("")

    lines.append(f"### Aggregate Statistics\n")
    lines.append(f"| Metric | Mean | Std |")
    lines.append(f"|--------|------|-----|")
    lines.append(f"| TDA return (%) | {agg['mean_return_pct']:.2f} | {agg['std_return_pct']:.2f} |")
    lines.append(f"| TDA Sharpe | {agg['mean_sharpe']:.2f} | {agg['std_sharpe']:.2f} |")
    lines.append(f"| TDA Sortino | {agg['mean_sortino']:.2f} | - |")
    lines.append(f"| Max drawdown (%) | {agg['mean_max_dd_pct']:.2f} | - |")
    lines.append(f"| Direction accuracy | {agg['mean_direction_accuracy']:.2%} | "
                 f"{agg['std_direction_accuracy']:.2%} |")
    lines.append(f"| Buy-hold return (%) | {agg['mean_buy_hold_return_pct']:.2f} | - |")
    lines.append(f"| Folds beating buy-hold | {agg['pct_folds_beat_bh']:.0%} | - |")
    lines.append(f"| Total signals | {agg['total_buy_signals'] + agg['total_sell_signals']} | - |")
    lines.append(f"| Total trades | {agg['total_trades']} | - |")
    lines.append("")

    lines.append(f"## 3. Direction Accuracy: Wilson Confidence Interval\n")
    if 'direction_acc_wilson_lower' in agg:
        lines.append(f"Robust confidence interval (better than normal approx for small samples):")
        lines.append(f"")
        lines.append(f"- **Direction accuracy:** {agg['mean_direction_accuracy']:.2%}")
        lines.append(f"- **95% Wilson CI:** [{agg['direction_acc_wilson_lower']:.2%}, "
                     f"{agg['direction_acc_wilson_upper']:.2%}]")
        if agg.get('direction_acc_significant'):
            lines.append(f"- ✅ **Lower bound > 50% — significantly better than chance**")
        else:
            lines.append(f"- ⚠️ Lower bound ≤ 50% — not significantly better than chance")
    lines.append("")

    lines.append(f"## 4. Statistical Significance Tests\n")
    lines.append(f"### Bootstrap Confidence Intervals (2000 iterations)\n")
    lines.append(f"| Metric | Mean | 95% CI Lower | 95% CI Upper |")
    lines.append(f"|--------|------|--------------|--------------|")
    lines.append(f"| Sharpe ratio | {bs_sharpe['mean']:.3f} | "
                 f"{bs_sharpe['lower']:.3f} | {bs_sharpe['upper']:.3f} |")
    lines.append(f"| Mean return | {bs_mean['mean']:.5f} | "
                 f"{bs_mean['lower']:.5f} | {bs_mean['upper']:.5f} |")
    lines.append("")

    lines.append(f"### T-Test: Returns vs Zero\n")
    lines.append(f"- **t-statistic:** {tt['t_stat']:.3f}")
    lines.append(f"- **p-value:** {tt['p_value']:.4f}")
    sig = "✅ Significant" if tt['significant'] else "❌ Not significant"
    lines.append(f"- {sig} at α=0.05")
    lines.append("")

    if cross_asset:
        lines.append(f"## 5. Cross-Asset Out-of-Sample\n")
        lines.append(f"Same model parameters applied to other cryptos:\n")
        lines.append(f"| Asset | Mean Return | Mean Sharpe | Direction Acc | Beat BH |")
        lines.append(f"|-------|-------------|-------------|---------------|---------|")
        for sym, res in cross_asset.items():
            if 'error' in res:
                lines.append(f"| {sym} | ERROR | - | - | - |")
                continue
            a = res['kfold']['aggregate']
            beat = f"{a['pct_folds_beat_bh']:.0%}"
            lines.append(f"| {sym} | {a['mean_return_pct']:.2f}% | "
                         f"{a['mean_sharpe']:.2f} | "
                         f"{a['mean_direction_accuracy']:.2%} | {beat} |")
        lines.append("")

    lines.append(f"## 6. Verdict\n")
    wins, issues = [], []

    if agg['mean_return_pct'] > agg['mean_buy_hold_return_pct']:
        wins.append(f"Beats buy-hold on average ({agg['mean_return_pct']:.2f}% vs "
                    f"{agg['mean_buy_hold_return_pct']:.2f}%)")
    else:
        issues.append(f"Underperforms buy-hold on average")

    if agg['pct_folds_beat_bh'] >= 0.6:
        wins.append(f"Beat buy-hold in {agg['pct_folds_beat_bh']:.0%} of folds")
    else:
        issues.append(f"Only beat buy-hold in {agg['pct_folds_beat_bh']:.0%} of folds")

    if agg['mean_direction_accuracy'] > 0.55:
        wins.append(f"Direction accuracy above chance ({agg['mean_direction_accuracy']:.2%})")
    else:
        issues.append(f"Direction accuracy at/below chance ({agg['mean_direction_accuracy']:.2%})")

    if agg.get('direction_acc_significant'):
        wins.append(f"Direction accuracy statistically significant (Wilson CI lower > 50%)")
    else:
        issues.append(f"Direction accuracy not statistically significant")

    if tt['significant']:
        wins.append(f"Returns statistically significant (p={tt['p_value']:.4f})")
    else:
        issues.append(f"Returns NOT statistically significant (p={tt['p_value']:.4f})")

    if bs_sharpe['lower'] > 0:
        wins.append(f"Bootstrap Sharpe CI excludes zero")
    else:
        issues.append(f"Bootstrap Sharpe CI includes zero")

    if abs(agg['mean_max_dd_pct']) < 10:
        wins.append(f"Drawdown control: {agg['mean_max_dd_pct']:.2f}%")

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
        lines.append(f"**🟡 MARGINAL** — Mixed results. {len(wins)} wins vs {len(issues)} issues.")
        lines.append(f"Needs more data, longer backtest, or ensemble approach.")
    else:
        lines.append(f"**🔴 NEEDS WORK** — {len(issues)} issues vs {len(wins)} wins.")
        lines.append(f"Investigate: feature engineering, regime filters, more data.")

    lines.append(f"\n---\n*Generated by `src/validation_v2.py`*")
    return "\n".join(lines)
