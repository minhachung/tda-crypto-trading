"""
Validation Framework for TDA Trading Strategy.

Methods:
  1. Walk-forward validation (no lookahead bias)
  2. Direction accuracy (does signal predict next-period direction?)
  3. Bootstrap confidence intervals (Sharpe, returns)
  4. Baseline comparisons (buy-hold, random, MA crossover)
  5. Statistical significance tests (t-test, p-value)
  6. Multi-asset out-of-sample testing
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from src.data_pipeline import run_pipeline
from src.persistent_homology import compute_features_for_windows
from src.trading_signals import TradingSignalGenerator
from src.backtester import Backtester


# ============================================================
# 1. DIRECTION ACCURACY
# ============================================================

def direction_accuracy(signals_df, prices, horizon=1):
    """
    Measure: when model says BUY, does price go UP next? When SELL, does it go DOWN?

    Returns: dict with accuracy for BUY signals, SELL signals, overall.
    """
    prices = np.asarray(prices)
    n = min(len(signals_df), len(prices) - horizon)

    if n <= 0:
        return {'overall_accuracy': 0.0, 'buy_accuracy': 0.0,
                'sell_accuracy': 0.0, 'n_buy': 0, 'n_sell': 0}

    correct_buy, total_buy = 0, 0
    correct_sell, total_sell = 0, 0

    for i in range(n):
        signal = signals_df.iloc[i]['signal']
        future_return = (prices[i + horizon] - prices[i]) / prices[i]

        if signal == 'BUY':
            total_buy += 1
            if future_return > 0:
                correct_buy += 1
        elif signal == 'SELL':
            total_sell += 1
            if future_return < 0:
                correct_sell += 1

    total = total_buy + total_sell
    correct = correct_buy + correct_sell

    return {
        'overall_accuracy': correct / total if total else 0.0,
        'buy_accuracy': correct_buy / total_buy if total_buy else 0.0,
        'sell_accuracy': correct_sell / total_sell if total_sell else 0.0,
        'n_buy': total_buy,
        'n_sell': total_sell,
        'n_signals': total,
    }


# ============================================================
# 2. BASELINE STRATEGIES
# ============================================================

def baseline_buy_hold(prices, initial_capital=10000):
    """Pure buy-and-hold."""
    units = initial_capital / prices[0]
    final = units * prices[-1]
    return {
        'final_equity': final,
        'total_return_pct': (final / initial_capital - 1) * 100,
        'sharpe_ratio': sharpe_ratio(pd.Series(prices).pct_change().fillna(0)),
        'strategy': 'buy_hold',
    }


def baseline_random(prices, n_signals=10, seed=42, initial_capital=10000):
    """Random buy/sell at random times."""
    np.random.seed(seed)
    n = len(prices)
    signal_indices = sorted(np.random.choice(n, min(n_signals, n), replace=False))

    df = pd.DataFrame({'signal': ['HOLD'] * n, 'position_size': [0.0] * n})
    for i, idx in enumerate(signal_indices):
        if i % 2 == 0:
            df.iloc[idx] = {'signal': 'BUY', 'position_size': 0.5}
        else:
            df.iloc[idx] = {'signal': 'SELL', 'position_size': -0.5}

    bt = Backtester(initial_capital=initial_capital)
    res = bt.run(prices, df)
    res['metrics']['strategy'] = 'random'
    return res['metrics']


def baseline_ma_crossover(prices, short_period=10, long_period=30, initial_capital=10000):
    """Classic moving average crossover (golden cross / death cross)."""
    prices_s = pd.Series(prices)
    short_ma = prices_s.rolling(window=short_period, min_periods=1).mean()
    long_ma = prices_s.rolling(window=long_period, min_periods=1).mean()

    signals = []
    sizes = []
    prev_above = None
    for i in range(len(prices)):
        curr_above = short_ma.iloc[i] > long_ma.iloc[i]
        if prev_above is None:
            signals.append('HOLD'); sizes.append(0.0)
        elif curr_above and not prev_above:
            signals.append('BUY'); sizes.append(0.5)
        elif not curr_above and prev_above:
            signals.append('SELL'); sizes.append(-0.5)
        else:
            signals.append('HOLD'); sizes.append(0.0)
        prev_above = curr_above

    df = pd.DataFrame({'signal': signals, 'position_size': sizes})
    bt = Backtester(initial_capital=initial_capital)
    res = bt.run(prices, df)
    res['metrics']['strategy'] = 'ma_crossover'
    return res['metrics']


# ============================================================
# 3. STATISTICAL TESTS
# ============================================================

def sharpe_ratio(returns, rf_annual=0.02, periods_per_year=365):
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf_annual / periods_per_year
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def bootstrap_sharpe(returns, n_iterations=1000, confidence=0.95, seed=42):
    """Bootstrap confidence interval for Sharpe ratio."""
    np.random.seed(seed)
    returns = np.asarray(returns)
    if len(returns) < 10:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    bootstrap_sharpes = []
    for _ in range(n_iterations):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_sharpes.append(sharpe_ratio(pd.Series(sample)))

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    alpha = (1 - confidence) / 2
    return {
        'mean': float(np.mean(bootstrap_sharpes)),
        'std': float(np.std(bootstrap_sharpes)),
        'lower': float(np.percentile(bootstrap_sharpes, alpha * 100)),
        'upper': float(np.percentile(bootstrap_sharpes, (1 - alpha) * 100)),
    }


def t_test_vs_zero(returns):
    """Test if returns are statistically different from zero."""
    if len(returns) < 2:
        return {'t_stat': 0.0, 'p_value': 1.0, 'significant': False}
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
    }


def t_test_vs_baseline(strategy_returns, baseline_returns):
    """Test if strategy returns differ significantly from baseline."""
    if len(strategy_returns) < 2 or len(baseline_returns) < 2:
        return {'t_stat': 0.0, 'p_value': 1.0, 'significant': False}
    t_stat, p_value = stats.ttest_ind(strategy_returns, baseline_returns)
    return {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
    }


# ============================================================
# 4. WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_split(features_df, prices, train_pct=0.7, val_pct=0.15):
    """Split into train / validation / test windows (no lookahead)."""
    n = len(features_df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return {
        'train_features': features_df.iloc[:train_end].reset_index(drop=True),
        'val_features': features_df.iloc[train_end:val_end].reset_index(drop=True),
        'test_features': features_df.iloc[val_end:].reset_index(drop=True),
        'train_prices': prices[:train_end],
        'val_prices': prices[train_end:val_end],
        'test_prices': prices[val_end:],
        'split_indices': (train_end, val_end, n),
    }


def walk_forward_validate(features_df, prices, threshold_grid=None, lookback=10):
    """
    Walk-forward validation:
      - Find best threshold on TRAIN data
      - Evaluate on VAL data
      - Final report on TEST data (out-of-sample)
    """
    if threshold_grid is None:
        threshold_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    splits = walk_forward_split(features_df, prices)
    results = {}

    print(f"  Train: {len(splits['train_features'])} windows | "
          f"Val: {len(splits['val_features'])} | "
          f"Test: {len(splits['test_features'])}")

    print(f"  Tuning threshold on validation set...")
    best_threshold = 1.0
    best_val_sharpe = -np.inf

    for threshold in threshold_grid:
        gen = TradingSignalGenerator(
            c1_threshold_std=threshold,
            c1_drop_threshold=-threshold * 0.5,
            confidence_threshold=0.3,
            lookback_window=lookback,
        )
        val_signals = gen.generate_signals(
            splits['val_features'], price_series=splits['val_prices']
        )
        bt = Backtester()
        if len(splits['val_prices']) >= len(val_signals):
            val_prices_aligned = splits['val_prices'][-len(val_signals):]
            val_results = bt.run(val_prices_aligned, val_signals)
            sharpe = val_results['metrics']['sharpe_ratio']
            if sharpe > best_val_sharpe:
                best_val_sharpe = sharpe
                best_threshold = threshold

    print(f"  Best threshold: {best_threshold} (val Sharpe: {best_val_sharpe:.2f})")
    results['best_threshold'] = best_threshold
    results['val_sharpe'] = best_val_sharpe

    print(f"  Evaluating on test set (out-of-sample)...")
    gen = TradingSignalGenerator(
        c1_threshold_std=best_threshold,
        c1_drop_threshold=-best_threshold * 0.5,
        confidence_threshold=0.3,
        lookback_window=lookback,
    )
    test_signals = gen.generate_signals(
        splits['test_features'], price_series=splits['test_prices']
    )

    if len(splits['test_prices']) >= len(test_signals):
        test_prices_aligned = splits['test_prices'][-len(test_signals):]
    else:
        test_prices_aligned = splits['test_prices']
        test_signals = test_signals.iloc[:len(test_prices_aligned)].reset_index(drop=True)

    bt = Backtester()
    test_results = bt.run(test_prices_aligned, test_signals)

    direction = direction_accuracy(test_signals, test_prices_aligned)

    bh = baseline_buy_hold(test_prices_aligned)

    results['test_metrics'] = test_results['metrics']
    results['direction_accuracy'] = direction
    results['buy_hold_baseline'] = bh
    results['returns'] = test_results['returns'].tolist()
    results['equity_curve'] = test_results['equity_curve'].tolist()

    bootstrap_ci = bootstrap_sharpe(test_results['returns'].values)
    results['bootstrap_sharpe'] = bootstrap_ci

    t_test = t_test_vs_zero(test_results['returns'].values)
    results['t_test_vs_zero'] = t_test

    return results


# ============================================================
# 5. MULTI-ASSET OUT-OF-SAMPLE
# ============================================================

def cross_asset_validation(symbols, days=365, threshold=1.0, lookback=10):
    """Test the strategy on multiple assets."""
    results = {}

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")
        try:
            data = run_pipeline(symbol=symbol, days=days, window_size=10, stride=1)
            df = data['df']
            point_clouds = data['point_clouds']
            end_indices = data['end_indices']

            features_df = compute_features_for_windows(
                point_clouds, end_indices=end_indices, verbose=False
            )

            gen = TradingSignalGenerator(
                c1_threshold_std=threshold,
                c1_drop_threshold=-threshold * 0.5,
                confidence_threshold=0.3,
                lookback_window=lookback,
            )
            signals_df = gen.generate_signals(features_df, price_series=df['close'].values)

            if 'end_idx' in signals_df.columns:
                idx = signals_df['end_idx'].astype(int).values
                idx = idx[idx < len(df)]
                prices_aligned = df['close'].values[idx]
                signals_df = signals_df.iloc[:len(prices_aligned)].reset_index(drop=True)
            else:
                prices_aligned = df['close'].values[-len(signals_df):]

            bt = Backtester()
            bt_results = bt.run(prices_aligned, signals_df)

            direction = direction_accuracy(signals_df, prices_aligned)
            bh = baseline_buy_hold(prices_aligned)
            ma = baseline_ma_crossover(prices_aligned)

            results[symbol] = {
                'tda_metrics': bt_results['metrics'],
                'direction': direction,
                'buy_hold': bh,
                'ma_crossover': ma,
                'n_windows': len(signals_df),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            results[symbol] = {'error': str(e)}

    return results


# ============================================================
# 6. REPORT FORMATTING
# ============================================================

def format_validation_report(symbol, wf_results, baselines, cross_asset):
    """Format full validation report as markdown."""
    lines = []
    lines.append(f"# Validation Report: TDA Trading Strategy")
    lines.append(f"\n**Primary symbol:** {symbol}")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n---\n")

    lines.append(f"## 1. Walk-Forward Validation (Out-of-Sample)\n")
    lines.append(f"Trained threshold on validation set, evaluated on held-out test set.\n")
    lines.append(f"- **Best threshold (from val):** {wf_results['best_threshold']}")
    lines.append(f"- **Val Sharpe:** {wf_results['val_sharpe']:.2f}\n")

    tm = wf_results['test_metrics']
    bh = wf_results['buy_hold_baseline']
    lines.append(f"### Test Set Performance\n")
    lines.append(f"| Metric | TDA Strategy | Buy & Hold |")
    lines.append(f"|--------|------|------|")
    lines.append(f"| Total Return | {tm['total_return_pct']:.2f}% | {bh['total_return_pct']:.2f}% |")
    lines.append(f"| Sharpe Ratio | {tm['sharpe_ratio']:.2f} | {bh['sharpe_ratio']:.2f} |")
    lines.append(f"| Sortino Ratio | {tm['sortino_ratio']:.2f} | - |")
    lines.append(f"| Max Drawdown | {tm['max_drawdown_pct']:.2f}% | - |")
    lines.append(f"| Trades | {tm.get('completed_trades', 0)} | 0 |")
    if tm.get('win_rate') is not None:
        lines.append(f"| Win Rate | {tm.get('win_rate', 0):.2%} | - |")
    lines.append("")

    lines.append(f"## 2. Direction Accuracy (Signal Quality)\n")
    da = wf_results['direction_accuracy']
    lines.append(f"Does the model correctly predict next-period direction?\n")
    lines.append(f"| Signal Type | Count | Accuracy |")
    lines.append(f"|-------------|-------|----------|")
    lines.append(f"| BUY signals  | {da['n_buy']} | {da['buy_accuracy']:.2%} |")
    lines.append(f"| SELL signals | {da['n_sell']} | {da['sell_accuracy']:.2%} |")
    lines.append(f"| **Overall**  | **{da['n_signals']}** | **{da['overall_accuracy']:.2%}** |")
    lines.append(f"\nRandom baseline accuracy = 50%. Above 50% = better than chance.\n")

    lines.append(f"## 3. Statistical Significance\n")
    bs = wf_results['bootstrap_sharpe']
    lines.append(f"### Bootstrap Sharpe Confidence Interval (1000 iterations)")
    lines.append(f"- **Mean Sharpe:** {bs['mean']:.3f}")
    lines.append(f"- **Std:** {bs['std']:.3f}")
    lines.append(f"- **95% CI:** [{bs['lower']:.3f}, {bs['upper']:.3f}]")
    if bs['lower'] > 0:
        lines.append(f"- ✅ Lower bound > 0: Sharpe is significantly positive\n")
    else:
        lines.append(f"- ⚠️ Lower bound ≤ 0: Sharpe not statistically positive\n")

    tt = wf_results['t_test_vs_zero']
    lines.append(f"### T-Test: Returns vs Zero")
    lines.append(f"- **t-statistic:** {tt['t_stat']:.3f}")
    lines.append(f"- **p-value:** {tt['p_value']:.4f}")
    sig_emoji = "✅" if tt['significant'] else "❌"
    lines.append(f"- {sig_emoji} {'Significant' if tt['significant'] else 'NOT significant'} at α=0.05\n")

    lines.append(f"## 4. Baseline Comparison\n")
    lines.append(f"| Strategy | Total Return | Sharpe |")
    lines.append(f"|----------|--------------|--------|")
    lines.append(f"| **TDA (test set)** | **{tm['total_return_pct']:.2f}%** | **{tm['sharpe_ratio']:.2f}** |")
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['total_return_pct']:.2f}% | {m['sharpe_ratio']:.2f} |")
    lines.append("")

    if cross_asset:
        lines.append(f"## 5. Cross-Asset Out-of-Sample\n")
        lines.append(f"Same model parameters applied to different cryptos.\n")
        lines.append(f"| Asset | TDA Return | TDA Sharpe | Buy&Hold Return | MA Crossover | Direction Acc. |")
        lines.append(f"|-------|------------|------------|-----------------|--------------|----------------|")
        for sym, res in cross_asset.items():
            if 'error' in res:
                lines.append(f"| {sym} | ERROR | - | - | - | - |")
                continue
            tda = res['tda_metrics']
            bh = res['buy_hold']
            ma = res['ma_crossover']
            da = res['direction']
            lines.append(f"| {sym} | {tda['total_return_pct']:.2f}% | "
                        f"{tda['sharpe_ratio']:.2f} | "
                        f"{bh['total_return_pct']:.2f}% | "
                        f"{ma['total_return_pct']:.2f}% | "
                        f"{da['overall_accuracy']:.2%} |")
        lines.append("")

    lines.append(f"## 6. Verdict\n")
    issues = []
    wins = []
    if tm['sharpe_ratio'] > bh['sharpe_ratio']:
        wins.append(f"Beats buy-hold Sharpe ({tm['sharpe_ratio']:.2f} vs {bh['sharpe_ratio']:.2f})")
    else:
        issues.append(f"Underperforms buy-hold Sharpe")
    if da['overall_accuracy'] > 0.55:
        wins.append(f"Direction accuracy above chance ({da['overall_accuracy']:.2%})")
    else:
        issues.append(f"Direction accuracy at/below chance ({da['overall_accuracy']:.2%})")
    if tt['significant']:
        wins.append(f"Returns statistically significant (p={tt['p_value']:.4f})")
    else:
        issues.append(f"Returns NOT statistically significant (p={tt['p_value']:.4f})")
    if bs['lower'] > 0:
        wins.append(f"Bootstrap Sharpe CI excludes zero")
    else:
        issues.append(f"Bootstrap Sharpe CI includes zero")
    if abs(tm.get('max_drawdown_pct', 0)) < abs(bh.get('total_return_pct', 0) if bh['total_return_pct'] < 0 else 100):
        wins.append(f"Drawdown control: {tm['max_drawdown_pct']:.2f}%")

    lines.append(f"### ✅ Strengths")
    if wins:
        for w in wins:
            lines.append(f"- {w}")
    else:
        lines.append(f"- None identified in this run.")

    lines.append(f"\n### ⚠️ Weaknesses")
    if issues:
        for i in issues:
            lines.append(f"- {i}")
    else:
        lines.append(f"- None identified.")

    lines.append(f"\n### Recommendation")
    score = len(wins) - len(issues)
    if score >= 3:
        lines.append(f"**🟢 PASS** — Strategy shows promise. {len(wins)} wins vs {len(issues)} weaknesses.")
        lines.append(f"Consider: more data, longer backtest, paper trading.")
    elif score >= 0:
        lines.append(f"**🟡 MARGINAL** — Mixed signals. {len(wins)} wins vs {len(issues)} weaknesses.")
        lines.append(f"Needs: parameter tuning, more data, ensemble with other features.")
    else:
        lines.append(f"**🔴 FAIL** — Strategy not validated. {len(issues)} weaknesses vs {len(wins)} wins.")
        lines.append(f"Don't trade live. Investigate: feature engineering, regime filters, ML overlay.")

    lines.append(f"\n---\n*Generated by `src/validation.py`*")

    return "\n".join(lines)


if __name__ == '__main__':
    print("Run: python examples/run_validation.py")
