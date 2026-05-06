#!/usr/bin/env python
"""
V7: Profitability test — focused ADA + SOL + 7d horizon + low fees + 365 days.

Tests the hypothesis from the v6 paper:
  "If you trade only the high-accuracy assets (ADA, SOL) at the 7-day horizon
   with strict probability threshold (>0.70) on a low-fee venue,
   the model crosses the profitability threshold."

Also addresses the 90-day sample limitation by:
  1. Fetching 365 days of hourly data (covers multiple regimes)
  2. Splitting into 3 regime sub-samples (early/middle/late)
  3. Reporting per-regime metrics so we know if the signal survives
     across different market conditions
  4. Running with realistic Binance.US maker fees (0.075%) AND Coinbase taker (0.4%)

Output:
  - results/V7_PROFITABILITY.md — markdown report
  - results/V7_PROFITABILITY.pdf — formatted PDF (built afterward)
  - results/v7_per_regime.csv — per-regime breakdown

Usage:
  python examples/run_validation_v7.py 365   # 1-year hourly = 8760 samples per asset
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
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.multi_asset_pipeline import (
    fetch_multi_asset_pool,
    get_combined_feature_cols,
)
from src.regime_filter import apply_regime_filter
from src.validation_v2 import wilson_interval, time_series_kfold
from src.backtester import Backtester


def make_classifier(model_type='rf', random_state=42):
    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=random_state, n_jobs=-1,
        )
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def add_targets_horizon(df, horizon=168, price_col='close'):
    df = df.copy()
    if 'symbol' in df.columns:
        df['target'] = (
            df.groupby('symbol')[price_col]
            .transform(lambda x: (x.shift(-horizon) > x).astype(float))
        )
    else:
        df['target'] = (df[price_col].shift(-horizon) > df[price_col]).astype(float)
    return df.dropna(subset=['target']).reset_index(drop=True)


def signals_from_proba(proba, prob_threshold=0.70, max_position_size=0.10):
    signals, sizes = [], []
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
    return pd.DataFrame({'signal': signals, 'position_size': sizes, 'p_up': proba})


def split_into_regimes(df, n_regimes=3):
    """Split each asset's time series into n equal-time regimes."""
    df = df.copy().sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    df['regime'] = -1
    for sym, group in df.groupby('symbol', sort=False):
        n = len(group)
        idx = group.index.values
        chunk = n // n_regimes
        for r in range(n_regimes):
            start = r * chunk
            end = (r + 1) * chunk if r < n_regimes - 1 else n
            df.loc[idx[start:end], 'regime'] = r
    return df


def evaluate_regime(pooled_df, regime_id, feature_cols, model_type='rf',
                    prob_threshold=0.70, horizon=168, n_splits=5,
                    regime_filter=None, trade_fee=0.00075, slippage=0.0005):
    """Evaluate within a single regime sub-sample using k-fold CV."""
    df = pooled_df[pooled_df['regime'] == regime_id].copy()
    df = add_targets_horizon(df, horizon=horizon)
    symbols = df['symbol'].unique().tolist()

    asset_folds = {}
    for symbol in symbols:
        asset_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(asset_df) < n_splits + 50:
            continue
        folds = time_series_kfold(len(asset_df), n_splits=n_splits)
        asset_folds[symbol] = (asset_df, folds)

    rows = []

    for fold_idx in range(n_splits):
        train_X_list, train_y_list = [], []
        for symbol in symbols:
            if symbol not in asset_folds:
                continue
            asset_df, folds = asset_folds[symbol]
            if fold_idx >= len(folds):
                continue
            tr_idx, _ = folds[fold_idx]
            train_X_list.append(asset_df.iloc[tr_idx][feature_cols].values)
            train_y_list.append(asset_df.iloc[tr_idx]['target'].values)
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
            if symbol not in asset_folds:
                continue
            asset_df, folds = asset_folds[symbol]
            if fold_idx >= len(folds):
                continue
            _, te_idx = folds[fold_idx]
            test_df = asset_df.iloc[te_idx].reset_index(drop=True)
            X_test = test_df[feature_cols].values
            valid = ~np.any(np.isnan(X_test), axis=1)
            if valid.sum() == 0:
                continue
            test_df = test_df.iloc[valid].reset_index(drop=True)
            X_test = X_test[valid]
            p_up = clf.predict_proba(X_test)[:, 1]
            signals = signals_from_proba(p_up, prob_threshold=prob_threshold)
            if regime_filter:
                signals = apply_regime_filter(signals, test_df, regime_filter)

            test_prices = test_df['close'].values
            true_targets = test_df['target'].values.astype(int)

            correct, total = 0, 0
            for i in range(min(len(signals), len(true_targets))):
                sig = signals.iloc[i]['signal']
                if sig == 'BUY':
                    total += 1
                    correct += int(true_targets[i] == 1)
                elif sig == 'SELL':
                    total += 1
                    correct += int(true_targets[i] == 0)
            acc = correct / total if total else 0.5

            bt = Backtester(trade_fee=trade_fee, slippage=slippage)
            res = bt.run(test_prices, signals)
            bh = (test_prices[-1] / test_prices[0] - 1) * 100

            rows.append({
                'regime': regime_id,
                'fold': fold_idx,
                'symbol': symbol,
                'n_test': len(test_df),
                'n_signals': total,
                'direction_accuracy': acc,
                'tda_return_pct': res['metrics']['total_return_pct'],
                'tda_sharpe': res['metrics']['sharpe_ratio'],
                'tda_n_trades': res['metrics'].get('completed_trades', 0),
                'tda_win_rate': res['metrics'].get('win_rate', 0) or 0,
                'tda_max_dd_pct': res['metrics']['max_drawdown_pct'],
                'buy_hold_return_pct': bh,
                'outperformed_bh': res['metrics']['total_return_pct'] > bh,
            })
    return pd.DataFrame(rows)


def run_v7(symbols=None, days=365, horizon_periods=168, prob_threshold=0.70):
    if symbols is None:
        symbols = ['ADA', 'SOL']

    print(f"\n{'#' * 70}")
    print(f"#  V7: Profitability Test")
    print(f"#  Assets: {symbols} (focused on best-performing v6 cohort)")
    print(f"#  Horizon: {horizon_periods}h ({horizon_periods/24:.1f}d)")
    print(f"#  Prob threshold: {prob_threshold}")
    print(f"#  Days: {days} (3-regime split for robustness check)")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/4] Fetching {days} days of hourly data...")
    pooled_df = fetch_multi_asset_pool(symbols, days=days, interval='1h')
    pooled_df = split_into_regimes(pooled_df, n_regimes=3)
    print(f"  Pool size: {len(pooled_df)}")
    for r in [0, 1, 2]:
        n = (pooled_df['regime'] == r).sum()
        ts = pooled_df[pooled_df['regime'] == r]['timestamp']
        print(f"  Regime {r}: {n} samples, {ts.min().date()} -> {ts.max().date()}")

    feature_cols = get_combined_feature_cols(pooled_df)
    print(f"  Features: {len(feature_cols)}")

    print(f"\n[2/4] Evaluating with TWO fee scenarios:")
    print(f"  Scenario A: Coinbase taker (0.40% fee + 0.05% slippage)")
    print(f"  Scenario B: Binance.US maker (0.075% fee + 0.05% slippage)")

    results = {}
    for scenario, (fee, slip) in {
        'coinbase_taker': (0.004, 0.0005),
        'binance_maker':  (0.00075, 0.0005),
    }.items():
        print(f"\n  --- Scenario: {scenario} (fee={fee*100:.3f}%, slip={slip*100:.2f}%) ---")
        per_regime = []
        for r in [0, 1, 2]:
            print(f"    Regime {r}...")
            fold_df = evaluate_regime(
                pooled_df, regime_id=r, feature_cols=feature_cols,
                model_type='rf', prob_threshold=prob_threshold,
                horizon=horizon_periods, n_splits=5,
                regime_filter={'vol': 'median'},
                trade_fee=fee, slippage=slip,
            )
            if len(fold_df) > 0:
                per_regime.append(fold_df)
        all_folds = pd.concat(per_regime, ignore_index=True) if per_regime else pd.DataFrame()
        results[scenario] = all_folds
        if len(all_folds) > 0:
            n_sig = int(all_folds['n_signals'].sum())
            avg_acc = float(all_folds['direction_accuracy'].mean())
            avg_sharpe = float(all_folds['tda_sharpe'].mean())
            avg_ret = float(all_folds['tda_return_pct'].mean())
            print(f"    Total signals: {n_sig}")
            print(f"    Mean direction accuracy: {avg_acc:.2%}")
            print(f"    Mean Sharpe: {avg_sharpe:.2f}")
            print(f"    Mean return per fold: {avg_ret:.2f}%")

    print(f"\n[3/4] Generating per-regime + per-asset breakdown...")
    by_regime = {}
    for scenario, fold_df in results.items():
        if len(fold_df) == 0:
            continue
        by_regime[scenario] = fold_df.groupby(['regime', 'symbol']).agg({
            'direction_accuracy': 'mean',
            'tda_return_pct': 'mean',
            'tda_sharpe': 'mean',
            'tda_n_trades': 'sum',
            'n_signals': 'sum',
            'buy_hold_return_pct': 'mean',
            'outperformed_bh': 'mean',
        }).reset_index()

    print(f"\n[4/4] Writing report and CSV...")
    write_v7_report(results, by_regime, symbols, days, horizon_periods, prob_threshold)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"\n{'#' * 70}")
    print(f"#  V7 Complete — see results/V7_PROFITABILITY.md")
    print(f"{'#' * 70}\n")

    return results, by_regime


def write_v7_report(results, by_regime, symbols, days, horizon, prob_threshold):
    os.makedirs('results', exist_ok=True)

    rows = []
    for scen, fold_df in results.items():
        if len(fold_df) == 0:
            continue
        rows.append({
            'scenario': scen,
            'n_signals': int(fold_df['n_signals'].sum()),
            'direction_accuracy': float(fold_df['direction_accuracy'].mean()),
            'mean_sharpe': float(fold_df['tda_sharpe'].mean()),
            'mean_return_pct': float(fold_df['tda_return_pct'].mean()),
            'mean_buy_hold_pct': float(fold_df['buy_hold_return_pct'].mean()),
            'pct_beat_bh': float(fold_df['outperformed_bh'].mean()),
            'total_trades': int(fold_df['tda_n_trades'].sum()),
            'mean_max_dd': float(fold_df['tda_max_dd_pct'].mean()),
        })
    summary = pd.DataFrame(rows)

    md = []
    md.append(f"# V7 Profitability Test\n")
    md.append(f"**Hypothesis:** Profitability is achievable on ADA+SOL at 7d horizon "
              f"with p>{prob_threshold} threshold + low-fee venue.\n")
    md.append(f"**Sample:** {days} days of hourly data, split into 3 equal-time regimes "
              f"to test cross-regime robustness.\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

    md.append(f"## 1. Summary\n")
    md.append(f"| Scenario | n Signals | Direction Acc. | Mean Sharpe | Mean Return | Beat B&H | Max DD |")
    md.append(f"|----------|----------:|---------------:|------------:|------------:|---------:|-------:|")
    for _, r in summary.iterrows():
        md.append(f"| {r['scenario']} | {int(r['n_signals'])} | "
                  f"{r['direction_accuracy']:.2%} | {r['mean_sharpe']:.2f} | "
                  f"{r['mean_return_pct']:.2f}% | {r['pct_beat_bh']:.0%} | "
                  f"{r['mean_max_dd']:.2f}% |")
    md.append("")

    md.append(f"## 2. Per-Regime Stability\n")
    md.append(f"This is the critical 90-day-sample fix. We split 365 days into 3 "
              f"sub-windows and check if the strategy works in EACH, not just on average.\n")
    for scen, df in by_regime.items():
        md.append(f"### {scen}\n")
        md.append(f"| Regime | Asset | Direction Acc. | Sharpe | Return | B&H | Beat B&H | Trades |")
        md.append(f"|-------:|-------|---------------:|-------:|-------:|----:|---------:|-------:|")
        for _, r in df.iterrows():
            md.append(f"| {int(r['regime'])} | {r['symbol']} | "
                      f"{r['direction_accuracy']:.2%} | {r['tda_sharpe']:.2f} | "
                      f"{r['tda_return_pct']:.2f}% | {r['buy_hold_return_pct']:.2f}% | "
                      f"{r['outperformed_bh']:.0%} | {int(r['tda_n_trades'])} |")
        md.append("")

    md.append(f"## 3. Verdict\n")
    if len(summary) > 0:
        bm = summary[summary['scenario'] == 'binance_maker']
        ct = summary[summary['scenario'] == 'coinbase_taker']

        md.append(f"### Direction accuracy persists?\n")
        md.append(f"v6 paper: 76.75% (ADA, n=317) and 73.12% (SOL, n=345). "
                  f"v7 365-day: see Scenario summary above. If similar, signal "
                  f"is robust across the longer sample.\n")

        md.append(f"### Sharpe profitable on Binance.US?\n")
        if len(bm) > 0:
            sharpe = float(bm.iloc[0]['mean_sharpe'])
            ret = float(bm.iloc[0]['mean_return_pct'])
            if sharpe > 0:
                md.append(f"- **YES**: Mean Sharpe {sharpe:.2f}, mean return {ret:.2f}%/fold "
                          f"with Binance.US maker fees. The math worked.")
            elif sharpe > -1:
                md.append(f"- **MARGINAL**: Mean Sharpe {sharpe:.2f}, mean return {ret:.2f}%/fold. "
                          f"Above breakeven but not strongly profitable. Consider tighter "
                          f"threshold (p>0.75) or longer horizon.")
            else:
                md.append(f"- **NO**: Mean Sharpe {sharpe:.2f}, mean return {ret:.2f}%/fold. "
                          f"Even with Binance.US fees, the model loses money. The ADA+SOL+7d "
                          f"hypothesis from the v6 paper is NOT confirmed at 1-year scale.")

        md.append(f"\n### Cross-regime stability?\n")
        for scen, df in by_regime.items():
            if len(df) > 0:
                regime_means = df.groupby('regime')['direction_accuracy'].mean()
                stable = all(a > 0.55 for a in regime_means.values)
                md.append(f"- **{scen}**: per-regime accuracies = "
                          f"{dict(regime_means.round(3))}. "
                          f"{'Stable across regimes ✓' if stable else 'NOT stable across regimes ✗'}")

    md.append(f"\n## 4. Recommendation\n")
    md.append(f"See verdict above for the data. If Sharpe is positive on Binance.US AND "
              f"per-regime accuracies are all >55%, paper trade for 60 days before "
              f"committing capital. Otherwise the v6 paper finding does not generalize "
              f"to a profitable strategy.\n")

    md_text = "\n".join(md)
    with open('results/V7_PROFITABILITY.md', 'w') as f:
        f.write(md_text)
    print(f"  Saved: results/V7_PROFITABILITY.md")

    csv_rows = []
    for scen, df in by_regime.items():
        for _, r in df.iterrows():
            csv_rows.append({**r.to_dict(), 'scenario': scen})
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv('results/v7_per_regime.csv', index=False)
        print(f"  Saved: results/v7_per_regime.csv")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 168
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    run_v7(days=days, horizon_periods=horizon, prob_threshold=threshold)
