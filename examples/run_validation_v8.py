#!/usr/bin/env python
"""
V8: Continuous Walk-Forward Profitability Test.

Fixes the v7 artifact: k-fold validation generated only 1-7 actual trades
per fold, making Sharpe estimates noise-dominated. v8 uses true continuous
walk-forward:

  1. Initial training window: first 60 days (1440 hours)
  2. Predict next 7 days
  3. Take any signals that fire, hold for 7d each
  4. Retrain weekly on growing window
  5. Continuously trade through full 365-day sample

This produces hundreds of realized trades, letting us measure actual
profitability with proper fee math.

Output:
  - results/V8_WALKFORWARD.md
  - results/v8_trades.csv (every trade with PnL)
  - results/v8_equity_curves.png
  - results/V8_WALKFORWARD.pdf

Usage:
  python examples/run_validation_v8.py 365 168 0.65
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


def make_classifier(random_state=42):
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=random_state, n_jobs=-1,
    )
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def add_targets(df, horizon=168, price_col='close'):
    df = df.copy().sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    df['future_price'] = df.groupby('symbol')[price_col].transform(lambda x: x.shift(-horizon))
    df['future_return'] = df['future_price'] / df[price_col] - 1.0
    df['target'] = np.where(df['future_price'].notna(), df['future_price'] > df[price_col], np.nan)
    return df.dropna(subset=['target', 'future_return']).reset_index(drop=True)


class ContinuousBacktester:
    """
    Walk-forward backtester:
      - Train on data[0:warmup_idx]
      - Predict & potentially trade at warmup_idx
      - Move forward 1 step, retrain every retrain_freq steps
      - Track equity, trades, PnL
    """

    def __init__(self, fee=0.00075, slippage=0.0005,
                 prob_threshold=0.65, max_position_pct=0.50,
                 horizon=168, retrain_freq=168, initial_capital=10000):
        self.fee = fee
        self.slippage = slippage
        self.prob_threshold = prob_threshold
        self.max_position_pct = max_position_pct
        self.horizon = horizon
        self.retrain_freq = retrain_freq
        self.initial_capital = initial_capital

    def run_one_asset(self, asset_df, feature_cols, warmup_steps=1440):
        """Run continuous walk-forward on one asset."""
        n = len(asset_df)
        if n < warmup_steps + self.horizon * 2:
            return None

        prices = asset_df['close'].values
        timestamps = asset_df['timestamp'].values
        targets = asset_df['target'].values
        X_all = asset_df[feature_cols].values

        cash = self.initial_capital
        position_units = 0.0
        position_entry_price = 0.0
        position_close_step = -1
        signal_at_open = None

        equity_curve = np.zeros(n)
        equity_curve[:warmup_steps] = self.initial_capital
        trades = []

        clf = None
        last_train_step = -self.retrain_freq

        for t in range(warmup_steps, n):
            if t - last_train_step >= self.retrain_freq:
                train_end = t - self.horizon
                if train_end <= 0:
                    equity_curve[t] = cash + position_units * prices[t]
                    continue
                X_train = X_all[:train_end]
                y_train = targets[:train_end]
                valid = ~np.any(np.isnan(X_train), axis=1) & ~np.isnan(y_train)
                X_train, y_train = X_train[valid], y_train[valid].astype(int)
                if len(np.unique(y_train)) >= 2 and len(X_train) >= 100:
                    clf = make_classifier()
                    clf.fit(X_train, y_train)
                    last_train_step = t

            current_price = prices[t]
            current_equity = cash + position_units * current_price

            if position_units != 0 and t >= position_close_step:
                exec_price = current_price * (1 - self.slippage if position_units > 0 else 1 + self.slippage)
                if position_units > 0:
                    proceeds = position_units * exec_price
                    fee_paid = proceeds * self.fee
                    cash += proceeds - fee_paid
                    entry_fee = trades[-1].get('entry_fee', 0.0) if trades else 0.0
                    pnl = (exec_price - position_entry_price) * position_units - fee_paid - entry_fee
                else:
                    cost = abs(position_units) * exec_price
                    fee_paid = cost * self.fee
                    cash -= cost + fee_paid
                    entry_fee = trades[-1].get('entry_fee', 0.0) if trades else 0.0
                    pnl = (position_entry_price - exec_price) * abs(position_units) - fee_paid - entry_fee

                if trades and trades[-1]['exit_step'] is None:
                    trades[-1].update({
                        'exit_step': t,
                        'exit_price': exec_price,
                        'exit_time': timestamps[t],
                        'pnl': pnl,
                        'return_pct': pnl / (abs(position_units) * position_entry_price),
                    })
                position_units = 0.0
                position_entry_price = 0.0
                position_close_step = -1
                signal_at_open = None

            if position_units == 0 and clf is not None:
                X_now = X_all[t].reshape(1, -1)
                if not np.any(np.isnan(X_now)):
                    p_up = clf.predict_proba(X_now)[0, 1]
                    confidence = abs(p_up - 0.5) * 2

                    if p_up > self.prob_threshold:
                        size_pct = self.max_position_pct * confidence
                        exec_price = current_price * (1 + self.slippage)
                        trade_value = current_equity * size_pct
                        fee_paid = trade_value * self.fee
                        position_units = (trade_value - fee_paid) / exec_price
                        position_entry_price = exec_price
                        position_close_step = t + self.horizon
                        signal_at_open = 'BUY'
                        cash -= trade_value
                        trades.append({
                            'entry_step': t,
                            'entry_price': exec_price,
                            'entry_time': timestamps[t],
                            'side': 'long',
                            'units': position_units,
                            'size_pct': size_pct,
                            'p_up_at_entry': p_up,
                            'entry_fee': fee_paid,
                            'exit_step': None,
                        })

                    elif p_up < 1 - self.prob_threshold:
                        size_pct = self.max_position_pct * confidence
                        exec_price = current_price * (1 - self.slippage)
                        trade_value = current_equity * size_pct
                        fee_paid = trade_value * self.fee
                        position_units = -trade_value / exec_price
                        position_entry_price = exec_price
                        position_close_step = t + self.horizon
                        signal_at_open = 'SELL'
                        cash += trade_value - fee_paid
                        trades.append({
                            'entry_step': t,
                            'entry_price': exec_price,
                            'entry_time': timestamps[t],
                            'side': 'short',
                            'units': position_units,
                            'size_pct': size_pct,
                            'p_up_at_entry': p_up,
                            'entry_fee': fee_paid,
                            'exit_step': None,
                        })

            equity_curve[t] = cash + position_units * current_price

        if position_units != 0:
            final_price = prices[-1]
            if position_units > 0:
                cash += position_units * final_price * (1 - self.slippage) * (1 - self.fee)
            else:
                exec_price = final_price * (1 + self.slippage)
                cost = abs(position_units) * exec_price
                cash -= cost + cost * self.fee
            equity_curve[-1] = cash

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_series = pd.Series(equity_curve, index=timestamps)
        returns = equity_series.pct_change().fillna(0)

        first_price = prices[warmup_steps]
        last_price = prices[-1]
        bh_units = self.initial_capital / first_price
        bh_equity = pd.Series(bh_units * prices, index=timestamps)
        bh_equity.iloc[:warmup_steps] = self.initial_capital

        return {
            'trades': trades_df,
            'equity': equity_series,
            'bh_equity': bh_equity,
            'returns': returns,
            'final_equity': float(equity_series.iloc[-1]),
            'bh_final_equity': float(bh_equity.iloc[-1]),
            'total_return_pct': float((equity_series.iloc[-1] / self.initial_capital - 1) * 100),
            'bh_return_pct': float((bh_equity.iloc[-1] / self.initial_capital - 1) * 100),
            'n_trades': len(trades_df),
            'sharpe': self._sharpe(returns),
            'max_drawdown_pct': float(self._max_drawdown(equity_series) * 100),
        }

    @staticmethod
    def _sharpe(returns, periods_per_year=8760):
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity):
        cummax = equity.expanding().max()
        dd = (equity - cummax) / cummax
        return float(dd.min())


def run_v8(symbols=None, days=365, horizon=168, prob_threshold=0.65,
           max_position_pct=0.50, warmup_days=60):
    if symbols is None:
        symbols = ['ADA', 'SOL']

    print(f"\n{'#' * 70}")
    print(f"#  V8: Continuous Walk-Forward Profitability Test")
    print(f"#  Assets: {symbols}")
    print(f"#  Horizon: {horizon}h ({horizon/24:.1f}d)")
    print(f"#  Prob threshold: {prob_threshold}, Max position: {max_position_pct*100:.0f}%")
    print(f"#  Warmup: {warmup_days} days, Total: {days} days")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    print(f"[1/3] Fetching data...")
    pooled_df = fetch_multi_asset_pool(symbols, days=days, interval='1h')
    pooled_df = add_targets(pooled_df, horizon=horizon)
    feature_cols = get_combined_feature_cols(pooled_df)
    print(f"  Pool size: {len(pooled_df)}, Features: {len(feature_cols)}")

    warmup_steps = warmup_days * 24

    fee_scenarios = {
        'binance_maker': (0.00075, 0.0005),
        'coinbase_taker': (0.004, 0.0005),
    }

    all_results = {}

    for scen_name, (fee, slip) in fee_scenarios.items():
        print(f"\n[2/3] Running scenario: {scen_name} (fee={fee*100:.3f}%, slip={slip*100:.2f}%)")
        scen_results = {}

        for symbol in symbols:
            asset_df = pooled_df[pooled_df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
            print(f"    {symbol}: {len(asset_df)} samples, walk-forward starting at step {warmup_steps}...")

            bt = ContinuousBacktester(
                fee=fee, slippage=slip,
                prob_threshold=prob_threshold,
                max_position_pct=max_position_pct,
                horizon=horizon,
                retrain_freq=horizon,
                initial_capital=10000,
            )

            result = bt.run_one_asset(asset_df, feature_cols, warmup_steps=warmup_steps)

            if result:
                print(f"      Trades: {result['n_trades']}, "
                      f"Final equity: ${result['final_equity']:,.0f}, "
                      f"Strategy return: {result['total_return_pct']:+.2f}%, "
                      f"Buy-hold: {result['bh_return_pct']:+.2f}%")
                print(f"      Sharpe: {result['sharpe']:.2f}, MaxDD: {result['max_drawdown_pct']:.2f}%")
                scen_results[symbol] = result

        all_results[scen_name] = scen_results

    print(f"\n[3/3] Writing report and plots...")
    write_v8_report(all_results, symbols, days, horizon, prob_threshold,
                    max_position_pct, warmup_days)
    plot_v8_equity(all_results, symbols)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"\n{'#' * 70}")
    print(f"#  V8 Complete — see results/V8_WALKFORWARD.md")
    print(f"{'#' * 70}\n")

    return all_results


def write_v8_report(all_results, symbols, days, horizon, prob_threshold,
                    max_position_pct, warmup_days):
    os.makedirs('results', exist_ok=True)

    md = []
    md.append(f"# V8: Continuous Walk-Forward Profitability Test\n")
    md.append(f"**Hypothesis:** Continuous trading (not k-fold) with realistic position sizing")
    md.append(f"will reveal whether the v6/v7 direction-accuracy edge converts to profit.\n")
    md.append(f"**Setup:** {days} days hourly data, walk-forward from day {warmup_days}, "
              f"7d holds, p>{prob_threshold} threshold, up to {max_position_pct*100:.0f}% position size.")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

    md.append(f"## Results\n")
    md.append(f"| Scenario | Asset | Trades | Strategy | Buy-Hold | Diff | Sharpe | Max DD |")
    md.append(f"|----------|-------|-------:|---------:|---------:|-----:|-------:|-------:|")
    for scen_name, scen_results in all_results.items():
        for symbol, r in scen_results.items():
            diff = r['total_return_pct'] - r['bh_return_pct']
            md.append(f"| {scen_name} | {symbol} | {r['n_trades']} | "
                      f"{r['total_return_pct']:+.2f}% | "
                      f"{r['bh_return_pct']:+.2f}% | "
                      f"{diff:+.2f}% | "
                      f"{r['sharpe']:.2f} | {r['max_drawdown_pct']:.2f}% |")
    md.append("")

    md.append(f"## Trade Counts (vs k-fold v7)\n")
    md.append(f"v7 had ~30 actual trades total across all folds. v8 has:")
    md.append(f"")
    for scen_name, scen_results in all_results.items():
        total = sum(r['n_trades'] for r in scen_results.values())
        md.append(f"- {scen_name}: {total} trades")
    md.append(f"\nThis is the trade count we need for proper Sharpe estimation.\n")

    md.append(f"## Verdict\n")

    bm_results = all_results.get('binance_maker', {})
    profitable = []
    losing = []
    for symbol, r in bm_results.items():
        if r['total_return_pct'] > 0:
            profitable.append((symbol, r))
        else:
            losing.append((symbol, r))

    if profitable:
        md.append(f"### Profitable on Binance.US:")
        for symbol, r in profitable:
            md.append(f"- **{symbol}**: +{r['total_return_pct']:.2f}% over {r['n_trades']} trades, "
                      f"Sharpe {r['sharpe']:.2f}")
    if losing:
        md.append(f"\n### Lost money on Binance.US:")
        for symbol, r in losing:
            md.append(f"- **{symbol}**: {r['total_return_pct']:.2f}% over {r['n_trades']} trades, "
                      f"Sharpe {r['sharpe']:.2f}")

    md.append(f"\n### Vs Buy-Hold:")
    bh_diff = []
    for symbol, r in bm_results.items():
        diff = r['total_return_pct'] - r['bh_return_pct']
        bh_diff.append((symbol, diff, r['bh_return_pct']))
        winner = "TDA strategy" if diff > 0 else "Buy-hold"
        md.append(f"- **{symbol}**: TDA {r['total_return_pct']:+.2f}% vs B&H "
                  f"{r['bh_return_pct']:+.2f}% — **{winner} wins by {abs(diff):.2f}%**")

    md_text = "\n".join(md)
    with open('results/V8_WALKFORWARD.md', 'w') as f:
        f.write(md_text)
    print(f"  Saved: results/V8_WALKFORWARD.md")

    all_trades = []
    for scen_name, scen_results in all_results.items():
        for symbol, r in scen_results.items():
            if not r['trades'].empty:
                tdf = r['trades'].copy()
                tdf['scenario'] = scen_name
                tdf['symbol'] = symbol
                all_trades.append(tdf)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv('results/v8_trades.csv', index=False)
        print(f"  Saved: results/v8_trades.csv")


def plot_v8_equity(all_results, symbols):
    os.makedirs('results/figures', exist_ok=True)

    n_scen = len(all_results)
    n_assets = len(symbols)
    fig, axes = plt.subplots(n_scen, n_assets, figsize=(7 * n_assets, 4.5 * n_scen),
                              squeeze=False)

    for i, (scen_name, scen_results) in enumerate(all_results.items()):
        for j, symbol in enumerate(symbols):
            ax = axes[i][j]
            if symbol not in scen_results:
                ax.set_visible(False)
                continue
            r = scen_results[symbol]
            ax.plot(r['equity'].index, r['equity'].values,
                    label=f"TDA Strategy ({r['total_return_pct']:+.1f}%)",
                    color='steelblue', linewidth=2)
            ax.plot(r['bh_equity'].index, r['bh_equity'].values,
                    label=f"Buy & Hold ({r['bh_return_pct']:+.1f}%)",
                    color='gray', linewidth=2, alpha=0.7)
            ax.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial')
            ax.set_title(f'{symbol} — {scen_name}\n({r["n_trades"]} trades, Sharpe {r["sharpe"]:.2f})')
            ax.set_ylabel('Equity ($)')
            ax.legend(loc='best')
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('results/figures/v8_equity_curves.png', dpi=120, bbox_inches='tight')
    plt.savefig('results/figures/v8_equity_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: results/figures/v8_equity_curves.{{png,pdf}}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 168
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
    pos_size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.50
    run_v8(days=days, horizon=horizon, prob_threshold=threshold,
           max_position_pct=pos_size)
