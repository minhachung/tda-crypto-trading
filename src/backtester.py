"""
Backtester: Evaluate trading strategy performance on historical data.

Computes Sharpe ratio, win rate, max drawdown, profit factor.
Supports long-only and long/short strategies.
"""

import os
import numpy as np
import pandas as pd


class Backtester:
    """Vectorized backtester for TDA-based trading signals."""

    def __init__(self, initial_capital=10000.0, trade_fee=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.trade_fee = trade_fee
        self.slippage = slippage

    def run(self, prices, signals_df, allow_short=False):
        """
        Run backtest aligning prices to signals.

        Args:
            prices: array-like of close prices
            signals_df: DataFrame with 'signal' and 'position_size' columns
            allow_short: If True, SELL on flat position opens short

        Returns:
            dict with trades, equity_curve, returns, metrics
        """
        prices = np.asarray(prices, dtype=float)
        n = min(len(prices), len(signals_df))
        prices = prices[-n:]
        signals_df = signals_df.iloc[-n:].reset_index(drop=True)

        cash = self.initial_capital
        position_units = 0.0
        position_entry_price = 0.0
        position_size_pct = 0.0

        trades = []
        equity_curve = np.zeros(n)

        for i in range(n):
            price = prices[i]
            signal = signals_df.iloc[i]['signal']
            target_size = float(signals_df.iloc[i]['position_size'])

            current_equity = cash + position_units * price

            if signal == 'BUY' and position_units == 0 and target_size > 0:
                exec_price = price * (1 + self.slippage)
                trade_value = current_equity * target_size
                fee = trade_value * self.trade_fee
                position_units = (trade_value - fee) / exec_price
                position_entry_price = exec_price
                position_size_pct = target_size
                cash -= trade_value
                trades.append({
                    'entry_idx': i, 'entry_price': exec_price, 'side': 'long',
                    'units': position_units, 'size_pct': target_size,
                    'exit_idx': None, 'exit_price': None, 'pnl': None, 'return_pct': None,
                })

            elif signal == 'SELL' and position_units > 0:
                exec_price = price * (1 - self.slippage)
                proceeds = position_units * exec_price
                fee = proceeds * self.trade_fee
                cash += proceeds - fee

                pnl = (exec_price - position_entry_price) * position_units - fee
                ret_pct = (exec_price - position_entry_price) / position_entry_price

                if trades:
                    trades[-1].update({
                        'exit_idx': i, 'exit_price': exec_price,
                        'pnl': pnl, 'return_pct': ret_pct,
                    })

                position_units = 0
                position_entry_price = 0
                position_size_pct = 0

            elif signal == 'SELL' and position_units == 0 and allow_short and target_size < 0:
                exec_price = price * (1 - self.slippage)
                trade_value = current_equity * abs(target_size)
                fee = trade_value * self.trade_fee
                position_units = -(trade_value - fee) / exec_price
                position_entry_price = exec_price
                position_size_pct = target_size
                cash += trade_value
                trades.append({
                    'entry_idx': i, 'entry_price': exec_price, 'side': 'short',
                    'units': position_units, 'size_pct': target_size,
                    'exit_idx': None, 'exit_price': None, 'pnl': None, 'return_pct': None,
                })

            elif signal == 'BUY' and position_units < 0:
                exec_price = price * (1 + self.slippage)
                cost = abs(position_units) * exec_price
                fee = cost * self.trade_fee
                cash -= cost + fee

                pnl = (position_entry_price - exec_price) * abs(position_units) - fee
                ret_pct = (position_entry_price - exec_price) / position_entry_price

                if trades:
                    trades[-1].update({
                        'exit_idx': i, 'exit_price': exec_price,
                        'pnl': pnl, 'return_pct': ret_pct,
                    })

                position_units = 0
                position_entry_price = 0
                position_size_pct = 0

            equity_curve[i] = cash + position_units * price

        if position_units != 0:
            final_price = prices[-1]
            if position_units > 0:
                proceeds = position_units * final_price * (1 - self.slippage)
                fee = proceeds * self.trade_fee
                cash += proceeds - fee
                pnl = (final_price - position_entry_price) * position_units - fee
                ret_pct = (final_price - position_entry_price) / position_entry_price
            else:
                cost = abs(position_units) * final_price * (1 + self.slippage)
                fee = cost * self.trade_fee
                cash -= cost + fee
                pnl = (position_entry_price - final_price) * abs(position_units) - fee
                ret_pct = (position_entry_price - final_price) / position_entry_price

            if trades:
                trades[-1].update({
                    'exit_idx': n - 1, 'exit_price': final_price,
                    'pnl': pnl, 'return_pct': ret_pct, 'closed_at_end': True,
                })

            equity_curve[-1] = cash

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().fillna(0)

        metrics = self._calculate_metrics(trades_df, equity_series, returns, prices)

        return {
            'trades': trades_df,
            'equity_curve': equity_series,
            'returns': returns,
            'metrics': metrics,
            'final_equity': float(equity_series.iloc[-1]),
        }

    def _calculate_metrics(self, trades_df, equity_curve, returns, prices):
        """Compute performance metrics."""
        m = {
            'initial_capital': self.initial_capital,
            'final_equity': float(equity_curve.iloc[-1]),
            'total_return_pct': float((equity_curve.iloc[-1] / self.initial_capital - 1) * 100),
            'buy_hold_return_pct': float((prices[-1] / prices[0] - 1) * 100),
        }

        m['total_trades'] = len(trades_df)
        if len(trades_df) > 0 and 'return_pct' in trades_df.columns:
            completed = trades_df.dropna(subset=['return_pct'])
            m['completed_trades'] = len(completed)

            if len(completed) > 0:
                wins = completed[completed['return_pct'] > 0]
                losses = completed[completed['return_pct'] < 0]

                m['winning_trades'] = len(wins)
                m['losing_trades'] = len(losses)
                m['win_rate'] = len(wins) / len(completed)
                m['avg_return_per_trade_pct'] = float(completed['return_pct'].mean() * 100)
                m['avg_win_pct'] = float(wins['return_pct'].mean() * 100) if len(wins) else 0.0
                m['avg_loss_pct'] = float(losses['return_pct'].mean() * 100) if len(losses) else 0.0

                gross_wins = wins['pnl'].sum() if 'pnl' in wins.columns else 0
                gross_losses = abs(losses['pnl'].sum()) if 'pnl' in losses.columns else 0
                m['profit_factor'] = float(gross_wins / gross_losses) if gross_losses > 0 else float('inf')

        m['max_drawdown_pct'] = float(self._max_drawdown(equity_curve) * 100)
        m['sharpe_ratio'] = float(self._sharpe_ratio(returns))
        m['sortino_ratio'] = float(self._sortino_ratio(returns))
        m['calmar_ratio'] = float(self._calmar_ratio(equity_curve, returns))

        return m

    @staticmethod
    def _max_drawdown(equity_curve):
        cummax = equity_curve.expanding().max()
        dd = (equity_curve - cummax) / cummax
        return dd.min()

    @staticmethod
    def _sharpe_ratio(returns, rf_annual=0.02, periods_per_year=365):
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        excess = returns - rf_annual / periods_per_year
        return excess.mean() / returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def _sortino_ratio(returns, rf_annual=0.02, periods_per_year=365):
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        excess = returns - rf_annual / periods_per_year
        return excess.mean() / downside.std() * np.sqrt(periods_per_year)

    def _calmar_ratio(self, equity_curve, returns):
        mdd = abs(self._max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        n = len(returns)
        if n == 0:
            return 0.0
        annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365 / n) - 1
        return annual_return / mdd


def format_metrics(metrics):
    """Pretty-print backtest metrics."""
    lines = ["", "=" * 60, "BACKTEST RESULTS", "=" * 60]
    lines.append(f"  Initial Capital:      ${metrics['initial_capital']:>12,.2f}")
    lines.append(f"  Final Equity:         ${metrics['final_equity']:>12,.2f}")
    lines.append(f"  Strategy Return:      {metrics['total_return_pct']:>12.2f}%")
    lines.append(f"  Buy & Hold Return:    {metrics['buy_hold_return_pct']:>12.2f}%")
    lines.append("-" * 60)
    lines.append(f"  Total Trades:         {metrics.get('total_trades', 0):>12d}")
    lines.append(f"  Completed Trades:     {metrics.get('completed_trades', 0):>12d}")
    if metrics.get('completed_trades', 0) > 0:
        lines.append(f"  Win Rate:             {metrics.get('win_rate', 0):>12.2%}")
        lines.append(f"  Avg Return/Trade:     {metrics.get('avg_return_per_trade_pct', 0):>12.2f}%")
        lines.append(f"  Avg Win:              {metrics.get('avg_win_pct', 0):>12.2f}%")
        lines.append(f"  Avg Loss:             {metrics.get('avg_loss_pct', 0):>12.2f}%")
        lines.append(f"  Profit Factor:        {metrics.get('profit_factor', 0):>12.2f}")
    lines.append("-" * 60)
    lines.append(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>12.2f}")
    lines.append(f"  Sortino Ratio:        {metrics['sortino_ratio']:>12.2f}")
    lines.append(f"  Calmar Ratio:         {metrics['calmar_ratio']:>12.2f}")
    lines.append(f"  Max Drawdown:         {metrics['max_drawdown_pct']:>12.2f}%")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_backtest(symbol='BTC', save_dir='data', allow_short=False, **bt_kwargs):
    """End-to-end backtest run."""
    signals_path = f'{save_dir}/persistence/{symbol}_signals.csv'
    prices_path = f'{save_dir}/raw/{symbol}_ohlcv.csv'

    print(f"[Backtest] Loading data for {symbol}")
    signals_df = pd.read_csv(signals_path)
    ohlcv = pd.read_csv(prices_path)
    prices = ohlcv['close'].values

    if 'end_idx' in signals_df.columns:
        idx = signals_df['end_idx'].astype(int).values
        idx = idx[idx < len(prices)]
        prices_aligned = prices[idx]
        signals_df = signals_df.iloc[:len(prices_aligned)].reset_index(drop=True)
    else:
        prices_aligned = prices[-len(signals_df):]

    print(f"  Aligned: {len(signals_df)} signals, {len(prices_aligned)} prices")

    print(f"[Backtest] Running backtest")
    backtester = Backtester(**bt_kwargs)
    results = backtester.run(prices_aligned, signals_df, allow_short=allow_short)

    print(format_metrics(results['metrics']))

    os.makedirs(f'{save_dir}/persistence', exist_ok=True)
    results['trades'].to_csv(f'{save_dir}/persistence/{symbol}_trades.csv', index=False)
    results['equity_curve'].to_csv(f'{save_dir}/persistence/{symbol}_equity.csv', index=False)
    print(f"  Saved trades and equity curve")

    return results


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    results = run_backtest(symbol=symbol)
