"""Test transaction-cost handling in the backtester."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester


def test_no_signals_no_change():
    """If all signals are HOLD, equity should equal initial capital."""
    prices = np.array([100.0, 101.0, 102.0, 101.5, 100.5])
    n = len(prices)
    signals = pd.DataFrame({
        'signal': ['HOLD'] * n,
        'position_size': [0.0] * n,
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals)
    assert res['metrics']['total_return_pct'] == pytest.approx(0.0)
    assert res['metrics']['total_trades'] == 0


def test_buy_then_sell_with_fees_loses_to_fees():
    """If price returns to start, fees should drag equity below initial."""
    prices = np.array([100.0, 100.0])
    n = len(prices)
    signals = pd.DataFrame({
        'signal': ['BUY', 'SELL'],
        'position_size': [0.5, -0.5],
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals)
    assert res['metrics']['final_equity'] < 10000


def test_buy_then_sell_profits_with_price_rise():
    """Price up 5% with 50% position → ~2.5% gain minus fees."""
    prices = np.array([100.0, 105.0])
    signals = pd.DataFrame({
        'signal': ['BUY', 'SELL'],
        'position_size': [0.5, -0.5],
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals)
    assert res['metrics']['total_return_pct'] > 0
    assert res['metrics']['total_return_pct'] < 5.0


def test_sharpe_zero_returns():
    """Constant-equity returns produce Sharpe = 0."""
    n = 50
    prices = np.array([100.0] * n)
    signals = pd.DataFrame({
        'signal': ['HOLD'] * n,
        'position_size': [0.0] * n,
    })
    bt = Backtester(initial_capital=10000)
    res = bt.run(prices, signals)
    assert res['metrics']['sharpe_ratio'] == 0.0


def test_drawdown_recorded():
    """Strategy that takes a loss should record a non-zero max drawdown."""
    prices = np.array([100.0, 90.0, 95.0])
    signals = pd.DataFrame({
        'signal': ['BUY', 'HOLD', 'SELL'],
        'position_size': [1.0, 0.0, -1.0],
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals)
    assert res['metrics']['max_drawdown_pct'] < 0


def test_lower_fees_higher_pnl():
    """Same trades, lower fees → higher final equity."""
    prices = np.array([100.0, 110.0, 105.0])
    signals = pd.DataFrame({
        'signal': ['BUY', 'SELL', 'HOLD'],
        'position_size': [0.5, -0.5, 0.0],
    })
    bt_high = Backtester(initial_capital=10000, trade_fee=0.004, slippage=0.0005)
    bt_low = Backtester(initial_capital=10000, trade_fee=0.00075, slippage=0.0005)
    r_high = bt_high.run(prices, signals)
    r_low = bt_low.run(prices, signals)
    assert r_low['metrics']['final_equity'] > r_high['metrics']['final_equity']


def test_short_flat_price_loses_to_costs():
    """Opening and closing a short at the same price should lose fees/slippage."""
    prices = np.array([100.0, 100.0])
    signals = pd.DataFrame({
        'signal': ['SELL', 'BUY'],
        'position_size': [-0.5, 0.5],
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals, allow_short=True)
    assert res['metrics']['final_equity'] < 10000
    assert res['trades'].iloc[0]['pnl'] < 0


def test_short_immediate_equity_reflects_entry_costs():
    """Short entry should not inflate equity above initial capital."""
    prices = np.array([100.0])
    signals = pd.DataFrame({
        'signal': ['SELL'],
        'position_size': [-0.5],
    })
    bt = Backtester(initial_capital=10000, trade_fee=0.001, slippage=0.0005)
    res = bt.run(prices, signals, allow_short=True)
    assert res['metrics']['final_equity'] < 10000
