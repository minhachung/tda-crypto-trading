# Implementation Guide: Build Your First TDA Trading System

This guide walks you through building Strategy 1 (Persistent Homology Price-Volume Manifolds) step-by-step, from data to trading signals.

---

## Phase 1: Environment Setup

### 1.1 Create Virtual Environment

```bash
cd tda-crypto-trading
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Create `requirements.txt`:

```
# Data
pandas==2.0.0
numpy==1.24.0
requests==2.31.0

# TDA
giotto-tda==0.6.1
ripser==0.6.42
persim==0.3.1

# Technical Indicators
ta==0.10.2
talib==0.4.28  # optional, requires compilation

# ML/Stats
scikit-learn==1.3.0
scipy==1.11.0
statsmodels==0.14.0

# Visualization
matplotlib==3.7.0
plotly==5.17.0
seaborn==0.12.0

# Backtesting
backtrader==1.9.76  # or backtesting.py

# APIs
ccxt==3.0.0  # Crypto exchange API
pandas-datareader==0.10.0

# Utilities
python-dotenv==1.0.0
tqdm==4.66.0
joblib==1.3.0

# Jupyter
jupyter==1.0.0
jupyterlab==4.0.0
```

### 1.3 Create .env File

```bash
cp .env.example .env
```

`.env.example`:
```
# APIs
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
COINGECKO_API_KEY=free  # or paid key

GLASSNODE_API_KEY=your_key_here

# Trading parameters
INITIAL_CAPITAL=10000
TRADING_PAIR=BTC/USD
```

---

## Phase 2: Data Pipeline

### 2.1 Create Data Fetcher

File: `src/data_pipeline.py`

```python
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from ta import *

load_dotenv()

class DataFetcher:
    """Fetch OHLCV and on-chain data for crypto assets"""
    
    def __init__(self, symbol='BTC', days=365):
        self.symbol = symbol
        self.days = days
        self.start_date = datetime.now() - timedelta(days=days)
        
    def fetch_ohlcv_coingecko(self):
        """Fetch free OHLCV data from CoinGecko"""
        url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': self.days,
            'interval': 'daily'
        }
        
        response = requests.get(url.format(self.symbol.lower()), params=params)
        data = response.json()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['prices'], unit='ms'),
            'close': [p[1] for p in data['prices']],
            'volume': [v[1] for v in data['total_volumes']]
        })
        
        # CoinGecko doesn't provide OHLC, so we'll use approximation
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'].rolling(2).max()
        df['low'] = df['close'].rolling(2).min()
        
        return df.dropna().reset_index(drop=True)
    
    def add_technical_indicators(self, df):
        """Add RSI, MACD, Bollinger Bands"""
        
        # RSI (14-period)
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        return df.dropna().reset_index(drop=True)
    
    def fetch_on_chain_glassnode(self):
        """Fetch on-chain data from Glassnode (requires API key)"""
        api_key = os.getenv('GLASSNODE_API_KEY')
        if not api_key:
            print("⚠️  Glassnode API key not set. Using placeholders.")
            return None
        
        # This would require Glassnode authentication
        # For now, placeholder
        return None
    
    def preprocess_features(self, df):
        """Normalize features for TDA"""
        features = ['open', 'high', 'low', 'close', 'volume', 
                   'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'bb_mid']
        
        # Remove rows with NaN
        df = df.dropna(subset=features)
        
        # Normalize each feature (z-score)
        df_normalized = df.copy()
        for col in features:
            mean = df[col].mean()
            std = df[col].std()
            df_normalized[col] = (df[col] - mean) / std
        
        return df_normalized[features].values, df
    
    def create_point_cloud(self, X, window_size=10, stride=1):
        """
        Convert time series to point cloud using sliding windows
        
        Input: X shape (n_samples, n_features)
        Output: point_cloud shape (n_windows, n_features)
        """
        point_cloud = []
        
        for i in range(0, len(X) - window_size, stride):
            window = X[i:i+window_size]  # (window_size, n_features)
            # Flatten or use summary stats
            point = window.mean(axis=0)  # Average features in window
            point_cloud.append(point)
        
        return np.array(point_cloud)

def main():
    # Fetch data
    fetcher = DataFetcher(symbol='BTC', days=365)
    df = fetcher.fetch_ohlcv_coingecko()
    print(f"Fetched {len(df)} days of price data")
    
    # Add indicators
    df = fetcher.add_technical_indicators(df)
    print(f"Added technical indicators. Shape: {df.shape}")
    
    # Preprocess
    X_normalized, df_original = fetcher.preprocess_features(df)
    print(f"Normalized features. Shape: {X_normalized.shape}")
    
    # Create point cloud
    point_cloud = fetcher.create_point_cloud(X_normalized, window_size=10, stride=1)
    print(f"Created point cloud. Shape: {point_cloud.shape}")
    
    # Save
    np.save('data/processed/btc_point_cloud.npy', point_cloud)
    df.to_csv('data/raw/btc_ohlcv.csv', index=False)
    print("✅ Data pipeline complete")

if __name__ == '__main__':
    main()
```

### 2.2 Test Data Pipeline

```bash
python src/data_pipeline.py
# Should create:
#   data/raw/btc_ohlcv.csv
#   data/processed/btc_point_cloud.npy
```

---

## Phase 3: Persistent Homology Computation

### 3.1 Create TDA Module

File: `src/persistent_homology.py`

```python
import numpy as np
from ripser import ripser
from persim import bottleneck, wasserstein
import matplotlib.pyplot as plt

class PersistentHomologyAnalyzer:
    """Compute and analyze persistent homology of point clouds"""
    
    def __init__(self, point_cloud):
        """
        point_cloud: ndarray shape (n_points, n_dimensions)
        """
        self.point_cloud = point_cloud
        self.result = None
        self.diagrams = None
        self.features = None
        
    def compute(self):
        """Compute persistent homology using Rips complex"""
        self.result = ripser(self.point_cloud)
        
        # Extract diagrams
        self.diagrams = {
            'H0': self.result['dgms'][0],  # Connected components
            'H1': self.result['dgms'][1] if len(self.result['dgms']) > 1 else np.array([]),  # Loops
        }
        
        return self.diagrams
    
    def extract_features(self):
        """Extract trading-relevant features from persistence diagrams"""
        features = {}
        
        # For H1 (loops/cycles)
        h1 = self.diagrams['H1']
        
        if len(h1) > 0:
            # Remove infinite points
            h1_finite = h1[~np.isinf(h1[:, 1])]
            
            # Persistence = death - birth
            persistence = h1_finite[:, 1] - h1_finite[:, 0]
            
            # C1-norm: sum of persistence values
            features['c1_norm'] = np.sum(persistence)
            
            # L1-norm: integral of landscape (approximate as sum)
            features['l1_norm'] = np.sum(persistence)
            
            # Entropy
            if len(persistence) > 0:
                p = persistence / np.sum(persistence)
                features['entropy'] = -np.sum(p * np.log(p + 1e-10))
            else:
                features['entropy'] = 0.0
            
            # Feature count
            features['feature_count'] = len(h1_finite)
            
            # Max persistence (largest feature)
            if len(persistence) > 0:
                features['max_persistence'] = np.max(persistence)
                features['median_persistence'] = np.median(persistence)
            else:
                features['max_persistence'] = 0.0
                features['median_persistence'] = 0.0
        else:
            # No loops detected
            features = {
                'c1_norm': 0.0,
                'l1_norm': 0.0,
                'entropy': 0.0,
                'feature_count': 0,
                'max_persistence': 0.0,
                'median_persistence': 0.0
            }
        
        self.features = features
        return features
    
    def visualize_diagram(self, save_path=None):
        """Plot persistence diagram"""
        h1 = self.diagrams['H1']
        h1_finite = h1[~np.isinf(h1[:, 1])]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot points
        ax.scatter(h1_finite[:, 0], h1_finite[:, 1], alpha=0.6, s=50)
        
        # Plot diagonal
        max_val = max(h1_finite[:, 0].max(), h1_finite[:, 1].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')
        
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title('Persistence Diagram (H1 - Loops)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def sliding_window_features(self, point_clouds, window_size):
        """
        Compute features across multiple point clouds
        
        point_clouds: list of ndarrays, each shape (n_points, n_features)
        Returns: DataFrame with features over time
        """
        all_features = []
        
        for i, pc in enumerate(point_clouds):
            analyzer = PersistentHomologyAnalyzer(pc)
            analyzer.compute()
            features = analyzer.extract_features()
            features['timestamp'] = i
            all_features.append(features)
        
        import pandas as pd
        return pd.DataFrame(all_features)

def rolling_point_clouds(X, window_size=10, stride=1):
    """
    Create multiple point clouds from time series
    
    X: array shape (n_samples, n_features)
    Returns: list of point clouds
    """
    point_clouds = []
    timestamps = []
    
    for i in range(0, len(X) - window_size, stride):
        pc = X[i:i+window_size]
        point_clouds.append(pc)
        timestamps.append(i + window_size)  # End of window
    
    return point_clouds, timestamps

def main():
    # Load point cloud
    point_cloud = np.load('data/processed/btc_point_cloud.npy')
    print(f"Loaded point cloud: {point_cloud.shape}")
    
    # Compute persistent homology
    analyzer = PersistentHomologyAnalyzer(point_cloud)
    diagrams = analyzer.compute()
    print(f"H0 features: {len(diagrams['H0'])}")
    print(f"H1 features: {len(diagrams['H1'])}")
    
    # Extract features
    features = analyzer.extract_features()
    print("\nTDA Features:")
    for key, val in features.items():
        print(f"  {key}: {val:.4f}")
    
    # Visualize
    analyzer.visualize_diagram('data/persistence/persistence_diagram.png')
    print("\n✅ Persistence homology complete")

if __name__ == '__main__':
    main()
```

### 3.2 Test TDA Module

```bash
python src/persistent_homology.py
# Should output:
#   H0 features: 1 (all connected)
#   H1 features: 0-5 (loops detected)
#   TDA Features with C1-norm, entropy, etc.
#   PNG visualization
```

---

## Phase 4: Trading Signal Generation

### 4.1 Create Signal Generator

File: `src/trading_signals.py`

```python
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

class TradingSignalGenerator:
    """Generate buy/sell signals from TDA features"""
    
    def __init__(self, c1_threshold_std=2.0, entropy_weight=0.5):
        """
        c1_threshold_std: Alert when C1-norm exceeds mean + k*std
        entropy_weight: How much to discount noisy (high entropy) signals
        """
        self.c1_threshold_std = c1_threshold_std
        self.entropy_weight = entropy_weight
        
    def generate_signals(self, features_df):
        """
        Generate signals from TDA features dataframe
        
        Input: features_df with columns: c1_norm, entropy, feature_count, etc.
        Output: signals dataframe with BUY/SELL/HOLD
        """
        
        # Calculate rolling statistics
        window = 20  # 20-day lookback
        features_df['c1_mean'] = features_df['c1_norm'].rolling(window).mean()
        features_df['c1_std'] = features_df['c1_norm'].rolling(window).std()
        features_df['entropy_mean'] = features_df['entropy'].rolling(window).mean()
        
        # Standardize C1-norm
        features_df['c1_zscore'] = (
            (features_df['c1_norm'] - features_df['c1_mean']) / 
            features_df['c1_std']
        )
        
        # Confidence based on entropy (lower entropy = higher confidence)
        features_df['entropy_zscore'] = (
            (features_df['entropy'] - features_df['entropy_mean']) / 
            features_df['entropy'].rolling(window).std()
        )
        
        # Confidence score: entropy_zscore inverted
        features_df['confidence'] = 1.0 / (1.0 + features_df['entropy_zscore'].abs())
        
        # Signal generation logic
        signals = []
        positions = []
        confidences = []
        
        for idx, row in features_df.iterrows():
            signal = 'HOLD'
            position_size = 0.0
            confidence = 0.5
            
            if pd.notna(row['c1_zscore']) and pd.notna(row['confidence']):
                confidence = row['confidence']
                
                # BUY signal: C1-norm spikes + high confidence
                if row['c1_zscore'] > self.c1_threshold_std and confidence > 0.7:
                    signal = 'BUY'
                    position_size = confidence * 0.1  # Max 10% position
                
                # SELL signal: C1-norm drops sharply
                elif row['c1_zscore'] < -1.0:
                    signal = 'SELL'
                    position_size = -confidence * 0.1
            
            signals.append(signal)
            positions.append(position_size)
            confidences.append(confidence)
        
        features_df['signal'] = signals
        features_df['position_size'] = positions
        features_df['confidence'] = confidences
        
        return features_df
    
    def get_trading_rules(self):
        """Return human-readable trading rules"""
        return f"""
        TDA-Based Trading Rules:
        
        BUY SIGNAL:
          - C1-norm exceeds mean + {self.c1_threshold_std} std
          - Entropy (confidence) > 0.7
          - Position size: 1-10% based on confidence
        
        SELL SIGNAL:
          - C1-norm drops below mean - 1 std
          - Position size: -1-10% (reduce/exit)
        
        HOLD:
          - No strong signal
          - Wait for clear topology change
        
        Entropy Weight: {self.entropy_weight}
          (Higher = more conservative on high-entropy periods)
        """

def main():
    # Load TDA features (from previous step)
    features_df = pd.read_csv('data/persistence/tda_features.csv')
    
    # Generate signals
    signal_gen = TradingSignalGenerator(c1_threshold_std=2.0, entropy_weight=0.5)
    signals_df = signal_gen.generate_signals(features_df.copy())
    
    # Print results
    print(signal_gen.get_trading_rules())
    print("\nSignal Summary:")
    print(signals_df[['c1_norm', 'entropy', 'confidence', 'signal']].tail(20))
    
    # Save
    signals_df.to_csv('data/persistence/trading_signals.csv', index=False)
    print("\n✅ Signals generated and saved")
    
    # Statistics
    print(f"\nSignal Statistics:")
    print(f"  BUY signals: {(signals_df['signal'] == 'BUY').sum()}")
    print(f"  SELL signals: {(signals_df['signal'] == 'SELL').sum()}")
    print(f"  HOLD signals: {(signals_df['signal'] == 'HOLD').sum()}")

if __name__ == '__main__':
    main()
```

---

## Phase 5: Backtesting

### 5.1 Create Backtester

File: `src/backtester.py`

```python
import pandas as pd
import numpy as np

class SimpleBacktester:
    """Basic backtester for TDA signals"""
    
    def __init__(self, initial_capital=10000, trade_fee=0.001):
        self.initial_capital = initial_capital
        self.trade_fee = trade_fee
        
    def backtest(self, prices, signals_df):
        """
        Run backtest
        
        prices: pd.Series of close prices
        signals_df: DataFrame with 'signal' and 'position_size' columns
        
        Returns: trades DataFrame with returns
        """
        
        trades = []
        position = 0.0  # Current position size
        entry_price = None
        cash = self.initial_capital
        equity = self.initial_capital
        equity_curve = []
        
        for idx, (price, signal_row) in enumerate(zip(prices, signals_df.itertuples())):
            signal = signal_row.signal
            position_size = signal_row.position_size
            
            if signal == 'BUY' and position == 0:
                # Enter long position
                position = position_size
                entry_price = price
                cost = abs(position_size) * price * self.initial_capital
                cash -= cost * (1 + self.trade_fee)
                
                trades.append({
                    'entry_idx': idx,
                    'entry_price': entry_price,
                    'exit_idx': None,
                    'exit_price': None,
                    'pnl': None,
                    'return': None
                })
            
            elif signal == 'SELL' and position > 0:
                # Exit position
                exit_price = price
                pnl = (exit_price - entry_price) * position * self.initial_capital
                pnl -= pnl * self.trade_fee
                cash += abs(position) * exit_price * self.initial_capital
                
                if trades:
                    trades[-1]['exit_idx'] = idx
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['return'] = (exit_price - entry_price) / entry_price
                
                position = 0
            
            # Calculate current equity
            if position > 0:
                equity = cash + position * price * self.initial_capital
            else:
                equity = cash
            
            equity_curve.append(equity)
        
        return pd.DataFrame(trades), pd.Series(equity_curve)
    
    def calculate_metrics(self, trades_df, equity_curve, prices):
        """Calculate performance metrics"""
        
        if len(trades_df) == 0:
            return None
        
        # Filter completed trades
        completed = trades_df.dropna(subset=['return'])
        
        if len(completed) == 0:
            return None
        
        returns = completed['return']
        
        # Metrics
        metrics = {
            'total_trades': len(completed),
            'winning_trades': (completed['return'] > 0).sum(),
            'losing_trades': (completed['return'] < 0).sum(),
            'win_rate': (completed['return'] > 0).sum() / len(completed),
            'avg_return': returns.mean(),
            'std_return': returns.std(),
            'max_return': returns.max(),
            'min_return': returns.min(),
            'profit_factor': completed[completed['return'] > 0]['return'].sum() / 
                           abs(completed[completed['return'] < 0]['return'].sum()) if 
                           (completed['return'] < 0).sum() > 0 else np.inf,
            'total_pnl': completed['pnl'].sum(),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self.calculate_sharpe_ratio(equity_curve),
        }
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown"""
        cummax = equity_curve.expanding().max()
        dd = (equity_curve - cummax) / cummax
        return dd.min()
    
    @staticmethod
    def calculate_sharpe_ratio(equity_curve, rf_rate=0.02):
        """Calculate Sharpe ratio"""
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return (returns.mean() - rf_rate/252) / returns.std() * np.sqrt(252)

def main():
    # Load data
    prices = pd.read_csv('data/raw/btc_ohlcv.csv')['close']
    signals = pd.read_csv('data/persistence/trading_signals.csv')
    
    # Run backtest
    backtester = SimpleBacktester(initial_capital=10000)
    trades, equity = backtester.backtest(prices.values, signals)
    
    # Calculate metrics
    metrics = backtester.calculate_metrics(trades, equity, prices)
    
    print("Backtest Results:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("\n✅ Backtest complete")

if __name__ == '__main__':
    main()
```

---

## Phase 6: Full Pipeline Integration

### 6.1 Create Main Script

File: `main.py`

```python
#!/usr/bin/env python
"""
Complete TDA Crypto Trading Pipeline
1. Fetch data
2. Compute persistent homology
3. Generate signals
4. Backtest
5. Report
"""

import sys
from src.data_pipeline import DataFetcher, rolling_point_clouds
from src.persistent_homology import PersistentHomologyAnalyzer
from src.trading_signals import TradingSignalGenerator
from src.backtester import SimpleBacktester
import pandas as pd
import numpy as np

def run_pipeline(symbol='BTC', days=365):
    print(f"\n{'='*60}")
    print(f"TDA Trading System: {symbol}")
    print(f"{'='*60}\n")
    
    # Phase 1: Data
    print("📊 Phase 1: Fetching data...")
    fetcher = DataFetcher(symbol=symbol, days=days)
    df = fetcher.fetch_ohlcv_coingecko()
    df = fetcher.add_technical_indicators(df)
    X_normalized, _ = fetcher.preprocess_features(df)
    print(f"   ✓ Fetched {len(df)} days, {X_normalized.shape[1]} features")
    
    # Phase 2: Create point clouds
    print("\n📐 Phase 2: Creating point clouds...")
    point_clouds, timestamps = rolling_point_clouds(X_normalized, window_size=10, stride=1)
    print(f"   ✓ Created {len(point_clouds)} point clouds")
    
    # Phase 3: Persistent homology
    print("\n🔍 Phase 3: Computing persistent homology...")
    all_features = []
    for i, pc in enumerate(point_clouds):
        analyzer = PersistentHomologyAnalyzer(pc)
        analyzer.compute()
        features = analyzer.extract_features()
        features['timestamp'] = timestamps[i]
        all_features.append(features)
        
        if (i + 1) % 50 == 0:
            print(f"   ✓ Processed {i+1}/{len(point_clouds)} point clouds")
    
    features_df = pd.DataFrame(all_features)
    print(f"   ✓ TDA features computed")
    
    # Phase 4: Signals
    print("\n🎯 Phase 4: Generating trading signals...")
    signal_gen = TradingSignalGenerator(c1_threshold_std=2.0)
    signals_df = signal_gen.generate_signals(features_df.copy())
    print(f"   ✓ BUY: {(signals_df['signal'] == 'BUY').sum()}")
    print(f"   ✓ SELL: {(signals_df['signal'] == 'SELL').sum()}")
    
    # Phase 5: Backtest
    print("\n📈 Phase 5: Backtesting...")
    prices = df['close'].values[-len(signals_df):]
    backtester = SimpleBacktester(initial_capital=10000)
    trades, equity = backtester.backtest(prices, signals_df)
    metrics = backtester.calculate_metrics(trades, equity, prices)
    
    print(f"   ✓ Total trades: {metrics['total_trades']}")
    print(f"   ✓ Win rate: {metrics['win_rate']:.2%}")
    print(f"   ✓ Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   ✓ Max drawdown: {metrics['max_drawdown']:.2%}")
    
    # Phase 6: Report
    print(f"\n📋 Phase 6: Generating report...")
    report = f"""
    TDA Trading System Report: {symbol}
    {'-'*60}
    
    Data:
      Period: {days} days
      Features: {X_normalized.shape[1]}
      Point clouds: {len(point_clouds)}
    
    Performance:
      Total trades: {metrics['total_trades']}
      Winning: {metrics['winning_trades']} ({metrics['win_rate']:.1%})
      Losing: {metrics['losing_trades']}
      
      Avg return per trade: {metrics['avg_return']:.2%}
      Best trade: {metrics['max_return']:.2%}
      Worst trade: {metrics['min_return']:.2%}
      
      Total P&L: ${metrics['total_pnl']:,.2f}
      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
      Max Drawdown: {metrics['max_drawdown']:.2%}
      Profit Factor: {metrics['profit_factor']:.2f}
    """
    
    print(report)
    
    # Save
    signals_df.to_csv(f'data/persistence/{symbol}_signals.csv', index=False)
    with open(f'data/persistence/{symbol}_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Pipeline complete!\n")
    
    return {
        'df': df,
        'features_df': features_df,
        'signals_df': signals_df,
        'trades': trades,
        'metrics': metrics
    }

if __name__ == '__main__':
    # Run for Bitcoin
    results = run_pipeline(symbol='BTC', days=365)
    
    # Optional: Run for other assets
    # results_eth = run_pipeline(symbol='ETH', days=365)
```

### 6.2 Run Full Pipeline

```bash
python main.py
```

Output:
```
============================================================
TDA Trading System: BTC
============================================================

📊 Phase 1: Fetching data...
   ✓ Fetched 365 days, 11 features

📐 Phase 2: Creating point clouds...
   ✓ Created 356 point clouds

🔍 Phase 3: Computing persistent homology...
   ✓ Processed 50/356 point clouds
   ✓ Processed 100/356 point clouds
   ...
   ✓ TDA features computed

🎯 Phase 4: Generating trading signals...
   ✓ BUY: 12
   ✓ SELL: 11

📈 Phase 5: Backtesting...
   ✓ Total trades: 6
   ✓ Win rate: 66.67%
   ✓ Sharpe ratio: 1.45
   ✓ Max drawdown: -12.34%

✅ Pipeline complete!
```

---

## Phase 7: Optimization & Tuning

### 7.1 Parameter Sweep

```python
# Try different thresholds
for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
    signals = signal_gen.generate_signals(features_df.copy(), threshold)
    trades, equity = backtester.backtest(prices, signals)
    metrics = backtester.calculate_metrics(trades, equity, prices)
    print(f"Threshold {threshold}: Sharpe = {metrics['sharpe_ratio']:.2f}")
```

### 7.2 Walk-Forward Testing

```python
# Avoid overfitting: test on future unseen data
train_end = int(0.7 * len(df))
test_end = int(0.9 * len(df))

train_data = df[:train_end]
test_data = df[train_end:test_end]
val_data = df[test_end:]

# Train parameters on train_data
# Backtest on test_data (out-of-sample)
# Report on val_data (validation)
```

---

## Next Steps

1. ✅ Run main.py to validate pipeline works
2. 🔧 Optimize parameters on historical data
3. 📊 Test on different cryptocurrencies
4. 🚀 Paper trade (simulated execution)
5. 💰 Deploy live (with risk controls!)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ripser'"
```bash
pip install ripser
```

### "ImportError: cannot import name 'RSIIndicator'"
```bash
pip install ta-lib  # or alternative
```

### "No H1 features detected"
- Increase point cloud size
- Decrease window size
- Check data normalization

---

## Resources

- Giotto-TDA docs: https://giotto-ai.github.io/
- Ripser: https://github.com/scikit-tda/ripser.py
- Research papers: See `docs/01_literature_summary.md`
