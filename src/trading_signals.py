"""
Trading Signal Generator: Convert TDA features into BUY/SELL/HOLD signals.

Based on Saengduean et al. (2018) and Gidea et al. (2020):
- C1-norm peaks before market regime changes
- High entropy = noisy, low confidence
- Combine with momentum confirmation
"""

import os
import numpy as np
import pandas as pd


class TradingSignalGenerator:
    """Generate trading signals from TDA features."""

    def __init__(self,
                 c1_threshold_std=1.5,
                 c1_drop_threshold=-1.0,
                 confidence_threshold=0.5,
                 lookback_window=20,
                 max_position_size=0.10):
        self.c1_threshold_std = c1_threshold_std
        self.c1_drop_threshold = c1_drop_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size

    def generate_signals(self, features_df, price_series=None):
        """
        Generate trading signals from TDA features.

        Args:
            features_df: DataFrame from persistent_homology pipeline
            price_series: Optional price series for momentum confirmation

        Returns:
            DataFrame with columns added: signal, position_size, confidence,
            c1_zscore, entropy_zscore
        """
        df = features_df.copy()

        c1_col = 'H1_c1_norm' if 'H1_c1_norm' in df.columns else 'H0_c1_norm'
        ent_col = 'H1_entropy' if 'H1_entropy' in df.columns else 'H0_entropy'

        if c1_col not in df.columns:
            raise ValueError(f"Required column not found. Available: {list(df.columns)}")

        df['c1_mean'] = df[c1_col].rolling(window=self.lookback_window, min_periods=5).mean()
        df['c1_std'] = df[c1_col].rolling(window=self.lookback_window, min_periods=5).std()
        df['c1_std'] = df['c1_std'].replace(0, np.nan)
        df['c1_zscore'] = (df[c1_col] - df['c1_mean']) / df['c1_std']
        df['c1_zscore'] = df['c1_zscore'].fillna(0)

        df['entropy_mean'] = df[ent_col].rolling(window=self.lookback_window, min_periods=5).mean()
        df['entropy_std'] = df[ent_col].rolling(window=self.lookback_window, min_periods=5).std()
        df['entropy_std'] = df['entropy_std'].replace(0, np.nan)
        df['entropy_zscore'] = (df[ent_col] - df['entropy_mean']) / df['entropy_std']
        df['entropy_zscore'] = df['entropy_zscore'].fillna(0)

        df['confidence'] = 1.0 / (1.0 + np.abs(df['entropy_zscore']))

        if price_series is not None and len(price_series) >= len(df):
            prices = pd.Series(price_series[-len(df):]).reset_index(drop=True)
            df['price_momentum'] = prices.pct_change(periods=5).fillna(0).values
        else:
            df['price_momentum'] = 0.0

        signals, sizes, confidences = [], [], []

        for _, row in df.iterrows():
            signal = 'HOLD'
            size = 0.0
            confidence = float(row['confidence'])

            if not np.isfinite(row['c1_zscore']):
                signals.append(signal)
                sizes.append(size)
                confidences.append(confidence)
                continue

            if (row['c1_zscore'] > self.c1_threshold_std
                    and confidence > self.confidence_threshold
                    and row['price_momentum'] >= 0):
                signal = 'BUY'
                size = min(self.max_position_size,
                           confidence * self.max_position_size * abs(row['c1_zscore']) / 2)

            elif (row['c1_zscore'] < self.c1_drop_threshold
                  or (row['c1_zscore'] > self.c1_threshold_std and row['price_momentum'] < 0)):
                signal = 'SELL'
                size = -min(self.max_position_size, confidence * self.max_position_size)

            signals.append(signal)
            sizes.append(size)
            confidences.append(confidence)

        df['signal'] = signals
        df['position_size'] = sizes
        df['confidence_final'] = confidences

        return df

    def signal_summary(self, signals_df):
        """Print signal statistics."""
        return {
            'total_windows': len(signals_df),
            'buy_signals': int((signals_df['signal'] == 'BUY').sum()),
            'sell_signals': int((signals_df['signal'] == 'SELL').sum()),
            'hold_signals': int((signals_df['signal'] == 'HOLD').sum()),
            'avg_confidence': float(signals_df['confidence_final'].mean()),
            'max_position_size': float(signals_df['position_size'].abs().max()),
        }


def run_signal_pipeline(features_path=None, price_path=None, symbol='BTC', save_dir='data',
                       **signal_kwargs):
    """End-to-end signal generation."""
    if features_path is None:
        features_path = f'{save_dir}/persistence/{symbol}_tda_features.csv'
    if price_path is None:
        price_path = f'{save_dir}/raw/{symbol}_ohlcv.csv'

    print(f"[Signals] Loading TDA features from {features_path}")
    features_df = pd.read_csv(features_path)
    print(f"  Loaded {len(features_df)} feature rows")

    price_series = None
    if os.path.exists(price_path):
        ohlcv = pd.read_csv(price_path)
        price_series = ohlcv['close'].values
        print(f"  Loaded {len(price_series)} price points for momentum confirmation")

    print(f"[Signals] Generating trading signals")
    generator = TradingSignalGenerator(**signal_kwargs)
    signals_df = generator.generate_signals(features_df, price_series=price_series)

    summary = generator.signal_summary(signals_df)
    print(f"  Signal summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    output_path = f'{save_dir}/persistence/{symbol}_signals.csv'
    signals_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return signals_df


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    signals_df = run_signal_pipeline(symbol=symbol)
    print(f"\n[Done] Signals generated: {signals_df.shape}")
