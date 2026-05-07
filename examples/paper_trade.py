#!/usr/bin/env python
"""
Paper-Trading Runner.

Runs the v9-selected model in real time WITHOUT executing actual trades.
Logs every signal to disk along with the contemporaneous price.

After 30-90 days of accumulated logs, run a review script to compare what
the model said vs. what the market actually did. This characterises the
live-vs-backtest gap before any real-money deployment.

Usage:
  # Single tick (intended to be invoked by cron every 1h)
  python examples/paper_trade.py tick

  # Continuous mode (runs forever, sleeps 1h between ticks)
  python examples/paper_trade.py continuous

  # Force retrain the cached model (otherwise rebuilt every 7 days)
  python examples/paper_trade.py retrain

Files written:
  data/paper_trades/{symbol}_signals.csv
    Columns: timestamp, symbol, signal, p_up, position_size, close_price, model_id
    Format:  UTC ISO 8601 timestamps, comma-separated, append-only.

  data/paper_trade_cache/production_model.pkl
    Pickled scikit-learn pipeline + feature column list + config dict.
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.binance_data import HighFreqFetcher
from src.advanced_features import build_advanced_features, TDA_FEATURE_SET
from src.persistent_homology import compute_features_for_windows
from src.multi_asset_pipeline import (
    fetch_multi_asset_pool,
    get_combined_feature_cols,
)
from examples.run_validation_v9 import add_targets


MODEL_CONFIG = {
    'model_type': 'logistic',
    'prob_threshold': 0.70,
    'horizon_hours': 72,
    'regime_filter': None,
    'symbols': ['ADA', 'SOL', 'DOT', 'LINK', 'AVAX', 'ETH'],
    'training_days': 365,
    'window_size': 20,
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'paper_trades')
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'data', 'paper_trade_cache')


def make_classifier(model_type, random_state=42):
    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state)
    elif model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=random_state, n_jobs=-1,
        )
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def train_production_model(force_retrain=False):
    """Train (or load) the production model. Cached for 7 days."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(MODEL_CACHE_DIR, 'production_model.pkl')

    age_days = -1
    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400

    if not force_retrain and age_days >= 0 and age_days < 7:
        print(f"[model] Loading cached model (age {age_days:.1f} days)")
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"[model] Training production model on last {MODEL_CONFIG['training_days']} days...")
    pooled = fetch_multi_asset_pool(
        MODEL_CONFIG['symbols'],
        days=MODEL_CONFIG['training_days'],
        interval='1h',
        window_size=MODEL_CONFIG['window_size'],
    )
    feature_cols = get_combined_feature_cols(pooled)
    pooled = add_targets(pooled, horizon=MODEL_CONFIG['horizon_hours'])

    X = pooled[feature_cols].values
    y = pooled['target'].values
    valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid].astype(int)

    clf = make_classifier(MODEL_CONFIG['model_type'])
    clf.fit(X, y)

    bundle = {
        'classifier': clf,
        'feature_cols': feature_cols,
        'config': MODEL_CONFIG.copy(),
        'trained_at': pd.Timestamp.utcnow().isoformat(),
    }
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[model] Saved cached model: {cache_path}")
    return bundle


def fetch_recent_features_for_asset(symbol, training_days=120):
    """Fetch enough hourly history to compute the current TDA window."""
    fetcher = HighFreqFetcher(symbol=symbol, interval='1h')
    raw_df = fetcher.fetch_history(days=training_days)
    df = build_advanced_features(raw_df)
    return df


def signal_for_symbol(symbol, classifier_bundle):
    """Generate one signal for one asset at the current moment."""
    df = fetch_recent_features_for_asset(symbol, training_days=60)
    if len(df) < 50:
        return None

    tda_cols = [c for c in TDA_FEATURE_SET if c in df.columns]
    X_tda = df[tda_cols].values

    mean = X_tda.mean(axis=0)
    std = X_tda.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X_tda - mean) / std

    window = MODEL_CONFIG['window_size']
    if len(X_norm) < window:
        return None
    last_window = X_norm[-window:]

    tda_features_df = compute_features_for_windows(
        [last_window], end_indices=[len(df) - 1], verbose=False,
    )
    if len(tda_features_df) == 0:
        return None

    feature_cols = classifier_bundle['feature_cols']
    last_row = df.iloc[-1:].copy()
    for col in tda_features_df.columns:
        if col in ('window_idx', 'end_idx'):
            continue
        last_row[col] = tda_features_df[col].iloc[0]
    for sym in MODEL_CONFIG['symbols']:
        last_row[f'asset_{sym}'] = 1 if sym == symbol else 0

    missing = [c for c in feature_cols if c not in last_row.columns]
    for c in missing:
        last_row[c] = 0.0

    X = last_row[feature_cols].values
    if np.any(np.isnan(X)):
        return None

    proba = classifier_bundle['classifier'].predict_proba(X)
    p_up = float(proba[0, 1] if proba.shape[1] == 2 else proba[0, 0])

    threshold = MODEL_CONFIG['prob_threshold']
    confidence = abs(p_up - 0.5) * 2
    if p_up > threshold:
        signal = 'BUY'
        size = 0.10 * confidence
    elif p_up < 1 - threshold:
        signal = 'SELL'
        size = -0.10 * confidence
    else:
        signal = 'HOLD'
        size = 0.0

    return {
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'symbol': symbol,
        'signal': signal,
        'p_up': round(p_up, 6),
        'position_size': round(size, 6),
        'close_price': round(float(df['close'].iloc[-1]), 4),
        'model_id': classifier_bundle.get('trained_at', 'unknown'),
    }


def append_signal(record):
    if record is None:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{record['symbol']}_signals.csv")
    new_file = not os.path.exists(path)
    with open(path, 'a') as f:
        if new_file:
            f.write('timestamp,symbol,signal,p_up,position_size,close_price,model_id\n')
        f.write(f"{record['timestamp']},{record['symbol']},{record['signal']},"
                f"{record['p_up']},{record['position_size']},{record['close_price']},"
                f"{record['model_id']}\n")


def run_one_tick(force_retrain=False):
    """Generate signals for all configured symbols once."""
    bundle = train_production_model(force_retrain=force_retrain)
    print(f"[tick] {pd.Timestamp.utcnow().isoformat()} - generating signals")
    for symbol in MODEL_CONFIG['symbols']:
        try:
            record = signal_for_symbol(symbol, bundle)
            if record is None:
                print(f"  [{symbol}] insufficient data, skipped")
                continue
            append_signal(record)
            print(f"  [{symbol}] {record['signal']:<5} p_up={record['p_up']:.3f}  "
                  f"price={record['close_price']:>10}")
        except Exception as e:
            print(f"  [{symbol}] ERROR: {e}")


def run_continuous(interval_seconds=3600):
    """Run forever, ticking every `interval_seconds`."""
    print(f"[continuous] tick interval = {interval_seconds}s")
    while True:
        try:
            run_one_tick()
        except Exception as e:
            print(f"[continuous] tick failed: {e}")
        print(f"[continuous] sleeping {interval_seconds}s...")
        time.sleep(interval_seconds)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'tick'
    if mode == 'tick':
        run_one_tick()
    elif mode == 'continuous':
        run_continuous()
    elif mode == 'retrain':
        run_one_tick(force_retrain=True)
    else:
        print(f"Unknown mode: {mode}")
        print(f"Usage: python {sys.argv[0]} [tick|continuous|retrain]")
        sys.exit(1)
