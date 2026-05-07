#!/usr/bin/env python
"""
V11: Leave-One-Asset-Out Validation + Stronger Baselines.

Addresses two open methodological items left over from v9/v10:

  3. LEAVE-ONE-ASSET-OUT (LOAO)
     Tests whether the multi-asset pooled story is real cross-asset
     transferability or asset-specific overfitting. For each asset A,
     train on the union of the other six assets and evaluate on A.

  4. STRONGER BASELINES
     Establishes a proper baseline ladder so the TDA story is judged
     against more than logistic/rf/gbm. Adds:
       - XGBoost, LightGBM, CatBoost (gradient boosters on Base+TDA)
       - Logistic on lagged returns only (returns-only baseline)
       - Momentum rule (sign of past N-period return)
       - Volatility-breakout rule (z-score of realized vol)
       - Buy-and-hold (always long, no signals)
       - Cash (always flat)

This script does NOT redo the v10 1000-permutation test; v10 already
covers items 1 (multi-year sample) and 2 (1000 permutations).

Output:
  - results/V11_LOAO_BASELINES.md
  - results/v11_loao.csv             (per-asset LOAO results)
  - results/v11_baselines.csv        (one row per baseline)
  - results/v11_loao_by_model.csv    (LOAO x model grid)

Usage:
  python examples/run_validation_v11.py                # 1095 days default
  python examples/run_validation_v11.py 365            # quicker run
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.multi_asset_pipeline import (
    fetch_multi_asset_pool,
    get_combined_feature_cols,
)
from src.advanced_features import ML_FEATURE_SET
from src.backtester import Backtester
from examples.run_validation_v9 import (
    add_targets, signals_from_proba, apply_regime_filter,
    wilson_interval,
)


# ============================================================
# Pluggable classifier registry (extends v9's logistic/rf/gbm)
# ============================================================

def _try_import_xgb():
    """Returns XGBClassifier only if both import AND a small instantiation succeed.
    Catches XGBoostError raised by missing libomp.dylib at first model construction.
    """
    try:
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=1, verbosity=0)
        return XGBClassifier
    except Exception:
        return None


def _try_import_lgbm():
    try:
        from lightgbm import LGBMClassifier
        LGBMClassifier(n_estimators=1, verbose=-1)
        return LGBMClassifier
    except Exception:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        CatBoostClassifier(iterations=1, verbose=0, allow_writing_files=False)
        return CatBoostClassifier
    except Exception:
        return None


def make_classifier_v11(model_type, random_state=42):
    """Extended classifier factory. Returns None if dep is missing."""
    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state)
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    if model_type == 'xgboost':
        XGB = _try_import_xgb()
        if XGB is None:
            return None
        clf = XGB(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric='logloss', use_label_encoder=False,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    if model_type == 'lightgbm':
        LGB = _try_import_lgbm()
        if LGB is None:
            return None
        clf = LGB(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    if model_type == 'catboost':
        CB = _try_import_catboost()
        if CB is None:
            return None
        clf = CB(
            iterations=300, depth=4, learning_rate=0.05,
            random_seed=random_state, verbose=0, allow_writing_files=False,
        )
        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Item 3: Leave-One-Asset-Out (LOAO)
# ============================================================

def evaluate_loao(pooled_df, feature_cols, model_type, prob_threshold,
                   regime_filter=None, fee=0.001, slippage=0.0005):
    """Train on N-1 assets pooled, test on the held-out asset, rotated."""
    symbols = sorted(pooled_df['symbol'].unique().tolist())
    rows = []

    for held_out in symbols:
        train_df = pooled_df[pooled_df['symbol'] != held_out]
        test_df = pooled_df[pooled_df['symbol'] == held_out].reset_index(drop=True)

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        valid_tr = ~np.any(np.isnan(X_train), axis=1) & ~np.isnan(y_train)
        X_train = X_train[valid_tr]
        y_train = y_train[valid_tr].astype(int)
        if len(np.unique(y_train)) < 2 or len(X_train) < 200:
            continue

        clf = make_classifier_v11(model_type)
        if clf is None:
            return pd.DataFrame()
        clf.fit(X_train, y_train)

        X_test = test_df[feature_cols].values
        valid_te = ~np.any(np.isnan(X_test), axis=1)
        X_test = X_test[valid_te]
        test_df = test_df.iloc[valid_te].reset_index(drop=True)
        if len(X_test) == 0:
            continue

        proba = clf.predict_proba(X_test)
        p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

        true_y = test_df['target'].values.astype(int)
        try:
            auc = roc_auc_score(true_y, p_up)
        except ValueError:
            auc = 0.5

        signals = signals_from_proba(p_up, prob_threshold=prob_threshold)
        if regime_filter:
            signals = apply_regime_filter(signals, test_df, regime_filter)

        correct = total = 0
        for i in range(min(len(signals), len(true_y))):
            sig = signals.iloc[i]['signal']
            if sig == 'BUY':
                total += 1
                correct += int(true_y[i] == 1)
            elif sig == 'SELL':
                total += 1
                correct += int(true_y[i] == 0)

        acc = correct / total if total else 0.5
        successes = int(round(acc * total))
        _, lo, hi = wilson_interval(successes, total) if total else (0.0, 0.0, 1.0)

        prices = test_df['close'].values
        bt = Backtester(trade_fee=fee, slippage=slippage)
        res = bt.run(prices, signals)

        rows.append({
            'held_out': held_out,
            'model': model_type,
            'n_train': int(len(X_train)),
            'n_test': int(len(test_df)),
            'n_signals': total,
            'direction_accuracy': acc,
            'wilson_lo': lo,
            'wilson_hi': hi,
            'auc': auc,
            'sharpe': res['metrics'].get('sharpe_ratio', 0.0),
            'return_pct': res['metrics'].get('total_return_pct', 0.0),
            'bh_return_pct': (prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0.0,
        })

    return pd.DataFrame(rows)


# ============================================================
# Item 4: Stronger baselines (rule-based and supervised)
# ============================================================

def _momentum_signals(asset_df, lookback_hours=24, threshold_bps=20):
    """BUY if past lookback return > +threshold, SELL if < -threshold."""
    close = asset_df['close'].values
    out = []
    for i in range(len(close)):
        if i < lookback_hours:
            out.append({'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5})
            continue
        ret_bps = (close[i] / close[i - lookback_hours] - 1) * 1e4
        if ret_bps > threshold_bps:
            out.append({'signal': 'BUY', 'position_size': 0.10, 'p_up': 0.7})
        elif ret_bps < -threshold_bps:
            out.append({'signal': 'SELL', 'position_size': -0.10, 'p_up': 0.3})
        else:
            out.append({'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5})
    return pd.DataFrame(out)


def _vol_breakout_signals(asset_df, vol_col='realized_vol_20', z_threshold=1.5,
                            lookback=100):
    """BUY when vol z-score > +z_threshold AND last bar return positive (and SELL when negative)."""
    if vol_col not in asset_df.columns:
        return pd.DataFrame([{'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5}
                             for _ in range(len(asset_df))])
    vol = asset_df[vol_col].values
    rets = asset_df['close'].pct_change().fillna(0).values
    out = []
    for i in range(len(vol)):
        if i < lookback:
            out.append({'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5})
            continue
        window = vol[max(0, i - lookback):i]
        mu = window.mean()
        sd = window.std() or 1.0
        z = (vol[i] - mu) / sd
        if z > z_threshold and rets[i - 1] > 0:
            out.append({'signal': 'BUY', 'position_size': 0.10, 'p_up': 0.65})
        elif z > z_threshold and rets[i - 1] < 0:
            out.append({'signal': 'SELL', 'position_size': -0.10, 'p_up': 0.35})
        else:
            out.append({'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5})
    return pd.DataFrame(out)


def _buy_and_hold_signals(asset_df):
    """Always long: BUY on the first bar, HOLD thereafter (no exit)."""
    out = [{'signal': 'BUY', 'position_size': 0.10, 'p_up': 0.5}]
    out.extend([{'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5}
                for _ in range(len(asset_df) - 1)])
    return pd.DataFrame(out)


def _cash_signals(asset_df):
    return pd.DataFrame([{'signal': 'HOLD', 'position_size': 0.0, 'p_up': 0.5}
                         for _ in range(len(asset_df))])


def evaluate_rule_baseline(pooled_df, signal_fn, name,
                            fee=0.001, slippage=0.0005):
    """Apply a rule-based signal generator per-asset, aggregate."""
    rows = []
    for symbol in sorted(pooled_df['symbol'].unique()):
        asset_df = pooled_df[pooled_df['symbol'] == symbol].reset_index(drop=True)
        if len(asset_df) < 50:
            continue
        signals = signal_fn(asset_df)
        true_y = asset_df['target'].values.astype(int)

        correct = total = 0
        for i in range(min(len(signals), len(true_y))):
            sig = signals.iloc[i]['signal']
            if sig == 'BUY':
                total += 1
                correct += int(true_y[i] == 1)
            elif sig == 'SELL':
                total += 1
                correct += int(true_y[i] == 0)

        acc = correct / total if total else 0.5
        successes = int(round(acc * total))
        _, lo, hi = wilson_interval(successes, total) if total else (0.0, 0.0, 1.0)

        prices = asset_df['close'].values
        bt = Backtester(trade_fee=fee, slippage=slippage)
        res = bt.run(prices, signals)
        rows.append({
            'baseline': name,
            'symbol': symbol,
            'n_signals': total,
            'direction_accuracy': acc,
            'wilson_lo': lo,
            'wilson_hi': hi,
            'sharpe': res['metrics'].get('sharpe_ratio', 0.0),
            'return_pct': res['metrics'].get('total_return_pct', 0.0),
            'bh_return_pct': (prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def evaluate_returns_only_baseline(pooled_df, prob_threshold=0.55,
                                     fee=0.001, slippage=0.0005):
    """Logistic on lagged returns only — no TDA, no microstructure beyond returns. LOAO."""
    return_cols = [c for c in ML_FEATURE_SET if 'return' in c]
    return_cols = [c for c in return_cols if c in pooled_df.columns]
    if not return_cols:
        return pd.DataFrame()

    rows = []
    for held_out in sorted(pooled_df['symbol'].unique()):
        train_df = pooled_df[pooled_df['symbol'] != held_out]
        test_df = pooled_df[pooled_df['symbol'] == held_out].reset_index(drop=True)

        X_tr = train_df[return_cols].values
        y_tr = train_df['target'].values
        valid = ~np.any(np.isnan(X_tr), axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[valid], y_tr[valid].astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ])
        clf.fit(X_tr, y_tr)

        X_te = test_df[return_cols].values
        valid_te = ~np.any(np.isnan(X_te), axis=1)
        X_te = X_te[valid_te]
        test_df = test_df.iloc[valid_te].reset_index(drop=True)
        if len(X_te) == 0:
            continue

        proba = clf.predict_proba(X_te)
        p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        signals = signals_from_proba(p_up, prob_threshold=prob_threshold)

        true_y = test_df['target'].values.astype(int)
        correct = total = 0
        for i in range(min(len(signals), len(true_y))):
            sig = signals.iloc[i]['signal']
            if sig == 'BUY':
                total += 1
                correct += int(true_y[i] == 1)
            elif sig == 'SELL':
                total += 1
                correct += int(true_y[i] == 0)
        acc = correct / total if total else 0.5
        successes = int(round(acc * total))
        _, lo, hi = wilson_interval(successes, total) if total else (0.0, 0.0, 1.0)

        prices = test_df['close'].values
        bt = Backtester(trade_fee=fee, slippage=slippage)
        res = bt.run(prices, signals)

        rows.append({
            'baseline': 'returns_only_logistic',
            'symbol': held_out,
            'n_signals': total,
            'direction_accuracy': acc,
            'wilson_lo': lo,
            'wilson_hi': hi,
            'sharpe': res['metrics'].get('sharpe_ratio', 0.0),
            'return_pct': res['metrics'].get('total_return_pct', 0.0),
            'bh_return_pct': (prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


# ============================================================
# Cached pooled-data loader (avoids re-fetching across runs)
# ============================================================

CACHE_PATH = 'data/pooled_v11_cache.parquet'


def get_pooled(symbols, days, force_refresh=False):
    """Cache pooled DataFrame to parquet; reload if same days+symbols."""
    cache_meta = CACHE_PATH + '.meta.txt'
    target_meta = f"days={days}; symbols={','.join(sorted(symbols))}"
    if (not force_refresh and os.path.exists(CACHE_PATH)
            and os.path.exists(cache_meta)):
        with open(cache_meta) as f:
            existing = f.read().strip()
        if existing == target_meta:
            print(f"[cache] Loading pooled data from {CACHE_PATH}")
            return pd.read_parquet(CACHE_PATH)

    print(f"[cache] Building pooled data ({days} days, {len(symbols)} symbols)...")
    pooled = fetch_multi_asset_pool(symbols, days=days, interval='1h')
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    pooled.to_parquet(CACHE_PATH)
    with open(cache_meta, 'w') as f:
        f.write(target_meta)
    return pooled


# ============================================================
# Orchestrator
# ============================================================

def run_v11(symbols=None, days=1095, horizon=72,
             prob_threshold=0.65, regime_filter=None):
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX']

    print(f"\n{'#' * 70}")
    print(f"#  V11: Leave-One-Asset-Out + Stronger Baselines")
    print(f"#  Symbols: {symbols} | Days: {days} ({days/365:.1f} years)")
    print(f"#  Horizon: {horizon}h ({horizon/24:.1f}d)")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print("[1/4] Loading multi-asset pool...")
    pooled = get_pooled(symbols, days=days)
    feature_cols = get_combined_feature_cols(pooled)
    pooled = add_targets(pooled, horizon=horizon)
    print(f"  Pool size: {len(pooled):,} samples, {len(feature_cols)} features")

    # --- LOAO across multiple models ---
    print(f"\n[2/4] Leave-one-asset-out (LOAO) — predict each asset from the other {len(symbols)-1}")
    model_grid = [
        ('logistic', prob_threshold, regime_filter),
        ('xgboost', prob_threshold, regime_filter),
        ('lightgbm', prob_threshold, regime_filter),
        ('catboost', prob_threshold, regime_filter),
    ]
    loao_rows = []
    for model_type, thresh, filt in model_grid:
        clf_check = make_classifier_v11(model_type)
        if clf_check is None:
            print(f"  [{model_type}] skipped — dependency not installed")
            continue
        print(f"  [{model_type}] running LOAO...")
        df = evaluate_loao(pooled, feature_cols, model_type, thresh,
                            regime_filter=filt)
        if len(df) > 0:
            df['threshold'] = thresh
            df['regime_filter'] = str(filt) if filt else 'none'
            loao_rows.append(df)
            print(f"    pooled accuracy: "
                  f"{(df['direction_accuracy'] * df['n_signals']).sum() / max(df['n_signals'].sum(), 1):.2%} "
                  f"on n={int(df['n_signals'].sum())}")

    loao_df = pd.concat(loao_rows, ignore_index=True) if loao_rows else pd.DataFrame()

    # --- Baselines ---
    print(f"\n[3/4] Stronger baselines on the same pooled data")
    baselines = []

    print("  [momentum] 24h lookback, 20 bps threshold")
    baselines.append(evaluate_rule_baseline(
        pooled, lambda d: _momentum_signals(d, lookback_hours=24, threshold_bps=20),
        name='momentum_24h_20bps'))

    print("  [vol_breakout] z>1.5 on realized_vol_20")
    baselines.append(evaluate_rule_baseline(
        pooled, lambda d: _vol_breakout_signals(d, z_threshold=1.5),
        name='vol_breakout_z15'))

    print("  [buy_and_hold]")
    baselines.append(evaluate_rule_baseline(pooled, _buy_and_hold_signals, name='buy_and_hold'))

    print("  [cash]")
    baselines.append(evaluate_rule_baseline(pooled, _cash_signals, name='cash'))

    print("  [returns_only_logistic] (LOAO)")
    baselines.append(evaluate_returns_only_baseline(pooled, prob_threshold=0.55))

    baselines_df = pd.concat([b for b in baselines if len(b) > 0], ignore_index=True)

    # --- Save ---
    print(f"\n[4/4] Saving outputs")
    if len(loao_df) > 0:
        loao_df.to_csv('results/v11_loao.csv', index=False)
        loao_summary = (
            loao_df.groupby('model')
            .apply(lambda g: pd.Series({
                'pooled_accuracy': (g['direction_accuracy'] * g['n_signals']).sum()
                                   / max(g['n_signals'].sum(), 1),
                'pooled_signals': int(g['n_signals'].sum()),
                'mean_sharpe': g['sharpe'].mean(),
            }))
            .reset_index()
        )
        loao_summary.to_csv('results/v11_loao_by_model.csv', index=False)
        print(f"  Saved: results/v11_loao.csv ({len(loao_df)} rows)")
        print(f"  Saved: results/v11_loao_by_model.csv")

    if len(baselines_df) > 0:
        baselines_df.to_csv('results/v11_baselines.csv', index=False)
        print(f"  Saved: results/v11_baselines.csv ({len(baselines_df)} rows)")

    write_v11_report(loao_df, baselines_df, symbols, days, horizon)

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"#  V11 Complete in {elapsed/60:.1f} min")
    print(f"#  See: results/V11_LOAO_BASELINES.md")
    print(f"{'#' * 70}\n")


def write_v11_report(loao_df, baselines_df, symbols, days, horizon):
    md = []
    md.append(f"# V11: Leave-One-Asset-Out + Stronger Baselines\n")
    md.append(f"**Goal:** Resolve two methodological items:")
    md.append(f"1. Cross-asset transferability (LOAO)")
    md.append(f"2. Comparison against stronger baselines (XGB / LightGBM / CatBoost / momentum / vol-breakout / B&H / cash / returns-only)\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    md.append(f"**Symbols:** {', '.join(symbols)}")
    md.append(f"**Sample:** {days} days hourly | Horizon: {horizon}h ({horizon/24:.1f}d)\n")

    md.append(f"## 1. Leave-One-Asset-Out\n")
    if len(loao_df) > 0:
        md.append(f"For each held-out asset, the model is trained on the union of the other "
                  f"{len(symbols)-1} assets and evaluated only on the held-out asset's signals.\n")
        md.append(f"### 1.1 Per-(model, held-out asset)\n")
        md.append(f"| Model | Held-Out | n Signals | Direction Acc | Wilson 95% | AUC | Sharpe | Return % |")
        md.append(f"|-------|----------|----------:|--------------:|-----------|----:|-------:|---------:|")
        for _, r in loao_df.iterrows():
            md.append(f"| {r['model']} | {r['held_out']} | {int(r['n_signals'])} | "
                      f"{r['direction_accuracy']:.2%} | "
                      f"[{r['wilson_lo']:.2%}, {r['wilson_hi']:.2%}] | "
                      f"{r['auc']:.3f} | {r['sharpe']:.2f} | {r['return_pct']:.2f}% |")
        md.append("")

        md.append(f"### 1.2 Per-model pooled summary\n")
        md.append(f"| Model | Pooled Accuracy | Pooled Signals | Mean Sharpe |")
        md.append(f"|-------|----------------:|---------------:|------------:|")
        for model, g in loao_df.groupby('model'):
            n_sig = int(g['n_signals'].sum())
            pooled_acc = ((g['direction_accuracy'] * g['n_signals']).sum()
                          / max(n_sig, 1))
            md.append(f"| {model} | {pooled_acc:.2%} | {n_sig} | "
                      f"{g['sharpe'].mean():.2f} |")
        md.append("")
    else:
        md.append("LOAO produced no rows (likely a missing dependency or insufficient data).\n")

    md.append(f"## 2. Baseline Ladder\n")
    if len(baselines_df) > 0:
        md.append(f"Each baseline is evaluated on the same pooled data with the same backtest "
                  f"cost model (0.001 fee per side, 0.0005 slippage per side).\n")
        md.append(f"### 2.1 Per-(baseline, asset)\n")
        md.append(f"| Baseline | Symbol | n Signals | Direction Acc | Wilson 95% | Sharpe | Return % | B&H Return |")
        md.append(f"|----------|--------|----------:|--------------:|-----------|-------:|---------:|-----------:|")
        for _, r in baselines_df.iterrows():
            md.append(f"| {r['baseline']} | {r['symbol']} | {int(r['n_signals'])} | "
                      f"{r['direction_accuracy']:.2%} | "
                      f"[{r['wilson_lo']:.2%}, {r['wilson_hi']:.2%}] | "
                      f"{r['sharpe']:.2f} | {r['return_pct']:.2f}% | "
                      f"{r['bh_return_pct']:.2f}% |")
        md.append("")

        md.append(f"### 2.2 Per-baseline pooled summary\n")
        md.append(f"| Baseline | Pooled Accuracy | Total Signals | Mean Sharpe | Mean Return % |")
        md.append(f"|----------|----------------:|--------------:|------------:|--------------:|")
        for name, g in baselines_df.groupby('baseline'):
            n_sig = int(g['n_signals'].sum())
            pooled_acc = ((g['direction_accuracy'] * g['n_signals']).sum()
                          / max(n_sig, 1))
            md.append(f"| {name} | {pooled_acc:.2%} | {n_sig} | "
                      f"{g['sharpe'].mean():.2f} | {g['return_pct'].mean():.2f}% |")
        md.append("")
    else:
        md.append("No baseline rows produced.\n")

    md.append(f"## 3. Verdict\n")
    md.append(f"Read this section against the v9/v10 headline numbers.\n")
    md.append(f"- **Cross-asset transferability:** if LOAO accuracy stays > 50% (Wilson lower bound) "
              f"on most held-out assets, the multi-asset story holds. If it drops to chance only on "
              f"specific assets, those assets are likely riding asset-specific quirks rather than "
              f"transferable structure.")
    md.append(f"- **Vs. stronger baselines:** XGBoost/LightGBM/CatBoost results bracket the realistic "
              f"upper bound for a Base+TDA gradient-boosted classifier. If the v9 logistic configuration "
              f"is within 1-2pp of the boosted ceiling, the v9 model selection was reasonable; a much "
              f"larger gap would suggest a boosted model is preferable.")
    md.append(f"- **Vs. momentum / vol-breakout / returns-only logistic:** if the TDA pipeline beats "
              f"these by more than the Wilson-CI margin, the case for TDA-feature contribution is "
              f"strengthened. If a returns-only logistic matches it, TDA's marginal value is below "
              f"the noise floor of the experiment.")

    with open('results/V11_LOAO_BASELINES.md', 'w') as f:
        f.write("\n".join(md))
    print(f"  Saved: results/V11_LOAO_BASELINES.md")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    run_v11(days=days)
