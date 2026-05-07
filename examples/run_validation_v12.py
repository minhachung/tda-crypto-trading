#!/usr/bin/env python
"""
V12: TDA-Representation Ablation (v1 scalars vs v2 persistence images).

Tests the central claim from v9: TDA features carry real structure
(perm p < 0.001, 58.5% feature importance) but adding them on top of
base features HURTS net accuracy by 5.6pp. v12 attributes that hurt
to the v1 representation (8 hand-engineered scalar stats per dim)
being too lossy, and tests whether persistence images recover
ablation-positive contribution.

Five-way ablation, on identical point clouds and identical
train/holdout splits, evaluated under both a linear (logistic) and a
non-linear (xgboost) classifier:

    base                   21 features
    tda_v1                 16 features (existing)
    tda_v2                200 features (10x10 persistence image per H0/H1)
    base + tda_v1          37 features
    base + tda_v2         221 features

Output:
  - results/V12_TDA_REP.md
  - results/v12_ablation.csv
  - results/figures/v12_ablation_compare.{png,pdf}

Usage:
  python examples/run_validation_v12.py            # 1095 days default
  python examples/run_validation_v12.py 365        # quicker run
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.binance_data import HighFreqFetcher
from src.advanced_features import (
    build_advanced_features, get_tda_features, ML_FEATURE_SET,
)
from src.persistent_homology import compute_features_for_windows
from src.tda_v2_features import (
    compute_v2_features_for_windows, get_v2_feature_cols,
)
from src.backtester import Backtester
from src.validation_v2 import wilson_interval, time_series_kfold


# ============================================================
# Data prep — both v1 and v2 features from the same point clouds
# ============================================================

def normalize_features_array(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def create_sliding_windows(X, window_size=20, stride=1):
    pcs, end_idx = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        pcs.append(X[i:i + window_size])
        end_idx.append(i + window_size - 1)
    return pcs, end_idx


def fetch_and_build_asset_v12(symbol, days=1095, interval='1h', window_size=20):
    """Per-asset: fetch -> features -> windows -> diagrams -> v1+v2 features.

    Returns (asset_df_aligned, point_clouds_list, end_idx_array).
    """
    print(f"  [{symbol}] Fetching {days} days of {interval}")
    fetcher = HighFreqFetcher(symbol=symbol, interval=interval)
    raw_df = fetcher.fetch_history(days=days)

    print(f"  [{symbol}] Building advanced features")
    df = build_advanced_features(raw_df)

    if len(df) < window_size + 50:
        print(f"  [{symbol}] WARNING: too few rows after feature build")
        return None, None, None

    X_tda, _ = get_tda_features(df)
    X_norm = normalize_features_array(X_tda)
    pcs, end_idx = create_sliding_windows(X_norm, window_size=window_size, stride=1)
    print(f"  [{symbol}] Created {len(pcs)} point clouds")

    aligned_idx = np.array(end_idx, dtype=int)
    aligned_idx = aligned_idx[aligned_idx < len(df)]
    aligned_df = df.iloc[aligned_idx].reset_index(drop=True)
    aligned_df['symbol'] = symbol

    return aligned_df, pcs[:len(aligned_df)], aligned_idx[:len(aligned_df)]


def build_pool_with_both_tda(symbols, days=1095, window_size=20):
    """Pool data across symbols, computing both v1 (16 scalar) and
    v2 (persistence-image) TDA features from the same point clouds."""
    pool_pieces = []
    for symbol in symbols:
        try:
            asset_df, pcs, end_idx = fetch_and_build_asset_v12(
                symbol, days=days, window_size=window_size,
            )
            if asset_df is None:
                continue

            print(f"  [{symbol}] Computing v1 (scalar) TDA features")
            tda_v1 = compute_features_for_windows(pcs, end_indices=end_idx,
                                                    verbose=False)
            tda_v1 = tda_v1.iloc[:len(asset_df)].reset_index(drop=True)

            print(f"  [{symbol}] Computing v2 (persistence-image) TDA features")
            tda_v2 = compute_v2_features_for_windows(pcs, end_indices=end_idx,
                                                      verbose=False)
            tda_v2 = tda_v2.iloc[:len(asset_df)].reset_index(drop=True)

            tda_v1 = tda_v1.drop(columns=[c for c in ('window_idx', 'end_idx')
                                            if c in tda_v1.columns])
            tda_v2 = tda_v2.drop(columns=[c for c in ('window_idx', 'end_idx')
                                            if c in tda_v2.columns])

            combined = pd.concat([
                asset_df.reset_index(drop=True),
                tda_v1.reset_index(drop=True),
                tda_v2.reset_index(drop=True),
            ], axis=1)
            print(f"  [{symbol}] Final: {len(combined)} samples, "
                  f"{combined.shape[1]} columns")
            pool_pieces.append(combined)
            time.sleep(1)
        except Exception as e:
            print(f"  [{symbol}] ERROR: {e}")

    if not pool_pieces:
        raise ValueError("No assets successfully processed")

    pooled = pd.concat(pool_pieces, ignore_index=True)
    print(f"\n[Pool] Combined: {len(pooled)} samples across {len(pool_pieces)} assets")
    return pooled


# ============================================================
# Targets + feature-set selectors
# ============================================================

def add_targets_v12(df, horizon=72):
    df = df.copy()
    df['target'] = (
        df.groupby('symbol')['close']
        .transform(lambda x: (x.shift(-horizon) > x).astype(float))
    )
    return df.dropna(subset=['target']).reset_index(drop=True)


def base_cols(df):
    return [c for c in ML_FEATURE_SET if c in df.columns]


def v1_cols(df):
    return [c for c in df.columns if c.startswith(('H0_', 'H1_'))]


def v2_cols(df):
    return get_v2_feature_cols(df)


# ============================================================
# Classifier factory + evaluation
# ============================================================

def _try_xgb():
    try:
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=1, verbosity=0)
        return XGBClassifier
    except Exception:
        return None


def make_clf(model_type, random_state=42):
    if model_type == 'logistic':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, C=0.5, random_state=random_state)),
        ])
    if model_type == 'xgboost':
        XGB = _try_xgb()
        if XGB is None:
            return None
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGB(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric='logloss', use_label_encoder=False,
                random_state=random_state, n_jobs=-1, verbosity=0,
            )),
        ])
    raise ValueError(f"Unknown model: {model_type}")


def evaluate_kfold(df, feature_cols, model_type, prob_threshold=0.65,
                    n_splits=5):
    """Time-series 5-fold CV across pooled symbols. Same protocol as v9."""
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
        train_X, train_y = [], []
        for symbol, (asset_df, folds) in asset_folds.items():
            if fold_idx >= len(folds):
                continue
            tr_idx, _ = folds[fold_idx]
            train_X.append(asset_df.iloc[tr_idx][feature_cols].values)
            train_y.append(asset_df.iloc[tr_idx]['target'].values)
        if not train_X:
            continue
        X = np.vstack(train_X)
        y = np.concatenate(train_y).astype(int)
        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]
        if len(np.unique(y)) < 2 or len(X) < 50:
            continue

        clf = make_clf(model_type)
        if clf is None:
            return pd.DataFrame()
        clf.fit(X, y)

        for symbol, (asset_df, folds) in asset_folds.items():
            if fold_idx >= len(folds):
                continue
            _, te_idx = folds[fold_idx]
            test_df = asset_df.iloc[te_idx].reset_index(drop=True)
            X_test = test_df[feature_cols].values
            valid_te = ~np.any(np.isnan(X_test), axis=1)
            if valid_te.sum() == 0:
                continue
            test_df = test_df.iloc[valid_te].reset_index(drop=True)
            X_test = X_test[valid_te]

            proba = clf.predict_proba(X_test)
            p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
            true_y = test_df['target'].values.astype(int)
            try:
                auc = roc_auc_score(true_y, p_up)
            except ValueError:
                auc = 0.5

            correct = total = 0
            for i in range(len(p_up)):
                p = p_up[i]
                if p > prob_threshold:
                    total += 1
                    correct += int(true_y[i] == 1)
                elif p < 1 - prob_threshold:
                    total += 1
                    correct += int(true_y[i] == 0)

            acc = correct / total if total else 0.5
            rows.append({
                'fold': fold_idx,
                'symbol': symbol,
                'n_signals': total,
                'direction_accuracy': acc,
                'auc': auc,
            })

    return pd.DataFrame(rows)


# ============================================================
# Orchestrator
# ============================================================

ABLATIONS = [
    ('base',           lambda df: base_cols(df)),
    ('tda_v1',         lambda df: v1_cols(df)),
    ('tda_v2',         lambda df: v2_cols(df)),
    ('base_plus_v1',   lambda df: base_cols(df) + v1_cols(df)),
    ('base_plus_v2',   lambda df: base_cols(df) + v2_cols(df)),
]

MODELS = ['logistic', 'xgboost']


def run_v12(symbols=None, days=1095, horizon=72, prob_threshold=0.65):
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX']

    print(f"\n{'#' * 70}")
    print(f"#  V12: TDA-Representation Ablation (v1 scalars vs v2 images)")
    print(f"#  Symbols: {symbols} | Days: {days} | Horizon: {horizon}h")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print("[1/3] Building pool with both v1 and v2 TDA features...")
    pooled = build_pool_with_both_tda(symbols, days=days, window_size=20)
    pooled = add_targets_v12(pooled, horizon=horizon)
    print(f"  Pool size: {len(pooled):,} samples after target alignment")
    print(f"  Base features:    {len(base_cols(pooled))}")
    print(f"  v1 TDA features:  {len(v1_cols(pooled))}")
    print(f"  v2 TDA features:  {len(v2_cols(pooled))}")

    print(f"\n[2/3] Running 5x{len(MODELS)} ablation grid")
    rows = []
    for ab_name, col_fn in ABLATIONS:
        cols = col_fn(pooled)
        if not cols:
            print(f"  [{ab_name}] no columns matched, skipping")
            continue
        for model_type in MODELS:
            clf_check = make_clf(model_type)
            if clf_check is None:
                print(f"  [{ab_name} | {model_type}] dep missing, skipping")
                continue
            print(f"  [{ab_name:>14} | {model_type:>8}] {len(cols)} features ...")
            df_eval = evaluate_kfold(pooled, cols, model_type,
                                       prob_threshold=prob_threshold)
            if len(df_eval) == 0:
                continue
            n_sig = int(df_eval['n_signals'].sum())
            if n_sig == 0:
                acc, lo, hi = 0.5, 0.0, 1.0
            else:
                acc = ((df_eval['direction_accuracy'] * df_eval['n_signals']).sum()
                       / n_sig)
                successes = int(round(acc * n_sig))
                _, lo, hi = wilson_interval(successes, n_sig)
            mean_auc = float(df_eval['auc'].mean())
            rows.append({
                'feature_set': ab_name,
                'model': model_type,
                'n_features': len(cols),
                'n_signals': n_sig,
                'direction_accuracy': acc,
                'wilson_lo': lo,
                'wilson_hi': hi,
                'mean_auc': mean_auc,
            })
            print(f"    -> {acc:.2%} on n={n_sig}, "
                  f"Wilson [{lo:.2%}, {hi:.2%}], AUC {mean_auc:.3f}")

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv('results/v12_ablation.csv', index=False)
    print(f"\n  Saved: results/v12_ablation.csv")

    print(f"\n[3/3] Writing report and figure...")
    write_v12_report(ab_df, symbols, days, horizon)
    plot_v12_figure(ab_df)

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"#  V12 Complete in {elapsed/60:.1f} min")
    print(f"#  See: results/V12_TDA_REP.md")
    print(f"{'#' * 70}\n")


def write_v12_report(ab_df, symbols, days, horizon):
    md = []
    md.append("# V12: TDA-Representation Ablation\n")
    md.append("**Question:** Is the v9 ablation-negative result (Base+TDA "
              "underperforms Base alone by 5.6pp) caused by the v1 representation "
              "(8 hand-engineered scalar stats per dimension) being too lossy?\n")
    md.append("**Test:** Re-run the same ablation with TDA features replaced by "
              "**persistence images** (Adams et al. 2017) — 10×10 grid per "
              "homology dimension = 200 features total, fit once per session, "
              "transform every window. Compare v1 vs v2 representations under "
              "two classifiers (logistic and xgboost), with all other "
              "experimental settings (windowing, point-cloud construction, "
              "k-fold protocol) held constant.\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    md.append(f"**Symbols:** {', '.join(symbols)}")
    md.append(f"**Sample:** {days} days hourly | Horizon: {horizon}h\n")

    md.append("## 1. Ablation Results\n")
    md.append("| Feature Set | Model | n Features | n Signals | Direction Acc | Wilson 95% | AUC |")
    md.append("|-------------|-------|-----------:|----------:|--------------:|-----------|----:|")
    for _, r in ab_df.iterrows():
        md.append(f"| {r['feature_set']} | {r['model']} | {int(r['n_features'])} | "
                  f"{int(r['n_signals'])} | {r['direction_accuracy']:.2%} | "
                  f"[{r['wilson_lo']:.2%}, {r['wilson_hi']:.2%}] | {r['mean_auc']:.3f} |")
    md.append("")

    md.append("## 2. Diagnostic\n")
    by_model = ab_df.set_index(['model', 'feature_set'])['direction_accuracy']
    for model in MODELS:
        if (model, 'base') not in by_model.index:
            continue
        base_acc = by_model.loc[(model, 'base')]
        v1_delta = by_model.get((model, 'base_plus_v1'), np.nan) - base_acc
        v2_delta = by_model.get((model, 'base_plus_v2'), np.nan) - base_acc
        md.append(f"### {model}\n")
        md.append(f"- Base alone: {base_acc:.2%}")
        md.append(f"- Base + TDA v1 (16 scalars): Δ = {v1_delta:+.2%}")
        md.append(f"- Base + TDA v2 (persistence images): Δ = {v2_delta:+.2%}\n")
        if not np.isnan(v2_delta) and v2_delta > 0:
            md.append("**Verdict for this model:** Persistence images add positive "
                      "marginal accuracy on top of base features. The v9 "
                      "ablation-negative result was driven by the v1 representation, "
                      "not by an absence of TDA signal.\n")
        elif not np.isnan(v2_delta) and v2_delta <= 0:
            md.append("**Verdict for this model:** Even the richer v2 representation "
                      "does not add positive marginal accuracy. The v9 ablation-negative "
                      "result reflects a deeper limitation than feature lossiness; "
                      "future work should explore window size, filtration choice, or "
                      "alternative TDA constructions (Mapper, persistence landscapes "
                      "at multiple resolutions).\n")

    md.append("## 3. Read against v9\n")
    md.append("v9 reported, on a similar 365-day pool: Base 65.93%, Base+v1_TDA 60.38% "
              "(Δ = −5.55pp). v12 results above are on a 1,095-day pool with the same "
              "windowing and identical point clouds for both representations — the only "
              "variable is the persistence-diagram → feature mapping. A positive "
              "v2 delta narrows the v9 gap and supports the diagnosis that v1's lossy "
              "scalar summaries, not TDA itself, were the issue.\n")

    with open('results/V12_TDA_REP.md', 'w') as f:
        f.write("\n".join(md))
    print(f"  Saved: results/V12_TDA_REP.md")


def plot_v12_figure(ab_df):
    if len(ab_df) == 0:
        return
    feature_sets = ['base', 'tda_v1', 'tda_v2', 'base_plus_v1', 'base_plus_v2']
    models = sorted(ab_df['model'].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_w = 0.35
    x = np.arange(len(feature_sets))

    for i, model in enumerate(models):
        ys, los, his = [], [], []
        for fs in feature_sets:
            row = ab_df[(ab_df['model'] == model) &
                        (ab_df['feature_set'] == fs)]
            if len(row) == 0:
                ys.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            ys.append(float(row['direction_accuracy'].iloc[0]) * 100)
            los.append(float(row['wilson_lo'].iloc[0]) * 100)
            his.append(float(row['wilson_hi'].iloc[0]) * 100)
        ys = np.array(ys); los = np.array(los); his = np.array(his)
        offsets = (i - (len(models) - 1) / 2) * bar_w
        yerr = np.stack([ys - los, his - ys])
        ax.bar(x + offsets, ys, bar_w, yerr=yerr, capsize=3,
                label=model, edgecolor='black', linewidth=0.5)

    ax.axhline(50.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label='50% chance')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, rotation=15, ha='right')
    ax.set_ylabel('Direction accuracy (%)')
    ax.set_title('V12: TDA Representation Ablation\n'
                 'v1 = 16 scalar stats  |  v2 = 10x10 persistence image per H0/H1')
    ax.legend(loc='lower right')
    upper = max(80.0, float(np.nanmax(ab_df['wilson_hi'])) * 100 + 5.0)
    ax.set_ylim(40, upper)
    plt.tight_layout()
    plt.savefig('results/figures/v12_ablation_compare.png', dpi=150)
    plt.savefig('results/figures/v12_ablation_compare.pdf')
    plt.close()
    print(f"  Saved: results/figures/v12_ablation_compare.{{png,pdf}}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    run_v12(days=days)
