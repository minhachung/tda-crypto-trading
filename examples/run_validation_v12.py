#!/usr/bin/env python
"""
V12: TDA-Representation Ablation (v1 scalars vs v2 persistence images).

Tests v9's ablation-negative finding (Base+TDA underperforms Base alone
by 5.6pp) under a richer TDA representation: persistence images
(Adams et al., JMLR 2017) instead of 8 hand-engineered scalar
statistics per homology dimension.

LEAKAGE-SAFETY
--------------
This is the critical correctness contract for v12. A persistence image
is the output of fit-then-transform: fitting determines (birth, pers)
grid bounds; transforming samples kernel density on that grid. Fitting
on the entire pool would leak the future TDA-feature distribution into
the representation — exactly the class of leak the post-fix pipeline
is supposed to prevent.

v12 therefore:
  - Computes raw persistence diagrams ONCE (intrinsic to each window;
    no leakage in the diagram step itself).
  - Caches those diagrams in memory for the duration of the run.
  - For each k-fold split, fits a fresh PersistenceImager on the
    TRAIN-fold diagrams ONLY, then transforms train+test diagrams
    with that fitted imager. The image features for the same window
    therefore differ across folds — that's the correct behaviour.

FIVE-WAY ABLATION
-----------------
For each classifier ∈ {logistic (L2-regularised), xgboost}:

    base                    21 features
    base + tda_v1           37 features  (existing 16 scalar stats)
    base + tda_v2          221 features  (new 200-dim persistence image)
    base + tda_v1 + tda_v2 237 features  (everything)
    base + shuffled_v2     221 features  (negative control:
                                          v2 features randomly
                                          permuted across windows
                                          inside each train fold)

A positive Δ on Base+v2 vs Base alone — under either classifier —
attributes the v9 ablation-negative result to the v1 representation.
A non-positive Δ on Base+v2, with the shuffled-v2 control matching
unshuffled v2, indicates the limitation is deeper than feature
lossiness (window size, filtration, point-cloud construction).

OUTPUT
------
  - results/V12_TDA_REP.md
  - results/v12_ablation.csv
  - results/figures/v12_ablation_compare.{png,pdf}

USAGE
-----
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
from sklearn.metrics import roc_auc_score

from src.binance_data import HighFreqFetcher
from src.advanced_features import (
    build_advanced_features, get_tda_features, ML_FEATURE_SET,
)
from src.persistent_homology import compute_features_for_windows
from src.tda_v2_features import (
    compute_diagrams_only, LeakSafePersistenceImagerFitter,
)
from src.validation_v2 import wilson_interval, time_series_kfold


# ============================================================
# Per-asset preprocessing
# ============================================================

def _normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _windows(X, window_size=20):
    pcs, end_idx = [], []
    for i in range(0, len(X) - window_size + 1):
        pcs.append(X[i:i + window_size])
        end_idx.append(i + window_size - 1)
    return pcs, end_idx


def prepare_asset(symbol, days=1095, window_size=20, verbose=True):
    """For one symbol, return everything v12 needs:
    {
      'df': aligned DataFrame with base features + 16 v1 TDA features + symbol,
      'h0_diagrams': list of (n_i, 2) arrays — one per window, finite-persistence H0,
      'h1_diagrams': list of (m_i, 2) arrays — one per window, finite-persistence H1,
    }
    The persistence diagrams in this dict carry no temporal leakage —
    each is intrinsic to a single window.
    """
    if verbose:
        print(f"  [{symbol}] Fetching {days} days hourly")
    fetcher = HighFreqFetcher(symbol=symbol, interval='1h')
    raw_df = fetcher.fetch_history(days=days)

    if verbose:
        print(f"  [{symbol}] Building advanced features")
    df = build_advanced_features(raw_df)
    if len(df) < window_size + 50:
        if verbose:
            print(f"  [{symbol}] Too few rows; skipped")
        return None

    X_tda, _ = get_tda_features(df)
    X_norm = _normalize(X_tda)
    pcs, end_idx = _windows(X_norm, window_size=window_size)
    if verbose:
        print(f"  [{symbol}] Created {len(pcs)} point clouds")

    aligned_idx = np.array(end_idx, dtype=int)
    aligned_idx = aligned_idx[aligned_idx < len(df)]
    aligned_df = df.iloc[aligned_idx].reset_index(drop=True)
    aligned_df['symbol'] = symbol

    if verbose:
        print(f"  [{symbol}] Computing v1 (scalar) TDA features")
    tda_v1 = compute_features_for_windows(pcs[:len(aligned_df)],
                                            end_indices=aligned_idx[:len(aligned_df)],
                                            verbose=False)
    tda_v1 = tda_v1.iloc[:len(aligned_df)].reset_index(drop=True)
    tda_v1 = tda_v1.drop(columns=[c for c in ('window_idx', 'end_idx')
                                    if c in tda_v1.columns])
    aligned_df = pd.concat([aligned_df.reset_index(drop=True),
                              tda_v1.reset_index(drop=True)], axis=1)

    if verbose:
        print(f"  [{symbol}] Computing raw persistence diagrams (no fit, no leak)")
    h0_list, h1_list = compute_diagrams_only(pcs[:len(aligned_df)],
                                                max_dim=1, verbose=False)

    if verbose:
        print(f"  [{symbol}] Final: {len(aligned_df)} samples, "
              f"{len(h0_list)} diagrams")
    return {'df': aligned_df, 'h0_diagrams': h0_list, 'h1_diagrams': h1_list}


# ============================================================
# Targets + feature-set selectors
# ============================================================

def add_targets(df, horizon=72):
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


# ============================================================
# Classifier factory
# ============================================================

def _try_xgb():
    try:
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=1, verbosity=0)
        return XGBClassifier
    except Exception:
        return None


def make_clf(model_type, random_state=42):
    """C=0.5 logistic for stronger L2 reg on the 200+ dim v2 inputs."""
    if model_type == 'logistic':
        return LogisticRegression(max_iter=500, C=0.5,
                                    random_state=random_state)
    if model_type == 'xgboost':
        XGB = _try_xgb()
        if XGB is None:
            return None
        return XGB(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric='logloss', use_label_encoder=False,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_type}")


# ============================================================
# Leak-safe k-fold evaluation
# ============================================================

FEATURE_SETS = [
    'base',
    'base_plus_v1',
    'base_plus_v2',
    'base_plus_v1_plus_v2',
    'base_plus_shuffled_v2',
]

MODELS = ['logistic', 'xgboost']


def evaluate_kfold_leaksafe(asset_data, feature_set, model_type,
                              prob_threshold=0.65, n_splits=5,
                              imager_resolution=10, random_state=42,
                              verbose=True):
    """Time-series 5-fold CV, leak-safe.

    asset_data : {symbol -> {'df', 'h0_diagrams', 'h1_diagrams'}}
                 with df containing target + base + v1 columns.
    feature_set : one of FEATURE_SETS.
    model_type  : one of MODELS.

    Inside each fold:
      1. Per-symbol time_series_kfold splits → train/test indices.
      2. If feature_set needs v2: fit a LeakSafePersistenceImagerFitter
         on the UNION of all assets' train diagrams ONLY, then transform
         train + test diagrams per asset.
      3. If feature_set is 'base_plus_shuffled_v2': permute train v2
         rows within each asset (negative control) before fitting clf.
      4. Train classifier on union of assets' train rows; evaluate per
         asset on test rows. Aggregate accuracy.
    """
    needs_v2 = 'v2' in feature_set
    rng = np.random.RandomState(random_state)

    # Per-asset cached fold splits
    fold_cache = {}
    for symbol, data in asset_data.items():
        if len(data['df']) < n_splits + 50:
            continue
        fold_cache[symbol] = (data, time_series_kfold(len(data['df']),
                                                       n_splits=n_splits))

    rows = []
    for fold_idx in range(n_splits):
        # --- Step A: collect train indices per asset and (if needed)
        # accumulate train diagrams across assets to fit a single imager.
        per_asset_split = {}
        all_train_h0, all_train_h1 = [], []
        for symbol, (data, folds) in fold_cache.items():
            if fold_idx >= len(folds):
                continue
            tr_idx, te_idx = folds[fold_idx]
            per_asset_split[symbol] = (data, tr_idx, te_idx)
            if needs_v2:
                all_train_h0.extend(data['h0_diagrams'][i] for i in tr_idx)
                all_train_h1.extend(data['h1_diagrams'][i] for i in tr_idx)

        imager = None
        if needs_v2:
            imager = LeakSafePersistenceImagerFitter(
                resolution=imager_resolution,
            ).fit(all_train_h0, all_train_h1)
            if verbose:
                print(f"    [fold {fold_idx}] imager fit on "
                      f"{len(all_train_h0)} train diagrams")

        # --- Step B: build train and test feature matrices per asset.
        train_X_parts, train_y_parts = [], []
        test_per_asset = {}
        for symbol, (data, tr_idx, te_idx) in per_asset_split.items():
            # Base
            base = data['df'][base_cols(data['df'])].values
            # v1 scalars
            v1 = data['df'][v1_cols(data['df'])].values

            # Train v2 (transform with imager)
            if needs_v2:
                tr_h0 = [data['h0_diagrams'][i] for i in tr_idx]
                tr_h1 = [data['h1_diagrams'][i] for i in tr_idx]
                te_h0 = [data['h0_diagrams'][i] for i in te_idx]
                te_h1 = [data['h1_diagrams'][i] for i in te_idx]
                v2_train = imager.transform(tr_h0, tr_h1)
                v2_test = imager.transform(te_h0, te_h1)
            else:
                v2_train = None
                v2_test = None

            tr_x = _assemble(feature_set, base[tr_idx], v1[tr_idx],
                              v2_train, rng=rng, is_train=True)
            te_x = _assemble(feature_set, base[te_idx], v1[te_idx],
                              v2_test, rng=rng, is_train=False)

            tr_y = data['df'].iloc[tr_idx]['target'].values.astype(int)
            te_y = data['df'].iloc[te_idx]['target'].values.astype(int)

            tr_valid = ~np.any(np.isnan(tr_x), axis=1)
            tr_x, tr_y = tr_x[tr_valid], tr_y[tr_valid]
            te_valid = ~np.any(np.isnan(te_x), axis=1)
            te_x, te_y = te_x[te_valid], te_y[te_valid]

            if len(tr_y) > 0:
                train_X_parts.append(tr_x)
                train_y_parts.append(tr_y)
            test_per_asset[symbol] = (te_x, te_y)

        if not train_X_parts:
            continue
        X = np.vstack(train_X_parts)
        y = np.concatenate(train_y_parts).astype(int)
        if len(np.unique(y)) < 2:
            continue

        # Standardise across all features (essential for logistic + L2 on
        # a 200-dim v2 representation).
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        clf = make_clf(model_type, random_state=42)
        if clf is None:
            return pd.DataFrame()
        clf.fit(Xz, y)

        for symbol, (te_x, te_y) in test_per_asset.items():
            if len(te_y) == 0:
                continue
            Xz_te = scaler.transform(te_x)
            try:
                proba = clf.predict_proba(Xz_te)
            except Exception:
                continue
            p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
            try:
                auc = roc_auc_score(te_y, p_up)
            except ValueError:
                auc = 0.5

            correct = total = 0
            for i in range(len(p_up)):
                p = p_up[i]
                if p > prob_threshold:
                    total += 1
                    correct += int(te_y[i] == 1)
                elif p < 1 - prob_threshold:
                    total += 1
                    correct += int(te_y[i] == 0)
            acc = correct / total if total else 0.5

            rows.append({
                'fold': fold_idx,
                'symbol': symbol,
                'n_signals': total,
                'direction_accuracy': acc,
                'auc': auc,
            })

    return pd.DataFrame(rows)


def _assemble(feature_set, base, v1, v2, rng=None, is_train=False):
    """Concatenate selected feature blocks for one asset's split."""
    parts = [base]
    if feature_set == 'base':
        pass
    elif feature_set == 'base_plus_v1':
        parts.append(v1)
    elif feature_set == 'base_plus_v2':
        parts.append(v2)
    elif feature_set == 'base_plus_v1_plus_v2':
        parts.append(v1)
        parts.append(v2)
    elif feature_set == 'base_plus_shuffled_v2':
        v2_use = v2
        if is_train and v2 is not None and len(v2) > 1:
            perm = rng.permutation(len(v2))
            v2_use = v2[perm]
        parts.append(v2_use)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return np.concatenate(parts, axis=1)


# ============================================================
# Orchestrator
# ============================================================

def run_v12(symbols=None, days=1095, horizon=72, prob_threshold=0.65,
              imager_resolution=10):
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'AVAX']

    print(f"\n{'#' * 70}")
    print(f"#  V12: TDA-Representation Ablation (v1 scalars vs v2 images)")
    print(f"#  Symbols: {symbols} | Days: {days} | Horizon: {horizon}h")
    print(f"#  Imager: {imager_resolution}x{imager_resolution} per H0/H1, "
          f"FIT PER FOLD ON TRAIN ONLY")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print("[1/3] Preparing asset data (diagrams cached, no fit)...")
    asset_data = {}
    for symbol in symbols:
        data = prepare_asset(symbol, days=days, window_size=20, verbose=True)
        if data is not None:
            asset_data[symbol] = data
        time.sleep(1)
    print(f"  {len(asset_data)} assets ready")

    # Apply target horizon to each asset.
    for symbol, data in asset_data.items():
        df_t = add_targets(data['df'], horizon=horizon)
        keep_idx = df_t.index.values
        # Reindex h0/h1 diagrams to align with target-trimmed df.
        # add_targets only drops rows from the tail (where target is NaN),
        # so we keep diagrams [:len(df_t)] from the head — same alignment.
        n_keep = len(df_t)
        data['df'] = df_t
        data['h0_diagrams'] = data['h0_diagrams'][:n_keep]
        data['h1_diagrams'] = data['h1_diagrams'][:n_keep]

    n_total = sum(len(d['df']) for d in asset_data.values())
    print(f"  Pool after target alignment: {n_total:,} samples")

    print(f"\n[2/3] Five-way ablation × {len(MODELS)} models, "
          f"leak-safe per-fold imager fit")
    rows = []
    for feature_set in FEATURE_SETS:
        for model_type in MODELS:
            clf_check = make_clf(model_type)
            if clf_check is None:
                print(f"  [{feature_set} | {model_type}] dep missing, skipping")
                continue
            print(f"  [{feature_set:>22} | {model_type:>8}] running...")
            df_eval = evaluate_kfold_leaksafe(
                asset_data, feature_set, model_type,
                prob_threshold=prob_threshold,
                imager_resolution=imager_resolution,
                verbose=False,
            )
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
                'feature_set': feature_set,
                'model': model_type,
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
    write_v12_report(ab_df, symbols, days, horizon, imager_resolution)
    plot_v12_figure(ab_df)

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"#  V12 Complete in {elapsed/60:.1f} min")
    print(f"#  See: results/V12_TDA_REP.md")
    print(f"{'#' * 70}\n")


def write_v12_report(ab_df, symbols, days, horizon, imager_resolution):
    md = []
    md.append("# V12: TDA-Representation Ablation\n")
    md.append("**Question:** Is v9's ablation-negative result (Base+TDA "
              "underperforms Base alone by 5.6pp) caused by the v1 "
              "representation (8 hand-engineered scalar stats per dimension) "
              "being too lossy?\n")
    md.append("**Test:** Five-way ablation under both a linear (logistic, L2) "
              "and a non-linear (xgboost) classifier, on identical point clouds, "
              "with persistence images replacing the scalar TDA features. "
              "Persistence imagers are fit *per fold* on training diagrams only "
              "— never on the full pool — so no future TDA-feature distribution "
              "information leaks into the representation.\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    md.append(f"**Symbols:** {', '.join(symbols)}")
    md.append(f"**Sample:** {days} days hourly | Horizon: {horizon}h | "
              f"Imager: {imager_resolution}×{imager_resolution} per H₀/H₁\n")

    md.append("## 1. Ablation Results (5-fold time-series CV)\n")
    md.append("| Feature Set | Model | n Signals | Direction Acc | Wilson 95% | AUC |")
    md.append("|-------------|-------|----------:|--------------:|-----------|----:|")
    for _, r in ab_df.iterrows():
        md.append(f"| {r['feature_set']} | {r['model']} | "
                  f"{int(r['n_signals'])} | {r['direction_accuracy']:.2%} | "
                  f"[{r['wilson_lo']:.2%}, {r['wilson_hi']:.2%}] | "
                  f"{r['mean_auc']:.3f} |")
    md.append("")

    md.append("## 2. Per-Model Diagnostic\n")
    md.append("Δ measures the absolute change in pooled direction accuracy "
              "vs the model's `base` row.\n")
    by_model = ab_df.set_index(['model', 'feature_set'])['direction_accuracy']
    for model in MODELS:
        if (model, 'base') not in by_model.index:
            continue
        base_acc = by_model.loc[(model, 'base')]
        v1_delta = by_model.get((model, 'base_plus_v1'), np.nan) - base_acc
        v2_delta = by_model.get((model, 'base_plus_v2'), np.nan) - base_acc
        v1v2_delta = by_model.get((model, 'base_plus_v1_plus_v2'), np.nan) - base_acc
        sh_delta = by_model.get((model, 'base_plus_shuffled_v2'), np.nan) - base_acc

        md.append(f"### {model}\n")
        md.append(f"| Component | Accuracy | Δ vs base |")
        md.append(f"|-----------|---------:|----------:|")
        md.append(f"| base                  | {base_acc:.2%} |  — |")
        md.append(f"| base + tda_v1         | "
                  f"{by_model.get((model, 'base_plus_v1'), np.nan):.2%} | "
                  f"{v1_delta:+.2%} |")
        md.append(f"| base + tda_v2         | "
                  f"{by_model.get((model, 'base_plus_v2'), np.nan):.2%} | "
                  f"{v2_delta:+.2%} |")
        md.append(f"| base + v1 + v2        | "
                  f"{by_model.get((model, 'base_plus_v1_plus_v2'), np.nan):.2%} | "
                  f"{v1v2_delta:+.2%} |")
        md.append(f"| base + shuffled_v2 (control) | "
                  f"{by_model.get((model, 'base_plus_shuffled_v2'), np.nan):.2%} | "
                  f"{sh_delta:+.2%} |\n")

        if not np.isnan(v2_delta) and v2_delta > 0:
            md.append("**Verdict:** Persistence images add positive marginal "
                      "accuracy on top of base features. The v9 ablation-negative "
                      "result was driven by the v1 representation, not by an "
                      "absence of TDA signal.")
        elif not np.isnan(v2_delta) and v2_delta <= 0:
            md.append("**Verdict:** The richer v2 representation does not add "
                      "positive marginal accuracy. The v9 ablation-negative "
                      "result reflects a deeper limitation than feature "
                      "lossiness; future work should explore window size, "
                      "filtration choice, or alternative TDA constructions.")
        if not np.isnan(sh_delta) and not np.isnan(v2_delta):
            sep = abs(v2_delta - sh_delta)
            md.append(f"\nUnshuffled-v2 vs shuffled-v2 separation: {sep:.2%}. "
                      f"A small separation suggests the classifier is exploiting "
                      f"the *distribution* of v2 features rather than their "
                      f"window-specific values.\n")

    md.append("## 3. Methodology Notes\n")
    md.append(f"- **Leak-safe imager fit.** A fresh `PersistenceImager` is fit on "
              f"the union of training-fold diagrams across all assets at every "
              f"fold, then used to transform that fold's training and test "
              f"diagrams. The full-pool fit (which would leak holdout "
              f"distribution into the feature representation) is never performed.\n")
    md.append(f"- **Fixed image grid.** The imager is configured with "
              f"`birth_range = pers_range = max(birth_max, pers_max)` of the "
              f"training diagrams, with `pixel_size` chosen to produce exactly "
              f"a {imager_resolution}×{imager_resolution} grid per homology "
              f"dimension. No silent padding or cropping.\n")
    md.append(f"- **Empty-diagram robustness.** If a fold's training diagrams "
              f"contain no finite-persistence content for a given homology "
              f"dimension, that dimension's image features are returned as "
              f"zeros instead of raising at fit time.\n")
    md.append(f"- **Regularisation.** Logistic uses `C=0.5` (stronger L2 than "
              f"v9's `C=1.0`) to control over-fit on the 200-dim v2 representation. "
              f"xgboost uses depth=4, n_estimators=200, subsample/colsample 0.9.\n")
    md.append(f"- **Negative control.** `base_plus_shuffled_v2` shuffles the v2 "
              f"feature rows within each train fold, breaking the v2-window "
              f"correspondence while preserving the v2 distribution. If this "
              f"control matches `base_plus_v2`, the classifier is responding to "
              f"the *distribution* of v2 rather than to per-window values.\n")

    md.append("## 4. Read against v9\n")
    md.append("v9 reported, on a 365-day pool: Base 65.93%, Base+v1_TDA 60.38% "
              "(Δ = −5.55pp). v12 results above are on a "
              f"{days}-day pool with identical point clouds for both "
              "representations — the only variable is the persistence-diagram → "
              "feature mapping. A non-negative v2 delta narrows the v9 gap and "
              "supports the diagnosis that v1's lossy scalar summaries, not TDA "
              "itself, were the issue.\n")

    with open('results/V12_TDA_REP.md', 'w') as f:
        f.write("\n".join(md))
    print(f"  Saved: results/V12_TDA_REP.md")


def plot_v12_figure(ab_df):
    if len(ab_df) == 0:
        return
    feature_sets = FEATURE_SETS
    models = sorted(ab_df['model'].unique())

    fig, ax = plt.subplots(figsize=(13, 6))
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
        # Build asymmetric error array; np.where for nan-safety.
        upper_err = np.where(np.isnan(ys), 0, his - ys)
        lower_err = np.where(np.isnan(ys), 0, ys - los)
        yerr = np.stack([lower_err, upper_err])
        ax.bar(x + offsets, np.where(np.isnan(ys), 0, ys), bar_w, yerr=yerr,
                capsize=3, label=model, edgecolor='black', linewidth=0.5)

    ax.axhline(50.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label='50% chance')
    pretty = {
        'base': 'base',
        'base_plus_v1': 'base + v1\n(scalars)',
        'base_plus_v2': 'base + v2\n(images)',
        'base_plus_v1_plus_v2': 'base + v1 + v2',
        'base_plus_shuffled_v2': 'base + shuffled v2\n(neg. control)',
    }
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(f, f) for f in feature_sets], fontsize=9)
    ax.set_ylabel('Direction accuracy (%)')
    ax.set_title('V12: TDA Representation Ablation (leak-safe per-fold imager fit)\n'
                 'v1 = 16 scalar stats  |  v2 = 10x10 persistence image per H0/H1')
    ax.legend(loc='lower right')
    valid_his = ab_df['wilson_hi'].dropna()
    upper = max(80.0, float(valid_his.max()) * 100 + 5.0) if len(valid_his) else 80.0
    ax.set_ylim(40, upper)
    plt.tight_layout()
    plt.savefig('results/figures/v12_ablation_compare.png', dpi=150)
    plt.savefig('results/figures/v12_ablation_compare.pdf')
    plt.close()
    print(f"  Saved: results/figures/v12_ablation_compare.{{png,pdf}}")


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    run_v12(days=days)
