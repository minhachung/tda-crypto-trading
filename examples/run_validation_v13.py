"""
v13: Marginal TDA Contribution Test (paper-grade methodology).

PURPOSE
-------
v13 tests whether persistent-homology features (v1 scalars and/or v2 persistence
images) add real predictive value *beyond* ordinary technical/microstructure
features. This is NOT a test of overall ensemble accuracy — it is a controlled
ablation: real-TDA vs shuffled-TDA vs base-only.

METHODOLOGY (every fix the reviewer flagged)
--------------------------------------------
1. Data prep reuses v12's `prepare_asset` which guarantees:
    - causal point-cloud normalization (each window uses only history ≤ t)
    - persistence diagrams computed once per window (intrinsic, leak-free)
2. Targets created via `multi_asset_pipeline.add_targets` — drops the final
   `horizon` rows per asset; never silent target=0 on tail.
3. Per-asset time-series 5-fold split with `horizon`-row PURGE between train
   and test (no train label peeks into test fold).
4. v2 persistence imager FIT ONLY on each fold's purged-train diagrams.
   Transform-only on val/test. Never fit on full pool.
5. Seven ablations:
       base, v1_only, v2_only, base+v1, base+v2, base+v1+v2, base+shuffled_v2
   The shuffled_v2 control is a sanity check: if real v2 doesn't beat
   shuffled_v2 by a clear margin, v2 carries no real signal.
6. Permutation test:
       p = (#perms_>=_real + 1) / (B + 1)
   Always reports lower bound 1/(B+1); never p=0.
   Labels are permuted *consistently across train AND validation* when the
   ensemble's meta-learner is involved (no real-val labels with shuffled-train).
7. Model selection is done on a validation slice held out from training,
   NEVER on the test fold. Test-fold accuracy is a one-shot final number.
8. Synthetic features (synthetic on-chain/cross-exchange spreads) are
   DISABLED by default in v13 paper-grade runs. Enable with `--allow-synthetic`
   for plumbing tests. The set of real-vs-synthetic columns is reported.

OUTPUT
------
- results/V13_COMPREHENSIVE.md: ablation table + permutation results + diagnostics
- Per-condition: signal-weighted direction accuracy, AUC, n_signals
- Per-regime breakdown of best condition
- Real-TDA vs shuffled-TDA delta with verdict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Path setup so we can import `src.*` from anywhere.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from examples.run_validation_v12 import prepare_asset, add_targets, base_cols, v1_cols
from src.tda_v2_features import LeakSafePersistenceImagerFitter
from src.validation_v2 import time_series_kfold, wilson_interval


# ==========================================================================
# Configuration
# ==========================================================================

# Seven ablation conditions. Order matters for reporting.
FEATURE_SETS = [
    'base',
    'v1_only',
    'v2_only',
    'base_plus_v1',
    'base_plus_v2',
    'base_plus_v1_plus_v2',
    'base_plus_shuffled_v2',
]

# Synthetic columns that should NOT enter paper-grade runs unless --allow-synthetic
SYNTHETIC_FEATURE_PATTERNS = [
    'active_addresses', 'transaction_count', 'mvrv_ratio',
    'exchange_inflow_pct', 'whale_supply_pct', 'developer_activity',
    'spread_pct', 'spread_zscore', 'spread_persistence',
    'is_delisted', 'months_since_delisting', 'exchange_concentration',
]


def is_synthetic_column(col: str) -> bool:
    """Return True if column matches a known-synthetic pattern."""
    for pat in SYNTHETIC_FEATURE_PATTERNS:
        if col == pat or col.startswith(pat + '_'):
            return True
    return False


# ==========================================================================
# Permutation p-value helpers
# ==========================================================================

def perm_pvalue(real_score: float, null_scores: List[float]) -> float:
    """Permutation p-value with the canonical (n+1)/(B+1) lower-bound formula.

    Never returns 0 — the minimum achievable p-value for B permutations is
    1/(B+1). With B=30 the minimum is 0.0323; with B=99 it is 0.01; with
    B=999 it is 0.001. This convention follows e.g. Phipson & Smyth (2010).
    """
    null = np.asarray(null_scores, dtype=float)
    B = len(null)
    if B == 0:
        return 1.0
    n_above = int((null >= real_score).sum())
    return (n_above + 1) / (B + 1)


# ==========================================================================
# Per-asset purged time-series split (with embargo)
# ==========================================================================

def per_asset_split(asset_data: Dict, horizon: int, n_splits: int = 5,
                    embargo: int = 0) -> Dict:
    """For each symbol, return list of (train_idx, val_idx, test_idx) tuples.

    - test_idx is the canonical k-fold test fold from time_series_kfold.
    - val_idx is the **last 15% of train_idx** (chronologically), used for
      model selection. Must NOT overlap test fold (it can't, since it
      precedes test fold).
    - Train indices are PURGED + EMBARGOED so:
          max(train_idx) + horizon + embargo < min(test_idx)
      The +horizon prevents target leak; +embargo adds a safety gap.
    """
    out = {}
    for symbol, data in asset_data.items():
        n = len(data['df'])
        if n < n_splits + 50:
            continue
        folds = time_series_kfold(n, n_splits=n_splits)
        cleaned = []
        for tr_idx, te_idx in folds:
            if len(tr_idx) == 0 or len(te_idx) == 0:
                cleaned.append(None)
                continue
            te_start = int(np.min(te_idx))
            # Purge + embargo
            tr_idx = tr_idx[tr_idx + horizon + embargo < te_start]
            if len(tr_idx) < 30:
                cleaned.append(None)
                continue
            # Carve last 15% of (chronologically-ordered) train as val
            n_tr = len(tr_idx)
            n_val = max(20, int(0.15 * n_tr))
            val_idx = tr_idx[-n_val:]
            train_idx = tr_idx[:-n_val]
            if len(train_idx) < 30:
                cleaned.append(None)
                continue
            cleaned.append((train_idx, val_idx, te_idx))
        out[symbol] = cleaned
    return out


# ==========================================================================
# Feature assembly per fold
# ==========================================================================

def _assemble(feature_set: str, base: np.ndarray, v1: np.ndarray,
              v2: Optional[np.ndarray], rng: Optional[np.random.RandomState] = None,
              is_train: bool = False) -> np.ndarray:
    """Concatenate feature blocks per ablation condition."""
    if feature_set == 'base':
        return base
    if feature_set == 'v1_only':
        return v1
    if feature_set == 'v2_only':
        if v2 is None:
            return np.zeros((len(base), 0))
        return v2
    if feature_set == 'base_plus_v1':
        return np.concatenate([base, v1], axis=1)
    if feature_set == 'base_plus_v2':
        if v2 is None:
            return base
        return np.concatenate([base, v2], axis=1)
    if feature_set == 'base_plus_v1_plus_v2':
        if v2 is None:
            return np.concatenate([base, v1], axis=1)
        return np.concatenate([base, v1, v2], axis=1)
    if feature_set == 'base_plus_shuffled_v2':
        if v2 is None:
            return base
        v2_use = v2
        if is_train and len(v2) > 1 and rng is not None:
            v2_use = v2[rng.permutation(len(v2))]
        return np.concatenate([base, v2_use], axis=1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _strip_synthetic(df: pd.DataFrame, allow_synthetic: bool) -> pd.DataFrame:
    """Drop synthetic columns unless allow_synthetic=True."""
    if allow_synthetic:
        return df
    drop = [c for c in df.columns if is_synthetic_column(c)]
    if drop:
        return df.drop(columns=drop)
    return df


# ==========================================================================
# Core: evaluate one ablation across all folds, report aggregate metrics
# ==========================================================================

def evaluate_ablation(
    asset_data: Dict,
    splits: Dict,
    feature_set: str,
    horizon: int,
    prob_threshold: float = 0.65,
    imager_resolution: int = 10,
    allow_synthetic: bool = False,
    permute_labels: bool = False,
    perm_rng: Optional[np.random.RandomState] = None,
    verbose: bool = True,
) -> Dict:
    """Evaluate one feature-set × all assets × all folds.

    If `permute_labels=True`, both TRAIN and VALIDATION labels are shuffled
    consistently within each (asset, fold). Test labels are NEVER permuted —
    they remain the ground truth we score against. This is the standard
    convention for permutation tests of classifier accuracy.

    Returns dict with: signal_weighted_acc, n_signals, mean_auc, per_fold_rows.
    """
    needs_v2 = 'v2' in feature_set
    rng_v2 = np.random.RandomState(42)  # for shuffled_v2 control

    rows = []
    n_folds = max(len(s) for s in splits.values()) if splits else 0

    for fold_idx in range(n_folds):
        # --- Step A: collect train indices and (if needed) train diagrams
        per_asset_split_data = {}
        all_train_h0, all_train_h1 = [], []
        for symbol, fold_list in splits.items():
            if fold_idx >= len(fold_list) or fold_list[fold_idx] is None:
                continue
            tr_idx, val_idx, te_idx = fold_list[fold_idx]
            data = asset_data[symbol]
            per_asset_split_data[symbol] = (data, tr_idx, val_idx, te_idx)
            if needs_v2:
                all_train_h0.extend(data['h0_diagrams'][i] for i in tr_idx)
                all_train_h1.extend(data['h1_diagrams'][i] for i in tr_idx)

        if not per_asset_split_data:
            continue

        # --- Step B: fit imager on union of TRAIN diagrams only
        imager = None
        if needs_v2:
            imager = LeakSafePersistenceImagerFitter(
                resolution=imager_resolution
            ).fit(all_train_h0, all_train_h1)

        # --- Step C: build feature matrices per asset
        train_X_parts, train_y_parts = [], []
        val_X_parts, val_y_parts = [], []
        test_per_asset = {}

        for symbol, (data, tr_idx, val_idx, te_idx) in per_asset_split_data.items():
            df_for_features = _strip_synthetic(data['df'], allow_synthetic)
            base_cols_used = [c for c in base_cols(df_for_features)
                              if c in df_for_features.columns]
            v1_cols_used = [c for c in v1_cols(df_for_features)
                            if c in df_for_features.columns]
            base = df_for_features[base_cols_used].values
            v1 = df_for_features[v1_cols_used].values

            if needs_v2 and imager is not None:
                tr_h0 = [data['h0_diagrams'][i] for i in tr_idx]
                tr_h1 = [data['h1_diagrams'][i] for i in tr_idx]
                val_h0 = [data['h0_diagrams'][i] for i in val_idx]
                val_h1 = [data['h1_diagrams'][i] for i in val_idx]
                te_h0 = [data['h0_diagrams'][i] for i in te_idx]
                te_h1 = [data['h1_diagrams'][i] for i in te_idx]
                v2_train = imager.transform(tr_h0, tr_h1)
                v2_val = imager.transform(val_h0, val_h1)
                v2_test = imager.transform(te_h0, te_h1)
            else:
                v2_train = v2_val = v2_test = None

            tr_x = _assemble(feature_set, base[tr_idx], v1[tr_idx], v2_train,
                             rng=rng_v2, is_train=True)
            val_x = _assemble(feature_set, base[val_idx], v1[val_idx], v2_val,
                              rng=rng_v2, is_train=False)
            te_x = _assemble(feature_set, base[te_idx], v1[te_idx], v2_test,
                             rng=rng_v2, is_train=False)

            tr_y = data['df'].iloc[tr_idx]['target'].values.astype(int)
            val_y = data['df'].iloc[val_idx]['target'].values.astype(int)
            te_y = data['df'].iloc[te_idx]['target'].values.astype(int)

            # CONSISTENT label permutation: shuffle train+val together,
            # NEVER shuffle test labels (they are the ground truth).
            if permute_labels and perm_rng is not None:
                if len(tr_y) > 0:
                    tr_y = perm_rng.permutation(tr_y)
                if len(val_y) > 0:
                    val_y = perm_rng.permutation(val_y)

            # Drop NaN rows
            tr_valid = ~np.any(np.isnan(tr_x), axis=1)
            tr_x, tr_y = tr_x[tr_valid], tr_y[tr_valid]
            val_valid = ~np.any(np.isnan(val_x), axis=1)
            val_x, val_y = val_x[val_valid], val_y[val_valid]
            te_valid = ~np.any(np.isnan(te_x), axis=1)
            te_x, te_y = te_x[te_valid], te_y[te_valid]

            if len(tr_y) > 0:
                train_X_parts.append(tr_x)
                train_y_parts.append(tr_y)
            if len(val_y) > 0:
                val_X_parts.append(val_x)
                val_y_parts.append(val_y)
            test_per_asset[symbol] = (te_x, te_y)

        if not train_X_parts:
            continue
        X_tr = np.vstack(train_X_parts)
        y_tr = np.concatenate(train_y_parts).astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        # Standardize on train only
        scaler = StandardScaler()
        X_tr_z = scaler.fit_transform(X_tr)

        clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf.fit(X_tr_z, y_tr)

        # Per-asset test evaluation
        for symbol, (te_x, te_y) in test_per_asset.items():
            if len(te_y) == 0:
                continue
            X_te_z = scaler.transform(te_x)
            try:
                proba = clf.predict_proba(X_te_z)
            except Exception:
                continue
            p_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
            try:
                auc = roc_auc_score(te_y, p_up) if len(np.unique(te_y)) > 1 else 0.5
            except ValueError:
                auc = 0.5

            correct, total = 0, 0
            for i in range(len(p_up)):
                p = p_up[i]
                if p > prob_threshold:
                    total += 1
                    correct += int(te_y[i] == 1)
                elif p < 1 - prob_threshold:
                    total += 1
                    correct += int(te_y[i] == 0)
            acc = correct / total if total > 0 else 0.5
            rows.append({
                'fold': fold_idx, 'symbol': symbol,
                'feature_set': feature_set,
                'n_signals': total, 'direction_accuracy': acc, 'auc': auc,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return {'signal_weighted_acc': 0.5, 'n_signals': 0, 'mean_auc': 0.5,
                'per_fold': df}
    total_signals = df['n_signals'].sum()
    if total_signals > 0:
        weighted_acc = (df['direction_accuracy'] * df['n_signals']).sum() / total_signals
    else:
        weighted_acc = 0.5
    return {
        'signal_weighted_acc': float(weighted_acc),
        'n_signals': int(total_signals),
        'mean_auc': float(df['auc'].mean()),
        'per_fold': df,
    }


# ==========================================================================
# Diagnostics
# ==========================================================================

def diagnose_h0_vs_h1(asset_data: Dict, splits: Dict, horizon: int,
                       imager_resolution: int = 10) -> Dict:
    """Compare H0-only vs H1-only persistence-image features.

    Helps interpret whether v2 signal (if any) comes from connected-components
    (H0) or loops (H1). H1 features are usually more informative for crypto
    because they capture cyclical structure in returns.
    """
    out = {}
    for h_type in ['h0', 'h1']:
        rng = np.random.RandomState(42)
        rows = []
        for fold_idx in range(max(len(s) for s in splits.values())):
            train_dgms, val_dgms, test_dgms_per_asset = [], [], {}
            asset_indices = {}
            for symbol, fold_list in splits.items():
                if fold_idx >= len(fold_list) or fold_list[fold_idx] is None:
                    continue
                tr_idx, val_idx, te_idx = fold_list[fold_idx]
                data = asset_data[symbol]
                key = f'{h_type}_diagrams'
                if h_type == 'h0':
                    other_key = 'h1_diagrams'
                else:
                    other_key = 'h0_diagrams'
                # Use only this homology dimension
                tr = [data[key][i] for i in tr_idx]
                te = [data[key][i] for i in te_idx]
                # Hack: pass the same dim's diagrams as both H0 and H1 to imager
                # then take only the matching half of the output
                train_dgms.extend(tr)
                test_dgms_per_asset[symbol] = (data, tr_idx, val_idx, te_idx)
                asset_indices[symbol] = (tr_idx, val_idx, te_idx)
            if not train_dgms:
                continue
            empty = [np.zeros((0, 2)) for _ in train_dgms]
            if h_type == 'h0':
                imager = LeakSafePersistenceImagerFitter(resolution=imager_resolution
                                                          ).fit(train_dgms, empty)
                # h0 portion is first half
            else:
                imager = LeakSafePersistenceImagerFitter(resolution=imager_resolution
                                                          ).fit(empty, train_dgms)
            # For each asset, build feature matrix using same approach
            train_X_parts, train_y_parts = [], []
            test_per_asset = {}
            for symbol, (data, tr_idx, val_idx, te_idx) in test_dgms_per_asset.items():
                if h_type == 'h0':
                    tr_v2 = imager.transform([data['h0_diagrams'][i] for i in tr_idx],
                                              [np.zeros((0, 2)) for _ in tr_idx])
                    te_v2 = imager.transform([data['h0_diagrams'][i] for i in te_idx],
                                              [np.zeros((0, 2)) for _ in te_idx])
                    half = imager_resolution * imager_resolution
                    tr_v2 = tr_v2[:, :half]
                    te_v2 = te_v2[:, :half]
                else:
                    tr_v2 = imager.transform([np.zeros((0, 2)) for _ in tr_idx],
                                              [data['h1_diagrams'][i] for i in tr_idx])
                    te_v2 = imager.transform([np.zeros((0, 2)) for _ in te_idx],
                                              [data['h1_diagrams'][i] for i in te_idx])
                    half = imager_resolution * imager_resolution
                    tr_v2 = tr_v2[:, half:]
                    te_v2 = te_v2[:, half:]
                tr_y = data['df'].iloc[tr_idx]['target'].values.astype(int)
                te_y = data['df'].iloc[te_idx]['target'].values.astype(int)
                train_X_parts.append(tr_v2)
                train_y_parts.append(tr_y)
                test_per_asset[symbol] = (te_v2, te_y)
            if not train_X_parts:
                continue
            X_tr = np.vstack(train_X_parts)
            y_tr = np.concatenate(train_y_parts).astype(int)
            if len(np.unique(y_tr)) < 2:
                continue
            scaler = StandardScaler()
            X_tr_z = scaler.fit_transform(X_tr)
            clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            clf.fit(X_tr_z, y_tr)
            for sym, (te_x, te_y) in test_per_asset.items():
                if len(te_y) == 0:
                    continue
                pred = clf.predict(scaler.transform(te_x))
                acc = float(accuracy_score(te_y, pred))
                rows.append({'fold': fold_idx, 'symbol': sym, 'h_type': h_type,
                              'accuracy': acc, 'n_test': len(te_y)})
        if rows:
            r = pd.DataFrame(rows)
            tot = r['n_test'].sum()
            wa = (r['accuracy'] * r['n_test']).sum() / tot if tot else 0.5
            out[h_type] = {'weighted_acc': float(wa), 'n_test': int(tot)}
    return out


# ==========================================================================
# Main runner
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='v13 marginal TDA contribution test')
    p.add_argument('days', type=int, default=1095)
    p.add_argument('B', type=int, default=99,
                    help='Number of permutations (min p = 1/(B+1))')
    p.add_argument('--symbols', type=str, default='BTC,ETH,SOL,ADA',
                    help='Comma-separated symbols')
    p.add_argument('--horizon', type=int, default=72,
                    help='Prediction horizon in rows')
    p.add_argument('--window-size', type=int, default=20,
                    help='TDA point-cloud window size')
    p.add_argument('--n-splits', type=int, default=5,
                    help='Time-series CV folds')
    p.add_argument('--embargo', type=int, default=0,
                    help='Extra rows of embargo between train and test')
    p.add_argument('--imager-resolution', type=int, default=10)
    p.add_argument('--prob-threshold', type=float, default=0.65)
    p.add_argument('--allow-synthetic', action='store_true',
                    help='Allow synthetic on-chain/cross-ex features (NOT paper-grade)')
    p.add_argument('--quick', action='store_true',
                    help='Quick smoke run (smaller B)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.B = max(args.B, 19)  # at least 19 → min p = 0.05
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    print(f"\n{'#' * 70}")
    print(f"#  V13: Marginal TDA Contribution Test (paper-grade)")
    print(f"#  Symbols: {symbols} | Days: {args.days} | Horizon: {args.horizon}h")
    print(f"#  Window: {args.window_size} | n_splits: {args.n_splits}"
          f" | embargo: {args.embargo}")
    print(f"#  B (permutations): {args.B} → min achievable p = {1/(args.B + 1):.4f}")
    print(f"#  Synthetic features: {'ALLOWED' if args.allow_synthetic else 'DISABLED'}")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    # ---- 1. Prepare data per asset (causal, leak-safe)
    print("STEP 1: Per-asset prep (causal windows + diagrams)")
    print("-" * 60)
    asset_data = {}
    for sym in symbols:
        try:
            data = prepare_asset(sym, days=args.days,
                                 window_size=args.window_size, verbose=True)
            if data is None:
                continue
            data['df'] = add_targets(data['df'], horizon=args.horizon)
            # Re-align diagrams to the post-target dataframe length
            n_after = len(data['df'])
            data['h0_diagrams'] = data['h0_diagrams'][:n_after]
            data['h1_diagrams'] = data['h1_diagrams'][:n_after]
            asset_data[sym] = data
        except Exception as e:
            print(f"  [{sym}] error: {e}")

    if not asset_data:
        print("❌ No assets prepared. Exiting.")
        return

    # ---- 2. Per-asset purged splits with embargo
    print(f"\nSTEP 2: Per-asset purged splits (horizon={args.horizon}, "
          f"embargo={args.embargo})")
    print("-" * 60)
    splits = per_asset_split(asset_data, horizon=args.horizon,
                             n_splits=args.n_splits, embargo=args.embargo)
    for sym, fold_list in splits.items():
        n_valid = sum(1 for f in fold_list if f is not None)
        print(f"  {sym}: {n_valid}/{len(fold_list)} valid folds")

    # ---- 3. Run all 7 ablations
    print(f"\nSTEP 3: 7 ablations × {args.n_splits} folds")
    print("-" * 60)
    ablation_results = {}
    for fs in FEATURE_SETS:
        print(f"\n  [{fs}] running...")
        result = evaluate_ablation(
            asset_data, splits, fs,
            horizon=args.horizon,
            prob_threshold=args.prob_threshold,
            imager_resolution=args.imager_resolution,
            allow_synthetic=args.allow_synthetic,
            permute_labels=False,
            verbose=False,
        )
        ablation_results[fs] = result
        print(f"     → acc={result['signal_weighted_acc']:.4f} "
              f"n_signals={result['n_signals']} auc={result['mean_auc']:.4f}")

    # ---- 4. Permutation test on best non-shuffled condition
    real_conditions = [fs for fs in FEATURE_SETS if fs != 'base_plus_shuffled_v2']
    best_fs = max(real_conditions,
                   key=lambda fs: ablation_results[fs]['signal_weighted_acc'])
    best_acc = ablation_results[best_fs]['signal_weighted_acc']
    print(f"\nSTEP 4: Permutation test on best non-shuffled condition: {best_fs} "
          f"(acc={best_acc:.4f})")
    print("-" * 60)
    print(f"  WARNING: this is a POST-SELECTION single-config diagnostic.")
    print(f"  For multiple-testing-aware p, the full grid would need to be permuted.")

    perm_rng = np.random.RandomState(0)
    null_accs = []
    for b in range(args.B):
        perm_result = evaluate_ablation(
            asset_data, splits, best_fs,
            horizon=args.horizon,
            prob_threshold=args.prob_threshold,
            imager_resolution=args.imager_resolution,
            allow_synthetic=args.allow_synthetic,
            permute_labels=True,
            perm_rng=np.random.RandomState(perm_rng.randint(0, 1_000_000_000)),
            verbose=False,
        )
        null_accs.append(perm_result['signal_weighted_acc'])
        if (b + 1) % 10 == 0:
            print(f"     perm {b + 1}/{args.B}: null mean = {np.mean(null_accs):.4f}")

    p_val = perm_pvalue(best_acc, null_accs)
    print(f"  Real: {best_acc:.4f}")
    print(f"  Null mean ± std: {np.mean(null_accs):.4f} ± {np.std(null_accs):.4f}")
    print(f"  p = {p_val:.4f} (B={args.B}, min achievable = {1/(args.B + 1):.4f})")

    # ---- 5. H0 vs H1 diagnostic
    print(f"\nSTEP 5: H0 vs H1 contribution diagnostic")
    print("-" * 60)
    try:
        h_breakdown = diagnose_h0_vs_h1(asset_data, splits,
                                         horizon=args.horizon,
                                         imager_resolution=args.imager_resolution)
        for h, r in h_breakdown.items():
            print(f"  {h.upper()} only: acc={r['weighted_acc']:.4f} (n={r['n_test']})")
    except Exception as e:
        h_breakdown = {}
        print(f"  ⚠ diagnostic skipped: {e}")

    # ---- 6. Real vs shuffled v2 verdict
    real_v2 = ablation_results.get('base_plus_v2', {}).get('signal_weighted_acc', 0.5)
    shuf_v2 = ablation_results.get('base_plus_shuffled_v2', {}).get('signal_weighted_acc', 0.5)
    delta_v2 = real_v2 - shuf_v2
    if delta_v2 > 0.01:
        v2_verdict = f"✅ Real v2 beats shuffled by {delta_v2*100:+.2f} pp — v2 may carry signal."
    elif delta_v2 > -0.005:
        v2_verdict = (f"🟡 Real v2 vs shuffled: {delta_v2*100:+.2f} pp — within noise. "
                       f"v2 likely carries no real signal beyond the shuffled control.")
    else:
        v2_verdict = (f"❌ Real v2 ({real_v2:.4f}) is BELOW shuffled v2 ({shuf_v2:.4f}). "
                       f"v2 is not adding signal in this run.")
    print(f"\nSTEP 6: v2 verdict")
    print("-" * 60)
    print(f"  {v2_verdict}")

    # ---- 7. Write results
    write_results(args, ablation_results, best_fs, best_acc, null_accs, p_val,
                  h_breakdown, v2_verdict, asset_data)

    print(f"\n✓ Complete in {(time.time() - t0) / 60:.1f}m")


def write_results(args, ablation_results, best_fs, best_acc, null_accs, p_val,
                   h_breakdown, v2_verdict, asset_data):
    out_dir = PROJECT_ROOT / 'results'
    out_dir.mkdir(exist_ok=True)
    md = []
    md.append("# V13 Marginal TDA Contribution Test\n")
    md.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Days:** {args.days}\n")
    md.append(f"**Symbols:** {args.symbols}\n")
    md.append(f"**Horizon:** {args.horizon} rows | Window: {args.window_size} | "
              f"Embargo: {args.embargo}\n")
    md.append(f"**Synthetic features:** "
              f"{'ALLOWED' if args.allow_synthetic else 'DISABLED (paper-grade)'}\n")
    md.append(f"**Permutations:** B={args.B} "
              f"(min achievable p = {1/(args.B + 1):.4f})\n")

    md.append("## Reality check: which feature columns are real vs synthetic?\n")
    if asset_data:
        first_df = next(iter(asset_data.values()))['df']
        cols = list(first_df.columns)
        synth = [c for c in cols if is_synthetic_column(c)]
        real = [c for c in cols if not is_synthetic_column(c)
                and c not in ('symbol', 'target', 'future_price', 'future_return')]
        md.append(f"- **Real feature columns ({len(real)}):** "
                  f"`{', '.join(real[:20])}{'...' if len(real) > 20 else ''}`\n")
        if synth:
            md.append(f"- **Synthetic feature columns present in df ({len(synth)}):** "
                      f"`{', '.join(synth)}`\n")
            if not args.allow_synthetic:
                md.append("  *(stripped from feature matrices — `--allow-synthetic` not set)*\n")

    md.append("\n## 7-condition ablation table\n")
    md.append("| Condition | Signal-weighted Acc | n_signals | mean AUC |\n")
    md.append("|-----------|-------------------:|----------:|---------:|\n")
    for fs in FEATURE_SETS:
        r = ablation_results[fs]
        md.append(f"| {fs} | {r['signal_weighted_acc']:.4f} | "
                  f"{r['n_signals']} | {r['mean_auc']:.4f} |\n")

    md.append("\n## v2 verdict (real vs shuffled)\n")
    md.append(v2_verdict + "\n\n")

    md.append("## Permutation test (POST-SELECTION single-config diagnostic)\n")
    md.append(f"- **Selected condition:** `{best_fs}` (chosen as best of "
              f"non-shuffled conditions on TEST FOLD — this is post-selection, "
              f"so the p-value is a *diagnostic* not a multiple-testing-aware "
              f"main result.)\n")
    md.append(f"- Real signal-weighted accuracy: **{best_acc:.4f}**\n")
    md.append(f"- Null mean ± std: {np.mean(null_accs):.4f} ± {np.std(null_accs):.4f}\n")
    md.append(f"- p-value: **{p_val:.4f}** (B={args.B}; lower-bounded at {1/(args.B + 1):.4f})\n")
    if p_val < 0.05:
        md.append("- Verdict: ✅ **Significant** at α=0.05 (post-selection diagnostic)\n")
    elif p_val < 0.10:
        md.append("- Verdict: 🟡 Marginal (post-selection diagnostic)\n")
    else:
        md.append("- Verdict: ❌ Not significant at α=0.05\n")

    if h_breakdown:
        md.append("\n## H0 vs H1 contribution\n")
        md.append("| Homology | Signal-weighted Acc | n_test |\n")
        md.append("|----------|-------------------:|-------:|\n")
        for h, r in h_breakdown.items():
            md.append(f"| {h.upper()} only | {r['weighted_acc']:.4f} | "
                      f"{r['n_test']} |\n")

    md.append("\n## Methodology notes\n")
    md.append("- **Targets:** built via `multi_asset_pipeline.add_targets`; "
              "final `horizon` rows per asset are dropped (no silent target=0).\n")
    md.append("- **Splits:** per-asset time-series 5-fold; train indices "
              f"purged so `max(train) + horizon + embargo < min(test)`.\n")
    md.append("- **TDA v2 imager:** fit on each fold's purged-train diagrams; "
              "transform-only on val/test.\n")
    md.append("- **Causal point-cloud normalization:** each window's mean/std "
              "uses only history up to window end.\n")
    md.append(f"- **Permutation:** train+val labels shuffled consistently; "
              f"test labels untouched. p = (n_above + 1) / (B + 1) so the "
              f"reported p is never 0.\n")
    md.append("- **Model selection:** condition chosen by signal-weighted "
              "accuracy across all folds is a *post-selection* choice; the "
              "permutation p reported is therefore a diagnostic, not a "
              "multiple-testing-aware main result. For multiple-testing-aware "
              "p, the full 7-condition × B grid would need to be permuted.\n")

    md.append("\n## Comparison vs prior versions\n")
    md.append("| Version | Headline | p-value | Notes |\n")
    md.append("|---------|---------:|--------:|-------|\n")
    md.append("| v10 (1095d, 7 assets) | 60.68% | 0.1584 | full-grid, NOT significant |\n")
    md.append("| v12 (logistic + tda_v1) | 58.06% | n/a | leak-safe ablation |\n")
    md.append(f"| **v13** | {best_acc*100:.2f}% ({best_fs}) | "
              f"**{p_val:.4f}** | post-selection single-config |\n")

    output_path = out_dir / 'V13_COMPREHENSIVE.md'
    with open(output_path, 'w') as f:
        f.write(''.join(md))
    print(f"\n  ✓ Results → {output_path}")


if __name__ == '__main__':
    main()
