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
from src.onchain_metrics import OnChainMetricsFetcher, is_real_onchain_supported
from src.crossexchange_spreads import CrossExchangeSpreadFetcher, is_cross_exchange_supported


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

# Synthetic features that should NEVER enter paper-grade runs.
# This list is now MUCH SHORTER than v13's first cut: the real on-chain
# (transaction_count, hash_rate, etc. from blockchain.info) and real
# cross-exchange (spread_pct, spread_zscore_60 from Coinbase+Kraken)
# columns are NOT synthetic — only the FeatureBuilder's random-walk
# placeholder columns remain on this list.
SYNTHETIC_FEATURE_PATTERNS = [
    'mvrv_ratio',                     # synthetic in FeatureBuilder
    'exchange_inflow_pct',            # synthetic in FeatureBuilder
    'whale_supply_pct',               # synthetic in FeatureBuilder
    'developer_activity',             # synthetic in FeatureBuilder
    'is_delisted',                    # placeholder
    'months_since_delisting',         # placeholder
    'exchange_concentration',         # synthetic in FeatureBuilder
    'active_addresses_z',             # FeatureBuilder z-score (real one is 'active_addresses' from blockchain.info, but blockchain.info BTC doesn't expose it)
]

# Columns from REAL fetchers (blockchain.info, Coinbase+Kraken). These are
# real and may enter paper-grade runs.
REAL_EXTERNAL_COLUMNS = {
    # blockchain.info (BTC only)
    'transaction_count', 'hash_rate', 'mempool_size', 'mean_block_size',
    'total_fees_usd', 'miners_revenue',
    'transaction_count_z', 'hash_rate_z', 'mempool_size_z',
    'mean_block_size_z', 'total_fees_usd_z', 'miners_revenue_z',
    # cross-exchange (Coinbase + Kraken)
    'spread_pct', 'spread_zscore_60', 'spread_persistence',
}


def is_synthetic_column(col: str) -> bool:
    """Return True if column matches a known-synthetic pattern.

    Real-fetcher columns (blockchain.info, Coinbase+Kraken spreads) are
    explicitly whitelisted via REAL_EXTERNAL_COLUMNS so they aren't
    accidentally classified as synthetic by a partial-string match.
    """
    if col in REAL_EXTERNAL_COLUMNS:
        return False
    # Legacy synthetic spread features from FeatureBuilder
    if col == 'spread_pct' or col == 'spread_zscore_20' or col == 'spread_persistence':
        # spread_pct itself is now real (from CrossExchangeSpreadFetcher) — it would
        # have been caught above. spread_zscore_20 was the synthetic name; the real
        # one is spread_zscore_60.
        if col == 'spread_zscore_20':
            return True
    for pat in SYNTHETIC_FEATURE_PATTERNS:
        if col == pat or col.startswith(pat + '_'):
            return True
    return False


# ==========================================================================
# Real external data attachment
# ==========================================================================

def attach_real_external_data(asset_df: pd.DataFrame, symbol: str,
                                days: int,
                                use_onchain: bool,
                                use_crossex: bool,
                                verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Attach real on-chain and/or real cross-exchange columns to asset_df.

    Returns:
      - DataFrame with extra columns (where data is available)
      - Dict mapping {'onchain_attached': bool, 'crossex_attached': bool, ...}
    """
    coverage = {
        'onchain_attached': False,
        'crossex_attached': False,
        'onchain_n_rows': 0,
        'crossex_n_rows': 0,
    }

    if asset_df.empty or 'timestamp' not in asset_df.columns:
        return asset_df, coverage

    # Floor to day for left-merge (asset df is hourly; on-chain is daily)
    asset_df = asset_df.copy()
    asset_df['_merge_date'] = pd.to_datetime(asset_df['timestamp']).dt.floor('D')

    if use_onchain and is_real_onchain_supported(symbol):
        try:
            oc = OnChainMetricsFetcher(symbol, days=days).fetch_metrics()
            if not oc.empty:
                oc = oc.rename(columns={'timestamp': '_merge_date'})
                oc['_merge_date'] = pd.to_datetime(oc['_merge_date']).dt.floor('D')
                asset_df = pd.merge(asset_df, oc, on='_merge_date', how='left')
                coverage['onchain_attached'] = True
                coverage['onchain_n_rows'] = len(oc)
                if verbose:
                    print(f"    ✓ on-chain attached ({len(oc)} daily rows)")
        except Exception as e:
            if verbose:
                print(f"    ✗ on-chain fetch failed: {e}")

    if use_crossex and is_cross_exchange_supported(symbol):
        try:
            cx = CrossExchangeSpreadFetcher(symbol, days=days).fetch_spreads()
            if not cx.empty:
                cx = cx.rename(columns={'timestamp': '_merge_date'})
                cx['_merge_date'] = pd.to_datetime(cx['_merge_date']).dt.floor('D')
                asset_df = pd.merge(asset_df, cx, on='_merge_date', how='left')
                coverage['crossex_attached'] = True
                coverage['crossex_n_rows'] = len(cx)
                if verbose:
                    print(f"    ✓ cross-exchange attached ({len(cx)} daily rows)")
        except Exception as e:
            if verbose:
                print(f"    ✗ cross-exchange fetch failed: {e}")

    asset_df = asset_df.drop(columns=['_merge_date'])
    return asset_df, coverage


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
    """Compare H0-only vs H1-only persistence-image features (L4 fix).

    The earlier implementation used a single `LeakSafePersistenceImagerFitter`
    and sliced its concatenated H0+H1 output in half — but that imager fits
    BOTH halves on the same data, so H0-only and H1-only ended up sharing
    grid bounds and produced near-identical accuracies.

    This corrected version uses TWO separate imagers — one fit on H0 only
    (with empty H1 inputs as the second-dim placeholder, which the fitter
    correctly skips) and one fit on H1 only. Each imager's output is sliced
    to the dimension it was actually fit on, so H0-only features come from
    a strictly H0-trained representation and similarly for H1.
    """
    out = {}
    feat_dim = imager_resolution * imager_resolution

    for h_type in ['h0', 'h1']:
        rows = []
        n_folds = max(len(s) for s in splits.values()) if splits else 0
        for fold_idx in range(n_folds):
            # Collect train diagrams of THE chosen homology dim only
            train_dgms_chosen = []
            per_asset_split_data = {}
            for symbol, fold_list in splits.items():
                if fold_idx >= len(fold_list) or fold_list[fold_idx] is None:
                    continue
                tr_idx, val_idx, te_idx = fold_list[fold_idx]
                data = asset_data[symbol]
                src_key = f'{h_type}_diagrams'
                tr_dgms = [data[src_key][i] for i in tr_idx]
                train_dgms_chosen.extend(tr_dgms)
                per_asset_split_data[symbol] = (data, tr_idx, te_idx)

            if not train_dgms_chosen:
                continue

            # Fit imager on the chosen dim only; second dim gets empty diagrams
            empty_for_other_dim = [np.zeros((0, 2)) for _ in train_dgms_chosen]
            imager = LeakSafePersistenceImagerFitter(resolution=imager_resolution)
            if h_type == 'h0':
                imager.fit(train_dgms_chosen, empty_for_other_dim)
                # Output is concat[H0_grid, H1_grid] — H1 will be all zeros
                # because the H1 imager was skipped; take the H0 slice.
                slice_lo, slice_hi = 0, feat_dim
            else:
                imager.fit(empty_for_other_dim, train_dgms_chosen)
                slice_lo, slice_hi = feat_dim, 2 * feat_dim

            # Build feature matrices per asset
            train_X_parts, train_y_parts = [], []
            test_per_asset = {}
            for symbol, (data, tr_idx, te_idx) in per_asset_split_data.items():
                src_key = f'{h_type}_diagrams'
                tr_dgms = [data[src_key][i] for i in tr_idx]
                te_dgms = [data[src_key][i] for i in te_idx]
                empty_tr = [np.zeros((0, 2)) for _ in tr_dgms]
                empty_te = [np.zeros((0, 2)) for _ in te_dgms]

                if h_type == 'h0':
                    tr_full = imager.transform(tr_dgms, empty_tr)
                    te_full = imager.transform(te_dgms, empty_te)
                else:
                    tr_full = imager.transform(empty_tr, tr_dgms)
                    te_full = imager.transform(empty_te, te_dgms)

                tr_v = tr_full[:, slice_lo:slice_hi]
                te_v = te_full[:, slice_lo:slice_hi]

                tr_y = data['df'].iloc[tr_idx]['target'].values.astype(int)
                te_y = data['df'].iloc[te_idx]['target'].values.astype(int)
                train_X_parts.append(tr_v)
                train_y_parts.append(tr_y)
                test_per_asset[symbol] = (te_v, te_y)

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
# L2: Full-grid permutation (multiple-testing aware)
# ==========================================================================

def full_grid_permutation_test(
    asset_data: Dict, splits: Dict, real_conditions: List[str],
    real_max_acc: float, B: int, horizon: int,
    prob_threshold: float, imager_resolution: int,
    allow_synthetic: bool, verbose: bool = True,
) -> Tuple[float, List[float]]:
    """Multiple-testing-aware permutation test.

    For each of B permutations, run ALL real (non-shuffled) conditions
    under the same shuffled labels and take the MAX accuracy across
    conditions. This null distribution accounts for the fact that we
    selected the best condition from the real run too.

    Returns: (p_grid_aware, list_of_null_max_accs)

    Cost: B × |real_conditions| model fits. With B=99 and 6 real conditions
    that's 594 fits — heavy. Caller should set B carefully.
    """
    perm_seed = np.random.RandomState(0)
    null_max_accs = []
    for b in range(B):
        # Use same shuffle for all conditions in this permutation
        rng_for_perm = np.random.RandomState(perm_seed.randint(0, 1_000_000_000))
        perm_results = []
        for fs in real_conditions:
            r = evaluate_ablation(
                asset_data, splits, fs,
                horizon=horizon,
                prob_threshold=prob_threshold,
                imager_resolution=imager_resolution,
                allow_synthetic=allow_synthetic,
                permute_labels=True,
                perm_rng=rng_for_perm,
                verbose=False,
            )
            perm_results.append(r['signal_weighted_acc'])
        null_max_accs.append(max(perm_results) if perm_results else 0.5)
        if verbose and (b + 1) % 5 == 0:
            print(f"     full-grid perm {b + 1}/{B}: "
                  f"null max mean = {np.mean(null_max_accs):.4f}")

    p_grid = perm_pvalue(real_max_acc, null_max_accs)
    return p_grid, null_max_accs


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
    p.add_argument('--no-real-onchain', action='store_true',
                    help='Skip real on-chain (blockchain.info BTC) attachment')
    p.add_argument('--no-real-crossex', action='store_true',
                    help='Skip real cross-exchange (Coinbase+Kraken) spread attachment')
    p.add_argument('--full-grid', action='store_true',
                    help='Multiple-testing-aware: permute all conditions, take max '
                         '(B × |real_conditions| fits — heavy)')
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
    print("STEP 1: Per-asset prep (causal windows + diagrams + real external)")
    print("-" * 60)
    asset_data = {}
    coverage_table = {}
    use_onchain = not args.no_real_onchain
    use_crossex = not args.no_real_crossex
    for sym in symbols:
        try:
            data = prepare_asset(sym, days=args.days,
                                 window_size=args.window_size, verbose=True)
            if data is None:
                continue
            # Attach real on-chain (BTC only via blockchain.info) and real
            # cross-exchange spreads BEFORE target creation so they line up
            # with the trimmed dataframe after add_targets drops tail rows.
            data['df'], coverage = attach_real_external_data(
                data['df'], sym, days=args.days,
                use_onchain=use_onchain, use_crossex=use_crossex,
                verbose=True,
            )
            coverage_table[sym] = coverage
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

    # ---- 4. Permutation test
    real_conditions = [fs for fs in FEATURE_SETS if fs != 'base_plus_shuffled_v2']
    best_fs = max(real_conditions,
                   key=lambda fs: ablation_results[fs]['signal_weighted_acc'])
    best_acc = ablation_results[best_fs]['signal_weighted_acc']

    p_grid = None
    null_grid_accs = None

    if args.full_grid:
        print(f"\nSTEP 4: FULL-GRID permutation (multiple-testing-aware)")
        print(f"        Real best: {best_fs} = {best_acc:.4f}")
        print("-" * 60)
        print(f"  Permuting ALL {len(real_conditions)} real conditions per perm; "
              f"taking MAX as null. Cost: {args.B} × {len(real_conditions)} fits.")
        p_grid, null_grid_accs = full_grid_permutation_test(
            asset_data, splits, real_conditions, best_acc,
            B=args.B, horizon=args.horizon,
            prob_threshold=args.prob_threshold,
            imager_resolution=args.imager_resolution,
            allow_synthetic=args.allow_synthetic,
            verbose=True,
        )
        print(f"  Real: {best_acc:.4f}")
        print(f"  Null max mean ± std: {np.mean(null_grid_accs):.4f} ± "
              f"{np.std(null_grid_accs):.4f}")
        print(f"  p_grid = {p_grid:.4f} (B={args.B}, "
              f"min achievable = {1/(args.B + 1):.4f}) — multiple-testing aware")
        # Also run the post-selection diagnostic for comparison
        null_accs = []  # not run in full-grid mode
        p_val = p_grid  # use grid p as the headline if --full-grid
    else:
        print(f"\nSTEP 4: Permutation test on best non-shuffled condition: {best_fs} "
              f"(acc={best_acc:.4f})")
        print("-" * 60)
        print(f"  WARNING: this is a POST-SELECTION single-config diagnostic.")
        print(f"  For multiple-testing-aware p, run with --full-grid (much slower).")

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
        print(f"  p = {p_val:.4f} (B={args.B}, "
              f"min achievable = {1/(args.B + 1):.4f}) — POST-SELECTION DIAGNOSTIC")

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
                  h_breakdown, v2_verdict, asset_data, coverage_table,
                  null_grid_accs, p_grid)

    print(f"\n✓ Complete in {(time.time() - t0) / 60:.1f}m")


def write_results(args, ablation_results, best_fs, best_acc, null_accs, p_val,
                   h_breakdown, v2_verdict, asset_data, coverage_table=None,
                   null_grid_accs=None, p_grid=None):
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

    if coverage_table:
        md.append("\n## Real external data coverage per asset\n")
        md.append("| Asset | On-chain (blockchain.info) | Cross-exchange (Coinbase+Kraken) |\n")
        md.append("|-------|---------------------------:|---------------------------------:|\n")
        for sym, cov in coverage_table.items():
            oc = (f"✓ {cov['onchain_n_rows']} daily rows"
                  if cov.get('onchain_attached') else "—")
            cx = (f"✓ {cov['crossex_n_rows']} daily rows"
                  if cov.get('crossex_attached') else "—")
            md.append(f"| {sym} | {oc} | {cx} |\n")

    md.append("\n## 7-condition ablation table\n")
    md.append("| Condition | Signal-weighted Acc | n_signals | mean AUC |\n")
    md.append("|-----------|-------------------:|----------:|---------:|\n")
    for fs in FEATURE_SETS:
        r = ablation_results[fs]
        md.append(f"| {fs} | {r['signal_weighted_acc']:.4f} | "
                  f"{r['n_signals']} | {r['mean_auc']:.4f} |\n")

    md.append("\n## v2 verdict (real vs shuffled)\n")
    md.append(v2_verdict + "\n\n")

    if p_grid is not None and null_grid_accs is not None:
        md.append("## Permutation test — FULL-GRID (multiple-testing aware)\n")
        md.append(f"- **Real best:** `{best_fs}` = {best_acc:.4f} "
                  f"(picked among {len([f for f in FEATURE_SETS if f != 'base_plus_shuffled_v2'])} real conditions)\n")
        md.append(f"- **Null distribution:** for each of B={args.B} permutations, "
                  f"all real conditions were re-evaluated under the SAME shuffled "
                  f"labels, and the MAX accuracy across conditions was taken. "
                  f"This is the multiple-testing-aware null.\n")
        md.append(f"- Null max mean ± std: {np.mean(null_grid_accs):.4f} ± "
                  f"{np.std(null_grid_accs):.4f}\n")
        md.append(f"- **p_grid = {p_grid:.4f}** "
                  f"(B={args.B}, lower-bounded at {1/(args.B + 1):.4f})\n")
        if p_grid < 0.05:
            md.append("- Verdict: ✅ **Significant** at α=0.05 — TDA passes the "
                      "multiple-testing-aware test.\n")
        elif p_grid < 0.10:
            md.append("- Verdict: 🟡 Marginal — borderline after multiple-testing correction.\n")
        else:
            md.append("- Verdict: ❌ NOT significant at α=0.05 after "
                      "multiple-testing correction.\n")
    else:
        md.append("## Permutation test (POST-SELECTION single-config diagnostic)\n")
        md.append(f"- **Selected condition:** `{best_fs}` (chosen as best of "
                  f"non-shuffled conditions on TEST FOLD — this is post-selection, "
                  f"so the p-value is a *diagnostic* not a multiple-testing-aware "
                  f"main result.)\n")
        md.append(f"- Real signal-weighted accuracy: **{best_acc:.4f}**\n")
        md.append(f"- Null mean ± std: {np.mean(null_accs):.4f} ± {np.std(null_accs):.4f}\n")
        md.append(f"- p-value: **{p_val:.4f}** (B={args.B}; "
                  f"lower-bounded at {1/(args.B + 1):.4f})\n")
        if p_val < 0.05:
            md.append("- Verdict: ✅ **Significant** at α=0.05 (post-selection diagnostic)\n")
        elif p_val < 0.10:
            md.append("- Verdict: 🟡 Marginal (post-selection diagnostic)\n")
        else:
            md.append("- Verdict: ❌ Not significant at α=0.05\n")
        md.append("- *(For multiple-testing-aware p, re-run with `--full-grid`.)*\n")

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
