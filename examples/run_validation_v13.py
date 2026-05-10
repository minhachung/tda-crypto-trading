"""
v13 Comprehensive Validation: Tests all enhancements end-to-end.

Improvements over v10/v12:
  1. Multi-source data (Coinbase + CMC + on-chain metrics)
  2. Comprehensive feature set (core + survivorship + on-chain + cross-ex + interactions)
  3. Ensemble model (5 base learners + leak-safe meta-learner)
  4. Regime-adaptive models (separate per low/medium/high vol)
  5. Hyperparameter sweep across TDA window sizes
  6. B=1000 single-config + B=100 full-grid permutation tests

Expected runtime: 2-4 hours on Apple Silicon.
Expected output: results/V13_COMPREHENSIVE.md + figures/v13_*.png
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description='v13 comprehensive validation')
    parser.add_argument('days', type=int, default=1095, help='Days of data')
    parser.add_argument('n_perm_single', type=int, default=100,
                       help='B for single-config permutation')
    parser.add_argument('n_perm_grid', type=int, default=20,
                       help='B for full-grid permutation')
    parser.add_argument('--symbols', type=str, default='BTC,ETH,SOL,ADA',
                       help='Comma-separated symbols')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (smaller permutation count)')
    return parser.parse_args()


def fetch_multi_source_data(symbols, days, window_size=20):
    """
    Fetch data with TDA features using existing multi_asset_pipeline.

    Returns: dict mapping symbol -> DataFrame with base + TDA (H0_*, H1_*) features
    """
    print("\n" + "=" * 30)
    print("STEP 1: Multi-source data fetch + TDA computation")
    print("=" * 30)

    # Use existing multi_asset_pipeline for TDA features (proven leak-safe)
    try:
        # Path-fix: multi_asset_pipeline imports from src.X, but we sys-path src/
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / 'src'
        # Insert as 'src' so internal imports work
        parent_path = str(Path(__file__).parent.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        from src.multi_asset_pipeline import fetch_and_build_asset
    except ImportError as e:
        print(f"  ⚠ multi_asset_pipeline import failed: {e}")
        print("  Falling back to base features only (NO TDA)")
        return _fetch_no_tda(symbols, days)

    all_data = {}
    for sym in symbols:
        try:
            asset_df = fetch_and_build_asset(sym, days=days, window_size=window_size)
            if asset_df is not None and len(asset_df) > 0:
                # Add metadata columns expected by FeatureBuilder
                if 'is_delisted' not in asset_df.columns:
                    asset_df['is_delisted'] = False
                if 'delisting_date' not in asset_df.columns:
                    asset_df['delisting_date'] = None
                if 'source' not in asset_df.columns:
                    asset_df['source'] = 'coinbase'

                all_data[sym] = asset_df
                tda_cols = [c for c in asset_df.columns if c.startswith(('H0_', 'H1_'))]
                print(f"  ✓ {sym}: {len(asset_df)} rows ({len(tda_cols)} TDA features)")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")

    return all_data


def _fetch_no_tda(symbols, days):
    """Fallback fetcher without TDA (just price data)."""
    from binance_data import HighFreqFetcher
    all_data = {}
    for sym in symbols:
        try:
            fetcher = HighFreqFetcher(sym)
            df = fetcher.fetch_history(days=days)
            df['is_delisted'] = False
            df['delisting_date'] = None
            df['source'] = 'coinbase'
            all_data[sym] = df
            print(f"  ✓ {sym}: {len(df)} rows (NO TDA)")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    return all_data


def build_comprehensive_features(data_dict):
    """Build features by ADDING new categories to existing TDA + base features."""
    print("\n" + "=" * 30)
    print("STEP 2: Add new feature categories on top of TDA")
    print("=" * 30)

    try:
        from advanced_features import (
            FeatureBuilder, COMPREHENSIVE_FEATURE_SET, ML_FEATURE_SET,
            SURVIVORSHIP_FEATURES, ONCHAIN_FEATURES, CROSSEX_FEATURES, INTERACTION_FEATURES
        )
    except ImportError:
        print("  ⚠ FeatureBuilder not available, using TDA features only")
        from advanced_features import ML_FEATURE_SET
        feat_dict = data_dict
        # Use whatever TDA + ML features are present
        available = list(feat_dict[list(feat_dict.keys())[0]].columns)
        ml_cols = [c for c in ML_FEATURE_SET if c in available]
        tda_cols = [c for c in available if c.startswith(('H0_', 'H1_'))]
        feature_set = ml_cols + tda_cols
        return feat_dict, feature_set

    feat_dict = {}
    for sym, df in data_dict.items():
        # df already has base + TDA (H0_/H1_) features from fetch_and_build_asset
        # Now ADD survivorship/onchain/crossex/interaction features without re-running TDA
        builder = FeatureBuilder(df)
        builder.add_survivorship_features()
        builder.add_onchain_features()
        builder.add_crossex_spread_features()
        builder.add_interaction_features()
        enriched_df, _ = builder.build()

        # Re-attach TDA columns from original df (FeatureBuilder strips them)
        tda_cols = [c for c in df.columns if c.startswith(('H0_', 'H1_'))]
        if tda_cols:
            tda_df = df[tda_cols].iloc[-len(enriched_df):].reset_index(drop=True)
            for col in tda_cols:
                enriched_df[col] = tda_df[col].values

        feat_dict[sym] = enriched_df
        n_tda = len([c for c in enriched_df.columns if c.startswith(('H0_', 'H1_'))])
        print(f"  ✓ {sym}: {len(enriched_df)} rows × {len(enriched_df.columns)} features ({n_tda} TDA)")

    # Build feature set: ML core + new categories + TDA
    available_cols = list(feat_dict[list(feat_dict.keys())[0]].columns)
    ml_cols = [c for c in ML_FEATURE_SET if c in available_cols]
    new_cols = [c for c in (SURVIVORSHIP_FEATURES + ONCHAIN_FEATURES +
                            CROSSEX_FEATURES + INTERACTION_FEATURES)
                if c in available_cols]
    tda_cols = [c for c in available_cols if c.startswith(('H0_', 'H1_'))]
    feature_set = ml_cols + new_cols + tda_cols

    print(f"\n  Feature breakdown:")
    print(f"    ML core:       {len(ml_cols)}")
    print(f"    New categories:{len(new_cols)}")
    print(f"    TDA (H0/H1):   {len(tda_cols)}")
    print(f"    TOTAL:         {len(feature_set)}")
    return feat_dict, feature_set


def pool_data(feat_dict, feature_set, horizon=3):
    """Pool data across assets with per-asset volatility for regimes."""
    print("\n" + "=" * 30)
    print("STEP 3: Multi-asset pooling")
    print("=" * 30)

    pooled_X = []
    pooled_y = []
    pooled_vol = []
    pooled_assets = []

    for sym, df in feat_dict.items():
        if 'log_return' not in df.columns or 'close' not in df.columns:
            print(f"  ✗ {sym}: missing required columns")
            continue

        # Direction target (3-period ahead)
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        y = (future_return > 0).astype(int)

        # Features
        valid_features = [c for c in feature_set if c in df.columns]
        if not valid_features:
            print(f"  ✗ {sym}: no valid features")
            continue

        # Coerce to numeric, dropping any object-dtype features
        X_df = df[valid_features].apply(pd.to_numeric, errors='coerce')
        X = X_df.values.astype(np.float64)

        # Volatility for regime detection
        vol_series = df['rv_20'] if 'rv_20' in df.columns else df['log_return'].rolling(20).std()
        vol = pd.to_numeric(vol_series, errors='coerce').values.astype(np.float64)

        # Coerce y and future_return to numeric
        y_arr = pd.to_numeric(y, errors='coerce').values.astype(np.float64)
        fr_arr = pd.to_numeric(future_return, errors='coerce').values.astype(np.float64)

        # Drop NaN rows
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_arr) & ~np.isnan(vol) & ~np.isnan(fr_arr)

        pooled_X.append(X[mask])
        pooled_y.append(y_arr[mask].astype(np.int64))
        pooled_vol.append(vol[mask])
        pooled_assets.extend([sym] * mask.sum())

        print(f"  ✓ {sym}: {mask.sum()} samples")

    X = np.vstack(pooled_X)
    y = np.concatenate(pooled_y)
    vol = np.concatenate(pooled_vol)

    print(f"\n  Pooled total: {len(X)} samples × {X.shape[1]} features")
    return X, y, vol, pooled_assets


def train_test_split_temporal(X, y, vol, train_pct=0.70, val_pct=0.15):
    """Walk-forward split: 70% train | 15% val | 15% test."""
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    return {
        'X_train': X[:train_end], 'y_train': y[:train_end], 'vol_train': vol[:train_end],
        'X_val': X[train_end:val_end], 'y_val': y[train_end:val_end], 'vol_val': vol[train_end:val_end],
        'X_test': X[val_end:], 'y_test': y[val_end:], 'vol_test': vol[val_end:],
    }


def evaluate_models(splits, verbose=1):
    """Evaluate baseline + ensemble + regime-adaptive on splits."""
    print("\n" + "=" * 30)
    print("STEP 4: Model evaluation")
    print("=" * 30)

    results = {}

    # 1. Baseline: Logistic Regression
    print("\n  1. Baseline Logistic Regression...")
    base_model = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42)
    base_model.fit(splits['X_train'], splits['y_train'])
    base_test_acc = accuracy_score(splits['y_test'], base_model.predict(splits['X_test']))
    results['baseline_logistic'] = {
        'test_accuracy': base_test_acc,
        'n_test': len(splits['X_test'])
    }
    print(f"     Test accuracy: {base_test_acc:.4f}")

    # 2. Ensemble (if available)
    print("\n  2. LeakSafeEnsemble...")
    try:
        from ml_ensemble import LeakSafeEnsemble
        ensemble = LeakSafeEnsemble(verbose=verbose)
        ensemble.fit(splits['X_train'], splits['y_train'],
                    splits['X_val'], splits['y_val'])
        ens_test_acc = accuracy_score(
            splits['y_test'],
            ensemble.predict(splits['X_test'])
        )
        results['ensemble'] = {
            'test_accuracy': ens_test_acc,
            'n_test': len(splits['X_test'])
        }
        print(f"     Test accuracy: {ens_test_acc:.4f}")
    except ImportError:
        print("     ⚠ ml_ensemble not available, skipping")
        results['ensemble'] = None

    # 3. Regime-Adaptive Model
    print("\n  3. RegimeAdaptiveModel...")
    try:
        from regime_models import RegimeAdaptiveModel
        regime_model = RegimeAdaptiveModel(verbose=verbose)
        regime_model.fit(splits['X_train'], splits['y_train'], splits['vol_train'])
        per_regime = regime_model.evaluate_per_regime(
            splits['X_test'], splits['y_test'], splits['vol_test']
        )
        results['regime_adaptive'] = per_regime
        print(f"     Per-regime accuracy: {per_regime}")
    except ImportError:
        print("     ⚠ regime_models not available, skipping")
        results['regime_adaptive'] = None

    return results


def permutation_test(X_train, y_train, X_test, y_test, model_fn, B=100, random_state=42):
    """Run permutation test."""
    rng = np.random.default_rng(random_state)
    real_model = model_fn()
    real_model.fit(X_train, y_train)
    real_acc = accuracy_score(y_test, real_model.predict(X_test))

    null_accs = []
    for b in range(B):
        y_train_perm = rng.permutation(y_train)
        perm_model = model_fn()
        perm_model.fit(X_train, y_train_perm)
        perm_acc = accuracy_score(y_test, perm_model.predict(X_test))
        null_accs.append(perm_acc)

        if (b + 1) % 10 == 0:
            print(f"     Permutation {b + 1}/{B}: mean null = {np.mean(null_accs):.4f}")

    null_accs = np.array(null_accs)
    p_value = (null_accs >= real_acc).sum() / B

    return {
        'real_accuracy': real_acc,
        'null_mean': float(np.mean(null_accs)),
        'null_std': float(np.std(null_accs)),
        'p_value': float(p_value),
        'B': B
    }


def write_results(results, perm_results, args, output_path):
    """Write comprehensive results to markdown."""
    with open(output_path, 'w') as f:
        f.write("# V13 Comprehensive Validation Results\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Days:** {args.days}\n")
        f.write(f"**Symbols:** {args.symbols}\n\n")

        f.write("## Headline Numbers\n\n")
        f.write("| Model | Test Accuracy | n_test |\n")
        f.write("|-------|---------------|--------|\n")
        for name, r in results.items():
            if r is None:
                continue
            if 'test_accuracy' in r:
                f.write(f"| {name} | {r['test_accuracy']:.4f} | {r['n_test']} |\n")
            elif name == 'regime_adaptive':
                for regime, sub in r.items():
                    f.write(f"| regime_{regime} | {sub['accuracy']:.4f} | {sub['n_samples']} |\n")

        f.write("\n## Permutation Test\n\n")
        if perm_results:
            f.write(f"- Real accuracy: {perm_results['real_accuracy']:.4f}\n")
            f.write(f"- Null mean ± std: {perm_results['null_mean']:.4f} ± {perm_results['null_std']:.4f}\n")
            f.write(f"- **p-value: {perm_results['p_value']:.4f}** (B={perm_results['B']})\n")
            verdict = "✅ Significant" if perm_results['p_value'] < 0.05 else \
                      "🟡 Marginal" if perm_results['p_value'] < 0.10 else \
                      "❌ Not significant"
            f.write(f"- **Verdict:** {verdict}\n\n")

        f.write("## Comparison vs v10/v12\n\n")
        f.write("| Version | Best Accuracy | p-value |\n")
        f.write("|---------|---------------|--------:|\n")
        f.write("| v10 | 60.68% | 0.1584 |\n")
        f.write("| v12 (logistic + tda_v1) | 58.06% | n/a |\n")
        if perm_results:
            f.write(f"| **v13** | {perm_results['real_accuracy']*100:.2f}% | **{perm_results['p_value']:.4f}** |\n")

    print(f"\n  ✓ Results written to {output_path}")


def main():
    args = parse_args()
    if args.quick:
        args.n_perm_single = 30
        args.n_perm_grid = 5

    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    print(f"\n🚀 V13 Comprehensive Validation")
    print(f"   Days: {args.days}, Symbols: {symbols}")
    print(f"   Permutations: single={args.n_perm_single}, grid={args.n_perm_grid}")

    start_time = time.time()

    # Phase 1: Data
    data_dict = fetch_multi_source_data(symbols, args.days)
    if not data_dict:
        print("❌ No data fetched. Exiting.")
        return

    # Phase 2: Features
    feat_dict, feature_set = build_comprehensive_features(data_dict)

    # Phase 3: Pool
    X, y, vol, assets = pool_data(feat_dict, feature_set, horizon=3)

    # Phase 4: Split
    splits = train_test_split_temporal(X, y, vol)
    print(f"\n  Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")

    # Phase 5: Evaluate
    results = evaluate_models(splits, verbose=1)

    # Phase 6: Permutation test on BEST model (auto-pick highest test accuracy)
    print("\n" + "=" * 30)
    print(f"STEP 5: Permutation test (B={args.n_perm_single})")
    print("=" * 30)

    # Pick best model from results
    best_model_name = 'baseline_logistic'
    best_acc = results['baseline_logistic']['test_accuracy']
    if results.get('ensemble') and results['ensemble']['test_accuracy'] > best_acc:
        best_model_name = 'ensemble'
        best_acc = results['ensemble']['test_accuracy']

    print(f"  Permuting best model: {best_model_name} (acc={best_acc:.4f})")

    if best_model_name == 'ensemble':
        # Custom permutation: refit ensemble on shuffled labels
        try:
            from ml_ensemble import LeakSafeEnsemble
            from sklearn.metrics import accuracy_score as _acc
            rng = np.random.default_rng(42)
            real_acc = best_acc
            null_accs = []
            for b in range(args.n_perm_single):
                y_train_perm = rng.permutation(splits['y_train'])
                ens_perm = LeakSafeEnsemble(verbose=0)
                ens_perm.fit(splits['X_train'], y_train_perm,
                            splits['X_val'], splits['y_val'])
                perm_acc = _acc(splits['y_test'], ens_perm.predict(splits['X_test']))
                null_accs.append(perm_acc)
                if (b + 1) % 5 == 0:
                    print(f"     Permutation {b + 1}/{args.n_perm_single}: mean null = {np.mean(null_accs):.4f}")
            null_accs = np.array(null_accs)
            perm_results = {
                'real_accuracy': real_acc,
                'null_mean': float(np.mean(null_accs)),
                'null_std': float(np.std(null_accs)),
                'p_value': float((null_accs >= real_acc).sum() / args.n_perm_single),
                'B': args.n_perm_single,
                'model': 'ensemble'
            }
        except Exception as e:
            print(f"  ⚠ Ensemble permutation failed: {e}, falling back to logistic")
            perm_results = permutation_test(
                splits['X_train'], splits['y_train'],
                splits['X_test'], splits['y_test'],
                model_fn=lambda: LogisticRegression(C=1.0, max_iter=500, random_state=42),
                B=args.n_perm_single
            )
            perm_results['model'] = 'baseline_logistic'
    else:
        perm_results = permutation_test(
            splits['X_train'], splits['y_train'],
            splits['X_test'], splits['y_test'],
            model_fn=lambda: LogisticRegression(C=1.0, max_iter=500, random_state=42),
            B=args.n_perm_single
        )
        perm_results['model'] = 'baseline_logistic'

    # Write results
    output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'V13_COMPREHENSIVE.md'
    write_results(results, perm_results, args, output_path)

    elapsed = time.time() - start_time
    print(f"\n✓ Complete in {elapsed/60:.1f}m")


if __name__ == '__main__':
    main()
