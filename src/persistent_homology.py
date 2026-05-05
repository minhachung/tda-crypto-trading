"""
Persistent Homology: Compute topological features from point clouds.

Uses Ripser to compute persistence diagrams, then extracts trading-relevant
features (C1-norm, L1-norm, entropy, feature count, max persistence).
"""

import os
import numpy as np
import pandas as pd

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("[Warning] ripser not installed. Run: pip install ripser")


class PersistentHomologyAnalyzer:
    """Compute persistent homology and extract topological features."""

    def __init__(self, point_cloud, max_dim=1):
        """
        Args:
            point_cloud: (n_points, n_features) array
            max_dim: Maximum homology dimension (1 = loops, 2 = voids)
        """
        self.point_cloud = np.asarray(point_cloud, dtype=float)
        self.max_dim = max_dim
        self.diagrams = None
        self.features = None

    def compute(self):
        """Compute persistent homology via Vietoris-Rips."""
        if not RIPSER_AVAILABLE:
            raise ImportError("ripser is required. pip install ripser")

        result = ripser(self.point_cloud, maxdim=self.max_dim)
        self.diagrams = {
            f'H{i}': result['dgms'][i] for i in range(len(result['dgms']))
        }
        return self.diagrams

    def extract_features(self):
        """Extract trading-relevant features from persistence diagrams."""
        if self.diagrams is None:
            self.compute()

        features = {}

        for dim_key, diagram in self.diagrams.items():
            if len(diagram) == 0:
                features.update(self._empty_features(dim_key))
                continue

            finite = diagram[~np.isinf(diagram[:, 1])]

            if len(finite) == 0:
                features.update(self._empty_features(dim_key))
                continue

            persistence = finite[:, 1] - finite[:, 0]
            persistence = persistence[persistence > 0]

            if len(persistence) == 0:
                features.update(self._empty_features(dim_key))
                continue

            features[f'{dim_key}_count'] = len(persistence)
            features[f'{dim_key}_l1_norm'] = float(np.sum(persistence))
            features[f'{dim_key}_c1_norm'] = float(np.max(persistence))
            features[f'{dim_key}_mean_persistence'] = float(np.mean(persistence))
            features[f'{dim_key}_median_persistence'] = float(np.median(persistence))
            features[f'{dim_key}_std_persistence'] = float(np.std(persistence))

            p_norm = persistence / np.sum(persistence)
            entropy = -np.sum(p_norm * np.log(p_norm + 1e-12))
            features[f'{dim_key}_entropy'] = float(entropy)

            features[f'{dim_key}_landscape_l2'] = float(np.sqrt(np.sum(persistence ** 2)))

        self.features = features
        return features

    @staticmethod
    def _empty_features(dim_key):
        """Default zero values when no features detected."""
        return {
            f'{dim_key}_count': 0,
            f'{dim_key}_l1_norm': 0.0,
            f'{dim_key}_c1_norm': 0.0,
            f'{dim_key}_mean_persistence': 0.0,
            f'{dim_key}_median_persistence': 0.0,
            f'{dim_key}_std_persistence': 0.0,
            f'{dim_key}_entropy': 0.0,
            f'{dim_key}_landscape_l2': 0.0,
        }


def compute_features_for_windows(point_clouds, end_indices=None, max_dim=1, verbose=True):
    """
    Compute TDA features for a sequence of point clouds.

    Args:
        point_clouds: List or array of shape (n_windows, window_size, n_features)
        end_indices: Optional list of original-data indices for each window
        max_dim: Maximum homology dimension
        verbose: Print progress

    Returns:
        DataFrame with TDA features per window
    """
    all_features = []
    n = len(point_clouds)

    for i, pc in enumerate(point_clouds):
        analyzer = PersistentHomologyAnalyzer(pc, max_dim=max_dim)
        try:
            features = analyzer.extract_features()
        except Exception as e:
            if verbose:
                print(f"  [Window {i}] Error: {e} -- using zeros")
            features = {**analyzer._empty_features('H0'), **analyzer._empty_features('H1')}

        features['window_idx'] = i
        if end_indices is not None:
            features['end_idx'] = end_indices[i]
        all_features.append(features)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Computed {i + 1}/{n} persistence diagrams")

    df = pd.DataFrame(all_features)
    return df


def save_diagrams(diagrams, save_path):
    """Persist diagrams to disk."""
    np.savez(save_path, **{k: v for k, v in diagrams.items()})


def load_diagrams(save_path):
    """Load diagrams from disk."""
    data = np.load(save_path)
    return {k: data[k] for k in data.files}


def run_persistence_pipeline(point_clouds_path=None, end_indices_path=None,
                              symbol='BTC', max_dim=1, save_dir='data'):
    """End-to-end persistence computation."""
    os.makedirs(f'{save_dir}/persistence', exist_ok=True)

    if point_clouds_path is None:
        point_clouds_path = f'{save_dir}/processed/{symbol}_point_clouds.npy'
    if end_indices_path is None:
        end_indices_path = f'{save_dir}/processed/{symbol}_window_indices.npy'

    print(f"[Persistence] Loading point clouds from {point_clouds_path}")
    point_clouds = np.load(point_clouds_path)
    end_indices = np.load(end_indices_path) if os.path.exists(end_indices_path) else None
    print(f"  Loaded {len(point_clouds)} point clouds, shape: {point_clouds.shape}")

    print(f"[Persistence] Computing features for all windows (max_dim={max_dim})")
    features_df = compute_features_for_windows(
        point_clouds, end_indices=end_indices, max_dim=max_dim
    )
    print(f"  Computed features: {features_df.shape}")

    output_path = f'{save_dir}/persistence/{symbol}_tda_features.csv'
    features_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return features_df


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC'
    features_df = run_persistence_pipeline(symbol=symbol)
    print(f"\n[Done] TDA features shape: {features_df.shape}")
    print(f"\nSample features:\n{features_df.head()}")
    print(f"\nSummary:\n{features_df.describe()}")
