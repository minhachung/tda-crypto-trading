# Tutorial: Persistent Homology of Crypto Returns

This tutorial builds intuition for the central object of this study: a **persistence diagram** computed from a sliding window of crypto market features.

## What persistent homology computes

Imagine you have a cloud of points in some high-dimensional space. As you grow a "ball" of radius $r$ around each point, balls start to overlap and form connected components. As $r$ grows further, those components merge, and **loops** (1-dimensional holes) and **voids** (2-dimensional cavities) appear and eventually disappear.

Persistent homology tracks the *birth* and *death* of these topological features as the radius parameter sweeps from 0 to infinity. The result is a **persistence diagram**: a scatter plot where each point $(b, d)$ represents a feature that was born at radius $b$ and died at radius $d$. Long-lived features (large $d - b$) capture genuine geometric structure; short-lived features are noise.

## A toy example: points on a circle

Let's verify our intuition. Sample 30 points roughly on a unit circle (with small noise), then compute the persistence diagram.

```python
import numpy as np
from src.persistent_homology import PersistentHomologyAnalyzer

np.random.seed(0)
angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
points = np.column_stack([np.cos(angles), np.sin(angles)])
points += np.random.randn(*points.shape) * 0.02   # small noise

analyzer = PersistentHomologyAnalyzer(points, max_dim=1)
diagrams = analyzer.compute()

print(f"H0 (components): {len(diagrams['H0'])} features")
print(f"H1 (loops):       {len(diagrams['H1'])} features")
```

You should see:
- Many H0 features (one per pair of points that gets connected as $r$ grows)
- **Exactly one prominent H1 feature** — the loop of the circle itself

The persistence of that H1 feature is roughly the gap between when the circle "fills in" (as opposite-side points become connected through the interior) and when adjacent points first connect. For a unit circle with 30 evenly-spaced points, the loop persists from $r \approx 0.21$ (adjacent point spacing) until $r \approx 1.0$ (when the ball at any point reaches the opposite side).

## From toy to crypto

The crypto pipeline does the same thing, but the point cloud is different:

1. **Each "point" is one timestep** in the OHLCV history, represented as a 15-dimensional vector of returns, volatility estimators, volume z-scores, and trend indicators.

2. **Each "point cloud" is a sliding window** of 20 consecutive timesteps. So we have a cloud of 20 points in $\mathbb{R}^{15}$ for every hour of every asset.

3. **From each cloud's persistence diagram** we extract 8 scalar statistics per homology dimension (count, $L^1$-norm, $C^1$-norm = max persistence, mean, median, standard deviation, entropy, $L^2$ landscape norm). 16 TDA features total.

4. **These TDA features are concatenated with the original 15** and fed to a random forest classifier predicting whether price will be higher or lower in $h$ hours.

## Why does this work?

The intuition: market regimes leave **topological fingerprints** in the local manifold of price-volume space. When the market is in a trending regime, recent timesteps form a low-dimensional, near-linear cloud — H1 features are small. When the market is in a transitioning regime (e.g., a rotation, a reversal, or accumulation/distribution), the cloud has more curvature, more loops, more variability — H1 features grow.

Our hypothesis is that this geometric distinction is exploitable: a classifier should learn that *certain* H1 patterns predict *certain* future moves. The empirical evidence (in [Validation Methodology](validation.md)) supports this for short horizons (1h–7d) on certain assets (notably ADA, SOL, ETH).

## Try it yourself

```python
from src.binance_data import HighFreqFetcher
from src.advanced_features import build_advanced_features, get_tda_features
from src.persistent_homology import compute_features_for_windows

# 1. Fetch some data
fetcher = HighFreqFetcher('ADA', '1h')
df = fetcher.fetch_history(days=7)

# 2. Engineer features
df = build_advanced_features(df)

# 3. Build sliding-window point clouds
X_tda, _ = get_tda_features(df)
window = 20
clouds = [X_tda[i:i+window] for i in range(len(X_tda) - window)]

# 4. Compute persistence statistics for every window
features_df = compute_features_for_windows(clouds, verbose=False)

print(features_df[['H1_count', 'H1_l1_norm', 'H1_c1_norm', 'H1_entropy']].describe())
```

You should see that H1 statistics vary considerably across windows — that's the signal the classifier picks up.

## Going deeper

- See `src/persistent_homology.py` for the full feature extraction code
- See `src/advanced_features.py` for the 15 base features
- Run `python examples/run_validation_v6.py 30` for a quick end-to-end pipeline on 30 days of data
