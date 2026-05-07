"""
TDA v2 Features: Persistence Images.

Replaces the v1 representation (8 hand-engineered scalar statistics per
homology dimension = 16 total features) with **persistence images** — a
fixed-dimension vectorisation of the persistence diagram itself.

Why persistence images:
  v1 summary statistics (count, L1-norm, max persistence, entropy, ...)
  collapse the entire persistence diagram into 8 scalars per dimension,
  discarding which (birth, persistence) regions the diagram occupies.
  v9 ablation showed Base+v1_TDA underperforms Base alone by 5.6pp —
  consistent with v1 features being too lossy to add value over a
  competitive base.

  Persistence images (Adams et al., JMLR 2017) place a Gaussian kernel
  on every diagram point, weighted by persistence, then sample the
  resulting density on a fixed grid. The result is a flattened vector
  whose dimensions correspond to specific (birth, persistence) regions
  — preserving the diagram's full geometric structure in a form
  classifiers can exploit.

Pipeline:
  point clouds ──ripser──▶ persistence diagrams (H0, H1)
                                      │
                                      ▼ (per dim)
                          PersistenceImager.fit(all_diagrams)
                                      │  (one fit per session)
                                      ▼
                          PersistenceImager.transform(diagrams)
                                      │
                                      ▼
                  flat 200-dim feature vector per window
                  (10×10 grid for H0 + 10×10 grid for H1)
"""

import numpy as np
import pandas as pd

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    from persim import PersistenceImager
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


# Default resolution: 10×10 = 100 features per homology dimension,
# 200 features total (H0 + H1). Chosen to keep dimensionality
# manageable on ~50k-sample training pools.
DEFAULT_RESOLUTION = 10


def compute_diagrams_for_windows(point_clouds, max_dim=1, verbose=True):
    """Compute Vietoris-Rips persistence diagrams for every window.

    Returns a list of dicts {'H0': ndarray(n,2), 'H1': ndarray(m,2)}.
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser required: pip install ripser")

    out = []
    n = len(point_clouds)
    for i, pc in enumerate(point_clouds):
        try:
            result = ripser(np.asarray(pc, dtype=float), maxdim=max_dim)
            out.append({f'H{d}': result['dgms'][d]
                        for d in range(len(result['dgms']))})
        except Exception as e:
            if verbose:
                print(f"  [v2 diag {i}] Error: {e}; using empty diagram")
            out.append({f'H{d}': np.zeros((0, 2)) for d in range(max_dim + 1)})
        if verbose and (i + 1) % 200 == 0:
            print(f"  [v2] {i + 1}/{n} diagrams computed")
    return out


def _strip_infinite(diagram):
    """Drop infinite-persistence points (e.g., always-alive H0 component)."""
    if len(diagram) == 0:
        return np.zeros((0, 2))
    finite = diagram[~np.isinf(diagram[:, 1])]
    if len(finite) == 0:
        return np.zeros((0, 2))
    pers = finite[:, 1] - finite[:, 0]
    return finite[pers > 1e-12]


def _dim_diagrams(all_dgms, dim_key):
    return [_strip_infinite(d.get(dim_key, np.zeros((0, 2)))) for d in all_dgms]


def _fit_imager_for_dim(diagrams, resolution=DEFAULT_RESOLUTION):
    """Fit one PersistenceImager with explicit grid so transform output
    has a fixed 2D shape across calls."""
    if not PERSIM_AVAILABLE:
        raise ImportError("persim required: pip install persim")

    # Compute global birth and persistence ranges from non-empty diagrams.
    births, perss = [], []
    for d in diagrams:
        if len(d) > 0:
            births.extend(d[:, 0].tolist())
            perss.extend((d[:, 1] - d[:, 0]).tolist())

    if not births or not perss:
        # No content — produce a degenerate but well-shaped imager.
        b_max, p_max = 1.0, 1.0
    else:
        b_max = max(float(np.max(births)) * 1.1, 1e-3)
        p_max = max(float(np.max(perss)) * 1.1, 1e-3)

    pixel_size = max(p_max, b_max) / resolution

    pim = PersistenceImager(
        birth_range=(0.0, b_max),
        pers_range=(0.0, p_max),
        pixel_size=pixel_size,
        weight='persistence',
    )
    pim.fit(diagrams, skew=True)
    return pim


def _transform_to_fixed_shape(pim, diagrams, resolution=DEFAULT_RESOLUTION):
    """Transform diagrams and pad/crop each image to (resolution, resolution).

    Persim's PersistenceImager output shape depends on the fitted ranges /
    pixel size; using birth_range/pers_range explicitly produces consistent
    shapes, but we still pad defensively to guarantee a fixed-dim feature.
    """
    raw_imgs = pim.transform(diagrams)
    out = np.zeros((len(diagrams), resolution * resolution), dtype=float)
    for i, img in enumerate(raw_imgs):
        arr = np.asarray(img, dtype=float)
        h, w = arr.shape if arr.ndim == 2 else (1, arr.size)
        if arr.ndim != 2:
            arr = arr.reshape(1, -1)
            h, w = arr.shape
        h_use, w_use = min(h, resolution), min(w, resolution)
        canvas = np.zeros((resolution, resolution), dtype=float)
        canvas[:h_use, :w_use] = arr[:h_use, :w_use]
        out[i] = canvas.ravel()
    return out


def compute_v2_features_for_windows(point_clouds, end_indices=None,
                                     resolution=DEFAULT_RESOLUTION,
                                     max_dim=1, verbose=True):
    """End-to-end v2 feature extraction.

    Returns a DataFrame with columns:
        pim_h0_0 .. pim_h0_(R²-1)        (resolution^2 columns)
        pim_h1_0 .. pim_h1_(R²-1)        (resolution^2 columns)
        window_idx, end_idx              (book-keeping)
    """
    if verbose:
        print(f"  [v2] Computing diagrams for {len(point_clouds)} windows")
    all_dgms = compute_diagrams_for_windows(point_clouds, max_dim=max_dim,
                                              verbose=verbose)

    n = len(all_dgms)

    h0_dgms = _dim_diagrams(all_dgms, 'H0')
    if verbose:
        print(f"  [v2] Fitting H0 imager (resolution={resolution})")
    pim_h0 = _fit_imager_for_dim(h0_dgms, resolution=resolution)
    h0_flat = _transform_to_fixed_shape(pim_h0, h0_dgms, resolution=resolution)

    if max_dim >= 1:
        h1_dgms = _dim_diagrams(all_dgms, 'H1')
        if verbose:
            print(f"  [v2] Fitting H1 imager (resolution={resolution})")
        pim_h1 = _fit_imager_for_dim(h1_dgms, resolution=resolution)
        h1_flat = _transform_to_fixed_shape(pim_h1, h1_dgms, resolution=resolution)
    else:
        h1_flat = np.zeros((n, resolution * resolution))

    if verbose:
        print(f"  [v2] Building feature DataFrame ({n} rows × "
              f"{2 * resolution * resolution} TDA dims)")
    rows = []
    for i in range(n):
        feats = {f'pim_h0_{j}': float(h0_flat[i, j])
                 for j in range(resolution * resolution)}
        feats.update({f'pim_h1_{j}': float(h1_flat[i, j])
                      for j in range(resolution * resolution)})
        feats['window_idx'] = i
        if end_indices is not None:
            feats['end_idx'] = end_indices[i]
        rows.append(feats)

    return pd.DataFrame(rows)


def get_v2_feature_cols(df):
    """All v2 TDA feature columns in df."""
    return [c for c in df.columns
            if c.startswith('pim_h0_') or c.startswith('pim_h1_')]


def get_v2_h0_cols(df):
    return [c for c in df.columns if c.startswith('pim_h0_')]


def get_v2_h1_cols(df):
    return [c for c in df.columns if c.startswith('pim_h1_')]
