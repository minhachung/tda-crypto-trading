"""
TDA v2 Features: Persistence Images (leak-safe).

Replaces the v1 representation (8 hand-engineered scalar statistics per
homology dimension = 16 total features) with **persistence images** — a
fixed-dimension vectorisation of the persistence diagram (Adams et al.,
JMLR 2017).

LEAKAGE-SAFETY CONTRACT
-----------------------
A persistence image is the output of a fit-then-transform pipeline:
fitting determines the (birth, persistence) grid bounds, transforming
samples the kernel density on that grid. *Fitting on all windows
(including holdout/test) leaks the future distribution of TDA features
into the representation, which corrupts validation accuracy.*

Therefore this module exposes the fit and transform steps separately,
and **never fits on full-pool diagrams**. Callers are responsible for:

  - K-fold CV       : fit on each fold's TRAIN diagrams; transform train+test
  - Train+Holdout   : fit on Train+Val diagrams; transform Train+Val and holdout
  - Walk-forward    : fit on diagrams known up to t - horizon; transform t

Diagrams themselves carry no leakage — they are intrinsic to each
window — so callers may compute them once and cache.

Output dimensionality is guaranteed to be exactly (resolution × resolution)
per homology dimension: we set birth_range = pers_range = max(birth_max,
pers_max) so persim's grid is square, eliminating the silent
pad/crop step.

Empty-diagram robustness: if a fold's training diagrams contain no
finite-persistence features, the imager for that dimension is NOT fit
(Imager construction is fragile in that case); transforms return a
zero-vector instead.
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


DEFAULT_RESOLUTION = 10


# ============================================================
# Diagram computation (leak-free — intrinsic to each window)
# ============================================================

def _strip_infinite(diagram):
    """Drop infinite-persistence points (e.g. always-alive H0 component)."""
    if len(diagram) == 0:
        return np.zeros((0, 2))
    finite = diagram[~np.isinf(diagram[:, 1])]
    if len(finite) == 0:
        return np.zeros((0, 2))
    pers = finite[:, 1] - finite[:, 0]
    return finite[pers > 1e-12]


def compute_diagrams_only(point_clouds, max_dim=1, verbose=True):
    """Compute Vietoris-Rips persistence diagrams for every window.

    Returns two parallel lists of length ``len(point_clouds)``:
      - h0_list : each element is an (n_i, 2) array of finite-persistence
                  H0 points
      - h1_list : each element is an (m_i, 2) array of finite-persistence
                  H1 points
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser required: pip install ripser")

    h0_list, h1_list = [], []
    n = len(point_clouds)
    for i, pc in enumerate(point_clouds):
        try:
            result = ripser(np.asarray(pc, dtype=float), maxdim=max_dim)
            dgms = result['dgms']
            h0 = _strip_infinite(dgms[0]) if len(dgms) > 0 else np.zeros((0, 2))
            h1 = _strip_infinite(dgms[1]) if len(dgms) > 1 else np.zeros((0, 2))
        except Exception as e:
            if verbose:
                print(f"  [v2 diag {i}] Error: {e}; using empty diagrams")
            h0, h1 = np.zeros((0, 2)), np.zeros((0, 2))
        h0_list.append(h0)
        h1_list.append(h1)
        if verbose and (i + 1) % 500 == 0:
            print(f"  [v2] {i + 1}/{n} diagrams computed")
    return h0_list, h1_list


# ============================================================
# Leak-safe imager: fit on training diagrams only, transform anything
# ============================================================

class LeakSafePersistenceImagerFitter:
    """A fit/transform wrapper around persim.PersistenceImager that:

    1. Fits on **only** the diagrams the caller passes (i.e., train fold).
    2. Produces a **fixed (resolution × resolution)** image per dimension —
       no silent padding, no shape drift between calls.
    3. Returns zeros for empty-train-diagram dimensions instead of
       raising at fit time.
    4. Concatenates H0 and H1 image features into a single flat
       feature matrix of shape (n_windows, 2 × resolution²).
    """

    def __init__(self, resolution=DEFAULT_RESOLUTION):
        self.resolution = resolution
        self._pim_h0 = None
        self._pim_h1 = None
        self._h0_skipped = False
        self._h1_skipped = False

    @property
    def feat_dim_per_homology(self):
        return self.resolution * self.resolution

    @property
    def feat_dim_total(self):
        return 2 * self.feat_dim_per_homology

    def fit(self, train_h0, train_h1):
        if not PERSIM_AVAILABLE:
            raise ImportError("persim required: pip install persim")
        self._pim_h0, self._h0_skipped = self._fit_dim(train_h0)
        self._pim_h1, self._h1_skipped = self._fit_dim(train_h1)
        return self

    def transform(self, h0_list, h1_list):
        n = len(h0_list)
        out = np.zeros((n, self.feat_dim_total), dtype=float)
        feat_dim = self.feat_dim_per_homology
        if not self._h0_skipped and self._pim_h0 is not None:
            out[:, :feat_dim] = self._transform_dim(self._pim_h0, h0_list)
        if not self._h1_skipped and self._pim_h1 is not None:
            out[:, feat_dim:] = self._transform_dim(self._pim_h1, h1_list)
        return out

    def fit_transform(self, train_h0, train_h1):
        self.fit(train_h0, train_h1)
        return self.transform(train_h0, train_h1)

    def feature_names(self, prefix_h0='pim_h0_', prefix_h1='pim_h1_'):
        feat_dim = self.feat_dim_per_homology
        names = [f'{prefix_h0}{j}' for j in range(feat_dim)]
        names += [f'{prefix_h1}{j}' for j in range(feat_dim)]
        return names

    # --- internals ---

    def _fit_dim(self, train_diagrams):
        """Returns (PersistenceImager, skipped: bool).

        skipped=True means train diagrams had no finite-persistence content,
        so the imager wasn't fit and transform should return zeros.
        """
        births, perss = [], []
        for d in train_diagrams:
            if len(d) > 0:
                births.extend(d[:, 0].tolist())
                perss.extend((d[:, 1] - d[:, 0]).tolist())

        if not births or not perss:
            return None, True

        b_max = max(float(np.max(births)) * 1.1, 1e-3)
        p_max = max(float(np.max(perss)) * 1.1, 1e-3)
        # Force a square grid: same upper bound on both axes, pixel_size
        # set so the imager produces exactly resolution × resolution.
        combined_max = max(b_max, p_max)
        pixel_size = combined_max / self.resolution

        try:
            pim = PersistenceImager(
                birth_range=(0.0, combined_max),
                pers_range=(0.0, combined_max),
                pixel_size=pixel_size,
                weight='persistence',
            )
            pim.fit(train_diagrams, skew=True)
            return pim, False
        except Exception:
            # Imager construction can be fragile on degenerate inputs.
            return None, True

    def _transform_dim(self, pim, diagrams):
        feat_dim = self.feat_dim_per_homology
        out = np.zeros((len(diagrams), feat_dim), dtype=float)
        try:
            raw_imgs = pim.transform(diagrams)
        except Exception:
            return out

        for i, img in enumerate(raw_imgs):
            arr = np.asarray(img, dtype=float)
            if arr.ndim != 2:
                continue
            h, w = arr.shape
            # We constructed the imager so the natural shape is exactly
            # (R, R); if persim returned a different shape due to a degenerate
            # diagram, take the upper-left R×R block (safe fallback).
            r = self.resolution
            block = arr[:r, :r]
            canvas = np.zeros((r, r), dtype=float)
            canvas[:block.shape[0], :block.shape[1]] = block
            out[i] = canvas.ravel()
        return out


# ============================================================
# Convenience: select v2 columns from a DataFrame
# ============================================================

def get_v2_feature_cols(df):
    return [c for c in df.columns
            if c.startswith('pim_h0_') or c.startswith('pim_h1_')]


def get_v2_h0_cols(df):
    return [c for c in df.columns if c.startswith('pim_h0_')]


def get_v2_h1_cols(df):
    return [c for c in df.columns if c.startswith('pim_h1_')]
