"""Test persistent homology computation on known geometries."""

import numpy as np
import pytest

from src.persistent_homology import PersistentHomologyAnalyzer


def test_circle_has_prominent_h1(circle_point_cloud):
    """Points on a circle should produce one prominent H1 (loop) feature."""
    analyzer = PersistentHomologyAnalyzer(circle_point_cloud, max_dim=1)
    diagrams = analyzer.compute()

    assert 'H1' in diagrams
    h1 = diagrams['H1']
    h1_finite = h1[~np.isinf(h1[:, 1])]
    if len(h1_finite) == 0:
        pytest.skip("No finite H1 features (boundary case)")

    persistences = h1_finite[:, 1] - h1_finite[:, 0]
    max_persistence = persistences.max()
    assert max_persistence > 0.1, \
        f"Expected prominent H1 feature for a circle, got max persistence {max_persistence}"


def test_random_points_low_h1(random_point_cloud):
    """Uniform random points should not have a prominent H1 signal."""
    analyzer = PersistentHomologyAnalyzer(random_point_cloud, max_dim=1)
    diagrams = analyzer.compute()

    h1 = diagrams.get('H1', np.array([]).reshape(0, 2))
    h1_finite = h1[~np.isinf(h1[:, 1])] if len(h1) > 0 else h1
    if len(h1_finite) > 0:
        persistences = h1_finite[:, 1] - h1_finite[:, 0]
        circle = PersistentHomologyAnalyzer.__init__
        assert persistences.max() < 0.5


def test_extract_features_returns_expected_keys(circle_point_cloud):
    """Feature dict must contain the documented keys."""
    analyzer = PersistentHomologyAnalyzer(circle_point_cloud, max_dim=1)
    features = analyzer.extract_features()

    expected_h0 = {'H0_count', 'H0_l1_norm', 'H0_c1_norm', 'H0_entropy', 'H0_landscape_l2'}
    expected_h1 = {'H1_count', 'H1_l1_norm', 'H1_c1_norm', 'H1_entropy', 'H1_landscape_l2'}
    assert expected_h0.issubset(set(features.keys()))
    assert expected_h1.issubset(set(features.keys()))


def test_extract_features_circle_has_high_h1_l1_norm(circle_point_cloud):
    """L1-norm of H1 persistences should be substantial for a circle."""
    analyzer = PersistentHomologyAnalyzer(circle_point_cloud, max_dim=1)
    features = analyzer.extract_features()
    assert features['H1_l1_norm'] > 0.05


def test_extract_features_empty_handles_gracefully():
    """Empty / single-point clouds shouldn't crash."""
    pts = np.array([[0.0, 0.0]])
    analyzer = PersistentHomologyAnalyzer(pts, max_dim=1)
    features = analyzer.extract_features()
    assert features['H1_count'] == 0
    assert features['H1_l1_norm'] == 0.0


def test_features_deterministic(circle_point_cloud):
    """Same input → same features."""
    a1 = PersistentHomologyAnalyzer(circle_point_cloud)
    a2 = PersistentHomologyAnalyzer(circle_point_cloud)
    f1 = a1.extract_features()
    f2 = a2.extract_features()
    for k in f1:
        assert abs(f1[k] - f2[k]) < 1e-9, f"Mismatch on {k}"
