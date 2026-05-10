"""
Regime-Adaptive Models: Train separate models for low/medium/high volatility regimes.

Hypothesis: TDA signal varies by market regime. Trending markets behave differently
from sideways or mean-reverting periods. By training a separate classifier for each
regime and routing test samples accordingly, we can:
  1. Improve per-regime accuracy
  2. Reveal which regime carries the actual signal
  3. Avoid contaminating signal with regime noise

Regime detection: Volatility-based (rolling 20-period quantiles).
Models: Separate logistic regression per regime (configurable).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class RegimeClassifier:
    """Detects market regime from rolling volatility."""

    def __init__(self, lookback: int = 20, low_quantile: float = 0.33,
                 high_quantile: float = 0.66):
        """
        Initialize regime classifier.

        Args:
            lookback: Rolling window for vol calculation
            low_quantile: Threshold for "low" regime
            high_quantile: Threshold for "high" regime
        """
        self.lookback = lookback
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.low_threshold_ = None
        self.high_threshold_ = None

    def fit(self, vol_series: np.ndarray) -> 'RegimeClassifier':
        """
        Fit regime thresholds from training volatility.

        Args:
            vol_series: Volatility series (e.g., rolling std of returns)

        Returns:
            self
        """
        vol_clean = vol_series[~np.isnan(vol_series)]
        self.low_threshold_ = np.quantile(vol_clean, self.low_quantile)
        self.high_threshold_ = np.quantile(vol_clean, self.high_quantile)
        return self

    def predict_regime(self, vol_series: np.ndarray) -> np.ndarray:
        """
        Predict regime for each sample.

        Args:
            vol_series: Volatility series

        Returns:
            Regime labels: 0=low, 1=medium, 2=high
        """
        if self.low_threshold_ is None or self.high_threshold_ is None:
            raise ValueError("Classifier not fit. Call fit() first.")

        regimes = np.ones(len(vol_series), dtype=int)  # Default: medium
        regimes[vol_series < self.low_threshold_] = 0  # Low
        regimes[vol_series >= self.high_threshold_] = 2  # High

        return regimes


class RegimeAdaptiveModel:
    """Train separate classifiers per regime; route test samples accordingly."""

    REGIME_NAMES = {0: 'low', 1: 'medium', 2: 'high'}

    def __init__(self, base_model_class=LogisticRegression,
                 base_model_kwargs: Optional[Dict] = None,
                 random_state: int = 42, verbose: int = 0):
        """
        Initialize regime-adaptive model.

        Args:
            base_model_class: sklearn-compatible classifier class
            base_model_kwargs: kwargs for base model
            random_state: Random seed
            verbose: Verbosity
        """
        self.base_model_class = base_model_class
        self.base_model_kwargs = base_model_kwargs or {
            'C': 1.0, 'max_iter': 500, 'solver': 'lbfgs'
        }
        self.random_state = random_state
        self.verbose = verbose
        self.regime_classifier = RegimeClassifier()
        self.models = {}  # regime -> trained model
        self.fallback_model = None  # For unseen regimes
        self.regime_accuracies_ = {}

    def _build_model(self):
        """Build a fresh base model instance."""
        kwargs = {**self.base_model_kwargs, 'random_state': self.random_state}
        return self.base_model_class(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, vol_series: np.ndarray) -> 'RegimeAdaptiveModel':
        """
        Train regime-specific models.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            vol_series: Volatility for regime classification (n_samples,)

        Returns:
            self
        """
        # Fit regime thresholds
        self.regime_classifier.fit(vol_series)
        regimes = self.regime_classifier.predict_regime(vol_series)

        # Train one model per regime
        for regime_id in [0, 1, 2]:
            mask = regimes == regime_id
            n_samples = mask.sum()

            if n_samples < 30:  # Insufficient data
                if self.verbose > 0:
                    print(f"  Regime {self.REGIME_NAMES[regime_id]}: insufficient samples ({n_samples})")
                continue

            X_regime = X[mask]
            y_regime = y[mask]

            # Need at least 2 classes for training
            if len(np.unique(y_regime)) < 2:
                if self.verbose > 0:
                    print(f"  Regime {self.REGIME_NAMES[regime_id]}: only one class present")
                continue

            model = self._build_model()
            model.fit(X_regime, y_regime)

            train_acc = accuracy_score(y_regime, model.predict(X_regime))
            self.regime_accuracies_[regime_id] = train_acc

            self.models[regime_id] = model

            if self.verbose > 0:
                print(f"  Regime {self.REGIME_NAMES[regime_id]} ({n_samples} samples): "
                      f"train_acc={train_acc:.4f}")

        # Fallback: train on full data
        self.fallback_model = self._build_model()
        self.fallback_model.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray, vol_series: np.ndarray) -> np.ndarray:
        """
        Predict probabilities by routing each sample to its regime's model.

        Args:
            X: Features (n_samples, n_features)
            vol_series: Volatility series (n_samples,)

        Returns:
            Probabilities (n_samples, 2)
        """
        regimes = self.regime_classifier.predict_regime(vol_series)
        proba = np.zeros((len(X), 2))

        for regime_id in [0, 1, 2]:
            mask = regimes == regime_id

            if not mask.any():
                continue

            if regime_id in self.models:
                proba[mask] = self.models[regime_id].predict_proba(X[mask])
            else:
                # Fallback to global model for unseen regimes
                proba[mask] = self.fallback_model.predict_proba(X[mask])

        return proba

    def predict(self, X: np.ndarray, vol_series: np.ndarray) -> np.ndarray:
        """Predict labels via regime routing."""
        proba = self.predict_proba(X, vol_series)
        return (proba[:, 1] > 0.5).astype(int)

    def evaluate_per_regime(self, X: np.ndarray, y: np.ndarray,
                            vol_series: np.ndarray) -> Dict[str, float]:
        """
        Evaluate accuracy per regime.

        Returns:
            Dict mapping regime name -> accuracy
        """
        regimes = self.regime_classifier.predict_regime(vol_series)
        predictions = self.predict(X, vol_series)
        results = {}

        for regime_id in [0, 1, 2]:
            mask = regimes == regime_id
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            acc = accuracy_score(y[mask], predictions[mask])
            results[self.REGIME_NAMES[regime_id]] = {
                'accuracy': acc,
                'n_samples': int(n_samples)
            }

        # Global accuracy
        results['global'] = {
            'accuracy': accuracy_score(y, predictions),
            'n_samples': len(y)
        }

        return results
