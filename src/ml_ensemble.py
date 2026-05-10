"""
Leak-Safe Ensemble Framework: Stacks 5 diverse base learners.

Base learners:
  1. Logistic Regression (L2, calibrated)
  2. Random Forest (100 trees, max_depth=5)
  3. XGBoost (early stopping)
  4. LightGBM (fast GBDT)
  5. Gradient Boosting (sklearn, flexible)

Meta-learner:
  - Logistic Regression on validation fold predictions

Leak-safety: Meta-learner trained only on validation fold, never sees test.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LeakSafeEnsemble:
    """Stack 5 base learners with leak-safe meta-learner."""

    def __init__(self, random_state: int = 42, verbose: int = 0):
        """
        Initialize ensemble.

        Args:
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.random_state = random_state
        self.verbose = verbose
        self.base_learners = {}
        self.meta_learner = None
        self.base_predictions_val = None
        self.meta_features_val = None

    def _build_base_learners(self) -> Dict:
        """Build 5 base learners."""
        learners = {}

        # 1. Logistic Regression
        learners['logistic'] = LogisticRegression(
            C=1.0, max_iter=500, random_state=self.random_state, solver='lbfgs'
        )

        # 2. Random Forest
        learners['rf'] = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=self.random_state, n_jobs=-1
        )

        # 3. XGBoost (if available)
        if xgb:
            learners['xgboost'] = xgb.XGBClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                random_state=self.random_state, verbosity=0, n_jobs=-1
            )

        # 4. LightGBM (if available)
        if lgb:
            learners['lgbm'] = lgb.LGBMClassifier(
                n_estimators=80, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, verbosity=-1, n_jobs=-1
            )

        # 5. Gradient Boosting
        learners['gbm'] = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            random_state=self.random_state
        )

        return learners

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> 'LeakSafeEnsemble':
        """
        Train ensemble with leak-safe meta-learner.

        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,)
            X_val: Validation features (n_val, n_features)
            y_val: Validation labels (n_val,)

        Returns:
            self
        """
        self.base_learners = self._build_base_learners()
        n_learners = len(self.base_learners)

        if self.verbose > 0:
            print(f"Training {n_learners} base learners...")

        # Train base learners on train set, predict on validation set
        self.base_predictions_val = np.zeros((len(X_val), n_learners))
        self.meta_features_val = np.zeros((len(X_val), n_learners))

        for idx, (name, learner) in enumerate(self.base_learners.items()):
            if self.verbose > 0:
                print(f"  {idx + 1}. Training {name}...")

            learner.fit(X_train, y_train)

            # Get validation fold predictions
            val_pred_proba = learner.predict_proba(X_val)[:, 1]
            self.base_predictions_val[:, idx] = val_pred_proba

            # Compute validation accuracy
            val_acc = accuracy_score(y_val, learner.predict(X_val))
            if self.verbose > 0:
                print(f"     Val Accuracy: {val_acc:.4f}")

        # Train meta-learner on validation fold predictions
        if self.verbose > 0:
            print(f"Training meta-learner on {len(X_val)} validation samples...")

        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=500, random_state=self.random_state, solver='lbfgs'
        )
        self.meta_learner.fit(self.base_predictions_val, y_val)
        meta_acc = accuracy_score(y_val, self.meta_learner.predict(self.base_predictions_val))
        if self.verbose > 0:
            print(f"  Meta Val Accuracy: {meta_acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities (n_samples, 2) where [:, 1] = P(y=1)
        """
        if not self.base_learners or self.meta_learner is None:
            raise ValueError("Ensemble not trained. Call fit() first.")

        # Get base predictions
        base_preds = np.zeros((len(X), len(self.base_learners)))
        for idx, (name, learner) in enumerate(self.base_learners.items()):
            base_preds[:, idx] = learner.predict_proba(X)[:, 1]

        # Meta-learner prediction
        meta_proba = self.meta_learner.predict_proba(base_preds)
        return meta_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def get_base_importances(self) -> Dict[str, np.ndarray]:
        """
        Extract feature importances from tree-based base learners.

        Returns:
            Dict mapping learner name -> importances
        """
        importances = {}

        for name, learner in self.base_learners.items():
            if hasattr(learner, 'feature_importances_'):
                importances[name] = learner.feature_importances_
            elif hasattr(learner, 'coef_'):
                importances[name] = np.abs(learner.coef_[0])

        return importances
