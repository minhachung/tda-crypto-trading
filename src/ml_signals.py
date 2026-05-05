"""
ML-Based Signal Generator: Train a classifier on TDA features.

Instead of hand-crafted threshold rules (which the v2 validation showed
don't work), train a classifier to predict next-period direction from
the full TDA feature vector + price-derived features.

Critical: training is done WITHIN each CV fold to avoid lookahead bias.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class MLSignalGenerator:
    """Train a classifier on TDA features to predict direction."""

    def __init__(self, model_type='logistic', horizon=1, prob_threshold=0.55,
                 max_position_size=0.10, random_state=42):
        """
        Args:
            model_type: 'logistic', 'rf', or 'gbm'
            horizon: forward steps to predict (1 = next period)
            prob_threshold: minimum probability to take a position
            max_position_size: max fraction of capital per trade
        """
        self.model_type = model_type
        self.horizon = horizon
        self.prob_threshold = prob_threshold
        self.max_position_size = max_position_size
        self.random_state = random_state
        self.pipeline = None
        self.feature_cols = None

    def _make_pipeline(self):
        if self.model_type == 'logistic':
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=self.random_state)
        elif self.model_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=1,
            )
        elif self.model_type == 'gbm':
            clf = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                min_samples_leaf=10, random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    @staticmethod
    def _make_target(prices, horizon=1):
        """Binary target: 1 if price up after `horizon` periods, else 0."""
        prices = np.asarray(prices, dtype=float)
        future = np.roll(prices, -horizon)
        future[-horizon:] = np.nan
        target = (future > prices).astype(float)
        target[-horizon:] = np.nan
        return target

    def _select_features(self, features_df):
        """Pick numeric TDA columns; drop indices/identifiers."""
        drop = {'window_idx', 'end_idx'}
        cols = [c for c in features_df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(features_df[c])]
        return cols

    def fit(self, features_df, prices):
        """Train on (features, future-direction) pairs."""
        self.feature_cols = self._select_features(features_df)
        X = features_df[self.feature_cols].values
        y = self._make_target(prices, horizon=self.horizon)

        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask].astype(int)

        if len(np.unique(y)) < 2:
            self.pipeline = None
            return self

        self.pipeline = self._make_pipeline()
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, features_df):
        """Predict probability of UP move next period."""
        if self.pipeline is None:
            return np.full(len(features_df), 0.5)
        X = features_df[self.feature_cols].values
        probas = self.pipeline.predict_proba(X)
        if probas.shape[1] == 2:
            return probas[:, 1]
        return probas[:, 0]

    def generate_signals(self, features_df, price_series=None):
        """
        Predict probabilities and convert to BUY/SELL/HOLD.

        BUY when P(up) > prob_threshold
        SELL when P(up) < 1 - prob_threshold
        HOLD otherwise
        """
        df = features_df.copy()
        proba = self.predict_proba(features_df)
        df['p_up'] = proba

        signals, sizes, confs = [], [], []
        for p in proba:
            confidence = abs(p - 0.5) * 2
            if p > self.prob_threshold:
                signal = 'BUY'
                size = self.max_position_size * confidence
            elif p < 1 - self.prob_threshold:
                signal = 'SELL'
                size = -self.max_position_size * confidence
            else:
                signal = 'HOLD'
                size = 0.0
            signals.append(signal)
            sizes.append(size)
            confs.append(confidence)

        df['signal'] = signals
        df['position_size'] = sizes
        df['confidence_final'] = confs
        return df


def evaluate_ml_kfold(features_df, prices, n_splits=5, model_type='logistic',
                      prob_threshold=0.55, horizon=1, verbose=False):
    """
    K-fold time-series CV with ML classifier.

    For each fold:
      - Train classifier on training portion
      - Predict on test portion
      - Score direction accuracy + run backtest
    """
    from src.validation_v2 import time_series_kfold, _direction_score
    from src.backtester import Backtester

    folds = time_series_kfold(len(features_df), n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_features = features_df.iloc[train_idx].reset_index(drop=True)
        train_prices = prices[train_idx]
        test_features = features_df.iloc[test_idx].reset_index(drop=True)
        test_prices = prices[test_idx]

        gen = MLSignalGenerator(
            model_type=model_type,
            horizon=horizon,
            prob_threshold=prob_threshold,
        )
        try:
            gen.fit(train_features, train_prices)
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx}: fit error -- {e}")
            continue

        if gen.pipeline is None:
            continue

        signals = gen.generate_signals(test_features, price_series=test_prices)

        bt = Backtester()
        res = bt.run(test_prices, signals)
        direction = _direction_score(signals, test_prices, horizon=horizon)
        bh_return = (test_prices[-1] / test_prices[0] - 1) * 100

        n_buy = int((signals['signal'] == 'BUY').sum())
        n_sell = int((signals['signal'] == 'SELL').sum())

        fold_results.append({
            'fold': fold_idx,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'tda_return_pct': res['metrics']['total_return_pct'],
            'tda_sharpe': res['metrics']['sharpe_ratio'],
            'tda_sortino': res['metrics']['sortino_ratio'],
            'tda_max_dd_pct': res['metrics']['max_drawdown_pct'],
            'tda_n_trades': res['metrics'].get('completed_trades', 0),
            'direction_accuracy': direction,
            'n_buy': n_buy, 'n_sell': n_sell,
            'buy_hold_return_pct': bh_return,
            'outperformed_bh': res['metrics']['total_return_pct'] > bh_return,
        })

    return pd.DataFrame(fold_results)


def grid_search_ml(features_df, prices, n_splits=5, verbose=False):
    """Find best ML model + threshold combination."""
    grid = []
    for model_type in ['logistic', 'rf', 'gbm']:
        for thresh in [0.52, 0.55, 0.58, 0.60]:
            for horizon in [1, 3, 5]:
                grid.append((model_type, thresh, horizon))

    if verbose:
        print(f"  Grid: {len(grid)} combos x {n_splits} folds")

    results = []
    for model_type, thresh, horizon in grid:
        try:
            fold_df = evaluate_ml_kfold(
                features_df, prices, n_splits=n_splits,
                model_type=model_type, prob_threshold=thresh, horizon=horizon,
            )
            if len(fold_df) == 0:
                continue
            results.append({
                'model_type': model_type,
                'prob_threshold': thresh,
                'horizon': horizon,
                'mean_direction_acc': float(fold_df['direction_accuracy'].mean()),
                'std_direction_acc': float(fold_df['direction_accuracy'].std()),
                'mean_sharpe': float(fold_df['tda_sharpe'].mean()),
                'mean_return_pct': float(fold_df['tda_return_pct'].mean()),
                'mean_buy_hold_pct': float(fold_df['buy_hold_return_pct'].mean()),
                'pct_beat_bh': float(fold_df['outperformed_bh'].mean()),
                'mean_n_trades': float(fold_df['tda_n_trades'].mean()),
            })
        except Exception as e:
            if verbose:
                print(f"  Skip {model_type}/{thresh}/{horizon}: {e}")

    df = pd.DataFrame(results).sort_values('mean_direction_acc', ascending=False)
    return df
