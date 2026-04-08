"""Regime classifier: identifies market regime for meta-policy routing.

This is the MOST IMPORTANT model in the system — if you know the regime,
even simple strategies work. If you misidentify, even brilliant strategies fail.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
from loguru import logger


class MarketRegime(IntEnum):
    """Market regime categories for XAUUSD."""

    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3


class RegimeLabeler:
    """Label historical data with market regimes using rule-based heuristics.

    These labels become the supervised training targets for the regime classifier.
    """

    def __init__(
        self,
        trend_threshold: float = 0.3,
        volatility_threshold: float = 2.0,
        lookback: int = 60,
    ):
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.lookback = lookback

    def label(
        self,
        close: np.ndarray,
        volatility: np.ndarray,
        adx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate regime labels from price data.

        Args:
            close: Close prices.
            volatility: Rolling volatility (e.g., ATR or realized vol).
            adx: ADX indicator (optional, improves trend detection).

        Returns:
            Array of MarketRegime integer labels.
        """
        n = len(close)
        labels = np.full(n, MarketRegime.RANGING, dtype=np.int64)

        for i in range(self.lookback, n):
            window = close[i - self.lookback : i]
            vol = volatility[i] if i < len(volatility) else 0

            # Returns over lookback
            ret = (close[i] - window[0]) / window[0]
            vol_z = vol / (np.mean(volatility[max(0, i - 200) : i]) + 1e-8)

            # High volatility overrides trend
            if vol_z > self.volatility_threshold:
                labels[i] = MarketRegime.VOLATILE
            elif adx is not None and i < len(adx) and adx[i] > 25:
                # ADX > 25 indicates strong trend
                labels[i] = MarketRegime.TRENDING_UP if ret > 0 else MarketRegime.TRENDING_DOWN
            elif abs(ret) > self.trend_threshold * (self.lookback / 60):
                labels[i] = MarketRegime.TRENDING_UP if ret > 0 else MarketRegime.TRENDING_DOWN
            else:
                labels[i] = MarketRegime.RANGING

        return labels


class RegimeClassifier:
    """Gradient Boosting regime classifier.

    Uses sklearn's HistGradientBoostingClassifier for speed and robustness.
    Designed to be lightweight and fast for real-time inference.
    """

    def __init__(self, n_regimes: int = 4, max_depth: int = 6, n_estimators: int = 200):
        self.n_regimes = n_regimes
        self._model = None
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.feature_names: list[str] = []

    def train(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None
    ) -> dict:
        """Train the regime classifier.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Regime labels.
            feature_names: Optional feature names for interpretability.

        Returns:
            Dict with training metrics.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        self._model = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.n_estimators,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )

        # Cross-validation score
        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring="accuracy")
        logger.info(
            f"Regime classifier CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        # Fit on full data
        self._model.fit(X, y)

        return {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_samples": len(y),
            "class_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
            },
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities (for confidence-based routing)."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self._model.predict_proba(X)

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get confidence of the predicted regime (max probability)."""
        proba = self.predict_proba(X)
        return proba.max(axis=1)

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib

        joblib.dump({"model": self._model, "feature_names": self.feature_names}, path)
        logger.info(f"Regime classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        self._model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info(f"Regime classifier loaded from {path}")
