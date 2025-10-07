"""Lightweight predictive model for next-bar return estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PredictionResult:
    """Container for a return prediction and supporting metadata."""

    expected_return: float
    confidence: float
    samples: int

    @property
    def direction(self) -> int:
        """Return the trading direction implied by the prediction."""

        if self.expected_return > 0:
            return 1
        if self.expected_return < 0:
            return -1
        return 0


class AIPredictor:
    """Simple ridge-regression model built on lagged return features."""

    def __init__(
        self,
        lookback: int = 20,
        regularization: float = 1e-4,
        min_history: Optional[int] = None,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.lookback = lookback
        self.regularization = regularization
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self.min_history = max(lookback + 1, min_history or 0)

    @staticmethod
    def _to_series(values: Iterable[float]) -> pd.Series:
        series = pd.Series(values, dtype="float64")
        return series.dropna()

    def _build_design_matrix(self, closes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        returns = closes.pct_change().dropna()
        if len(returns) <= self.lookback:
            raise ValueError("insufficient data to build features")
        features = []
        targets = []
        for end in range(self.lookback, len(returns)):
            window = returns.iloc[end - self.lookback : end]
            features.append(window.to_numpy())
            targets.append(returns.iloc[end])
        X = np.asarray(features, dtype="float64")
        y = np.asarray(targets, dtype="float64")
        return X, y

    def fit(self, closes: Iterable[float]) -> PredictionResult:
        """Fit ridge regression weights using *closes* price history."""

        series = self._to_series(closes)
        if len(series) < self.min_history:
            raise ValueError("not enough history to fit predictor")
        X, y = self._build_design_matrix(series)
        XtX = X.T @ X
        ridge = XtX + self.regularization * np.identity(XtX.shape[0])
        XtY = X.T @ y
        weights = np.linalg.solve(ridge, XtY)
        # The intercept should recenter predictions relative to the mean of the
        # training features. Using the feature mean instead of ``weights.mean``
        # prevents runaway offsets when the per-feature scales differ.
        feature_mean = X.mean(axis=0)
        bias = float(y.mean() - feature_mean @ weights)
        self._weights = weights
        self._bias = bias
        # Provide a backtest-style in-sample prediction for transparency.
        mean_pred = float((X @ weights + bias).mean())
        variance = float(np.var(y - (X @ weights + bias))) if len(y) > 1 else 0.0
        confidence = 1.0 / (1.0 + variance)
        return PredictionResult(expected_return=mean_pred, confidence=confidence, samples=len(y))

    def is_trained(self) -> bool:
        return self._weights is not None

    def predict(self, closes: Iterable[float]) -> PredictionResult:
        """Predict the next-bar return from closing prices."""

        series = self._to_series(closes)
        if len(series) <= self.lookback:
            raise ValueError("not enough history to predict")
        if not self.is_trained():
            self.fit(series)
        returns = series.pct_change().dropna()
        if len(returns) < self.lookback:
            raise ValueError("not enough return history to predict")
        assert self._weights is not None
        window = returns.iloc[-self.lookback :]
        features = window.to_numpy()
        expected = float(features @ self._weights + self._bias)
        # Confidence decays with prediction magnitude relative to historical dispersion.
        dispersion = float(np.std(features)) if len(features) > 1 else 1.0
        confidence = 1.0 / (1.0 + abs(expected) / max(dispersion, 1e-9))
        return PredictionResult(expected_return=expected, confidence=confidence, samples=len(series))
