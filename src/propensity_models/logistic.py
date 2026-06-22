import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BasePropensityEstimator


class LogisticPropensityEstimator(BasePropensityEstimator):
    """Propensity score estimator based on logistic regression

    Args:
        **kwargs: forwarded to sklearn.linear_model.LogisticRegression
    """

    def __init__(self, **kwargs):
        self._model = LogisticRegression(max_iter=1000, **kwargs)

    def fit(self, X: np.ndarray, T: np.ndarray) -> "LogisticPropensityEstimator":
        self._model.fit(X, T)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def coefficients(self) -> np.ndarray:
        """Return the fitted coefficients of the logistic regression."""
        return self._model.coef_[0]


class QuadraticLogisticPropensityEstimator(LogisticPropensityEstimator):
    """Logistic regression augmented with a single quadratic term.

    Exactly mirrors the QuadraticDGP propensity form:
    P(T=1|X) = logistic(beta^T X + alpha * X[i]^2)

    Args:
        feature_idx: index of the feature to square
        **kwargs:    forwarded to sklearn.linear_model.LogisticRegression
    """

    def __init__(self, feature_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.feature_idx = feature_idx

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Append X[i]^2 as the last column"""
        quad_term = X[:, [self.feature_idx]] ** 2
        return np.hstack([X, quad_term])

    def fit(self, X: np.ndarray, T: np.ndarray) -> "QuadraticLogisticPropensityEstimator":
        X_aug = self._augment(X)
        self._model.fit(X_aug, T)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = self._augment(X)
        return self._model.predict_proba(X_aug)[:, 1]
