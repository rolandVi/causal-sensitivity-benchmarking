from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .linear import LinearDGP, LinearDGPConfig


@dataclass
class QuadraticDGPConfig(LinearDGPConfig):
    """Hyperparameters for the quadratic DGP (Continuous Misspecification)

    Following Task 2 of the Research Plan (page 2):
    "P(T = 1 | X, U) = logistic(β^T X + α X1^2 + βu U)"

    Attributes:
        alpha: strength of the quadratic term for X1 in the propensity score
    """
    alpha: float = 0.5


class QuadraticDGP(LinearDGP):
    """Quadratic DGP for binary treatment (Continuous Misspecification).

    Inherits from LinearDGP but overrides the true propensity score to include
    a non-linear term (α X1²), while the nominal propensity remains linear (misspecified).
    """

    def true_propensity(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """e(X, U) = logistic(β_X^T X + α X1^2 + β_U^T U) — true propensity including U and non-linearity"""
        alpha = getattr(self.config, 'alpha', 0.0)
        logit = (
            X @ self.beta_x
            + alpha * (X[:, 0] ** 2 - (1.0/3.0))
            + U @ self.beta_u
        )
        return expit(logit)
