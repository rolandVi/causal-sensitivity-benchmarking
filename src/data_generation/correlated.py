from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from .base import BaseDGP, BaseDGPConfig


@dataclass
class UniformProxyDGPConfig(BaseDGPConfig):
    """Hyperparameters for the Uniform Proxy DGP

    Attributes:
        alpha: strength of the correlated variable X2
        rho: Gaussian copula correlation between X1 and X2
    """
    alpha: float = 0.0
    rho: float = 0.5
    p_x: int = 5
    p_u: int = 2


class UniformProxyDGP(BaseDGP):
    """Uniform Proxy DGP with correlated uniform variables.

    Implementation details:
    - (X1, X2) are correlated uniform variables in [-1, 1].
    - Generation uses a Gaussian Copula with configurable normal correlation rho.
    - Resulting uniform (Spearman) correlation is approximately 6/pi * arcsin(rho/2).
    - Propensity model: logit = X @ beta_x + alpha * X2 + U @ beta_u
    - beta_x = [1.0, 0.0, 0.3, 0.3, 0.3]
    - beta_u = [0.3, 0.3]
    """

    def __init__(self, config: UniformProxyDGPConfig):
        super().__init__(config)
        self.beta_x = np.array([1.0, 0.0, 0.3, 0.3, 0.3])
        self.beta_u = np.array([0.3, 0.3])

    def sample_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample X with correlated uniform variables X1 and X2.

        Uses Gaussian Copula to map bivariate normal to uniform [-1, 1].
        """
        rho = getattr(self.config, "rho", 0.5)
        cov = np.array([[1.0, rho], [rho, 1.0]])
        Z = rng.multivariate_normal([0.0, 0.0], cov, size=n)

        X12 = 2.0 * norm.cdf(Z) - 1.0

        remaining_p = self.config.p_x - 2
        if remaining_p > 0:
            X_rest = rng.uniform(-1.0, 1.0, size=(n, remaining_p))
            return np.hstack([X12, X_rest])
        return X12

    def nominal_propensity(self, X: np.ndarray) -> np.ndarray:
        """e(X) = logistic(β_X^T X + α X_2)"""
        alpha = getattr(self.config, "alpha", 0.0)
        logit = X @ self.beta_x + alpha * X[:, 1]
        return expit(logit)

    def true_propensity(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """e(X, U) = logistic(β_X^T X + α X_2 + β_U^T U)"""
        alpha = getattr(self.config, "alpha", 0.0)
        logit = X @ self.beta_x + alpha * X[:, 1] + U @ self.beta_u
        return expit(logit)
