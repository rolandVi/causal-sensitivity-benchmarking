from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit


@dataclass
class DGPData:
    """Generated dataset"""

    X: np.ndarray           # observed confounders          (n, p_x)
    T: np.ndarray           # binary treatment {0, 1}       (n,)
    U: np.ndarray           # unobserved confounders        (n, p_u)
    e_true: np.ndarray      # true propensity  e(X, U)      (n,)
    e_nominal: np.ndarray   # nominal propensity e(X)       (n,)


@dataclass
class BaseDGPConfig:
    """Base hyperparameters for the DGP

    Attributes:
        p_x:      number of observed confounders X
        p_u:      number of unobserved confounders U
        lambda_:  controls the X–U correlation
    """
    p_x: int = 5
    p_u: int = 2
    lambda_: float = 0.6


class BaseDGP(ABC):
    """Base class for Data Generating Processes."""

    def __init__(self, config: BaseDGPConfig):
        self.config = config
        c = config
        self.beta_x: np.ndarray = np.full(c.p_x, 0.3)
        self.beta_u: np.ndarray = np.full(c.p_u, 0.3)
        self.rho: np.ndarray = np.full((c.p_u, c.p_x), 1.75)

    def nominal_propensity(self, X: np.ndarray) -> np.ndarray:
        """e(X) = logistic(β_X^T X)"""
        return expit(X @ self.beta_x)

    def true_propensity(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """e(X, U) = logistic(β_X^T X + β_U^T U) — true propensity including U"""
        return expit(X @ self.beta_x + U @ self.beta_u)

    @abstractmethod
    def sample_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample observed confounders X."""
        pass

    def sample_U(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample U_j | X=x ~ N((1−λ)·ρⱼᵀx, λ²) for each j

        Args:
            X:   observed confounders (n, p_x)
            rng: seeded random generator

        Returns:
            U: (n, p_u) matrix of unobserved confounders
        """
        c = self.config
        n = X.shape[0]
        U = np.zeros((n, c.p_u))
        for j in range(c.p_u):
            mean_j = (1.0 - c.lambda_) * (X @ self.rho[j])
            U[:, j] = rng.normal(loc=mean_j, scale=c.lambda_, size=n)
        return U

    def sample(self, n: int, seed: Optional[int] = None) -> DGPData:
        """Generate n i.i.d. observations.

        Args:
            n:    number of samples
            seed: random seed for reproducibility

        Returns:
            DGPData with ground-truth propensity quantities included
        """
        rng = np.random.default_rng(seed)
        X = self.sample_X(n, rng)
        U = self.sample_U(X, rng)
        e_nominal = self.nominal_propensity(X)
        e_true = self.true_propensity(X, U)
        T = rng.binomial(1, e_true).astype(float)

        return DGPData(
            X=X,
            T=T,
            U=U,
            e_true=e_true,
            e_nominal=e_nominal,
        )
