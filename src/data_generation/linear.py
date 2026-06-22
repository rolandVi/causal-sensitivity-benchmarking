from dataclasses import dataclass

import numpy as np

from .base import BaseDGP, BaseDGPConfig, DGPData


@dataclass
class LinearDGPConfig(BaseDGPConfig):
    """Hyperparameters for the Linear DGP

    Attributes:
        p_x:      number of observed confounders X
        p_u:      number of unobserved confounders U
        lambda_:  controls the X–U correlation
                  lambda_=1  →  X and U are uncorrelated (U is pure noise)
                  lambda_=0.2 →  X and U are highly correlated
    """
    pass


class LinearDGP(BaseDGP):
    """Linear DGP for binary treatment.

    Data generating process
    -----------------------
    Observed confounders:
        X ~ Uniform(-1, 1)^p_x

    Unobserved confounders (j = 1 … p_u), correlated with X via lambda:
        U_j | X=x ~ N( (1-λ) · ρ_j^T x,  λ² )

    True propensity score:
        e(X, U) = logistic( β_X^T X + β_U^T U )

    Nominal propensity score (no U, estimable from data):
        e(X) = logistic( β_X^T X )

    Treatment:
        T | X, U ~ Bernoulli( e(X, U) )

    Parameters β_X = 0.3, β_U = 0.3, ρ = 1.75 are fixed scalars
    broadcast across their respective dimensions.
    """

    def sample_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """X ~ Uniform(-1, 1)^p_x"""
        return rng.uniform(-1.0, 1.0, size=(n, self.config.p_x))
