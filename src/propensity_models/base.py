from abc import ABC, abstractmethod

import numpy as np


class BasePropensityEstimator(ABC):
    """Abstract base class for propensity score estimators"""

    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray) -> "BasePropensityEstimator":
        """Fit the propensity model

        Args:
            X: observed confounders (n, p_x)
            T: binary treatment labels {0, 1} (n,)

        Returns:
            self, to allow method chaining
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate P(T=1 | X) for each row of X

        Args:
            X: observed confounders (n, p_x)

        Returns:
            Estimated propensity scores (n,) in (0, 1)
        """
        ...
