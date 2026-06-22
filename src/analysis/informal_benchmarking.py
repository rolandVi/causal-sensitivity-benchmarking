from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from ..propensity_models.base import BasePropensityEstimator


@dataclass
class IBResult:
    """Results of an informal benchmarking run.

    Attributes:
        gammas:          mapping from omitted feature indices (tuple) to Γ̂ for that subset
        gamma_low:       Γ̂_low  = min over all subsets
        gamma_high:      Γ̂_high = max over all subsets (= Γ̂_IB, the recommended estimate)
        e_full:          cross-fitted propensity scores from the full model
        full_estimators: list of fitted estimators from the full model cross-fitting
    """

    gammas: dict[tuple[int, ...], float]
    gamma_low: float
    gamma_high: float
    e_full: Optional[np.ndarray] = None
    full_estimators: Optional[list[BasePropensityEstimator]] = None


class InformalBenchmarking:
    """Informal benchmarking for the Marginal Sensitivity Model (binary treatment).

    Estimates the sensitivity parameter Γ by treating each observed confounder
    (or group of confounders) as a proxy for potential unobserved confounding.
    Implements Algorithm 1 from Baitairian et al. (2025) for both the
    leave-one-out and leave-multiple-out variants.

    Propensity scores are estimated via K-fold cross-fitting to avoid
    overfitting: for each observation j, both ê_full(X_j) and ê_reduced(X_j)
    are predicted by a model that was never trained on X_j.
    The same fold split is reused for all models in a single run.

    Args:
        estimator_factory: callable that returns a fresh BasePropensityEstimator
        n_folds:           number of cross-fitting folds (default 5)
        random_state:      seed for the fold split
        clip_eps:          propensity scores are clipped to [clip_eps, 1 - clip_eps]
                           before computing odds ratios to avoid division by zero
    """

    def __init__(
        self,
        estimator_factory: Callable[[], BasePropensityEstimator],
        n_folds: int = 5,
        random_state: Optional[int] = None,
        clip_eps: float = 1e-6,
        verbose: bool = True,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.n_folds = n_folds
        self.random_state = random_state
        self.clip_eps = clip_eps
        self.verbose = verbose

    def leave_one_out(self, X: np.ndarray, T: np.ndarray) -> IBResult:
        """Leave-one-out informal benchmarking.

        For each feature i, treats X(i) as unobserved and estimates the
        confounding strength it would imply if it were hidden.

        Args:
            X: observed confounders (n, p_x)
            T: binary treatment labels {0, 1} (n,)

        Returns:
            IBResult with one Γ̂ entry per feature.
        """
        subsets = [(i,) for i in range(X.shape[1])]
        return self.run(X, T, subsets)

    def leave_multiple_out(
        self, X: np.ndarray, T: np.ndarray, max_k: int = 2
    ) -> IBResult:
        """Leave-multiple-out informal benchmarking.

        For all subsets of size 1 up to max_k, treats the omitted subset as
        unobserved and estimates the implied confounding strength.

        Args:
            X:     observed confounders (n, p_x)
            T:     binary treatment labels {0, 1} (n,)
            max_k: maximum number of features to leave out simultaneously.
                   Must satisfy 1 <= max_k < p_x. Defaults to 2.

        Returns:
            IBResult with one Γ̂ entry per omission subset.
        """
        p = X.shape[1]
        if not (1 <= max_k < p):
            raise ValueError(f"max_k must satisfy 1 <= max_k < p_x ({p}), got {max_k}")
        subsets = [
            combo
            for k in range(1, max_k + 1)
            for combo in combinations(range(p), k)
        ]
        return self.run(X, T, subsets)

    def run(
        self,
        X: np.ndarray,
        T: np.ndarray,
        subsets: Iterable[tuple[int, ...]],
    ) -> IBResult:
        """Build folds once, cross-fit the full model, then compute Γ̂ per subset."""
        folds = self.make_folds(T)
        e_full, full_estimators = self.cross_fit_predict(
            X, T, folds, desc="full model", return_estimators=True
        )

        subset_list = list(subsets)
        gammas: dict[tuple[int, ...], float] = {}
        pbar = tqdm(subset_list, disable=not self.verbose, desc="subsets", unit="subset")
        for omit in pbar:
            label = "{" + ",".join(f"X{i+1}" for i in omit) + "}"
            pbar.set_postfix(omit=label)
            gammas[omit] = self.gamma_for_omission(
                e_full, X, T, omit, folds
            )

        gamma_low = float(min(gammas.values()))
        gamma_high = float(max(gammas.values()))
        return IBResult(
            gammas=gammas,
            gamma_low=gamma_low,
            gamma_high=gamma_high,
            e_full=e_full,
            full_estimators=full_estimators,
        )

    def make_folds(self, T: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return stratified K-fold (train, test) index pairs."""
        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        return list(cv.split(T, T))

    def cross_fit_predict(
        self,
        X: np.ndarray,
        T: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        desc: str = "cross-fit",
        return_estimators: bool = False,
    ) -> tuple[np.ndarray, list[BasePropensityEstimator]] | np.ndarray:
        """Out-of-fold propensity score predictions using pre-computed folds."""
        scores = np.empty(len(T))
        estimators = []
        pbar = tqdm(
            enumerate(folds),
            total=len(folds),
            disable=not self.verbose,
            desc=f"  {desc}",
            unit="fold",
            leave=False,
        )
        for fold_i, (train_idx, test_idx) in pbar:
            pbar.set_postfix(fold=f"{fold_i+1}/{len(folds)}")
            est = self.estimator_factory()
            est.fit(X[train_idx], T[train_idx])
            scores[test_idx] = est.predict_proba(X[test_idx])
            if return_estimators:
                estimators.append(est)

        scores = np.clip(scores, self.clip_eps, 1.0 - self.clip_eps)
        if return_estimators:
            return scores, estimators
        return scores

    def gamma_for_omission(
        self,
        e_full: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        omit_cols: tuple[int, ...],
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Compute Γ̂ when the features in omit_cols are treated as unobserved.

        Per-sample odds ratio:
            OR_j = [ê(Xj) / (1 - ê(Xj))] / [ê(Xj^{-S}) / (1 - ê(Xj^{-S}))]

        Γ̂_S = max(max_j OR_j, max_j 1/OR_j)
        """
        keep = [i for i in range(X.shape[1]) if i not in omit_cols]
        e_reduced = self.cross_fit_predict(X[:, keep], T, folds)

        or_ratios = (e_full / (1.0 - e_full)) / (e_reduced / (1.0 - e_reduced))

        gamma_plus = float(np.max(or_ratios))
        gamma_minus = float(1.0 / np.min(or_ratios))

        return max(gamma_plus, gamma_minus)
