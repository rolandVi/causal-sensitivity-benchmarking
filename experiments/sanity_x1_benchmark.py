"""Sanity check: what happens if X1 is included in the IB benchmark set
under the misspecified (linear) model

Claim to test:
- The omitted curvature is symmetric in X1, so X1 carries almost no linear
  signal; the misspecified linear model fits its coefficient near zero
- Benchmarking on X1 therefore reads ~no confounding (Gamma_1 ~ 1) for the
  single strongest driver of treatment
- Because Gamma_IB is a max over covariates, adding X1 does not lower the
  headline value
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generation.quadratic import QuadraticDGP, QuadraticDGPConfig
from src.analysis.informal_benchmarking import InformalBenchmarking
from src.propensity_models.logistic import LogisticPropensityEstimator

N = 5000
ALPHA = 3.0
M = 100
SEED_BASE = 42

config = QuadraticDGPConfig(p_x=5, p_u=2, lambda_=0.6, alpha=ALPHA)

coef_abs = []           # mean |coef| per covariate from the misspecified full fit
gamma_per_cov = []      # Gamma_i per covariate (leave-one-out, linear model)

for m in range(M):
    dgp = QuadraticDGP(config)
    data = dgp.sample(n=N, seed=SEED_BASE + m + 100)

    ib = InformalBenchmarking(
        estimator_factory=LogisticPropensityEstimator,
        n_folds=5,
        verbose=False,
        random_state=SEED_BASE + m,
    )
    # leave-one-out over ALL covariates, including X1 (index 0)
    res = ib.leave_one_out(data.X, data.T)

    fold_coefs = np.array([est.coefficients for est in res.full_estimators])
    coef_abs.append(np.mean(np.abs(fold_coefs), axis=0))
    gamma_per_cov.append([res.gammas[(i,)] for i in range(config.p_x)])

coef_abs = np.array(coef_abs)
gamma_per_cov = np.array(gamma_per_cov)

print(f"\nQuadraticDGP, alpha={ALPHA}, N={N}, trials={M}, misspecified LINEAR model")
print("=" * 60)
print("Mean |fitted coefficient| per covariate (linear model):")
for i in range(config.p_x):
    tag = "  <- carries the quadratic term" if i == 0 else ""
    print(f"  X{i+1}: {coef_abs[:, i].mean():.4f} +/- {coef_abs[:, i].std():.4f}{tag}")

print("\nLeave-one-out Gamma_i per covariate (linear model):")
for i in range(config.p_x):
    tag = "  <- X1 benchmark" if i == 0 else ""
    print(f"  X{i+1}: {gamma_per_cov[:, i].mean():.4f} +/- {gamma_per_cov[:, i].std():.4f}{tag}")

gamma_all = gamma_per_cov.max(axis=1)
gamma_excl_x1 = gamma_per_cov[:, 1:].max(axis=1)
print("\nGamma_IB = max over covariates:")
print(f"  including X1: {gamma_all.mean():.4f} +/- {gamma_all.std():.4f}")
print(f"  excluding X1: {gamma_excl_x1.mean():.4f} +/- {gamma_excl_x1.std():.4f}")
print(f"  X1 is the argmax in {np.mean(gamma_per_cov.argmax(axis=1) == 0) * 100:.0f}% of trials")
