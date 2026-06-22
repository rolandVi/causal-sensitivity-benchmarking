# Causal Sensitivity Benchmarking under Propensity Model Misspecification

Code for the bachelor thesis **"When the Propensity Model Is Wrong: Informal Benchmarking and a False Sense of Robustness in Causal Sensitivity Analysis"**.

- **Author:** Roland Vízner
- **Supervisors:** Jesse Krijthe, Matej Havelka

## Overview

Causal conclusions from observational data rely on the assumption that every confounder (a variable that affects both treatment and outcome) is observed. Sensitivity analysis with the **Marginal Sensitivity Model (MSM)** relaxes this assumption through a parameter $\Gamma \ge 1$ that bounds how strongly a hidden confounder could shift an individual's odds of treatment. Choosing a credible $\Gamma$ is hard, so practitioners use **Informal Benchmarking (IB)**: each observed covariate is removed from the propensity model in turn and treated as a stand-in for an unobserved confounder, and the largest resulting odds-ratio shift is taken as a plausible value of $\Gamma$ (denoted $\hat{\Gamma}_{IB}$).

Because IB is read entirely off a fitted propensity model, this thesis asks what happens to it when that model is **misspecified**. Using controlled simulations, it shows that omitting a non-linear term from the propensity model **deflates** $\hat{\Gamma}_{IB}$: the benchmark drifts below its correctly specified value, narrowing the sensitivity interval and making a causal conclusion look more robust to hidden confounding than it really is, which is the more dangerous direction of error.

## Research question

> What is the interaction between the parameter from informal benchmarking and the bound from sensitivity analysis if the propensity model is incorrect?

## Key findings

- Omitting a quadratic term from a parametric propensity model biases $\hat{\Gamma}_{IB}$ **downward**, producing falsely robust sensitivity bounds.
- The mechanism is coefficient attenuation: in logistic regression an omitted term shrinks every fitted coefficient toward zero (non-collapsibility), so the error leaks into the benchmark even when it is measured only on covariates that are individually well specified.
- A practical safeguard: refit the propensity model with a richer, still cross-fitted specification and rerun IB; a rise in $\hat{\Gamma}_{IB}$ signals that the original estimate was deflated.

## Repository structure

```text
├── src/                      # Core library
│   ├── data_generation/      # Synthetic DGPs: LinearDGP (base), QuadraticDGP, UniformProxyDGP
│   ├── propensity_models/    # LogisticPropensityEstimator, QuadraticLogisticPropensityEstimator
│   └── analysis/             # InformalBenchmarking (K-fold cross-fitted, leave-one-out)
├── experiments/              # Reproducible experiment drivers and shared plot style
└── results/                  # Saved data (CSV) and generated plots
```

The thesis write-up, poster, presentation, and background literature are kept out of this public code repository.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the experiments

Run from the repository root:

```powershell
# Experiment 1 - functional-form misspecification (main paper results)
python experiments/quadratic_misspecification_gamma_scaling.py

# Experiment 2 - structural omission of a correlated covariate (appendix)
python experiments/correlated_omission_gamma_scaling.py

# Sanity check - including X1 in the benchmark set
python experiments/sanity_x1_benchmark.py
```

Every sweep is seeded for reproducibility and writes its results to `results/data/`. Pass `--replot` to an experiment script to regenerate its figures from the saved CSVs without rerunning the sweep.

## Library usage

```python
from src.data_generation.quadratic import QuadraticDGP, QuadraticDGPConfig
from src.analysis.informal_benchmarking import InformalBenchmarking
from src.propensity_models.logistic import LogisticPropensityEstimator

# True propensity has a quadratic term in X1; alpha sets its strength
config = QuadraticDGPConfig(p_x=5, p_u=2, lambda_=0.6, alpha=2.0)
data = QuadraticDGP(config).sample(n=5000, seed=42)

# Informal Benchmarking with a (misspecified) linear logistic model
ib = InformalBenchmarking(estimator_factory=LogisticPropensityEstimator, n_folds=5)
result = ib.leave_one_out(data.X, data.T)
print(f"Gamma_IB = {result.gamma_high:.3f}")
```