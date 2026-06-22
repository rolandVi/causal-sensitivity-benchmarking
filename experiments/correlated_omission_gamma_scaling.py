import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plot_style import COL_WIDTH, apply_paper_style, figsize, plot_band, save_figure
from src.analysis.informal_benchmarking import InformalBenchmarking
from src.data_generation.correlated import UniformProxyDGP, UniformProxyDGPConfig
from src.propensity_models.logistic import LogisticPropensityEstimator

# alpha is the misspecification strength (swept), distinct from Gamma_IB, the IB sensitivity estimate
ALPHA_VALUES = np.linspace(0.0, 2.0, 31)
RHO = 0.5
N_SAMPLES = 5000
M_TRIALS = 100
N_FOLDS = 5
SEED_BASE = 42

# Leave-one-out IB omission subsets, indexed into each model's own design matrix.
# Correct model keeps all five covariates; misspecified model has already dropped X2, so
# its column 0 is X1 and it has four columns (X1, X3, X4, X5).
CORRECT_SUBSETS = [(0,), (1,), (2,), (3,), (4,)]
MIS_SUBSETS = [(0,), (1,), (2,), (3,)]

# Rho-sweep parameters: at fixed (large) alpha, sweep the X1-X2 copula correlation to test
# the prediction that the misspecified X1 coefficient saturates at a rho-determined ceiling,
# independent of alpha. Coefficient-only (no IB), so it is cheap; fewer trials are enough.
RHO_SWEEP_VALUES = np.linspace(0.0, 0.9, 16)
RHO_SWEEP_ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0]
RHO_SWEEP_TRIALS = 50

DATA_DIR = os.path.join("results", "data")


def _loo_max(gammas, n_features):
    """Leave-one-out Gamma_IB = max over single-covariate omissions."""
    return max(gammas[(i,)] for i in range(n_features))


def run_correlated_experiment(
    alpha_values=ALPHA_VALUES,
    rho=RHO,
    n_samples=N_SAMPLES,
    m_trials=M_TRIALS,
    n_folds=N_FOLDS,
    seed_base=SEED_BASE,
    desc="Alpha Scaling Sweep",
):
    """Evaluate proxy inflation and IB Gamma for the misspecified (omitted correlated
    covariate) vs. the correctly specified model across an alpha sweep.

    For each alpha we record, averaged over m_trials Monte-Carlo trials:
      * the leave-one-out Gamma_IB of both models,
      * the per-covariate drivers (omit X1 / omit X2) behind each Gamma_IB,
      * the fitted X1 coefficient of both models (the proxy-inflation diagnostic).
    """
    factory = LogisticPropensityEstimator
    results = []

    for alpha in tqdm(alpha_values, desc=desc):
        g_correct_loo = []
        g_mis_loo = []
        coef_x1_correct = []
        coef_x1_mis = []

        config = UniformProxyDGPConfig(alpha=alpha, rho=rho)

        for m in range(m_trials):
            dgp = UniformProxyDGP(config)
            data = dgp.sample(n=n_samples, seed=seed_base + m)

            # Correct model: leave-one-out over all covariates
            ib_correct = InformalBenchmarking(
                estimator_factory=factory,
                n_folds=n_folds,
                verbose=False,
                random_state=seed_base + m,
            )
            res_correct = ib_correct.run(data.X, data.T, CORRECT_SUBSETS)
            gc = res_correct.gammas
            g_correct_loo.append(_loo_max(gc, data.X.shape[1]))
            coef_x1_correct.append(
                np.mean([est.coefficients[0] for est in res_correct.full_estimators])
            )

            # Misspecified model drops X2 (index 1), so column 0 is X1
            X_mis = data.X[:, [0, 2, 3, 4]]
            ib_mis = InformalBenchmarking(
                estimator_factory=factory,
                n_folds=n_folds,
                verbose=False,
                random_state=seed_base + m,
            )
            res_mis = ib_mis.run(X_mis, data.T, MIS_SUBSETS)
            gm = res_mis.gammas
            g_mis_loo.append(_loo_max(gm, X_mis.shape[1]))
            coef_x1_mis.append(
                np.mean([est.coefficients[0] for est in res_mis.full_estimators])
            )

        def stats(name, values):
            return {f"{name}_mean": np.mean(values), f"{name}_std": np.std(values)}

        row = {"alpha": alpha}
        row.update(stats("gamma_correct", g_correct_loo))
        row.update(stats("gamma_mis", g_mis_loo))
        row.update(stats("coef_x1_correct", coef_x1_correct))
        row.update(stats("coef_x1_mis", coef_x1_mis))
        results.append(row)

    return pd.DataFrame(results)


def run_rho_sweep(
    rho_values=RHO_SWEEP_VALUES,
    alphas=RHO_SWEEP_ALPHAS,
    n_samples=N_SAMPLES,
    m_trials=RHO_SWEEP_TRIALS,
    seed_base=SEED_BASE,
    desc="Rho Sweep",
):
    """Sweep the X1-X2 copula correlation rho at fixed alpha values.

    Tests the saturation mechanism: as alpha grows the misspecified X1 coefficient does
    not diverge but converges to a ceiling set by rho (how well X1 proxies the sign of X2).
    We therefore expect the coefficient to (a) rise with rho and (b) stack toward a common
    rho-determined curve as alpha increases. Only the fitted X1 coefficient is needed, so we
    fit each model once on the full sample rather than running the full IB cross-fitting.

    Returns a long-format DataFrame with one row per (alpha, rho).
    """
    rows = []
    for alpha in tqdm(alphas, desc=desc):
        for rho in rho_values:
            coef_mis = []
            coef_correct = []
            config = UniformProxyDGPConfig(alpha=alpha, rho=rho)
            for m in range(m_trials):
                dgp = UniformProxyDGP(config)
                data = dgp.sample(n=n_samples, seed=seed_base + m)

                # Misspecified model drops X2 (index 1); column 0 is X1
                est_mis = LogisticPropensityEstimator().fit(
                    data.X[:, [0, 2, 3, 4]], data.T
                )
                coef_mis.append(est_mis.coefficients[0])

                # Correct model keeps all covariates; X1 coefficient is the baseline
                est_correct = LogisticPropensityEstimator().fit(data.X, data.T)
                coef_correct.append(est_correct.coefficients[0])

            rows.append(
                {
                    "alpha": alpha,
                    "rho": rho,
                    "coef_x1_mis_mean": np.mean(coef_mis),
                    "coef_x1_mis_std": np.std(coef_mis),
                    "coef_x1_correct_mean": np.mean(coef_correct),
                    "coef_x1_correct_std": np.std(coef_correct),
                }
            )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

CORRECT_COLOR = "forestgreen"
MIS_COLOR = "steelblue"


def plot_proxy_inflation(df, filename="exp02_proxy_inflation.png"):
    """Estimated X1 coefficient vs alpha for both models."""
    plt.figure(figsize=figsize(COL_WIDTH))
    plot_band(
        df["alpha"], df["coef_x1_correct_mean"], df["coef_x1_correct_std"],
        color=CORRECT_COLOR, label="Correctly Specified (All X)",
    )
    plot_band(
        df["alpha"], df["coef_x1_mis_mean"], df["coef_x1_mis_std"],
        color=MIS_COLOR, label="Misspecified (Dropped $X_2$)",
    )
    plt.xlabel(r"Strength of Omitted Covariate ($\alpha$)")
    plt.ylabel(r"Estimated Coefficient for $X_1$ ($\hat{\beta}_1$)")
    plt.legend()
    save_figure(filename, paper=True)


def plot_ib_consequence(df, filename="exp02_ib_consequence.png"):
    """Leave-one-out Gamma_IB vs alpha for both models."""
    plt.figure(figsize=figsize(COL_WIDTH))
    plot_band(
        df["alpha"], df["gamma_correct_mean"], df["gamma_correct_std"],
        color=CORRECT_COLOR, label="Correctly Specified (All X)",
    )
    plot_band(
        df["alpha"], df["gamma_mis_mean"], df["gamma_mis_std"],
        color=MIS_COLOR, label="Misspecified (Dropped $X_2$)",
    )
    plt.xlabel(r"Strength of Omitted Covariate ($\alpha$)")
    plt.ylabel(r"Estimated Sensitivity Parameter $\hat{\Gamma}_{IB}$")
    plt.legend()
    save_figure(filename, paper=True)


def plot_rho_saturation(df, filename="exp02_rho_saturation.png"):
    """Misspecified X1 coefficient vs rho, one curve per alpha.

    Confirms the saturation mechanism: the proxy-inflation ceiling rises with rho, and the
    per-alpha curves stack toward a common rho-determined asymptote as alpha grows (the
    coefficient stops responding to alpha once X1's information about sign(X2) is exhausted).
    """
    plt.figure(figsize=figsize(COL_WIDTH))
    alphas_sorted = sorted(df["alpha"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(alphas_sorted)))
    for color, alpha in zip(colors, alphas_sorted):
        sub = df[df["alpha"] == alpha].sort_values("rho")
        plot_band(
            sub["rho"], sub["coef_x1_mis_mean"], sub["coef_x1_mis_std"],
            color=color,
            label=rf"misspecified, $\alpha={alpha:g}$",
        )
    # Correct-model baseline (X1 coefficient stays near its true value, flat in rho)
    base = df[df["alpha"] == max(df["alpha"].unique())].sort_values("rho")
    plt.plot(
        base["rho"], base["coef_x1_correct_mean"],
        "--", color="black", label=r"correct $\hat{\beta}_1$ (baseline)",
    )
    plt.xlabel(r"Copula correlation $\rho$ between $X_1$ and $X_2$")
    plt.ylabel(r"Estimated Coefficient for $X_1$ ($\hat{\beta}_1$)")
    plt.legend()
    save_figure(filename, paper=True)


ALPHA_CSV = "exp02_correlated.csv"
RHO_CSV = "exp02_rho_sweep.csv"


def _save_csv(df, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, name), index=False)


def _load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def make_plots(df, df_rho):
    plot_proxy_inflation(df)
    plot_ib_consequence(df)
    plot_rho_saturation(df_rho)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: correlated omission")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Skip the sweeps and regenerate the plots from the saved CSVs",
    )
    args = parser.parse_args()

    apply_paper_style()

    if args.replot:
        print("Replotting Experiment 2 from saved CSVs...")
        df = _load_csv(ALPHA_CSV)
        df_rho = _load_csv(RHO_CSV)
    else:
        print("Running Experiment 2: Correlated Omission (alpha sweep)...")
        df = run_correlated_experiment(desc="Alpha sweep")
        _save_csv(df, ALPHA_CSV)

        print("Running Experiment 2: Rho sweep (saturation ceiling)...")
        df_rho = run_rho_sweep(desc="Rho sweep")
        _save_csv(df_rho, RHO_CSV)

    print("Generating plots...")
    make_plots(df, df_rho)

    print("\nDone.")
