import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plot_style import COL_WIDTH, apply_paper_style, figsize, fill_band, plot_band, save_figure
from src.data_generation.quadratic import QuadraticDGP, QuadraticDGPConfig
from src.analysis.informal_benchmarking import InformalBenchmarking
from src.propensity_models.logistic import LogisticPropensityEstimator, QuadraticLogisticPropensityEstimator

# alpha is the misspecification strength (the swept quadratic coefficient),
# distinct from Gamma_IB, the IB sensitivity estimate
ALPHA_VALUES = np.linspace(0.0, 3.0, 31)
ALPHA_DIAGNOSTIC_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
N_SAMPLES = 5000
M_TRIALS = 100
N_FOLDS = 5
SEED_BASE = 42

DATA_DIR = os.path.join('results', 'data')
LOO_CSV = 'exp01_quadratic_loo.csv'
PROB_SQUISH_CSV = 'exp01_prob_squish.csv'

# Grid on which the propensity-score densities are evaluated and stored, so the
# squish diagnostic can be redrawn on --replot without rerunning the sweep
PROB_GRID = np.linspace(0, 1, 200)

def run_scaling_experiment(
    dgp_class,
    config_class,
    alpha_values=ALPHA_VALUES,
    n_samples=N_SAMPLES,
    m_trials=M_TRIALS,
    n_folds=N_FOLDS,
    seed_base=SEED_BASE,
    leave_multiple_out=False,
    desc="Scaling Alpha"
):
    """
    Evaluates Gamma_IB for misspecified vs correctly specified models across the alpha sweep.
    Also collects diagnostics: coefficient attenuation and probability squish.
    """
    results = []
    prob_diagnostics = {a: {"linear": [], "quad": []} for a in ALPHA_DIAGNOSTIC_VALUES}

    for a_val in tqdm(alpha_values, desc=desc):
        trial_gammas_linear = []
        trial_gammas_quad = []
        trial_coefs_linear = [] # Mean abs(coef) for X2-X5

        config = config_class(
            p_x=5,
            p_u=2,
            lambda_=0.6,
            alpha=a_val
        )

        allowed_indices = list(range(1, config.p_x))
        max_k = 2 if leave_multiple_out else 1
        subsets = []
        for k in range(1, max_k + 1):
            subsets.extend(combinations(allowed_indices, k))

        for m in range(m_trials):
            dgp = dgp_class(config)
            data = dgp.sample(n=n_samples, seed=seed_base + m + 100)

            # Misspecified (linear) model
            ib_linear = InformalBenchmarking(
                estimator_factory=LogisticPropensityEstimator,
                n_folds=n_folds,
                verbose=False,
                random_state=seed_base + m
            )
            res_linear = ib_linear.run(data.X, data.T, subsets)
            trial_gammas_linear.append(res_linear.gamma_high)

            fold_coefs = [est.coefficients[1:5] for est in res_linear.full_estimators]
            trial_coefs_linear.append(np.mean(np.abs(fold_coefs)))

            # Correctly specified (quadratic) model
            ib_quad = InformalBenchmarking(
                estimator_factory=QuadraticLogisticPropensityEstimator,
                n_folds=n_folds,
                verbose=False,
                random_state=seed_base + m
            )
            res_quad = ib_quad.run(data.X, data.T, subsets)
            trial_gammas_quad.append(res_quad.gamma_high)

            # Collect predicted scores at the diagnostic alphas
            if any(np.isclose(a_val, a_diag) for a_diag in ALPHA_DIAGNOSTIC_VALUES):
                a_key = next(a for a in ALPHA_DIAGNOSTIC_VALUES if np.isclose(a_val, a))
                prob_diagnostics[a_key]["linear"].extend(res_linear.e_full)
                prob_diagnostics[a_key]["quad"].extend(res_quad.e_full)

        results.append({
            "alpha_dgp": a_val,
            "gamma_linear_mean": np.mean(trial_gammas_linear),
            "gamma_linear_std": np.std(trial_gammas_linear),
            "gamma_quad_mean": np.mean(trial_gammas_quad),
            "gamma_quad_std": np.std(trial_gammas_quad),
            "coef_linear_mean": np.mean(trial_coefs_linear),
            "coef_linear_std": np.std(trial_coefs_linear)
        })
    return pd.DataFrame(results), prob_diagnostics

def plot_gamma_results(df, filename, xlabel=r'Non-linearity Parameter $\alpha$', linear_label=r'Misspecified (Linear) $\hat{\Gamma}_{IB}$', paper=True):
    """Plot Gamma_IB vs alpha for misspecified vs correct models (main figure)."""
    plt.figure(figsize=figsize(COL_WIDTH))
    plot_band(df['alpha_dgp'], df['gamma_quad_mean'], df['gamma_quad_std'],
              color='forestgreen',
              label=r'Correctly Specified (Quadratic) $\hat{\Gamma}_{IB}$')
    plot_band(df['alpha_dgp'], df['gamma_linear_mean'], df['gamma_linear_std'],
              color='steelblue', label=linear_label)
    plt.xlabel(xlabel)
    plt.ylabel(r'Estimated Sensitivity Parameter $\hat{\Gamma}_{IB}$')
    plt.legend()
    save_figure(filename, paper=paper)

def plot_coefficient_attenuation(df, filename='exp01_coef_attenuation.png'):
    """Plot average absolute coefficients for misspecified model (main figure)."""
    plt.figure(figsize=figsize(COL_WIDTH))
    plot_band(df['alpha_dgp'], df['coef_linear_mean'], df['coef_linear_std'],
              color='firebrick')
    plt.xlabel(r'Non-linearity Parameter $\alpha$')
    plt.ylabel(r'Average $|\beta|$ for $X_2 \dots X_5$')
    save_figure(filename, paper=True)

def compute_prob_squish_curves(prob_diagnostics):
    """Evaluate the propensity-score KDEs on PROB_GRID for each diagnostic alpha.

    Stores the finished density curves rather than the raw probabilities, so the
    squish diagnostic can be redrawn on --replot without rerunning the sweep.

    Args:
        prob_diagnostics: dict of alpha to per-model lists of predicted scores

    Returns:
        Long-format DataFrame with columns alpha, x, density_quad, density_linear
    """
    from scipy.stats import gaussian_kde

    rows = []
    for a_val, data in prob_diagnostics.items():
        if not data["linear"]:
            continue
        y_quad = gaussian_kde(data['quad'])(PROB_GRID)
        y_linear = gaussian_kde(data['linear'])(PROB_GRID)
        for x, yq, yl in zip(PROB_GRID, y_quad, y_linear):
            rows.append({"alpha": a_val, "x": x, "density_quad": yq, "density_linear": yl})
    return pd.DataFrame(rows)

def plot_probability_squish(curves, filename_prefix='exp01_prob_squish'):
    """Plot the predicted-probability densities for the selected alphas.

    Every frame shares one fixed y-axis and a fixed legend position so the panels are
    directly comparable and play as a clean flip-book animation: the density visibly
    collapses toward 0.5 as alpha grows, without the axes or legend moving between
    frames. The alpha=3.0 panel is also promoted to a paper figure.

    Args:
        curves: DataFrame from compute_prob_squish_curves
        filename_prefix: prefix for the per-alpha output files
    """
    ymax = 1.05 * max(curves["density_quad"].max(), curves["density_linear"].max())
    for a_val, sub in curves.groupby("alpha"):
        sub = sub.sort_values("x")
        x = sub["x"].to_numpy()
        y_quad = sub["density_quad"].to_numpy()
        y_linear = sub["density_linear"].to_numpy()

        plt.figure(figsize=figsize(COL_WIDTH))

        # Correctly Specified
        fill_band(x, 0.0, y_quad, color='forestgreen')
        plt.plot(x, y_quad, label='Correctly Specified', color='forestgreen', lw=1.5, zorder=2)

        # Misspecified
        fill_band(x, 0.0, y_linear, color='steelblue')
        plt.plot(x, y_linear, label='Misspecified', color='steelblue', lw=1.5, zorder=2)

        plt.xlabel(r'Predicted Propensity Score $\hat{e}(X)$')
        plt.ylabel('Density')
        plt.legend(loc='upper left')
        plt.xlim(0, 1)
        plt.ylim(0, ymax)

        # Promote the alpha=3.0 panel to a paper figure
        is_paper = abs(a_val - 3.0) < 1e-9

        save_figure(f"{filename_prefix}_alpha_{a_val}.png", paper=is_paper)

def save_csv(df, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, name), index=False)

def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: quadratic misspecification")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Skip the sweep and regenerate the main plots from the saved CSV",
    )
    args = parser.parse_args()

    apply_paper_style()

    if args.replot:
        print("Replotting Experiment 1 from saved CSV...")
        df_loo = load_csv(LOO_CSV)
        plot_gamma_results(df_loo, filename='exp01_quadratic_loo.png')
        plot_coefficient_attenuation(df_loo)
        if os.path.exists(os.path.join(DATA_DIR, PROB_SQUISH_CSV)):
            plot_probability_squish(load_csv(PROB_SQUISH_CSV))
        else:
            print("Note: probability-squish curves not found, run a full sweep once to generate them")
    else:
        print("Running Experiment: Quadratic Scaling (Leave-One-Out)...")
        df_loo, prob_diags = run_scaling_experiment(QuadraticDGP, QuadraticDGPConfig, desc="Quadratic LOO")
        save_csv(df_loo, LOO_CSV)
        squish_curves = compute_prob_squish_curves(prob_diags)
        save_csv(squish_curves, PROB_SQUISH_CSV)

        plot_gamma_results(df_loo, filename='exp01_quadratic_loo.png')
        print("Generating Diagnostic Plots...")
        plot_coefficient_attenuation(df_loo)
        plot_probability_squish(squish_curves)

    print("\nDone.")
