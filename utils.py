# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# ============================================================
# 1. Data Generation
# ============================================================

def simulate_dgp(
    T_train,
    T_test,
    c=10.0,
    b_star=0.2,
    seed=1
):
    """
    Simulate the finite-sample DGP:

        R_{t+1} = S_t' beta* + epsilon_{t+1}

    where
        P / T_train = c,
        S_t ~ N(0, I_P),
        epsilon_{t+1} ~ N(0, 1),

    and beta* is dense and normalized such that

        ||beta*||_2^2 = b_star.

    Returns
    -------
    S_train   : ndarray, shape (T_train, P)
    R_train   : ndarray, shape (T_train,)
    S_test    : ndarray, shape (T_test, P)
    R_test    : ndarray, shape (T_test,)
    beta_star : ndarray, shape (P,)
    """

    rng = np.random.default_rng(seed)

    # DGP complexity: c = P / T_train
    P = int(c * T_train)
    T_total = T_train + T_test

    # Generate signals: S_t ~ N(0, I_P)
    S = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(T_total, P)
    )

    # Generate dense isotropic beta*
    v = rng.normal(
        loc=0.0,
        scale=1.0,
        size=P
    )

    beta_star = (
        np.sqrt(b_star)
        * v
        / np.linalg.norm(v)
    )

    assert np.isclose(
        np.linalg.norm(beta_star) ** 2,
        b_star
    )

    # Generate returns
    epsilon = rng.normal(
        loc=0.0,
        scale=1.0,
        size=T_total
    )

    R = S @ beta_star + epsilon

    # Fixed random permutation for partial observability
    permutation = rng.permutation(P)

    S = S[:, permutation]
    beta_star = beta_star[permutation]

    # Train/test split
    S_train = S[:T_train]
    R_train = R[:T_train]

    S_test = S[T_train:]
    R_test = R[T_train:]

    return (
        S_train,
        R_train,
        S_test,
        R_test,
        beta_star
    )

# ============================================================
# 2. Model Fitting
# ============================================================

def train_model(
    X_train,
    y_train,
    X_test,
    T_train,
    z,
    l1_ratio=None,
    model_type="ridge"
):
    """
    Fit Ridge or Lasso and return OOS predictions
    and estimated coefficients.

    For z = 0, use the Moore-Penrose pseudoinverse.

    Parameterization
    ----------------
    Ridge:
        sklearn alpha = T_train * z

    Lasso:
        sklearn alpha = z

    """

    # --------------------------------------------------------
    # Ridgeless / unregularized case
    # --------------------------------------------------------

    if z == 0:

        beta_hat = (
            np.linalg.pinv(X_train)
            @ y_train
        )

    # --------------------------------------------------------
    # Regularized models
    # --------------------------------------------------------

    else:

        if model_type == "ridge":

            model = Ridge(
                alpha=T_train * z,
                fit_intercept=False
            )

        elif model_type == "lasso":

            model = Lasso(
                alpha=z,
                fit_intercept=False,
                max_iter=10000
            )
        elif model_type == "elasticnet":
            
            model = ElasticNet(
                alpha=z,
                l1_ratio=l1_ratio,
                fit_intercept=False,
                max_iter=10000
            )
        else:

            raise ValueError(
                f"Unknown model_type: {model_type}"
            )

        model.fit(
            X_train,
            y_train
        )

        beta_hat = model.coef_

    # Genuine out-of-sample prediction
    y_hat_test = X_test @ beta_hat

    return y_hat_test, beta_hat

# ============================================================
# 3. Evaluation Metrics
# ============================================================

def compute_paper_r2(
    y_test,
    y_hat_test
):
    """
    Finite-sample analogue of KMZ out-of-sample R^2:

        R^2 = 1 - MSE / E[R^2]

    where the null prediction is zero.
    """

    # OOS prediction MSE
    mse = np.mean(
        (y_test - y_hat_test) ** 2
    )

    # Null MSE from zero-return prediction
    null_mse = np.mean(
        y_test ** 2
    )

    r2 = 1.0 - mse / null_mse

    return r2

def compute_metrics(
    y_test,
    y_hat_test,
    params=None
):
    """
    Compute the evaluation metrics used in the project.

    Returns
    -------
    r2
    timing_return
    sharpe_ratio
    parameter_size
    """

    # OOS R^2
    r2 = compute_paper_r2(
        y_test,
        y_hat_test
    )

    # Timing strategy returns
    timing_returns = (
        y_hat_test * y_test
    )

    # Expected timing return
    timing_return = np.mean(
        timing_returns
    )

    # Sharpe ratio
    second_moment = np.mean(
        timing_returns ** 2
    )

    if second_moment == 0:

        sharpe_ratio = np.nan

    else:

        sharpe_ratio = (
            timing_return
            / np.sqrt(second_moment)
        )

    # Parameter size
    if params is None:

        parameter_size = np.nan

    else:

        parameter_size = np.sum(
            np.asarray(params) ** 2
        )

    return (
        r2,
        timing_return,
        sharpe_ratio,
        parameter_size
    )

# ============================================================
# 4. Parameter-Grid Experiments
# ============================================================

def parameter_grid(
    S_train,
    R_train,
    S_test,
    R_test,
    cq_grid,
    z_grid=None,
    model_type="ridge",
    l1_ratio_grid=None
):
    """
    Evaluate model performance over parameter grids.

    Ridge / Lasso:
        vary c_q and z

    Elastic Net:
        vary c_q and l1_ratio while holding z fixed
    """

    P = S_train.shape[1]
    T_train = S_train.shape[0]

    results = []

    for c_q in cq_grid:

        # Number of observed features
        P1 = int(round(c_q * T_train))
        P1 = min(P1, P)
        q = P1 / P

        X_train = S_train.iloc[:, :P1]
        X_test = S_test.iloc[:, :P1]

        y_train = R_train
        y_test = R_test

        # ====================================================
        # Elastic Net
        # ====================================================

        for z in z_grid:

            if model_type == "elasticnet" and l1_ratio_grid is not None:

                for l1_ratio in l1_ratio_grid:

                    y_hat_test, beta_hat = train_model(
                        X_train,
                        y_train,
                        X_test,
                        T_train,
                        z,
                        l1_ratio=l1_ratio,
                        model_type="elasticnet"
                    )

                    (
                        r2,
                        timing_return,
                        sharpe_ratio,
                        parameter_size
                    ) = compute_metrics(
                        y_test,
                        y_hat_test,
                        beta_hat
                    )

                    results.append({
                        "c_q": c_q,
                        "q": q,
                        "P1": P1,
                        "z": z,
                        "l1_ratio": l1_ratio,
                        "r2": r2,
                        "timing_return": timing_return,
                        "sharpe_ratio": sharpe_ratio,
                        "parameter_size": parameter_size
                    })

            else:

                y_hat_test, beta_hat = train_model(
                    X_train,
                    y_train,
                    X_test,
                    T_train,
                    z,
                    model_type=model_type
                )

                (
                    r2,
                    timing_return,
                    sharpe_ratio,
                    parameter_size
                ) = compute_metrics(
                    y_test,
                    y_hat_test,
                    beta_hat
                )

                results.append({
                    "c_q": c_q,
                    "q": q,
                    "P1": P1,
                    "z": z,
                    "r2": r2,
                    "timing_return": timing_return,
                    "sharpe_ratio": sharpe_ratio,
                    "parameter_size": parameter_size
                })


    return pd.DataFrame(results)

# ============================================================
# 5. Plotting
# ============================================================

def plot_metric_vs_cq(
    df,
    metric="r2",
    y_min=None,
    y_max=None,
    plot_title=None,
    group_param="z"
):
    """
    Plot a selected metric against observed complexity c_q
    for different values of z or l1_ratio.
    """

    metric_labels = {
        "r2": r"$R^2$",
        "timing_return": "Timing Return",
        "sharpe_ratio": "Sharpe Ratio",
        "parameter_size": "Parameter Size",
        "mse": "MSE"
    }

    param_labels = {
        "z": "z",
        "l1_ratio": r"$\ell_1$ ratio"
    }

    label = metric_labels.get(
        metric,
        metric
    )

    param_label = param_labels.get(
        group_param,
        group_param
    )

    plt.figure(figsize=(10, 6))

    # Values used for different curves
    param_values = sorted(
        df[group_param].unique()
    )

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "black"
    ]

    for i, value in enumerate(param_values):

        subset = (
            df[df[group_param] == value]
            .sort_values("c_q")
        )

        plt.plot(
            subset["c_q"],
            subset[metric],
            color=colors[i % len(colors)],
            label=f"{param_label} = {value:g}"
        )

    # Interpolation threshold
    plt.axvline(
        x=1,
        color="dimgray",
        linestyle="--",
        label=r"$c_q=1$ (interpolation threshold)"
    )

    # Optional y-axis limits
    plt.ylim(
        bottom=y_min,
        top=y_max
    )

    plt.xlabel(
        r"Observed Complexity $c_q=P_1/T_{\mathrm{tr}}$"
    )

    plt.ylabel(label)

    if plot_title is not None:
        plt.title(plot_title)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs(
        "plots",
        exist_ok=True
    )

    if plot_title is not None:
        plt.savefig(
            f"plots/{plot_title}.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()