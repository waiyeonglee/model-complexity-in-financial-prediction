# Task 3: Predictions on three data sets A, B, C
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import parameter_grid, plot_metric_vs_cq, train_model

def prepare_task3_data(S_train, S_test, train_ratio=0.8):
    """
    Prepare data for model training and testing.

    Parameters
    ----------
    S_train : pd.DataFrame
        Training features with 'return' column.
    S_test : pd.DataFrame
        Test features.

    Returns
    -------
    X_train : pd.DataFrame
        Training features without 'return'.
    y_train : pd.Series
        Training target variable (returns).
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target variable (returns).
    X_test : pd.DataFrame
        Test features.
    """
    y = S_train["return"]

    # Keep time index separately
    t_test = S_test["t"].copy()

    # Do not use t or return as predictors
    X = S_train.drop(columns=["t", "return"])
    X_test = S_test.drop(columns=["t"])

    # Check train/test feature alignment
    assert list(X.columns) == list(X_test.columns), \
        "Train and test feature names/order do not match"

    # ============================================================
    # Chronological train-validation split
    # ============================================================

    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    # T_train = X_train.shape[0]

    # Checks
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(X_test.columns)

    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape)

    P = X_train.shape[1]
    T = X_train.shape[0]

    print("cq = P/T =", P / T)

    return X_train, y_train, X_val, y_val, X_test, t_test

def load_task3_datasets(
    dataset_names=("A", "B", "C"),
    train_ratio=0.8,
    data_path="data/"
):
    datasets = {}

    for name in dataset_names:

        print(f"Dataset {name}:")

        S_train = pd.read_csv(
            f"{data_path}pair{name}_train.csv"
        )

        S_test = pd.read_csv(
            f"{data_path}pair{name}_test_features.csv"
        )

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            t_test
        ) = prepare_task3_data(
            S_train,
            S_test,
            train_ratio=train_ratio
        )

        datasets[name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "t_test": t_test
        }

        print()

    return datasets
    
def plot_feature_correlations(datasets):

    correlation_data = {
        name: (
            data["X_train"],
            data["y_train"]
        )
        for name, data in datasets.items()
    }
    
    plt.figure(figsize=(10, 6))

    for dataset_name, (X, y) in correlation_data.items():

        feature_corr_abs = (
            X.corrwith(y)
            .abs()
            .sort_values(ascending=False)
        )

        plt.plot(
            np.arange(1, len(feature_corr_abs) + 1),
            feature_corr_abs.values,
            label=f"Dataset {dataset_name}"
        )

    plt.xlabel("Feature Rank")
    plt.ylabel(r"$|\mathrm{Corr}(X_j,R)|$")

    plt.title(
        "t3_correlations"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "plots/t3_correlations.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def train_ridge_and_lasso(data, dataset_name, cq_grids, z_grid, best_params):

    for model_type in ["ridge", "lasso"]:

        df = parameter_grid(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            cq_grids[dataset_name],
            z_grid,
            model_type=model_type
        )
        if dataset_name == "B" or dataset_name == "C":
            return_y_min = -0.1
            return_y_max = 0.1
        else:
            return_y_min = None
            return_y_max = 0.5

        # Determine the best z and cq values based on the maximum Sharpe ratio
        best_params_df = df.loc[
            df["sharpe_ratio"].idxmax()
        ]

        best_cq = best_params_df["c_q"]
        best_z = best_params_df["z"]

        best_params.append({
            "dataset": dataset_name,
            "model_type": model_type,
            "best_cq": best_cq,
            "best_z": best_z,
            "sharpe_ratio": best_params_df["sharpe_ratio"]
        })
            
        plot_metric_vs_cq(
            df,
            metric="timing_return",
            y_min=return_y_min,
            y_max=return_y_max,
            plot_title=(
                f"t3_{dataset_name}_{model_type}_timing_return"
            ),
            group_param="z"
        )

        plot_metric_vs_cq(
            df,
            metric="sharpe_ratio",
            y_min=-0.2,
            y_max=0.2,
            plot_title=(
                f"t3_{dataset_name}_{model_type}_sharpe_ratio"
            ),
            group_param="z"
        )

def train_elasticnet(data, dataset_name, cq_grids, z_grid, model_type="elasticnet", l1_ratio_grid=None, best_params=None):

    df = parameter_grid(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        cq_grids[dataset_name],
        z_grid = z_grid,
        model_type=model_type,
        l1_ratio_grid=l1_ratio_grid,
    )

    # Determine the best z, cq and l1 values based on the maximum Sharpe ratio
    best_params_df = df.loc[
        df["sharpe_ratio"].idxmax()
    ]

    best_cq = best_params_df["c_q"]
    best_z = best_params_df["z"]
    best_l1_ratio = best_params_df["l1_ratio"]

    best_params.append({
        "dataset": dataset_name,
        "model_type": model_type,
        "best_cq": best_cq,
        "best_z": best_z,
        "best_l1_ratio": best_l1_ratio,
        "sharpe_ratio": best_params_df["sharpe_ratio"]
    })

    # Filter the DataFrame for the best z value
    df_best_z = df[
        df["z"] == best_z
    ].copy()

    plot_metric_vs_cq(
        df_best_z,
        metric="timing_return",
        y_min=-0.1,
        y_max=0.1,
        plot_title=(
            f"t3_{dataset_name}_{model_type}_z={best_z}_timing_return"
        ),
        group_param="l1_ratio"
    )

    plot_metric_vs_cq(
        df_best_z,
        metric="sharpe_ratio",
        plot_title=(
            f"t3_{dataset_name}_{model_type}_z={best_z}_sharpe_ratio"
        ),
        group_param="l1_ratio"
    )

def model_inference(best_params_df, datasets):

    os.makedirs(
        "predictions",
        exist_ok=True
    )
    
    best_params_df.to_csv(
        "predictions/t3_best_parameters.csv",
        index=False
    )

    best_models_df = best_params_df.loc[
        best_params_df.groupby("dataset")["sharpe_ratio"].idxmax()
    ].reset_index(drop=True)

    predictions = {}

    for _, row in best_models_df.iterrows():

        dataset_name = row["dataset"]
        model_type = row["model_type"]
        best_cq = row["best_cq"]
        best_z = row["best_z"]
        best_l1_ratio = (
            row["best_l1_ratio"]
            if model_type == "elasticnet"
            else None
        )
        data = datasets[dataset_name]

        # ====================================================
        # 1. Combine training + validation data
        # ====================================================

        X_full = pd.concat(
            [
                data["X_train"],
                data["X_val"]
            ],
            axis=0,
            ignore_index=True
        )

        y_full = pd.concat(
            [
                data["y_train"],
                data["y_val"]
            ],
            axis=0,
            ignore_index=True
        )

        X_test = data["X_test"]

        # Sanity checks
        assert len(X_full) == len(y_full), \
            "X_full and y_full row counts do not match"

        assert list(X_full.columns) == list(X_test.columns), \
            "Train and test feature columns do not match"

        # ====================================================
        # 2. Determine final number of observed features
        # ====================================================

        P = X_full.shape[1]
        T_train = X_full.shape[0]

        P1 = int(
            round(best_cq * T_train)
        )

        # Cannot use more features than available
        P1 = min(P1, P)
        q = P1 / P

        # ====================================================
        # 3. Select observed features
        # ====================================================

        X_train_final = X_full.iloc[:, :P1]
        X_test_final = X_test.iloc[:, :P1]

        # ====================================================
        # 4. Fit final model and predict hidden test sample
        # ====================================================

        y_hat_test, beta_hat = train_model(
            X_train=X_train_final,
            y_train=y_full,
            X_test=X_test_final,
            T_train=T_train,
            z=best_z,
            model_type=model_type,
            l1_ratio=best_l1_ratio
        )

        # ====================================================
        # 5. Store predictions
        # ====================================================

        predictions[dataset_name] = {
            "y_hat_test": y_hat_test,
            "beta_hat": beta_hat,
            "model_type": model_type,
            "best_cq": best_cq,
            "P1": P1,
            "q": q,
            "best_z": best_z,
            "best_l1_ratio": best_l1_ratio
        }

        print(
            f"Dataset {dataset_name}: "
            f"{model_type}, "
            f"P1={P1}/{P}, "
            f"selected cq={best_cq:g}, "
            f"z={best_z:g}"
        )

        output_df = pd.DataFrame({
            "t": data["t_test"].values,
            "yhat": y_hat_test
        })

        # Sanity checks
        assert len(output_df) == len(data["X_test"])
        assert output_df["yhat"].notna().all()

        # Save prediction
        output_df.to_csv(
            f"predictions/A0352666W_predictions_{dataset_name}.csv",
            index=False
        )

        print(f"Saved predictions for Dataset {dataset_name}")
    return predictions

def task3():
    train_ratio = 0.8
    data_path = "data/"
    datasets = load_task3_datasets(
        dataset_names=("A", "B", "C"), 
        train_ratio=train_ratio, 
        data_path=data_path
    )

    plot_feature_correlations(
        datasets
    )

    # Ridge Estimation and Fixed Grid
    cq_grids = {

        "A": np.array([
            0.50, 0.75, 0.90, 0.95, 0.98,
            1.02, 1.05, 1.10, 1.25
        ]),

        "B": np.array([
            0.50, 0.75, 0.90, 0.95, 0.98,
            1.02, 1.05, 1.10, 1.25, 1.50,
            2.0, 3.0, 5.0, 7.0, 10.0, 12.5
        ]),

        "C": np.array([
            0.50, 0.75, 0.90, 0.95, 0.98,
            1.02, 1.05, 1.10, 1.25, 1.50,
            2.0, 3.0, 5.0, 6.25
        ])
    }

    z_grid = np.array([
        0.0,
        0.1, 0.25, 0.5,
        1.0, 2.0, 5.0,
        10.0, 25.0, 50.0, 100.0
    ])

    best_params = []
    # Compare Ridge and Lasso for datasets A, B and C
    for dataset_name, data in datasets.items():
        train_ridge_and_lasso(data, dataset_name, cq_grids, z_grid, best_params)

    # Elastic Net for dataset C
    l1_ratio_grid = np.array([
        0.0, 0.1, 0.25, 0.5,
        0.75, 0.9, 1.0
    ])
    train_elasticnet(
        datasets["C"], 
        "C", 
        cq_grids, 
        z_grid,
        model_type="elasticnet", 
        l1_ratio_grid=l1_ratio_grid,
        best_params=best_params
    )
    
    best_params_df = pd.DataFrame(best_params)
    _ = model_inference(best_params_df, datasets)

if __name__ == "__main__":
    task3()