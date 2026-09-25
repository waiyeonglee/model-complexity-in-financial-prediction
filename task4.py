# Task 4: Generalization and Overfitting in Neural Networks

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from task1 import initialize_task1
from utils import compute_metrics

# Prepare simulated data for the neural network experiment
def prepare_task4_data(c_q=10.0):

    S_train, R_train, S_test, R_test, _ = initialize_task1()

    P = S_train.shape[1]
    T_train = S_train.shape[0]

    P1 = min(int(round(c_q * T_train)), P)

    X_train = S_train.iloc[:, :P1].to_numpy(dtype=np.float32)
    X_test = S_test.iloc[:, :P1].to_numpy(dtype=np.float32)

    y_train = R_train.to_numpy(dtype=np.float32)
    y_test = R_test.to_numpy(dtype=np.float32)

    return X_train, y_train, X_test, y_test

# Train and evaluate a two-layer neural network
def train_nn_model(
    X_train,
    y_train,
    X_test,
    y_test,
    width=200,
    epochs=1000,
    lr=1e-3,
    eval_every=1,
    seed=1
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Two-layer neural network
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], width, bias=False),
        nn.ReLU(),
        nn.Linear(width, 1, bias=False)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(epochs + 1):

        # Evaluate model
        if epoch % eval_every == 0:

            model.eval()

            with torch.no_grad():
                y_hat_train = (
                    model(X_train_tensor)
                    .squeeze(-1)
                    .numpy()
                )

                y_hat_test = (
                    model(X_test_tensor)
                    .squeeze(-1)
                    .numpy()
                )

            train_mse = np.mean((y_train - y_hat_train) ** 2)
            test_mse = np.mean((y_test - y_hat_test) ** 2)

            params = np.concatenate([
                p.detach().numpy().ravel()
                for p in model.parameters()
            ])

            (
                r2,
                timing_return,
                sharpe_ratio,
                parameter_size
            ) = compute_metrics(
                y_test,
                y_hat_test,
                params
            )

            second_layer_weights = (
                model[2]
                .weight
                .detach()
                .numpy()
                .ravel()
            )

            history.append({
                "epoch": epoch,
                "width": width,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "r2": r2,
                "timing_return": timing_return,
                "sharpe_ratio": sharpe_ratio,
                "parameter_size": parameter_size,
                "second_layer_l1": np.sum(
                    np.abs(second_layer_weights)
                ),
                "second_layer_l2": np.sum(
                    second_layer_weights ** 2
                )
            })

        if epoch == epochs:
            break

        # Gradient descent
        model.train()
        optimizer.zero_grad()

        y_hat_train = model(X_train_tensor).squeeze(-1)
        loss = loss_fn(y_hat_train, y_train_tensor)

        loss.backward()
        optimizer.step()

    return model, pd.DataFrame(history)

# Plot test error across different network widths
def plot_width_results(df_widths, widths):

    plt.figure(figsize=(10, 6))

    for width in widths:

        temp = df_widths[
            df_widths["width"] == width
        ]

        plt.plot(
            temp["epoch"],
            temp["test_mse"],
            label=f"m = {width}"
        )

        best_idx = temp["test_mse"].idxmin()

        best_epoch = df_widths.loc[
            best_idx,
            "epoch"
        ]

        best_mse = df_widths.loc[
            best_idx,
            "test_mse"
        ]

        plt.scatter(
            best_epoch,
            best_mse,
            marker="o"
        )

    plt.scatter([], [], marker="o", color="black", label="Minimum Test MSE")
    plt.xlabel("Training Epoch")
    plt.ylabel("Test MSE")
    plt.title("Test MSE vs Training Epoch for Different Network Widths")
    plt.legend(title="Network Width")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        "plots/t4_widths.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Compare Sharpe ratios across selected network widths
def plot_sharpe_widths(df_widths, widths=[50, 400]):

    plt.figure(figsize=(10, 6))

    for width in widths:

        temp = df_widths[
            df_widths["width"] == width
        ]

        plt.plot(
            temp["epoch"],
            temp["sharpe_ratio"],
            label=f"m = {width}"
        )

        # Find epoch where test MSE is lowest
        best_idx = temp["test_mse"].idxmin()

        best_epoch = df_widths.loc[
            best_idx,
            "epoch"
        ]

        best_sharpe = df_widths.loc[
            best_idx,
            "sharpe_ratio"
        ]

        # Mark this point on the Sharpe ratio curve
        plt.scatter(
            best_epoch,
            best_sharpe,
            marker="o"
        )

    plt.scatter([], [], marker="o", color="black", label="Minimum Test MSE")
    plt.xlabel("Training Epoch")
    plt.ylabel("OOS Sharpe Ratio")
    plt.title("OOS Sharpe Ratio Across Network Widths")
    plt.legend(title="Network Width")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        "plots/t4_sharpe_widths.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Plot model performance across training epochs
def plot_mse(df):

    best_mse_epoch = int(
        df.loc[df["test_mse"].idxmin(), "epoch"]
    )

    os.makedirs("plots", exist_ok=True)

    # Training and test MSE
    plt.figure(figsize=(10, 6))

    plt.plot(df["epoch"], df["train_mse"], label="Training MSE")
    plt.plot(df["epoch"], df["test_mse"], label="Test MSE")

    plt.axvline(
        best_mse_epoch,
        linestyle="--",
        label=f"Minimum test MSE: epoch {best_mse_epoch}"
    )

    plt.xlabel("Training Epoch")
    plt.ylabel("MSE")
    plt.title("MSE vs Training Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "plots/t4_mse.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Run the Task 4 experiment
def task4():

    X_train, y_train, X_test, y_test = prepare_task4_data(
        c_q=10.0
    )

    widths = [50, 100, 200, 400]
    all_results = []

    # Train networks with different widths
    for width in widths:

        _, df_width = train_nn_model(
            X_train,
            y_train,
            X_test,
            y_test,
            width=width,
            epochs=1000,
            lr=1e-3,
            eval_every=5,
            seed=1
        )

        all_results.append(df_width)

    df_widths = pd.concat(
        all_results,
        ignore_index=True
    )

    # Compare generalization across network widths
    plot_width_results(
        df_widths,
        widths
    )

    # Compare Sharpe ratios across selected network widths
    plot_sharpe_widths(
        df_widths,
        widths=[50, 400]
    )
    # Analyze the largest network in more detail
    df_400 = df_widths[
        df_widths["width"] == 400
    ].copy()

    plot_mse(
        df_400
    )

if __name__ == "__main__":
    task4()