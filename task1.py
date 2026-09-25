# Task 1: Ridge Regression on Simulated Data

import pandas as pd
import numpy as np
from utils import simulate_dgp, parameter_grid, plot_metric_vs_cq

def initialize_task1():
    # Simulate DGP
    T_train = 1000
    T_test = 100
    c = 10.0
    b_star = 0.2
    seed = 1
    
    S_train, R_train, S_test, R_test, _ = simulate_dgp(
        T_train=T_train,
        T_test=T_test,
        c=c,
        b_star=b_star,
        seed=seed
    )

    # Convert features to DataFrames
    S_train = pd.DataFrame(S_train)
    S_test = pd.DataFrame(S_test)

    # Convert returns to Series
    R_train = pd.Series(R_train, name="return")
    R_test = pd.Series(R_test, name="return")

    cq_grid = np.array([
        0.50, 0.75, 0.90, 0.95, 0.98,
        1.02, 1.05, 1.10, 1.25, 1.50,
        2.00, 3.00, 5.00, 7.00, 10.00
    ])
    return S_train, R_train, S_test, R_test, cq_grid

def task1():
    S_train, R_train, S_test, R_test, cq_grid = initialize_task1()

    # Parameter grid for Ridge Regression
    z_grid = np.array([
        0.0,
        0.1, 0.25, 0.5,
        1.0, 2.0, 5.0,
        10.0, 25.0, 50.0, 100.0
    ])
    
    df = parameter_grid(S_train, R_train, S_test, R_test, cq_grid, z_grid, model_type="ridge")

    # plot different metrics against cq for different values of z
    plot_metric_vs_cq(df, metric="r2", y_min=-10, plot_title="t1_ridge_R2", group_param="z")

    plot_metric_vs_cq(df, metric="timing_return", y_min=-0.1, y_max=0.1, plot_title="t1_ridge_timing_return", group_param="z")

    plot_metric_vs_cq(df, metric="sharpe_ratio", plot_title="t1_ridge_sharpe_ratio", group_param="z")

    plot_metric_vs_cq(df, metric="parameter_size", y_max=10, plot_title="t1_ridge_parameter_size", group_param="z")

if __name__ == "__main__":
    task1()