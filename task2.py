# Task 2: Lasso Regression on Simulated Data

import numpy as np
from task1 import initialize_task1
from utils import parameter_grid, plot_metric_vs_cq

def task2():

    S_train, R_train, S_test, R_test, cq_grid = initialize_task1()
    
    # Parameter grid for Lasso Regression
    z_grid = np.array([
        0.0, 0.001, 0.01,
        0.1, 0.25, 0.5,
        1.0, 2.0, 5.0
    ])
    
    df = parameter_grid(S_train, R_train, S_test, R_test, cq_grid, z_grid, model_type="lasso")

    # plot different metrics against cq for different values of z
    plot_metric_vs_cq(df, metric="r2", y_min=-10, plot_title="t2_lasso_R2", group_param="z")

    plot_metric_vs_cq(df, metric="timing_return", y_min=-0.2, y_max=0.2, plot_title="t2_lasso_timing_return", group_param="z")

    plot_metric_vs_cq(df, metric="sharpe_ratio", plot_title="t2_lasso_sharpe_ratio", group_param="z")

    plot_metric_vs_cq(df, metric="parameter_size", y_max=10, plot_title="t2_lasso_parameter_size", group_param="z")

if __name__ == "__main__":
    task2()