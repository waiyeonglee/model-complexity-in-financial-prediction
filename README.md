# DSA5205 Project 1

Code for **DSA5205 Project 1: Reproduce Key Results in a ML-in-Finance
Study**.

## Repository Structure

``` text
dsa5205_p1/
├── task1.py       # Task 1: Ridge regression
├── task2.py       # Task 2: Lasso regression
├── task3.py       # Task 3: Predictions on datasets A, B and C
├── task4.py       # Task 4: Neural network experiment
├── utils.py       # Shared functions
├── data/          # Input data for Task 3
├── plots/         # Generated plots
└── predictions/   # Generated Task 3 predictions
├── requirements.txt

```

## Requirements

The code requires Python 3 and the following packages:

``` bash
pip install -r requirements.txt
```

## Running the Code

Clone the repository and enter the project directory:

``` bash
git clone https://github.com/waiyeonglee/dsa5205_p1.git
cd dsa5205_p1
```

Each task can be run independently.

### Task 1 --- Ridge Regression

``` bash
python task1.py
```

Runs the simulated-data experiment using Ridge regression and saves the
resulting figures to `plots/`.

### Task 2 --- Lasso Regression

``` bash
python task2.py
```

Runs the same simulated-data experiment using Lasso regression and saves
the figures to `plots/`.

### Task 3 --- Sample Datasets

Ensure the supplied datasets are placed in the `data/` directory:

``` text
data/
├── pairA_train.csv
├── pairA_test_features.csv
├── pairB_train.csv
├── pairB_test_features.csv
├── pairC_train.csv
└── pairC_test_features.csv
```

Then run:

``` bash
python task3.py
```

The script performs model selection using Ridge, Lasso and Elastic Net
where applicable, retrains the selected models using the full training
data, and generates the final predictions.

Outputs are saved to:

``` text
predictions/
├── A0352666W_predictions_A.csv
├── A0352666W_predictions_B.csv
├── A0352666W_predictions_C.csv
└── t3_best_parameters.csv
```

Task 3 figures are saved to `plots/`.

### Task 4 --- Neural Network Experiment

``` bash
python task4.py
```

Trains two-layer neural networks with different hidden-layer widths and
saves the training/test performance figures to `plots/`.

## Notes

-   Run all commands from the repository root directory.
-   `task2.py` and `task4.py` import functions from `task1.py`.
-   All tasks use shared functions from `utils.py`.
-   The `plots/` and `predictions/` directories are created
    automatically when required.

## Author

**Lee Wai Yeong**\
DSA5205 --- Data Science for Quantitative Finance\
National University of Singapore
