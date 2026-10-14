# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko

Demonstration of ICO-likeness using NGPR1 and leave-one-out cross-validation.
Reference: "What is the outlier - consistent outlier or inconsistent outlier?"
Run in the DCEKit directory containing numerical_simulation_data.csv.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import LeaveOneOut

# Settings
sample_numbers = [0, 15, 50]  # Zero-based row positions: 0 means the first sample.
dataset_name = 'numerical_simulation_data.csv'  # Also used in demo_elo_pls.py.


def loo_prediction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict each sample using only the other samples for preprocessing/fitting."""
    estimated_y = np.empty(y.size)
    for train, test in LeaveOneOut().split(x):
        x_train, y_train = x[train], y[train]

        # Remove constant variables using the training samples only.
        x_std = x_train.std(axis=0, ddof=1)
        selected_variables = x_std > 0
        if not selected_variables.any():
            raise ValueError('All x variables are constant in a training fold.')
        x_mean = x_train[:, selected_variables].mean(axis=0)
        x_std = x_std[selected_variables]
        autoscaled_x_train = (x_train[:, selected_variables] - x_mean) / x_std
        autoscaled_x_test = (x[test][:, selected_variables] - x_mean) / x_std

        # Autoscale y using the training samples only (ddof=1).
        y_mean, y_std = y_train.mean(), y_train.std(ddof=1)
        if y_std == 0:
            y_std = 1.0
        autoscaled_y_train = (y_train - y_mean) / y_std

        # NGPR1: optimize kernel parameters by log-marginal likelihood in fit().
        # No inner CV/grid search is used, as in the supplied NGPR1 code.
        model = GaussianProcessRegressor(
            ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
        model.fit(autoscaled_x_train, autoscaled_y_train)
        estimated_y[test] = model.predict(autoscaled_x_test) * y_std + y_mean
    return estimated_y

# CSV format: sample names in column 1, y in column 2, x in columns 3 onward.
dataset = pd.read_csv(dataset_name, index_col=0)
if dataset.shape[0] < 4 or dataset.shape[1] < 2:
    raise ValueError('The dataset must contain at least 4 samples and y + x.')
y = dataset.iloc[:, 0].to_numpy(dtype=float)
x = dataset.iloc[:, 1:].to_numpy(dtype=float)
if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise ValueError('x and y must not contain missing or infinite values.')
targets = np.asarray(sample_numbers)
if (targets.ndim != 1 or targets.size == 0
        or not np.issubdtype(targets.dtype, np.integer)):
    raise ValueError('sample_numbers must be a nonempty list of integers.')
if np.any(targets < 0) or np.any(targets >= y.size):
    raise ValueError(f'sample_numbers must be between 0 and {y.size - 1}.')

# Paper step 1: LOO predictions on the full dataset, computed only once.
print('Calculating leave-one-out predictions for the full dataset...')
estimated_y = loo_prediction(x, y)
absolute_error = np.abs(y - estimated_y)
pd.DataFrame({
    'sample_number': np.arange(y.size),
    'sample_name': dataset.index,
    'actual_y': y,
    'estimated_y_in_loo': estimated_y,
    'absolute_error_in_loo': absolute_error,
}).to_csv('ico_likeness_loo_predictions.csv', index=False)

# Evaluate each target separately; all other original samples are retained.
# These targets are not automatically treated as a set of known outliers.
# For several previously detected outliers, first remove them all and then
# add back one at a time to a clean reference dataset, as stated in the paper.
results = []
for i, sample_number in enumerate(targets):
    remaining = np.arange(y.size) != sample_number
    # Paper step 2: repeat LOO after excluding sample A entirely.
    estimated_y_without = loo_prediction(x[remaining], y[remaining])
    # Steps 3-5: both MAEs use exactly the same samples, excluding A.
    # In MAEwOS, A is in every training fold used for this comparison.
    mae_with = absolute_error[remaining].mean()
    mae_without = np.abs(y[remaining] - estimated_y_without).mean()
    ico_likeness = mae_with - mae_without
    results.append({
        'sample_number': sample_number,
        'sample_name': dataset.index[sample_number],
        'absolute_error_in_loo': absolute_error[sample_number],
        'MAEwOS': mae_with,
        'MAEwoOS': mae_without,
        'ICO-likeness': ico_likeness,
    })
    print(f'[{i + 1}/{targets.size}] sample {sample_number}: '
          f'ICO-likeness = {ico_likeness:.6g}')

results = pd.DataFrame(results)
results.to_csv('ico_likeness_results.csv', index=False)
print(results.to_string(index=False))
# A positive value alone is not a universal ICO/CO classification threshold.
