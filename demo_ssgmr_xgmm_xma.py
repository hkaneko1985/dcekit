# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of ssGMR-xGMM-xMA
#
# The data generation block below is identical to the one in demo_gmr.py.
# ssGMR uses the x values of training and test samples for x-only GMM
# pretraining, but the y values of the test samples are not used for fitting.

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from sklearn.model_selection import train_test_split

from dcekit.generative_model import SSGMRXGMMXMA


# Settings
number_of_components = 13
covariance_type = 'full'  # 'full', 'diag', 'tied', 'spherical'
rep = 'mode'  # 'mode', 'mean'

number_of_all_samples = 500
number_of_test_samples = 200

numbers_of_x = [0, 1, 2]
numbers_of_y = [3, 4]

# Generate samples for demonstration
np.random.seed(seed=100)
x = np.random.rand(number_of_all_samples, 3) * 10 - 5
y1 = 3 * x[:, 0:1] - 2 * x[:, 1:2] + 0.5 * x[:, 2:3]
y2 = 5 * x[:, 0:1] + 2 * x[:, 1:2] ** 3 - x[:, 2:3] ** 2
y1 = y1 + y1.std(ddof=1) * 0.05 * np.random.randn(number_of_all_samples, 1)
y2 = y2 + y2.std(ddof=1) * 0.05 * np.random.randn(number_of_all_samples, 1)

variables = np.c_[x, y1, y2]
variables_train, variables_test = train_test_split(variables, test_size=number_of_test_samples, random_state=100)


# Autoscaling using the training data, as in demo_gmr.py
variables_train_mean = variables_train.mean(axis=0)
variables_train_std = variables_train.std(axis=0, ddof=1)
autoscaled_variables_train = (variables_train - variables_train_mean) / variables_train_std
autoscaled_variables_test = (variables_test - variables_train_mean) / variables_train_std

# Transductive semi-supervised setting.
# The x-only GMM uses x values of both the training and test samples.
# The y values of the test samples are not used in model fitting.
autoscaled_x_for_pretraining = np.r_[
    autoscaled_variables_train[:, numbers_of_x],
    autoscaled_variables_test[:, numbers_of_x],
]


# ssGMR-xGMM-xMA
model = SSGMRXGMMXMA(
    n_components=number_of_components,
    covariance_type=covariance_type,
    rep=rep,
    random_state=100,
)
model.fit(
    autoscaled_variables_train,
    x_for_pretraining=autoscaled_x_for_pretraining,
    numbers_of_x=numbers_of_x,
    numbers_of_y=numbers_of_y,
)


# Forward analysis, i.e., prediction of y from x
predicted_y_test_all = model.predict_rep(
    autoscaled_variables_test[:, numbers_of_x],
    numbers_of_x,
    numbers_of_y,
)

# Direct inverse analysis, i.e., prediction of x from y
estimated_x_test_all = model.predict_rep(
    autoscaled_variables_test[:, numbers_of_y],
    numbers_of_y,
    numbers_of_x,
)


# Check results of forward analysis
print('Results of forward analysis with ssGMR-xGMM-xMA')
plt.rcParams['font.size'] = 18
for y_number in range(len(numbers_of_y)):
    predicted_y_test = np.ndarray.flatten(predicted_y_test_all[:, y_number])
    predicted_y_test = (
        predicted_y_test
        * variables_train[:, numbers_of_y[y_number]].std(ddof=1)
        + variables_train[:, numbers_of_y[y_number]].mean()
    )

    # y-y plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(variables_test[:, numbers_of_y[y_number]], predicted_y_test)
    y_max = np.max(
        np.array([
            np.array(variables_test[:, numbers_of_y[y_number]]),
            predicted_y_test,
        ])
    )
    y_min = np.min(
        np.array([
            np.array(variables_test[:, numbers_of_y[y_number]]),
            predicted_y_test,
        ])
    )
    plt.plot(
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        'k-',
    )
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Estimated Y')
    plt.show()

    actual_y_test = variables_test[:, numbers_of_y[y_number]]
    r2p = float(
        1 - sum((actual_y_test - predicted_y_test) ** 2)
        / sum((actual_y_test - variables_train[:, numbers_of_y[y_number]].mean()) ** 2)
    )
    rmsep = float((sum((actual_y_test - predicted_y_test) ** 2) / len(actual_y_test)) ** 0.5)
    maep = float(sum(abs(actual_y_test - predicted_y_test)) / len(actual_y_test))
    print('r2p: {0}'.format(r2p))
    print('RMSEp: {0}'.format(rmsep))
    print('MAEp: {0}'.format(maep))


# Check results of direct inverse analysis
print('---------------------------')
print('Results of direct inverse analysis with ssGMR-xGMM-xMA')
estimated_x_test = (
    estimated_x_test_all
    * np.matlib.repmat(
        variables_train[:, numbers_of_x].std(ddof=1, axis=0),
        estimated_x_test_all.shape[0],
        1,
    )
    + np.matlib.repmat(
        variables_train[:, numbers_of_x].mean(axis=0),
        estimated_x_test_all.shape[0],
        1,
    )
)
calculated_y_from_estimated_x_test = np.empty([number_of_test_samples, 2])
calculated_y_from_estimated_x_test[:, 0:1] = (
    3 * estimated_x_test[:, 0:1]
    - 2 * estimated_x_test[:, 1:2]
    + 0.5 * estimated_x_test[:, 2:3]
)
calculated_y_from_estimated_x_test[:, 1:2] = (
    5 * estimated_x_test[:, 0:1]
    + 2 * estimated_x_test[:, 1:2] ** 3
    - estimated_x_test[:, 2:3] ** 2
)

for y_number in range(len(numbers_of_y)):
    predicted_y_test = np.ndarray.flatten(calculated_y_from_estimated_x_test[:, y_number])

    # y-y plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(variables_test[:, numbers_of_y[y_number]], predicted_y_test)
    y_max = np.max(
        np.array([
            np.array(variables_test[:, numbers_of_y[y_number]]),
            predicted_y_test,
        ])
    )
    y_min = np.min(
        np.array([
            np.array(variables_test[:, numbers_of_y[y_number]]),
            predicted_y_test,
        ])
    )
    plt.plot(
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        'k-',
    )
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Estimated Y')
    plt.show()

    actual_y_test = variables_test[:, numbers_of_y[y_number]]
    r2p = float(
        1 - sum((actual_y_test - predicted_y_test) ** 2)
        / sum((actual_y_test - variables_train[:, numbers_of_y[y_number]].mean()) ** 2)
    )
    rmsep = float((sum((actual_y_test - predicted_y_test) ** 2) / len(actual_y_test)) ** 0.5)
    maep = float(sum(abs(actual_y_test - predicted_y_test)) / len(actual_y_test))
    print('r2p: {0}'.format(r2p))
    print('RMSEp: {0}'.format(rmsep))
    print('MAEp: {0}'.format(maep))
