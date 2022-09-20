# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of CVPFI (Cross-Validated Permutation Feature Importance considering correlation between features)
# in GPR (Gaussian Process Regression)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
from dcekit.variable_selection import cvpfi

# settings
n_repeats = 5  # the number of repetition J
alpha_r = 0.999  # alpha (the significance level) in the r (correlation) consideration, 1 means the correlations between features are not considered 

number_of_samples = 30
number_of_x = 15
number_of_important_x = 10
rate_of_noise = 0.1
weights_of_x_to_y = np.ones(number_of_important_x)

np.random.seed(12)
x = np.random.rand(number_of_samples, number_of_x)
y = np.dot(x[:, :number_of_important_x], weights_of_x_to_y.reshape([len(weights_of_x_to_y), 1]))
y = y[:, 0]
y += np.random.randn(number_of_samples) * y.std(ddof=1) * rate_of_noise
x_train = pd.DataFrame(x)
y_train = pd.Series(y)

autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

fold_number = x_train.shape[0]
    
numbers_of_x = list(range(x_train.shape[1]))
numbers_of_y = [x_train.shape[1]]
regression_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
regression_model.fit(autoscaled_x_train, autoscaled_y_train)

# CVPFI calculation
importances_mean, importances_std, importances = cvpfi(
    regression_model,
    autoscaled_x_train,
    autoscaled_y_train,
    fold_number=fold_number,
    scoring='r2',
    n_repeats=n_repeats,
    alpha_r=alpha_r,
    random_state=9,
)

plt.rcParams['font.size'] = 16
plt.bar(range(1, number_of_important_x + 1), importances_mean[range(0, number_of_important_x)], color='b', width=1)
plt.bar(range(number_of_important_x + 1, number_of_x + 1), importances_mean[range(number_of_important_x, number_of_x)], color='k', width=1)
plt.xlabel('x')
plt.xlabel('feature number')
plt.ylabel('importance')
plt.show()
