# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Bayesian optimization

import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d

from dcekit.design_of_exmeriments import bayesian_optimization

# settings
number_of_samples = 10000
number_of_first_samples = 30
number_of_iteration = 100
acquisition_function_flag = 2  # 1: Mutual information (MI), 2: Expected Improvement(EI), 3: Probability of improvement (PI)
do_maximization = False  # true: maximization, false: minimization

# generate dataset
np.random.seed(seed=5)
X = np.random.rand(number_of_samples, 2) * 4 - 2
function1 = 1 + ((X[:, 0] + X[:, 1] + 1) ** 2) * (
        19 - 14 * X[:, 0] + 3 * X[:, 0] ** 2 - 14 * X[:, 1] + 6 * X[:, 0] * X[:, 1] + 3 * X[:, 1] ** 2)
function2 = 30 + ((2 * X[:, 0] - 3 * X[:, 1]) ** 2) * (
        18 - 32 * X[:, 0] + 12 * X[:, 0] ** 2 + 48 * X[:, 1] - 36 * X[:, 0] * X[:, 1] + 27 * X[:, 1] ** 2)
y = np.log(function1 * function2)
print('min of y : {0}'.format(min(y)))

# plot
plt.rcParams['font.size'] = 18
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.scatter(X[:, 0], X[:, 1], y, c=y)
plt.tight_layout()
plt.show()

# set first samples
bad_sample_number = np.where(y > 10)[0]
bad_X = X[bad_sample_number, :]
bad_y = y[bad_sample_number]
np.random.seed(seed=1)
first_sample_numbers = np.random.randint(0, len(bad_y), number_of_first_samples)
X_train = bad_X[first_sample_numbers, :]
y_train = bad_y[first_sample_numbers]
X = np.delete(X, bad_sample_number[first_sample_numbers], 0)
y = np.delete(y, bad_sample_number[first_sample_numbers])

# plot
plt.rcParams['font.size'] = 18
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue')
plt.tight_layout()
plt.show()

# Bayesian optimization
if not do_maximization:
    y = -y
    y_train = -y_train
cumulative_variance = np.empty(len(y))
selected_candidate_number, selected_X_candidate, cumulative_variance = bayesian_optimization(X_train, y_train, X,
                                                                                             acquisition_function_flag,
                                                                                             cumulative_variance)
print('next experiment : {0}'.format(selected_X_candidate)) 
