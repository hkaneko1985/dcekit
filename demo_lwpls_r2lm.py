# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Locally-Weighted Partial Least Squares (LWPLS)

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np

from dcekit.just_in_tme import lwpls
from dcekit.validation import r2lm

# hyperparameters of LWPLS
component_number = 2
lambda_in_similarity = 2 ** -2

sample_number = 100
np.random.seed(10)
x = 5 * np.random.rand(sample_number, 2)
y = 3 * x[:, 0] ** 2 + 10 * np.log(x[:, 1]) + np.random.randn(sample_number)
y = y + 0.1 * y.std(ddof=1) * np.random.randn(sample_number)
np.random.seed()
x_train = x[0:70, :]
y_train = y[0:70]
x_test = x[70:, :]
y_test = y[70:]

autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

estimated_y_test = lwpls(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test, component_number,
                         lambda_in_similarity)
estimated_y_test = estimated_y_test[:, component_number - 1] * y_train.std(ddof=1) + y_train.mean()

# yy-plot
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, estimated_y_test)
max_y = np.max(np.array([np.array(y_test), estimated_y_test]))
min_y = np.min(np.array([np.array(y_test), estimated_y_test]))
plt.plot([min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],
         [min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)], 'k-')
plt.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlabel('simulated y')
plt.ylabel('estimated y')
plt.show()

# r2, r2lm, RMSE and MAE for test data
print('r2 for test data : {0}'.format(
    float(1 - sum((y_test - estimated_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('r2lm for test data : {0}'.format(r2lm(y_test, estimated_y_test)))
print('RMSE for test data : {0}'.format(float((sum((y_test - estimated_y_test) ** 2) / len(y_test)) ** (1 / 2))))
print('MAE for test data : {0}'.format(float(sum(abs(y_test - estimated_y_test)) / len(y_test))))
