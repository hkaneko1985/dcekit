# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of fast optimization of SVR hyperparameters with Gaussian kernel using cross-validation

import time

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.optimization import fast_opt_svr_hyperparams
# import pandas as pd
from sklearn import model_selection, svm, datasets
from sklearn.model_selection import train_test_split

# Settings
cs = 2 ** np.arange(-5, 11, dtype=float)  # Candidates of C
epsilons = 2 ** np.arange(-10, 1, dtype=float)  # Candidates of epsilon
gammas = 2 ** np.arange(-20, 11, dtype=float)  # Candidates of gamma
fold_number = 5  # "fold_number"-fold cross-validation
number_of_training_samples = 1000
number_of_test_samples = 1000

# Generate samples for demonstration
x, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=100,
                                n_informative=100, noise=100, random_state=0)

# Divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# Standardize X and y
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# Measure time in hyperparameter optimization
start_time = time.time()

# Hyperparameter optimization
optimal_c, optimal_epsilon, optimal_gamma = fast_opt_svr_hyperparams(autoscaled_x_train, autoscaled_y_train, cs,
                                                                     epsilons, gammas, 'cv', fold_number)

# Check time in hyperparameter optimization
elapsed_time = time.time() - start_time
print('Elapsed time in hyperparameter optimization : {0} [sec]'.format(elapsed_time))

# Check optimized hyperparameters
print('C: {0}, Epsion: {1}, Gamma: {2}'.format(optimal_c, optimal_epsilon, optimal_gamma))

# Construct SVR model
model = svm.SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=optimal_gamma)
model.fit(autoscaled_x_train, autoscaled_y_train)

# Calculate y of training dataset
calculated_y_train = np.ndarray.flatten(model.predict(autoscaled_x_train))
calculated_y_train = calculated_y_train * y_train.std(ddof=1) + y_train.mean()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))
# yy-plot
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, calculated_y_train)
y_max = np.max(np.array([np.array(y_train), calculated_y_train]))
y_min = np.min(np.array([np.array(y_train), calculated_y_train]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()

# Estimate y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_in_cv)
y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()

# Estimate y of test dataset
predicted_y_test = np.ndarray.flatten(model.predict(autoscaled_x_test))
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
# r2p, RMSEp, MAEp
print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, predicted_y_test)
y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y')
plt.show()
