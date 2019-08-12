# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
Demonstration of SVR hyperparameter optimization using the midpoints between
k-nearest-neighbor data points of a training dataset (midknn)
as a validation dataset in regression
"""

import time

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.validation import midknn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Settings
k = 10  # k in k-nearest-neighbor algorithm
cs = 2 ** np.arange(-5, 10, dtype=float)  # Candidates of C
epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Candidates of epsilon
gammas = 2 ** np.arange(-20, 10, dtype=float)  # Candidates of gamma
number_of_training_samples = 300
number_of_test_samples = 100

# Generate samples for demonstration
x, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=10,
                                n_informative=10, noise=10, random_state=0)

# Divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# Standardize X and y
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# Measure time in hyperparameter optimization
start_time = time.time()

# Optimize gamma by maximizing variance in Gram matrix
numpy_autoscaled_x_train = np.array(autoscaled_x_train)
variance_of_gram_matrix = list()
for svr_gamma in gammas:
    gram_matrix = np.exp(
        -svr_gamma * ((numpy_autoscaled_x_train[:, np.newaxis] - numpy_autoscaled_x_train) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_gamma = gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

# Optimize C and epsilon with midknn
midknn_index = midknn(autoscaled_x_train, k)  # generate indexes of midknn
x_midknn = (autoscaled_x_train[midknn_index[:, 0], :] + autoscaled_x_train[midknn_index[:, 1], :]) / 2
y_midknn = (y_train[midknn_index[:, 0]] + y_train[midknn_index[:, 1]]) / 2
r2_midknns = np.empty((len(cs), len(epsilons)))
rmse_midknns = np.empty((len(cs), len(epsilons)))
for c_number, c in enumerate(cs):
    for epsilon_number, epsilon in enumerate(epsilons):
        regression_model = svm.SVR(kernel='rbf', C=c, epsilon=epsilon, gamma=optimal_gamma)
        regression_model.fit(autoscaled_x_train, autoscaled_y_train)
        estimated_y_midknn = np.ndarray.flatten(regression_model.predict(x_midknn))
        estimated_y_midknn = estimated_y_midknn * y_train.std(ddof=1) + y_train.mean()
        r2_midknns[c_number, epsilon_number] = float(
            1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2))
        rmse_midknns[c_number, epsilon_number] = float(
            (2 * (len(y_train) + 1) * sum((y_midknn - estimated_y_midknn) ** 2) / len(y_train) / (
                    len(y_midknn) - 1)) ** 0.5)

optimal_c_epsilon_index = np.where(r2_midknns == r2_midknns.max())
optimal_c = cs[optimal_c_epsilon_index[0][0]]
optimal_epsilon = epsilons[optimal_c_epsilon_index[1][0]]

# Check time in hyperparameter optimization
elapsed_time = time.time() - start_time

# Check optimized hyperparameters
print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_c, optimal_epsilon, optimal_gamma))

# Construct SVR model
regression_model = svm.SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=optimal_gamma)
regression_model.fit(autoscaled_x_train, autoscaled_y_train)

# Calculate y of training dataset
calculated_y_train = np.ndarray.flatten(regression_model.predict(autoscaled_x_train))
calculated_y_train = calculated_y_train * y_train.std(ddof=1) + y_train.mean()
# r2, RMSE, MAE
print("r2: {0}".format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print("RMSE: {0}".format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print("MAE: {0}".format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))
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
plt.xlabel("Actual Y")
plt.ylabel("Calculated Y")
plt.show()

# Estimate y of midknn
estimated_y_midknn = np.ndarray.flatten(regression_model.predict(x_midknn))
estimated_y_midknn = estimated_y_midknn * y_train.std(ddof=1) + y_train.mean()
# r2cv, RMSEcv, MAEcv
print("r2midknn: {0}".format(
    float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2))))
print("RMSEmidknn: {0}".format(
    float((2 * (len(y_train) + 1) * sum((y_midknn - estimated_y_midknn) ** 2) / len(y_train) / (
            len(y_midknn) - 1)) ** 0.5)))
print("MAEmidknn: {0}".format(float(sum(abs(y_midknn - estimated_y_midknn)) / len(y_midknn) * (
        2 * (len(y_train) + 1) / len(y_train) * len(y_midknn) / (len(y_midknn) - 1)) ** 0.5)))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_midknn, estimated_y_midknn)
y_max = np.max(np.array([np.array(y_midknn), estimated_y_midknn]))
y_min = np.min(np.array([np.array(y_midknn), estimated_y_midknn]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y of midknn")
plt.show()

# Estimate y of test dataset
predicted_y_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
# r2p, RMSEp, MAEp
print("r2p: {0}".format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print("RMSEp: {0}".format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print("MAEp: {0}".format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, predicted_y_test)
y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel("Actual Y")
plt.ylabel("Estimated Y")
plt.show()
