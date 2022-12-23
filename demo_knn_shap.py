# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of kNN-SHAP (for more details, please go to https://www.sciencedirect.com/science/article/pii/S2772508122000692)
# in GPR (Gaussian Process Regression)
# SHAP must be installed. https://anaconda.org/conda-forge/shap

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression 
import shap
from sklearn.neighbors import NearestNeighbors

# setting
k_in_knn = 10

number_of_training_samples = 100
number_of_test_samples = 100

# Generate samples for demonstration
x, y = make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=10,
                       n_informative=10, noise=5, random_state=0)

# Divide samples into training samples and test samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# Standardize X and y
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# modeling
model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
model.fit(autoscaled_x_train, autoscaled_y_train)

# calculate SHAP for training data
explainer_shap = shap.KernelExplainer(model.predict, autoscaled_x_train)
shap_train = np.zeros(autoscaled_x_train.shape)
print('SHAP calculation')
for sample_number in range(autoscaled_x_train.shape[0]):
    print(sample_number + 1, '/', autoscaled_x_train.shape[0])
    shap_train[sample_number, :] = explainer_shap.shap_values(autoscaled_x_train[sample_number, :])
#shap_test = np.zeros(autoscaled_x_test.shape)
#for sample_number in range(autoscaled_x_test.shape[0]):
#    shap_test[sample_number, :] = explainer_shap.shap_values(autoscaled_x_test[sample_number, :])

# calculate kNN-SHAP for training data
knn = NearestNeighbors(n_neighbors=k_in_knn)
knn.fit(autoscaled_x_train)
knn_dist_train, knn_index_train = knn.kneighbors(None)
knn_shap_train = np.zeros(autoscaled_x_train.shape)
for sample_number in range(autoscaled_x_train.shape[0]):
    knn_shap_train[sample_number, :] = (shap_train[knn_index_train[sample_number, :], :].mean(axis=0))
# calculate kNN-SHAP for test data
knn_dist_test, knn_index_test = knn.kneighbors(autoscaled_x_test)
knn_shap_test = np.zeros(autoscaled_x_test.shape)
for sample_number in range(autoscaled_x_test.shape[0]):
    knn_shap_test[sample_number, :] = (shap_train[knn_index_test[sample_number, :], :].mean(axis=0))

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
