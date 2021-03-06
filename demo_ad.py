# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcekit.validation import ApplicabilityDomain
from sklearn.datasets import load_boston
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import train_test_split

method_name = 'ocsvm'  # 'ocsvm' or 'knn' or 'lof'
rate_of_outliers = 0.05

number_of_test_samples = 400
fold_number = 5

# load dataset
boston = load_boston()
y = boston.target
x = boston.data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

#index = np.argsort(boston.target)
#y = boston.target[index]
#x = boston.data[index, :]
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=False)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# Gaussian process regression
model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
model.fit(autoscaled_x_train, autoscaled_y_train)

# AD
ad = ApplicabilityDomain(method_name=method_name, rate_of_outliers=rate_of_outliers)
ad.fit(autoscaled_x_train)

# calculate y in training data
calculated_y_train = model.predict(autoscaled_x_train) * y_train.std(ddof=1) + y_train.mean()
# yy-plot
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, calculated_y_train, c='blue')
y_max = np.max(np.array([np.array(y_train), calculated_y_train]))
y_min = np.min(np.array([np.array(y_train), calculated_y_train]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))

# prediction
predicted_y_test, predicted_y_test_std = model.predict(autoscaled_x_test, return_std=True)
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
predicted_y_test_std = predicted_y_test_std * y_train.std(ddof=1)
predicted_ad = ad.predict(autoscaled_x_test)

# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, predicted_y_test, c='blue')
y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.show()
# r2p, RMSEp, MAEp
print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
print('')

#plt.scatter(predicted_y_test_std, abs(y_test - predicted_y_test), c='blue')
#plt.xlabel('Std. of estimated Y')
#plt.ylabel('Error of Y')
#plt.show()

plt.scatter(predicted_ad, abs(y_test - predicted_y_test), c='blue')
plt.xlabel('AD index (lower than 0 means outside of AD)')
plt.ylabel('Error of Y')
plt.show()
