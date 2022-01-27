# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of transfer learning with frustratingly easy domain adaptation (FEDA) [with hyperparameter for a regression method]
# If autoscaling is required, please autoscale target domain and source domain separately.

import warnings

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcekit.learning import TransferLearningSample
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV

warnings.filterwarnings('ignore')

number_of_training_samples = 3
number_of_test_samples = 100
number_of_source_samples = 100
number_of_samples = number_of_training_samples + number_of_test_samples
noise_ratio_in_simulation = 0.1
fold_number = 3
max_pls_component_number = 30

# generate samples in simulation                              
np.random.seed(0)
x_source = np.random.rand(number_of_source_samples, 2)
y_source = 2 * x_source[:, 0] + 3 * x_source[:, 1] + 1
y_source = y_source + noise_ratio_in_simulation * y_source.std() * np.random.rand(len(y_source))
x_target = np.random.rand(number_of_samples, 2)
y_target = 2 * x_target[:, 0] + 4 * x_target[:, 1] + 1
y_target = y_target + noise_ratio_in_simulation * y_target.std() * np.random.rand(len(y_target))

np.random.seed()
x_train, x_test, y_train, y_test = train_test_split(x_target, y_target, test_size=number_of_test_samples,
                                                    random_state=0)

fold_number = min(fold_number, len(y_train))

# PLS
pls_components = np.arange(1, min(max_pls_component_number, np.linalg.matrix_rank(x_train) * 3) + 1)
cv_model = GridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=fold_number)
model = TransferLearningSample(base_estimator=cv_model, x_source=x_source, y_source=y_source, cv_flag=True)
model.fit(x_train, y_train)

# calculate y in training data
calculated_y_train = model.predict(x_train)
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

# estimate y in cross-validation in training data
estimated_y_in_cv = cross_val_predict(model, x_train, y_train, cv=fold_number)
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_in_cv, c='blue')
y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))

# prediction
if x_test.shape[0]:
    predicted_y_test = model.predict(x_test)
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
