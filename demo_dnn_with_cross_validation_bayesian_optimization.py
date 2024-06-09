# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of hyperparameter optimization based on Bayesian optimization for DNN regression

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.optimization import bo_dnn_hyperparams
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

number_of_test_samples = 200
hidden_layer_sizes_candidates = [(50,), (100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)]
activation_candidates = ['identity', 'logistic', 'tanh', 'relu']
alpha_candidates = 10 ** np.arange(-6, -1, dtype=float)
learning_rate_init_candidates = 10 ** np.arange(-5, 0, dtype=float)
fold_number = 5

# load dataset
boston = load_boston()
y = boston.target
x = boston.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# DNN hyperparameter optimization
optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init = bo_dnn_hyperparams(autoscaled_x_train, autoscaled_y_train,
                       hidden_layer_sizes_candidates=hidden_layer_sizes_candidates,
                       activation_candidates=activation_candidates,
                       alpha_candidates=alpha_candidates,
                       learning_rate_init_candidates=learning_rate_init_candidates,
                       validation_method='cv', parameter=fold_number, bo_iteration_number=15, display_flag=True)

# DNN
model = MLPRegressor(hidden_layer_sizes=optimal_hidden_layer_sizes,
                     activation=optimal_activation,
                     alpha=optimal_alpha,
                     learning_rate_init=optimal_learning_rate_init,
                     random_state=99)
model.fit(autoscaled_x_train, autoscaled_y_train)

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
if x_test.shape[0]:
    predicted_y_test = model.predict(autoscaled_x_test) * y_train.std(ddof=1) + y_train.mean()
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
