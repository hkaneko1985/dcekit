# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of hyperparameter optimization based on Bayesian optimization for DNN classification

import numpy as np
import pandas as pd
from dcekit.optimization import bo_dnn_hyperparams_class
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

number_of_test_samples = 200
hidden_layer_sizes_candidates = [(50,), (100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)]
activation_candidates = ['identity', 'logistic', 'tanh', 'relu']
alpha_candidates = 10 ** np.arange(-6, -1, dtype=float)
learning_rate_init_candidates = 10 ** np.arange(-5, 0, dtype=float)
fold_number = 5

# load dataset
breast_cancer = load_breast_cancer()
y = breast_cancer.target
x = breast_cancer.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)
class_types = list(set(y_train))  # クラスの種類。これで混同行列における縦と横のクラスの順番を定めます
class_types.sort(reverse=True)  # 並び替え

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# DNN hyperparameter optimization
optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init = bo_dnn_hyperparams_class(autoscaled_x_train, y_train,
                       hidden_layer_sizes_candidates=hidden_layer_sizes_candidates,
                       activation_candidates=activation_candidates,
                       alpha_candidates=alpha_candidates,
                       learning_rate_init_candidates=learning_rate_init_candidates,
                       parameter=fold_number, bo_iteration_number=15, display_flag=True)

# DNN
model = MLPClassifier(hidden_layer_sizes=optimal_hidden_layer_sizes,
                     activation=optimal_activation,
                     alpha=optimal_alpha,
                     learning_rate_init=optimal_learning_rate_init,
                     random_state=99)
model.fit(autoscaled_x_train, y_train)

# calculate y in training data
calculated_y_train = model.predict(autoscaled_x_train)
# confusion matrix
confusion_matrix_train = pd.DataFrame(
    confusion_matrix(y_train, calculated_y_train, labels=class_types), index=class_types,
    columns=class_types)
print('training data')
print(confusion_matrix_train)
print('')

# prediction
if x_test.shape[0]:
    predicted_y_test = model.predict(autoscaled_x_test)
    # confusion matrix
    confusion_matrix_test = pd.DataFrame(
        confusion_matrix(y_test, predicted_y_test, labels=class_types), index=class_types,
        columns=class_types)
    print('test data')
    print(confusion_matrix_test)
