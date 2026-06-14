# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of hyperparameter optimization with Bayesian optimization and cross-validation for kNNPC

import numpy as np
import pandas as pd
from dcekit.learning import KNeighborsPerClassClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

number_of_test_samples = 200
# Settings
ks_in_knnpc_for_class1 = np.arange(1, 31, 1)
ks_in_knnpc_for_class0 = np.arange(1, 31, 1)
ks_in_knnpc_per_class = {1 : ks_in_knnpc_for_class1, 0 : ks_in_knnpc_for_class0}
fold_number = 5
bo_iteration_number = 15

# load dataset
breast_cancer = load_breast_cancer()
y = breast_cancer.target
x = breast_cancer.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, stratify=y, random_state=0)
class_types = list(set(y_train))  # クラスの種類。これで混同行列における縦と横のクラスの順番を定めます
class_types.sort(reverse=True)  # 並び替え

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# kNNPC model
model = KNeighborsPerClassClassifier(n_neighbors=None, metric='euclidean')
# Bayesian optimization with cross-validation
model.cv_bo(autoscaled_x_train, y_train, ks_in_knnpc_per_class, fold_number, bo_iteration_number)
print('Best Accuracy score :', model.acccv)

# Modeling
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
