# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of hyperparameter optimization with grid search and cross-validation (DCEGridSearchCV) for SVM

import numpy as np
import pandas as pd
from dcekit.validation import DCEGridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

number_of_test_samples = 200
fold_number = 5
svm_cs = 2 ** np.arange(-10, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 10, dtype=float)

# load dataset
breast_cancer = load_breast_cancer()
y = breast_cancer.target
x = breast_cancer.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)
class_types = list(set(y_train))  # クラスの種類。これで混同行列における縦と横のクラスの順番を定めます
class_types.sort(reverse=True)  # 並び替え

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# SVM hyperparameter optimization with grid search and cross-validation (DCEGridSearchCV)
cv_model = DCEGridSearchCV(SVC(kernel='rbf'), {'C': svm_cs, 'gamma': svm_gammas},
                           cv=fold_number, random_state=11, display_flag=True)
cv_model.fit(autoscaled_x_train, y_train)

optimal_svm_c = cv_model.best_params_['C']
optimal_svm_gamma = cv_model.best_params_['gamma']
print('Optimal C in CV :', optimal_svm_c)
print('Optimal γ in CV :', optimal_svm_gamma)

# SVM
model = SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)
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
