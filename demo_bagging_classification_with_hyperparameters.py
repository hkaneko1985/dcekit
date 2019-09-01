# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
from dcekit.learning import DCEBaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer

number_of_test_samples = 200
number_of_submodels = 50
rate_of_selected_variables = 0.7
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

# SVM
cv_classifier = GridSearchCV(SVC(kernel='rbf'), {'C': svm_cs, 'gamma': svm_gammas},
                             cv=fold_number)
model = DCEBaggingClassifier(base_estimator=cv_classifier, n_estimators=number_of_submodels,
                             max_features=rate_of_selected_variables, autoscaling_flag=True,
                             cv_flag=True, random_state=0)
model.fit(x_train, y_train)

# calculate y in training data
calculated_y_train = model.predict(x_train)
# confusion matrix
confusion_matrix_train = pd.DataFrame(
        confusion_matrix(y_train, calculated_y_train, labels=class_types), index=class_types,
        columns=class_types)
print('training data')
print(confusion_matrix_train)
print('')

# estimate y in cross-validation in training data
estimated_y_in_cv = cross_val_predict(model, x_train, y_train, cv=fold_number)
# confusion matrix
confusion_matrix_train_cv = pd.DataFrame(
        confusion_matrix(y_train, estimated_y_in_cv, labels=class_types), index=class_types,
        columns=class_types)
print('training data in cross-validation')
print(confusion_matrix_train_cv)
print('')

# prediction
if x_test.shape[0]:
    predicted_y_test, predicted_y_test_probability = model.predict(x_test, return_probability=True)
    # confusion matrix
    confusion_matrix_test = pd.DataFrame(
            confusion_matrix(y_test, predicted_y_test, labels=class_types), index=class_types,
            columns=class_types)
    print('test data')
    print(confusion_matrix_test)
        
