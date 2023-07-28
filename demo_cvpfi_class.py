# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of CVPFI (Cross-Validated Permutation Feature Importance considering correlation between features) for classification
# in Support Vector Machine (SVM) with Gaussian kernel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from dcekit.validation import DCEGridSearchCV
from dcekit.variable_selection import cvpfi

# settings
n_repeats = 5  # the number of repetition J
alpha_r = 0.999  # alpha (the significance level) in the r (correlation) consideration, 1 means the correlations between features are not considered 

fold_number = 5
svm_cs = 2 ** np.arange(-10, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 10, dtype=float)
svm_cs = 2 ** np.arange(-5, -3, dtype=float)
svm_gammas = 2 ** np.arange(-15, -13, dtype=float)

# load dataset
breast_cancer = load_breast_cancer()
y = breast_cancer.target
x = breast_cancer.data
x_train = pd.DataFrame(x)
y_train = pd.Series(y)

autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

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

# CVPFI calculation
importances_mean, importances_std, importances = cvpfi(
    model,
    autoscaled_x_train,
    y,
    fold_number=fold_number,
    scoring='accuracy',
    n_repeats=n_repeats,
    alpha_r=alpha_r,
    random_state=9,
)

plt.rcParams['font.size'] = 16
plt.bar(range(1, x_train.shape[1] + 1), importances_mean[range(0, x_train.shape[1])], color='b', width=1)
plt.xlabel('x')
plt.xlabel('feature number')
plt.ylabel('importance')
plt.show()
