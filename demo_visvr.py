# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Variable Importance-considering Support Vector Regression (VI-SVR) 

import math

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')


rate_of_test_samples = 0.25 # rate of the number of test samples
fold_number = 5 # fold number in cross-validation (CV)
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Gamma for nonlinear svr
random_forest_number_of_trees = 500  # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10, dtype=float) / 10  # Ratio of the number of X-variables for random forest
weights_of_feature_importances = list(np.arange(0, 3.1, 0.1)) # p in VI-SVR

x, y = load_boston(return_X_y=True)
x = pd.DataFrame(x)
y = pd.Series(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rate_of_test_samples, shuffle=True)
    
# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
           
# VI-SVR
rmse_oob_all = []
for random_forest_x_variables_rate in random_forest_x_variables_rates:
    RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
        max(math.ceil(x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
    RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
    estimated_y_in_cv = RandomForestResult.oob_prediction_
    estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
    rmse_oob_all.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)
optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
    np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
    max(math.ceil(x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
regression_model.fit(autoscaled_x_train, autoscaled_y_train)
feature_importances = pd.DataFrame(regression_model.feature_importances_ / max(regression_model.feature_importances_), index=x_train.columns)

autoscaled_x_train_original = autoscaled_x_train.copy()
r2cvs = []
for weight in weights_of_feature_importances:
    autoscaled_x_train = autoscaled_x_train_original * (feature_importances.iloc[:, 0] ** weight)
    # svr
    variance_of_gram_matrix = []
    numpy_autoscaled_x_train = np.array(autoscaled_x_train)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(-nonlinear_svr_gamma * cdist(numpy_autoscaled_x_train, numpy_autoscaled_x_train, metric='sqeuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    # optimize ε with CV
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                               cv=fold_number, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
    # optimize C with CV
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                               {'C': nonlinear_svr_cs}, cv=fold_number, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_c = model_in_cv.best_params_['C']
    # optimize γ with CV
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                               {'gamma': nonlinear_svr_gammas}, cv=fold_number, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
    regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                               gamma=optimal_nonlinear_gamma)
    regression_model.fit(autoscaled_x_train, autoscaled_y_train)
    estimated_y_in_cv = np.ndarray.flatten(
        cross_val_predict(regression_model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
    r2cvs.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum((autoscaled_y_train - autoscaled_y_train.mean()) ** 2)))
    
optimal_weight = weights_of_feature_importances[np.where(r2cvs == np.max(r2cvs))[0][0]]
print(optimal_weight)
autoscaled_x_train = autoscaled_x_train_original * (feature_importances.iloc[:, 0] ** optimal_weight)
autoscaled_x_test = autoscaled_x_test * (feature_importances.iloc[:, 0] ** optimal_weight)
# svr
variance_of_gram_matrix = list()
numpy_autoscaled_x_train = np.array(autoscaled_x_train)
for nonlinear_svr_gamma in nonlinear_svr_gammas:
    gram_matrix = np.exp(-nonlinear_svr_gamma * cdist(numpy_autoscaled_x_train, numpy_autoscaled_x_train, metric='sqeuclidean'))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_nonlinear_gamma = nonlinear_svr_gammas[
    np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
# optimize ε with CV
model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                           cv=fold_number, verbose=0)
model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
# optimize C with CV
model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                           {'C': nonlinear_svr_cs}, cv=fold_number, verbose=0)
model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
optimal_nonlinear_c = model_in_cv.best_params_['C']
# optimize γ with CV
model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                           {'gamma': nonlinear_svr_gammas}, cv=fold_number, verbose=0)
model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                           gamma=optimal_nonlinear_gamma)
regression_model.fit(autoscaled_x_train, autoscaled_y_train)

# calculate y
calculated_y_train = np.ndarray.flatten(regression_model.predict(autoscaled_x_train))
calculated_y_train = calculated_y_train * y_train.std(ddof=1) + y_train.mean()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))
# yy-plot
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

# prediction
predicted_y_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
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
plt.ylabel('Predicted Y')
plt.show()
# r2p, RMSEp, MAEp
print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
