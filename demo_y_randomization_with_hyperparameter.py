# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of y-randomization with hyperparameters

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.validation import y_randomization_with_hyperparam_opt
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, train_test_split

# settings
fold_number = 5  # "fold_number"-fold cross-validation
max_pls_component_number = 20
number_of_training_samples = 100
number_of_test_samples = 10000
number_of_x_variables = 30

# generate sample dataset
x, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples,
                                n_features=number_of_x_variables, n_informative=10, noise=15, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# cross-validation
pls_components = np.arange(1, max_pls_component_number + 1)
cv_model = GridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=fold_number)
cv_model.fit(autoscaled_x_train, autoscaled_y_train)

# modeling and prediction
model = getattr(cv_model, 'estimator')
hyperparameters = list(cv_model.best_params_.keys())
for hyperparameter in hyperparameters:
    setattr(model, hyperparameter, cv_model.best_params_[hyperparameter])
model.fit(autoscaled_x_train, autoscaled_y_train)
estimated_y_train = np.ndarray.flatten(model.predict(autoscaled_x_train))
estimated_y_train = estimated_y_train * y_train.std(ddof=1) + y_train.mean()
predicted_y_test = np.ndarray.flatten(model.predict(autoscaled_x_test))
predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()

# y-randomization
y_train_rand, estimated_y_train_rand = y_randomization_with_hyperparam_opt(cv_model, x_train, y_train)

# yy-plot for training data
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_train)
y_max = np.max(np.array([np.array(y_train), estimated_y_train]))
y_min = np.min(np.array([np.array(y_train), estimated_y_train]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
# r2, RMSE, MAE for training data
print('r2: {0}'.format(float(1 - sum((y_train - estimated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - estimated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - estimated_y_train)) / len(y_train))))

# yy-plot in y-randomization
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train_rand, estimated_y_train_rand)
y_max = np.max(np.array([np.array(y_train_rand), estimated_y_train_rand]))
y_min = np.min(np.array([np.array(y_train_rand), estimated_y_train_rand]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in y-randomization')
plt.show()
# r2, RMSE, MAE in y-randomization
print('r2rand: {0}'.format(
    float(1 - sum((y_train_rand - estimated_y_train_rand) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSErand: {0}'.format(float((sum((y_train_rand - estimated_y_train_rand) ** 2) / len(y_train)) ** 0.5)))
print('MAErand: {0}'.format(float(sum(abs(y_train_rand - estimated_y_train_rand)) / len(y_train))))

# yy-plot for test data
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
# r2p, RMSEp, MAEp for test data
print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
