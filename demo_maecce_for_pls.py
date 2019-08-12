# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of MAEcce in PLS modeling

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.validation import mae_cce
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, train_test_split

# settings
number_of_training_samples = 50  # 30, 50, 100, 300, 500, 1000, 3000, for example
number_of_test_samples = 10000
number_of_x_variables = 30  # 10, 30, 50, 100, 300, 500, 1000, 3000, for example
number_of_y_randomization = 50
max_pls_component_number = 20
fold_number = 5

# generate sample dataset
x, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples,
                                n_features=number_of_x_variables, n_informative=10, noise=30,
                                random_state=number_of_training_samples + number_of_x_variables)
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

# MAEcce
mae_cce_train = mae_cce(cv_model, x_train, y_train, number_of_y_randomization=number_of_y_randomization, do_autoscaling=True, random_state=0)

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
mae_test = float(sum(abs(y_test - predicted_y_test)) / len(y_test))
print('MAEp: {0}'.format(mae_test))

# histgram of MAEcce
plt.rcParams["font.size"] = 18
plt.hist(mae_cce_train, bins=30)
plt.plot(mae_test, 0.2, 'r.', markersize=30)
plt.xlabel('MAEcce(histgram), MAEp(red point)')
plt.ylabel('frequency')
plt.show()
