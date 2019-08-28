# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Locally-Weighted Partial Least Squares (LWPLS) and decision to set hyperparameters using LWPLS

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
from dcekit.just_in_tme import LWPLS
from dcekit.validation import r2lm
from sklearn.model_selection import GridSearchCV

# settings
dynamics_max = 10  # if this is 0, no time delayed variables are used
dynamics_span = 2
y_measurement_delay = 5
max_component_number = 20
candidates_of_lambda_in_similarity = 2 ** np.arange(-9, 6, dtype=float)
fold_number = 5
number_of_training_data = 1000
random_state_number = 0

# load and pre-process dataset
dataset = pd.read_csv('debutanizer_y_measurement_span_10.csv')
# dataset = pd.read_csv( 'debutanizer.csv' )
dataset = np.array(dataset)
if dynamics_max:
    dataset_with_dynamics = np.empty((dataset.shape[0] - dynamics_max, 0))
    dataset_with_dynamics = np.append(dataset_with_dynamics, dataset[dynamics_max:, 0:1], axis=1)
    for x_variable_number in range(dataset.shape[1] - 1):
        dataset_with_dynamics = np.append(dataset_with_dynamics,
                                          dataset[dynamics_max:, x_variable_number + 1:x_variable_number + 2], axis=1)
        for time_delay_number in range(int(np.floor(dynamics_max / dynamics_span))):
            dataset_with_dynamics = np.append(dataset_with_dynamics, dataset[dynamics_max - (
                        time_delay_number + 1) * dynamics_span:-(time_delay_number + 1) * dynamics_span,
                                                                     x_variable_number + 1:x_variable_number + 2],
                                              axis=1)
else:
    dataset_with_dynamics = dataset

x_train_with_999 = dataset_with_dynamics[0:number_of_training_data, 1:]
y_train_with_999 = dataset_with_dynamics[0:number_of_training_data, 0]
x_test_with_999 = dataset_with_dynamics[number_of_training_data:, 1:]
y_test_with_999 = dataset_with_dynamics[number_of_training_data:, 0]

x_train = x_train_with_999[y_train_with_999 != 999, :]
y_train = y_train_with_999[y_train_with_999 != 999]

np.random.seed(random_state_number)
suffled_x_train = np.random.permutation(x_train)  # サンプルをシャッフル
np.random.seed(random_state_number)
suffled_y_train = np.random.permutation(y_train)  # サンプルをシャッフル
    
autoscaled_x_train = (suffled_x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (suffled_y_train - y_train.mean()) / y_train.std(ddof=1)

# grid search + cross-validation
lwpls_components = np.arange(1, max_component_number + 1)
cv_model = GridSearchCV(LWPLS(), {'n_components': lwpls_components, 'lambda_in_similarity': candidates_of_lambda_in_similarity, }, cv=fold_number)
cv_model.fit(autoscaled_x_train, autoscaled_y_train)
optimal_component_number = cv_model.best_params_['n_components']
optimal_lambda_in_similarity = cv_model.best_params_['lambda_in_similarity']

estimated_y_test_with_999 = np.empty((len(y_test_with_999)))
model = LWPLS(n_components=optimal_component_number, lambda_in_similarity=optimal_lambda_in_similarity)
for test_sample_number in range(len(y_test_with_999)):
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    autoscaled_x_test = (x_test_with_999[test_sample_number:test_sample_number + 1, ] - x_train.mean(
        axis=0)) / x_train.std(axis=0, ddof=1)
    model.fit(autoscaled_x_train, autoscaled_y_train)
    autoscaled_estimated_y_test = model.predict(autoscaled_x_test)
    if np.isnan(autoscaled_estimated_y_test):
        if test_sample_number == 0:
            estimated_y_test_with_999[test_sample_number] = y_train.mean()
        else:
            estimated_y_test_with_999[test_sample_number] = estimated_y_test_with_999[test_sample_number - 1]
    else:
        estimated_y_test_with_999[test_sample_number] = autoscaled_estimated_y_test * y_train.std(ddof=1) + y_train.mean()

    if test_sample_number - y_measurement_delay >= 0:
        if y_test_with_999[test_sample_number - y_measurement_delay] != 999:
            x_train = np.append(x_train, x_test_with_999[
                                         test_sample_number - y_measurement_delay:test_sample_number - y_measurement_delay + 1, ],
                                axis=0)
            y_train = np.append(y_train, y_test_with_999[test_sample_number - y_measurement_delay])

estimated_y_test = estimated_y_test_with_999[y_test_with_999 != 999]
y_test = y_test_with_999[y_test_with_999 != 999]

# yy-plot
plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, estimated_y_test)
max_y = np.max(np.array([np.array(y_test), estimated_y_test]))
min_y = np.min(np.array([np.array(y_test), estimated_y_test]))
plt.plot([min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)],
         [min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y)], 'k-')
plt.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
plt.xlabel("measured y")
plt.ylabel("estimated y")
plt.show()

plt.figure()
elapsed_time = np.arange(0, len(y_test_with_999))
plt.plot(elapsed_time, estimated_y_test_with_999, 'r.', label='estimated y')
plt.plot(elapsed_time[y_test_with_999 != 999], y_test, 'b.', label='measured y')
plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100),
           borderaxespad=0., )
plt.xlabel('time')
plt.ylabel('y')
plt.show()

plt.figure()
elapsed_time = np.arange(0, len(y_test_with_999))
plt.plot(elapsed_time, estimated_y_test_with_999, 'r.', label='estimated y')
plt.plot(elapsed_time[y_test_with_999 != 999], y_test, 'b.', label='measured y')
plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100),
           borderaxespad=0., )
plt.xlabel('time')
plt.xlim(900, 1100)
plt.ylim(0, 1)
plt.ylabel('y')
plt.show()

# r2, r2lm, RMSE and MAE for test data
print('r2 for test data : {0}'.format(
    float(1 - sum((y_test - estimated_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
print('r2lm for test data : {0}'.format(r2lm(y_test, estimated_y_test)))
print('RMSE for test data : {0}'.format(float((sum((y_test - estimated_y_test) ** 2) / len(y_test)) ** (1 / 2))))
print('MAE for test data : {0}'.format(float(sum(abs(y_test - estimated_y_test)) / len(y_test))))
