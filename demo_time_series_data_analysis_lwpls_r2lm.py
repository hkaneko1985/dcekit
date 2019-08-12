# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Locally-Weighted Partial Least Squares (LWPLS) and decision to set hyperparameters using LWPLS

import math

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
from dcekit.just_in_tme import lwpls
from dcekit.validation import r2lm

# settings
dynamics_max = 10  # if this is 0, no time delayed variables are used
dynamics_span = 2
y_measurement_delay = 5
max_component_number = 20
candidates_of_lambda_in_similarity = 2 ** np.arange(-9, 6, dtype=float)
number_of_fold_in_cv = 5
number_of_training_data = 1000

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

autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# grid search + cross-validation
r2cvs = np.empty(
    (min(np.linalg.matrix_rank(autoscaled_x_train), max_component_number), len(candidates_of_lambda_in_similarity)))
min_number = math.floor(x_train.shape[0] / number_of_fold_in_cv)
mod_numbers = x_train.shape[0] - min_number * number_of_fold_in_cv
index = np.matlib.repmat(np.arange(1, number_of_fold_in_cv + 1, 1), 1, min_number).ravel()
if mod_numbers != 0:
    index = np.r_[index, np.arange(1, mod_numbers + 1, 1)]
indexes_for_division_in_cv = np.random.permutation(index)
np.random.seed()
for parameter_number, lambda_in_similarity in enumerate(candidates_of_lambda_in_similarity):
    estimated_y_in_cv = np.empty((len(y_train), r2cvs.shape[0]))
    for fold_number in np.arange(1, number_of_fold_in_cv + 1, 1):
        autoscaled_x_train_in_cv = autoscaled_x_train[indexes_for_division_in_cv != fold_number, :]
        autoscaled_y_train_in_cv = autoscaled_y_train[indexes_for_division_in_cv != fold_number]
        autoscaled_x_validation_in_cv = autoscaled_x_train[indexes_for_division_in_cv == fold_number, :]

        estimated_y_validation_in_cv = lwpls(autoscaled_x_train_in_cv, autoscaled_y_train_in_cv,
                                             autoscaled_x_validation_in_cv, r2cvs.shape[0], lambda_in_similarity)
        estimated_y_in_cv[indexes_for_division_in_cv == fold_number, :] = estimated_y_validation_in_cv * y_train.std(
            ddof=1) + y_train.mean()

    estimated_y_in_cv[np.isnan(estimated_y_in_cv)] = 99999
    ss = (y_train - y_train.mean()).T.dot(y_train - y_train.mean())
    press = np.diag(
        (np.matlib.repmat(y_train.reshape(len(y_train), 1), 1, estimated_y_in_cv.shape[1]) - estimated_y_in_cv).T.dot(
            np.matlib.repmat(y_train.reshape(len(y_train), 1), 1, estimated_y_in_cv.shape[1]) - estimated_y_in_cv))
    r2cvs[:, parameter_number] = 1 - press / ss

best_candidate_number = np.where(r2cvs == r2cvs.max())

optimal_component_number = best_candidate_number[0][0] + 1
optimal_lambda_in_similarity = candidates_of_lambda_in_similarity[best_candidate_number[1][0]]

estimated_y_test_with_999 = np.empty((len(y_test_with_999)))
for test_sample_number in range(len(y_test_with_999)):
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    autoscaled_x_test = (x_test_with_999[test_sample_number:test_sample_number + 1, ] - x_train.mean(
        axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_estimated_y_test = lwpls(autoscaled_x_train, autoscaled_y_train, autoscaled_x_test,
                                        optimal_component_number,
                                        optimal_lambda_in_similarity)
    if np.isnan(autoscaled_estimated_y_test[:, optimal_component_number - 1]):
        if test_sample_number == 0:
            estimated_y_test_with_999[test_sample_number] = 0
        else:
            estimated_y_test_with_999[test_sample_number] = estimated_y_test_with_999[test_sample_number - 1]
    else:
        estimated_y_test_with_999[test_sample_number] = autoscaled_estimated_y_test[:,
                                                        optimal_component_number - 1] * y_train.std(
            ddof=1) + y_train.mean()

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
