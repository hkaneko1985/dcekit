# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of interpolation based on Gaussian Mixture Regression (GMR)

import numpy as np
import pandas as pd
from dcekit.generative_model import GMR

# settings
iterations = 10
max_number_of_components = 20
covariance_types = ['full', 'diag', 'tied', 'spherical']

# load dataset
arranged_x = pd.read_csv('iris_with_nan.csv', index_col=0)

# select nan sample numbers
nan_indexes = np.where(arranged_x.isnull().sum(axis=1) > 0)[0]
nan_variables = []
effective_variables = []
for sample_number in nan_indexes:
    nan_variables.append(np.where(arranged_x.iloc[sample_number, :].isnull() == True)[0])
    effective_variables.append(np.where(arranged_x.iloc[sample_number, :].isnull() == False)[0])

for iteration in range(iterations):
    print(iteration + 1, '/', iterations)
    if iteration == 0:
        x = arranged_x.dropna(axis=0)
        autoscaled_x_arranged = (arranged_x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    else:
        x = arranged_x.copy()

    # standardize x and y
    autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    x_mean = np.array(x.mean(axis=0))
    x_std = np.array(x.std(axis=0, ddof=1))

    # grid search using BIC
    bic_values = np.empty([max_number_of_components, len(covariance_types)])
    for covariance_type_index, covariance_type in enumerate(covariance_types):
        for number_of_components in range(max_number_of_components):
            model = GMR(n_components=number_of_components + 1, covariance_type=covariance_type)
            model.fit(autoscaled_x)
            bic_values[number_of_components, covariance_type_index] = model.bic(autoscaled_x)

    # set optimal parameters
    optimal_index = np.where(bic_values == bic_values.min())
    optimal_number_of_components = optimal_index[0][0] + 1
    optimal_covariance_type = covariance_types[optimal_index[1][0]]
    #    print(iteration + 1, '/', iterations, ', BIC :', bic_values.min())

    # GMM
    model = GMR(n_components=optimal_number_of_components, covariance_type=optimal_covariance_type)
    model.fit(autoscaled_x)

    # interpolation
    for index, sample_number in enumerate(nan_indexes):
        if iteration == 0:
            mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights_for_x = \
                model.predict(autoscaled_x_arranged.iloc[sample_number:sample_number + 1, effective_variables[index]],
                              effective_variables[index], nan_variables[index])
        else:
            mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights_for_x = \
                model.predict(autoscaled_x.iloc[sample_number:sample_number + 1, effective_variables[index]],
                              effective_variables[index], nan_variables[index])
        interpolated_value = mode_of_estimated_mean[0] * x_std[nan_variables[index]] + x_mean[nan_variables[index]]
        arranged_x.iloc[sample_number, nan_variables[index]] = interpolated_value
#        print(interpolated_value)

# save interpolated dataset
arranged_x.to_csv('interpolated_dataset.csv')
