# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of generating samples restrected

# settings
file_name = 'virtual_resin_x.csv'
number_of_samples_generated = 10000
x_max_rate = 1.1  # this value is multiplied to the maximum value in dataset and is used as the upper limit for generated samples 
x_min_rate = 0.9  # this value is multiplied to the minimum value in dataset and is used as the lower limit for generated samples
#zero_variable_numbers = [0, 2]  # numbers of x-variables whose values are 0. Empty (zero_variable_numbers = []) is ok
zero_variable_numbers = []
list_of_component_numbers = [0, 1, 2]  # numbers of x-variables whose sum is 'desired_sum_of_components' below. Empty (list_of_component_numbers = []) is ok
#list_of_component_numbers = []
desired_sum_of_components = 1  # sum of x-variables whose numbers are 'list_of_component_numbers'
decimals = [2, 2, 2, 0, -1]  # numbers of decimals for x-variables. The length must be the same as the number of x-variables. If empty (decimals = []), not rounded off
#decimals = []


import numpy as np
import pandas as pd

# load dataset
x = pd.read_csv(file_name, index_col=0, header=0)

# set upper and lower limits
x_upper = x.max() * x_max_rate
x_lower = x.min() * x_min_rate

# generate x
generated_x = np.random.rand(number_of_samples_generated, x.shape[1]) * (x_upper.values - x_lower.values) + x_lower.values
if len(zero_variable_numbers):
    generated_x[:, zero_variable_numbers] = np.zeros([generated_x.shape[0], len(zero_variable_numbers)])
if len(list_of_component_numbers):
    from numpy import matlib
    actual_sum_of_components = generated_x[:, list_of_component_numbers].sum(axis=1)
    actual_sum_of_components_converted = np.matlib.repmat(np.reshape(actual_sum_of_components, (generated_x.shape[0], 1)) , 1, len(list_of_component_numbers))
    generated_x[:, list_of_component_numbers] = generated_x[:, list_of_component_numbers] / actual_sum_of_components_converted * desired_sum_of_components # 対象の特徴量を合計で割り、制約の値を掛けます
    deleting_sample_numbers, _ = np.where(generated_x > x_upper.values)
    generated_x = np.delete(generated_x, deleting_sample_numbers, axis=0)
    deleting_sample_numbers, _ = np.where(generated_x < x_lower.values)
    generated_x = np.delete(generated_x, deleting_sample_numbers, axis=0)
if len(decimals):
    for feature_number in range(generated_x.shape[1]):
        generated_x[:, feature_number] = np.round(generated_x[:, feature_number], decimals[feature_number])

# save
generated_x = pd.DataFrame(generated_x, columns=x.columns)
generated_x.to_csv('generated_samples.csv')
