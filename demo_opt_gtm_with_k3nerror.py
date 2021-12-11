# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of optimization of GTM hyperparameters with k3n-error

import matplotlib.figure as figure
import matplotlib.pyplot as plt
# settings
import numpy as np
from dcekit.generative_model import GTM
from dcekit.validation import k3nerror
from sklearn.datasets import load_iris

# settings
candidates_of_shape_of_map = np.arange(30, 31, dtype=int)
candidates_of_shape_of_rbf_centers = np.arange(2, 22, 2, dtype=int)
candidates_of_variance_of_rbfs = 2 ** np.arange(-5, 4, 2, dtype=float)
candidates_of_lambda_in_em_algorithm = 2 ** np.arange(-4, 0, dtype=float)
candidates_of_lambda_in_em_algorithm = np.append(0, candidates_of_lambda_in_em_algorithm)
number_of_iterations = 300
display_flag = False
k_in_k3nerror = 10

# load an iris dataset
iris = load_iris()
# input_dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
input_dataset = iris.data
color = iris.target

# autoscaling
input_dataset = (input_dataset - input_dataset.mean(axis=0)) / input_dataset.std(axis=0, ddof=1)

# grid search
parameters_and_k3nerror = []
all_calculation_numbers = len(candidates_of_shape_of_map) * len(candidates_of_shape_of_rbf_centers) * len(
    candidates_of_variance_of_rbfs) * len(candidates_of_lambda_in_em_algorithm)
calculation_number = 0
for shape_of_map_grid in candidates_of_shape_of_map:
    for shape_of_rbf_centers_grid in candidates_of_shape_of_rbf_centers:
        for variance_of_rbfs_grid in candidates_of_variance_of_rbfs:
            for lambda_in_em_algorithm_grid in candidates_of_lambda_in_em_algorithm:
                calculation_number += 1
                print([calculation_number, all_calculation_numbers])
                # construct GTM model
                model = GTM([shape_of_map_grid, shape_of_map_grid],
                            [shape_of_rbf_centers_grid, shape_of_rbf_centers_grid],
                            variance_of_rbfs_grid, lambda_in_em_algorithm_grid, number_of_iterations, display_flag)
                model.fit(input_dataset)
                if model.success_flag:
                    # calculate of responsibilities
                    responsibilities = model.responsibility(input_dataset)
                    # calculate the mean of responsibilities
                    means = responsibilities.dot(model.map_grids)
                    # calculate k3n-error
                    k3nerror_of_gtm = k3nerror(input_dataset, means, k_in_k3nerror) + k3nerror(means, input_dataset, k_in_k3nerror)
                else:
                    k3nerror_of_gtm = 10 ** 100
                parameters_and_k3nerror.append(
                    [shape_of_map_grid, shape_of_rbf_centers_grid, variance_of_rbfs_grid, lambda_in_em_algorithm_grid,
                     k3nerror_of_gtm])

# optimized GTM
parameters_and_k3nerror = np.array(parameters_and_k3nerror)
optimized_hyperparameter_number = \
    np.where(parameters_and_k3nerror[:, 4] == np.min(parameters_and_k3nerror[:, 4]))[0][0]
shape_of_map = [int(parameters_and_k3nerror[optimized_hyperparameter_number, 0]),
                int(parameters_and_k3nerror[optimized_hyperparameter_number, 0])]
shape_of_rbf_centers = [int(parameters_and_k3nerror[optimized_hyperparameter_number, 1]),
                        int(parameters_and_k3nerror[optimized_hyperparameter_number, 1])]
variance_of_rbfs = parameters_and_k3nerror[optimized_hyperparameter_number, 2]
lambda_in_em_algorithm = parameters_and_k3nerror[optimized_hyperparameter_number, 3]

# construct GTM model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(input_dataset)

# calculate of responsibilities
responsibilities = model.responsibility(input_dataset)

# plot the mean of responsibilities
plt.rcParams['font.size'] = 18
means = responsibilities.dot(model.map_grids)
plt.figure(figsize=figure.figaspect(1))
plt.scatter(means[:, 0], means[:, 1], c=color)
plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel('z1 (mean)')
plt.ylabel('z2 (mean)')
plt.show()

print('Optimized hyperparameters')
print('Optimal map size: {0}, {1}'.format(shape_of_map[0], shape_of_map[1]))
print('Optimal shape of RBF centers: {0}, {1}'.format(shape_of_rbf_centers[0], shape_of_rbf_centers[1]))
print('Optimal variance of RBFs: {0}'.format(variance_of_rbfs))
print('Optimal lambda in EM algorithm: {0}'.format(lambda_in_em_algorithm))
