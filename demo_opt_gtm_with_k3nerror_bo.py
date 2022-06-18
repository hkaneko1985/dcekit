# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of optimization of GTM hyperparameters with k3n-error and Bayesian optimization (BO)

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.generative_model import GTM
from sklearn.datasets import load_iris

# settings
candidates_of_shape_of_map = np.arange(30, 31, dtype=int)
candidates_of_shape_of_rbf_centers = np.arange(2, 22, 2, dtype=int)
candidates_of_variance_of_rbfs = 2 ** np.arange(-5, 4, 2, dtype=float)
candidates_of_lambda_in_em_algorithm = 2 ** np.arange(-4, 0, dtype=float)
candidates_of_lambda_in_em_algorithm = np.append(0, candidates_of_lambda_in_em_algorithm)
number_of_iterations = 300
k_in_k3nerror = 10
bo_iteration_number = 15

# load an iris dataset
iris = load_iris()
# input_dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
input_dataset = iris.data
color = iris.target

# autoscaling
input_dataset = (input_dataset - input_dataset.mean(axis=0)) / input_dataset.std(axis=0, ddof=1)

# optimize hyperparameter in GTMR with CV and BO
model = GTM(display_flag = True)
model.k3nerror_bo(input_dataset, candidates_of_shape_of_map,
                  candidates_of_shape_of_rbf_centers, candidates_of_variance_of_rbfs,
                  candidates_of_lambda_in_em_algorithm, number_of_iterations,
                  k_in_k3nerror, bo_iteration_number)
print('Optimized hyperparameters')
print('optimized shape of map :', model.shape_of_map)
print('optimized shape of RBF centers :', model.shape_of_rbf_centers)
print('optimized variance of RBFs :', model.variance_of_rbfs)
print('optimized lambda in EM algorithm :', model.lambda_in_em_algorithm)

# construct GTM model
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
