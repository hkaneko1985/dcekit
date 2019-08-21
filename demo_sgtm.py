# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of sparce generative topographic mapping
# (SGTM) https://pubs.acs.org/doi/10.1021/acs.jcim.8b00528

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.generative_model import GTM
from sklearn.datasets import load_iris

# settings
shape_of_map = [10, 10]
shape_of_rbf_centers = [5, 5]
variance_of_rbfs = 4
lambda_in_em_algorithm = 0.001
number_of_iterations = 300
display_flag = 1

# load an iris dataset
iris = load_iris()
# input_dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
input_dataset = iris.data
color = iris.target

# autoscaling
input_dataset = (input_dataset - input_dataset.mean(axis=0)) / input_dataset.std(axis=0, ddof=1)

# construct SGTM model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag, sparse_flag=True)
model.fit(input_dataset)

if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(input_dataset)
    means, modes = model.means_modes(input_dataset)

    # plot the mean of responsibilities
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(means[:, 0], means[:, 1], c=color)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('z1 (mean)')
    plt.ylabel('z2 (mean)')
    plt.show()

    # plot the mode of responsibilities
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(modes[:, 0], modes[:, 1], c=color)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('z1 (mode)')
    plt.ylabel('z2 (mode)')
    plt.show()

# integration of clusters based on Bayesian information criterion (BIC)
clustering_results = np.empty([3, responsibilities.shape[1]])
cluster_numbers = np.empty(responsibilities.shape)
original_mixing_coefficients = model.mixing_coefficients
for i in range(len(original_mixing_coefficients)):
    likelihood = model.likelihood(input_dataset)
    non0_indexes = np.where(model.mixing_coefficients != 0)[0]

    responsibilities = model.responsibility(input_dataset)
    cluster_numbers[:, i] = responsibilities.argmax(axis=1)

    bic = -2 * likelihood + len(non0_indexes) * np.log(input_dataset.shape[0])
    clustering_results[:, i] = np.array([len(non0_indexes), bic, len(np.unique(responsibilities.argmax(axis=1)))])

    if len(non0_indexes) == 1:
        break

    non0_mixing_coefficient = model.mixing_coefficients[non0_indexes]
    model.mixing_coefficients[non0_indexes[non0_mixing_coefficient.argmin()]] = 0
    non0_indexes = np.delete(non0_indexes, non0_mixing_coefficient.argmin())
    model.mixing_coefficients[non0_indexes] = model.mixing_coefficients[non0_indexes] + min(
        non0_mixing_coefficient) / len(non0_indexes)

clustering_results = np.delete(clustering_results, np.arange(i, responsibilities.shape[1]), axis=1)
cluster_numbers = np.delete(cluster_numbers, np.arange(i, responsibilities.shape[1]), axis=1)

plt.figure(figsize=figure.figaspect(1))
plt.plot(clustering_results[0, :], clustering_results[1, :], 'b.')
plt.xlabel('Number of clusters')
plt.ylabel('BIC')
plt.show()

number = np.where(clustering_results[1, :] == min(clustering_results[1, :]))[0][0]
print('Optimal number of clusters: {0}'.format(int(clustering_results[0, number])))
clusters = cluster_numbers[:, number].astype('int64')
