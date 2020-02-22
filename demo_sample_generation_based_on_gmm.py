# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of generating samples using Gaussian Mixture Models (GMM)

# settings
file_name = 'iris_without_species.csv'
number_of_samples_generated = 10000

do_autoscaling = True  # if True, autoscaling is conducted
max_number_of_components = 20
covariance_types = ['full', 'diag', 'tied', 'spherical']

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# load dataset
x = pd.read_csv(file_name, index_col=0)

# standardize x
if do_autoscaling:
    autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
else:
    autoscaled_x = x.copy()

# grid search using BIC
bic_values = np.empty([max_number_of_components, len(covariance_types)])
for covariance_type_index, covariance_type in enumerate(covariance_types):
    for number_of_components in range(max_number_of_components):
        gmm_model = GaussianMixture(n_components=number_of_components + 1, covariance_type=covariance_type)
        gmm_model.fit(autoscaled_x)
        bic_values[number_of_components, covariance_type_index] = gmm_model.bic(autoscaled_x)

# plot
plt.rcParams["font.size"] = 18
plt.figure()
plt.plot(bic_values[:, 3], 'r-', label='spherical')
plt.plot(bic_values[:, 2], 'k-', label='tied')
plt.plot(bic_values[:, 1], 'g-', label='diag')
plt.plot(bic_values[:, 0], 'b-', label='full')
plt.xlabel('Number of components')
plt.ylabel('BIC values')
plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0., )
plt.show()

# optimal parameters
optimal_index = np.where(bic_values == bic_values.min())
optimal_number_of_components = optimal_index[0][0] + 1
optimal_covariance_type = covariance_types[optimal_index[1][0]]

# GMM
gmm = GaussianMixture(n_components=optimal_number_of_components, covariance_type=optimal_covariance_type)
gmm.fit(autoscaled_x)

# mean and covariance
means = gmm.means_
if gmm.covariance_type == 'full':
    all_covariances = gmm.covariances_
elif gmm.covariance_type == 'diag':
    all_covariances = np.empty(
        [gmm.n_components, gmm.covariances_.shape[1], gmm.covariances_.shape[1]])
    for component_number in range(gmm.n_components):
        all_covariances[component_number, :, :] = np.diag(gmm.covariances_[component_number, :])
elif gmm.covariance_type == 'tied':
    all_covariances = np.tile(gmm.covariances_, (gmm.n_components, 1, 1))
elif gmm.covariance_type == 'spherical':
    all_covariances = np.empty([gmm.n_components, gmm.means_.shape[1], gmm.means_.shape[1]])
    for component_number in range(gmm.n_components):
        all_covariances[component_number, :, :] = np.diag(
            gmm.covariances_[component_number] * np.ones(gmm.means_.shape[1]))

# sample generation
all_samples_generated = np.zeros([0, x.shape[1]])
for component in range(gmm.n_components):
    generated_samples = np.random.multivariate_normal(means[component, :], all_covariances[component, :, :],
                                                      int(np.ceil(number_of_samples_generated * gmm.weights_[component])))
    all_samples_generated = np.r_[all_samples_generated, generated_samples]

all_samples_generated = pd.DataFrame(all_samples_generated, columns=x.columns)
if do_autoscaling:
    all_samples_generated = all_samples_generated * x.std(axis=0, ddof=1) + x.mean(axis=0)

# save
all_samples_generated.to_csv('generated_samples.csv')

# scatter plot
variable_number_1 = 0
variable_number_2 = 2
plt.rcParams['font.size'] = 18
plt.scatter(all_samples_generated.iloc[:, variable_number_1], all_samples_generated.iloc[:, variable_number_2], c='black', label='generated samples')
plt.scatter(x.iloc[:, variable_number_1], x.iloc[:, variable_number_2], c='red', label='training samples')
plt.xlabel(x.columns[variable_number_1])
plt.ylabel(x.columns[variable_number_2])
plt.legend(bbox_to_anchor=(1.25, 0.2, 0.5, .100), borderaxespad=0., )
plt.show()
