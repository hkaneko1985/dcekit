# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of inverse GTMR (Generative Topographic Mapping Regression)

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from dcekit.generative_model import GTM
# import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll

target_y_value = 4  # y-target for inverse analysis

# settings
shape_of_map = [30, 30]
shape_of_rbf_centers = [4, 4]
variance_of_rbfs = 0.5
lambda_in_em_algorithm = 0.001
number_of_iterations = 300
display_flag = 1
noise_ratio_of_y = 0.1
random_state_number = 30000

number_of_samples = 1000

numbers_of_x = [0, 1, 2]
numbers_of_y = [3]

# load a swiss roll dataset and make a y-variable
x, color = make_swiss_roll(number_of_samples, 0, random_state=10)
raw_y = 0.3 * x[:, 0] - 0.1 * x[:, 1] + 0.2 * x[:, 2]
y = raw_y + noise_ratio_of_y * raw_y.std(ddof=1) * np.random.randn(len(raw_y))
# plot
plt.rcParams['font.size'] = 18
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
fig.colorbar(p)
plt.show()

variables = np.c_[x, y]
# standardize x and y
autoscaled_variables = (variables - variables.mean(axis=0)) / variables.std(axis=0, ddof=1)
autoscaled_target_y_value = (target_y_value - variables.mean(axis=0)[numbers_of_y]) / variables.std(axis=0, ddof=1)[
    numbers_of_y]

# construct GTMR model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(autoscaled_variables)

if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(autoscaled_variables)
    means = responsibilities.dot(model.map_grids)
    modes = model.map_grids[responsibilities.argmax(axis=1), :]

    plt.rcParams['font.size'] = 18
    for y_number in numbers_of_y:
        # plot the mean of responsibilities
        plt.scatter(means[:, 0], means[:, 1], c=variables[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mean)')
        plt.ylabel('z2 (mean)')
        plt.show()
        # plot the mode of responsibilities
        plt.scatter(modes[:, 0], modes[:, 1], c=variables[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mode)')
        plt.ylabel('z2 (mode)')
        plt.show()

    # GTMR prediction for inverse analysis
    mean_of_estimated_mean_of_x, mode_of_estimated_mean_of_x, responsibilities_y, py = \
        model.predict(autoscaled_target_y_value, numbers_of_y, numbers_of_x)

    # Check results of inverse analysis
    print('Results of inverse analysis')
    mean_of_estimated_mean_of_x = mean_of_estimated_mean_of_x * x.std(axis=0, ddof=1) + x.mean(axis=0)
    mode_of_estimated_mean_of_x = mode_of_estimated_mean_of_x * x.std(axis=0, ddof=1) + x.mean(axis=0)
    #    print('estimated x-mean: {0}'.format(mean_of_estimated_mean_of_x))
    print('estimated x-mode: {0}'.format(mode_of_estimated_mean_of_x))

    estimated_x_mean_on_map = responsibilities_y.dot(model.map_grids)
    estimated_x_mode_on_map = model.map_grids[np.argmax(responsibilities_y), :]
    #    print('estimated x-mean on map: {0}'.format(estimated_x_mean_on_map))
    print('estimated x-mode on map: {0}'.format(estimated_x_mode_on_map))

    plt.scatter(modes[:, 0], modes[:, 1], c='blue')
    plt.scatter(estimated_x_mode_on_map[0], estimated_x_mode_on_map[1], c='red', marker='x', s=100)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('z1 (mode)')
    plt.ylabel('z2 (mode)')
    plt.show()
