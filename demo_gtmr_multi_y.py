# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of GTMR (Generative Topographic Mapping Regression) with multiple y-variables

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.generative_model import GTM
from sklearn.datasets.samples_generator import make_s_curve
from sklearn.model_selection import train_test_split

# settings
shape_of_map = [30, 30]
shape_of_rbf_centers = [6, 6]
variance_of_rbfs = 0.125
lambda_in_em_algorithm = 0.0625
number_of_iterations = 300
display_flag = True
noise_ratio_of_y = 0.1
random_state_number = 30000

number_of_samples = 1000
number_of_test_samples = 200

numbers_of_x = [0, 1, 2]
numbers_of_y = [3, 4]

# Generate samples for demonstration
np.random.seed(seed=100)
x, color = make_s_curve(number_of_samples, random_state=10)
raw_y1 = 0.3 * x[:, 0] - 0.1 * x[:, 1] + 0.2 * x[:, 2]
y1 = raw_y1 + noise_ratio_of_y * raw_y1.std(ddof=1) * np.random.randn(len(raw_y1))
raw_y2 = np.arcsin(x[:, 0]) + np.log(x[:, 1]) - 0.5 * x[:, 2] ** 4 + 5
y2 = raw_y2 + noise_ratio_of_y * raw_y2.std(ddof=1) * np.random.randn(len(raw_y2))

# plot y1 vs. y2
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y1, y2)
plt.xlabel('y1')
plt.ylabel('y2')
plt.show()

variables = np.c_[x, y1, y2]
variables_train, variables_test = train_test_split(variables, test_size=number_of_test_samples, random_state=100)

# standardize x and y
autoscaled_variables_train = (variables_train - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)
autoscaled_variables_test = (variables_test - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)

# construct GTMR model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(autoscaled_variables_train)
if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(autoscaled_variables_train)
    means, modes = model.means_modes(autoscaled_variables_train)

    plt.rcParams['font.size'] = 18
    for y_number in numbers_of_y:
        # plot the mean of responsibilities
        plt.scatter(means[:, 0], means[:, 1], c=variables_train[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mean)')
        plt.ylabel('z2 (mean)')
        plt.show()
        # plot the mode of responsibilities
        plt.scatter(modes[:, 0], modes[:, 1], c=variables_train[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mode)')
        plt.ylabel('z2 (mode)')
        plt.show()

    # GTMR prediction
    # Forward analysis (Regression)
    mean_of_estimated_mean_of_y, mode_of_estimated_mean_of_y, responsibilities_y, py = \
        model.predict(autoscaled_variables_test[:, numbers_of_x], numbers_of_x, numbers_of_y)

    # Inverse analysis
    mean_of_estimated_mean_of_x, mode_of_estimated_mean_of_x, responsibilities_y, py = \
        model.predict(autoscaled_variables_test[:, numbers_of_y], numbers_of_y, numbers_of_x)

    # Check results of forward analysis (regression)
    print('Results of forward analysis (regression)')
    predicted_y_test_all = mode_of_estimated_mean_of_y.copy()
    plt.rcParams['font.size'] = 18
    for y_number in range(len(numbers_of_y)):
        predicted_y_test = np.ndarray.flatten(predicted_y_test_all[:, y_number])
        predicted_y_test = predicted_y_test * variables_train[:, numbers_of_y[y_number]].std(ddof=1) + variables_train[
                                                                                                       :, numbers_of_y[
                                                                                                              y_number]].mean()
        # yy-plot
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(variables_test[:, numbers_of_y[y_number]], predicted_y_test)
        y_max = np.max(np.array([np.array(variables_test[:, numbers_of_y[y_number]]), predicted_y_test]))
        y_min = np.min(np.array([np.array(variables_test[:, numbers_of_y[y_number]]), predicted_y_test]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Estimated Y')
        plt.show()
        # r2p, RMSEp, MAEp
        print(
            'r2p: {0}'.format(float(1 - sum((variables_test[:, numbers_of_y[y_number]] - predicted_y_test) ** 2) / sum(
                (variables_test[:, numbers_of_y[y_number]] - variables_train[:, numbers_of_y[y_number]].mean()) ** 2))))
        print('RMSEp: {0}'.format(float((sum((variables_test[:, numbers_of_y[y_number]] - predicted_y_test) ** 2) / len(
            variables_test[:, numbers_of_y[y_number]])) ** 0.5)))
        print('MAEp: {0}'.format(float(sum(abs(variables_test[:, numbers_of_y[y_number]] - predicted_y_test)) / len(
            variables_test[:, numbers_of_y[y_number]]))))
