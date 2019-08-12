# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of GTM

import matplotlib.figure as figure
import matplotlib.pyplot as plt
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

# construct GTM model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(input_dataset)

if model.success_flag:
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

    # plot the mode of responsibilities
    modes = model.map_grids[responsibilities.argmax(axis=1), :]
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(modes[:, 0], modes[:, 1], c=color)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('z1 (mode)')
    plt.ylabel('z2 (mode)')
    plt.show()
