# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd

from dcekit.variable_selection import search_high_rate_of_same_values, clustering_based_on_correlation_coefficients

threshold_of_r = 0.95  # variable whose absolute correlation coefficnent with other variables is higher than threshold_of_r is searched
threshold_of_rate_of_same_value = 1

# load data set
dataset = pd.read_csv('descriptors_with_logS.csv', encoding='SHIFT-JIS', index_col=0)

dataset = dataset.loc[:, dataset.mean().index]  # 平均を計算できる変数だけ選択
dataset = dataset.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
dataset = dataset.dropna(axis=1)  # nanのある変数を削除

x = dataset.iloc[:, 1:]

# delete variables with high rate of the same values
deleting_variable_numbers = search_high_rate_of_same_values(x, threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where(x.var() == 0)
"""

if len(deleting_variable_numbers[0]) != 0:
    x = x.drop(x.columns[deleting_variable_numbers], axis=1)

print('# of X-variables: {0}'.format(x.shape[1]))

# clustering
cluster_numbers = clustering_based_on_correlation_coefficients(x, threshold_of_r)
print('# of cluesters in X-variables: {0}'.format(cluster_numbers.max()))

# select one variable in each cluster
x_selected = pd.DataFrame([])
for i in range(cluster_numbers.max()):
    variable_numbers = np.where(cluster_numbers == i)[0]
    x_selected = pd.concat([x_selected, x.iloc[:, variable_numbers[0]]], axis=1, sort=False)

# average variables in each cluster
x_averaged = pd.DataFrame([])
for i in range(cluster_numbers.max()):
    variable_numbers = np.where(cluster_numbers == i)[0]
    if variable_numbers.shape[0] == 1:
        x_averaged = pd.concat([x_averaged, x.iloc[:, variable_numbers[0]]], axis=1, sort=False)
    else:
        x_each_cluster = x.iloc[:, variable_numbers]
        autoscaled_x_each_cluster = (x_each_cluster - x_each_cluster.mean()) / x_each_cluster.std()
        averaged_x_each_cluster = autoscaled_x_each_cluster.mean(axis=1)
        averaged_x_each_cluster.name = 'mean_in_cluster_{0}'.format(i)
        x_averaged = pd.concat([x_averaged, averaged_x_each_cluster], axis=1, sort=False)
