# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd

from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables

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

highly_correlated_variable_numbers = search_highly_correlated_variables(x, threshold_of_r)
print('# of highly correlated X-variables: {0}'.format(len(highly_correlated_variable_numbers)))

x_selected = x.drop(x.columns[highly_correlated_variable_numbers], axis=1)
print('# of selected X-variables: {0}'.format(x_selected.shape[1]))
