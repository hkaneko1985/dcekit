# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""
# 

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings('ignore')


def search_high_rate_of_same_values(x, threshold_of_rate_of_same_values):
    """
    Search variables with high rate of the same values

    Parameters
    ----------
    x: numpy.array or pandas.DataFrame
    threshold_of_rate_of_same_values: float
        threshold of the rate of the same values

    Returns
    -------
    high_rate_variable_numbers : list
        the number of variables that should be deleted
    """
    x = pd.DataFrame(x)
    rate_of_same_value = []
    num = 0
    for x_variable_name in x.columns:
        num += 1
        #    print('{0} / {1}'.format(num, x.shape[1]))
        same_value_number = x[x_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x.shape[0]))
    high_rate_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_values)

    return high_rate_variable_numbers

def search_highly_correlated_variables(x, threshold_of_r):
    """
    Search variables whose absolute correlation coefficient is higher than threshold_of_r

    Parameters
    ----------
    x: numpy.array or pandas.DataFrame
    threshold_of_r: float
        threshold of correlation coefficient

    Returns
    -------
    highly_correlated_variable_numbers : list
        the number of variables that should be deleted
    """
    x = pd.DataFrame(x)
    r_in_x = x.corr()
    r_in_x = abs(r_in_x)
    for i in range(r_in_x.shape[0]):
        r_in_x.iloc[i, i] = 0
    highly_correlated_variable_numbers = []
    for i in range(r_in_x.shape[0]):
        r_max = r_in_x.max()
        r_max_max = r_max.max()
        if r_max_max >= threshold_of_r:
            print(i + 1)
            variable_number_1 = np.where(r_max == r_max_max)[0][0]
            variable_number_2 = np.where(r_in_x.iloc[:, variable_number_1] == r_max_max)[0][0]
            r_sum_1 = r_in_x.iloc[:, variable_number_1].sum()
            r_sum_2 = r_in_x.iloc[:, variable_number_2].sum()
            if r_sum_1 >= r_sum_2:
                delete_x_number = variable_number_1
            else:
                delete_x_number = variable_number_2
            highly_correlated_variable_numbers.append(delete_x_number)
            r_in_x.iloc[:, delete_x_number] = 0
            r_in_x.iloc[delete_x_number, :] = 0
        else:
            break

    return highly_correlated_variable_numbers


def clustering_based_on_correlation_coefficients(x, threshold_of_r):
    """
    Clustering variables based on absolute correlation coefficient

    Parameters
    ----------
    x: numpy.array or pandas.DataFrame
    threshold_of_r: float
        threshold of correlation coefficient

    Returns
    -------
    cluster_numbers : numpy.array
        cluster number for each variable
    """
    x = pd.DataFrame(x)
    r_in_x = x.corr()
    r_in_x = abs(r_in_x)
    distance_in_x = 1 / r_in_x

    # clustering
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree='True',
                                         distance_threshold=1 / threshold_of_r, linkage='complete')
    clustering.fit(distance_in_x)
    cluster_numbers = clustering.labels_

    return cluster_numbers
