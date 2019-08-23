# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
from scipy.spatial import distance


def k3nerror(x1, x2, k=10):
    """
    k-nearest neighbor normalized error (k3n-error)

    When x1 is data of X-variables and x2 is data of Z-variables
    (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
    When x1 is Z-variables (low-dimensional data) and x2 is data of data of
    X-variables, this is k3n error in reconstruction (k3n-X-error).

    k3n-error = k3n-Z-error + k3n-X-error

    Parameters
    ----------
    x1 : numpy.array or pandas.DataFrame
         (autoscaled) m x n matrix of X-variables of training data,
         m is the number of training sammples and
         n is the number of X-variables
    x2 : numpy.array or pandas.DataFrame
         m x k matrix of latent (Z-) variables of training data,
         k is the number of latent variables
    k : int
        The number of neighbors

    Returns
    -------
    k3nerror : float
        k3n-Z-error or k3n-X-error
    """
    x1 = np.array(x1)
    x2 = np.array(x2)

    x1_dist = distance.cdist(x1, x1)
    x1_sorted_indices = np.argsort(x1_dist, axis=1)
    x2_dist = distance.cdist(x2, x2)

    for i in range(x2.shape[0]):
        _replace_zero_with_the_smallest_positive_values(x2_dist[i, :])

    I = np.eye(len(x1_dist), dtype=bool)
    neighbor_dist_in_x1 = np.sort(x2_dist[:, x1_sorted_indices[:, 1:k + 1]][I])
    neighbor_dist_in_x2 = np.sort(x2_dist)[:, 1:k + 1]

    sum_k3nerror = (
            (neighbor_dist_in_x1 - neighbor_dist_in_x2) / neighbor_dist_in_x2
    ).sum()
    return sum_k3nerror / x1.shape[0] / k


def _replace_zero_with_the_smallest_positive_values(arr):
    """
    Replace zeros in array with the smallest positive values.

    Parameters
    ----------
    arr: numpy.array
    """
    arr[arr == 0] = np.min(arr[arr != 0])


def r2lm(measured_y, estimated_y):
    """
    r^2 based on the latest measured y-values (r2lm)

    Calculate r^2 based on the latest measured y-values. Measured_y and estimated_y must be vectors.

    Parameters
    ----------
    measured_y: numpy.array or pandas.DataFrame
    estimated_y: numpy.array or pandas.DataFrame

    Returns
    -------
    r2lm : float
        r^2 based on the latest measured y-values
    """

    measured_y = np.array(measured_y).flatten()
    estimated_y = np.array(estimated_y).flatten()
    return float(1 - sum((measured_y - estimated_y) ** 2) / sum((measured_y[1:] - measured_y[:-1]) ** 2))
