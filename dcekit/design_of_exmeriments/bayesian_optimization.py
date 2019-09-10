# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel


def bayesian_optimization(X, y, candidates_of_X, acquisition_function_flag, cumulative_variance=None):
    """
    Bayesian optimization
    
    Gaussian process regression model is constructed between X and y.
    A candidate of X with the highest acquisition function is selected using the model from candidates of X.

    Parameters
    ----------
    X: numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
    y: numpy.array or pandas.DataFrame
        m x 1 vector of a y-variable of training dataset
    candidates_of_X: numpy.array or pandas.DataFrame
        Candidates of X
    acquisition_function_flag: int
        1: Mutual information (MI), 2: Expected improvement(EI), 
        3: Probability of improvement (PI) [0: Estimated y-values]
    cumulative_variance: numpy.array or pandas.DataFrame
        cumulative variance in mutual information (MI)[acquisition_function_flag=1]

    Returns
    -------
    selected_candidate_number : int
        selected number of candidates_of_X
    selected_X_candidate : numpy.array
        selected X candidate
    cumulative_variance: numpy.array
        cumulative variance in mutual information (MI)[acquisition_function_flag=1]
    """

    X = np.array(X)
    y = np.array(y)
    if cumulative_variance is None:
        cumulative_variance = np.empty(len(y))
    else:
        cumulative_variance = np.array(cumulative_variance)

    relaxation_value = 0.01
    delta = 10 ** -6
    alpha = np.log(2 / delta)

    autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_candidates_of_X = (candidates_of_X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_y = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    gaussian_process_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
    gaussian_process_model.fit(autoscaled_X, autoscaled_y)
    autoscaled_estimated_y_test, autoscaled_std_of_estimated_y_test = gaussian_process_model.predict(
        autoscaled_candidates_of_X, return_std=True)

    if acquisition_function_flag == 1:
        acquisition_function_values = autoscaled_estimated_y_test + alpha ** 0.5 * (
                (autoscaled_std_of_estimated_y_test ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
        cumulative_variance = cumulative_variance + autoscaled_std_of_estimated_y_test ** 2
    elif acquisition_function_flag == 2:
        acquisition_function_values = (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) * \
                                      norm.cdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               autoscaled_std_of_estimated_y_test) + \
                                      autoscaled_std_of_estimated_y_test * \
                                      norm.pdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               autoscaled_std_of_estimated_y_test)
    elif acquisition_function_flag == 3:
        acquisition_function_values = norm.cdf(
            (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) / autoscaled_std_of_estimated_y_test)
    elif acquisition_function_flag == 0:
        acquisition_function_values = autoscaled_estimated_y_test

    selected_candidate_number = np.where(acquisition_function_values == max(acquisition_function_values))[0][0]
    selected_X_candidate = candidates_of_X[selected_candidate_number, :]

    return selected_candidate_number, selected_X_candidate, cumulative_variance
