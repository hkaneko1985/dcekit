# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from ..validation import make_midknn_dataset

def fast_opt_svr_hyperparams_cv(x, y, cs, epsilons, gammas, fold_number):
    """
    Fast optimization of SVR hyperparameters
    
    Optimize SVR hyperparameters based on variance of gram matrix and cross-validation

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    cs : numpy.array or pandas.DataFrame
        vector of candidates of C
    epsilons : numpy.array or pandas.DataFrame
        vector of candidates of epsilons
    gammass : numpy.array or pandas.DataFrame
        vector of candidates of gammas
    fold_number : int
        "fold_number"-fold cross-validation

    Returns
    -------
    optimal_c : float
        optimized C
    optimal_epsilon : float
        optimized epsilon
    optimal_gamma : float
        optimized gamma
    """

    x = np.array(x)
    y = np.array(y)
    cs = np.array(cs)
    epsilons = np.array(epsilons)
    gammas = np.array(gammas)
    
    print('1/4 ... pre-optimization of gamma')
    optimal_gamma = maximize_variance_of_gram_matrix(x, gammas)
    
    # Optimize epsilon with cross-validation
    print('2/4 ... optimization of epsilon')
    model = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_gamma), {'epsilon': epsilons}, cv=fold_number)
    model.fit(x, y)
    optimal_epsilon = model.best_params_['epsilon']
    
    # Optimize C with cross-validation
    print('3/4 ... optimization of c')
    model = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_epsilon, gamma=optimal_gamma), {'C': cs}, cv=fold_number)
    model.fit(x, y)
    optimal_c = model.best_params_['C']
    
    # Optimize gamma with cross-validation (optional)
    print('4/4 ... optimization of gamma')
    model = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_epsilon, C=optimal_c), {'gamma': gammas}, cv=fold_number)
    model.fit(x, y)
    optimal_gamma = model.best_params_['gamma']
        
    return optimal_c, optimal_epsilon, optimal_gamma


def fast_opt_svr_hyperparams_midknn(x, y, cs, epsilons, gammas, k):
    """
    Fast optimization of SVR hyperparameters
    
    Optimize SVR hyperparameters based on variance of gram matrix and cross-validation

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    cs : numpy.array or pandas.DataFrame
        vector of candidates of C
    epsilons : numpy.array or pandas.DataFrame
        vector of candidates of epsilons
    gammass : numpy.array or pandas.DataFrame
        vector of candidates of gammas
    k : int
        The number of neighbors

    Returns
    -------
    optimal_c : float
        optimized C
    optimal_epsilon : float
        optimized epsilon
    optimal_gamma : float
        optimized gamma
    """

    x = np.array(x)
    y = np.array(y)
    cs = np.array(cs)
    epsilons = np.array(epsilons)
    gammas = np.array(gammas)
    
    print('1/4 ... pre-optimization of gamma')
    optimal_gamma = maximize_variance_of_gram_matrix(x, gammas)
    
    # make midknn data points
    x_midknn, y_midknn = make_midknn_dataset(x, y, k)
    
    # Optimize epsilon with midknn
    print('2/4 ... optimization of epsilon')
    r2_midknns = []
    for epsilon in epsilons:
        model = svm.SVR(kernel='rbf', C=3, epsilon=epsilon, gamma=optimal_gamma)
        model.fit(x, y)
        estimated_y_midknn = np.ndarray.flatten(model.predict(x_midknn))
        r2_midknns.append(float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2)))
    optimal_epsilon = epsilons[np.where(r2_midknns == np.max(r2_midknns))[0][0]]
    
    # Optimize C with midknn
    print('3/4 ... optimization of c')
    r2_midknns = []
    for c in cs:
        model = svm.SVR(kernel='rbf', C=c, epsilon=optimal_epsilon, gamma=optimal_gamma)
        model.fit(x, y)
        estimated_y_midknn = np.ndarray.flatten(model.predict(x_midknn))
        r2_midknns.append(float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2)))
    optimal_c = cs[np.where(r2_midknns == np.max(r2_midknns))[0][0]]
    
    # Optimize gamma with midknn
    print('4/4 ... optimization of gamma')
    r2_midknns = []
    for gamma in gammas:
        model = svm.SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=gamma)
        model.fit(x, y)
        estimated_y_midknn = np.ndarray.flatten(model.predict(x_midknn))
        r2_midknns.append(float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2)))
    optimal_gamma = gammas[np.where(r2_midknns == np.max(r2_midknns))[0][0]]
        
    return optimal_c, optimal_epsilon, optimal_gamma


def maximize_variance_of_gram_matrix(x, gammas):
    
    variance_of_gram_matrix = []
    for svr_gamma in gammas:
        gram_matrix = np.exp(
            -svr_gamma * ((x[:, np.newaxis] - x) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_gamma = gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    
    return optimal_gamma
    