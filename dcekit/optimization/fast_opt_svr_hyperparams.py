# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

import sys

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from ..validation import make_midknn_dataset

def fast_opt_svr_hyperparams(x, y, cs, epsilons, gammas, validation_method, parameter):
    """
    Fast optimization of SVR hyperparameters
    
    Optimize SVR hyperparameters based on variance of gram matrix and cross-validation or midknn

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    cs : numpy.array or list
        vector of candidates of C
    epsilons : numpy.array or list
        vector of candidates of epsilons
    gammas : numpy.array or list
        vector of candidates of gammas
    validation_method : 'cv' or 'midknn'
        if 'cv', cross-validation is used, and if 'midknn', midknn is used 
    parameter : int
        "fold_number"-fold cross-validation in cross-validation, and k in midknn

    Returns
    -------
    optimal_c : float
        optimized C
    optimal_epsilon : float
        optimized epsilon
    optimal_gamma : float
        optimized gamma
    """
    
    if validation_method != 'cv' and validation_method != 'midknn':
#        print('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))
#        return 0, 0, 0
        sys.exit('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))

 
    x = np.array(x)
    y = np.array(y)
    cs = np.array(cs)
    epsilons = np.array(epsilons)
    gammas = np.array(gammas)
    
    print('1/4 ... pre-optimization of gamma')
    optimal_gamma = maximize_variance_of_gram_matrix(x, gammas)

    if validation_method == 'midknn':
        # make midknn data points
        x_midknn, y_midknn = make_midknn_dataset(x, y, parameter)
    
    # Optimize epsilon with cross-validation
    print('2/4 ... optimization of epsilon')
    if validation_method == 'cv':
        cross_validation = KFold(n_splits=parameter, random_state=9, shuffle=True)
        r2cvs = []
        for epsilon in epsilons:
            model = SVR(kernel='rbf', C=3, epsilon=epsilon, gamma=optimal_gamma)
            estimated_y_in_cv = cross_val_predict(model, x, y, cv=cross_validation)
            r2cvs.append(r2_score(y, estimated_y_in_cv))
        optimal_epsilon = epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]]
#        model = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_gamma), {'epsilon': epsilons}, cv=parameter)
#        model.fit(x, y)
#        optimal_epsilon = model.best_params_['epsilon']
    elif validation_method == 'midknn':
        r2_midknns = []
        for epsilon in epsilons:
            model = SVR(kernel='rbf', C=3, epsilon=epsilon, gamma=optimal_gamma)
            model.fit(x, y)
            estimated_y_midknn = np.ndarray.flatten(model.predict(x_midknn))
            r2_midknns.append(float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2)))
        optimal_epsilon = epsilons[np.where(r2_midknns == np.max(r2_midknns))[0][0]]
    
    # Optimize C with cross-validation
    print('3/4 ... optimization of c')
    if validation_method == 'cv':
        r2cvs = []
        for c in cs:
            model = SVR(kernel='rbf', C=c, epsilon=optimal_epsilon, gamma=optimal_gamma)
            estimated_y_in_cv = cross_val_predict(model, x, y, cv=cross_validation)
            r2cvs.append(r2_score(y, estimated_y_in_cv))
        optimal_c = cs[np.where(r2cvs==np.max(r2cvs))[0][0]]
#        model = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_epsilon, gamma=optimal_gamma), {'C': cs}, cv=parameter)
#        model.fit(x, y)
#        optimal_c = model.best_params_['C']
    elif validation_method == 'midknn':
        r2_midknns = []
        for c in cs:
            model = SVR(kernel='rbf', C=c, epsilon=optimal_epsilon, gamma=optimal_gamma)
            model.fit(x, y)
            estimated_y_midknn = np.ndarray.flatten(model.predict(x_midknn))
            r2_midknns.append(float(1 - sum((y_midknn - estimated_y_midknn) ** 2) / sum((y_midknn - y_midknn.mean()) ** 2)))
        optimal_c = cs[np.where(r2_midknns == np.max(r2_midknns))[0][0]]
    
    # Optimize gamma with cross-validation (optional)
    print('4/4 ... optimization of gamma')
    if validation_method == 'cv':
        r2cvs = []
        for gamma in gammas:
            model = SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=gamma)
            estimated_y_in_cv = cross_val_predict(model, x, y, cv=cross_validation)
            r2cvs.append(r2_score(y, estimated_y_in_cv))
        optimal_gamma = gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
#        model = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_epsilon, C=optimal_c), {'gamma': gammas}, cv=parameter)
#        model.fit(x, y)
#        optimal_gamma = model.best_params_['gamma']
    elif validation_method == 'midknn':
        r2_midknns = []
        for gamma in gammas:
            model = SVR(kernel='rbf', C=optimal_c, epsilon=optimal_epsilon, gamma=gamma)
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
    