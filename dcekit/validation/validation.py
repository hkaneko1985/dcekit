# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import math

import numpy as np
import numpy.matlib
from scipy.spatial import distance


def midknn(x, k):
    """
    Midpoints between k-nearest-neighbor data points (midknn)

    Calculate index of midknn of training dataset for validation dataset in regression

    Parameters
    ----------
    x: numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
         m is the number of training sammples and
         n is the number of X-variables
    k : int
        The number of neighbors

    Returns
    -------
    midknn_index : numpy.array
        indexes of two samples for midpoints between k-nearest-neighbor data points
    """

    x = np.array(x)
    x_distance = distance.cdist(x, x)
    sample_pair_numbers = np.argsort(x_distance, axis=1)
    sample_pair_numbers = sample_pair_numbers[:, 1:k + 1]

    midknn_index = np.empty((x.shape[0] * k, 2), dtype='int64')
    for nearest_sample_number in range(k):
        midknn_index[nearest_sample_number * x.shape[0]:(nearest_sample_number + 1) * x.shape[0], 0] = \
            np.arange(x.shape[0])
        midknn_index[nearest_sample_number * x.shape[0]:(nearest_sample_number + 1) * x.shape[0], 1] = \
            sample_pair_numbers[:, nearest_sample_number]

    return midknn_index


def make_midknn_dataset(x, y, k):
    """
    Midpoints between k-nearest-neighbor data points (midknn)

    Get dataset of midknn

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
         m is the number of training sammples and
         n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    k : int
        The number of neighbors

    Returns
    -------
    x_midknn : numpy.array
        x of midknn
    y_midknn : numpy.array
        y of midknn
    """

    x = np.array(x)
    y = np.array(y)
    midknn_index = midknn(x, k)  # generate indexes of midknn
    x_midknn = (x[midknn_index[:, 0], :] + x[midknn_index[:, 1], :]) / 2
    y_midknn = (y[midknn_index[:, 0]] + y[midknn_index[:, 1]]) / 2

    return x_midknn, y_midknn


def double_cross_validation(gs_cv, x, y, outer_fold_number, do_autoscaling=True, random_state=None):
    """
    Double Cross-Validation (DCV)

    Estimate y-values in DCV

    Parameters
    ----------
    gs_cv : object of GridSearchCV (sklearn.model_selection.GridSearchCV)
        for more details, please go to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    x : numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        m x 1 vector of a Y-variable of training data
    outer_fold_number : int
        Fold number in outer CV (fold number in inner CV is included in gs_cv)
    do_autoscaling : bool
        flag of autoscaling, if True, do autoscaling
    random_state : int
        random seed, if None, random seed is not set

    Returns
    -------
    estimated_y : numpy.array
        estimated y-values in DCV
    """

    x = np.array(x)
    y = np.array(y)

    # how to divide datase in outer CV
    min_number = math.floor(x.shape[0] / outer_fold_number)
    mod_number = x.shape[0] - min_number * outer_fold_number
    index = np.matlib.repmat(np.arange(1, outer_fold_number + 1, 1), 1, min_number).ravel()
    if mod_number != 0:
        index = np.r_[index, np.arange(1, mod_number + 1, 1)]
    if random_state != None:
        np.random.seed(random_state)
    fold_index_in_outer_cv = np.random.permutation(index)
    np.random.seed()

    estimated_y = np.zeros(len(y))
    for fold_number_in_outer_cv in np.arange(1, outer_fold_number + 1, 1):
        print(fold_number_in_outer_cv, '/', outer_fold_number)
        # divide training data and test data
        x_train = x[fold_index_in_outer_cv != fold_number_in_outer_cv, :].copy()
        y_train = y[fold_index_in_outer_cv != fold_number_in_outer_cv].copy()
        x_test = x[fold_index_in_outer_cv == fold_number_in_outer_cv, :].copy()
        # shuffle samples
        if random_state != -999:
            np.random.seed(random_state)
        random_numbers = np.random.permutation(np.arange(x_train.shape[0]))
        x_train = x_train[random_numbers, :]
        y_train = y_train[random_numbers]
        np.random.seed()
        # autoscaling
        if do_autoscaling:
            autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
            autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
            autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
        else:
            autoscaled_x_train = x_train.copy()
            autoscaled_y_train = y_train.copy()
            autoscaled_x_test = x_test.copy()
        # inner CV
        gs_cv.fit(autoscaled_x_train, autoscaled_y_train)
        # modeling
        model = getattr(gs_cv, 'estimator')
        hyperparameters = list(gs_cv.best_params_.keys())
        for hyperparameter in hyperparameters:
            setattr(model, hyperparameter, gs_cv.best_params_[hyperparameter])
        model.fit(autoscaled_x_train, autoscaled_y_train)
        # prediction
        estimated_y_test = np.ndarray.flatten(model.predict(autoscaled_x_test))
        if do_autoscaling:
            estimated_y_test = estimated_y_test * y_train.std(ddof=1) + y_train.mean()

        estimated_y[fold_index_in_outer_cv == fold_number_in_outer_cv] = estimated_y_test  # 格納

    return estimated_y


def y_randomization(model, x, y, do_autoscaling=True, random_state=None):
    """
    y-randomization
    
    Estimated y-values after shuffling y-values of dataset without hyperparameters

    Parameters
    ----------
    model : model in sklearn before fitting
    x : numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        m x 1 vector of a Y-variable of training data
    do_autoscaling : bool
        flag of autoscaling, if True, do autoscaling
    random_state : int
        random seed, if None, random seed is not set

    Returns
    -------
    y_shuffle : numpy.array
        k x 1 vector of shuffled y-values of training data
    estimated_y_shuffle : numpy.array
        k x 1 vector of shuffled y-values of randomized training data
    """

    x = np.array(x)
    y = np.array(y)

    if random_state != None:
        np.random.seed(random_state)
    y_shuffle = np.random.permutation(y)
    if do_autoscaling:
        autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        autoscaled_y_shuffle = (y_shuffle - y_shuffle.mean()) / y_shuffle.std(ddof=1)
    else:
        autoscaled_x = x.copy()
        autoscaled_y_shuffle = y_shuffle.copy()

    model.fit(autoscaled_x, autoscaled_y_shuffle)
    estimated_y_shuffle = np.ndarray.flatten(model.predict(autoscaled_x))
    if do_autoscaling:
        estimated_y_shuffle = estimated_y_shuffle * y_shuffle.std(ddof=1) + y_shuffle.mean()

    return y_shuffle, estimated_y_shuffle


def y_randomization_with_hyperparam_opt(gs_cv, x, y, do_autoscaling=True, random_state=None):
    """
    y-randomization
    
    Estimated y-values after shuffling y-values of dataset with hyperparameters

    Parameters
    ----------
    gs_cv : object of GridSearchCV (sklearn.model_selection.GridSearchCV)
        for more details, please go to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    x : numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        m x 1 vector of a Y-variable of training data
    do_autoscaling : bool
        flag of autoscaling, if True, do autoscaling
    random_state : int
        random seed, if None, random seed is not set

    Returns
    -------
    y_shuffle : numpy.array
        k x 1 vector of randomized y-values of training data
    estimated_y_shuffle : numpy.array
        k x 1 vector of estimated y-values of randomized training data
    """

    x = np.array(x)
    y = np.array(y)

    if random_state != None:
        np.random.seed(random_state)
    y_shuffle = np.random.permutation(y)
    if do_autoscaling:
        autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        autoscaled_y_shuffle = (y_shuffle - y_shuffle.mean()) / y_shuffle.std(ddof=1)
    else:
        autoscaled_x = x.copy()
        autoscaled_y_shuffle = y_shuffle.copy()

    # hyperparameter optimiation with cross-validation
    gs_cv.fit(autoscaled_x, autoscaled_y_shuffle)
    # modeling
    model = getattr(gs_cv, 'estimator')
    hyperparameters = list(gs_cv.best_params_.keys())
    for hyperparameter in hyperparameters:
        setattr(model, hyperparameter, gs_cv.best_params_[hyperparameter])

    model.fit(autoscaled_x, autoscaled_y_shuffle)
    estimated_y_shuffle = np.ndarray.flatten(model.predict(autoscaled_x))
    if do_autoscaling:
        estimated_y_shuffle = estimated_y_shuffle * y_shuffle.std(ddof=1) + y_shuffle.mean()

    return y_shuffle, estimated_y_shuffle


def mae_cce(gs_cv, x, y, number_of_y_randomization=30, do_autoscaling=True, random_state=None):
    """
    Chance Correlation‐Excluded Mean Absolute Error (MAEcce)
    
    Calculate MAEcce

    Parameters
    ----------
    gs_cv : object of GridSearchCV (sklearn.model_selection.GridSearchCV)
        for more details, please go to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    x : numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        m x 1 vector of a Y-variable of training data
    number_of_y_randomization : int, default 30
        number of y_randomization
    do_autoscaling : bool
        flag of autoscaling, if True, do autoscaling
    random_state : int
        random seed, if None, random seed is not set

    Returns
    -------
    mae_cce : numpy.array
        values of MAEcce
    """

    x = np.array(x)
    y = np.array(y)

    # general analysis
    if do_autoscaling:
        autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        autoscaled_y = (y - y.mean()) / y.std(ddof=1)
    else:
        autoscaled_x = x.copy()
        autoscaled_y = y.copy()
    # hyperparameter optimiation with cross-validation
    gs_cv.fit(autoscaled_x, autoscaled_y)
    # modeling
    model = getattr(gs_cv, 'estimator')
    hyperparameters = list(gs_cv.best_params_.keys())
    for hyperparameter in hyperparameters:
        setattr(model, hyperparameter, gs_cv.best_params_[hyperparameter])

    model.fit(autoscaled_x, autoscaled_y)
    estimated_y = np.ndarray.flatten(model.predict(autoscaled_x))
    if do_autoscaling:
        estimated_y = estimated_y * y.std(ddof=1) + y.mean()
    mae_train = float(sum(abs(y - estimated_y)) / len(y))
    mae_mean = float(sum(abs(y - y.mean())) / len(y))

    # y-randomization
    mae_yrand = []
    for y_randomizatoin_number in range(number_of_y_randomization):
        if random_state != None:
            np.random.seed(random_state + y_randomizatoin_number + 1)
        y_rand = np.random.permutation(y)
        if do_autoscaling:
            autoscaled_y_rand = (y_rand - y_rand.mean()) / y_rand.std(ddof=1)
        else:
            autoscaled_y_rand = y_rand.copy()
        # hyperparameter optimiation with cross-validation
        gs_cv.fit(autoscaled_x, autoscaled_y_rand)
        # modeling
        model = getattr(gs_cv, 'estimator')
        hyperparameters = list(gs_cv.best_params_.keys())
        for hyperparameter in hyperparameters:
            setattr(model, hyperparameter, gs_cv.best_params_[hyperparameter])
        model.fit(autoscaled_x, autoscaled_y_rand)
        estimated_y_rand = np.ndarray.flatten(model.predict(autoscaled_x))
        if do_autoscaling:
            estimated_y_rand = estimated_y_rand * y_rand.std(ddof=1) + y_rand.mean()
        mae_yrand.append(float(sum(abs(y_rand - estimated_y_rand)) / len(y_rand)))

    mae_cce = mae_train + mae_mean - np.array(mae_yrand)

    return mae_cce
