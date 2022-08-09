# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import math

import numpy as np
import numpy.matlib
from scipy.spatial import distance
from sklearn.base import is_classifier, clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold, cross_val_predict, GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from itertools import chain

class DCEGridSearchCV(BaseSearchCV):
    """
    Hyperparameter optimization with grid search and cross-validation,
    which is similar to GridSearchCV in scikit-learn https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    Parameters
    ----------
    Parameters are basically the same as the ones in GridSearchCV, KFold, and StratifiedKFold
    GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    KFold : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    StratifiedKFold : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
            
    """
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        random_state = None,
        shuffle = True,
        display_flag = False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        self.random_state = random_state
        self.shuffle = shuffle
        self.display_flag = display_flag
    def fit(self, x, y):
        if type(self.cv) == int:
            if is_classifier(self.estimator):
                cross_validation = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=self.shuffle)
            else:
                cross_validation = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=self.shuffle)
        else:
            cross_validation = self.cv
            
        param_dicts = list(ParameterGrid(self.param_grid))
        scores = []
        for i, param_dict in enumerate(param_dicts):
            self.estimator.set_params(**param_dict)
            estimated_y_in_cv = cross_val_predict(self.estimator, x, y, cv=cross_validation,
                                                  n_jobs=self.n_jobs, verbose=self.verbose, 
                                                  pre_dispatch=self.pre_dispatch)
            if is_classifier(self.estimator):
                score = accuracy_score(y, estimated_y_in_cv)
            else:
                score = r2_score(y, estimated_y_in_cv)
            if self.display_flag:
                print(i + 1, '/', len(param_dicts), '... ' 'score :', score)
            scores.append(score)
        self.best_score_ = max(scores)
        self.best_index_ = scores.index(self.best_score_)
        self.best_params_ = param_dicts[self.best_index_]
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.cv_results_ = {'params': param_dicts, 'score': scores}
    
    def predict(self, x):
        return self.best_estimator_.predict(x)


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
    random_state : int, default None
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

def double_cross_validation_group(gs_cv, x, y, groups, outer_fold_number, do_autoscaling=True, random_state=None):
    """
    Double Cross-Validation (DCV) with groups
    Train and test are randomly selected according to a third-party provided group,

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
    groups : numpy.array or pandas.DataFrame
        m x 1 vector of group ID of training data
    outer_fold_number : int
        Fold number in outer CV (fold number in inner CV is included in gs_cv)
    do_autoscaling : bool
        flag of autoscaling, if True, do autoscaling
    random_state : int, default None
        random seed, if None, random seed is not set

    Returns
    -------
    estimated_y : numpy.array
        estimated y-values in DCV
    """

    x = np.array(x)
    y = np.array(y)
    groups = np.array(groups)
    
    unique_groups = np.array(list(set(groups)))
    # how to divide datase in outer CV
    kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=random_state)
    estimated_y = np.zeros(len(y))
    fold_number_in_outer_cv = 0
    for train_group_idx, test_group_idx in kf.split(unique_groups):
        fold_number_in_outer_cv += 1
        print(fold_number_in_outer_cv, '/', outer_fold_number)
        train_group_numbers, test_group_numbers = unique_groups[train_group_idx], unique_groups[test_group_idx]
        # group to sample number
        train_sample_numbers = np.array([], dtype=np.int64)
        for i in train_group_numbers:
            numbers = np.where(groups == i)[0]
            if len(numbers):
                train_sample_numbers = np.r_[train_sample_numbers, numbers]
        test_sample_numbers = np.array([], dtype=np.int64)
        for i in test_group_numbers:
            numbers = np.where(groups == i)[0]
            if len(numbers):
                test_sample_numbers = np.r_[test_sample_numbers, numbers]

        # divide training data and test data
        x_train = x[train_sample_numbers, :].copy()
        y_train = y[train_sample_numbers].copy()
        x_test = x[test_sample_numbers, :].copy()
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

        estimated_y[test_sample_numbers] = estimated_y_test  # 格納

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

def train_test_split_group(
    *arrays,
    test_size=None,
    train_size=None,
    groups=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):

    """
    Split arrays or matrices into random train and test subsets according to a third-party provided group,
    which is similar to train_test_split in scikit-learn https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    Parameters
    ----------
    Parameters are basically the same as the ones in train_test_split, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
            
    groups : numpy.array or pandas.DataFrame
        m x 1 vector of group ID
    """

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)
    
    if groups is not None:
        n_samples = len(set(groups))
    else:
        n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if groups is not None:
            print()
            cv = GroupShuffleSplit(test_size=n_test, train_size=n_train, random_state=random_state)
            train, test = next(cv.split(X=arrays[0], y=stratify, groups=groups))
        else:
            if stratify is not None:
                CVClass = StratifiedShuffleSplit
            else:
                CVClass = ShuffleSplit
            cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)
            train, test = next(cv.split(X=arrays[0], y=stratify))
        

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )    
    
