# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import copy
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class DCEBaggingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, n_estimators=100, max_features=1.0, autoscaling_flag=False,
                 cv_flag=False, robust_flag=True, random_state=None):
        """
        Ensemble Learning based on Bagging Regression
        
        Parameters
        ----------
        base_estimator: object
            The base estimator in scikit-learn. If cv_flag is True, this must be the object of DCEGridSearchCV or GridSearchCV
        n_estimators: int, default 100
            number of sub-models
        max_features: int or float, default 1.0
            If int, max_features features are selected in sub-dataset, 
            if float, max_features * x.shape[1] features are selected in sub-dataset
        autoscaling_flag : boolean, default False
            If True, autoscaling is done, and if False, autoscaling is not done 
        cv_flag: boolean, default False
            If True, base_estimator must be the object of 
        robust_flag: boolean, default True
            If True, median-based statistics are used, if False, mean-based statistics are used
        random_state : int, default None
            random seed, if None, random seed is not set
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.autoscaling_flag = autoscaling_flag
        self.cv_flag = cv_flag
        self.robust_flag = robust_flag
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_features_ = []
        self.x_means_ = []
        self.x_stds_ = []
        self.y_means_ = []
        self.y_stds_ = []

    def fit(self, x, y):
        """
        Ensemble Learning based on Bagging Regression
        
        Fit x and y to model of base_estimator
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            m x n matrix of X-variables of training data,
            m is the number of training sammples and
            n is the number of X-variables
        y : numpy.array or pandas.DataFrame
            m x 1 vector of a Y-variable of training data
        """

        x = np.array(x)
        y = np.array(y)

        if type(self.max_features) == int:
            number_of_selected_variables = self.max_features
        else:
            number_of_selected_variables = int(np.ceil(self.max_features * x.shape[1]))

        for submodel_number in range(self.n_estimators):
            #        print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
            if self.random_state is not None:
                np.random.seed((submodel_number + 1) * (self.random_state + 9))
            # 0 から (サンプル数) までの間に一様に分布する乱数をサンプルの数だけ生成して、その floor の値の番号のサンプルを選択
            selected_sample_numbers = np.floor(np.random.rand(x.shape[0]) * x.shape[0]).astype(int)
            selected_variable_numbers = np.random.permutation(x.shape[1])[0:number_of_selected_variables]
            np.random.seed()
            selected_x = x[selected_sample_numbers, :]
            selected_x = selected_x[:, selected_variable_numbers]
            selected_y = y[selected_sample_numbers]
            self.estimators_features_.append(selected_variable_numbers)
            if self.autoscaling_flag:
                # autoscaling
                x_std = selected_x.std(axis=0, ddof=1)
                if (x_std == 0).any():
                    sys.exit(
                        'There exist X-variable(s) with zero variance. Please delete these X-variables or set autoscaling_flag as False.')
                x_mean = selected_x.mean(axis=0)
                y_mean = selected_y.mean()
                y_std = selected_y.std(ddof=1)
                selected_autoscaled_x = (selected_x - x_mean) / x_std
                selected_autoscaled_y = (selected_y - y_mean) / y_std
                self.x_means_.append(x_mean)
                self.x_stds_.append(x_std)
                self.y_means_.append(y_mean)
                self.y_stds_.append(y_std)
            else:
                selected_autoscaled_x = selected_x.copy()
                selected_autoscaled_y = selected_y.copy()

            if self.cv_flag:
                # hyperparameter optimization with cross-validation
                self.base_estimator.fit(selected_autoscaled_x, selected_autoscaled_y)
                # modeling
                submodel = getattr(self.base_estimator, 'estimator')
                hyperparameters = list(self.base_estimator.best_params_.keys())
                for hyperparameter in hyperparameters:
                    setattr(submodel, hyperparameter, self.base_estimator.best_params_[hyperparameter])

            else:
                submodel = copy.deepcopy(self.base_estimator)
            submodel.fit(selected_autoscaled_x, selected_autoscaled_y)
            self.estimators_.append(submodel)

    def predict(self, x, return_std=False):
        """
        Ensemble Learning based on Bagging Regression
        
        Predict y-values of samples using all submodels
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
        return_std : boolean, default False
            If True, the standard-deviation of the predictive distribution at the query points is returned along with the mean.
    
        Returns
        -------
        estimated_y : numpy.array, shape (n_samples,)
            k x 1 vector of estimated y-values of test data
        estimated_y_std : numpy.array, shape (n_samples,)
            standard-deviation of the predictive distribution at the query points
        """

        x = np.array(x)
        estimated_y_all = np.zeros([x.shape[0], self.n_estimators])
        for submodel_number in range(self.n_estimators):
            selected_x = x[:, self.estimators_features_[submodel_number]]
            if self.autoscaling_flag:
                selected_autoscaled_x = (selected_x - self.x_means_[submodel_number]) / self.x_stds_[submodel_number]
            else:
                selected_autoscaled_x = selected_x.copy()
            estimated_y = np.ndarray.flatten(self.estimators_[submodel_number].predict(selected_autoscaled_x))
            if self.autoscaling_flag:
                estimated_y = estimated_y * self.y_stds_[submodel_number] + self.y_means_[submodel_number]
            estimated_y_all[:, submodel_number] = estimated_y.copy()

        if self.robust_flag:
            estimated_y = np.median(estimated_y_all, axis=1)
            estimated_y_std = 1.4826 * np.median(abs(estimated_y_all - np.reshape(estimated_y, [x.shape[0], 1])),
                                                 axis=1)
        else:
            estimated_y = np.mean(estimated_y_all, axis=1)
            estimated_y_std = np.std(estimated_y_all, axis=1)
        if return_std:
            return estimated_y, estimated_y_std
        else:
            return estimated_y

    def predict_all(self, x):
        """
        Ensemble Learning based on Bagging Regression
        
        Predict y-values of samples using all submodels
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
    
        Returns
        -------
        estimated_y_all : numpy.array, shape (n_samples, n_estimators)
            Estimated y-values of test data for all submodels
        """

        x = np.array(x)
        estimated_y_all = np.zeros([x.shape[0], self.n_estimators])
        for submodel_number in range(self.n_estimators):
            selected_x = x[:, self.estimators_features_[submodel_number]]
            if self.autoscaling_flag:
                selected_autoscaled_x = (selected_x - self.x_means_[submodel_number]) / self.x_stds_[submodel_number]
            else:
                selected_autoscaled_x = selected_x.copy()
            estimated_y = np.ndarray.flatten(self.estimators_[submodel_number].predict(selected_autoscaled_x))
            if self.autoscaling_flag:
                estimated_y = estimated_y * self.y_stds_[submodel_number] + self.y_means_[submodel_number]
            estimated_y_all[:, submodel_number] = estimated_y.copy()
        return estimated_y_all


class DCEBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=100, max_features=1.0, autoscaling_flag=False,
                 cv_flag=False, random_state=None):
        """
        Ensemble Learning based on Bagging Classification
        
        Parameters
        ----------
        base_estimator: object
            The base estimator in scikit-learn. If cv_flag is True, this must be the object of GridSearchCV
        n_estimators: int, default 100
            number of sub-models
        max_features: int or float, default 1.0
            If int, max_features features are selected in sub-dataset, 
            if float, max_features * x.shape[1] features are selected in sub-dataset
        autoscaling_flag : boolean, default False
            If True, autoscaling is done, and if False, autoscaling is not done 
        cv_flag: boolean, default False
            If True, base_estimator must be the object of 
        random_state : int, default None
            random seed, if None, random seed is not set
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.autoscaling_flag = autoscaling_flag
        self.cv_flag = cv_flag
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_features_ = []
        self.x_means_ = []
        self.x_stds_ = []
        self.class_types_ = None

    def fit(self, x, y):
        """
        Ensemble Learning based on Bagging Classification
        
        Fit x and y to model of base_estimator
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            m x n matrix of X-variables of training data,
            m is the number of training sammples and
            n is the number of X-variables
        y : numpy.array or pandas.DataFrame
            m x 1 vector of a Y-variable of training data
        """

        x = np.array(x)
        y = np.array(y)

        class_types = list(set(y))
        class_types.sort(reverse=True)
        self.class_types_ = class_types

        if type(self.max_features) == int:
            number_of_selected_variables = self.max_features
        else:
            number_of_selected_variables = int(np.ceil(self.max_features * x.shape[1]))

        for submodel_number in range(self.n_estimators):
            #        print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
            if self.random_state is not None:
                np.random.seed((submodel_number + 1) * (self.random_state + 9))
            # 0 から (サンプル数) までの間に一様に分布する乱数をサンプルの数だけ生成して、その floor の値の番号のサンプルを選択
            selected_sample_numbers = np.floor(np.random.rand(x.shape[0]) * x.shape[0]).astype(int)
            selected_variable_numbers = np.random.permutation(x.shape[1])[0:number_of_selected_variables]
            np.random.seed()
            selected_x = x[selected_sample_numbers, :]
            selected_x = selected_x[:, selected_variable_numbers]
            selected_y = y[selected_sample_numbers]
            self.estimators_features_.append(selected_variable_numbers)

            if self.autoscaling_flag:
                # autoscaling
                x_std = selected_x.std(axis=0, ddof=1)
                if (x_std == 0).any():
                    sys.exit(
                        'There exist X-variable(s) with zero variance. Please delete these X-variables or set autoscaling_flag as False.')
                x_mean = selected_x.mean(axis=0)
                selected_autoscaled_x = (selected_x - x_mean) / x_std
                self.x_means_.append(x_mean)
                self.x_stds_.append(x_std)
            else:
                selected_autoscaled_x = selected_x.copy()

            if self.cv_flag:
                # hyperparameter optimization with cross-validation
                self.base_estimator.fit(selected_autoscaled_x, selected_y)
                # modeling
                submodel = getattr(self.base_estimator, 'estimator')
                hyperparameters = list(self.base_estimator.best_params_.keys())
                for hyperparameter in hyperparameters:
                    setattr(submodel, hyperparameter, self.base_estimator.best_params_[hyperparameter])

            else:
                submodel = copy.deepcopy(self.base_estimator)
            submodel.fit(selected_autoscaled_x, selected_y)
            self.estimators_.append(submodel)

    def predict(self, x, return_probability=False):
        """
        Ensemble Learning based on Bagging Classification
        
        Predict y-values of samples using all submodels
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
        return_probability : boolean, default False
            If True, the probability at the query points is returned
    
        Returns
        -------
        estimated_y : numpy.array, shape (n_samples,)
            k x 1 vector of estimated y-values of test data
        estimated_y_probability : numpy.array, shape (n_samples, n_classes)
            Probability at the query points
        """

        x = np.array(x)
        estimated_y_all = np.zeros([x.shape[0], self.n_estimators])
        estimated_y_count = np.zeros([x.shape[0], len(self.class_types_)])  # クラスごとに、推定したサブモデルの数をカウントして値をここに格納

        for submodel_number in range(self.n_estimators):
            selected_x = x[:, self.estimators_features_[submodel_number]]
            if self.autoscaling_flag:
                selected_autoscaled_x = (selected_x - self.x_means_[submodel_number]) / self.x_stds_[submodel_number]
            else:
                selected_autoscaled_x = selected_x.copy()
            estimated_y = np.ndarray.flatten(self.estimators_[submodel_number].predict(selected_autoscaled_x))
            estimated_y_all[:, submodel_number] = estimated_y.copy()
            for sample_number in range(x.shape[0]):
                estimated_y_count[sample_number, self.class_types_.index(estimated_y[sample_number])] += 1

        # テストデータにおける、クラスごとの推定したサブモデルの数
        estimated_y_count = pd.DataFrame(estimated_y_count, columns=self.class_types_)

        # テストデータにおける、クラスごとの確率
        estimated_y_probability = estimated_y_count / self.n_estimators

        # テストデータにおける、多数決で推定された結果
        estimated_y = np.array(estimated_y_count.idxmax(axis=1))

        if return_probability:
            return estimated_y, estimated_y_probability
        else:
            return estimated_y


def ensemble_outlier_sample_detection(base_estimator, x, y, cv_flag, n_estimators=100, iteration=30,
                                      autoscaling_flag=True, random_state=None):
    """
    Ensemble Learning Outlier sample detection (ELO)
    https://datachemeng.com/ensembleoutliersampledetection/
    https://www.sciencedirect.com/science/article/abs/pii/S0169743917305919

    Detect outlier samples based on ELO y-values in DCV

    Parameters
    ----------
    base_estimator: object
        The base estimator in scikit-learn. If cv_flag is True, this must be the object of GridSearchCV
        for more details on GridSearchCV, please go to
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    x : numpy.array or pandas.DataFrame
        m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        m x 1 vector of a Y-variable of training data
    cv_flag: boolen, default False
        If this is True, base_estimator must be the object of GridSearchCV
    n_estimators: int, default 100
        number of sub-models
    iteration: int, default 30
        max number of iteration
    autoscaling_flag : boolean, default False
        If True, autoscaling is done, and if False, autoscaling is not done 
    random_state : int
        random seed, if None, random seed is not set

    Returns
    -------
    outlier_sample_flags : numpy.array, shape (n_samples,)
        TRUE if outlier samples
    """

    x = np.array(x)
    y = np.array(y)

    # 初期化
    outlier_sample_flags = ~(y == y)
    previous_outlier_sample_flags = ~(y == y)
    for iteration_number in range(iteration):
        print(iteration_number + 1, '/', iteration)  # 進捗状況の表示
        normal_x = x[~outlier_sample_flags, :]
        normal_y = y[~outlier_sample_flags]

        model = DCEBaggingRegressor(base_estimator=base_estimator, n_estimators=n_estimators,
                                    max_features=1.0, autoscaling_flag=autoscaling_flag,
                                    cv_flag=False, robust_flag=True, random_state=random_state)
        model.fit(normal_x, normal_y)
        estimated_y_all = model.predict_all(x)
        # 外れサンプルの判定
        estimated_y_all_normal = estimated_y_all[~outlier_sample_flags, :]
        estimated_y_median_normal = np.median(estimated_y_all_normal, axis=1)
        estimated_y_mad_normal = np.median(abs(estimated_y_all_normal - np.median(estimated_y_median_normal)))
        y_error = abs(y - np.median(estimated_y_all, axis=1))
        outlier_sample_flags = y_error > 3 * 1.4826 * estimated_y_mad_normal
        #        print('外れサンプル検出結果が一致した数 :', sum(outlier_sample_flags == previous_outlier_sample_flags))
        if sum(outlier_sample_flags == previous_outlier_sample_flags) == x.shape[0]:
            #            print('計算終了')
            break
        previous_outlier_sample_flags = outlier_sample_flags.copy()

    #    outlier_sample_flags = pd.DataFrame(outlier_sample_flags)
    #    outlier_sample_flags.columns = ['TRUE if outlier samples']
    return outlier_sample_flags

#
