# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import sys

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TransferLearningSample(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, x_source=None, y_source=None, autoscaling_flag=False, cv_flag=False):
        """
        Transfer Learning based on samples
        
        Parameters
        ----------
        base_estimator: object
            The base estimator in scikit-learn. If cv_flag is True, this must be the object of GridSearchCV
        x_source : numpy.array or pandas.DataFrame, shape (n_samples, n_features)
            Dataset of x to transfer samples
        y_source : numpy.array or pandas.DataFrame, shape (n_samples,)
            Dataset of y to transfer samples
        autoscaling_flag : boolen, default True
            If True, autoscaling is done, and if False, autoscaling is not done 
        cv_flag: boolen, default False
            If this is True, base_estimator must be the object of GridSearchCV
        """
        self.base_estimator = base_estimator
        self.x_source = x_source
        self.y_source = y_source
        self.autoscaling_flag = autoscaling_flag
        self.cv_flag = cv_flag
        self.model = None
        self.combined_x = None
        self.combined_y = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, x, y):
        """
        Transfer Learning based on samples
        
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

        x_source = np.array(self.x_source)
        y_source = np.array(self.y_source)
        x_train_target = np.array(x)
        y_train_target = np.array(y)

        x_source_arranged = np.c_[x_source, x_source, np.zeros(x_source.shape)]
        x_train_target_arranged = np.c_[x_train_target, np.zeros(x_train_target.shape), x_train_target]
        x = np.r_[x_source_arranged, x_train_target_arranged]
        y = np.r_[y_source, y_train_target]
        
        self.combined_x = x
        self.combined_y = y
        
        if self.autoscaling_flag:
            self.x_std = x.std(axis=0, ddof=1)
            if (self.x_std == 0).any():
                zero_variance_variable_number = list(np.where(self.x_std == 0)[0])
                sys.exit('X-variables of the number {0} have zero variance. Please delete those X-variables.'.format(
                    zero_variance_variable_number))
            self.x_mean = x.mean(axis=0)
            self.y_mean = y.mean()
            self.y_std = y.std(ddof=1)
            autoscaled_x = (x - self.x_mean) / self.x_std
            autoscaled_y = (y - self.y_mean) / self.y_std
        else:
            autoscaled_x = x.copy()
            autoscaled_y = y.copy()

        if self.cv_flag:
            # hyperparameter optimization with cross-validation
            self.base_estimator.fit(autoscaled_x, autoscaled_y)
            # modeling
            self.model = getattr(self.base_estimator, 'estimator')
            hyperparameters = list(self.base_estimator.best_params_.keys())
            for hyperparameter in hyperparameters:
                setattr(self.model, hyperparameter, self.base_estimator.best_params_[hyperparameter])

        else:
            self.model = self.base_estimator

        self.model.fit(autoscaled_x, autoscaled_y)

    def predict(self, x, target_flag=True):
        """
        Transfer Learning based on samples
        
        Predict y-values of samples using transfer Learning based on samples
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
        target_flag : boolean, default True
            if this is True, x of target test data is estimated,
            if this is False, x of supporing test data is estimated
    
        Returns
        -------
        estimated_y : numpy.array
            k x 1 vector of estimated y-values of test data
        """

        x = np.array(x)
        if target_flag:
            x_test = np.c_[x, np.zeros(x.shape), x]
        else:
            x_test = np.c_[x, x, np.zeros(x.shape)]

        if self.autoscaling_flag:
            autoscaled_x_test = (x_test - self.x_mean) / self.x_std
        else:
            autoscaled_x_test = x_test.copy()

        estimated_y = np.ndarray.flatten(self.model.predict(autoscaled_x_test))

        if self.autoscaling_flag:
            estimated_y = estimated_y * self.y_std + self.y_mean

        return estimated_y
