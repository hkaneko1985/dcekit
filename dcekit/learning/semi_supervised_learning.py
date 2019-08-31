# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin


class SemiSupervisedLearningLowDimension(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, base_dimension_reductioner=None, x_unsupervised=None,
                 autoscaling_flag=True, cv_flag=False, ad_flag=False, k=5, within_ad_rate=0.997):
        """
        Semi-Supervised Learning in Low Dimensional transform and regression
        
        Parameters
        ----------
        base_estimator: object
            The base estimator in scikit-learn. If cv_flag is True, this must be the object of GridSearchCV
        base_dimension_reductioner: object
            The base model of dimension reduction in scikit-learn with 'transform' function. 
            GTM in DCEKit can be used.
        x_unsupervised : numpy.array or pandas.DataFrame, shape (n_samples, n_features)
            Unsupervised dataset of x
        autoscaling_flag : boolen, default True
            If True, autoscaling is done, and if False, autoscaling is not done 
        cv_flag: boolen, default False
            If True, base_estimator must be the object of GridSearchCV
        ad_flag: boolen, default False
            If True, AD of k-NN is considered and unsupervised samples within AD are selected first.
            https://www.sciencedirect.com/science/article/abs/pii/S0169743919300875
        k: int, default 5
            k in k-NN. This is required when ad_flag is True
        within_ad_rate: float, default 0.997
            Rate of trainng samples within AD, which is used to determine the threshold of k-NN distance as AD
        """
        self.base_estimator = base_estimator
        self.base_dimension_reductioner = base_dimension_reductioner
        self.x_unsupervised = x_unsupervised
        self.autoscaling_flag = autoscaling_flag
        self.cv_flag = cv_flag
        self.ad_flag = ad_flag
        self.k = k
        self.within_ad_rate = within_ad_rate
        self.model = None
        self.combined_x = None
        self.transformed_x = None
        self.x_mean = None
        self.x_std = None
        self.z_mean = None
        self.z_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, x, y):
        """
        Semi-Supervised Learning in Low Dimensional transform and regression
        
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

        x_unsupervised = np.array(self.x_unsupervised)
        x_supervised = np.array(x)
        y = np.array(y)
        
        if self.ad_flag:
            # select unsupervised samples using AD based on k-NN
            if self.autoscaling_flag:
                if (x_supervised.std(axis=0, ddof=1) == 0).any():
                    zero_variance_variable_number = list(np.where(x_supervised.std(axis=0, ddof=1) == 0)[0])
                    sys.exit('X-variables of the number {0} have zero variance. Please delete those X-variables.'.format(
                        zero_variance_variable_number))
                # autoscaling      
                autoscaled_supervised_x = (x_supervised - x_supervised.mean(axis=0)) / x_supervised.std(axis=0, ddof=1)
                autoscaled_unsupervised_x = (x_unsupervised - x_supervised.mean(axis=0)) / x_supervised.std(axis=0, ddof=1)
            else:
                autoscaled_supervised_x = x_supervised.copy()
                autoscaled_unsupervised_x = x_unsupervised.copy()            
            # set AD
            distance_between_autoscaled_supervised_x = cdist(autoscaled_supervised_x, autoscaled_supervised_x)
            distance_between_autoscaled_supervised_x.sort()
            knn_distance = np.mean(distance_between_autoscaled_supervised_x[:, 1:self.k + 1], axis=1)
            # set AD threshold
            knn_distance.sort()
            if self.within_ad_rate == 0:
                ad_threshold = 0
            else:
                ad_threshold = knn_distance[round(distance_between_autoscaled_supervised_x.shape[0] * self.within_ad_rate) - 1]
            # AD check
            distance_between_autoscaled_x = cdist(autoscaled_unsupervised_x, autoscaled_supervised_x)
            distance_between_autoscaled_x.sort()
            knn_distance_x = np.mean(distance_between_autoscaled_x[:, 0:self.k], axis=1)
            # select unsupervised samples
            selected_unsupervised_sample_numbers = np.where(knn_distance_x <= ad_threshold)[0]
            x_unsupervised = x_unsupervised[selected_unsupervised_sample_numbers, :].copy()

        x_all = np.r_[x_supervised, x_unsupervised]
        self.combined_x = x_all

        if self.autoscaling_flag:
            self.x_std = x_all.std(axis=0, ddof=1)
            if (self.x_std == 0).any():
                zero_variance_variable_number = list(np.where(self.x_std == 0)[0])
                sys.exit('X-variables of the number {0} have zero variance. Please delete those X-variables.'.format(
                    zero_variance_variable_number))
            self.x_mean = x_all.mean(axis=0)
            self.y_mean = y.mean()
            self.y_std = y.std(ddof=1)
            autoscaled_x_all = (x_all - self.x_mean) / self.x_std
            autoscaled_y = (y - self.y_mean) / self.y_std
        else:
            autoscaled_x_all = x_all.copy()
            autoscaled_y = y.copy()

        # Dimension reduction
        transformed_x_all = self.base_dimension_reductioner.fit_transform(autoscaled_x_all)
        self.transformed_x = transformed_x_all
        x_train = transformed_x_all[0:x.shape[0], :]

        if self.autoscaling_flag:
            self.z_std = x_train.std(axis=0, ddof=1)
            self.z_mean = x_train.mean(axis=0)
            autoscaled_x = (x_train - self.z_mean) / self.z_std
        else:
            autoscaled_x = x_train.copy()

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

    def predict(self, x):
        """
        Semi-Supervised Learning in Low Dimensional transform and regression
        
        Predict y-values of samples
    
        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            k x n matrix of X-variables of test data, which is autoscaled with training data,
            and k is the number of test samples
    
        Returns
        -------
        estimated_y : numpy.array
            k x 1 vector of estimated y-values of test data
        """

        x = np.array(x)
        if self.autoscaling_flag:
            autoscaled_x = (x - self.x_mean) / self.x_std
        else:
            autoscaled_x = x.copy()

        # Dimension reduction
        transformed_x = self.base_dimension_reductioner.transform(autoscaled_x)

        if self.autoscaling_flag:
            autoscaled_x_test = (transformed_x - self.z_mean) / self.z_std
        else:
            autoscaled_x_test = transformed_x.copy()

        estimated_y = np.ndarray.flatten(self.model.predict(autoscaled_x_test))

        if self.autoscaling_flag:
            estimated_y = estimated_y * self.y_std + self.y_mean

        return estimated_y
