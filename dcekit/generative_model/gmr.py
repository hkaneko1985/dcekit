# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Class of Gaussian Mixture Regression (GMR), which is supervised Gaussian Mixture Model (GMM)

import math

import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GMR(GaussianMixture):

    def __init__(self, covariance_type='full', n_components=10, max_iter=100,
                 random_state=None, display_flag=False):
        super(GMR, self).__init__(n_components=n_components, covariance_type=covariance_type,
                                  max_iter=max_iter, random_state=random_state)

        self.display_flag = display_flag

    def predict(self, dataset, numbers_of_input_variables, numbers_of_output_variables):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis
    
        Parameters
        ----------
        gmm_model: mixture.gaussian_mixture.GaussianMixture
            GMM model constructed using scikit-learn
        dataset: numpy.array or pandas.DataFrame
            (autoscaled) m x n matrix of dataset of training data or test data,
            m is the number of sammples and
            n is the number of input variables
            When this is X-variables, it is forward analysis (regression) and
            when this is Y-variables, it is inverse analysis
        numbers_of_input_variables: list
            vector of numbers of input variables
            When this is numbers of X-variables, it is forward analysis (regression) and
            when this is numbers of Y-variables, it is inverse analysis
        numbers_of_output_variables: list
            vector of numbers of output variables
            When this is numbers of Y-variables, it is forward analysis (regression) and
            when this is numbers of X-variables, it is inverse analysis
    
        Returns
        -------
        mode_of_estimated_mean : numpy.array
            (autoscaled) m x k matrix of output variables estimated using mode of weights,
            k is the number of output variables
        weighted_estimated_mean : numpy.array
            (autoscaled) m x k matrix of output variables estimated using weighted mean,
        estimated_mean_for_all_components : numpy.array
            (autoscaled) l x m x k matrix of output variables estimated for all components,
        weights : numpy.array
            m x l matrix of weights,
        """

        dataset = np.array(dataset)
        if dataset.ndim == 0:
            dataset = np.reshape(dataset, (1, 1))
        elif dataset.ndim == 1:
            dataset = np.reshape(dataset, (1, dataset.shape[0]))

        input_means = self.means_[:, numbers_of_input_variables]
        output_means = self.means_[:, numbers_of_output_variables]

        if self.covariance_type == 'full':
            all_covariances = self.covariances_
        elif self.covariance_type == 'diag':
            all_covariances = np.empty(
                [self.n_components, self.covariances_.shape[1], self.covariances_.shape[1]])
            for component_number in range(self.n_components):
                all_covariances[component_number, :, :] = np.diag(self.covariances_[component_number, :])
        elif self.covariance_type == 'tied':
            all_covariances = np.tile(self.covariances_, (self.n_components, 1, 1))
        elif self.covariance_type == 'spherical':
            all_covariances = np.empty([self.n_components, len(self.means_), len(self.means_)])
            for component_number in range(self.n_components):
                all_covariances[component_number, :, :] = np.diag(
                    self.covariances_[component_number] * np.ones(len(self.means_)))

        if all_covariances.shape[2] == len(numbers_of_input_variables) + len(numbers_of_output_variables):
            input_output_covariances = all_covariances[:, numbers_of_input_variables, :]
            input_covariances = input_output_covariances[:, :, numbers_of_input_variables]
            input_output_covariances = input_output_covariances[:, :, numbers_of_output_variables]

            # estimated means and weights for all components
            estimated_mean_for_all_components = np.empty(
                [self.n_components, dataset.shape[0], len(numbers_of_output_variables)])
            weights = np.empty([self.n_components, dataset.shape[0]])
            for component_number in range(self.n_components):
                estimated_mean_for_all_components[component_number, :, :] = output_means[component_number, :] + (
                        dataset - input_means[component_number, :]).dot(
                    np.linalg.inv(input_covariances[component_number, :, :])).dot(
                    input_output_covariances[component_number, :, :])
                weights[component_number, :] = self.weights_[component_number] * \
                                               multivariate_normal.pdf(dataset,
                                                                       input_means[component_number, :],
                                                                       input_covariances[component_number, :, :])

            weights = weights / weights.sum(axis=0)

            # calculate mode of estimated means and weighted estimated means
            mode_of_estimated_mean = np.empty([dataset.shape[0], len(numbers_of_output_variables)])
            weighted_estimated_mean = np.empty([dataset.shape[0], len(numbers_of_output_variables)])
            index_for_mode = np.argmax(weights, axis=0)
            for sample_number in range(dataset.shape[0]):
                mode_of_estimated_mean[sample_number, :] = estimated_mean_for_all_components[
                                                           index_for_mode[sample_number],
                                                           sample_number, :]
                weighted_estimated_mean[sample_number, :] = weights[:, sample_number].dot(
                    estimated_mean_for_all_components[:, sample_number, :])
        else:
            mode_of_estimated_mean = np.ones([dataset.shape[0], len(numbers_of_output_variables)]) * -99999
            weighted_estimated_mean = np.ones([dataset.shape[0], len(numbers_of_output_variables)]) * -99999
            weights = np.zeros([self.n_components, dataset.shape[0]])
            estimated_mean_for_all_components = np.zeros(
                [self.n_components, dataset.shape[0], len(numbers_of_output_variables)])

        return mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights

    def cv_opt(self, dataset, numbers_of_input_variables, numbers_of_output_variables, covariance_types,
              max_number_of_components, fold_number):
        """
        Hyperparameter optimization for Gaussian Mixture Regression (GMR) using cross-validation
    
        Parameters
        ----------
        dataset: numpy.array or pandas.DataFrame
            m x n matrix of dataset of training data,
            m is the number of sammples and
            n is the number of both input and output variables
        numbers_of_input_variables: list
            vector of numbers of input variables
            When this is numbers of X-variables, it is forward analysis (regression) and
            when this is numbers of Y-variables, it is inverse analysis
        numbers_of_output_variables: list
            vector of numbers of output variables
            When this is numbers of Y-variables, it is forward analysis (regression) and
            when this is numbers of X-variables, it is inverse analysis
        covariance_types: list
            candidates of covariance types such as ['full', 'diag', 'tied', 'spherical']
        max_number_of_components: int
            number of maximum components in GMM
        fold_number: int
            number of fold in cross-validation        
    
        Returns
        -------
        best_covariance_type : str
            best covariance type
        best_number_of_components : int
            best number of components
        """

        dataset = np.array(dataset)
        autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)

        r2cvs = []
        for covariance_type in covariance_types:
            for number_of_components in range(max_number_of_components):
                estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])

                min_number = math.floor(dataset.shape[0] / fold_number)
                mod_number = dataset.shape[0] - min_number * fold_number
                index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
                if mod_number != 0:
                    index = np.r_[index, np.arange(1, mod_number + 1, 1)]
                #            np.random.seed(999)
                fold_index_in_cv = np.random.permutation(index)
                np.random.seed()
                for fold_number_in_cv in np.arange(1, fold_number + 1, 1):
                    dataset_train_in_cv = autoscaled_dataset[fold_index_in_cv != fold_number_in_cv, :]
                    dataset_test_in_cv = autoscaled_dataset[fold_index_in_cv == fold_number_in_cv, :]
                    self.covariance_type = covariance_type
                    self.n_components = number_of_components + 1
                    self.fit(dataset_train_in_cv)

                    mode_of_estimated_mean_of_y, weighted_estimated_mean_of_y, estimated_mean_of_y_for_all_components, weights_for_x = \
                        self.predict(dataset_test_in_cv[:, numbers_of_input_variables],
                                     numbers_of_input_variables, numbers_of_output_variables)

                    estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = mode_of_estimated_mean_of_y  # 格納

                y = np.ravel(autoscaled_dataset[:, numbers_of_output_variables])
                y_pred = np.ravel(estimated_y_in_cv)
                r2 = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
                r2cvs.append(r2)
                if self.display_flag:
                    print(covariance_type, number_of_components + 1)
        max_r2cv_number = np.where(r2cvs == np.max(r2cvs))[0][0]
        best_covariance_type = covariance_types[max_r2cv_number // max_number_of_components]
        best_number_of_components = max_r2cv_number % max_number_of_components + 1

        self.covariance_type = best_covariance_type
        self.n_components = best_number_of_components
