# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Class of Variational Bayesian Gaussian Mixture Regression (VBGMR), which is supervised Variational Bayesian Gaussian Mixture Model (VBGMM)

import math

import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture


class VBGMR(BayesianGaussianMixture):

    def __init__(self, n_components=30, covariance_type='full',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=0.01, rep='mean', max_iter=100,
                 random_state=None, display_flag=False):
        super(VBGMR, self).__init__(n_components=n_components, covariance_type=covariance_type,
                                    weight_concentration_prior_type=weight_concentration_prior_type,
                                    weight_concentration_prior=weight_concentration_prior,
                                    max_iter=max_iter, random_state=random_state)

        self.display_flag = display_flag

    def predict(self, dataset, numbers_of_input_variables, numbers_of_output_variables):
        """
        Variational Bayesian Gaussian Mixture Regression (VBGMR) based on Variational Bayesian Gaussian Mixture Model (VBGMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis
    
        Parameters
        ----------
        vbgmm_model: mixture.gaussian_mixture.BayesianGaussianMixture
            VBGMM model constructed using scikit-learn
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
            all_covariances = np.empty([self.n_components, self.means_.shape[1], self.means_.shape[1]])
            for component_number in range(self.n_components):
                all_covariances[component_number, :, :] = np.diag(
                    self.covariances_[component_number] * np.ones(self.means_.shape[1]))

#        print(all_covariances.shape[2], len(numbers_of_input_variables), len(numbers_of_output_variables))
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
            if len(np.where(weights.sum(axis=0)==0)[0]) > 0:
                weights = np.ones(weights.shape)
            if np.isnan(weights.sum(axis=0)).any():
                weights = np.ones(weights.shape)
            if np.isinf(weights.sum(axis=0)).any():
                weights = np.ones(weights.shape)
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

    def predict_rep(self, dataset, numbers_of_input_variables, numbers_of_output_variables):
        """
        Variational Bayesian Gaussian Mixture Regression (VBGMR) based on Variational Bayesian Gaussian Mixture Model (VBGMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis. The way to calculate representative values can be set with 'rep'
    
        Parameters
        ----------
        vbgmm_model: mixture.gaussian_mixture.BayesianGaussianMixture
            VBGMM model constructed using scikit-learn
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
            all_covariances = np.empty([self.n_components, self.means_.shape[1], self.means_.shape[1]])
            for component_number in range(self.n_components):
                all_covariances[component_number, :, :] = np.diag(
                    self.covariances_[component_number] * np.ones(self.means_.shape[1]))

#        print(all_covariances.shape[2], len(numbers_of_input_variables), len(numbers_of_output_variables))
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
                                                                       input_covariances[component_number, :, :], allow_singular=True)
            if len(np.where(weights.sum(axis=0)==0)[0]) > 0:
                weights = np.ones(weights.shape)
            if np.isnan(weights.sum(axis=0)).any():
                weights = np.ones(weights.shape)
            if np.isinf(weights.sum(axis=0)).any():
                weights = np.ones(weights.shape)
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

        if self.rep == 'mean':
            values = mode_of_estimated_mean.copy()
        elif self.rep == 'mode':
            values = weighted_estimated_mean.copy()
        return values
    
    def cv_opt(self, dataset, numbers_of_input_variables, numbers_of_output_variables, covariance_types,
               numbers_of_components, weight_concentration_prior_types, weight_concentration_priors, fold_number):
        """
        Hyperparameter optimization for Variational Bayesian Gaussian Mixture Regression (VBGMR) using cross-validation
    
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
        numbers_of_components: list or numpy.array
            candidates of number of components in GMM
        weight_concentration_prior_types: list
            candidates of weight_concentration_prior_type in VBGMM such as ['dirichlet_process', 'dirichlet_distribution']
        weight_concentration_priors: list or numpy.array
            candidates of weight_concentration_prior in VBGMM
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
        reps = ['mean', 'mode']
        
        r2cvs = []
        hyperparameters = []
        for covariance_type in covariance_types:
            for number_of_components in numbers_of_components:
                for weight_concentration_prior_type in weight_concentration_prior_types:
                        for weight_concentration_prior in weight_concentration_priors:
                            for rep in reps:
                                self.covariance_type = covariance_type
                                self.n_components = number_of_components
                                self.weight_concentration_prior_type = weight_concentration_prior_type
                                self.weight_concentration_prior = weight_concentration_prior
                                self.rep = rep
                                hyperparameters.append([covariance_type, number_of_components, weight_concentration_prior_type, weight_concentration_prior, rep])
                                estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
                
                                min_number = math.floor(dataset.shape[0] / fold_number)
                                mod_number = dataset.shape[0] - min_number * fold_number
                                index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
                                if mod_number != 0:
                                    index = np.r_[index, np.arange(1, mod_number + 1, 1)]
                #                np.random.seed(999)
                                fold_index_in_cv = np.random.permutation(index)
                                np.random.seed()
                                for fold_number_in_cv in np.arange(1, fold_number + 1, 1):
                                    dataset_train_in_cv = autoscaled_dataset[fold_index_in_cv != fold_number_in_cv, :]
                                    dataset_test_in_cv = autoscaled_dataset[fold_index_in_cv == fold_number_in_cv, :]
                                    
                                    self.fit(dataset_train_in_cv)
                
                                    values = self.predict_rep(dataset_test_in_cv[:, numbers_of_input_variables],
                                                              numbers_of_input_variables, numbers_of_output_variables)
                
                                    estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = values  # 格納
                
                                y = np.ravel(autoscaled_dataset[:, numbers_of_output_variables])
                                y_pred = np.ravel(estimated_y_in_cv)
                                r2 = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
                                r2cvs.append(r2)
                                if self.display_flag:
                                    print(covariance_type, number_of_components, weight_concentration_prior_type, weight_concentration_prior, rep)
        r2cvs = np.nan_to_num(r2cvs, nan=-10**10)
        max_r2cv_number = np.where(r2cvs == np.max(r2cvs))[0][0]
        max_r2cv_hyperparameter = hyperparameters[max_r2cv_number]
        
        self.covariance_type = max_r2cv_hyperparameter[0]
        self.n_components = max_r2cv_hyperparameter[1]
        self.weight_concentration_prior_type = max_r2cv_hyperparameter[2]
        self.weight_concentration_prior = max_r2cv_hyperparameter[3]
        self.rep = max_r2cv_hyperparameter[4]
        self.r2cv = r2cvs[max_r2cv_number]
        
