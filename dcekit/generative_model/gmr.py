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
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize, LinearConstraint

class GMR(GaussianMixture):

    def __init__(self, n_components=1, covariance_type='full', rep='mean',
                 max_iter=100, random_state=None, display_flag=False):
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

        estimated_mean_for_all_components, _, weights = self.predict_mog(dataset, numbers_of_input_variables, numbers_of_output_variables)
        
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
        
        return mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights

    def predict_rep(self, dataset, numbers_of_input_variables, numbers_of_output_variables):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis to maximize PDF 
    
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

        mode_of_estimated_mean, weighted_estimated_mean, _, _ = self.predict(dataset, numbers_of_input_variables, numbers_of_output_variables)
        if self.rep == 'mean':
            values = mode_of_estimated_mean.copy()
        elif self.rep == 'mode':
            values = weighted_estimated_mean.copy()
        return values
    
    def predict_true(self, dataset, numbers_of_input_variables, numbers_of_output_variables, bounds=[]):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis. The way to calculate representative values can be set with 'rep' 
    
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
        bounds : list
            (does not work) upper and lower bounds of output variables
    
        Returns
        -------
        estimated_values : numpy.array
            (autoscaled) m x k matrix of output variables estimated using true GMR,
            k is the number of output variables
        """
        
        if len(bounds) == 0:
            for i in range(len(numbers_of_output_variables)):
                bounds.append([-float('inf'), float('inf')])
        
        mode_of_estimated_mean_of_y, weighted_estimated_mean_of_y, _, _ = \
                self.predict(dataset, numbers_of_input_variables, numbers_of_output_variables)
        estimated_means, estimated_covariances, weights = self.predict_mog(dataset, numbers_of_input_variables, numbers_of_output_variables)
        
        estimated_values_mode = self.true_gmr(estimated_means, estimated_covariances, weights, mode_of_estimated_mean_of_y)
        estimated_values_mean = self.true_gmr(estimated_means, estimated_covariances, weights, weighted_estimated_mean_of_y)
        
        estimated_values = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
        for i in range(dataset.shape[0]):
            tmp_func = np.zeros(4)
            tmp_estimated_values = []
            tmp_func[0] = self.true_gmr_obj_func(mode_of_estimated_mean_of_y[i, :], estimated_means[:, i, :], estimated_covariances, weights[:, i])
            tmp_estimated_values.append(mode_of_estimated_mean_of_y[i, :])
            tmp_func[1] = self.true_gmr_obj_func(weighted_estimated_mean_of_y[i, :], estimated_means[:, i, :], estimated_covariances, weights[:, i])
            tmp_estimated_values.append(weighted_estimated_mean_of_y[i, :])
            tmp_func[2] = self.true_gmr_obj_func(estimated_values_mode[i, :], estimated_means[:, i, :], estimated_covariances, weights[:, i])
            tmp_estimated_values.append(estimated_values_mode[i, :])
            tmp_func[3] = self.true_gmr_obj_func(estimated_values_mean[i, :], estimated_means[:, i, :], estimated_covariances, weights[:, i])
            tmp_estimated_values.append(estimated_values_mean[i, :])
            best = np.where(tmp_func == min(tmp_func))[0][0]
            estimated_values[i, :] = tmp_estimated_values[best]

        return estimated_values
    
    def true_gmr_obj_func(self, variable, means, covariances, weights):
        """
        Objective function of True GMR
        
        Parameters
        ----------
    
        Returns
        -------
        -logsumexp : float
        
        """
        
        tmps = []
        for i in range(covariances.shape[0]):
            tmp = np.log(weights[i]) + multivariate_normal.logpdf(variable, mean=means[i, :], cov=covariances[i, :, :])
            tmps.append(tmp)
        value = -logsumexp(tmps)
        
        return value
    
    def true_gmr(self, means_all, covariances, weights_all, init_values, bounds=[]):
        means_all = np.array(means_all)
        covariances = np.array(covariances)
        weights_all = np.array(weights_all)
        init_values = np.array(init_values)
        number_of_samples = means_all.shape[1]
        predicted_results = np.zeros([number_of_samples, init_values.shape[1]])
        for i in range(number_of_samples):
            means = means_all[:, i, :]
            weights = weights_all[:, i]
            init_value = init_values[i, :]
            if len(bounds) == 0:
                pred_results = minimize(self.true_gmr_obj_func,
                                        x0=init_value,
                                        args=(means, covariances, weights),
    #                                    bounds=bounds,
    #                                    constraints=LinearConstraint(np.ones(init_values.shape[1]), 1, 1),
                                        method='SLSQP'
                                        )

            predicted_results[i, :] = pred_results.x.copy()

        return predicted_results
    
    def predict_logpdf(self, dataset, numbers_of_input_variables, numbers_of_output_variables, estimated_results):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis. Predition results are given as probability density function
    
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
        logpdf : numpy.array
            (autoscaled) m vector of log of probability density function,
        """
        
        dataset = np.array(dataset)
        if dataset.ndim == 0:
            dataset = np.reshape(dataset, (1, 1))
        elif dataset.ndim == 1:
            dataset = np.reshape(dataset, (1, dataset.shape[0]))
        estimated_means_all, estimated_covariances, weights_all = self.predict_mog(dataset, numbers_of_input_variables, numbers_of_output_variables)

        estimated_results = np.array(estimated_results)
        if estimated_results.ndim == 0:
            estimated_results = np.reshape(estimated_results, (1, 1))
        elif estimated_results.ndim == 1:
            estimated_results = np.reshape(estimated_results, (1, estimated_results.shape[0]))
            
        logpdf = np.zeros(estimated_results.shape[0])
        for sample_number in range(estimated_results.shape[0]):
            estimated_means = estimated_means_all[:, sample_number, :]
            weights = weights_all[:, sample_number]        
            tmps = []
            for component_number in range(estimated_covariances.shape[0]):
                tmp = np.log(weights[component_number]) + multivariate_normal.logpdf(estimated_results[sample_number, :], mean=estimated_means[component_number, :], cov=estimated_covariances[component_number, :, :])
                tmps.append(tmp)
            logpdf[sample_number] = logsumexp(tmps)

        return logpdf
    
    def predict_pdf(self, dataset, numbers_of_input_variables, numbers_of_output_variables, estimated_results):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis. Predition results are given as probability density function
    
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
        pdf : numpy.array
            (autoscaled) m vector of probability density function,
        """
        
        logpdf = self.predict_logpdf(dataset, numbers_of_input_variables, numbers_of_output_variables, estimated_results)

        return np.exp(logpdf)
    
    def predict_mog(self, dataset, numbers_of_input_variables, numbers_of_output_variables):
        """
        Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
        
        Predict values of variables for forward analysis (regression) and inverse analysis. Predition results are given as mixture of Gaussians
    
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
        estimated_means : numpy.array
            (autoscaled) l x m x k matrix of output variables estimated for all components,
            k is the number of output variables
        estimated_covariances : numpy.array
            (autoscaled) l x k x k variance-covariance matrix of output variables estimated for all components,
        weights : numpy.array
            l x m matrix of weights,
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

        estimated_means = np.zeros([self.n_components, dataset.shape[0], len(numbers_of_output_variables)])
        estimated_covariances = np.zeros([self.n_components, len(numbers_of_output_variables), len(numbers_of_output_variables)])
        weights = np.zeros([self.n_components, dataset.shape[0]])
#        print(all_covariances.shape[2], len(numbers_of_input_variables), len(numbers_of_output_variables))
        if all_covariances.shape[2] == len(numbers_of_input_variables) + len(numbers_of_output_variables):
            input_output_covariances = all_covariances[:, numbers_of_input_variables, :]
            input_covariances = input_output_covariances[:, :, numbers_of_input_variables]
            input_output_covariances = input_output_covariances[:, :, numbers_of_output_variables]
            output_input_covariances = all_covariances[:, numbers_of_output_variables, :]
            output_covariances = output_input_covariances[:, :, numbers_of_output_variables]
            output_input_covariances = output_input_covariances[:, :, numbers_of_input_variables]
            
            # estimated means and weights for all components
            for component_number in range(self.n_components):
                estimated_means[component_number, :, :] = output_means[component_number, :] + (
                        dataset - input_means[component_number, :]).dot(
                    np.linalg.inv(input_covariances[component_number, :, :])).dot(
                    input_output_covariances[component_number, :, :])
                estimated_covariances[component_number, :, :] = output_covariances[component_number, :, :] - output_input_covariances[component_number, :, :].dot(
                        np.linalg.inv(input_covariances[component_number, :, :])).dot(input_output_covariances[component_number, :, :])
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

        return estimated_means, estimated_covariances, weights
    
    def cv_opt(self, dataset, numbers_of_input_variables, numbers_of_output_variables, covariance_types,
               numbers_of_components, fold_number):
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
        numbers_of_components: list or numpy.array
            candidates of number of components in GMM
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
        
        min_number = math.floor(dataset.shape[0] / fold_number)
        mod_number = dataset.shape[0] - min_number * fold_number
        index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
        if mod_number != 0:
            index = np.r_[index, np.arange(1, mod_number + 1, 1)]
#        np.random.seed(999)
        fold_index_in_cv = np.random.permutation(index)
        np.random.seed()
                    
        r2cvs = []
        hyperparameters = []
        for covariance_type in covariance_types:
            for number_of_components in numbers_of_components:
                for rep in reps:
                    self.covariance_type = covariance_type
                    self.n_components = number_of_components
                    self.rep = rep
                    
                    hyperparameters.append([covariance_type, number_of_components, rep])
                    estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
                    
                    for fold_number_in_cv in np.arange(1, fold_number + 1, 1):
                        dataset_train_in_cv = autoscaled_dataset[fold_index_in_cv != fold_number_in_cv, :]
                        dataset_test_in_cv = autoscaled_dataset[fold_index_in_cv == fold_number_in_cv, :]
                        self.fit(dataset_train_in_cv)
                        try:
                            values = self.predict_rep(dataset_test_in_cv[:, numbers_of_input_variables],
                                                      numbers_of_input_variables, numbers_of_output_variables)
                        except:
                            values = np.ones([dataset_test_in_cv.shape[0], len(numbers_of_output_variables)]) * (- 10 ** 10)

                        estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = values  # 格納
    
                    y = np.ravel(autoscaled_dataset[:, numbers_of_output_variables])
                    y_pred = np.ravel(estimated_y_in_cv)
                    r2 = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
                    r2cvs.append(r2)
                    if self.display_flag:
                        print(covariance_type, number_of_components, rep)
        r2cvs = np.nan_to_num(r2cvs, nan=-10**10)
        max_r2cv_number = np.where(r2cvs == np.max(r2cvs))[0][0]
        max_r2cv_hyperparameter = hyperparameters[max_r2cv_number]
        
        self.covariance_type = max_r2cv_hyperparameter[0]
        self.n_components = max_r2cv_hyperparameter[1]
        self.rep = max_r2cv_hyperparameter[2]
        self.r2cv = r2cvs[max_r2cv_number]
