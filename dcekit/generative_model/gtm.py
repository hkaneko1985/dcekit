# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# GTM (generative topographic mapping) class

import math

import numpy as np
import numpy.matlib
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA


class GTM:

    def __init__(self, shape_of_map=[30, 30], shape_of_rbf_centers=[10, 10],
                 variance_of_rbfs=4, lambda_in_em_algorithm=0.001,
                 number_of_iterations=200, display_flag=True, sparse_flag=False, rep='mean'):
        self.shape_of_map = shape_of_map
        self.shape_of_rbf_centers = shape_of_rbf_centers
        self.variance_of_rbfs = variance_of_rbfs
        self.lambda_in_em_algorithm = lambda_in_em_algorithm
        self.number_of_iterations = number_of_iterations
        self.display_flag = display_flag
        self.sparse_flag = sparse_flag
        self.rep = rep

    def calculate_grids(self, num_x, num_y):
        """
        Calculate grid coordinates on the GTM map
        
        Parameters
        ----------
        num_x : int
            number_of_x_grids
        num_y : int
            number_of_y_grids
        """
        grids_x, grids_y = np.meshgrid(np.arange(0.0, num_x), np.arange(0.0, num_y))
        grids = np.c_[np.ndarray.flatten(grids_x)[:, np.newaxis],
                      np.ndarray.flatten(grids_y)[:, np.newaxis]]
        max_grids = grids.max(axis=0)
        grids[:, 0] = 2 * (grids[:, 0] - max_grids[0] / 2) / max_grids[0]
        grids[:, 1] = 2 * (grids[:, 1] - max_grids[1] / 2) / max_grids[1]
        return grids

    def fit(self, input_dataset):
        """
        Train the GTM map
                
        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.
        
        Returns
        ----------
        self : returns an instance of self.
        """
        input_dataset = np.array(input_dataset)
        self.success_flag = True
        self.shape_of_map = [int(self.shape_of_map[0]), int(self.shape_of_map[1])]
        self.shape_of_rbf_centers = [int(self.shape_of_rbf_centers[0]), int(self.shape_of_rbf_centers[1])]

        # make rbf grids
        self.rbf_grids = self.calculate_grids(self.shape_of_rbf_centers[0],
                                              self.shape_of_rbf_centers[1])

        # make map grids
        self.map_grids = self.calculate_grids(self.shape_of_map[0],
                                              self.shape_of_map[1])

        # calculate phi of map_grids and rbf_grids
        distance_between_map_and_rbf_grids = cdist(self.map_grids, self.rbf_grids,
                                                   'sqeuclidean')
        self.phi_of_map_rbf_grids = np.exp(-distance_between_map_and_rbf_grids / 2.0
                                           / self.variance_of_rbfs)

        # PCA for initializing W and beta
        pca_model = PCA(n_components=3)
        pca_model.fit_transform(input_dataset)
        if np.linalg.matrix_rank(self.phi_of_map_rbf_grids) < min(self.phi_of_map_rbf_grids.shape):
            self.success_flag = False
            return
        self.W = np.linalg.pinv(self.phi_of_map_rbf_grids).dot(
            self.map_grids.dot(pca_model.components_[0:2, :]))
        self.beta = min(pca_model.explained_variance_[2], 1 / (
                (
                        cdist(self.phi_of_map_rbf_grids.dot(self.W),
                              self.phi_of_map_rbf_grids.dot(self.W))
                        + np.diag(np.ones(np.prod(self.shape_of_map)) * 10 ** 100)
                ).min(axis=0).mean() / 2))
        self.bias = input_dataset.mean(axis=0)

        self.mixing_coefficients = np.ones(np.prod(self.shape_of_map)) / np.prod(self.shape_of_map)

        # EM algorithm
        phi_of_map_rbf_grids_with_one = np.c_[self.phi_of_map_rbf_grids,
                                              np.ones((np.prod(self.shape_of_map), 1))]
        for iteration in range(self.number_of_iterations):
            responsibilities = self.responsibility(input_dataset)

            phi_t_G_phi_etc = phi_of_map_rbf_grids_with_one.T.dot(
                np.diag(responsibilities.sum(axis=0)).dot(phi_of_map_rbf_grids_with_one)
            ) + self.lambda_in_em_algorithm / self.beta * np.identity(
                phi_of_map_rbf_grids_with_one.shape[1])
            if 1 / np.linalg.cond(phi_t_G_phi_etc) < 10 ** -15:
                self.success_flag = False
                break
            self.W_with_one = np.linalg.inv(phi_t_G_phi_etc).dot(
                phi_of_map_rbf_grids_with_one.T.dot(responsibilities.T.dot(input_dataset)))
            self.beta = input_dataset.size / (responsibilities
                                              * cdist(input_dataset,
                                                      phi_of_map_rbf_grids_with_one.dot(self.W_with_one)) ** 2).sum()

            self.W = self.W_with_one[:-1, :]
            self.bias = self.W_with_one[-1, :]
            if self.sparse_flag == True:
                self.mixing_coefficients = sum(responsibilities) / input_dataset.shape[0]

            if self.display_flag:
                print("{0}/{1} ... likelihood: {2}".format(iteration + 1, self.number_of_iterations,
                                                           self.likelihood_value))

    def calculate_distance_between_phi_w_and_input_distances(self, input_dataset):
        """
        Calculate distance between phi*W
        
        Parameters
        ----------
        input_dataset : numpy.array
             Training dataset for GTM.
             
        Returns
        -------
        distance : distance between phi*W
        """
        distance = cdist(
            input_dataset,
            self.phi_of_map_rbf_grids.dot(self.W)
            + np.ones((np.prod(self.shape_of_map), 1)).dot(
                np.reshape(self.bias, (1, len(self.bias)))
            ),
            'sqeuclidean')
        return distance

    def means_modes(self, input_dataset):
        """
        Get means and modes

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             input_dataset must be autoscaled.

        Returns
        -------
        means : numpy.array, shape (n_samples, 2)
            Coordinate of means of input_dataset for each sample.
            
        modes : numpy.array, shape (n_samples, 2)
            Grid of modes of input_dataset for each sample.

        """
        input_dataset = np.array(input_dataset)
        responsibilities = self.responsibility(input_dataset)
        means = responsibilities.dot(self.map_grids)
        modes = self.map_grids[responsibilities.argmax(axis=1), :]
        
        return means, modes
    
    def fit_transform(self, x, mean_flag=True):
        """
        Fit GTM model and transform X

        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            input_dataset must be autoscaled.
        mean_flag ; boolean, default True
            If True, output is mean, and if False, output is mode

        Returns
        -------
        means or modes : numpy.array, shape (n_samples, 2)
            Coordinate of means or modes of x for each sample.

        """
        
        self.fit(x)
        means, modes = self.means_modes(x)
        if mean_flag:
            return means
        else:
            return modes
    
    def transform(self, x, mean_flag=True):
        """
        Transform X using constructed GTM model

        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            input_dataset must be autoscaled.
        mean_flag ; boolen, default True
            If True, output is mean, and if False, output is mode

        Returns
        -------
        means or modes : numpy.array, shape (n_samples, 2)
            Coordinate of means or modes of x for each sample.

        """
        
        means, modes = self.means_modes(x)
        if mean_flag:
            return means
        else:
            return modes
        
    def responsibility(self, input_dataset):
        """
        Get responsibilities and likelihood.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.

        Returns
        -------
        reponsibilities : numpy.array
            Responsibilities of input_dataset for each grid point.
        likelihood_value : float
            likelihood of input_dataset.
        """
        input_dataset = np.array(input_dataset)
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)
        rbf_for_responsibility = np.exp(-self.beta / 2.0 * distance) * self.mixing_coefficients
        sum_of_rbf_for_responsibility = rbf_for_responsibility.sum(axis=1)
        zero_sample_index = np.where(sum_of_rbf_for_responsibility == 0)[0]
        if len(zero_sample_index):
            sum_of_rbf_for_responsibility[zero_sample_index] = 1
            rbf_for_responsibility[zero_sample_index, :] = 1 / rbf_for_responsibility.shape[1]

        reponsibilities = rbf_for_responsibility / np.reshape(sum_of_rbf_for_responsibility,
                                                              (rbf_for_responsibility.shape[0], 1))
        likelihood_value_tmp = (input_dataset.shape[1] / 2.0) * np.log(self.beta / 2.0 / np.pi) + np.log(sum_of_rbf_for_responsibility)
        self.likelihood_value = likelihood_value_tmp.sum()

        return reponsibilities

    def likelihood(self, input_dataset):
        """
        Get likelihood.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.

        Returns
        -------
        likelihood : float
            likelihood of input_dataset.
        """
        input_dataset = np.array(input_dataset)
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)
        
        rbf_for_responsibility = np.exp(-self.beta / 2.0 * distance) * self.mixing_coefficients
        sum_of_rbf_for_responsibility = rbf_for_responsibility.sum(axis=1)
        likelihood_value_tmp = (input_dataset.shape[1] / 2.0) * np.log(self.beta / 2.0 / np.pi) + np.log(sum_of_rbf_for_responsibility)
        return likelihood_value_tmp.sum()

    def mlr(self, X, y):
        """
        Train the MLR model
        
        Parameters
        ----------
        X, y : numpy.array or pandas.DataFrame
            Both X and y must NOT be autoscaled.
        """
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))
        # autoscaling
        self.Xmean = X.mean(axis=0)
        self.Xstd = X.std(axis=0, ddof=1)
        autoscaled_X = (X - self.Xmean) / self.Xstd
        self.y_mean = y.mean(axis=0)
        self.ystd = y.std(axis=0, ddof=1)
        autoscaled_y = (y - self.y_mean) / self.ystd
        self.regression_coefficients = np.linalg.inv(
            np.dot(autoscaled_X.T, autoscaled_X)
        ).dot(autoscaled_X.T.dot(autoscaled_y))
        calculated_y = (autoscaled_X.dot(self.regression_coefficients)
                        * self.ystd + self.y_mean)
        self.sigma = sum((y - calculated_y) ** 2) / len(y)

    def mlr_predict(self, X):
        """
        Predict y-values from X-values using the MLR model
        
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            X must NOT be autoscaled.
        """
        autoscaled_X = (X - self.Xmean) / self.Xstd
        return (autoscaled_X.dot(self.regression_coefficients)
                * self.ystd + self.y_mean)

    def inverse_gtm_mlr(self, target_y_value):
        """
        Predict X-values from a y-value using the MLR model
        
        Parameters
        ----------
        target_v_alue : a target y-value
            scaler

        Returns
        -------
        responsibilities_inverse can be used to discussed assigned grids on
        the GTM map.
        """
        #        target_y_values = np.ndarray.flatten(np.array(target_y_values))
        myu_i = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
            (np.prod(self.shape_of_map), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))
        sigma_i = np.diag(np.ones(len(self.regression_coefficients))) / self.beta
        inverse_sigma_i = np.diag(np.ones(len(self.regression_coefficients))) * self.beta
        delta_i = np.linalg.inv(inverse_sigma_i
                                + self.regression_coefficients.dot(self.regression_coefficients.T) / self.sigma)
        #        for target_y_value in target_y_values:
        pxy_means = np.empty(myu_i.shape)
        for i in range(pxy_means.shape[0]):
            pxy_means[i, :] = np.ndarray.flatten(
                delta_i.dot(
                    self.regression_coefficients / self.sigma * target_y_value
                    + inverse_sigma_i.dot(np.reshape(myu_i[i, :], [myu_i.shape[1], 1]))
                ))

        pyz_means = myu_i.dot(self.regression_coefficients)
        pyz_var = self.sigma + self.regression_coefficients.T.dot(
            sigma_i.dot(self.regression_coefficients))
        pyzs = np.empty(len(pyz_means))
        for i in range(len(pyz_means)):
            pyzs[i] = norm.pdf(target_y_value, pyz_means[i], pyz_var ** (1 / 2))

        responsibilities_inverse = pyzs / pyzs.sum()
        estimated_x_mean = responsibilities_inverse.dot(pxy_means)
        estimated_x_mode = pxy_means[np.argmax(responsibilities_inverse), :]

        # pyzs : vector of probability of y given zi, which can be used to
        #        discuss applicability domains
        return estimated_x_mean, estimated_x_mode, responsibilities_inverse

    def predict(self, input_variables, numbers_of_input_variables, numbers_of_output_variables):
        """
        
        Predict values of variables for forward analysis (regression) and inverse analysis
    
        Parameters
        ----------
        input_variables: numpy.array or pandas.DataFrame
            (autoscaled) m x n matrix of input variables of training data or test data,
            m is the number of sammples and
            n is the number of input variables
            When this is X-variables, it is forward analysis (regression) and
            when this is Y-variables, it is inverse analysis
        numbers_of_input_variables: list or numpy.array
            vector of numbers of input variables
            When this is numbers of X-variables, it is forward analysis (regression) and
            when this is numbers of Y-variables, it is inverse analysis
        numbers_of_output_variables: list or numpy.array
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

        input_variables = np.array(input_variables)
        if input_variables.ndim == 0:
            input_variables = np.reshape(input_variables, (1, 1))
        elif input_variables.ndim == 1:
            input_variables = np.reshape(input_variables, (1, input_variables.shape[0]))
        if self.success_flag:
            means = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
                (np.prod(self.shape_of_map), 1)
            ).dot(np.reshape(self.bias, (1, len(self.bias))))
            input_means = means[:, numbers_of_input_variables]
            output_means = means[:, numbers_of_output_variables]
            input_covariances = np.diag(np.ones(len(numbers_of_input_variables))) / self.beta
            px = np.empty([input_variables.shape[0], input_means.shape[0]])
            for sample_number in range(input_means.shape[0]):
                px[:, sample_number] = multivariate_normal.pdf(input_variables, input_means[sample_number, :],
                                                               input_covariances)

            responsibilities = px.T / px.T.sum(axis=0)
            responsibilities = responsibilities.T
            estimated_y_mean = responsibilities.dot(output_means)
            estimated_y_mode = output_means[np.argmax(responsibilities, axis=1), :]
        else:
            estimated_y_mean = np.zeros(input_variables.shape[0])
            estimated_y_mode = np.zeros(input_variables.shape[0])
            px = np.empty([input_variables.shape[0], np.prod(self.shape_of_map)])
            responsibilities = np.empty([input_variables.shape[0], np.prod(self.shape_of_map)])

        return estimated_y_mean, estimated_y_mode, responsibilities, px

    def predict_rep(self, input_variables, numbers_of_input_variables, numbers_of_output_variables):
        """
        
        Predict values of variables for forward analysis (regression) and inverse analysis. The way to calculate representative values can be set with 'rep' 
    
        Parameters
        ----------
        input_variables: numpy.array or pandas.DataFrame
            (autoscaled) m x n matrix of input variables of training data or test data,
            m is the number of sammples and
            n is the number of input variables
            When this is X-variables, it is forward analysis (regression) and
            when this is Y-variables, it is inverse analysis
        numbers_of_input_variables: list or numpy.array
            vector of numbers of input variables
            When this is numbers of X-variables, it is forward analysis (regression) and
            when this is numbers of Y-variables, it is inverse analysis
        numbers_of_output_variables: list or numpy.array
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

        estimated_y_mean, estimated_y_mode, responsibilities, px = self.predict(input_variables, numbers_of_input_variables, numbers_of_output_variables)
        if self.rep == 'mean':
            values = estimated_y_mean.copy()
        elif self.rep == 'mode':
            values = estimated_y_mode.copy()

        return values
    
    def cv_opt(self, dataset, numbers_of_input_variables, numbers_of_output_variables, candidates_of_shape_of_map,
               candidates_of_shape_of_rbf_centers,
               candidates_of_variance_of_rbfs, candidates_of_lambda_in_em_algorithm, fold_number,
               number_of_iterations):
        """
        
        Optimize hyperparameter values of GTMR model using cross-validation
    
        """

        self.display_flag = False
        self.number_of_iterations = number_of_iterations
        dataset = np.array(dataset)
        numbers_of_output_variables = np.array(numbers_of_output_variables)
#        numbers_of_input_variables = np.arange(dataset.shape[1])
#        numbers_of_input_variables = np.delete(numbers_of_input_variables, numbers_of_output_variables)
        reps = ['mean', 'mode']
        
        min_number = math.floor(dataset.shape[0] / fold_number)
        mod_number = dataset.shape[0] - min_number * fold_number
        index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
        if mod_number != 0:
            index = np.r_[index, np.arange(1, mod_number + 1, 1)]
        #            np.random.seed(999)
        fold_index_in_cv = np.random.permutation(index)
        np.random.seed()

        # grid search
        y = np.ravel(dataset[:, numbers_of_output_variables])
        parameters_and_r2_cv = []
        all_calculation_numbers = len(candidates_of_shape_of_map) * len(candidates_of_shape_of_rbf_centers) * len(
            candidates_of_variance_of_rbfs) * len(candidates_of_lambda_in_em_algorithm)
        calculation_number = 0
        for shape_of_map_grid in candidates_of_shape_of_map:
            for shape_of_rbf_centers_grid in candidates_of_shape_of_rbf_centers:
                for variance_of_rbfs_grid in candidates_of_variance_of_rbfs:
                    for lambda_in_em_algorithm_grid in candidates_of_lambda_in_em_algorithm:
                        for rep in reps:
                            calculation_number += 1
                            estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
                            success_flag_cv = True
                            for fold_number_in_cv in np.arange(1, fold_number + 1, 1):
                                dataset_train_in_cv = dataset[fold_index_in_cv != fold_number_in_cv, :]
                                dataset_test_in_cv = dataset[fold_index_in_cv == fold_number_in_cv, :]
                                self.shape_of_map = [shape_of_map_grid, shape_of_map_grid]
                                self.shape_of_rbf_centers = [shape_of_rbf_centers_grid, shape_of_rbf_centers_grid]
                                self.variance_of_rbfs = variance_of_rbfs_grid
                                self.lambda_in_em_algorithm = lambda_in_em_algorithm_grid
                                self.rep = rep
                                self.fit(dataset_train_in_cv)
                                if self.success_flag:
                                    values = self.predict_rep(dataset_test_in_cv[:, numbers_of_input_variables],
                                                              numbers_of_input_variables, numbers_of_output_variables)
                                    estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = values
                                else:
                                    success_flag_cv = False
                                    break
    
                            if success_flag_cv:
                                y_pred = np.ravel(estimated_y_in_cv)
                                r2_cv = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
                            else:
                                r2_cv = -10 ** 10
                            parameters_and_r2_cv.append(
                                [shape_of_map_grid, shape_of_rbf_centers_grid, variance_of_rbfs_grid,
                                 lambda_in_em_algorithm_grid, rep, r2_cv])
                            print([calculation_number, all_calculation_numbers, r2_cv])

        # optimized GTMR
        parameters_and_r2_cv = np.array(parameters_and_r2_cv)
        optimized_hyperparameter_number = \
            np.where(parameters_and_r2_cv[:, 5] == np.max(parameters_and_r2_cv[:, 5]))[0][0]
        self.shape_of_map = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 0]),
                             int(parameters_and_r2_cv[optimized_hyperparameter_number, 0])]
        self.shape_of_rbf_centers = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 1]),
                                     int(parameters_and_r2_cv[optimized_hyperparameter_number, 1])]
        self.variance_of_rbfs = parameters_and_r2_cv[optimized_hyperparameter_number, 2]
        self.lambda_in_em_algorithm = parameters_and_r2_cv[optimized_hyperparameter_number, 3]
        self.rep = parameters_and_r2_cv[optimized_hyperparameter_number, 4]
