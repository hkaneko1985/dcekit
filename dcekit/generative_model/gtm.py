# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# GTM (generative topographic mapping) class

import math

import numpy as np
import numpy.matlib
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score
from ..validation import k3nerror


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

    def fit(self, input_dataset, y=None):
        """
        Train the GTM map
                
        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.
        y : None
            Ignored.
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
    
    def fit_transform(self, x, y=None, mean_flag=True):
        """
        Fit GTM model and transform X

        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            input_dataset must be autoscaled.
        y : None
            Ignored.
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
    
    def transform(self, x, y=None, mean_flag=True):
        """
        Transform X using constructed GTM model

        Parameters
        ----------
        x : numpy.array or pandas.DataFrame
            input_dataset must be autoscaled.
        y : None
            Ignored.
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
            candidates_of_variance_of_rbfs) * len(candidates_of_lambda_in_em_algorithm) * len(reps)
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
        r2_cv = np.array(list(map(float, parameters_and_r2_cv[:, 5])))
        r2_cv[np.isnan(r2_cv)] = -10 ** 10
        optimized_hyperparameter_number = np.where(r2_cv == np.max(r2_cv))[0][0]
        self.shape_of_map = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 0]),
                             int(parameters_and_r2_cv[optimized_hyperparameter_number, 0])]
        self.shape_of_rbf_centers = [int(parameters_and_r2_cv[optimized_hyperparameter_number, 1]),
                                     int(parameters_and_r2_cv[optimized_hyperparameter_number, 1])]
        self.variance_of_rbfs = float(parameters_and_r2_cv[optimized_hyperparameter_number, 2])
        self.lambda_in_em_algorithm = float(parameters_and_r2_cv[optimized_hyperparameter_number, 3])
        self.rep = parameters_and_r2_cv[optimized_hyperparameter_number, 4]

    def cv_bo(self, dataset, numbers_of_input_variables, numbers_of_output_variables, candidates_of_shape_of_map,
               candidates_of_shape_of_rbf_centers,
               candidates_of_variance_of_rbfs, candidates_of_lambda_in_em_algorithm, fold_number,
               number_of_iterations, bo_iteration_number=15):
        """
        
        Optimize hyperparameter values of GTMR model using cross-validation and Bayesian optimization
    
        """

        if self.display_flag:
            bo_display_flag = True
            self.display_flag = False
        self.number_of_iterations = number_of_iterations
        
        # GTMRのパラメータ設定
        gtmr_kfold = KFold(n_splits=fold_number, shuffle=True, random_state=2) # CVの設定
#        candidates_of_shape_of_map = np.arange(30, 32, dtype=int)
#        candidates_of_shape_of_rbf_centers = np.arange(2, 22, 2, dtype=int)
#        candidates_of_variance_of_rbfs = 2 ** np.arange(-5, 4, 2, dtype=float)
#        candidates_of_lambda_in_em_algorithm = 2 ** np.arange(-4, 0, dtype=float)
#        candidates_of_lambda_in_em_algorithm = np.append(0, gtmr_candidates_of_lambda_in_em_algorithm)
        reps = ['mean', 'mode']
        # 実験計画法の条件
        doe_number_of_selecting_samples = 15  # 選択するサンプル数
        doe_number_of_random_searches = 100  # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数
        # BOの設定
        bo_iterations = np.arange(0, bo_iteration_number + 1)
        bo_gp_fold_number = 5 # BOのGPを構築するためのcvfold数
        bo_number_of_selecting_samples = 1  # 選択するサンプル数
        #bo_regression_method = 'gpr_kernels'  # gpr_one_kernel', 'gpr_kernels'
        bo_regression_method = 'gpr_one_kernel'  # gpr_one_kernel', 'gpr_kernels'
        bo_kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        acquisition_function = 'PTR'  # 'PTR', 'PI', 'EI', 'MI'
        target_range = [1, 100]  # PTR
        relaxation = 0.01  # EI, PI
        delta = 10 ** -6  # MI

        dataset = np.array(dataset)
#        autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)
        
        # GTMRパラメータの全組み合わせ生成
        params_temp_list = []
        for gtmr_map in candidates_of_shape_of_map:
            for gtmr_rbf_c in candidates_of_shape_of_rbf_centers:
                for gtmr_var in candidates_of_variance_of_rbfs:
                    for gtmr_lambda in candidates_of_lambda_in_em_algorithm:
                        for r in reps:
                            params_temp_list.append([gtmr_map, gtmr_rbf_c, gtmr_var, gtmr_lambda, r])
        params_df = pd.DataFrame(params_temp_list, columns=['shape of map', 'shape of rbf centers', 'variance of rbfs', 'lambda', 'rep']) # 全組み合わせのDataframe
        params_df = pd.get_dummies(params_df, columns=['rep']) # ダミー変数化
        
        # ベイズ最適化の繰り返し
        for bo_iter in bo_iterations:
            if bo_display_flag:
                print(f'Bayesian optimization iteration : {bo_iter} / {bo_iteration_number}')
        #    print('='*10)
            if bo_iter == 0: # 最初の試行ではD最適基準を計算
                # D最適基準の計算
                param_std = params_df.std(axis=0, ddof=1)
                param_std[np.where(param_std == 0)[0]] = 1
                autoscaled_params_df = (params_df - params_df.mean(axis=0)) / param_std # 計算のために標準化
        
                all_indexes = list(range(autoscaled_params_df.shape[0])) # indexを取得
        
                np.random.seed(110) # 乱数を生成するためのシードを固定
                for random_search_number in range(doe_number_of_random_searches):
                    # 1. ランダムに候補を選択
                    new_selected_indexes = np.random.choice(all_indexes, doe_number_of_selecting_samples, replace=False)
                    new_selected_samples = autoscaled_params_df.iloc[new_selected_indexes, :]
                    # 2. D 最適基準を計算
                    xt_x = np.dot(new_selected_samples.T, new_selected_samples)
                    d_optimal_value = np.linalg.det(xt_x) 
                    # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
                    if random_search_number == 0:
                        best_d_optimal_value = d_optimal_value.copy()
                        selected_sample_indexes = new_selected_indexes.copy()
                    else:
                        if best_d_optimal_value < d_optimal_value:
                            best_d_optimal_value = d_optimal_value.copy()
                            selected_sample_indexes = new_selected_indexes.copy()
                selected_sample_indexes = list(selected_sample_indexes) # リスト型に変換
        
                # 選択されたサンプル、選択されなかったサンプル
                selected_params_df = params_df.iloc[selected_sample_indexes, :]  # 選択されたサンプル
                bo_params_df = selected_params_df.copy() # BOのGPモデル構築用データを作成
                remaining_indexes = np.delete(all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
                remaining_params_df = params_df.iloc[remaining_indexes, :]  # 選択されなかったサンプル
        
                # 選択された全候補でGTMRの計算
                params_with_score_df = params_df.copy() # cvのscoreが含まれるdataframe
                params_with_score_df['r2cv score'] = np.nan # 初期値はnanを設定
        
            else: # 2回目以降では前回の結果をもとにする
                selected_sample_indexes = next_samples_df.index # 提案サンプルのindex
                selected_params_df = params_df.loc[selected_sample_indexes, :] # 次に計算するサンプル
                bo_params_df = pd.concat([bo_params_df, selected_params_df], axis=0) # BOのGPモデル構築用データは前回のデータと提案サンプルをマージする
                remaining_params_df = params_df.loc[params_with_score_df['r2cv score'].isna(), :] # 選択されなかったサンプル
                remaining_params_df = remaining_params_df.drop(index=selected_sample_indexes)
        
            # 選ばれたサンプル（パラメータの組み合わせ）を一つずつ計算する
            for i_n, selected_params_idx in enumerate(selected_sample_indexes):
                selected_params = selected_params_df.loc[selected_params_idx, :] # サンプルの選択
                # shape of mapの決定
                shape_of_map = selected_params['shape of map']
                # shape of rbf centerの決定
                shape_of_rbf_centers = selected_params['shape of rbf centers']
                # variance of rbfの決定
                variance_of_rbfs = selected_params['variance of rbfs']
                # lambdaの決定
                lambda_in_em = selected_params['lambda']
                # repの決定
                rep_series = selected_params[[x for x in selected_params.index if 'rep' in x]]
                rep_type = rep_series.loc[rep_series == 1]
                rep_type = (rep_type.index[0]).replace('rep_', '')
                # GTMRモデルの構築
                self.shape_of_map = [shape_of_map, shape_of_map]
                self.shape_of_rbf_centers = [shape_of_rbf_centers, shape_of_rbf_centers]
                self.variance_of_rbfs = variance_of_rbfs
                self.lambda_in_em_algorithm = lambda_in_em
                self.rep = rep_type
                estimated_y_in_cv = np.zeros((dataset.shape[0],  len(numbers_of_output_variables))) # yの保存先
                success_flag_cv = True
                for i, (cv_train_idx, cv_test_idx) in enumerate(gtmr_kfold.split(dataset)): # CVによる検証
                    # CVのinnerとouterを設定
                    autoscaled_train_innercv = dataset[cv_train_idx, :]
                    autoscaled_train_outercv = dataset[cv_test_idx, :]
                    try:
                        # modelにfitさせる
                        self.fit(autoscaled_train_innercv)
                        if self.success_flag:
                            predict_y_train_outercv = self.predict_rep(
                                autoscaled_train_outercv[:, numbers_of_input_variables], numbers_of_input_variables, numbers_of_output_variables)
                        else:
                            success_flag_cv = False
                            break
                    except:
                        predict_y_train_outercv = np.ones([autoscaled_train_outercv.shape[0], 1]) * (-10**10)
                    estimated_y_in_cv[cv_test_idx, :] = predict_y_train_outercv
                
                if success_flag_cv:
                    # r2を計算
                    y_train_one = np.ravel(dataset[:, numbers_of_output_variables])
                    y_pred = np.ravel(estimated_y_in_cv)
                    gtmr_r2_score = r2_score(y_train_one, y_pred)
                else:
                    gtmr_r2_score = (np.random.rand() / 10 + 1) * (-100)
                params_with_score_df.loc[selected_params_idx, 'r2cv score'] = gtmr_r2_score # データの保存
                
            if bo_display_flag:
                print('Best r2cv :', params_with_score_df['r2cv score'].max())
                print('='*10)
            
            # 最後はBOの計算をしないためbreak
            if bo_iter + 1 == bo_iteration_number:
                break
                    
            # Bayesian optimization
            bo_x_data = bo_params_df.copy() # GP学習用データはGMRの結果があるサンプル
            bo_x_prediction = remaining_params_df.copy() # predictionは選択されていない（GMRの結果がない）サンプル
            bo_y_data = params_with_score_df.loc[bo_params_df.index, 'r2cv score'] # yはGMRのr2cv
            
            # カーネル 11 種類
            bo_kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
                        ConstantKernel() * RBF() + WhiteKernel(),
                        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel(),
                        ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]
        
            next_samples = pd.DataFrame([], columns=selected_params_df.columns)  # 次のサンプルを入れる変数を準備
        
            # 次の候補を複数提案する繰り返し工程
            for bo_sample_number in range(bo_number_of_selecting_samples):
                # オートスケーリング
                x_std = bo_x_data.std(axis=0, ddof=1)
                x_std[np.where(x_std == 0)[0]] = 1
                autoscaled_bo_y_data = (bo_y_data - bo_y_data.mean()) / bo_y_data.std()
                autoscaled_bo_x_data = (bo_x_data - bo_x_data.mean()) / x_std
                autoscaled_bo_x_prediction = (bo_x_prediction - bo_x_data.mean()) / x_std
                
                # モデル構築
                if bo_regression_method == 'gpr_one_kernel':
                    bo_selected_kernel = bo_kernels[bo_kernel_number]
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_selected_kernel)
        
                elif bo_regression_method == 'gpr_kernels':
                    # クロスバリデーションによるカーネル関数の最適化
                    bo_cross_validation = KFold(n_splits=bo_gp_fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
                    bo_r2cvs = [] # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
                    for index, bo_kernel in enumerate(bo_kernels):
                        bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_kernel)
                        estimated_bo_y_in_cv = np.ndarray.flatten(cross_val_predict(bo_model, autoscaled_bo_x_data, autoscaled_bo_y_data, cv=bo_cross_validation))
                        estimated_bo_y_in_cv = estimated_bo_y_in_cv * bo_y_data.std(ddof=1) + bo_y_data.mean()
                        bo_r2cvs.append(r2_score(bo_y_data, estimated_bo_y_in_cv))
                    optimal_bo_kernel_number = np.where(bo_r2cvs == np.max(bo_r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
                    optimal_bo_kernel = bo_kernels[optimal_bo_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
                    
                    # モデル構築
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=optimal_bo_kernel) # GPR モデルの宣言
                
                bo_model.fit(autoscaled_bo_x_data, autoscaled_bo_y_data)  # モデルの学習
                
                # 予測
                estimated_bo_y_prediction, estimated_bo_y_prediction_std = bo_model.predict(autoscaled_bo_x_prediction, return_std=True)
                estimated_bo_y_prediction = estimated_bo_y_prediction * bo_y_data.std() + bo_y_data.mean()
                estimated_bo_y_prediction_std = estimated_bo_y_prediction_std * bo_y_data.std()
                
                cumulative_variance = np.zeros(bo_x_prediction.shape[0])
                # 獲得関数の計算
                if acquisition_function == 'MI':
                    acquisition_function_prediction = estimated_bo_y_prediction + np.log(2 / delta) ** 0.5 * (
                            (estimated_bo_y_prediction_std ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
                    cumulative_variance = cumulative_variance + estimated_bo_y_prediction_std ** 2
                elif acquisition_function == 'EI':
                    acquisition_function_prediction = (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) * \
                                                    norm.cdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
                                                                estimated_bo_y_prediction_std) + \
                                                    estimated_bo_y_prediction_std * \
                                                    norm.pdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
                                                                estimated_bo_y_prediction_std)
                elif acquisition_function == 'PI':
                    acquisition_function_prediction = norm.cdf(
                            (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) / estimated_bo_y_prediction_std)
                elif acquisition_function == 'PTR':
                    acquisition_function_prediction = norm.cdf(target_range[1],
                                                            loc=estimated_bo_y_prediction,
                                                            scale=estimated_bo_y_prediction_std
                                                            ) - norm.cdf(target_range[0],
                                                                            loc=estimated_bo_y_prediction,
                                                                            scale=estimated_bo_y_prediction_std)
                acquisition_function_prediction[estimated_bo_y_prediction_std <= 0] = 0
                
                # 保存
                estimated_bo_y_prediction = pd.DataFrame(estimated_bo_y_prediction, bo_x_prediction.index, columns=['estimated_y'])
                estimated_bo_y_prediction_std = pd.DataFrame(estimated_bo_y_prediction_std, bo_x_prediction.index, columns=['std_of_estimated_y'])
                acquisition_function_prediction = pd.DataFrame(acquisition_function_prediction, index=bo_x_prediction.index, columns=['acquisition_function'])
        #        
                # 次のサンプル
                next_samples = pd.concat([next_samples, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                
                # x, y, x_prediction, cumulative_variance の更新
                bo_x_data = pd.concat([bo_x_data, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                bo_y_data = pd.concat([bo_y_data, estimated_bo_y_prediction.loc[acquisition_function_prediction.idxmax()].iloc[0]], axis=0)
                bo_x_prediction = bo_x_prediction.drop(acquisition_function_prediction.idxmax(), axis=0)
                cumulative_variance = np.delete(cumulative_variance, np.where(acquisition_function_prediction.index == acquisition_function_prediction.iloc[:, 0].idxmax())[0][0])
            next_samples_df = next_samples.copy()
        
        # 結果の保存
        #params_with_score_df.sort_values('r2cv score', ascending=False).to_csv('params_with_score.csv')
        params_with_score_df_best = params_with_score_df.sort_values('r2cv score', ascending=False).iloc[0, :] # r2が高い順にソート
        shape_of_map = params_with_score_df_best['shape of map']
        # shape of rbf centerの決定
        shape_of_rbf_centers = params_with_score_df_best['shape of rbf centers']
        # variance of rbfの決定
        variance_of_rbfs = params_with_score_df_best['variance of rbfs']
        # lambdaの決定
        lambda_in_em = params_with_score_df_best['lambda']
        # repの決定
        rep_series = params_with_score_df_best[[x for x in params_with_score_df_best.index if 'rep' in x]]
        rep_type = rep_series.loc[rep_series == 1]
        self.shape_of_map =[int(shape_of_map), int(shape_of_map)]
        self.shape_of_rbf_centers = [shape_of_rbf_centers, shape_of_rbf_centers]
        self.variance_of_rbfs = variance_of_rbfs
        self.lambda_in_em_algorithm = lambda_in_em
        self.rep = (rep_type.index[0]).replace('rep_', '')
        self.r2cv = params_with_score_df_best['r2cv score']
        if bo_display_flag:
            self.display_flag = True
    
    def k3nerror_bo(self, dataset, candidates_of_shape_of_map, candidates_of_shape_of_rbf_centers,
               candidates_of_variance_of_rbfs, candidates_of_lambda_in_em_algorithm, number_of_iterations,
               k_in_k3nerror=10, bo_iteration_number=15):
        """
        
        Optimize hyperparameter values of GTM model using k3n-error and Bayesian optimization
    
        """

        if self.display_flag:
            bo_display_flag = True
            self.display_flag = False
        self.number_of_iterations = number_of_iterations
        
        # GTMのパラメータ設定
#        candidates_of_shape_of_map = np.arange(30, 32, dtype=int)
#        candidates_of_shape_of_rbf_centers = np.arange(2, 22, 2, dtype=int)
#        candidates_of_variance_of_rbfs = 2 ** np.arange(-5, 4, 2, dtype=float)
#        candidates_of_lambda_in_em_algorithm = 2 ** np.arange(-4, 0, dtype=float)
#        candidates_of_lambda_in_em_algorithm = np.append(0, gtmr_candidates_of_lambda_in_em_algorithm)
        rep = 'mean'
#        reps = ['mean', 'mode']
        # 実験計画法の条件
        doe_number_of_selecting_samples = 15  # 選択するサンプル数
        doe_number_of_random_searches = 100  # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数
        # BOの設定
        bo_iterations = np.arange(0, bo_iteration_number + 1)
        bo_gp_fold_number = 5 # BOのGPを構築するためのcvfold数
        bo_number_of_selecting_samples = 1  # 選択するサンプル数
        #bo_regression_method = 'gpr_kernels'  # gpr_one_kernel', 'gpr_kernels'
        bo_regression_method = 'gpr_one_kernel'  # gpr_one_kernel', 'gpr_kernels'
        bo_kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        acquisition_function = 'PTR'  # 'PTR' のみ ['PI', 'EI', 'MI'は、なし]
        target_range = [-100, 0]  # PTR
#        relaxation = 0.01  # EI, PI
#        delta = 10 ** -6  # MI

        dataset = np.array(dataset)
#        autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)
        
        # GTMパラメータの全組み合わせ生成
        params_temp_list = []
        for gtmr_map in candidates_of_shape_of_map:
            for gtmr_rbf_c in candidates_of_shape_of_rbf_centers:
                for gtmr_var in candidates_of_variance_of_rbfs:
                    for gtmr_lambda in candidates_of_lambda_in_em_algorithm:
                        params_temp_list.append([gtmr_map, gtmr_rbf_c, gtmr_var, gtmr_lambda])
        params_df = pd.DataFrame(params_temp_list, columns=['shape of map', 'shape of rbf centers', 'variance of rbfs', 'lambda']) # 全組み合わせのDataframe
        
        # ベイズ最適化の繰り返し
        for bo_iter in bo_iterations:
            if bo_display_flag:
                print(f'Bayesian optimization iteration : {bo_iter} / {bo_iteration_number}')
        #    print('='*10)
            if bo_iter == 0: # 最初の試行ではD最適基準を計算
                # D最適基準の計算
                param_std = params_df.std(axis=0, ddof=1)
                param_std[np.where(param_std == 0)[0]] = 1
                autoscaled_params_df = (params_df - params_df.mean(axis=0)) / param_std # 計算のために標準化
        
                all_indexes = list(range(autoscaled_params_df.shape[0])) # indexを取得
        
                np.random.seed(110) # 乱数を生成するためのシードを固定
                for random_search_number in range(doe_number_of_random_searches):
                    # 1. ランダムに候補を選択
                    new_selected_indexes = np.random.choice(all_indexes, doe_number_of_selecting_samples, replace=False)
                    new_selected_samples = autoscaled_params_df.iloc[new_selected_indexes, :]
                    # 2. D 最適基準を計算
                    xt_x = np.dot(new_selected_samples.T, new_selected_samples)
                    d_optimal_value = np.linalg.det(xt_x) 
                    # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
                    if random_search_number == 0:
                        best_d_optimal_value = d_optimal_value.copy()
                        selected_sample_indexes = new_selected_indexes.copy()
                    else:
                        if best_d_optimal_value < d_optimal_value:
                            best_d_optimal_value = d_optimal_value.copy()
                            selected_sample_indexes = new_selected_indexes.copy()
                selected_sample_indexes = list(selected_sample_indexes) # リスト型に変換
        
                # 選択されたサンプル、選択されなかったサンプル
                selected_params_df = params_df.iloc[selected_sample_indexes, :]  # 選択されたサンプル
                bo_params_df = selected_params_df.copy() # BOのGPモデル構築用データを作成
                remaining_indexes = np.delete(all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
                remaining_params_df = params_df.iloc[remaining_indexes, :]  # 選択されなかったサンプル
        
                # 選択された全候補でGTMRの計算
                params_with_score_df = params_df.copy() # cvのscoreが含まれるdataframe
                params_with_score_df['k3nerror score'] = np.nan # 初期値はnanを設定
        
            else: # 2回目以降では前回の結果をもとにする
                selected_sample_indexes = next_samples_df.index # 提案サンプルのindex
                selected_params_df = params_df.loc[selected_sample_indexes, :] # 次に計算するサンプル
                bo_params_df = pd.concat([bo_params_df, selected_params_df], axis=0) # BOのGPモデル構築用データは前回のデータと提案サンプルをマージする
                remaining_params_df = params_df.loc[params_with_score_df['k3nerror score'].isna(), :] # 選択されなかったサンプル
                remaining_params_df = remaining_params_df.drop(index=selected_sample_indexes)
        
            # 選ばれたサンプル（パラメータの組み合わせ）を一つずつ計算する
            for i_n, selected_params_idx in enumerate(selected_sample_indexes):
                selected_params = selected_params_df.loc[selected_params_idx, :] # サンプルの選択
                # shape of mapの決定
                shape_of_map = selected_params['shape of map']
                # shape of rbf centerの決定
                shape_of_rbf_centers = selected_params['shape of rbf centers']
                # variance of rbfの決定
                variance_of_rbfs = selected_params['variance of rbfs']
                # lambdaの決定
                lambda_in_em = selected_params['lambda']
                # GTMモデルの構築
                self.shape_of_map = [shape_of_map, shape_of_map]
                self.shape_of_rbf_centers = [shape_of_rbf_centers, shape_of_rbf_centers]
                self.variance_of_rbfs = variance_of_rbfs
                self.lambda_in_em_algorithm = lambda_in_em
                self.fit(dataset)
                if self.success_flag:
                    means, modes = self.means_modes(dataset)
                    # calculate k3n-error
                    if rep == 'mean':
                        k3nerror_of_gtm = k3nerror(dataset, means, k_in_k3nerror) + k3nerror(means, dataset, k_in_k3nerror)
                    elif rep == 'mode':
                        k3nerror_of_gtm = k3nerror(dataset, modes, k_in_k3nerror) + k3nerror(modes, dataset, k_in_k3nerror)
                else:
#                    k3nerror_of_gtm = 10 ** 100
                    k3nerror_of_gtm = (np.random.rand()/10 + 1) * 50

                params_with_score_df['k3nerror score'].loc[selected_params_idx] = k3nerror_of_gtm # データの保存
                
            if bo_display_flag:
                print('Best k3n-error :', params_with_score_df['k3nerror score'].min())
                print('='*10)
            
            # 最後はBOの計算をしないためbreak
            if bo_iter + 1 == bo_iteration_number:
                break
                    
            # Bayesian optimization
            bo_x_data = bo_params_df.copy() # GP学習用データはGMRの結果があるサンプル
            bo_x_prediction = remaining_params_df.copy() # predictionは選択されていない（GMRの結果がない）サンプル
            bo_y_data = params_with_score_df.loc[bo_params_df.index, 'k3nerror score'] # yはGMRのr2cv
            
            # カーネル 11 種類
            bo_kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
                        ConstantKernel() * RBF() + WhiteKernel(),
                        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel(),
                        ConstantKernel() * RBF(np.ones(bo_x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]
        
            next_samples = pd.DataFrame([], columns=selected_params_df.columns)  # 次のサンプルを入れる変数を準備
        
            # 次の候補を複数提案する繰り返し工程
            for bo_sample_number in range(bo_number_of_selecting_samples):
                # オートスケーリング
                x_std = bo_x_data.std(axis=0, ddof=1)
                x_std[np.where(x_std == 0)[0]] = 1
                autoscaled_bo_y_data = (bo_y_data - bo_y_data.mean()) / bo_y_data.std()
                autoscaled_bo_x_data = (bo_x_data - bo_x_data.mean()) / x_std
                autoscaled_bo_x_prediction = (bo_x_prediction - bo_x_data.mean()) / x_std
                
                # モデル構築
                if bo_regression_method == 'gpr_one_kernel':
                    bo_selected_kernel = bo_kernels[bo_kernel_number]
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_selected_kernel)
        
                elif bo_regression_method == 'gpr_kernels':
                    # クロスバリデーションによるカーネル関数の最適化
                    bo_cross_validation = KFold(n_splits=bo_gp_fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
                    bo_r2cvs = [] # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
                    for index, bo_kernel in enumerate(bo_kernels):
                        bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_kernel)
                        estimated_bo_y_in_cv = np.ndarray.flatten(cross_val_predict(bo_model, autoscaled_bo_x_data, autoscaled_bo_y_data, cv=bo_cross_validation))
                        estimated_bo_y_in_cv = estimated_bo_y_in_cv * bo_y_data.std(ddof=1) + bo_y_data.mean()
                        bo_r2cvs.append(r2_score(bo_y_data, estimated_bo_y_in_cv))
                    optimal_bo_kernel_number = np.where(bo_r2cvs == np.max(bo_r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
                    optimal_bo_kernel = bo_kernels[optimal_bo_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
                    
                    # モデル構築
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=optimal_bo_kernel) # GPR モデルの宣言
                
                bo_model.fit(autoscaled_bo_x_data, autoscaled_bo_y_data)  # モデルの学習
                
                # 予測
                estimated_bo_y_prediction, estimated_bo_y_prediction_std = bo_model.predict(autoscaled_bo_x_prediction, return_std=True)
                estimated_bo_y_prediction = estimated_bo_y_prediction * bo_y_data.std() + bo_y_data.mean()
                estimated_bo_y_prediction_std = estimated_bo_y_prediction_std * bo_y_data.std()
                
#                cumulative_variance = np.zeros(bo_x_prediction.shape[0])
                # 獲得関数の計算
#                if acquisition_function == 'MI':
#                    acquisition_function_prediction = estimated_bo_y_prediction + np.log(2 / delta) ** 0.5 * (
#                            (estimated_bo_y_prediction_std ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
#                    cumulative_variance = cumulative_variance + estimated_bo_y_prediction_std ** 2
#                elif acquisition_function == 'EI':
#                    acquisition_function_prediction = (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) * \
#                                                    norm.cdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
#                                                                estimated_bo_y_prediction_std) + \
#                                                    estimated_bo_y_prediction_std * \
#                                                    norm.pdf((estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) /
#                                                                estimated_bo_y_prediction_std)
#                elif acquisition_function == 'PI':
#                    acquisition_function_prediction = norm.cdf(
#                            (estimated_bo_y_prediction - max(bo_y_data) - relaxation * bo_y_data.std()) / estimated_bo_y_prediction_std)
                if acquisition_function == 'PTR':
                    acquisition_function_prediction = norm.cdf(target_range[1],
                                                            loc=estimated_bo_y_prediction,
                                                            scale=estimated_bo_y_prediction_std
                                                            ) - norm.cdf(target_range[0],
                                                                            loc=estimated_bo_y_prediction,
                                                                            scale=estimated_bo_y_prediction_std)
                acquisition_function_prediction[estimated_bo_y_prediction_std <= 0] = 0
                
                # 保存
                estimated_bo_y_prediction = pd.DataFrame(estimated_bo_y_prediction, bo_x_prediction.index, columns=['estimated_y'])
                estimated_bo_y_prediction_std = pd.DataFrame(estimated_bo_y_prediction_std, bo_x_prediction.index, columns=['std_of_estimated_y'])
                acquisition_function_prediction = pd.DataFrame(acquisition_function_prediction, index=bo_x_prediction.index, columns=['acquisition_function'])
        #        
                # 次のサンプル
                next_samples = pd.concat([next_samples, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                
                # x, y, x_prediction, cumulative_variance の更新
                bo_x_data = pd.concat([bo_x_data, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                bo_y_data = pd.concat([bo_y_data, estimated_bo_y_prediction.loc[acquisition_function_prediction.idxmax()].iloc[0]], axis=0)
                bo_x_prediction = bo_x_prediction.drop(acquisition_function_prediction.idxmax(), axis=0)
#                cumulative_variance = np.delete(cumulative_variance, np.where(acquisition_function_prediction.index == acquisition_function_prediction.iloc[:, 0].idxmax())[0][0])
            next_samples_df = next_samples.copy()
        
        # 結果の保存
        #params_with_score_df.sort_values('r2cv score', ascending=True).to_csv('params_with_score.csv')
        params_with_score_df_best = params_with_score_df.sort_values('k3nerror score', ascending=True).iloc[0, :] # k3n error が低い順にソート
        shape_of_map = params_with_score_df_best['shape of map']
        # shape of rbf centerの決定
        shape_of_rbf_centers = params_with_score_df_best['shape of rbf centers']
        # variance of rbfの決定
        variance_of_rbfs = params_with_score_df_best['variance of rbfs']
        # lambdaの決定
        lambda_in_em = params_with_score_df_best['lambda']
        self.shape_of_map =[int(shape_of_map), int(shape_of_map)]
        self.shape_of_rbf_centers = [shape_of_rbf_centers, shape_of_rbf_centers]
        self.variance_of_rbfs = variance_of_rbfs
        self.lambda_in_em_algorithm = lambda_in_em
        self.k3nerror = params_with_score_df_best['k3nerror score']
        if bo_display_flag:
            self.display_flag = True
            
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """
        Parameters
        ----------
        deep : Ignored. (for compatibility with sklearn)

        Returns
        ----------
        self : returns an dictionary of parameters.
        """
        
        params = {'shape_of_map' : self.shape_of_map,
                  'shape_of_rbf_centers' : self.shape_of_rbf_centers,
                  'variance_of_rbfs' : self.variance_of_rbfs,
                  'lambda_in_em_algorithm' : self.lambda_in_em_algorithm,
                   'number_of_iterations' : self.number_of_iterations}
        return params
