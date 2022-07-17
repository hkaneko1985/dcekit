# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Class of Variational Bayesian Gaussian Mixture Regression (VBGMR), which is supervised Variational Bayesian Gaussian Mixture Model (VBGMM)

import math

import numpy as np
import numpy.matlib
import pandas as pd
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score
import warnings
warnings.simplefilter('ignore')

class VBGMR(BayesianGaussianMixture):

    def __init__(self, n_components=30, covariance_type='full',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=0.01, rep='mean', max_iter=100,
                 random_state=None, display_flag=False):
        super(VBGMR, self).__init__(n_components=n_components, covariance_type=covariance_type,
                                    weight_concentration_prior_type=weight_concentration_prior_type,
                                    weight_concentration_prior=weight_concentration_prior,
                                    max_iter=max_iter, random_state=random_state)

        self.rep = rep
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
        
        try:
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
        except:
            mode_of_estimated_mean = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
            weighted_estimated_mean = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
            estimated_mean_for_all_components = np.zeros([self.n_components, dataset.shape[0], len(numbers_of_output_variables)])
            weights = np.zeros([dataset.shape[0], self.n_components])

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
        
        mode_of_estimated_mean, weighted_estimated_mean, _, _ = self.predict(dataset, numbers_of_input_variables, numbers_of_output_variables)
        if self.rep == 'mean':
            values = mode_of_estimated_mean.copy()
        elif self.rep == 'mode':
            values = weighted_estimated_mean.copy()
        return values
    
    def predict_logpdf(self, dataset, numbers_of_input_variables, numbers_of_output_variables, estimated_results):
        """
        Variational Bayesian Gaussian Mixture Regression (VBGMR) based on Variational Bayesian Gaussian Mixture Model (VBGMM)
        
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
        Variational Bayesian Gaussian Mixture Regression (VBGMR) based on Variational Bayesian Gaussian Mixture Model (VBGMM)
        
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
        Variational Bayesian Gaussian Mixture Regression (VBGMR) based on Variational Bayesian Gaussian Mixture Model (VBGMM)
        
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
#        autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)
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
                                
                                for fold_number_in_cv in np.arange(1, fold_number + 1, 1):
                                    dataset_train_in_cv = dataset[fold_index_in_cv != fold_number_in_cv, :]
                                    dataset_test_in_cv = dataset[fold_index_in_cv == fold_number_in_cv, :]
                                    
                                    try:
                                        self.fit(dataset_train_in_cv)
                                        values = self.predict_rep(dataset_test_in_cv[:, numbers_of_input_variables],
                                                                  numbers_of_input_variables, numbers_of_output_variables)
                                    except:
                                        values = np.ones([dataset_test_in_cv.shape[0], len(numbers_of_output_variables)]) * (- 10 ** 10)
                
                                    estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = values  # 格納
                
                                y = np.ravel(dataset[:, numbers_of_output_variables])
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
    
    def cv_bo(self, dataset, numbers_of_input_variables, numbers_of_output_variables, covariance_types,
              numbers_of_components, weight_concentration_prior_types, weight_concentration_priors,
              fold_number, bo_iteration_number=15):
        """
        Hyperparameter optimization for Variational Bayesian Gaussian Mixture Regression (VBGMR) using cross-validation and Bayesian optimization
    
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
        bo_iteration_number: int
            number of iteration in Bayesian optimization
    
        Returns
        -------
        best_covariance_type : str
            best covariance type
        best_number_of_components : int
            best number of components
        """

        dataset = np.array(dataset)
#        autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)
        
        # VBGMRのパラメータ設定
#        numbers_of_components = np.arange(1, 51, 1)
#        covariance_types = ['full', 'diag', 'tied', 'spherical']
#        weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']
#        weight_concentration_priors = 10 ** np.arange(-4, 2, 2, dtype=float)
        reps = ['mean', 'mode']
        vbgmr_kfold = KFold(n_splits=fold_number, shuffle=True, random_state=3) # CVの設定
        # 実験計画法の条件
        doe_number_of_selecting_samples = 15  # 選択するサンプル数
        doe_number_of_random_searches = 100  # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数
        # BOの設定
        bo_iteration_number = 20 # BOの繰り返し回数
        bo_iterations = np.arange(0, bo_iteration_number + 1)
        bo_gp_fold_number = 5 # BOのGPを構築するためのcvfold数
        bo_number_of_selecting_samples = 5  # 選択するサンプル数
        bo_regression_method = 'gpr_one_kernel'  # gpr_one_kernel', 'gpr_kernels'
        bo_kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        acquisition_function = 'PI'  # 'PTR', 'PI', 'EI', 'MI'
        target_range = [1, 2]  # PTR
        relaxation = 0.01  # EI, PI
        delta = 10 ** -6  # MI
        
        # GMRパラメータの全組み合わせ生成
        params_temp_list = []
        for num_comp in numbers_of_components:
            for cov_type in covariance_types:
                for weight_concentration_prior_type in weight_concentration_prior_types:
                    for weight_concentration_prior in weight_concentration_priors:
                        for r in reps:
                            params_temp_list.append([num_comp, cov_type, weight_concentration_prior_type, weight_concentration_prior, r])
        params_df = pd.DataFrame(params_temp_list, columns=['number of components', 'covariance type', 'weight concentration prior type', 'weight concentration prior', 'rep']) # 全組み合わせのDataframe
        params_df = pd.get_dummies(params_df, columns=['covariance type', 'weight concentration prior type', 'rep']) # ダミー変数化
        
        # ベイズ最適化の繰り返し
        for bo_iter in bo_iterations:
            if self.display_flag:
                print(f'Bayesian optimization iteration : {bo_iter + 1} / {bo_iteration_number}')
            if bo_iter == 0: # 最初の試行ではD最適基準を計算
                # D最適基準の計算
                autoscaled_params_df = (params_df - params_df.mean(axis=0)) / params_df.std(axis=0, ddof=1) # 計算のために標準化
        
                all_indexes = list(range(autoscaled_params_df.shape[0])) # indexを取得
        
                np.random.seed(12) # 乱数を生成するためのシードを固定
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
        
                # 選択された全候補でGMRの計算
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
                # componentの数の決定
                number_of_components = int(selected_params['number of components'])
                # covariance typeの決定
                covariance_series = selected_params[[x for x in selected_params.index if 'covariance type' in x]]
                covariance_type = covariance_series.loc[covariance_series == 1]
                covariance_type = (covariance_type.index[0]).replace('covariance type_', '')
                # weight concentration prior typesの決定
                weight_concentration_prior_type_series = selected_params[[x for x in selected_params.index if 'weight concentration prior type' in x]]
                weight_concentration_prior_type = weight_concentration_prior_type_series.loc[weight_concentration_prior_type_series == 1]
                weight_concentration_prior_type = (weight_concentration_prior_type.index[0]).replace('weight concentration prior type_', '')
                # weight concentration priorsの決定
                weight_concentration_prior = selected_params['weight concentration prior']
                # repの決定
                rep_series = selected_params[[x for x in selected_params.index if 'rep' in x]]
                rep_type = rep_series.loc[rep_series == 1]
                rep_type = (rep_type.index[0]).replace('rep_', '')
                # VBGMRモデルの構築
                self.covariance_type = covariance_type
                self.n_components = number_of_components
                self.weight_concentration_prior_type = weight_concentration_prior_type
                self.weight_concentration_prior = weight_concentration_prior
                self.rep = rep_type
                estimated_y_in_cv = np.empty((dataset.shape[0],  len(numbers_of_output_variables))) # yの保存先
                for i, (cv_train_idx, cv_test_idx) in enumerate(vbgmr_kfold.split(dataset)): # CVによる検証
                    # CVのinnerとouterを設定
                    autoscaled_train_innercv = dataset[cv_train_idx, :]
                    autoscaled_train_outercv = dataset[cv_test_idx, :]
                    try:
                        # modelにfitさせる
                        self.fit(autoscaled_train_innercv)
                        # 学習モデルにouterを入力
                        predict_y_train_outercv = self.predict_rep(autoscaled_train_outercv[:, numbers_of_input_variables], numbers_of_input_variables, numbers_of_output_variables)
        #                predict_y_train_outercv = predict_y_train_outercv * y_train.std() + y_train.mean()
                    except:
                        predict_y_train_outercv = np.ones([autoscaled_train_outercv.shape[0], 1]) * (-10**10)
                    
                    estimated_y_in_cv[cv_test_idx, :] = predict_y_train_outercv
                
                # r2を計算
                y_train_one = np.ravel(dataset[:, numbers_of_output_variables])
                y_pred = np.ravel(estimated_y_in_cv)
                vbgmr_r2_score = r2_score(y_train_one, y_pred)
                params_with_score_df.loc[selected_params_idx, 'r2cv score'] = vbgmr_r2_score # データの保存
            if self.display_flag:
                print('Best r2cv :', params_with_score_df['r2cv score'].max())
                print('='*10)
                
            # 最後はBOの計算をしないためbreak
            if bo_iter + 1 == bo_iteration_number:
                break
            
            # Bayesian optimization
            bo_x_data = bo_params_df.copy() # GP学習用データはVBGMRの結果があるサンプル
            bo_x_prediction = remaining_params_df.copy() # predictionは選択されていない（VBGMRの結果がない）サンプル
            bo_y_data = params_with_score_df.loc[bo_params_df.index, 'r2cv score'] # yはVBGMRのr2cv
            
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
                autoscaled_bo_y_data = (bo_y_data - bo_y_data.mean()) / bo_y_data.std()
                autoscaled_bo_x_data = (bo_x_data - bo_x_data.mean()) / bo_x_data.std()
                autoscaled_bo_x_prediction = (bo_x_prediction - bo_x_data.mean()) / bo_x_data.std()
                
                # モデル構築
                if bo_regression_method == 'gpr_one_kernel':
                    bo_selected_kernel = bo_kernels[bo_kernel_number]
                    bo_model = GaussianProcessRegressor(alpha=0, kernel=bo_selected_kernel)
        
                elif bo_regression_method == 'gpr_kernels':
                    # クロスバリデーションによるカーネル関数の最適化
                    bo_cross_validation = KFold(n_splits=bo_gp_fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
                    bo_r2cvs = [] # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
                    for index, bo_kernel in enumerate(bo_kernels):
        #                print('\r\tGP cv kernel selection', index + 1, '/', len(bo_kernels), end='')
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
                
                # 次のサンプル
                next_samples = pd.concat([next_samples, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                next_samples_df = next_samples.copy()
                
                # x, y, x_prediction, cumulative_variance の更新
                bo_x_data = pd.concat([bo_x_data, bo_x_prediction.loc[acquisition_function_prediction.idxmax()]], axis=0)
                bo_y_data = pd.concat([bo_y_data, estimated_bo_y_prediction.loc[acquisition_function_prediction.idxmax()].iloc[0]], axis=0)
                bo_x_prediction = bo_x_prediction.drop(acquisition_function_prediction.idxmax(), axis=0)
                cumulative_variance = np.delete(cumulative_variance, np.where(acquisition_function_prediction.index == acquisition_function_prediction.iloc[:, 0].idxmax())[0][0])
        
        # 結果の保存
        params_with_score_df_best = params_with_score_df.sort_values('r2cv score', ascending=False).iloc[0, :] # r2が高い順にソート
        # covariance typeの決定
        covariance_series = params_with_score_df_best[[x for x in params_with_score_df_best.index if 'covariance type' in x]]
        covariance_type = covariance_series.loc[covariance_series == 1]
        # weight concentration prior typesの決定
        weight_concentration_prior_type_series = params_with_score_df_best[[x for x in params_with_score_df_best.index if 'weight concentration prior type' in x]]
        weight_concentration_prior_type = weight_concentration_prior_type_series.loc[weight_concentration_prior_type_series == 1]
        # repの決定
        rep_series = params_with_score_df_best[[x for x in params_with_score_df_best.index if 'rep' in x]]
        rep_type = rep_series.loc[rep_series == 1]
        
        self.covariance_type = (covariance_type.index[0]).replace('covariance type_', '')
        self.n_components = int(params_with_score_df_best['number of components'])
        self.weight_concentration_prior_type = (weight_concentration_prior_type.index[0]).replace('weight concentration prior type_', '')
        self.weight_concentration_prior = params_with_score_df_best['weight concentration prior']
        self.rep = (rep_type.index[0]).replace('rep_', '')
        self.r2cv = params_with_score_df_best['r2cv score']
    
    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)
    
    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.
        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.
        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
