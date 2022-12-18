# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

import sys

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score
from scipy.stats import norm
from ..validation import make_midknn_dataset
import warnings
warnings.simplefilter('ignore')

def bo_dnn_hyperparams(x, y,
                       hidden_layer_sizes_candidates=[(50,), (100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)],
                       activation_candidates=['identity', 'logistic', 'tanh', 'relu'],
                       alpha_candidates=10 ** np.arange(-6, -1, dtype=float),
                       learning_rate_init_candidates=10 ** np.arange(-5, 0, dtype=float),
                       validation_method='cv', parameter=5, bo_iteration_number=15, display_flag=False):
    """
    Bayesian optimization of scikit-learn[MLPRegressor]-based deep neural network (DNN) hyperparameters
    
    Optimize DNN hyperparameters based on Bayesian optimization and cross-validation or midknn

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    hidden_layer_sizes_candidates : list
        vector of candidates of hidden_layer_sizes in MLPRegressor
    activation_candidates : list
        vector of candidates of activation in MLPRegressor
    alpha_candidates : numpy.array or list
        vector of candidates of alpha in MLPRegressor
    learning_rate_init_candidates : numpy.array or list
        vector of candidates of learning_rate_init in MLPRegressor    
    validation_method : 'cv' or 'midknn'
        if 'cv', cross-validation is used, and if 'midknn', midknn is used 
    parameter : int
        "fold_number"-fold cross-validation in cross-validation, and k in midknn

    Returns
    -------
    optimal_hidden_layer_sizes : tuple
        optimized hidden_layer_sizes in MLPRegressor
    optimal_activation : str
        optimized activation in MLPRegressor
    optimal_alpha : float
        optimized alpha in MLPRegressor
    optimal_learning_rate_init : float
        optimized learning_rate_init in MLPRegressor
    """
    
    if validation_method != 'cv' and validation_method != 'midknn':
#        print('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))
#        return 0, 0, 0
        sys.exit('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))
        
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
 
    x = np.array(x)
    y = np.array(y)
    alpha_candidates = np.array(alpha_candidates)
    learning_rate_init_candidates = np.array(learning_rate_init_candidates)

    hidden_layer_sizes_indexes = [i for i in range(len(hidden_layer_sizes_candidates))]
    activation_candidates_indexes = [i for i in range(len(activation_candidates))]
    all_candidate_combinations = []
    for hidden_layer_sizes_index in hidden_layer_sizes_indexes:
        for activation_candidates_indexe in activation_candidates_indexes:
            for alpha_candidate in alpha_candidates:
                for learning_rate_init_candidate in learning_rate_init_candidates:
                    all_candidate_combinations.append([hidden_layer_sizes_index, activation_candidates_indexe, alpha_candidate, learning_rate_init_candidate])
    all_candidate_combinations = np.array(all_candidate_combinations)
    all_candidate_combinations_df = pd.DataFrame(all_candidate_combinations, columns=['hidden_layer_sizes', 'activation', 'alpha', 'learning_rate_init'])
    
    if all_candidate_combinations_df.shape[0] <= doe_number_of_selecting_samples + bo_iteration_number:
        optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init = gs_dnn_hyperparams(x, y,
                       hidden_layer_sizes_candidates=hidden_layer_sizes_candidates,
                       activation_candidates=activation_candidates,
                       alpha_candidates=alpha_candidates,
                       learning_rate_init_candidates=learning_rate_init_candidates,
                       validation_method=validation_method, parameter=parameter, display_flag=display_flag)
        return optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init
    
    numerical_variable_numbers = np.array([2, 3])
    category_variable_numbers = np.array([0, 1])
    #category_variable_names = all_candidate_combinations_df.columns[category_variable_numbers]
    numerical_x = all_candidate_combinations_df.iloc[:, numerical_variable_numbers]
    if len(np.where(alpha_candidates <= 0)[0]) == 0:
        numerical_x.iloc[:, 0] = np.log10(numerical_x.iloc[:, 0])
    if len(np.where(learning_rate_init_candidates <= 0)[0]) == 0:
        numerical_x.iloc[:, 1] = np.log10(numerical_x.iloc[:, 1])
    category_x = all_candidate_combinations_df.iloc[:, category_variable_numbers]
    dummy_x = pd.get_dummies(category_x.astype(str))
#    all_x = pd.concat([numerical_x, dummy_x], axis=1)
    params_df = pd.concat([numerical_x, dummy_x], axis=1)
    
    
    cross_validation = KFold(n_splits=parameter, random_state=9, shuffle=True)
    if validation_method == 'midknn':
        # make midknn data points
        x_midknn, y_midknn = make_midknn_dataset(x, y, parameter)
    # ベイズ最適化の繰り返し
    for bo_iter in bo_iterations:
        if display_flag:
            print(f'Bayesian optimization iteration : {bo_iter + 1} / {bo_iteration_number}')
    #    print('='*10)
        if bo_iter == 0: # 最初の試行ではD最適基準を計算
            # D最適基準の計算
            autoscaled_params_df = (params_df - params_df.mean(axis=0)) / params_df.std(axis=0, ddof=1) # 計算のために標準化
    
            all_indexes = list(range(autoscaled_params_df.shape[0])) # indexを取得
    
            np.random.seed(11) # 乱数を生成するためのシードを固定
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
            true_selected_params_df = all_candidate_combinations_df.iloc[selected_sample_indexes, :]
            bo_params_df = selected_params_df.copy() # BOのGPモデル構築用データを作成
            remaining_indexes = np.delete(all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
            remaining_params_df = params_df.iloc[remaining_indexes, :]  # 選択されなかったサンプル
            true_remaining_params_df = all_candidate_combinations_df.iloc[remaining_indexes, :]
    
            # 選択された全候補でGMRの計算
            params_with_score_df = params_df.copy() # cvのscoreが含まれるdataframe
            params_with_score_df['score'] = np.nan # 初期値はnanを設定
    
        else: # 2回目以降では前回の結果をもとにする
            selected_sample_indexes = next_samples_df.index # 提案サンプルのindex
            selected_params_df = params_df.loc[selected_sample_indexes, :] # 次に計算するサンプル
            true_selected_params_df = all_candidate_combinations_df.loc[selected_sample_indexes, :] # 次に計算するサンプル
            bo_params_df = pd.concat([bo_params_df, selected_params_df], axis=0) # BOのGPモデル構築用データは前回のデータと提案サンプルをマージする
            remaining_params_df = params_df.loc[params_with_score_df['score'].isna(), :] # 選択されなかったサンプル
            remaining_params_df = remaining_params_df.drop(index=selected_sample_indexes)
            true_remaining_params_df = all_candidate_combinations_df.loc[params_with_score_df['score'].isna(), :] # 選択されなかったサンプル
            true_remaining_params_df = true_remaining_params_df.drop(index=selected_sample_indexes)
    
        # 選ばれたサンプル（パラメータの組み合わせ）を一つずつ計算する
        for i_n, selected_params_idx in enumerate(selected_sample_indexes):
            selected_params = true_selected_params_df.loc[selected_params_idx, :] # サンプルの選択
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes_candidates[int(selected_params.iloc[0])],
                                 activation=activation_candidates[int(selected_params.iloc[1])],
                                 alpha=selected_params.iloc[2],
                                 learning_rate_init=selected_params.iloc[3],
                                 random_state=99)
            if validation_method == 'cv':
                estimated_y = cross_val_predict(model, x, y, cv=cross_validation)
            else:
                model.fit(x, y)
                estimated_y = np.ndarray.flatten(model.predict(x_midknn))
            params_with_score_df.loc[selected_params_idx, 'score'] = r2_score(y, estimated_y) # データの保存
        if display_flag:
            print('Best score :', params_with_score_df['score'].max())
            print('='*10)
        
        # 最後はBOの計算をしないためbreak
        if bo_iter + 1 == bo_iteration_number:
            break
                
        # Bayesian optimization
        bo_x_data = bo_params_df.copy() # GP学習用データはGMRの結果があるサンプル
        bo_x_prediction = remaining_params_df.copy() # predictionは選択されていない（GMRの結果がない）サンプル
        bo_y_data = params_with_score_df.loc[bo_params_df.index, 'score'] # yはGMRのr2cv
        
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
            bo_x_data_std = bo_x_data.std()
            bo_x_data_std[bo_x_data_std == 0] = 1
            autoscaled_bo_y_data = (bo_y_data - bo_y_data.mean()) / bo_y_data.std()
            autoscaled_bo_x_data = (bo_x_data - bo_x_data.mean()) / bo_x_data_std
            autoscaled_bo_x_prediction = (bo_x_prediction - bo_x_data.mean()) / bo_x_data_std
            
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
                bo_model = GaussianProcessRegressor(alpha=0, kernel=optimal_bo_kernel, random_state=9) # GPR モデルの宣言
                
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
    #params_with_score_df.sort_values('score', ascending=False).to_csv('params_with_score.csv')
    params_with_score_df_best = params_with_score_df.sort_values('score', ascending=False).iloc[0, :] # r2が高い順にソー
    best_r2cv = params_with_score_df_best['score'].copy()
    print(best_r2cv)
    best_candidate_combination = all_candidate_combinations_df.loc[params_with_score_df_best.name].copy()
    
    optimal_hidden_layer_sizes = hidden_layer_sizes_candidates[int(best_candidate_combination.iloc[0])]
    optimal_activation = activation_candidates[int(best_candidate_combination.iloc[1])]
    optimal_alpha = best_candidate_combination.iloc[2]
    optimal_learning_rate_init = best_candidate_combination.iloc[3]
       
    return optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init


def gs_dnn_hyperparams(x, y,
                       hidden_layer_sizes_candidates=[(50,), (100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)],
                       activation_candidates=['identity', 'logistic', 'tanh', 'relu'],
                       alpha_candidates=10 ** np.arange(-6, -1, dtype=float),
                       learning_rate_init_candidates=10 ** np.arange(-5, 0, dtype=float),
                       validation_method='cv', parameter=5, bo_iteration_number=15, display_flag=False):
    """
    Grid search of scikit-learn[MLPRegressor]-based deep neural network (DNN) hyperparameters
    
    Optimize DNN hyperparameters based on Grid search and cross-validation or midknn

    Parameters
    ----------
    x : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y : numpy.array or pandas.DataFrame
        (autoscaled) m x 1 vector of a Y-variable of training data
    hidden_layer_sizes_candidates : list
        vector of candidates of hidden_layer_sizes in MLPRegressor
    activation_candidates : list
        vector of candidates of activation in MLPRegressor
    alpha_candidates : numpy.array or list
        vector of candidates of alpha in MLPRegressor
    learning_rate_init_candidates : numpy.array or list
        vector of candidates of learning_rate_init in MLPRegressor    
    validation_method : 'cv' or 'midknn'
        if 'cv', cross-validation is used, and if 'midknn', midknn is used 
    parameter : int
        "fold_number"-fold cross-validation in cross-validation, and k in midknn

    Returns
    -------
    optimal_hidden_layer_sizes : tuple
        optimized hidden_layer_sizes in MLPRegressor
    optimal_activation : str
        optimized activation in MLPRegressor
    optimal_alpha : float
        optimized alpha in MLPRegressor
    optimal_learning_rate_init : float
        optimized learning_rate_init in MLPRegressor
    """
    
    if validation_method != 'cv' and validation_method != 'midknn':
#        print('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))
#        return 0, 0, 0
        sys.exit('\'{0}\' is unknown. Please check \'validation_method\'.'.format(validation_method))
        
    x = np.array(x)
    y = np.array(y)
    cross_validation = KFold(n_splits=parameter, random_state=9, shuffle=True)
    if validation_method == 'midknn':
        # make midknn data points
        x_midknn, y_midknn = make_midknn_dataset(x, y, parameter)
        
    alpha_candidates = np.array(alpha_candidates)
    learning_rate_init_candidates = np.array(learning_rate_init_candidates)

    hidden_layer_sizes_indexes = [i for i in range(len(hidden_layer_sizes_candidates))]
    activation_candidates_indexes = [i for i in range(len(activation_candidates))]
    all_candidate_combinations = []
    scores = []
    number = 0
    total_number = len(hidden_layer_sizes_indexes) * len(activation_candidates_indexes) * len(alpha_candidates) * len(learning_rate_init_candidates)
    for hidden_layer_sizes_index in hidden_layer_sizes_indexes:
        for activation_candidates_indexe in activation_candidates_indexes:
            for alpha_candidate in alpha_candidates:
                for learning_rate_init_candidate in learning_rate_init_candidates:
                    if display_flag:
                        number += 1
                        print(number, '/', total_number)
                    all_candidate_combinations.append([hidden_layer_sizes_index, activation_candidates_indexe, alpha_candidate, learning_rate_init_candidate])
                    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes_candidates[int(hidden_layer_sizes_index)],
                                         activation=activation_candidates[int(activation_candidates_indexe)],
                                         alpha=alpha_candidate,
                                         learning_rate_init=learning_rate_init_candidate,
                                         random_state=99)
                    if validation_method == 'cv':
                        estimated_y = cross_val_predict(model, x, y, cv=cross_validation)
                    else:
                        model.fit(x, y)
                        estimated_y = np.ndarray.flatten(model.predict(x_midknn))
                    scores.append(r2_score(y, estimated_y))
    scores = np.array(scores)
    best_index = np.where(scores == scores.max())[0][0]
    all_candidate_combinations = np.array(all_candidate_combinations)
    
    optimal_hidden_layer_sizes = hidden_layer_sizes_candidates[int(all_candidate_combinations[best_index, 0])]
    optimal_activation = activation_candidates[int(all_candidate_combinations[best_index, 1])]
    optimal_alpha = all_candidate_combinations[best_index, 2]
    optimal_learning_rate_init = all_candidate_combinations[best_index, 3]
       
    return optimal_hidden_layer_sizes, optimal_activation, optimal_alpha, optimal_learning_rate_init
