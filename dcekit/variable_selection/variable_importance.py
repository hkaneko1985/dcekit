# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
from numpy import matlib
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from scipy.stats import norm
import copy

def cvpfi(
    estimator,
    X,
    y,
    fold_number=5,
    scoring='r2',
    n_repeats=5,
    alpha_r = 0.999,
    random_state=None,
    replace=False
):
    
    X = np.array(X)
    y = np.array(y)
    estimator_copy = copy.deepcopy(estimator)    
    
    if alpha_r == 1:  # no correlation correction
        x_corr = np.eye(X.shape[1])
    else:
        x_corr = abs(np.corrcoef(X.T))
        if alpha_r != 0:
            for i in range(x_corr.shape[1]):
                for j in range(i + 1, x_corr.shape[1]):
                    z = 1 / 2 * np.log((1 + x_corr[i, j]) / (1 - x_corr[i, j])) # フィッシャーの z 変換
                    interval_of_normal_distribution = norm.interval(alpha_r, loc=z, scale=(1 / (X.shape[0] - 3)) ** 0.5)
                    interval_of_normal_distribution = np.array(interval_of_normal_distribution)
                    interval_of_r = (np.exp(2 * interval_of_normal_distribution) - 1) / (np.exp(2 * interval_of_normal_distribution) + 1)
#                    print(interval_of_r)
                    if interval_of_r[0] * interval_of_r[1] < 0:
                        x_corr[i, j] = 0
                        x_corr[j, i] = 0
                    else:
                        x_corr[i, j] = interval_of_r[0]
                        x_corr[j, i] = interval_of_r[0]
    
    min_number = X.shape[0] // fold_number
    mod_number = X.shape[0] % fold_number
    index = np.matlib.repmat(np.arange(fold_number), 1, min_number).ravel()
    if mod_number != 0:
        index = np.r_[index, np.arange(mod_number)]
    if random_state:
        np.random.seed(random_state)
    fold_index = np.random.permutation(index)
    
    estimated_y_in_cv = y.copy()
    estimated_y_in_cv_shuffled = np.zeros([X.shape[0], n_repeats, X.shape[1]])
    # cross-validatoin
    for fold_number_in_outer_cv in range(fold_number):
#        print(fold_number_in_outer_cv + 1, '/', fold_number)
        x_train = X[fold_index != fold_number_in_outer_cv, :]
        y_train = y[fold_index != fold_number_in_outer_cv]
        x_test = X[fold_index == fold_number_in_outer_cv, :]
        estimator_copy.fit(x_train, y_train)
        estimated_y_test = estimator_copy.predict(x_test)
        estimated_y_in_cv[fold_index==fold_number_in_outer_cv] = np.ndarray.flatten(estimated_y_test)
        for variable_number in range(X.shape[1]):
            target_variable_all = X[:, variable_number].copy()
            target_x_corr = x_corr[:, variable_number]
            if target_x_corr.sum() == 1:
                corr_flag = False
            else:
                corr_flag = True
                
            for n_repeat in range(n_repeats):
                x_test_shuffled = x_test.copy()
                shuffled_variable = np.random.choice(target_variable_all, x_test.shape[0], replace=replace)
                x_test_shuffled[:, variable_number] = shuffled_variable
                if corr_flag:
                    for corr_variable_number in range(X.shape[1]):
                        if corr_variable_number != variable_number and target_x_corr[corr_variable_number]:
                            corr_variable_all = X[:, corr_variable_number].copy()                            
                            for sample_number in range(x_test.shape[0]):
                                if np.random.rand(1)[0] < target_x_corr[corr_variable_number]:
                                    x_test_shuffled[sample_number, corr_variable_number] = np.random.choice(corr_variable_all, 1, replace=replace)
                                
                estimated_y_test_shuffled = np.ndarray.flatten(estimator_copy.predict(x_test_shuffled))
                estimated_y_in_cv_shuffled[fold_index==fold_number_in_outer_cv, n_repeat, variable_number] = estimated_y_test_shuffled
    importances = np.zeros([X.shape[1], n_repeats])
    if scoring == 'r2':
        r2cv = r2_score(y, estimated_y_in_cv)
#        print(r2cv)
        for variable_number in range(X.shape[1]):
            for n_repeat in range(n_repeats):
#                print(r2_score(y, estimated_y_in_cv_shuffled[:, n_repeat, variable_number]))
                importances[variable_number, n_repeat] = r2cv - r2_score(y, estimated_y_in_cv_shuffled[:, n_repeat, variable_number])
    elif scoring == 'accuracy':
        accuracy_cv = accuracy_score(y, estimated_y_in_cv)
#        print(accuracy_score)
        for variable_number in range(X.shape[1]):
            for n_repeat in range(n_repeats):
#                print(accuracy_score(y, estimated_y_in_cv_shuffled[:, n_repeat, variable_number]))
                importances[variable_number, n_repeat] = accuracy_cv - accuracy_score(y, estimated_y_in_cv_shuffled[:, n_repeat, variable_number])
        
    
    importances_mean = importances.mean(axis=1)
    importances_std = importances.std(axis=1, ddof=1)
    np.random.seed()
    
    return importances_mean, importances_std, importances

def cvpfi_gmr(
    estimator,
    dataset,
    numbers_of_x,
    numbers_of_y,
    fold_number=5,
    scoring='r2',
    n_repeats=5,
    alpha_r = 0.999,
    random_state=None,
    replace=False
):
    
    dataset = np.array(dataset)
    y = dataset[:, numbers_of_y]
    X = dataset[:, numbers_of_x]
    estimator_copy = copy.deepcopy(estimator)   
    
    if alpha_r == 1:  # no correlation correction
        x_corr = np.eye(X.shape[1])
    else:
        x_corr = abs(np.corrcoef(X.T))
        if alpha_r != 0:
            for i in range(x_corr.shape[1]):
                for j in range(i + 1, x_corr.shape[1]):
                    z = 1 / 2 * np.log((1 + x_corr[i, j]) / (1 - x_corr[i, j])) # フィッシャーの z 変換
                    interval_of_normal_distribution = norm.interval(alpha_r, loc=z, scale=(1 / (X.shape[0] - 3)) ** 0.5)
                    interval_of_normal_distribution = np.array(interval_of_normal_distribution)
                    interval_of_r = (np.exp(2 * interval_of_normal_distribution) - 1) / (np.exp(2 * interval_of_normal_distribution) + 1)
                    if interval_of_r[0] * interval_of_r[1] < 0:
                        x_corr[i, j] = 0
                        x_corr[j, i] = 0
    
    min_number = X.shape[0] // fold_number
    mod_number = X.shape[0] % fold_number
    index = np.matlib.repmat(np.arange(fold_number), 1, min_number).ravel()
    if mod_number != 0:
        index = np.r_[index, np.arange(mod_number)]
    if random_state:
        np.random.seed(random_state)
    fold_index = np.random.permutation(index)
    
    estimated_y_in_cv = y.copy()
    estimated_y_in_cv_shuffled = np.zeros([X.shape[0], y.shape[1], n_repeats, X.shape[1]])
    # cross-validatoin
    for fold_number_in_outer_cv in range(fold_number):
        dataset_train = dataset[fold_index != fold_number_in_outer_cv, :]
        x_test = X[fold_index == fold_number_in_outer_cv, :]
        estimator_copy.fit(dataset_train)
        estimated_y_test = estimator_copy.predict_rep(x_test, numbers_of_x, numbers_of_y)
        estimated_y_in_cv[fold_index==fold_number_in_outer_cv, :] = estimated_y_test
        for variable_number in range(X.shape[1]):
            target_variable_all = X[:, variable_number].copy()
            target_x_corr = x_corr[:, variable_number]
            if target_x_corr.sum() == 1:
                corr_flag = False
            else:
                corr_flag = True
            
            for n_repeat in range(n_repeats):
                x_test_shuffled = x_test.copy()
                shuffled_variable = np.random.choice(target_variable_all, x_test.shape[0], replace=replace)
                
                x_test_shuffled[:, variable_number] = shuffled_variable
                
                if corr_flag:
                    for corr_variable_number in range(X.shape[1]):
                        if corr_variable_number != variable_number and target_x_corr[corr_variable_number]:
                            corr_variable_all = X[:, corr_variable_number].copy()                            
                            for sample_number in range(x_test.shape[0]):
                                if np.random.rand(1)[0] < target_x_corr[corr_variable_number]:
                                    x_test_shuffled[sample_number, corr_variable_number] = np.random.choice(corr_variable_all, 1, replace=replace)
                estimated_y_in_cv_shuffled[fold_index==fold_number_in_outer_cv, :, n_repeat, variable_number] = estimator_copy.predict_rep(x_test_shuffled, numbers_of_x, numbers_of_y)
    
    importances = np.zeros([X.shape[1], n_repeats])
    if scoring == 'r2':
        r2cv = r2_score(np.ravel(y), np.ravel(estimated_y_in_cv))
        for variable_number in range(X.shape[1]):
            for n_repeat in range(n_repeats):
                importances[variable_number, n_repeat] = r2cv - r2_score(np.ravel(y), np.ravel(estimated_y_in_cv_shuffled[:, :, n_repeat, variable_number]))
        
    importances_mean = importances.mean(axis=1)
    importances_std = importances.std(axis=1, ddof=1)
    np.random.seed()
    
    return importances_mean, importances_std, importances
