# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Evaluation and optimization methods for applicability domain methods and their hyperparameters, considering the prediction performance of machine learning models
# Support vector regression is used as a regression method, as an example

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from dcekit.validation import ApplicabilityDomain
from dcekit.validation import DCEGridSearchCV

start_number_of_samples = 30  # # of samples to calculate AUCR first
ad_methods = ['kNN', 'LOF', 'OCSVM']  # AD methods

ks_in_knn = list(range(2, 32))
ks_in_lof = list(range(2, 32))
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)
nus_in_ocsvm = list(range(1, 51))

outer_fold_number = 10  # Outer fold number in DCV
inner_fold_number = 5  # Inner fold number in DCV
threshold_of_rate_of_same_value = 0.8
threshold_of_r = 0.99
rate_of_training_samples_inside_ad = 0.99
random_seed_constant = 9
do_autoscaling = True

nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # Gamma for nonlinear svr

plt.rcParams['font.size'] = 18

dataset = pd.read_csv(r'descriptors_with_logS.csv', index_col=0, header=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]
x = x.drop(['Ipc', 'Kappa3', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'], axis=1)  

rate_of_same_value = list()
num = 0
for x_variable_name in x.columns:
    num += 1
    same_value_number = x[x_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)[0]
#    print(deleting_variable_numbers[0])
deleting_variable_numbers_df = pd.DataFrame(deleting_variable_numbers, index=x.columns[deleting_variable_numbers])
deleting_variable_numbers_df = deleting_variable_numbers_df.T
x = x.drop(x.columns[deleting_variable_numbers], axis=1)

r_in_x = x.corr()
r_in_x = abs(r_in_x)
delete_x_numbers = []
for i in range(r_in_x.shape[0]):
    r_in_x.iloc[i, i] = 0
for i in range(r_in_x.shape[0]):
    r_max = r_in_x.max()
    r_max_max = r_max.max()
    if r_max_max >= threshold_of_r:
        variable_number_1 = np.where(r_max == r_max_max)[0][0]
        variable_number_2 = np.where(r_in_x.iloc[:, variable_number_1] == r_max_max)[0][0]
        r_sum_1 = r_in_x.iloc[:, variable_number_1].sum()
        r_sum_2 = r_in_x.iloc[:, variable_number_2].sum()
        if r_sum_1 >= r_sum_2:
            delete_x_number = variable_number_1
        else:
            delete_x_number = variable_number_2
        delete_x_numbers.append(delete_x_number)
        r_in_x.iloc[:, delete_x_number] = 0
        r_in_x.iloc[delete_x_number, :] = 0
    else:
        break   
x = x.drop(x.columns[delete_x_numbers], axis=1)

# DCV
fold = KFold(n_splits=inner_fold_number, shuffle=True, random_state=9)
if outer_fold_number == 0:
    true_outer_fold_number = len(y)
else:
    true_outer_fold_number = outer_fold_number
    
indexes = [sample_number % true_outer_fold_number for sample_number in range(x.shape[0])]
if random_seed_constant != 0:
    np.random.seed(random_seed_constant) # 再現性のため乱数の種を固定
fold_index_in_outer_cv = np.random.permutation(indexes) # シャッフル
np.random.seed() # 乱数の種の固定を解除

estimated_y_in_outer_cv = np.zeros(len(y))
for fold_number_in_outer_cv in np.arange(true_outer_fold_number):
    x_train_in_outer_cv = x.iloc[fold_index_in_outer_cv != fold_number_in_outer_cv, :]
    y_train = y[fold_index_in_outer_cv != fold_number_in_outer_cv]
    x_test_in_outer_cv = x.iloc[fold_index_in_outer_cv == fold_number_in_outer_cv, :]
    deleting_variable_numbers = np.where(x_train_in_outer_cv.var(axis=0, ddof=1) == 0)
    if len(deleting_variable_numbers[0]) == 0:
        x_train = x_train_in_outer_cv.copy()
        x_test = x_test_in_outer_cv.copy()
    else:
        x_train = x_train_in_outer_cv.drop(x_train_in_outer_cv.columns[deleting_variable_numbers], axis=1)
        x_test = x_test_in_outer_cv.drop(x_test_in_outer_cv.columns[deleting_variable_numbers], axis=1)
    
    # autoscaling
    if do_autoscaling:
        autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
        autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
        autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    else:
        autoscaled_x_train = x_train.copy()
        autoscaled_y_train = y_train.copy()
        autoscaled_x_test = x_test.copy()
    
    # SVR modeling
    variance_of_gram_matrix = list()
    numpy_autoscaled_Xtrain = np.array(autoscaled_x_train)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(
            -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    model_in_cv = DCEGridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                               cv=fold)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
    model_in_cv = DCEGridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                               {'C': nonlinear_svr_cs}, cv=fold)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_c = model_in_cv.best_params_['C']
    model_in_cv = DCEGridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                               {'gamma': nonlinear_svr_gammas}, cv=fold)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
    regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                               gamma=optimal_nonlinear_gamma)
    regression_model.fit(autoscaled_x_train, autoscaled_y_train)
    # prediction
    predicted_y_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
    if do_autoscaling:
        predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
        
    estimated_y_in_outer_cv[fold_index_in_outer_cv==fold_number_in_outer_cv] = predicted_y_test # 格納

# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, estimated_y_in_outer_cv, c='blue')
y_max = np.max(np.array([np.array(y), estimated_y_in_outer_cv]))
y_min = np.min(np.array([np.array(y), estimated_y_in_outer_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.show()
# r2p, RMSEp, MAEp
r2_dcv = float(1 - sum((y - estimated_y_in_outer_cv) ** 2) / sum((y - y.mean()) ** 2))
rmse_dcv = float((sum((y - estimated_y_in_outer_cv) ** 2) / len(y)) ** 0.5)
mae_dcv = float(sum(abs(y - estimated_y_in_outer_cv)) / len(y))
print('r2(DCV): {0}'.format(r2_dcv))
print('RMSE(DCV): {0}'.format(rmse_dcv))
print('MAE(DCV): {0}'.format(mae_dcv))
print('\n\n')

estimated_y_in_outer_cv = pd.DataFrame(estimated_y_in_outer_cv, index=y.index, columns=['estimated y'])
#estimated_y_in_outer_cv.to_csv(r'estimated_y_in_dcv.csv', encoding='utf_8')

y_df = pd.DataFrame(y)
samples = pd.concat([y_df, estimated_y_in_outer_cv], axis=1)

# autoscaling
if do_autoscaling:
    autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
else:
    autoscaled_x = x.copy()
    
# AD
all_results = np.zeros([x.shape[0], len(ks_in_knn) + len(ks_in_lof) + len(nus_in_ocsvm)])
ad_names = []
number = 0
for ad_method in ad_methods:
    if ad_method == 'kNN':
        hyperparameters = ks_in_knn.copy()
    elif ad_method == 'LOF':
        hyperparameters = ks_in_lof.copy()
    elif ad_method == 'OCSVM':
        hyperparameters = nus_in_ocsvm.copy()
    
    for hyperparameter in hyperparameters:
        if ad_method == 'kNN':
            ad = ApplicabilityDomain(method_name='knn', n_neighbors=hyperparameter, rate_of_outliers=1 - rate_of_training_samples_inside_ad)
            ad_names.append('kNN {0}'.format(hyperparameter - 1))
        elif ad_method == 'LOF':
            ad = ApplicabilityDomain(method_name='lof', n_neighbors=hyperparameter, rate_of_outliers=1 - rate_of_training_samples_inside_ad)
            ad_names.append('LOF {0}'.format(hyperparameter - 1))
        elif ad_method == 'OCSVM':
            ad = ApplicabilityDomain(method_name='ocsvm', gamma='auto', nu=hyperparameter / 100, rate_of_outliers=1 - rate_of_training_samples_inside_ad)
            ad_names.append('OCSVM {0}'.format(hyperparameter / 100))
        ad.fit(autoscaled_x)
        predicted_ad = ad.predict(autoscaled_x)
        predicted_ad = pd.DataFrame(predicted_ad, index=x.index, columns=['AD'])

        samples_ad = pd.concat([samples, predicted_ad], axis=1)
        samples_ad.sort_values('AD', ascending=False, inplace=True)
        coverage_rmse_mae = np.zeros([samples_ad.shape[0], 2])
        for i in range(1, samples_ad.shape[0]):
            coverage_rmse_mae[i, 0] = (i + 1) / samples_ad.shape[0]
            y_tmp = samples_ad.iloc[:i+1, 0]
            yp_tmp = samples_ad.iloc[:i+1, 1]
            coverage_rmse_mae[i, 1] = float((sum((y_tmp - yp_tmp) ** 2) / len(y_tmp)) ** 0.5)
        coverage_rmse_mae = pd.DataFrame(coverage_rmse_mae, columns=['coverage', 'RMSE'])
        plt.plot(coverage_rmse_mae.iloc[start_number_of_samples:, 0], coverage_rmse_mae.iloc[start_number_of_samples:, 1], color='0.8')
        all_results[:, number] = coverage_rmse_mae.values[:, 1]
        number += 1
        
all_results = pd.DataFrame(all_results, columns=ad_names)        
aucr = all_results.iloc[start_number_of_samples:, :].sum(axis=0)
print('Optimal AD method and its hyperparameter :', aucr.idxmin())
min_results = all_results[aucr.idxmin()]
max_results = all_results[aucr.idxmax()]
plt.plot(coverage_rmse_mae.iloc[start_number_of_samples:, 0], min_results.iloc[start_number_of_samples:], 'b-')
plt.xlim(0, 1)
plt.xlabel('coverage')
plt.ylabel('RMSE')
plt.show()

#plt.plot(coverage_rmse_mae.iloc[start_number_of_samples:, 0], max_results.iloc[start_number_of_samples:], color='0.5')
#plt.plot(coverage_rmse_mae.iloc[start_number_of_samples:, 0], min_results.iloc[start_number_of_samples:], 'b-')
#plt.xlim(0, 1)
#plt.xlabel('coverage')
#plt.ylabel('RMSE')
#plt.show()
