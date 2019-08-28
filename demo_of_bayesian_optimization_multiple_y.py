# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Bayesian optimization for multiple y variables

import warnings

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel

warnings.filterwarnings('ignore')

# settings
fold_number = 10
relaxation_value = 0.01

# load datasets and settings
training_data = pd.read_csv('training_data.csv', encoding='SHIFT-JIS', index_col=0)
x_for_prediction = pd.read_csv('x_for_prediction.csv', encoding='SHIFT-JIS', index_col=0)
settings = pd.read_csv('settings.csv', encoding='SHIFT-JIS', index_col=0)

# check datasets and settings
number_of_y_variables = settings.shape[1]
if not number_of_y_variables == (training_data.shape[1] - x_for_prediction.shape[1]):
    raise Exception(
        'Check the numbers of y-variables and X-variables in training_data.csv, data_for_prediction.csv and settings.csv.')
for i in range(number_of_y_variables):
    if settings.iloc[0, i] == 0 and settings.iloc[1, i] >= settings.iloc[2, i]:
        raise Exception('`lower_limit` must be lower than `upper_limit` in settings.csv.')

# autoscaling
y = training_data.iloc[:, 0:number_of_y_variables]
x = training_data.iloc[:, number_of_y_variables:]
x_for_prediction.columns = x.columns
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_for_prediction = (x_for_prediction - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()
mean_of_y = y.mean()
std_of_y = y.std()

# Gaussian process regression
estimated_y_for_prediction = np.zeros([x_for_prediction.shape[0], number_of_y_variables])
std_of_estimated_y_for_prediction = np.zeros([x_for_prediction.shape[0], number_of_y_variables])
plt.rcParams['font.size'] = 18
for y_number in range(number_of_y_variables):
    model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel())
    model.fit(autoscaled_x, autoscaled_y.iloc[:, y_number])
    estimated_y_for_prediction_tmp, std_of_estimated_y_for_prediction_tmp = model.predict(
        autoscaled_x_for_prediction, return_std=True)
    estimated_y_for_prediction[:, y_number] = estimated_y_for_prediction_tmp
    std_of_estimated_y_for_prediction[:, y_number] = std_of_estimated_y_for_prediction_tmp

    estimated_y = model.predict(autoscaled_x)
    estimated_y = estimated_y * std_of_y.iloc[y_number] + mean_of_y.iloc[y_number]
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y.iloc[:, y_number], estimated_y)
    y_max = max(y.iloc[:, y_number].max(), estimated_y.max())
    y_min = min(y.iloc[:, y_number].min(), estimated_y.min())
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('actual y')
    plt.ylabel('estimated y')
    plt.show()
    print('y{0}, r2: {1}'.format(y_number + 1, float(1 - sum((y.iloc[:, y_number] - estimated_y) ** 2) / sum(
        (y.iloc[:, y_number] - y.iloc[:, y_number].mean()) ** 2))))
    print('y{0}, RMSE: {1}'.format(y_number + 1, float(
        (sum((y.iloc[:, y_number] - estimated_y) ** 2) / len(y.iloc[:, y_number])) ** 0.5)))
    print('y{0}, MAE: {1}'.format(y_number + 1,
                                  float(sum(abs(y.iloc[:, y_number] - estimated_y)) / len(y.iloc[:, y_number]))))

    estimated_y_in_cv = model_selection.cross_val_predict(model, autoscaled_x, autoscaled_y.iloc[:, y_number],
                                                          cv=fold_number)
    estimated_y_in_cv = estimated_y_in_cv * std_of_y.iloc[y_number] + mean_of_y.iloc[y_number]
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y.iloc[:, y_number], estimated_y_in_cv)
    y_max = max(y.iloc[:, y_number].max(), estimated_y_in_cv.max())
    y_min = min(y.iloc[:, y_number].min(), estimated_y_in_cv.min())
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('actual y')
    plt.ylabel('estimated y in CV')
    plt.show()
    print('y{0}, r2cv: {1}'.format(y_number + 1, float(1 - sum((y.iloc[:, y_number] - estimated_y_in_cv) ** 2) / sum(
        (y.iloc[:, y_number] - y.iloc[:, y_number].mean()) ** 2))))
    print('y{0}, RMSEcv: {1}'.format(y_number + 1, float(
        (sum((y.iloc[:, y_number] - estimated_y_in_cv) ** 2) / len(y.iloc[:, y_number])) ** 0.5)))
    print('y{0}, MAEcv: {1}'.format(y_number + 1, float(
        sum(abs(y.iloc[:, y_number] - estimated_y_in_cv)) / len(y.iloc[:, y_number]))))
    print('')

estimated_y_for_prediction = pd.DataFrame(estimated_y_for_prediction)
estimated_y_for_prediction.columns = y.columns
estimated_y_for_prediction = estimated_y_for_prediction * y.std() + y.mean()
std_of_estimated_y_for_prediction = pd.DataFrame(std_of_estimated_y_for_prediction)
std_of_estimated_y_for_prediction.columns = y.columns
std_of_estimated_y_for_prediction = std_of_estimated_y_for_prediction * y.std()

# calculate probabilities
probabilities = np.zeros(estimated_y_for_prediction.shape)
for y_number in range(number_of_y_variables):
    if settings.iloc[0, y_number] == 1:
        probabilities[:, y_number] = 1 - norm.cdf(max(y.iloc[:, y_number]) + std_of_y.iloc[y_number] * relaxation_value,
                                                  loc=estimated_y_for_prediction.iloc[:, y_number],
                                                  scale=std_of_estimated_y_for_prediction.iloc[:, y_number])
    elif settings.iloc[0, y_number] == -1:
        probabilities[:, y_number] = norm.cdf(min(y.iloc[:, y_number]) - std_of_y.iloc[y_number] * relaxation_value,
                                              loc=estimated_y_for_prediction.iloc[:, y_number],
                                              scale=std_of_estimated_y_for_prediction.iloc[:, y_number])

    elif settings.iloc[0, y_number] == 0:
        probabilities[:, y_number] = norm.cdf(settings.iloc[2, y_number],
                                              loc=estimated_y_for_prediction.iloc[:, y_number],
                                              scale=std_of_estimated_y_for_prediction.iloc[:, y_number]) - norm.cdf(
            settings.iloc[1, y_number],
            loc=estimated_y_for_prediction.iloc[:, y_number],
            scale=std_of_estimated_y_for_prediction.iloc[:, y_number])

    probabilities[std_of_estimated_y_for_prediction.iloc[:, y_number] <= 0, y_number] = 0

# save results
probabilities = pd.DataFrame(probabilities)
probabilities.columns = y.columns
probabilities.index = x_for_prediction.index
probabilities.to_csv('probabilities.csv')

sum_of_log_probabilities = (np.log(probabilities)).sum(axis=1)
sum_of_log_probabilities = pd.DataFrame(sum_of_log_probabilities)
sum_of_log_probabilities[sum_of_log_probabilities == -np.inf] = -10 ** 100
sum_of_log_probabilities.columns = ['sum_of_log_probabilities']
sum_of_log_probabilities.index = x_for_prediction.index
sum_of_log_probabilities.to_csv('sum_of_log_probabilities.csv')

print('max of sum of log(probability) : {0}'.format(sum_of_log_probabilities.iloc[:, 0].max()))
print('index of max sum of log(probability) : {0}'.format(sum_of_log_probabilities.iloc[:, 0].idxmax()))
