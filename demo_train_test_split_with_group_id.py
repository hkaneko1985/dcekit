# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Demonstration of hyperparameter optimization with grid search and cross-validation (DCEGridSearchCV) for EN

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcekit.validation import DCEGridSearchCV, train_test_split_group
from sklearn.linear_model import ElasticNet

ratio_of_test_groups = 0.25  # this is the rate of the number of GROUPS (NOT samples)
#number_of_test_groups = 40  # when not the rate but the number is set, this is teh number of GROUPS (NOT samples) 
lambdas = 2 ** np.arange(-15, 0, dtype=float) # λ for EN
alphas = np.arange(0, 1.01, 0.01, dtype=float) # α for EN
fold_number = 5

# generate sample dataset
number_of_all_samples = 500
np.random.seed(seed=100)
x_raw = np.random.rand(number_of_all_samples, 3) * 10 - 5
y1 = 3 * x_raw[:, 0:1] - 2 * x_raw[:, 1:2] + 0.5 * x_raw[:, 2:3]
y1 = y1 + y1.std(ddof=1) * 0.05 * np.random.randn(number_of_all_samples, 1)
tmp = np.c_[y1, x_raw]
id_tmp = []
id_number = 1
times = 0
for i in range(tmp.shape[0]):
    times += 1
    id_tmp.append(id_number) 
    if times == 3:
        times = 0
        id_number += 1
dataset = np.c_[id_tmp, tmp]
column_names = ['group_id', 'y', 'x1', 'x2', 'x3']

# sample dataset
dataset = pd.DataFrame(dataset, columns=column_names)
group_id = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]
x = dataset.iloc[:, 2:]

x_train, x_test, y_train, y_test = train_test_split_group(x, y, test_size=ratio_of_test_groups, groups=group_id, random_state=0)
#x_train, x_test, y_train, y_test = train_test_split_group(x, y, test_size=number_of_test_groups, groups=group_id, random_state=0)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# EN hyperparameter optimization with grid search and cross-validation (DCEGridSearchCV)
cv_model = DCEGridSearchCV(ElasticNet(), {'l1_ratio': lambdas, 'alpha': alphas},
                           cv=fold_number, random_state=11, display_flag=True)
cv_model.fit(autoscaled_x_train, autoscaled_y_train)

optimal_en_lambda = cv_model.best_params_['l1_ratio']
optimal_en_alpha = cv_model.best_params_['alpha']
print('Optimal lambda in CV :', optimal_en_lambda)
print('Optimal alpha in CV :', optimal_en_alpha)

# EN
model = ElasticNet(l1_ratio=optimal_en_lambda, alpha=optimal_en_alpha)
model.fit(autoscaled_x_train, autoscaled_y_train)

# calculate y in training data
calculated_y_train = model.predict(autoscaled_x_train) * y_train.std(ddof=1) + y_train.mean()
# yy-plot
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, calculated_y_train, c='blue')
y_max = np.max(np.array([np.array(y_train), calculated_y_train]))
y_min = np.min(np.array([np.array(y_train), calculated_y_train]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))

# prediction
if x_test.shape[0]:
    predicted_y_test = model.predict(autoscaled_x_test) * y_train.std(ddof=1) + y_train.mean()
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y_test, predicted_y_test, c='blue')
    y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
    y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.show()
    # r2p, RMSEp, MAEp
    print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
    print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
    print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
