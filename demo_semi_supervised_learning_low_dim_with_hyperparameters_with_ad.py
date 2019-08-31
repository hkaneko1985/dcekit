# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcekit.learning import SemiSupervisedLearningLowDimension
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV

number_of_training_samples = 30
number_of_pca_components = 5
max_pls_component_number = 30
fold_number = 10

# load dataset
dataset = pd.read_csv('descriptors8_with_boiling_point.csv', encoding='SHIFT-JIS', index_col=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=x.shape[0] - number_of_training_samples,
                                                    random_state=0)
fold_number = min(fold_number, len(y_train))

# PLS and PCA
pls_components = np.arange(1, min(number_of_pca_components, max_pls_component_number, np.linalg.matrix_rank(x_train)) + 1)
cv_model = GridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=fold_number)
low_dimension_model = PCA(n_components=number_of_pca_components)
model = SemiSupervisedLearningLowDimension(base_estimator=cv_model,
                                           base_dimension_reductioner=low_dimension_model,
                                           x_unsupervised=x_test, autoscaling_flag=True, cv_flag=False,
                                           ad_flag=True)
model.fit(x_train, y_train)

# calculate y in training data
calculated_y_train = model.predict(x_train)
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

# estimate y in cross-validation in training data
estimated_y_in_cv = cross_val_predict(model, x_train, y_train, cv=fold_number)
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_in_cv, c='blue')
y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))

# prediction
if x_test.shape[0]:
    predicted_y_test = model.predict(x_test)
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
