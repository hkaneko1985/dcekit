# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

k = 10
rate_of_outliers = 0.01

# generation of samples
n_samples_1 = 500
mean_1 = [0, 0]
cov_1 = [[0.3, 0.28], [0.28, 0.3]]

n_samples_2 = 50
mean_2 = [-2, 2]
cov_2 = [[0.5, 0], [0, 0.5]]

np.random.seed(11)
x_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples_1)
x_2 = np.random.multivariate_normal(mean_2, cov_2, n_samples_2)
np.random.seed()
x = np.r_[x_1, x_2]
# plot
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.plot(x[:, 0], x[:, 1], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-4.5, 2.5])
plt.ylim([-2.5, 4.5])
plt.show()


xx, yy = np.meshgrid(np.linspace(-4.5, 2.5, 500), np.linspace(-2.5, 4.5, 500))
x_grids = np.c_[xx.ravel(), yy.ravel()]

autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0)
autoscaled_x_grids = (x_grids - x.mean(axis=0)) / x.std(axis=0)

# k-NN
model_knn = NearestNeighbors(n_neighbors=k)
model_knn.fit(autoscaled_x)
knn_dist_all, knn_ind_all = model_knn.kneighbors(None)
knn_dist = knn_dist_all.mean(axis=1)
sorted_knn_dist = np.sort(knn_dist)[::-1]
knn_dist_threshold = sorted_knn_dist[round(autoscaled_x.shape[0] * rate_of_outliers) - 1]
knn_dist_all_grids, knn_ind_grids = model_knn.kneighbors(autoscaled_x_grids)
knn_dist_grids = knn_dist_all_grids.mean(axis=1)
knn_dist_grids = knn_dist_grids.reshape(xx.shape)
# plot
plt.title('k-NN')
plt.contour(xx, yy, knn_dist_grids, levels=[knn_dist_threshold], linewidths=2, colors='darkred')
plt.plot(x[:, 0], x[:, 1], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# LOF
model_lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=rate_of_outliers)
model_lof.fit(autoscaled_x)
lof_grids = model_lof.decision_function(autoscaled_x_grids)
lof_grids = lof_grids.reshape(xx.shape)
# plot
plt.title('LOF')
plt.contour(xx, yy, lof_grids, levels=[0], linewidths=2, colors='darkred')
plt.plot(x[:, 0], x[:, 1], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
