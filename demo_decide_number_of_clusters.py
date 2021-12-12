# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

number_of_max_clusters = 10  # maximum number of clusters
perplexity = 50  # perplexity in tSNE
k_in_knn = 3  # k in k-NN

x = pd.read_csv('iris_without_species.csv', index_col=0)
autoscaled_x = (x - x.mean()) / x.std()

# tSNE
score = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=0).fit_transform(autoscaled_x)
plt.rcParams['font.size'] = 18
plt.scatter(score[:, 0], score[:, 1], c='b')
plt.xlabel('t1')
plt.ylabel('t2')
plt.show()

# k-NN
knn = NearestNeighbors(n_neighbors=k_in_knn)
knn.fit(score)
knn_dist_all, knn_ind_all = knn.kneighbors(None)

# clustering
clustering_results = linkage(score, metric='euclidean', method='ward')

true_rate = []
for number_of_clusters in range(1, number_of_max_clusters + 1):
    print(number_of_clusters, number_of_max_clusters)
    cluster_numbers = fcluster(clustering_results, number_of_clusters, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
    true_number = 0
    for i in range(knn_ind_all.shape[0]):
        true_number += len(np.where(cluster_numbers[knn_ind_all[i, :]] == cluster_numbers[i])[0])
    true_rate.append(true_number / (knn_ind_all.shape[0] * knn_ind_all.shape[1]))

plt.scatter(range(1, number_of_max_clusters + 1), true_rate, c='blue')  # 散布図の作成。クラスター番号ごとにプロットの色を変えています
plt.xlabel('number of cluster')
plt.ylabel('matching ratio')
plt.show()

true_rate = np.array(true_rate)
optimal_cluster_number = np.where(true_rate == 1)[0][-1] + 1
print('Optimum number of clusters :', optimal_cluster_number)

cluster_numbers = fcluster(clustering_results, optimal_cluster_number, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
plt.rcParams['font.size'] = 18
plt.scatter(score[:, 0], score[:, 1], c=cluster_numbers,
            cmap=plt.get_cmap('jet'))  # 散布図の作成。クラスター番号ごとにプロットの色を変えています
plt.xlabel('t1')
plt.ylabel('t2')
plt.show()
