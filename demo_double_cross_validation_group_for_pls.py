# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of Double Cross-Validation (DCV) for PLS

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from dcekit.validation import double_cross_validation_group
from sklearn import datasets
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from dcekit.validation import DCEGridSearchCV

# Settings
max_pls_component_number = 30
inner_fold_number = 5  # "fold_number"-fold cross-validation (CV) for inter CV 
outer_fold_number = 10  # "fold_number"-fold CV for outer CV
number_of_training_samples = 1000
number_of_test_samples = 1000

# Generate samples for demonstration
x, y = datasets.make_regression(n_samples=number_of_training_samples + number_of_test_samples, n_features=100,
                                n_informative=100, noise=100, random_state=0)
tmp = np.c_[y, x]
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
column_names = ['group_id', 'y']
for i in range(100):
    column_names.append('x{0}'.format(i + 1))
    
# sample dataset
dataset = pd.DataFrame(dataset, columns=column_names)
group_id = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]
x = dataset.iloc[:, 2:]

# DCV
pls_components = np.arange(1, max_pls_component_number + 1)
inner_cv = DCEGridSearchCV(PLSRegression(), {'n_components': pls_components}, cv=inner_fold_number)
estimated_y = double_cross_validation_group(gs_cv=inner_cv, x=x, y=y, groups=group_id, outer_fold_number=outer_fold_number,
                                            do_autoscaling=True, random_state=0)

# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, estimated_y)
y_max = np.max(np.array([np.array(y), estimated_y]))
y_min = np.min(np.array([np.array(y), estimated_y]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()
# r2dcv, RMSEdcv, MAEdcv
print('r2dcv: {0}'.format(float(1 - sum((y - estimated_y) ** 2) / sum((y - y.mean()) ** 2))))
print('RMSEdcv: {0}'.format(float((sum((y - estimated_y) ** 2) / len(y)) ** 0.5)))
print('MAEdcv: {0}'.format(float(sum(abs(y - estimated_y)) / len(y))))
