# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def lomp(
    estimator,
    x,
    x_half_range, # half range around x
    n_virtual_samples=100,  # number of virtually generated samples
    random_state=None,
):
    
    x = np.array(x)
        
    if x.ndim == 1:
        x = x.reshape([1, len(x)])
        
    lomp = np.zeros(x.shape)
    model_i = LinearRegression()
    for sample_number in range(x.shape[0]):
        if random_state:
            np.random.seed(random_state)
        x_upper = x[sample_number, :] + x_half_range
        x_lower = x[sample_number, :] - x_half_range
        x_generated = np.random.rand(n_virtual_samples, x.shape[1]) * (x_upper - x_lower) + x_lower
        # prediction
        y_pred_i = np.ndarray.flatten(estimator.predict(x_generated))
        x_generated = x_generated - x_generated.mean(axis=0)
        y_pred_i = y_pred_i - y_pred_i.mean(axis=0)
        model_i.fit(x_generated, y_pred_i)
        lomp[sample_number, :] = model_i.coef_
        
    np.random.seed()
    return lomp
