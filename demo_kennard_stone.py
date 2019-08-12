# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of Kennard-Stone (KS) algorighm

import matplotlib.pyplot as plt
import numpy as np

from dcekit.sampling import kennard_stone

number_of_samples = 50
number_of_selected_samples = 20

# generate samples 0f samples for demonstration
x = np.random.rand(number_of_samples, 2)

# standardize x
autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

# select samples using Kennard-Stone algorithm
selected_sample_numbers, remaining_sample_numbers = kennard_stone(autoscaled_x, number_of_selected_samples)
print('selected sample numbers')
print(selected_sample_numbers)
print('---')
print('remaining sample numbers')
print(remaining_sample_numbers)

selected_x = x[selected_sample_numbers, :]

# plot samples
plt.rcParams['font.size'] = 18
plt.figure()
plt.scatter(autoscaled_x[:, 0], autoscaled_x[:, 1], label='all samples')
plt.scatter(autoscaled_x[selected_sample_numbers, 0], autoscaled_x[selected_sample_numbers, 1], marker='*',
            label='selected samples')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')
plt.show()
