# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import numpy as np
from dcekit.optimization import iot

# settings
number_of_pure_spectra = 6  # number of pure spectra

number_of_norm_dist = 10
number_of_mixture_spectra = 1
number_of_wavelengthes = 1100
np.random.seed(1)
number_of_norm_dist = len(np.where(np.random.rand(30) > 0.5)[0])
stds = np.random.rand(number_of_pure_spectra, number_of_norm_dist) * 100 + 10
means = np.random.rand(number_of_pure_spectra, number_of_norm_dist) * 1000 + 1000
intensities = np.random.rand(number_of_pure_spectra, number_of_norm_dist) * 30

# make and plot pure spectra
x_axis = np.arange(1100, 2200)
pure_spectra = np.zeros([number_of_pure_spectra, len(x_axis)])
for raw_spectra_number in range(number_of_pure_spectra):
    for dist in range(number_of_norm_dist):
        pure_spectra[raw_spectra_number, :] += intensities[raw_spectra_number, dist] * (1 / (2 * np.pi * stds[raw_spectra_number, dist] ** 2) ** (0.5) * np.exp(-(x_axis - means[raw_spectra_number, dist]) ** 2 / 2 / stds[raw_spectra_number, dist] ** 2))
plt.rcParams['font.size'] = 18
for pure_spectra_number in range(number_of_pure_spectra):
    plt.plot(x_axis, pure_spectra[pure_spectra_number, :], color='b', label='pure spectra {0}'.format(pure_spectra_number + 1))
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity')
    plt.legend()
    plt.show()

# make true mole fractions
np.random.seed()
true_mol_fracs = np.random.rand(number_of_pure_spectra)
true_mol_fracs = true_mol_fracs / true_mol_fracs.sum()

# make and plot mixture spectra
mixture_spectra = np.dot(true_mol_fracs.reshape([1, len(true_mol_fracs)]), pure_spectra)
plt.plot(x_axis, mixture_spectra[0, :], color='r', label='mixture spectra')
plt.xlabel('wavelength [nm]')
plt.ylabel('intensity')
plt.legend()
plt.show()

# IOT prediction
pred_mol_fracs = iot(mixture_spectra, pure_spectra)

# show results
print('\n')
print('Predicted mole fractions :', pred_mol_fracs[0, :])
print('True mole fractions :', true_mol_fracs)

# plot esidual spectra
residual_spectra = mixture_spectra - np.dot(pred_mol_fracs, pure_spectra)
plt.plot(x_axis, residual_spectra[0, :], color='k', label='redisual spectra')
plt.xlabel('wavelength [nm]')
plt.ylabel('intensity')
plt.legend()
plt.show()
