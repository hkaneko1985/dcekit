# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from dcekit.optimization import obj_func

# settings
number_of_raw_spectra = 6  # number of raw spectra

number_of_norm_dist = 10
number_of_mixture_spectra = 1
number_of_wavelengthes = 1100
np.random.seed(1)
number_of_norm_dist = len(np.where(np.random.rand(30) > 0.5)[0])
stds = np.random.rand(number_of_raw_spectra, number_of_norm_dist) * 100 + 10
means = np.random.rand(number_of_raw_spectra, number_of_norm_dist) * 1000 + 1000
intensities = np.random.rand(number_of_raw_spectra, number_of_norm_dist) * 30

# make and plot raw spectra
x_axis = np.arange(1100, 2200)
raw_spectra = np.zeros([number_of_raw_spectra, len(x_axis)])
for raw_spectra_number in range(number_of_raw_spectra):
    for dist in range(number_of_norm_dist):
        raw_spectra[raw_spectra_number, :] += intensities[raw_spectra_number, dist] * (1 / (2 * np.pi * stds[raw_spectra_number, dist] ** 2) ** (0.5) * np.exp(-(x_axis - means[raw_spectra_number, dist]) ** 2 / 2 / stds[raw_spectra_number, dist] ** 2))
plt.rcParams['font.size'] = 18
for raw_spectra_number in range(number_of_raw_spectra):
    plt.plot(x_axis, raw_spectra[raw_spectra_number, :], color='b', label='raw spectra {0}'.format(raw_spectra_number + 1))
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity')
    plt.legend()
    plt.show()

# make true mole fractions
np.random.seed()
true_mol_fracs = np.random.rand(number_of_raw_spectra)
true_mol_fracs = true_mol_fracs / true_mol_fracs.sum()

# make and plot mixture spectra
mixture_spectra = np.dot(true_mol_fracs.reshape([1, len(true_mol_fracs)]), raw_spectra)
plt.plot(x_axis, mixture_spectra[0, :], color='r', label='mixture spectra')
plt.xlabel('wavelength [nm]')
plt.ylabel('intensity')
plt.legend()
plt.show()

# IOT prediction
bounds = []
for i in range(number_of_raw_spectra):
    bounds.append([0, 1])
init_mol_fracs = np.zeros(number_of_raw_spectra)
pred_results = minimize(obj_func,
                        x0=init_mol_fracs,
                        args=(mixture_spectra, raw_spectra),
                        bounds=bounds,
                        constraints=LinearConstraint(np.ones(number_of_raw_spectra), 1, 1),
                        method='SLSQP')
pred_mol_fracs = pred_results.x.copy()

# show results
print('\n')
#print(calc_mol_fracs)
print('Predicted mole fractions :', pred_mol_fracs)
print('True mole fractions :', true_mol_fracs)

# plot esidual spectra
residual_spectra = mixture_spectra - np.dot(pred_mol_fracs.reshape([1, len(pred_mol_fracs)]), raw_spectra)
plt.plot(x_axis, residual_spectra[0, :], color='k', label='redisual spectra')
plt.xlabel('wavelength [nm]')
plt.ylabel('intensity')
plt.legend()
plt.show()
