# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint

def iot_obj_func(mol_fracs, x_mix, x_pure):
    """
    Objective function of Iterative Optimization Technology (IOT)
    
    Parameters
    ----------
    x_mix : numpy.array or pandas.DataFrame
        vector (size m) of mixture speatra
        m is the number of wavelength or wavenumber
    x_pure : numpy.array or pandas.DataFrame
        n x m matrix of raw spectra
        n is the number of raw materials

    Returns
    -------
    sum of squre of redisuals : float
    
    """
    
    x_mix = np.array(x_mix)
    x_pure = np.array(x_pure)
    calc_x_mix = np.dot(mol_fracs.reshape([1, len(mol_fracs)]), x_pure)
    return ((x_mix - calc_x_mix) ** 2).sum()

def iot(x_mix, x_pure):
    """
    Iterative Optimization Technology (IOT)
    
    Parameters
    ----------
    x_mix : numpy.array or pandas.DataFrame
        k x m matrix of mixture speatra
        k is the number of mixtures
        m is the number of wavelength or wavenumber
    x_pure : numpy.array or pandas.DataFrame
        n x m matrix of raw spectra
        n is the number of raw materials

    Returns
    -------
    pred_mol_fracs : numpy.array
        k x n matrix of raw spectra
    
    """
    
    x_mix = np.array(x_mix)
    x_pure = np.array(x_pure)
    number_of_pure_materials = x_pure.shape[0]
    pred_mol_fracs = np.zeros([x_mix.shape[0], number_of_pure_materials])
    bounds = []
    for i in range(number_of_pure_materials):
        bounds.append([0, 1])
    init_mol_fracs = np.zeros(number_of_pure_materials)
    for i in range(x_mix.shape[0]):
        pred_results = minimize(iot_obj_func,
                                x0=init_mol_fracs,
                                args=(x_mix, x_pure),
                                bounds=bounds,
                                constraints=LinearConstraint(np.ones(number_of_pure_materials), 1, 1),
                                method='SLSQP')
        pred_mol_fracs[i, :] = pred_results.x.copy()
    
    return pred_mol_fracs
