# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np

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
