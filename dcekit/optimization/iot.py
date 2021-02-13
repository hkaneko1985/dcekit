# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np

def obj_func(mol_fracs, x_mix, x_pure):
    calc_x_mix = np.dot(mol_fracs.reshape([1, len(mol_fracs)]), x_pure)
    return ((x_mix - calc_x_mix) ** 2).sum()
