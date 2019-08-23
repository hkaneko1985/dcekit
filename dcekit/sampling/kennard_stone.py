# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import numpy as np


def kennard_stone(dataset, number_of_samples_to_be_selected):
    """
    Sample selection based on Kennard-Stone (KS) algorithm
    
    Select samples using KS algorithm

    Parameters
    ----------
    dataset : numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of dataset,
        m is the number of sammples and
        n is the number of (X-)variables
    number_of_samples_to_be_selected : int
        number of samples to be selected

    Returns
    -------
    selected_sample_numbers : list
        indexes of selected samples (training data)
    remaining_sample_numbers : list
        indexes of remaining samples (test data)
    """

    dataset = np.array(dataset)
    original_x = dataset
    distance_to_average = ((dataset - np.tile(dataset.mean(axis=0), (dataset.shape[0], 1))) ** 2).sum(axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = []
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, dataset.shape[0], 1)
    dataset = np.delete(dataset, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    for iteration in range(1, number_of_samples_to_be_selected):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, dataset.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(dataset[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(
                axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        dataset = np.delete(dataset, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

    return selected_sample_numbers, remaining_sample_numbers
