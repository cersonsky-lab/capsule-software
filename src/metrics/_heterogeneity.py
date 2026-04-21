"""Metrics to calculate API heterogeneity in a capsule."""

import numpy as np


def calculate_mean(sub_array, sample_no):
    """Calculates the mean for a given sub-array.

    The macro-voxel mean is defined as:

    Number of voxels of sample / Total number of voxels

    Args:
        sub_array (np.ndarray): Array with which to calculate the mean.
        sample_no (int): Number corresponding to the sample.

    Returns:
        float: Poole index
    """
    unique_labels = np.unique(sub_array)
    total_voxels = sub_array.size
    observed_species = len(
        np.where(sub_array.flatten() == sample_no)[0]
    )  # exclude the background (typically labeled as 0)
    mean = observed_species / total_voxels
    return mean


def calculate_vmr(sub_array, sample_no=None):
    """Calculates the variance-to-mean ratio.

    The variance-to-mean ratio is defined as:

    Variance of sample in a voxel / Mean of sample in the voxel.

    Args:
        sub_array (np.ndarray): Array with which to compute the VMR.
        sample_no (int): _description_. Number corresponding to the sample.

    Returns:
        float: Lacey index
    """
    is_species = (sub_array == sample_no).astype(int)

    if not np.any(is_species):
        return 0

    lacey_index = np.var(is_species) / np.mean(is_species)
    return lacey_index


# def calculate_segregation_index(sub_array, sample_no=None):
    
#     is_species = (sub_array == sample_no).astype(int)
#     if not np.any(is_species):
#         return 0
#     segregation_index = 1 - (np.var(is_species) / np.mean(is_species))

#     return segregation_index
    

