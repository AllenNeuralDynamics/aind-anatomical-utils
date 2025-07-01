"""Utility functions"""

import numpy as np


def find_indices_equal_to(arr, v):
    """Find array indices equal to v"""
    return np.column_stack(np.nonzero(arr == v))
