"""Utility functions"""

import numpy as np


def find_indices_equal_to(arr: np.ndarray, v: int) -> np.ndarray:
    """Find array indices equal to v"""
    return np.column_stack(np.nonzero(arr == v))
