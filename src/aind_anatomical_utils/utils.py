"""Utility functions"""

import numpy as np


from typing import Union

def find_indices_equal_to(arr: np.ndarray, v: Union[int, float, bool]) -> np.ndarray:
    """Find array indices equal to v"""
    return np.column_stack(np.nonzero(arr == v))
