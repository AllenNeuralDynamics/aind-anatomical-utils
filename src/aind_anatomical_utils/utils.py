"""Utility functions"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def find_indices_equal_to(arr: NDArray, v: Union[int, float, bool]) -> NDArray:
    """Find array indices equal to v"""
    return np.column_stack(np.nonzero(arr == v))
