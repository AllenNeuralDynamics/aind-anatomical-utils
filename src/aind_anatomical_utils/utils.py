"""Utility functions."""

import numpy as np
from numpy.typing import NDArray


def find_indices_equal_to(arr: NDArray[np.integer], v: float | bool) -> NDArray[np.intp]:
    """Find array indices equal to v."""
    return np.column_stack(np.nonzero(arr == v))
