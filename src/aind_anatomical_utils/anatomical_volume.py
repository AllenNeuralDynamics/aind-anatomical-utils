"""
Functions for working with the headers of anatomical volumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray

from aind_anatomical_utils.coordinate_systems import _norm_code, _orientation


def _corner_indices(size: NDArray, outer: bool = True) -> NDArray[np.float64]:
    size = np.asarray(size, float)
    lo = -0.5 if outer else 0.0
    hi = (size - 0.5) if outer else (size - 1.0)
    return np.array(
        list(product([lo, hi[0]], [lo, hi[1]], [lo, hi[2]])), float
    )


def fix_corner_compute_origin(
    size: Sequence[int],
    spacing: Sequence[float],
    direction: NDArray[np.float64],
    target_point: Sequence[float],
    corner_code: str = "RAS",
    target_frame: str | None = None,
    use_outer_box: bool = False,
) -> tuple[tuple[float, float, float], NDArray[np.float64], int]:
    """
    Compute the image origin such that a specified corner of the image
    aligns with a given physical point in a specified coordinate frame.

    Parameters
    ----------
    size : Sequence of int
        The image size along each spatial axis (e.g., [nx, ny, nz]).
    spacing : Sequence of float
        The voxel spacing along each axis in millimeters (e.g., [sx, sy, sz]).
    direction : NDArray[np.float64]
        3x3 direction cosine matrix (row-major) in ITK/LPS convention.
    target_point : Sequence of float
        Physical coordinates (in mm) of the desired corner in the target frame.
    corner_code : str, optional
        3-letter code specifying which image corner to align (e.g., "LPI",
        "RAS").  Default is "LPI".
    target_frame : str, optional
        3-letter code specifying the coordinate frame of `target_point`.
        Defaults to `corner_code`.
    use_outer_box : bool, optional
        If True, use bounding box corners (-0.5, size-0.5); if False, use voxel
        centers (0, size-1).  Default is False.

    Returns
    -------
    origin_lps : tuple of float
        The computed image origin in LPS coordinates (mm).
    chosen_corner_index : NDArray[np.float64]
        The continuous index (ijk) of the chosen corner.
    corner_idx_number : int
        The index (0..7) of the chosen corner.

    Notes
    -----
    This function is useful for setting the image origin so that a particular
    image corner matches a desired physical location, taking into account
    direction cosines and coordinate conventions.
    """
    if target_frame is None:
        target_frame = corner_code

    # Normalize to 3D
    size_arr = np.array(list(size) + [1, 1, 1])[:3].astype(float)
    spacing_arr = np.array(list(spacing) + [1, 1, 1])[:3].astype(float)
    D = np.asarray(direction, float).reshape(3, 3)

    # All 8 corners in continuous index space and their LPS offsets from origin
    corners_idx = _corner_indices(size_arr, outer=use_outer_box)  # (8,3)
    offsets_lps = (corners_idx * spacing_arr) @ D.T  # (8,3)

    # Pick the corner that is "most" along the requested code axes
    s_align = _orientation(_norm_code(corner_code), "LPS").sign
    vals = offsets_lps * s_align  # convert to that code's axis sense
    # lexicographic argmax: prioritize x, then y, then z in that code
    idx = np.lexsort((vals[:, 2], vals[:, 1], vals[:, 0]))[-1]
    corner_offset_lps = offsets_lps[idx]

    # Convert target point to LPS and solve: target = origin + corner_offset
    s_target = _orientation(_norm_code(target_frame), "LPS").sign
    target_lps = np.asarray(target_point, float) * s_target
    origin_lps = target_lps - corner_offset_lps

    return tuple(origin_lps), corners_idx[idx], idx
